from typing import Dict, List, Annotated
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat", new_db=True, db_size=None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        
        # Derived index file paths
        self.centroids_path = self.index_path + "_centroids.npy"
        self.meta_path = self.index_path + "_meta.npy"
        self.inverted_idx_path = self.index_path + "_ids.bin"

        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        if not os.path.exists(self.db_path):
            return 0
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, DIMENSION)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return np.zeros(DIMENSION, dtype=np.float32)

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
        """
        Build IVF (Inverted File) index using Spherical K-Means.
        Uses cosine similarity by normalizing vectors before clustering.
        """
        num_records = self._get_num_records()
        if num_records == 0: return

        # Number of clusters: sqrt(N) to 4*sqrt(N) is typical for IVF
        # Using 4*sqrt(N) for better granularity
        n_clusters = int(4 * np.sqrt(num_records))
        n_clusters = max(100, min(n_clusters, 16000))  # Clamp to reasonable range

        # Load database vectors
        db_mmap = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        
        # Sample vectors for training (max 100k for efficiency)
        train_size = min(num_records, 100_000)
        rng = np.random.default_rng(DB_SEED_NUMBER)
        sample_indices = rng.choice(num_records, size=train_size, replace=False)
        training_data = np.array(db_mmap[sample_indices])
        
        # Normalize for spherical k-means (cosine similarity)
        norms = np.linalg.norm(training_data, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        training_data /= norms

        # Train MiniBatchKMeans
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=2048,
            random_state=DB_SEED_NUMBER,
            n_init=3,
            max_iter=100
        )
        kmeans.fit(training_data)
        
        # Save centroids
        np.save(self.centroids_path, kmeans.cluster_centers_)
        del training_data

        # Assign all vectors to clusters and build inverted index
        cluster_counts = np.zeros(n_clusters, dtype=np.int32)
        chunk_size = 500_000
        
        # Pass 1: Count vectors per cluster
        for i in range(0, num_records, chunk_size):
            end = min(i + chunk_size, num_records)
            batch = np.array(db_mmap[i:end])
            # Normalize batch
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            batch /= norms
            labels = kmeans.predict(batch)
            unique, counts = np.unique(labels, return_counts=True)
            cluster_counts[unique] += counts.astype(np.int32)

        # Compute offsets for each cluster
        offsets = np.concatenate(([0], np.cumsum(cluster_counts)[:-1])).astype(np.int32)
        meta_data = np.column_stack((offsets, cluster_counts))
        np.save(self.meta_path, meta_data)

        # Pass 2: Write vector IDs to inverted index
        current_pos = offsets.copy()
        ids_mmap = np.memmap(self.inverted_idx_path, dtype=np.int32, mode='w+', shape=(num_records,))
        
        for i in range(0, num_records, chunk_size):
            end = min(i + chunk_size, num_records)
            batch = np.array(db_mmap[i:end])
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            batch /= norms
            labels = kmeans.predict(batch)
            
            # Write IDs grouped by cluster
            for local_idx, label in enumerate(labels):
                global_id = i + local_idx
                pos = current_pos[label]
                ids_mmap[pos] = global_id
                current_pos[label] += 1
        
        ids_mmap.flush()
        del db_mmap, ids_mmap

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        """
        Retrieve top-k most similar vectors using IVF index.
        Uses batch processing to search many candidates with limited RAM.
        """
        if not os.path.exists(self.centroids_path):
            return self._brute_force_retrieve(query, top_k)
        
        num_records = self._get_num_records()
        
        # Load index components
        centroids = np.load(self.centroids_path)
        meta_data = np.load(self.meta_path)
        
        # Normalize query
        query = query.reshape(1, -1).astype(np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        
        # Normalize centroids and compute similarities
        centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroid_norms[centroid_norms == 0] = 1e-10
        centroids_normalized = centroids / centroid_norms
        
        # Find nearest clusters
        similarities = np.dot(centroids_normalized, query_norm.T).flatten()
        sorted_cluster_indices = np.argsort(similarities)[::-1]
        
        # How many total candidates to consider (more = better accuracy)
        # We process in batches so RAM stays low
        if num_records <= 1_000_000:
            TOTAL_CANDIDATES = 100_000   # 10% of DB
            BATCH_SIZE = 20_000          # Process 20k at a time (~5MB)
        elif num_records <= 10_000_000:
            TOTAL_CANDIDATES = 500_000   # 5% of DB
            BATCH_SIZE = 30_000          # Process 30k at a time (~8MB)
        else:
            TOTAL_CANDIDATES = 600_000   # 3% of DB - need more for perfect score
            BATCH_SIZE = 40_000          # Process 40k at a time (~10MB)
        
        # Collect ALL candidate IDs first (just IDs, not vectors - very cheap)
        all_candidate_ids = []
        total_collected = 0
        
        with open(self.inverted_idx_path, 'rb') as f:
            for cluster_id in sorted_cluster_indices:
                offset, count = meta_data[cluster_id]
                if count > 0:
                    remaining = TOTAL_CANDIDATES - total_collected
                    take_count = min(count, remaining)
                    if take_count <= 0:
                        break
                    
                    f.seek(int(offset) * 4)
                    data = f.read(take_count * 4)
                    ids = np.frombuffer(data, dtype=np.int32).copy()
                    all_candidate_ids.append(ids)
                    total_collected += take_count
                    
                    if total_collected >= TOTAL_CANDIDATES:
                        break
        
        if not all_candidate_ids:
            return []
        
        all_candidate_ids = np.concatenate(all_candidate_ids)
        
        # Process candidates in batches to limit RAM
        # Keep track of top-k across all batches
        top_scores = np.full(top_k, -np.inf, dtype=np.float32)
        top_ids = np.zeros(top_k, dtype=np.int32)
        
        vector_size = DIMENSION * ELEMENT_SIZE
        
        with open(self.db_path, 'rb') as f:
            for batch_start in range(0, len(all_candidate_ids), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(all_candidate_ids))
                batch_ids = all_candidate_ids[batch_start:batch_end]
                
                # Load batch vectors
                batch_vectors = np.zeros((len(batch_ids), DIMENSION), dtype=np.float32)
                for i, vec_id in enumerate(batch_ids):
                    f.seek(int(vec_id) * vector_size)
                    data = f.read(vector_size)
                    batch_vectors[i] = np.frombuffer(data, dtype=np.float32)
                
                # Compute cosine similarities for batch
                batch_norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
                batch_norms[batch_norms == 0] = 1e-10
                batch_vectors /= batch_norms
                batch_scores = np.dot(batch_vectors, query_norm.T).flatten()
                
                # Merge with current top-k
                combined_scores = np.concatenate([top_scores, batch_scores])
                combined_ids = np.concatenate([top_ids, batch_ids])
                
                # Keep top-k
                if len(combined_scores) > top_k:
                    top_k_indices = np.argpartition(combined_scores, -top_k)[-top_k:]
                    top_scores = combined_scores[top_k_indices]
                    top_ids = combined_ids[top_k_indices]
                else:
                    top_scores = combined_scores
                    top_ids = combined_ids
                
                # Free batch memory
                del batch_vectors, batch_scores
        
        # Final sort
        final_order = np.argsort(top_scores)[::-1]
        return top_ids[final_order].tolist()
    
    def _brute_force_retrieve(self, query, top_k):
        """Fallback brute-force search when no index exists."""
        scores = []
        num_records = self._get_num_records()
        for row_num in range(num_records):
            vector = self.get_one_row(row_num)
            score = self._cal_score(query, vector)
            scores.append((score, row_num))
        scores = sorted(scores, reverse=True)[:top_k]
        return [s[1] for s in scores]