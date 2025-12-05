from typing import Dict, List, Annotated
import numpy as np
import os
import json
import heapq
import math
from sklearn.cluster import MiniBatchKMeans

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64  # 2025 evaluation uses 64-dim embeddings
VECTOR_DTYPE = np.float32  # Avoid aggressive quantization per TA feedback

class VecDB:
    def __init__(self, database_file_path="saved_db.dat", index_file_path="index.dat", new_db=True, db_size=None,
                 n_clusters=None, n_probe=None):
        """
        Initializes the VecDB.
        Constraints: Retain exact parameters.
        """
        self.db_path = database_file_path
        self.base_index_path = index_file_path
        self.meta_path = self.base_index_path + ".meta.json"
        self.centroids_path = self.base_index_path + ".centroids.npy"
        self.index_data_path = self.base_index_path + ".data"
        
        # --- FIX: Define this internally ---
        self.build_batch_size = 100000 
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            
            # Determine n_clusters based on db_size
            if n_clusters is None:
                if db_size <= 1_000_000:
                    self.n_clusters = max(100, db_size // 1000)
                else:
                    self.n_clusters = 8000
            else:
                self.n_clusters = n_clusters
            
            self.n_probe = n_probe if n_probe is not None else 35
                
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            
            print(f"Generating new database of size {db_size}...")
            self.generate_database(db_size)
        else:
            self._load_meta()
            
            if n_probe is not None:
                self.n_probe = n_probe
            else:
                # Dynamic n_probe assignment based on DB size
                db_size = self._get_num_records()
                if db_size <= 1_000_000:
                    self.n_probe = 8
                elif db_size <= 5_000_000:
                    self.n_probe = 15
                else:
                    self.n_probe = 35

    def _load_meta(self):
        """Loads small metadata from disk."""
        try:
            with open(self.meta_path, 'r') as f:
                meta = json.load(f)
                self.n_clusters = meta['n_clusters']
            print(f"Loaded existing index with {self.n_clusters} clusters.")
        except FileNotFoundError:
            raise FileNotFoundError(f"No index metadata file found at {self.meta_path}. "
                                     "Ensure the index was built or 'new_db=True'.")

    def _save_meta(self):
        """Saves small metadata to disk."""
        with open(self.meta_path, 'w') as f:
            json.dump({'n_clusters': self.n_clusters}, f)

    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32).astype(VECTOR_DTYPE)
        self._write_vectors_to_file(vectors)
        print("Database generation complete. Building index...")
        self._build_index()
        print("Index build complete.")

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=VECTOR_DTYPE, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        if not os.path.exists(self.db_path):
            return 0
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        
        mmap_vectors = np.memmap(self.db_path, dtype=VECTOR_DTYPE, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        
        print("Rebuilding index after insertion...")
        self._build_index()
        print("Index rebuild complete.")

    def get_one_row(self, row_num: int) -> np.ndarray:
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            num_records = self._get_num_records()
            if row_num >= num_records:
                raise IndexError(f"Row number {row_num} is out of bounds.")
                
            mmap_vector = np.memmap(self.db_path, dtype=VECTOR_DTYPE, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            print(f"Error in get_one_row for row {row_num}: {e}")
            return None

    def get_all_rows(self) -> np.ndarray:
        num_records = self._get_num_records()
        if num_records == 0:
            return np.array([], dtype=VECTOR_DTYPE)
        vectors = np.memmap(self.db_path, dtype=VECTOR_DTYPE, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def _cal_score(self, vec1, vec2):
        epsilon = 1e-9
        dot_product = np.dot(vec1, vec2.T)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 < epsilon or norm_vec2 < epsilon:
            return 0.0
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity.item()

    def _build_index(self):
        """
        Builds the Index using Two-Pass strategy to keep RAM Usage constant.
        """
        num_records = self._get_num_records()
        if num_records == 0: return

        print(f"Starting index build for {num_records} records.")
        
        # 1. Train KMeans
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, init='k-means++', 
                                 batch_size=self.build_batch_size, max_iter=50, n_init=1, 
                                 random_state=DB_SEED_NUMBER)
        
        mmap_vectors = np.memmap(self.db_path, dtype=VECTOR_DTYPE, mode='r', shape=(num_records, DIMENSION))

        # Train centroids
        for i in range(0, num_records, self.build_batch_size):
            chunk = mmap_vectors[i : i + self.build_batch_size]
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            kmeans.partial_fit(chunk / norms)
        
        np.save(self.centroids_path, kmeans.cluster_centers_)
        print("Centroids saved.")

        # 2. Pass 1: Count cluster sizes
        cluster_counts = np.zeros(self.n_clusters, dtype=np.int32)
        for i in range(0, num_records, self.build_batch_size):
            chunk = mmap_vectors[i : i + self.build_batch_size]
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            labels = kmeans.predict(chunk / norms)
            cluster_counts += np.bincount(labels, minlength=self.n_clusters).astype(np.int32)

        # Calculate Offsets
        header_bytes = self.n_clusters * 8
        bytes_per_cluster = 4 + (cluster_counts * 4)
        cumulative_offsets = np.concatenate(([0], np.cumsum(bytes_per_cluster)[:-1]))
        final_offsets = header_bytes + cumulative_offsets
        
        # Write Header
        with open(self.index_data_path, 'wb') as f:
            f.seek(header_bytes + np.sum(bytes_per_cluster) - 1)
            f.write(b'\0')
            f.seek(0)
            f.write(final_offsets.astype(np.int64).tobytes())

        # 3. Pass 2: Write IDs
        current_write_pos = final_offsets.copy() + 4
        index_mmap = np.memmap(self.index_data_path, mode='r+')
        
        # Write counts
        for i in range(self.n_clusters):
            start = final_offsets[i]
            if start + 4 <= len(index_mmap):
                index_mmap[start : start + 4] = np.array([cluster_counts[i]], dtype=np.int32).view(np.uint8)

        # Write IDs
        for i in range(0, num_records, self.build_batch_size):
            chunk = mmap_vectors[i : i + self.build_batch_size]
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            labels = kmeans.predict(chunk / norms)
            
            batch_ids = np.arange(i, i + len(chunk), dtype=np.int32)
            sort_idx = np.argsort(labels)
            sorted_labels = labels[sort_idx]
            sorted_ids = batch_ids[sort_idx]
            
            unique_labels, split_idx = np.unique(sorted_labels, return_index=True)
            for j, label in enumerate(unique_labels):
                start = split_idx[j]
                end = split_idx[j+1] if j+1 < len(split_idx) else len(sorted_labels)
                ids_to_write = sorted_ids[start:end]
                
                pos = current_write_pos[label]
                index_mmap[pos : pos + len(ids_to_write)*4] = ids_to_write.view(np.uint8)
                current_write_pos[label] += len(ids_to_write)*4
                
        index_mmap.flush()
        self._save_meta()
        print("Index build complete.")

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        # 1. Load Metadata & Centroids
        try:
            with open(self.meta_path, 'r') as f:
                n_clusters = json.load(f)['n_clusters']
            centroids = np.load(self.centroids_path)
        except:
            return self._brute_force_retrieve(query, top_k)

        # 2. Find Candidate Clusters
        query_float = query.astype(np.float32).flatten()
        query_norm = query_float / (np.linalg.norm(query_float) + 1e-9)
        dists = np.dot(centroids, query_norm)
        
        n_probe_actual = min(self.n_probe, n_clusters)
        nearest_clusters = np.argpartition(dists, -n_probe_actual)[-n_probe_actual:]

        # 3. Read Candidate IDs
        candidate_ids_list = []
        with open(self.index_data_path, 'rb') as f:
            header = np.fromfile(f, dtype=np.int64, count=n_clusters)
            for cluster_id in nearest_clusters:
                f.seek(header[cluster_id])
                count_bytes = f.read(4)
                if not count_bytes: continue
                count = int(np.frombuffer(count_bytes, dtype=np.int32)[0])
                if count > 0:
                    ids = np.frombuffer(f.read(count * 4), dtype=np.int32)
                    candidate_ids_list.append(ids)

        if not candidate_ids_list: return []
        candidate_ids = np.concatenate(candidate_ids_list)

        # 4. Batched Scoring (Time & RAM Optimized)
        num_records = self._get_num_records()
        mmap_vectors = np.memmap(self.db_path, dtype=VECTOR_DTYPE, mode='r', shape=(num_records, DIMENSION))
        
        score_batch_size = 20000 
        top_k_heap = [] 
        
        # Sort IDs for better disk access
        candidate_ids.sort()
        
        for i in range(0, len(candidate_ids), score_batch_size):
            batch_ids = candidate_ids[i : i + score_batch_size]
            
            vec_batch = np.array(mmap_vectors[batch_ids], dtype=np.float32)
            
            batch_norms = np.linalg.norm(vec_batch, axis=1)
            batch_norms[batch_norms == 0] = 1e-9
            scores = np.dot(vec_batch, query_float) / (batch_norms * np.linalg.norm(query_float))
            
            for j, score in enumerate(scores):
                if len(top_k_heap) < top_k:
                    heapq.heappush(top_k_heap, (score, batch_ids[j]))
                else:
                    if score > top_k_heap[0][0]:
                        heapq.heapreplace(top_k_heap, (score, batch_ids[j]))

        final_results = sorted(top_k_heap, key=lambda x: x[0], reverse=True)
        return [int(x[1]) for x in final_results]

    def _brute_force_retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        scores = []
        num_records = self._get_num_records()
        if num_records == 0: return []
        top_k_heap = []
        for row_num in range(num_records):
            vector = self.get_one_row(row_num)
            if vector is None: continue
            score = self._cal_score(query, vector)
            if len(top_k_heap) < top_k:
                heapq.heappush(top_k_heap, (score, row_num))
            else:
                heapq.heappushpop(top_k_heap, (score, row_num))
        final_results = sorted(top_k_heap, reverse=True)
        return [s[1] for s in final_results]