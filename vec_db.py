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
                 n_clusters=None, n_probe=None, build_batch_size=100000):
        """
        Initializes the VecDB.
        
        Constraints: The __init__ function is not allowed to store large objects or caches. [cite: 109, 110]
        It only sets up file paths and configuration.
        
        IVFFlat tuning (from pgvector docs):
        - n_clusters: rows/1000 for up to 1M rows, sqrt(rows) for over 1M rows
        - n_probe: sqrt(n_clusters) for good recall/speed tradeoff
        """
        self.db_path = database_file_path
        
        # Define file paths for the index components
        self.base_index_path = index_file_path
        self.meta_path = self.base_index_path + ".meta.json"
        self.centroids_path = self.base_index_path + ".centroids.npy"
        self.index_data_path = self.base_index_path + ".data"
        
        self.build_batch_size = build_batch_size
        
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            
            # Determine n_clusters based on db_size if not provided.
            # More clusters = smaller clusters = can probe more while staying under RAM limit
            if n_clusters is None:
                if db_size <= 1_000_000:
                    self.n_clusters = max(100, db_size // 1000)
                else:
                    # Use 8000 clusters for large DBs to allow higher n_probe
                    self.n_clusters = 8000
            else:
                self.n_clusters = n_clusters
            
            # n_probe: with 8000 clusters, avg cluster = 2500 vectors
            # 35 probes × 2500 × 64 × 4 bytes ≈ 22MB (safe under 50MB)
            self.n_probe = n_probe if n_probe is not None else 35
                
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            
            print(f"Generating new database of size {db_size}...")
            self.generate_database(db_size)
        else:
            # If using an existing DB, load the metadata.
            self._load_meta()
            
            # Determine n_probe based on DB size to meet RAM constraints:
            # 1M: 20MB limit → n_probe ~8
            # 10M+: 50MB limit → n_probe ~35
            if n_probe is not None:
                self.n_probe = n_probe
            else:
                # Get DB size to determine appropriate n_probe
                db_size = self._get_num_records()
                if db_size <= 1_000_000:
                    # 1M: 20MB RAM limit - very conservative
                    self.n_probe = 8
                elif db_size <= 5_000_000:
                    self.n_probe = 15
                else:
                    # 10M, 15M, 20M: 50MB RAM limit
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
        # Generate as float32 and convert to float16 (numpy random doesn't support float16 directly)
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
        
        # 'r+' mode opens for reading and writing, doesn't truncate
        mmap_vectors = np.memmap(self.db_path, dtype=VECTOR_DTYPE, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        
        # Per project spec, we rebuild the whole index. [cite: 83]
        # A real system would support incremental indexing.
        print("Rebuilding index after insertion...")
        self._build_index()
        print("Index rebuild complete.")

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            num_records = self._get_num_records()
            if row_num >= num_records:
                raise IndexError(f"Row number {row_num} is out of bounds for {num_records} records.")
                
            mmap_vector = np.memmap(self.db_path, dtype=VECTOR_DTYPE, mode='r', shape=(1, DIMENSION), offset=offset)
            # Create a copy to ensure we release the mmap file
            return np.array(mmap_vector[0])
        except Exception as e:
            print(f"Error in get_one_row for row {row_num}: {e}")
            return None

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        if num_records == 0:
            return np.array([], dtype=VECTOR_DTYPE)
        vectors = np.memmap(self.db_path, dtype=VECTOR_DTYPE, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)

    def _cal_score(self, vec1, vec2):
        # Add epsilon to prevent division by zero
        epsilon = 1e-9
        dot_product = np.dot(vec1, vec2.T)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 < epsilon or norm_vec2 < epsilon:
            return 0.0
            
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        # Ensure result is a single float
        return cosine_similarity.item()


    def _build_index(self):
        """
        Builds the Inverted File (IVF) index.
        1. Trains KMeans on normalized data to find centroids.
        2. Assigns each vector to a cluster.
        3. Saves the centroids and the inverted file (cluster_id -> [vec_ids]) to disk.
        """
        num_records = self._get_num_records()
        if num_records == 0:
            return

        print(f"Starting index build for {num_records} records with {self.n_clusters} clusters.")

        # 1. Train KMeans using MiniBatchKMeans to handle large data
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters,
                                 init='k-means++',
                                 batch_size=self.build_batch_size,
                                 max_iter=100,
                                 n_init=1, # Speed up build time
                                 random_state=DB_SEED_NUMBER)
        
        mmap_vectors = np.memmap(self.db_path, dtype=VECTOR_DTYPE, mode='r', shape=(num_records, DIMENSION))

        # We cluster on *normalized* vectors for cosine similarity
        for i in range(0, num_records, self.build_batch_size):
            chunk = mmap_vectors[i : i + self.build_batch_size]
            # Normalize chunk
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9 # Avoid division by zero
            chunk_norm = chunk / norms
            kmeans.partial_fit(chunk_norm)
        
        centroids_norm = kmeans.cluster_centers_
        # Save centroids
        np.save(self.centroids_path, centroids_norm)
        print("Centroids trained and saved.")

        # 2. Build the inverted file (in memory, as this is the build phase)
        #    This maps cluster_id -> list of vector_ids
        inverted_file = [[] for _ in range(self.n_clusters)]

        for i in range(0, num_records, self.build_batch_size):
            chunk = mmap_vectors[i : i + self.build_batch_size]
            # Normalize chunk
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            chunk_norm = chunk / norms
            
            # Find cluster for each vector in chunk
            labels = kmeans.predict(chunk_norm)
            
            for j, label in enumerate(labels):
                global_id = i + j
                inverted_file[label].append(global_id)
        
        del mmap_vectors # Release memmap file

        # 3. Save the inverted file to disk in our custom format
        #    Format: [Header: N*int64 offsets] [Data: all ID lists concatenated]
        header_size = self.n_clusters * 8  # 8 bytes for int64
        header = np.zeros(self.n_clusters, dtype=np.int64)
        
        with open(self.index_data_path, 'wb') as f:
            # Write a placeholder header
            f.write(header.tobytes())
            
            current_offset = f.tell()
            if current_offset != header_size:
                raise IOError("Header write error.")

            # Now write the data lists and record their offsets
            for i in range(self.n_clusters):
                header[i] = current_offset
                
                id_list = np.array(inverted_file[i], dtype=np.int32)
                
                # Write the length of the list (as int32)
                f.write(np.int32(len(id_list)).tobytes())
                
                # Write the list items
                f.write(id_list.tobytes())
                
                current_offset = f.tell()
            
            # Go back to the beginning and write the real header
            f.seek(0)
            f.write(header.tobytes())

        # 4. Save metadata
        self._save_meta()
        print(f"Index data file written to {self.index_data_path}.")

    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        """
        Retrieves the top_k nearest vectors using the IVF index.
        This function loads the index from disk *every time* to respect the no-cache rule. [cite: 111, 112]
        """
        
        # --- 1. Load Index Components (from disk) ---
        # This is compliant with the 50MB RAM limit as the centroids file is very small.
        # e.g., 20k clusters * 70 dims * 4 bytes/float = 5.6MB
        try:
            # We must load n_clusters from meta, as it's not stored in __init__
            with open(self.meta_path, 'r') as f:
                n_clusters = json.load(f)['n_clusters']
            
            centroids = np.load(self.centroids_path)
        except FileNotFoundError:
            print("Index files not found. Returning brute-force scan.")
            return self._brute_force_retrieve(query, top_k)
        
        if n_clusters != len(centroids):
            raise ValueError("Index metadata mismatch. Rebuild index.")

        # --- 2. Coarse Quantization: Find n_probe nearest clusters ---
        # Normalize query vector for cosine similarity
        query_norm = query / np.linalg.norm(query)
        
        # Calculate scores against all centroids (fast, all in memory)
        centroid_scores = query_norm.dot(centroids.T).flatten()
        
        # Get the IDs of the top n_probe best matching clusters
        # Use argpartition for speed (finds top N without sorting all)
        n_probe_actual = min(self.n_probe, n_clusters)
        if n_probe_actual < n_clusters:
            nearest_centroid_ids = np.argpartition(centroid_scores, -n_probe_actual)[-n_probe_actual:]
        else:
            nearest_centroid_ids = np.arange(n_clusters)

        # --- 3. Gather Candidate Vectors (from disk) ---
        candidate_ids = set()
        
        with open(self.index_data_path, 'rb') as f:
            # Read the *entire* header (offsets) into memory.
            # e.g., 20k clusters * 8 bytes/offset = 0.16MB (tiny)
            header = np.fromfile(f, dtype=np.int64, count=n_clusters)
            
            for cluster_id in nearest_centroid_ids:
                # Get the offset for this cluster's list
                offset = header[cluster_id]
                f.seek(offset)
                
                # Read the length of the list (int32)
                try:
                    list_len = np.fromfile(f, dtype=np.int32, count=1)[0]
                except IndexError:
                    continue # Empty list
                
                # Read the list of IDs
                ids = np.fromfile(f, dtype=np.int32, count=list_len)
                candidate_ids.update(ids)

        if not candidate_ids:
            return []

        # --- 4. Re-ranking: Score only the candidates ---
        candidate_list = sorted(candidate_ids)  # Sort for sequential disk access
        num_candidates = len(candidate_list)
        
        # Read candidate vectors directly from file (avoids memmap RSS bloat)
        candidate_vectors = np.zeros((num_candidates, DIMENSION), dtype=np.float32)
        vector_size = DIMENSION * ELEMENT_SIZE
        
        with open(self.db_path, 'rb') as f:
            for i, vec_id in enumerate(candidate_list):
                f.seek(int(vec_id) * vector_size)  # Cast to int to avoid overflow
                candidate_vectors[i] = np.frombuffer(f.read(vector_size), dtype=VECTOR_DTYPE)
        
        # Vectorized cosine similarity computation
        query_float = query.astype(np.float32).flatten()
        query_norm = np.linalg.norm(query_float)
        candidate_norms = np.linalg.norm(candidate_vectors, axis=1)
        
        # Avoid division by zero
        epsilon = 1e-9
        candidate_norms = np.maximum(candidate_norms, epsilon)
        
        # Compute all scores at once: (candidates @ query) / (||candidates|| * ||query||)
        scores = candidate_vectors @ query_float / (candidate_norms * max(query_norm, epsilon))
        
        # Get top_k indices using argpartition (faster than full sort)
        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        # --- 5. Format and Return Results ---
        return [int(candidate_list[i]) for i in top_indices]


    def _brute_force_retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        """The original, slow retrieve method. Used as a fallback."""
        scores = []
        num_records = self._get_num_records()
        if num_records == 0:
            return []
            
        # Use a min-heap for efficiency instead of sorting the whole list
        top_k_heap = []

        for row_num in range(num_records):
            vector = self.get_one_row(row_num)
            if vector is None:
                continue
                
            score = self._cal_score(query, vector)
            
            if len(top_k_heap) < top_k:
                heapq.heappush(top_k_heap, (score, row_num))
            else:
                heapq.heappushpop(top_k_heap, (score, row_num))
                
        final_results = sorted(top_k_heap, reverse=True)
        return [s[1] for s in final_results]