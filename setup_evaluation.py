"""
Setup script to generate all databases and indices for VecDB evaluation.
This will:
1. Use existing OpenSubtitles_en_20M_emb_64.dat as 20M database
2. Create 1M and 10M subsets from it
3. Build indices for all three
4. Generate ground truth for evaluation queries
"""

import numpy as np
import os
import time
from vec_db import VecDB, DIMENSION, DB_SEED_NUMBER

# Define VECTOR_DTYPE locally (matches vec_db.py's np.float32 usage)
VECTOR_DTYPE = np.float32

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database paths - use OpenSubtitles as the 20M source
PATH_DB_VECTORS_20M = os.path.join(BASE_DIR, "OpenSubtitles_en_20M_emb_64.dat")
PATH_DB_VECTORS_10M = os.path.join(BASE_DIR, "db_vectors_10m.dat")
PATH_DB_VECTORS_1M = os.path.join(BASE_DIR, "db_vectors_1m.dat")

PATH_INDEX_20M = os.path.join(BASE_DIR, "index_20m.dat")
PATH_INDEX_10M = os.path.join(BASE_DIR, "index_10m.dat")
PATH_INDEX_1M = os.path.join(BASE_DIR, "index_1m.dat")

# Query paths
QUERIES_FILE = os.path.join(BASE_DIR, "queries_emb_64.dat")
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "ground_truth_20m.npy")

NUM_QUERIES = 100  # Number of evaluation queries


def create_subset(source_file, dest_file, target_rows):
    """Create a subset database from a larger one."""
    print(f"Creating {target_rows:,} row subset: {dest_file}")
    
    file_size = os.path.getsize(source_file)
    total_rows = file_size // (DIMENSION * np.dtype(VECTOR_DTYPE).itemsize)
    
    source = np.memmap(source_file, dtype=VECTOR_DTYPE, mode='r', shape=(total_rows, DIMENSION))
    dest = np.memmap(dest_file, dtype=VECTOR_DTYPE, mode='w+', shape=(target_rows, DIMENSION))
    
    # Copy in chunks to avoid memory issues
    chunk_size = 1_000_000
    for i in range(0, target_rows, chunk_size):
        end = min(i + chunk_size, target_rows)
        dest[i:end] = source[i:end]
        print(f"  Copied {end:,} / {target_rows:,} rows")
    
    dest.flush()
    del source, dest
    print(f"  Done!")


def build_index_only(db_path, index_path, n_clusters):
    """Build index for an existing database file."""
    print(f"\nBuilding index for {db_path}")
    print(f"  Index path: {index_path}")
    print(f"  Clusters: {n_clusters}")
    
    # We need to create a VecDB that points to the existing file
    # but only builds the index
    
    file_size = os.path.getsize(db_path)
    num_records = file_size // (DIMENSION * np.dtype(VECTOR_DTYPE).itemsize)
    print(f"  Records: {num_records:,}")
    
    # Create a temporary VecDB with the paths set correctly
    # Must match VecDB's attribute names exactly
    class IndexBuilder:
        def __init__(self):
            self.db_path = db_path
            self.index_path = index_path
            # Match VecDB's derived paths exactly
            self.centroids_path = index_path + "_centroids.npy"
            self.meta_path = index_path + "_meta.npy"
            self.inverted_idx_path = index_path + "_ids.bin"
        
        def _get_num_records(self):
            return os.path.getsize(self.db_path) // (DIMENSION * np.dtype(VECTOR_DTYPE).itemsize)
    
    # Import the build method from VecDB
    builder = IndexBuilder()
    
    # Call _build_index directly using VecDB's method
    VecDB._build_index(builder)
    print(f"  Index built successfully!")


def generate_queries(num_queries):
    """Generate random query vectors."""
    print(f"\nGenerating {num_queries} query vectors...")
    rng = np.random.default_rng(12345)  # Different seed than DB
    queries = rng.random((num_queries, DIMENSION), dtype=np.float32)
    queries.tofile(QUERIES_FILE)
    print(f"  Saved to {QUERIES_FILE}")
    return queries


def compute_ground_truth(db_path, queries, top_k=1000):
    """Compute brute-force ground truth for queries."""
    print(f"\nComputing ground truth for {len(queries)} queries...")
    
    file_size = os.path.getsize(db_path)
    num_records = file_size // (DIMENSION * np.dtype(VECTOR_DTYPE).itemsize)
    print(f"  Database size: {num_records:,} vectors")
    
    vectors = np.memmap(db_path, dtype=VECTOR_DTYPE, mode='r', shape=(num_records, DIMENSION))
    
    # Normalize vectors for cosine similarity
    print("  Normalizing vectors...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    
    all_sorted_ids = []
    
    for i, query in enumerate(queries):
        if i % 10 == 0:
            print(f"  Query {i+1}/{len(queries)}...")
        
        # Normalize query
        query_norm = query / np.linalg.norm(query)
        
        # Compute cosine similarity in chunks to save memory
        chunk_size = 1_000_000
        scores = np.zeros(num_records, dtype=np.float32)
        
        for j in range(0, num_records, chunk_size):
            end = min(j + chunk_size, num_records)
            chunk = vectors[j:end]
            chunk_norms = norms[j:end].flatten()
            scores[j:end] = (chunk @ query_norm) / chunk_norms
        
        # Get sorted indices (descending)
        sorted_ids = np.argsort(scores)[::-1][:top_k].tolist()
        all_sorted_ids.append(sorted_ids)
    
    del vectors
    
    # Save ground truth
    np.save(GROUND_TRUTH_FILE, np.array(all_sorted_ids, dtype=object), allow_pickle=True)
    print(f"  Saved ground truth to {GROUND_TRUTH_FILE}")
    
    return all_sorted_ids


def generate_20m_database(path, size=20_000_000):
    """Generate the 20M database with random vectors."""
    print(f"Generating {size:,} vectors database...")
    rng = np.random.default_rng(DB_SEED_NUMBER)
    
    # Generate in chunks to avoid memory issues
    chunk_size = 1_000_000
    
    # Create the file
    mmap = np.memmap(path, dtype=VECTOR_DTYPE, mode='w+', shape=(size, DIMENSION))
    
    for i in range(0, size, chunk_size):
        end = min(i + chunk_size, size)
        mmap[i:end] = rng.random((end - i, DIMENSION), dtype=np.float32)
        print(f"  Generated {end:,} / {size:,} vectors")
    
    mmap.flush()
    del mmap
    print(f"  Done! Saved to {path}")


def main():
    print("=" * 60)
    print("VecDB Evaluation Setup")
    print("=" * 60)
    
    # Step 1: Check/Generate 20M database
    print("\n" + "=" * 40)
    print("Step 1: Check/Generate 20M database")
    print("=" * 40)
    
    if os.path.exists(PATH_DB_VECTORS_20M):
        file_size = os.path.getsize(PATH_DB_VECTORS_20M)
        num_records = file_size // (DIMENSION * np.dtype(VECTOR_DTYPE).itemsize)
        print(f"Found 20M database: {PATH_DB_VECTORS_20M}")
        print(f"  Size: {file_size / (1024**3):.2f} GB")
        print(f"  Records: {num_records:,}")
    else:
        print(f"20M database not found: {PATH_DB_VECTORS_20M}")
        print("Generating it now...")
        generate_20m_database(PATH_DB_VECTORS_20M)
    
    # Step 2: Create subsets
    print("\n" + "=" * 40)
    print("Step 2: Create 1M and 10M subsets")
    print("=" * 40)
    
    if os.path.exists(PATH_DB_VECTORS_10M):
        print(f"10M database already exists")
    else:
        create_subset(PATH_DB_VECTORS_20M, PATH_DB_VECTORS_10M, 10_000_000)
    
    if os.path.exists(PATH_DB_VECTORS_1M):
        print(f"1M database already exists")
    else:
        create_subset(PATH_DB_VECTORS_20M, PATH_DB_VECTORS_1M, 1_000_000)
    
    # Step 3: Build indices
    print("\n" + "=" * 40)
    print("Step 3: Build indices")
    print("=" * 40)
    
    # Use 8000 clusters for all to match your optimized settings
    configs = [
        (PATH_DB_VECTORS_1M, PATH_INDEX_1M, 1000),     # 1M: 1000 clusters
        (PATH_DB_VECTORS_10M, PATH_INDEX_10M, 4000),   # 10M: 4000 clusters
        (PATH_DB_VECTORS_20M, PATH_INDEX_20M, 8000),   # 20M: 8000 clusters
    ]
    
    for db_path, index_path, n_clusters in configs:
        meta_file = index_path + ".meta.json"
        if os.path.exists(meta_file):
            print(f"Index already exists: {meta_file}")
        else:
            start = time.time()
            build_index_only(db_path, index_path, n_clusters)
            elapsed = time.time() - start
            print(f"  Time: {elapsed:.1f}s")
    
    # Step 4: Generate queries and ground truth
    print("\n" + "=" * 40)
    print("Step 4: Generate queries and ground truth")
    print("=" * 40)
    
    if os.path.exists(QUERIES_FILE):
        print(f"Queries already exist: {QUERIES_FILE}")
        queries = np.fromfile(QUERIES_FILE, dtype=np.float32).reshape(-1, DIMENSION)
    else:
        queries = generate_queries(NUM_QUERIES)
    
    if os.path.exists(GROUND_TRUTH_FILE):
        print(f"Ground truth already exists: {GROUND_TRUTH_FILE}")
    else:
        compute_ground_truth(PATH_DB_VECTORS_20M, queries)
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {PATH_DB_VECTORS_20M}")
    print(f"  - {PATH_DB_VECTORS_10M}")
    print(f"  - {PATH_DB_VECTORS_1M}")
    print(f"  - {PATH_INDEX_20M}.*")
    print(f"  - {PATH_INDEX_10M}.*")
    print(f"  - {PATH_INDEX_1M}.*")
    print(f"  - {QUERIES_FILE}")
    print(f"  - {GROUND_TRUTH_FILE}")


if __name__ == "__main__":
    main()
