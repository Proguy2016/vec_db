"""
RAM Usage Monitor for VecDB retrieve() function
This measures peak memory used ONLY during retrieve(), not the whole script.
"""
import numpy as np
import tracemalloc
from vec_db import VecDB
import os

def measure_retrieve_ram(db, num_queries=5):
    """Measure peak RAM usage during retrieve() calls only."""
    peak_memories = []
    
    for i in range(num_queries):
        query = np.random.random((1, 64))
        
        # Start tracking memory
        tracemalloc.start()
        
        # Run retrieve
        results = db.retrieve(query, top_k=5)
        
        # Get peak memory
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / (1024 * 1024)
        peak_memories.append(peak_mb)
        print(f"  Query {i+1}: Peak RAM = {peak_mb:.2f} MB, returned {len(results)} results")
    
    avg_peak = sum(peak_memories) / len(peak_memories)
    max_peak = max(peak_memories)
    
    return avg_peak, max_peak

if __name__ == "__main__":
    import sys
    
    # Test different DB sizes
    test_configs = [
        ("db_vectors_1m.dat", "index_1m.dat", 20),   # 1M: 20MB limit
        ("db_vectors_10m.dat", "index_10m.dat", 50), # 10M: 50MB limit
        ("db_vectors_20m.dat", "index_20m.dat", 50), # 20M: 50MB limit
    ]
    
    for db_file, index_file, ram_limit in test_configs:
        db_path = os.path.join(os.path.dirname(__file__), db_file)
        index_path = os.path.join(os.path.dirname(__file__), index_file)
        
        if not os.path.exists(db_path):
            print(f"\nSkipping {db_file} - not found")
            continue
            
        print(f"\n{'='*50}")
        print(f"Testing: {db_file} (RAM limit: {ram_limit} MB)")
        print(f"{'='*50}")
        
        db = VecDB(database_file_path=db_path, index_file_path=index_path, new_db=False)
        print(f"n_probe = {db.n_probe}, n_clusters = {db.n_clusters}")
        
        avg_peak, max_peak = measure_retrieve_ram(db, num_queries=5)
        
        print("-" * 50)
        print(f"Average Peak RAM: {avg_peak:.2f} MB")
        print(f"Maximum Peak RAM: {max_peak:.2f} MB")
        
        if max_peak <= ram_limit:
            print(f"✅ PASS: Within {ram_limit}MB RAM limit")
        else:
            print(f"❌ FAIL: Exceeds {ram_limit}MB RAM limit")
