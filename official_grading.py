# Official Grading Script - Exact recreation from TA notebook
import numpy as np
import os
import time
from dataclasses import dataclass
from typing import List
from memory_profiler import memory_usage
import gc
import tracemalloc

# Configuration
DIMENSION = 64
TEAM_NUMBER = "YOUR_TEAM_NUMBER"  # Update this

# Paths - Update these to match your setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_DB_VECTORS_20M = os.path.join(BASE_DIR, "OpenSubtitles_en_20M_emb_64.dat")
PATH_DB_VECTORS_10M = os.path.join(BASE_DIR, "db_vectors_10m.dat")
PATH_DB_VECTORS_1M = os.path.join(BASE_DIR, "db_vectors_1m.dat")
PATH_DB_20M = os.path.join(BASE_DIR, "index_20m.dat")
PATH_DB_10M = os.path.join(BASE_DIR, "index_10m.dat")
PATH_DB_1M = os.path.join(BASE_DIR, "index_1m.dat")

def create_other_DB_size(input_file, output_file, target_rows, embedding_dim=DIMENSION):
    dtype = 'float32'
    file_size_bytes = os.path.getsize(input_file)
    itemsize = np.dtype(dtype).itemsize
    total_rows = file_size_bytes // (embedding_dim * itemsize)

    print(f"Source detected: {total_rows} rows.")

    source_memmap = np.memmap(input_file, dtype=dtype, mode='r', shape=(total_rows, embedding_dim))
    dest_memmap = np.memmap(output_file, dtype=dtype, mode='w+', shape=(target_rows, embedding_dim))

    print("Copying data...")
    dest_memmap[:] = source_memmap[:target_rows]
    dest_memmap.flush()

    print(f"Success! Saved first {target_rows} rows to {output_file}")

# Load or create queries
queries_embed_file = os.path.join(BASE_DIR, "queries_emb_64.dat")

# Always generate exactly 4 queries for evaluation (1 dummy + 3 eval)
print("Generating evaluation queries...")
np.random.seed(12345)
queries_np = np.random.random((4, DIMENSION)).astype(np.float32)
queries_np.tofile(queries_embed_file)

query_dummy = queries_np[0].reshape(1, DIMENSION)
queries = [queries_np[1].reshape(1, DIMENSION), queries_np[2].reshape(1, DIMENSION), queries_np[3].reshape(1, DIMENSION)]
queries_np_eval = queries_np[1:, :]  # For ground truth computation (3 queries)

# Load or compute ground truth
actual_sorted_ids_file = os.path.join(BASE_DIR, "actual_sorted_ids_20m.dat")
saved_top_k = 30_000
needed_top_k = 10_000

if not os.path.exists(actual_sorted_ids_file):
    print("Computing ground truth (this takes a while)...")
    vectors = np.memmap(PATH_DB_VECTORS_20M, dtype='float32', mode='r', shape=(20_000_000, DIMENSION))
    actual_sorted_ids_20m = np.argsort(
        np.dot(vectors, queries_np_eval.T) / (1e-45 + np.linalg.norm(vectors, axis=1)[:, None] * np.linalg.norm(queries_np_eval, axis=1)),
        axis=0
    )[-saved_top_k:][::-1].T
    actual_sorted_ids_20m = actual_sorted_ids_20m.astype(np.int32)
    actual_sorted_ids_20m.tofile(actual_sorted_ids_file)
    print("Ground truth saved.")
else:
    actual_sorted_ids_20m = np.fromfile(actual_sorted_ids_file, dtype=np.int32).reshape(-1, saved_top_k)
    print(f"Loaded ground truth: shape {actual_sorted_ids_20m.shape}")

@dataclass
class Result:
    run_time: float
    top_k: int
    db_ids: List[int]
    actual_ids: List[int]

results = []

def run_queries(db, queries, top_k, actual_ids, num_runs):
    global results
    results = []
    for i in range(num_runs):
        tic = time.time()
        db_ids = db.retrieve(queries[i], top_k)
        toc = time.time()
        run_time = toc - tic
        results.append(Result(run_time, top_k, db_ids, actual_ids[i]))
    return results

def memory_usage_run_queries(args):
    global results
    mem_before = max(memory_usage())
    mem = memory_usage(proc=(run_queries, args, {}), interval=1e-3)
    return results, max(mem) - mem_before

def evaluate_result(results: List[Result]):
    scores = []
    run_time = []
    for res in results:
        run_time.append(res.run_time)
        if len(set(res.db_ids)) != res.top_k or len(res.db_ids) != res.top_k:
            scores.append(-1 * len(res.actual_ids) * res.top_k)
            continue
        score = 0
        for id in res.db_ids:
            try:
                ind = res.actual_ids.index(id)
                if ind > res.top_k * 3:
                    score -= ind
            except:
                score -= len(res.actual_ids)
        scores.append(score)

    return sum(scores) / len(scores), sum(run_time) / len(run_time)

def get_actual_ids_first_k(actual_sorted_ids, k, out_len=10_000):
    return [[id for id in actual_sorted_ids_one_q if id < k] for actual_sorted_ids_one_q in actual_sorted_ids][:out_len]


if __name__ == "__main__":
    # Check memory usage for import
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()

    from vec_db import VecDB

    end_snapshot = tracemalloc.take_snapshot()
    stats = end_snapshot.compare_to(start_snapshot, 'lineno')
    for stat in stats[:5]:
        print(stat)
    tracemalloc.stop()

    results = []
    to_print_arr = []
    print("Team Number", TEAM_NUMBER)

    database_info = {
        "1M": {
            "database_file_path": PATH_DB_VECTORS_1M,
            "index_file_path": PATH_DB_1M,
            "size": 10**6
        },
        "10M": {
            "database_file_path": PATH_DB_VECTORS_10M,
            "index_file_path": PATH_DB_10M,
            "size": 10 * 10**6
        },
        "20M": {
            "database_file_path": PATH_DB_VECTORS_20M,
            "index_file_path": PATH_DB_20M,
            "size": 20 * 10**6
        }
    }

    for db_name, info in database_info.items():
        if not os.path.exists(info["database_file_path"]):
            print(f"Skipping {db_name} - file not found: {info['database_file_path']}")
            continue
            
        print(f"*" * 40)
        print(f"Evaluating DB of size {db_name}")

        # Check RAM usage for class init
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()

        db = VecDB(database_file_path=info["database_file_path"], index_file_path=info["index_file_path"], new_db=False)

        end_snapshot = tracemalloc.take_snapshot()
        stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        for stat in stats[:5]:
            print(stat)
        tracemalloc.stop()

        actual_ids = get_actual_ids_first_k(actual_sorted_ids_20m, info["size"], needed_top_k)

        # Warm-up run
        tracemalloc.start()
        start_snapshot = tracemalloc.take_snapshot()

        res = run_queries(db, [query_dummy], 5, actual_ids, 1)

        end_snapshot = tracemalloc.take_snapshot()
        stats = end_snapshot.compare_to(start_snapshot, 'lineno')
        for stat in stats[:5]:
            print(stat)
        tracemalloc.stop()

        # Actual evaluation runs
        res, mem = memory_usage_run_queries((db, queries, 5, actual_ids, 3))
        eval_result = evaluate_result(res)
        to_print = f"{db_name}\tscore\t{eval_result[0]}\ttime\t{eval_result[1]:.2f}\tRAM\t{mem:.2f} MB"
        print(to_print)
        to_print_arr.append(to_print)

        del db
        del actual_ids
        del res
        del mem
        del eval_result
        gc.collect()

    print("Team Number", TEAM_NUMBER)
    print("\n".join(to_print_arr))
