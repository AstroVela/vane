#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Multi-caller benchmark: 8 threads call submit/poll like Daft's 8 partitions.

Tests whether having multiple callers improves throughput vs single caller.
Uses Vane's LocalVLLMExecutor (pa.Table compatible).
"""

import os
import sys
import threading
import time

import pyarrow as pa
import pyarrow.fs
import pyarrow.parquet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INPUT_LIMIT,
    MODEL_NAME,
    get_sampling_params_dict,
    get_vllm_engine_args,
    init_ray_runtime,
    print_benchmark_results,
)


def worker_fn(worker_id, actor_refs, prompts, table, results, lock, start_event):
    """Simulate one Daft partition task: submit then poll in a loop."""
    import ray

    start_event.wait()

    n_actors = len(actor_refs)
    n_prompts = len(prompts)
    submitted = 0
    polled = 0
    actor_idx = worker_id % n_actors  # simple round-robin start

    batch_size = 128
    while submitted < n_prompts:
        # Submit a batch
        end = min(submitted + batch_size, n_prompts)
        batch_prompts = prompts[submitted:end]
        batch_rows = table.slice(submitted, end - submitted)

        # Route like Daft: route via PrefixRouter (simplified: just round-robin)
        target = actor_idx
        actor_idx = (actor_idx + 1) % n_actors

        # Blocking submit (like Daft's ray.get(actor.submit_async.remote()))
        ray.get(actor_refs[target].submit_async.remote(batch_prompts, batch_rows))
        submitted += len(batch_prompts)

    # Now poll until we get all results
    while polled < n_prompts:
        for i in range(n_actors):
            result = ray.get(actor_refs[i].poll.remote())
            if result is not None:
                texts, _tbl = result
                n = len(texts) if texts else 0
                polled += n
                with lock:
                    results.append(n)

    return polled


def main():
    import ray

    init_ray_runtime()
    input_limit = INPUT_LIMIT or 10000
    n_workers = 8  # Match Daft's 8 partitions

    # Read data
    print(f"Reading data (limit={input_limit})...")
    table = pa.parquet.read_table(
        "vllm-prefix-caching-partitioned/67k_0-5_512.parquet",
        filesystem=pa.fs.S3FileSystem(
            endpoint_override=os.environ.get("AWS_ENDPOINT_URL", "http://127.0.0.1:9000"),
            access_key=os.environ.get("AWS_ACCESS_KEY_ID", ""),
            secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            scheme="http",
            region=os.environ.get("AWS_REGION", "us-east-1"),
        ),
    )
    if input_limit > 0 and len(table) > input_limit:
        table = table.slice(0, input_limit)
    prompts = table.column("prompt").to_pylist()
    print(f"Read {len(prompts)} prompts")

    # Create actors (Vane's LocalVLLMExecutor)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from vane.execution.vllm import LocalVLLMExecutor

    num_gpus = max(1, int(ray.cluster_resources().get("GPU", 1)))
    engine_args = get_vllm_engine_args()
    generate_args = {"sampling_params": get_sampling_params_dict()}

    print(f"Creating {num_gpus} vLLM actors...")
    actor_cls = ray.remote(num_gpus=1)(LocalVLLMExecutor)
    actors = [actor_cls.remote(MODEL_NAME, engine_args, generate_args) for _ in range(num_gpus)]

    # Wait for engines
    print("Waiting for engines...")
    ray.get([a.poll.remote() for a in actors])
    print("Engines ready")

    # Split data across workers
    chunk_size = (len(prompts) + n_workers - 1) // n_workers
    worker_prompts = []
    worker_tables = []
    for i in range(n_workers):
        start = i * chunk_size
        end = min(start + chunk_size, len(prompts))
        if start >= len(prompts):
            break
        worker_prompts.append(prompts[start:end])
        worker_tables.append(table.slice(start, end - start))

    actual_workers = len(worker_prompts)
    print(f"\n=== Starting benchmark: {len(prompts)} prompts, {actual_workers} workers, {num_gpus} GPUs ===\n")

    results = []
    lock = threading.Lock()
    start_event = threading.Event()

    threads = []
    for i in range(actual_workers):
        t = threading.Thread(
            target=worker_fn,
            args=(i, actors, worker_prompts[i], worker_tables[i], results, lock, start_event),
        )
        t.start()
        threads.append(t)

    start_time = time.perf_counter()
    start_event.set()

    # Monitor progress
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        with lock:
            total = sum(results)
        elapsed = time.perf_counter() - start_time
        rate = total / elapsed if elapsed > 0 else 0
        remaining = len(prompts) - total
        eta = remaining / rate if rate > 0 else float("inf")
        print(f"  progress: {total}/{len(prompts)} @{elapsed:.1f}s ({rate:.0f}/s) ETA={eta:.0f}s")
        if total >= len(prompts):
            break

    for t in threads:
        t.join(timeout=5)

    end_time = time.perf_counter()
    with lock:
        total = sum(results)

    # Signal finished
    for a in actors:
        a.finished_submitting.remote()

    elapsed = end_time - start_time
    print(f"\n=== RESULT: {total} results in {elapsed:.2f}s ({total / elapsed:.0f}/s) ===")
    print_benchmark_results("multi_caller_bench.py", start_time, end_time)


if __name__ == "__main__":
    main()
