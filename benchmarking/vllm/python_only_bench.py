#!/usr/bin/env python3
"""Python-only benchmark: same vLLM actors, no C++ overhead.

Measures raw throughput of submit/poll on the shared vLLM actors.
This isolates whether the bottleneck is in C++ pipeline or the actors.
"""

import os
import sys
import threading
import time
from collections import deque

import pyarrow as pa
import pyarrow.fs
import pyarrow.parquet

# Ensure we can import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INPUT_LIMIT,
    MODEL_NAME,
    get_vllm_engine_args,
    init_ray_runtime,
    print_benchmark_results,
)


def main():
    import ray

    init_ray_runtime()
    input_limit = INPUT_LIMIT or 10000

    # Read data
    print(f"Reading data (limit={input_limit})...")
    t0 = time.perf_counter()
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
    print(f"Read {len(prompts)} prompts in {time.perf_counter() - t0:.2f}s")

    # Create actors — use DAFT's LocalVLLMExecutor for comparison
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from daft.execution.vllm import LocalVLLMExecutor as DaftLocalVLLMExecutor

    num_gpus = max(1, int(ray.cluster_resources().get("GPU", 1)))
    engine_args = get_vllm_engine_args()
    # Daft's LocalVLLMExecutor expects SamplingParams object, not dict
    from vllm import SamplingParams

    generate_args = {"sampling_params": SamplingParams(min_tokens=128, max_tokens=128)}

    print(f"Creating {num_gpus} vLLM actors (using Daft's LocalVLLMExecutor)...")
    actor_cls = ray.remote(num_gpus=1, max_restarts=4)(DaftLocalVLLMExecutor)
    actors = [actor_cls.remote(MODEL_NAME, engine_args, generate_args) for _ in range(num_gpus)]

    # Wait for engines to be ready by calling a simple method
    print("Waiting for actors to init engines...")
    ray.get([a.poll.remote() for a in actors])
    print("All actors ready")

    # Background poll threads (same as Vane's RemoteVLLMExecutor)
    result_buffer = deque()
    poll_stop = threading.Event()
    result_cv = threading.Condition(threading.Lock())
    total_polled = [0]
    inflight_per_actor = [0] * num_gpus
    inflight_lock = threading.Lock()
    poll_results_per_actor = [0] * num_gpus

    def poll_loop(actor_idx):
        actor = actors[actor_idx]
        while not poll_stop.is_set():
            try:
                result = ray.get(actor.poll.remote())
            except Exception as e:
                print(f"Poll error actor {actor_idx}: {e}")
                time.sleep(0.1)
                continue
            if result is not None:
                texts, _tbl = result
                n = len(texts) if texts else 0
                with inflight_lock:
                    inflight_per_actor[actor_idx] -= n
                poll_results_per_actor[actor_idx] += n
                result_buffer.append(result)
                with result_cv:
                    total_polled[0] += n
                    result_cv.notify_all()
                    if total_polled[0] % 1000 == 0:
                        elapsed = time.perf_counter() - start_time
                        rate = total_polled[0] / elapsed
                        print(
                            f"  polled {total_polled[0]}/{len(prompts)} "
                            f"@{elapsed:.1f}s ({rate:.0f} results/s) "
                            f"per_actor={poll_results_per_actor} "
                            f"inflight={inflight_per_actor}"
                        )
                time.sleep(0.05)  # Same as current Vane config
            else:
                time.sleep(0.1)

    # Start poll threads
    poll_threads = []
    for i in range(num_gpus):
        t = threading.Thread(target=poll_loop, args=(i,), daemon=True)
        t.start()
        poll_threads.append(t)

    # BENCHMARK: Submit all prompts with inflight-aware routing
    inflight_limit = 1024
    batch_size = 512
    print(
        f"\n=== Starting benchmark: {len(prompts)} prompts, "
        f"inflight_limit={inflight_limit}, batch_size={batch_size} ===\n"
    )

    start_time = time.perf_counter()
    submitted = 0

    while submitted < len(prompts):
        # Check inflight
        total_inflight = sum(inflight_per_actor)
        if total_inflight >= inflight_limit:
            # Wait for some results
            with result_cv:
                result_cv.wait(timeout=0.1)
            continue

        # How many to submit
        can_submit = min(batch_size, inflight_limit - total_inflight, len(prompts) - submitted)
        batch_prompts = prompts[submitted : submitted + can_submit]
        batch_rows = table.slice(submitted, can_submit)

        # Route to actor with lowest inflight
        with inflight_lock:
            route_to = min(range(num_gpus), key=lambda i: inflight_per_actor[i])
            inflight_per_actor[route_to] += len(batch_prompts)

        # Fire-and-forget submit
        actors[route_to].submit_async.remote(batch_prompts, batch_rows)
        submitted += len(batch_prompts)

        if submitted % 2000 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  submitted {submitted}/{len(prompts)} @{elapsed:.1f}s inflight={inflight_per_actor}")

    print(f"\nAll {submitted} prompts submitted in {time.perf_counter() - start_time:.2f}s")

    # Signal finished
    for actor in actors:
        actor.finished_submitting.remote()

    # Wait for all results
    with result_cv:
        while total_polled[0] < len(prompts):
            result_cv.wait(timeout=1.0)
            elapsed = time.perf_counter() - start_time
            remaining = len(prompts) - total_polled[0]
            rate = total_polled[0] / elapsed if elapsed > 0 else 0
            if remaining > 0:
                eta = remaining / rate if rate > 0 else float("inf")
                print(f"  waiting: {total_polled[0]}/{len(prompts)} @{elapsed:.1f}s ({rate:.0f}/s) ETA={eta:.0f}s")

    end_time = time.perf_counter()

    poll_stop.set()
    for t in poll_threads:
        t.join(timeout=2)

    elapsed = end_time - start_time
    print(f"\n=== RESULT: {len(prompts)} prompts in {elapsed:.2f}s ({len(prompts) / elapsed:.0f} results/s) ===")
    print(f"Per-actor results: {poll_results_per_actor}")
    print_benchmark_results("python_only_bench.py", start_time, end_time)


if __name__ == "__main__":
    main()
