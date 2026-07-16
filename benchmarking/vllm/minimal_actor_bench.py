#!/usr/bin/env python3
"""Minimal actor benchmark: stripped-down actor matching Daft's _generate exactly.

Tests whether Vane's extra overhead (error handling, condition vars) slows things down.
"""

import asyncio
import os
import sys
import threading
import time
from collections import deque

import pyarrow as pa
import pyarrow.fs
import pyarrow.parquet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    INPUT_LIMIT,
    MODEL_NAME,
    get_vllm_engine_args,
    init_ray_runtime,
    print_benchmark_results,
)


class MinimalVLLMExecutor:
    """Minimal actor matching Daft's LocalVLLMExecutor as closely as possible."""

    def __init__(self, model, engine_args, generate_args):
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

        # Daft-style: create engine synchronously in __init__
        args = AsyncEngineArgs(model=model, **engine_args)
        self.llm = AsyncLLMEngine.from_engine_args(args)

        sp = generate_args.get("sampling_params", {})
        if isinstance(sp, dict):
            self.sampling_params = SamplingParams(**sp)
        else:
            self.sampling_params = sp

        self.counter = 0
        self.counter_lock = threading.Lock()
        self.completed_tasks = deque()

        # Dedicated event loop (same as Daft)
        self.loop_ready = threading.Event()
        self.loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.loop_thread.start()
        self.loop_ready.wait()

    def _run_event_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop_ready.set()
        self.loop.run_forever()

    async def _generate(self, prompt, row):
        """Minimal _generate — matches Daft's exactly."""
        with self.counter_lock:
            rid = self.counter
            self.counter += 1

        final_output = None
        async for output in self.llm.generate(prompt, self.sampling_params, str(rid)):
            final_output = output

        output_text = final_output.outputs[0].text
        self.completed_tasks.append((output_text, row))

    async def submit_async(self, prompts, rows):
        if isinstance(rows, pa.RecordBatch):
            rows = pa.Table.from_batches([rows])

        for i in range(len(prompts)):
            prompt = prompts[i]
            row = rows.slice(i, 1)
            asyncio.run_coroutine_threadsafe(self._generate(prompt, row), self.loop)

    def poll(self):
        completed_outputs = []
        completed_rows = []
        while True:
            try:
                output, row = self.completed_tasks.popleft()
                completed_outputs.append(output)
                completed_rows.append(row)
            except IndexError:
                break

        if not completed_outputs:
            return None

        completed_rows_batch = pa.concat_tables(completed_rows)
        return completed_outputs, completed_rows_batch


def main():
    import ray

    init_ray_runtime()
    input_limit = INPUT_LIMIT or 10000

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

    num_gpus = max(1, int(ray.cluster_resources().get("GPU", 1)))
    engine_args = get_vllm_engine_args()
    generate_args = {"sampling_params": {"min_tokens": 128, "max_tokens": 128}}

    print(f"Creating {num_gpus} minimal vLLM actors...")
    actor_cls = ray.remote(num_gpus=1)(MinimalVLLMExecutor)
    actors = [actor_cls.remote(MODEL_NAME, engine_args, generate_args) for _ in range(num_gpus)]

    # Wait for engines
    print("Waiting for engines...")
    ray.get([a.poll.remote() for a in actors])
    print("Engines ready")

    # Same poll pattern as python_only_bench
    result_buffer = deque()
    poll_stop = threading.Event()
    result_cv = threading.Condition()
    total_polled = [0]
    inflight_per_actor = [0] * num_gpus
    inflight_lock = threading.Lock()
    poll_results_per_actor = [0] * num_gpus

    def poll_loop(actor_idx):
        actor = actors[actor_idx]
        while not poll_stop.is_set():
            try:
                result = ray.get(actor.poll.remote())
            except Exception:
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
                    if total_polled[0] % 2000 == 0:
                        elapsed = time.perf_counter() - start_time
                        rate = total_polled[0] / elapsed
                        print(
                            f"  polled {total_polled[0]}/{len(prompts)} "
                            f"@{elapsed:.1f}s ({rate:.0f}/s) "
                            f"per_actor={poll_results_per_actor}"
                        )
                time.sleep(0.05)
            else:
                time.sleep(0.1)

    poll_threads = []
    for i in range(num_gpus):
        t = threading.Thread(target=poll_loop, args=(i,), daemon=True)
        t.start()
        poll_threads.append(t)

    inflight_limit = 1024
    batch_size = 512
    print(f"\n=== Starting benchmark: {len(prompts)} prompts, inflight_limit={inflight_limit}, minimal actor ===\n")

    start_time = time.perf_counter()
    submitted = 0

    while submitted < len(prompts):
        total_inflight = sum(inflight_per_actor)
        if total_inflight >= inflight_limit:
            with result_cv:
                result_cv.wait(timeout=0.1)
            continue

        can_submit = min(batch_size, inflight_limit - total_inflight, len(prompts) - submitted)
        batch_prompts = prompts[submitted : submitted + can_submit]
        batch_rows = table.slice(submitted, can_submit)

        with inflight_lock:
            route_to = min(range(num_gpus), key=lambda i: inflight_per_actor[i])
            inflight_per_actor[route_to] += len(batch_prompts)

        actors[route_to].submit_async.remote(batch_prompts, batch_rows)
        submitted += len(batch_prompts)

    print(f"All {submitted} prompts submitted in {time.perf_counter() - start_time:.2f}s")

    with result_cv:
        while total_polled[0] < len(prompts):
            result_cv.wait(timeout=5.0)
            elapsed = time.perf_counter() - start_time
            rate = total_polled[0] / elapsed if elapsed > 0 else 0
            print(f"  waiting: {total_polled[0]}/{len(prompts)} @{elapsed:.1f}s ({rate:.0f}/s)")

    end_time = time.perf_counter()
    poll_stop.set()

    elapsed = end_time - start_time
    print(f"\n=== RESULT: {len(prompts)} prompts in {elapsed:.2f}s ({len(prompts) / elapsed:.0f}/s) ===")
    print(f"Per-actor: {poll_results_per_actor}")
    print_benchmark_results("minimal_actor_bench.py", start_time, end_time)


if __name__ == "__main__":
    main()
