import time

import daft
import ray
from config import (
    GPU_MEMORY_UTILIZATION,
    INPUT_LIMIT,
    INPUT_PATH,
    MAX_MODEL_LEN,
    MODEL_NAME,
    SAMPLING_PARAMS,
    init_daft_ray,
    print_benchmark_results,
)
from daft.functions import prompt


def main():
    print("Starting benchmark...")

    init_daft_ray()
    prefix_actor_count = max(1, int(ray.cluster_resources().get("GPU", 1)))

    df = daft.read_parquet(INPUT_PATH).into_partitions(8)
    if INPUT_LIMIT > 0:
        df = df.limit(INPUT_LIMIT)
    df = df.with_column(
        "output",
        prompt(
            df["prompt"],
            provider="vllm-prefix-caching",
            model=MODEL_NAME,
            engine_args={
                "max_model_len": MAX_MODEL_LEN,
                "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            },
            generate_args={
                "sampling_params": SAMPLING_PARAMS,
            },
            # The vllm-prefix-caching provider interprets concurrency as actor count.
            concurrency=prefix_actor_count,
        ),
    )

    print("Running benchmark...")
    start_time = time.perf_counter()
    df = df.collect()
    end_time = time.perf_counter()
    print("Benchmark completed!")

    df = df.with_columns(
        {
            "prompt_len": df["prompt"].length(),
            "output_len": df["output"].length(),
        }
    )
    df.show()

    print_benchmark_results("prefix-bucketing.py", start_time, end_time)


if __name__ == "__main__":
    main()
