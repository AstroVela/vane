import time

import ray
from config import (
    GPU_MEMORY_UTILIZATION,
    INPUT_LIMIT,
    INPUT_PATH,
    MAX_MODEL_LEN,
    MODEL_NAME,
    NATIVE_BATCH_SIZE,
    SAMPLING_PARAMS,
    get_cluster_gpu_count,
    init_ray_runtime,
    print_benchmark_results,
)


class VLLMPredictor:
    def __init__(self):
        from vllm import LLM

        print("Initializing LLM...")
        start = time.perf_counter()
        self.llm = LLM(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            enable_prefix_caching=True,
            disable_log_stats=True,
        )
        print(f"LLM initialized in {time.perf_counter() - start:.2f}s")

    def __call__(self, batch):
        outputs = self.llm.generate(batch["prompt"].tolist(), SAMPLING_PARAMS)
        batch["output"] = [o.outputs[0].text for o in outputs]
        return batch


def main():
    print("Starting benchmark...")

    init_ray_runtime()

    num_gpus = get_cluster_gpu_count()
    print(f"Cluster GPUs: {num_gpus}")

    print("Running benchmark...")
    start_time = time.perf_counter()

    ds = ray.data.read_parquet(INPUT_PATH)
    if INPUT_LIMIT > 0:
        ds = ds.limit(INPUT_LIMIT)
    ds = ds.map_batches(
        VLLMPredictor,
        batch_size=NATIVE_BATCH_SIZE,
        num_gpus=1,
        concurrency=num_gpus,
    )
    ds.take_all()

    end_time = time.perf_counter()

    print_benchmark_results("ray-data.py", start_time, end_time)


if __name__ == "__main__":
    main()
