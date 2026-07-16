import time

import daft
from config import (
    GPU_MEMORY_UTILIZATION,
    INPUT_LIMIT,
    INPUT_PATH,
    MAX_MODEL_LEN,
    MODEL_NAME,
    SAMPLING_PARAMS,
    get_cluster_gpu_count,
    init_daft_ray,
    print_benchmark_results,
)
from daft import Series
from vllm import LLM

NUM_GPUS = get_cluster_gpu_count()


@daft.cls(max_concurrency=NUM_GPUS, gpus=1)
class VLLM:
    def __init__(self):
        print("Initializing LLM...")
        start_time = time.perf_counter()
        self.llm = LLM(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            disable_log_stats=True,
        )
        end_time = time.perf_counter()
        print(f"LLM initialized in {end_time - start_time:.2f} seconds.")

    @daft.method.batch(return_dtype=str, batch_size=512)
    def generate(self, prompts: Series) -> Series:
        outputs = self.llm.generate(prompts.to_pylist(), SAMPLING_PARAMS)
        return Series.from_pylist([o.outputs[0].text for o in outputs])


def main():
    print("Starting benchmark...")

    init_daft_ray()

    df = daft.read_parquet(INPUT_PATH).into_partitions(32)
    if INPUT_LIMIT > 0:
        df = df.limit(INPUT_LIMIT)

    vllm = VLLM()
    df = df.with_column("output", vllm.generate(df["prompt"]))

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

    print_benchmark_results("naive-batch.py", start_time, end_time)


if __name__ == "__main__":
    main()
