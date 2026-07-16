import os

from vane_common import run_vane_native_vllm_benchmark


def main():
    distributed = os.environ.get("BENCHMARK_DISTRIBUTED", "1") != "0"
    run_vane_native_vllm_benchmark(
        "vane-prefix-bucketing.py",
        do_prefix_routing=True,
        sorted_by_prompt=False,
        distributed=distributed,
    )


if __name__ == "__main__":
    main()
