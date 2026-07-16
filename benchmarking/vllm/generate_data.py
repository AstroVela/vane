import argparse
import os

DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"
DEFAULT_ROOT_DIR = "s3://vllm-prefix-caching-partitioned"


def configure_s3_env() -> dict[str, str]:
    os.environ.setdefault(
        "AWS_ENDPOINT_URL",
        "http://127.0.0.1:9000",
    )
    os.environ.setdefault(
        "AWS_ACCESS_KEY_ID",
        "",
    )
    os.environ.setdefault(
        "AWS_SECRET_ACCESS_KEY",
        "",
    )
    os.environ.setdefault(
        "AWS_REGION",
        "us-east-1",
    )
    os.environ.setdefault("AWS_DEFAULT_REGION", os.environ["AWS_REGION"])
    return {
        "AWS_ENDPOINT_URL": os.environ["AWS_ENDPOINT_URL"],
        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
        "AWS_REGION": os.environ["AWS_REGION"],
        "AWS_DEFAULT_REGION": os.environ["AWS_DEFAULT_REGION"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VLLM prefix-caching benchmark data and write it as parquet.",
    )
    parser.add_argument(
        "--request-k",
        type=int,
        required=True,
        help="Number of requests in thousands. For example, 200 produces 200000 prompts.",
    )
    parser.add_argument(
        "--num-prefixes",
        type=int,
        help="Number of shared prefixes. Required unless --no-prefix is set.",
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Generate prompts without shared prefixes.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Tokenizer model name.",
    )
    parser.add_argument(
        "--root-dir",
        default=DEFAULT_ROOT_DIR,
        help="Destination directory, for example s3://vllm-prefix-caching-partitioned.",
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=256,
        help="Prefix token length for prefix-sharing mode.",
    )
    parser.add_argument(
        "--suffix-len",
        type=int,
        default=256,
        help="Suffix token length for prefix-sharing mode.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Target output token length.",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=8,
        help="Number of parquet output partitions.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Dataset sampling seed.",
    )
    parser.add_argument(
        "--ray-address",
        default=os.getenv("RAY_ADDRESS", "auto"),
        help="Ray cluster address. Use 'auto' to connect to the existing cluster.",
    )
    args = parser.parse_args()
    if args.no_prefix == (args.num_prefixes is not None):
        parser.error("Specify exactly one of --num-prefixes or --no-prefix.")
    if args.request_k <= 0:
        parser.error("--request-k must be positive.")
    if args.num_prefixes is not None and args.num_prefixes <= 0:
        parser.error("--num-prefixes must be positive.")
    if args.partitions <= 0:
        parser.error("--partitions must be positive.")
    return args


def build_output_path(args: argparse.Namespace) -> str:
    root_dir = args.root_dir.rstrip("/")
    if args.no_prefix:
        return f"{root_dir}/{args.request_k}k_0.parquet"
    return f"{root_dir}/{args.request_k}k_0-5_{args.num_prefixes}.parquet"


def init_runtime(ray_address: str, env_vars: dict[str, str]) -> None:
    import daft
    import ray

    if not ray.is_initialized():
        ray.init(
            address=ray_address,
            runtime_env={"env_vars": env_vars},
        )
    daft.set_runner_ray()


def sample_prompts(args: argparse.Namespace) -> list[str]:
    from vllm.benchmarks.datasets import PrefixRepetitionRandomDataset
    from vllm.transformers_utils.tokenizer import get_tokenizer

    dataset = PrefixRepetitionRandomDataset(random_seed=args.random_seed)
    tokenizer = get_tokenizer(args.model_name)
    if args.no_prefix:
        sample = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.request_k * 1000,
            prefix_len=0,
            suffix_len=args.prefix_len + args.suffix_len,
            output_len=args.output_len,
        )
    else:
        sample = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.request_k * 1000,
            prefix_len=args.prefix_len,
            suffix_len=args.suffix_len,
            num_prefixes=args.num_prefixes,
            output_len=args.output_len,
        )
    return [entry.prompt for entry in sample]


def main() -> None:
    import daft
    from daft.functions import monotonically_increasing_id

    args = parse_args()
    env_vars = configure_s3_env()
    init_runtime(args.ray_address, env_vars)

    output_path = build_output_path(args)
    print(f"Generating prompts for {output_path}")
    prompts = sample_prompts(args)

    df = daft.from_pydict({"prompt": prompts})
    df = df.select(monotonically_increasing_id().alias("id"), "prompt")
    df = df.repartition(args.partitions)

    print(f"Writing parquet to {output_path}")
    df.write_parquet(output_path)
    print("Done")


if __name__ == "__main__":
    main()
