import json
import os
from urllib.parse import urlparse

from vllm import SamplingParams

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(BENCHMARK_DIR))


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _get_int_env(name: str, default: int = 0) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _build_pythonpath() -> str:
    entries: list[str] = []
    existing = os.getenv("PYTHONPATH", "")
    if existing:
        entries.extend([entry for entry in existing.split(os.pathsep) if entry])
    for entry in (BENCHMARK_DIR, REPO_ROOT):
        if entry not in entries:
            entries.insert(0, entry)
    return os.pathsep.join(entries)


os.environ["VANE_RUNNER"] = os.getenv("VANE_RUNNER", "").strip() or "ray"
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
os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "0").strip())
os.environ.setdefault(
    "TRANSFORMERS_OFFLINE",
    os.getenv("TRANSFORMERS_OFFLINE", os.environ["HF_HUB_OFFLINE"]).strip(),
)
os.environ.setdefault(
    "VANE_RAY_SCAN_TASK_SIZE_GROUPING",
    os.getenv("VANE_RAY_SCAN_TASK_SIZE_GROUPING", "1").strip(),
)
os.environ.setdefault(
    "VANE_UDF_RAY_READY_TIMEOUT_SECS",
    os.getenv("VANE_UDF_RAY_READY_TIMEOUT_SECS", "180").strip(),
)
os.environ["PYTHONPATH"] = _build_pythonpath()


def _quote_sql(value: str) -> str:
    return value.replace("'", "''")


def _build_vane_ray_init_sql() -> str:
    endpoint_url = os.environ["AWS_ENDPOINT_URL"].strip()
    if "://" not in endpoint_url:
        endpoint_url = f"http://{endpoint_url}"
    parsed = urlparse(endpoint_url)
    endpoint = parsed.netloc or parsed.path
    use_ssl = parsed.scheme == "https"
    session_token = os.getenv("AWS_SESSION_TOKEN", "").strip()

    sqls = [
        f"SET s3_region='{_quote_sql(os.environ['AWS_REGION'])}'",
        f"SET s3_endpoint='{_quote_sql(endpoint)}'",
        f"SET s3_access_key_id='{_quote_sql(os.environ['AWS_ACCESS_KEY_ID'])}'",
        f"SET s3_secret_access_key='{_quote_sql(os.environ['AWS_SECRET_ACCESS_KEY'])}'",
    ]
    if session_token:
        sqls.append(f"SET s3_session_token='{_quote_sql(session_token)}'")
    sqls.extend(
        [
            f"SET s3_use_ssl={'true' if use_ssl else 'false'}",
            "SET s3_url_style='path'",
        ]
    )
    return "; ".join(sqls)


os.environ.setdefault("VANE_RAY_INIT_SQL", _build_vane_ray_init_sql())


def get_runtime_env_vars() -> dict[str, str]:
    env_vars = {
        "AWS_ENDPOINT_URL": os.environ["AWS_ENDPOINT_URL"],
        "AWS_ACCESS_KEY_ID": os.environ["AWS_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["AWS_SECRET_ACCESS_KEY"],
        "AWS_REGION": os.environ["AWS_REGION"],
        "AWS_DEFAULT_REGION": os.environ["AWS_DEFAULT_REGION"],
        "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
        "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
        "VANE_RUNNER": os.environ["VANE_RUNNER"],
        "VANE_RAY_SCAN_TASK_SIZE_GROUPING": os.environ["VANE_RAY_SCAN_TASK_SIZE_GROUPING"],
        "VANE_RAY_INIT_SQL": os.environ["VANE_RAY_INIT_SQL"],
        "VANE_UDF_RAY_READY_TIMEOUT_SECS": os.environ["VANE_UDF_RAY_READY_TIMEOUT_SECS"],
        "PYTHONPATH": os.environ["PYTHONPATH"],
    }
    session_token = os.getenv("AWS_SESSION_TOKEN", "").strip()
    if session_token:
        env_vars["AWS_SESSION_TOKEN"] = session_token
    return env_vars


def get_ray_runtime_env() -> dict[str, object]:
    return {
        "env_vars": get_runtime_env_vars(),
        "working_dir": BENCHMARK_DIR,
        "excludes": [
            "data",
            "__pycache__",
            "*.ipynb",
            "*.parquet",
        ],
    }


def init_ray_runtime() -> None:
    import ray

    if not ray.is_initialized():
        ray.init(
            address=os.getenv("RAY_ADDRESS", "auto"),
            log_to_driver=True,
            runtime_env=get_ray_runtime_env(),
        )


def init_daft_ray() -> None:
    import daft

    init_ray_runtime()
    daft.set_runner_ray()


def init_vane_runtime() -> None:
    import vane

    init_ray_runtime()
    vane.configure(runner="ray", ray_scan_task_size_grouping=1)


def get_cluster_gpu_count() -> int:
    import ray

    init_ray_runtime()
    return max(1, int(ray.cluster_resources().get("GPU", 1)))


def connect_vane():
    import vane

    init_vane_runtime()
    con = vane.connect()
    configure_vane_connection(con)
    return con


def configure_vane_connection(con) -> None:
    try:
        con.execute("LOAD httpfs")
    except Exception:
        try:
            con.execute("INSTALL httpfs")
            con.execute("LOAD httpfs")
        except Exception:
            pass

    for stmt in os.environ["VANE_RAY_INIT_SQL"].split("; "):
        if stmt:
            con.execute(stmt)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
INPUT_PATH = "s3://vllm-prefix-caching-partitioned/67k_0-5_512.parquet"
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.75"))
OUTPUT_LEN = 128
SAMPLING_PARAMS = SamplingParams(min_tokens=OUTPUT_LEN, max_tokens=OUTPUT_LEN)
CONCURRENCY = 128
NATIVE_BATCH_SIZE = 512
PARTITION_COUNT = 8
INPUT_LIMIT = max(0, int(os.getenv("INPUT_LIMIT", "0")))
VLLM_ACTOR_COUNT = max(0, _get_int_env("VLLM_ACTOR_COUNT", 0))
NAIVE_ACTOR_COUNT = max(0, _get_int_env("NAIVE_ACTOR_COUNT", 0))
VLLM_BATCH_SIZE = max(1, _get_int_env("VLLM_BATCH_SIZE", NATIVE_BATCH_SIZE))
NAIVE_BATCH_SIZE = max(1, _get_int_env("NAIVE_BATCH_SIZE", NATIVE_BATCH_SIZE))
NAIVE_PARTITION_COUNT = max(0, _get_int_env("NAIVE_PARTITION_COUNT", 0))
VLLM_ENFORCE_EAGER = _get_bool_env("VLLM_ENFORCE_EAGER", False)


def get_sampling_params_dict() -> dict[str, int]:
    return {
        "min_tokens": OUTPUT_LEN,
        "max_tokens": OUTPUT_LEN,
    }


def get_vllm_engine_args() -> dict[str, object]:
    engine_args: dict[str, object] = {
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
    }
    if VLLM_ENFORCE_EAGER:
        engine_args["enforce_eager"] = True
    return engine_args


def get_vane_native_actor_count() -> int:
    return VLLM_ACTOR_COUNT or get_cluster_gpu_count()


def get_vane_naive_actor_count() -> int:
    return NAIVE_ACTOR_COUNT or get_cluster_gpu_count()


def get_vane_naive_partition_count() -> int:
    """Number of DuckDB pipeline partitions for naive-batch benchmark.

    More partitions → more pipeline threads feeding the GPU actors,
    reducing data-starvation gaps.  Default: max(actor_count * 4, 8).
    """
    if NAIVE_PARTITION_COUNT > 0:
        return NAIVE_PARTITION_COUNT
    return max(get_vane_naive_actor_count() * 4, 8)


def build_vane_native_vllm_options(
    *,
    do_prefix_routing: bool,
    batch_size: int | None = VLLM_BATCH_SIZE,
) -> dict[str, object]:
    return {
        "use_ray": True,
        "concurrency": get_vane_native_actor_count(),
        "gpus_per_actor": 1,
        "do_prefix_routing": do_prefix_routing,
        "batch_size": batch_size,
        "engine_args": get_vllm_engine_args(),
        "generate_args": {
            "sampling_params": get_sampling_params_dict(),
        },
    }


def build_input_sql(*, sorted_by_prompt: bool = False) -> str:
    sql = f"SELECT id, prompt FROM read_parquet('{get_vane_input_path()}')"
    if sorted_by_prompt:
        sql += " ORDER BY prompt"
    if INPUT_LIMIT > 0:
        sql += f" LIMIT {INPUT_LIMIT}"
    return sql


def sql_literal(value: str) -> str:
    return value.replace("'", "''")


def json_sql_literal(value: dict[str, object]) -> str:
    return sql_literal(json.dumps(value))


def get_vane_input_path() -> str:
    if any(token in INPUT_PATH for token in ("*", "?", "[")):
        return INPUT_PATH
    if INPUT_PATH.endswith(".parquet"):
        return f"{INPUT_PATH.rstrip('/')}/**/*.parquet"
    return INPUT_PATH


def print_preview_rows(
    rows: list[tuple[object, ...]],
    *,
    limit: int = 8,
) -> None:
    print(f"Showing first {min(limit, len(rows))} rows")
    for row in rows[:limit]:
        preview = list(row[:5])
        normalized: list[object] = []
        for value in preview:
            if isinstance(value, str) and len(value) > 80:
                normalized.append(value[:77] + "...")
            else:
                normalized.append(value)
        print(tuple(normalized))


def print_benchmark_results(script: str, start_time: float, end_time: float):
    print("========== BENCHMARK RESULTS ==========")
    print(f"Script: {script}")
    print(f"Execution time: {end_time - start_time:.2f} seconds.")
    print()
    print("Benchmark configuration:")
    print(f"\t{MODEL_NAME=}")
    print(f"\t{INPUT_PATH=}")
    print(f"\t{MAX_MODEL_LEN=}")
    print(f"\t{GPU_MEMORY_UTILIZATION=}")
    print(f"\t{OUTPUT_LEN=}")
    print(f"\t{CONCURRENCY=}")
    print(f"\t{PARTITION_COUNT=}")
    print(f"\t{NATIVE_BATCH_SIZE=}")
    print(f"\t{INPUT_LIMIT=}")
    print(f"\t{VLLM_ACTOR_COUNT=}")
    print(f"\t{NAIVE_ACTOR_COUNT=}")
    print(f"\t{NAIVE_PARTITION_COUNT=}")
    print(f"\t{VLLM_BATCH_SIZE=}")
    print(f"\t{NAIVE_BATCH_SIZE=}")
    print(f"\t{VLLM_ENFORCE_EAGER=}")
    print(f"\tVANE_UDF_RAY_READY_TIMEOUT_SECS={os.environ['VANE_UDF_RAY_READY_TIMEOUT_SECS']}")
    print("======================================")
