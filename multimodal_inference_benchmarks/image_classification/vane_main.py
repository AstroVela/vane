from __future__ import annotations

import io
import os
import sys
import time
import uuid
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from torchvision import transforms as tv_transforms
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as tv_F

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("VANE_RUNNER", "ray")
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
os.environ.setdefault("AWS_REGION", "us-east-1")
os.environ.setdefault("AWS_DEFAULT_REGION", os.environ["AWS_REGION"])
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

for _proxy_key in (
    "http_proxy",
    "HTTP_PROXY",
    "https_proxy",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
):
    os.environ.pop(_proxy_key, None)
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

if PROJECT_ROOT not in os.getenv("PYTHONPATH", "").split(os.pathsep):
    os.environ["PYTHONPATH"] = os.pathsep.join([PROJECT_ROOT, os.getenv("PYTHONPATH", "")]).strip(os.pathsep)

import vane

USE_RAY = (os.getenv("VANE_RUNNER", "").strip().lower() or "ray") == "ray"
_RAY_STARTED_BY_SCRIPT = False


def _read_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _read_int_env(name: str, default: int, minimum: int | None = None) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        result = default
    else:
        try:
            result = int(value)
        except Exception:
            result = default
    if minimum is not None:
        result = max(minimum, result)
    return result


def _read_first_int_env(
    names: tuple[str, ...],
    default: int,
    minimum: int | None = None,
) -> int:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return _read_int_env(name, default, minimum=minimum)
    if minimum is not None:
        return max(minimum, default)
    return default


def _read_optional_text_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        text = value.strip()
        if text:
            return text
    return None


def _read_subprocess_backend(names: tuple[str, ...], default: str) -> str:
    value = (_read_optional_text_env(names) or default).strip().lower()
    allowed = {"subprocess_task", "subprocess_actor"}
    if value not in allowed:
        raise ValueError(f"subprocess backend must be one of {sorted(allowed)}, got {value!r}")
    return value


def _split_csv_env(value: str | None) -> list[str]:
    if value is None:
        return []
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def _default_cpu_partition_count() -> int:
    worker_slots = os.getenv("VANE_DISTRIBUTED_WORKER_SLOTS")
    if worker_slots is not None and worker_slots.strip():
        try:
            return max(1, int(worker_slots))
        except Exception:
            pass
    return max(1, int(os.cpu_count() or 16))


def _default_gpu_actor_count() -> int:
    if USE_RAY:
        try:
            import ray

            gpu_count = int(float(ray.cluster_resources().get("GPU", 0)))
            if gpu_count > 0:
                return gpu_count
        except Exception:
            pass
    try:
        gpu_count = int(torch.cuda.device_count())
        if gpu_count > 0:
            return gpu_count
    except Exception:
        pass
    return 1


def _default_local_gpu_actor_count() -> int:
    visible_devices = _split_csv_env(os.getenv("CUDA_VISIBLE_DEVICES"))
    if visible_devices:
        return len(visible_devices)
    count = _default_gpu_actor_count()
    return max(1, count)


def _resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return path
    if "://" in path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _is_s3_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def _normalize_parquet_input(path: str) -> str:
    if any(ch in path for ch in ("*", "?", "[")):
        return path
    lower = path.lower()
    if lower.endswith((".parquet", ".parquet.gz")):
        return path
    if _is_s3_path(path):
        return path.rstrip("/") + "/**/*.parquet"
    if os.path.isdir(path):
        return os.path.join(path.rstrip("/"), "**", "*.parquet")
    return path


def _maybe_create_local_output_dir(path: str) -> None:
    if "://" in path:
        return
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)


def _quote(value: str) -> str:
    return value.replace("'", "''")


def _s3_connection_config() -> tuple[str, str, str, str, str, str, bool]:
    endpoint_url = os.getenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9000").strip()
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    session_token = os.getenv("AWS_SESSION_TOKEN", "").strip()
    region = (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1").strip()
    if "://" not in endpoint_url:
        endpoint_url = f"http://{endpoint_url}"
    parsed = urlparse(endpoint_url)
    endpoint = parsed.netloc or parsed.path
    use_ssl = parsed.scheme == "https"
    return endpoint_url, endpoint, access_key, secret_key, session_token, region, use_ssl


def _configure_vane_s3(con, paths: tuple[str, ...] = ()) -> None:
    if paths and not any(_is_s3_path(path) for path in paths):
        return
    try:
        con.execute("LOAD httpfs")
    except Exception:
        try:
            con.execute("INSTALL httpfs")
            con.execute("LOAD httpfs")
        except Exception:
            pass

    _, endpoint, access_key, secret_key, session_token, region, use_ssl = _s3_connection_config()
    init_sqls = [
        f"SET s3_region='{_quote(region)}'",
        f"SET s3_endpoint='{_quote(endpoint)}'",
        f"SET s3_access_key_id='{_quote(access_key)}'",
        f"SET s3_secret_access_key='{_quote(secret_key)}'",
    ]
    if session_token:
        init_sqls.append(f"SET s3_session_token='{_quote(session_token)}'")
    init_sqls.append(f"SET s3_use_ssl={'true' if use_ssl else 'false'}")
    init_sqls.append("SET s3_url_style='path'")
    for stmt in init_sqls:
        con.execute(stmt)
    os.environ["VANE_RAY_INIT_SQL"] = "; ".join(init_sqls)


def _s3_filesystem():
    import pyarrow.fs as pa_fs

    _, endpoint, access_key, secret_key, session_token, region, use_ssl = _s3_connection_config()
    kwargs: dict[str, object] = {
        "endpoint_override": endpoint,
        "region": region,
        "scheme": "https" if use_ssl else "http",
    }
    if access_key or secret_key or session_token:
        kwargs["access_key"] = access_key
        kwargs["secret_key"] = secret_key
        if session_token:
            kwargs["session_token"] = session_token
        kwargs["anonymous"] = False
    else:
        kwargs["anonymous"] = True
    return pa_fs.S3FileSystem(**kwargs)


def _read_s3_bytes(path: str) -> bytes:
    fs = _s3_filesystem()
    object_path = path[len("s3://") :]
    with fs.open_input_file(object_path) as handle:
        return handle.read()


def _ensure_ray_initialized() -> None:
    if not USE_RAY:
        return
    import ray

    global _RAY_STARTED_BY_SCRIPT
    if ray.is_initialized():
        return
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    env_vars = {
        "http_proxy": "",
        "HTTP_PROXY": "",
        "https_proxy": "",
        "HTTPS_PROXY": "",
        "all_proxy": "",
        "ALL_PROXY": "",
        "no_proxy": "*",
        "NO_PROXY": "*",
        "PYTHONPATH": os.environ.get("PYTHONPATH", PROJECT_ROOT),
        "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL", ""),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "AWS_SESSION_TOKEN": os.environ.get("AWS_SESSION_TOKEN", ""),
        "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
        "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "0"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "0"),
        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": os.environ.get(
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO",
            "0",
        ),
    }
    env_vars.update(
        {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("VANE_", "DUCKDB_", "HF_", "TRANSFORMERS_"))
        }
    )
    ray_address = os.getenv("RAY_ADDRESS", "auto").strip()
    try:
        ray.init(
            address=ray_address,
            ignore_reinit_error=True,
            runtime_env={"env_vars": env_vars},
        )
    except Exception:
        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": env_vars})
        _RAY_STARTED_BY_SCRIPT = True


def _shutdown_ray() -> None:
    if not USE_RAY:
        return
    if not _read_bool_env("SHUTDOWN_RAY", True):
        return
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


BATCH_SIZE = _read_int_env("BATCH_SIZE", 100, minimum=1)
CPU_BATCH_SIZE = _read_int_env("CPU_UDF_INPUT_BATCH_SIZE", BATCH_SIZE, minimum=1)
CPU_UDF_TIMING_LOG = _read_bool_env("IMAGE_CPU_UDF_TIMING_LOG", _read_bool_env("CPU_UDF_TIMING_LOG", False))
CPU_UDF_TIMING_SAMPLE_RATE = _read_int_env("IMAGE_CPU_UDF_TIMING_SAMPLE_RATE", 1, minimum=1)
GPU_UDF_TIMING_LOG = _read_bool_env("IMAGE_GPU_UDF_TIMING_LOG", False)
GPU_UDF_TIMING_SAMPLE_RATE = _read_int_env("IMAGE_GPU_UDF_TIMING_SAMPLE_RATE", 1, minimum=1)
UDF_TIMING_LOG_PATH = _read_optional_text_env(("IMAGE_UDF_TIMING_LOG_PATH", "UDF_TIMING_LOG_PATH"))
BENCHMARK_RUN_ID = os.getenv("IMAGE_BENCHMARK_RUN_ID", "-").strip() or "-"
FULL_INPUT_ROW_COUNT = 803580
INPUT_LIMIT = _read_int_env("INPUT_LIMIT", FULL_INPUT_ROW_COUNT, minimum=0)
INPUT_OFFSET = _read_int_env("INPUT_OFFSET", 0, minimum=0)
EXECUTION_WIDTH = _read_int_env("IMAGE_EXECUTION_WIDTH", 0, minimum=0)
CPU_UDF_STREAMING_BREAKER = True
GPU_UDF_STREAMING_BREAKER = True
LOCAL_CPU_UDF_BACKEND = _read_subprocess_backend(
    ("IMAGE_CPU_SUBPROCESS_BACKEND", "CPU_SUBPROCESS_BACKEND"),
    "subprocess_task",
)
LOCAL_GPU_UDF_BACKEND = _read_subprocess_backend(
    ("IMAGE_GPU_SUBPROCESS_BACKEND", "GPU_SUBPROCESS_BACKEND"),
    "subprocess_actor",
)


def _cpu_udf_streaming_kwargs() -> dict[str, object]:
    if not CPU_UDF_STREAMING_BREAKER:
        return {}
    return {"streaming_breaker": True}


def _gpu_udf_streaming_kwargs() -> dict[str, object]:
    if not GPU_UDF_STREAMING_BREAKER:
        return {}
    return {"streaming_breaker": True}


_DEFAULT_INPUT_PATH = "s3://datasets/multimodal_inference_benchmarks/imagenet/metadata_file_rg10000.parquet"
_DEFAULT_IMAGE_ROOT = "s3://datasets/multimodal_inference_benchmarks/imagenet/train"
INPUT_PATH = _resolve_path(
    os.getenv("INPUT_PATH") or _DEFAULT_INPUT_PATH,
    SCRIPT_DIR,
)
LOCAL_IMAGE_ROOT = _resolve_path(
    os.getenv("LOCAL_IMAGE_ROOT") or _DEFAULT_IMAGE_ROOT,
    SCRIPT_DIR,
)
OUTPUT_PATH = _resolve_path(
    os.getenv("OUTPUT_PATH")
    or f"s3://datasets/multimodal_inference_benchmarks/image_classification_output/{uuid.uuid4().hex}",
    SCRIPT_DIR,
)
if "://" not in OUTPUT_PATH and not OUTPUT_PATH.lower().endswith(".parquet"):
    OUTPUT_PATH = os.path.join(OUTPUT_PATH.rstrip("/"), "result.parquet")
S3_IMAGE_PREFIX = "s3://ray-example-data/imagenet/train"

IMAGE_DIM = (3, 224, 224)
NORM_IMAGE_SQL_TYPE = vane.tensor_type(vane.sqltypes.FLOAT, IMAGE_DIM)

MODEL_WEIGHTS = ResNet18_Weights.DEFAULT
weights = MODEL_WEIGHTS
_to_tensor_transform = tv_transforms.ToTensor()
_weights_transform = weights.transforms()
transform = tv_transforms.Compose([_to_tensor_transform, _weights_transform])

_S3_CLIENT = None
_CPU_UDF_TIMING_CALLS = 0


def _localize_image_path_sql() -> str:
    return (
        "CASE WHEN image_url LIKE '"
        + S3_IMAGE_PREFIX
        + "/%' THEN replace(image_url, '"
        + S3_IMAGE_PREFIX
        + "', '"
        + LOCAL_IMAGE_ROOT
        + "') ELSE image_url END AS image_url"
    )


def _is_http_url(path: str) -> bool:
    return urlparse(path).scheme in ("http", "https")


def _download_image_bytes(path: str) -> bytes:
    if _is_s3_path(path):
        return _read_s3_bytes(path)
    if _is_http_url(path):
        with urlopen(path, timeout=15) as response:
            return response.read()
    with open(path, "rb") as handle:
        return handle.read()


def _process_thread_count() -> int:
    try:
        with open("/proc/self/status") as status_file:
            for line in status_file:
                if line.startswith("Threads:"):
                    return int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return -1


def _format_seconds_and_ms(name: str, seconds: float) -> str:
    value = max(0.0, float(seconds))
    return f"{name}_s={value:.6f} {name}_ms={value * 1000.0:.3f}"


def _emit_udf_timing_line(line: str) -> None:
    print(line, file=sys.stderr, flush=True)
    if not UDF_TIMING_LOG_PATH:
        return
    try:
        log_path = os.path.abspath(os.path.expanduser(UDF_TIMING_LOG_PATH))
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
    except Exception as exc:
        print(
            "[vane_image][udf_timing_log_error] "
            f"run_id={BENCHMARK_RUN_ID} pid={os.getpid()} path={UDF_TIMING_LOG_PATH!r} error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _format_cpu_udf_timing_fields(
    *,
    read_s: float,
    decode_s: float,
    transform_s: float,
    stack_s: float,
    arrow_s: float,
    total_s: float,
    rows_per_s: float,
    transform_to_tensor_s: float,
    transform_resize_s: float,
    transform_center_crop_s: float,
    transform_convert_dtype_s: float,
    transform_normalize_s: float,
    transform_numpy_s: float,
) -> str:
    fields = [
        _format_seconds_and_ms("read", read_s),
        _format_seconds_and_ms("decode", decode_s),
        _format_seconds_and_ms("transform", transform_s),
        _format_seconds_and_ms("stack", stack_s),
        _format_seconds_and_ms("arrow", arrow_s),
        _format_seconds_and_ms("total", total_s),
        f"rows_per_s={float(rows_per_s):.2f}",
        _format_seconds_and_ms("transform_to_tensor", transform_to_tensor_s),
        _format_seconds_and_ms("transform_resize", transform_resize_s),
        _format_seconds_and_ms("transform_center_crop", transform_center_crop_s),
        _format_seconds_and_ms("transform_convert_dtype", transform_convert_dtype_s),
        _format_seconds_and_ms("transform_normalize", transform_normalize_s),
        _format_seconds_and_ms("transform_numpy", transform_numpy_s),
    ]
    return " ".join(fields)


def _transform_image_timed(image: Image.Image):
    stage_start = time.perf_counter()
    tensor = _to_tensor_transform(image)
    to_tensor_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    tensor = tv_F.resize(
        tensor,
        _weights_transform.resize_size,
        interpolation=_weights_transform.interpolation,
        antialias=_weights_transform.antialias,
    )
    resize_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    tensor = tv_F.center_crop(tensor, _weights_transform.crop_size)
    center_crop_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    tensor = tv_F.convert_image_dtype(tensor, torch.float)
    convert_dtype_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    tensor = tv_F.normalize(tensor, mean=_weights_transform.mean, std=_weights_transform.std)
    normalize_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    array = np.ascontiguousarray(tensor.numpy())
    numpy_s = time.perf_counter() - stage_start

    return array, to_tensor_s, resize_s, center_crop_s, convert_dtype_s, normalize_s, numpy_s


def _decode_and_transform(table):
    global _CPU_UDF_TIMING_CALLS

    total_start = time.perf_counter()
    image_urls = table.column("image_url").to_pylist()
    arrays: list[np.ndarray] = []
    payload_bytes = 0
    failures = 0
    read_s = 0.0
    decode_s = 0.0
    transform_s = 0.0
    transform_to_tensor_s = 0.0
    transform_resize_s = 0.0
    transform_center_crop_s = 0.0
    transform_convert_dtype_s = 0.0
    transform_normalize_s = 0.0
    transform_numpy_s = 0.0
    for path in image_urls:
        try:
            stage_start = time.perf_counter()
            payload = _download_image_bytes(path)
            read_s += time.perf_counter() - stage_start
            payload_bytes += len(payload)

            stage_start = time.perf_counter()
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            decode_s += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            (
                array,
                to_tensor_s,
                resize_s,
                center_crop_s,
                convert_dtype_s,
                normalize_s,
                numpy_s,
            ) = _transform_image_timed(image)
            arrays.append(array)
            transform_s += time.perf_counter() - stage_start
            transform_to_tensor_s += to_tensor_s
            transform_resize_s += resize_s
            transform_center_crop_s += center_crop_s
            transform_convert_dtype_s += convert_dtype_s
            transform_normalize_s += normalize_s
            transform_numpy_s += numpy_s
        except Exception as exc:
            raise RuntimeError(f"Failed to decode/transform image {path!r}") from exc

    stage_start = time.perf_counter()
    batch = np.stack(arrays, axis=0).astype(np.float32, copy=False)
    stack_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    norm_images = pa.FixedShapeTensorArray.from_numpy_ndarray(batch)
    arrow_s = time.perf_counter() - stage_start
    total_s = time.perf_counter() - total_start

    _CPU_UDF_TIMING_CALLS += 1
    if CPU_UDF_TIMING_LOG and (_CPU_UDF_TIMING_CALLS % CPU_UDF_TIMING_SAMPLE_RATE == 0):
        rows = len(image_urls)
        rows_per_s = rows / total_s if total_s > 0 else 0.0
        timing_fields = _format_cpu_udf_timing_fields(
            read_s=read_s,
            decode_s=decode_s,
            transform_s=transform_s,
            stack_s=stack_s,
            arrow_s=arrow_s,
            total_s=total_s,
            rows_per_s=rows_per_s,
            transform_to_tensor_s=transform_to_tensor_s,
            transform_resize_s=transform_resize_s,
            transform_center_crop_s=transform_center_crop_s,
            transform_convert_dtype_s=transform_convert_dtype_s,
            transform_normalize_s=transform_normalize_s,
            transform_numpy_s=transform_numpy_s,
        )
        _emit_udf_timing_line(
            "[vane_image][cpu_udf_timing] "
            f"run_id={BENCHMARK_RUN_ID} pid={os.getpid()} call={_CPU_UDF_TIMING_CALLS} rows={rows} "
            f"payload_bytes={payload_bytes} failures={failures} "
            f"process_threads={_process_thread_count()} "
            f"torch_num_threads={torch.get_num_threads()} "
            f"torch_interop_threads={torch.get_num_interop_threads()} "
            f"{timing_fields}"
        )
    return pa.table({"image_url": image_urls, "norm_image": norm_images})


class ResNetModel:
    def __init__(self):
        self.weights = weights
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = resnet18(weights=self.weights)
        self._model = model.to(self._device)
        self._model.eval()
        self._timing_calls = 0

    def __call__(self, table):
        total_start = time.perf_counter()

        stage_start = time.perf_counter()
        image_urls = table.column("image_url").to_pylist()
        image_urls_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        norm_col = table.column("norm_image")
        if isinstance(norm_col, pa.ChunkedArray):
            norm_col = norm_col.combine_chunks()
        arr = norm_col.to_numpy_ndarray()
        arrow_numpy_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        arr = np.array(arr, dtype=np.float32, copy=True, order="C")
        cpu_copy_s = time.perf_counter() - stage_start

        should_time = GPU_UDF_TIMING_LOG and (self._timing_calls + 1) % GPU_UDF_TIMING_SAMPLE_RATE == 0
        with torch.inference_mode():
            stage_start = time.perf_counter()
            torch_batch = torch.from_numpy(arr).to(self._device)
            if should_time and self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            h2d_s = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            prediction = self._model(torch_batch)
            if should_time and self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            forward_s = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            classes = prediction.argmax(dim=1).detach().cpu().tolist()
            if should_time and self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            classes_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        labels = [self.weights.meta["categories"][idx] for idx in classes]
        result = pa.table({"image_url": image_urls, "label": labels})
        output_arrow_s = time.perf_counter() - stage_start
        total_s = time.perf_counter() - total_start

        self._timing_calls += 1
        if should_time:
            rows = len(image_urls)
            rows_per_s = rows / total_s if total_s > 0 else 0.0
            _emit_udf_timing_line(
                "[vane_image][gpu_udf_timing] "
                f"run_id={BENCHMARK_RUN_ID} pid={os.getpid()} call={self._timing_calls} rows={rows} "
                f"device={self._device} process_threads={_process_thread_count()} "
                f"image_urls_s={image_urls_s:.6f} arrow_numpy_s={arrow_numpy_s:.6f} "
                f"cpu_copy_s={cpu_copy_s:.6f} h2d_s={h2d_s:.6f} forward_s={forward_s:.6f} "
                f"classes_s={classes_s:.6f} output_arrow_s={output_arrow_s:.6f} "
                f"total_s={total_s:.6f} rows_per_s={rows_per_s:.2f}"
            )
        return result


def main() -> None:
    gpu_actor_env_names = (
        ("IMAGE_GPU_ACTOR_NUMBER", "GPU_ACTOR_NUMBER", "NUM_GPU_NODES")
        if USE_RAY
        else (
            "IMAGE_GPU_ACTOR_NUMBER",
            "GPU_ACTOR_NUMBER",
            "NUM_GPU_NODES",
        )
    )
    num_gpu_nodes = _read_first_int_env(
        gpu_actor_env_names,
        _default_gpu_actor_count() if USE_RAY else _default_local_gpu_actor_count(),
        minimum=1,
    )
    cpu_partition_count = _read_first_int_env(
        ("IMAGE_CPU_PARTITIONS",),
        _default_cpu_partition_count(),
        minimum=1,
    )
    if EXECUTION_WIDTH > 0:
        # The physical plan is finalized by the Ray driver actor, so propagate
        # the same width to its DuckDB connection before Ray runtime_env is
        # captured.  Repartitioning remains independently fixed above.
        os.environ["VANE_DUCKDB_THREADS"] = str(EXECUTION_WIDTH)
    _ensure_ray_initialized()
    con = None
    rel = None
    decoded_rel = None
    labeled_rel = None
    try:
        con = vane.connect(config={"local_exchange_streaming": "true"})
        _configure_vane_s3(con, (INPUT_PATH, LOCAL_IMAGE_ROOT, OUTPUT_PATH))
        con.execute("SET arrow_large_buffer_size=true")
        if EXECUTION_WIDTH > 0:
            con.execute(f"SET threads={EXECUTION_WIDTH}")
        execution_width = int(con.execute("SELECT current_setting('threads')").fetchone()[0])

        start_time = time.time()
        print(f"[vane_image] input_path={INPUT_PATH}", file=sys.stderr, flush=True)
        print(f"[vane_image] local_image_root={LOCAL_IMAGE_ROOT}", file=sys.stderr, flush=True)
        print(f"[vane_image] output_path={OUTPUT_PATH}", file=sys.stderr, flush=True)
        print(
            "[vane_image] "
            "udf_backends preprocess=%s gpu=%s cpu_partitions=%d execution_width=%d "
            "cpu_workers=%s cpu_native_threads=%s gpu_actors=%d"
            % (
                "ray_task" if USE_RAY else LOCAL_CPU_UDF_BACKEND,
                "ray_actor" if USE_RAY else LOCAL_GPU_UDF_BACKEND,
                cpu_partition_count,
                execution_width,
                "per-local-ray-task" if USE_RAY else "local-default",
                "ray-runtime" if USE_RAY else "runtime-env",
                num_gpu_nodes,
            ),
            file=sys.stderr,
            flush=True,
        )
        print(
            "[vane_image] "
            f"cpu_udf_streaming_breaker={CPU_UDF_STREAMING_BREAKER} "
            f"gpu_udf_streaming_breaker={GPU_UDF_STREAMING_BREAKER}",
            file=sys.stderr,
            flush=True,
        )

        rel = con.read_parquet(_normalize_parquet_input(INPUT_PATH))
        rel = rel.project(_localize_image_path_sql())
        if INPUT_OFFSET > 0 or 0 < INPUT_LIMIT < FULL_INPUT_ROW_COUNT:
            rel = rel.limit(INPUT_LIMIT, INPUT_OFFSET)
        rel = rel.repartition(cpu_partition_count)
        print(
            f"[vane_image] repartition_count={cpu_partition_count}",
            file=sys.stderr,
            flush=True,
        )

        decode_kwargs = {
            "schema": {"image_url": vane.sqltypes.VARCHAR, "norm_image": NORM_IMAGE_SQL_TYPE},
            "execution_backend": "ray_task" if USE_RAY else LOCAL_CPU_UDF_BACKEND,
            "cpus": 1.0,
            "gpus": 0.0,
            "batch_size": CPU_BATCH_SIZE,
            **_cpu_udf_streaming_kwargs(),
        }
        decoded_rel = rel.map_batches(_decode_and_transform, **decode_kwargs)
        labeled_rel = decoded_rel.map_batches(
            ResNetModel,
            schema={"image_url": vane.sqltypes.VARCHAR, "label": vane.sqltypes.VARCHAR},
            batch_size=BATCH_SIZE,
            **_gpu_udf_streaming_kwargs(),
            **(
                {
                    "execution_backend": "ray_actor",
                    "actor_number": num_gpu_nodes,
                    "gpus": 1.0,
                }
                if USE_RAY
                else {
                    "execution_backend": LOCAL_GPU_UDF_BACKEND,
                    "actor_number": num_gpu_nodes,
                    "gpus": 1.0,
                }
            ),
        )

        _maybe_create_local_output_dir(OUTPUT_PATH)
        labeled_rel.write_parquet(OUTPUT_PATH)
        print("Runtime:", time.time() - start_time)
    finally:
        labeled_rel = None
        decoded_rel = None
        rel = None
        try:
            import gc

            gc.collect()
        except Exception:
            pass
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
        _shutdown_ray()


if __name__ == "__main__":
    main()
