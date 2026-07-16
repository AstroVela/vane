from __future__ import annotations

import io
import math
import os
import sys
import time
import uuid
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from ultralytics import YOLO

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("VANE_RUNNER", "ray")
os.environ["VANE_UDF_EAGER_INIT"] = "1"
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
if (os.getenv("VANE_RUNNER", "").strip().lower() or "ray") == "ray":
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")

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

from video_inputs import (
    estimate_video_input_size_bytes,
    has_s3_video_files,
    path_is_s3_like,
    ray_data_read_task_count,
    resolve_video_files,
)
from video_kernels import (
    crop_bbox_to_png,
    frames_to_torch_tensor,
    video_gpu_transport_config_from_env,
    yolo_result_to_features,
)

import vane


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


def _read_optional_positive_float_env(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _read_optional_positive_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return result


def _video_source_target_block_bytes(gpu_target_block_bytes: int) -> int:
    """Choose source publication size without coupling it to import order.

    The benchmark-scoped setting takes precedence.  The datasource's legacy
    setting remains supported, but is read here explicitly instead of relying
    on the default captured when ``video_reader`` is imported.
    """
    for name in ("VIDEO_SOURCE_TARGET_BLOCK_BYTES", "VANE_VIDEO_MAX_PARTITION_BYTES"):
        configured = _read_optional_positive_int_env(name)
        if configured is not None:
            return configured
    return int(gpu_target_block_bytes)


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


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


def _read_crop_mode() -> str:
    value = (_read_optional_text_env(("VIDEO_CROP_MODE",)) or "map_batches").strip().lower()
    allowed = {"map_batches", "flat_map", "explode"}
    if value not in allowed:
        raise ValueError(f"VIDEO_CROP_MODE must be one of {sorted(allowed)}, got {value!r}")
    return value


def _is_s3_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def _maybe_create_local_output_dir(path: str) -> None:
    if "://" in path:
        return
    os.makedirs(os.path.abspath(path), exist_ok=True)


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


def _ensure_ray_initialized() -> None:
    if not USE_RAY:
        return
    import ray

    if ray.is_initialized():
        return
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
    }
    env_vars.update(
        {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("VANE_", "DUCKDB_", "HF_", "TRANSFORMERS_", "VIDEO_")) or key == "UDF_TIMING_LOG_PATH"
        }
    )
    yolo_model = os.environ.get("YOLO_MODEL", "").strip()
    if yolo_model:
        env_vars["YOLO_MODEL"] = yolo_model
    image_height = os.environ.get("IMAGE_HEIGHT")
    if image_height is not None:
        env_vars["IMAGE_HEIGHT"] = image_height
    image_width = os.environ.get("IMAGE_WIDTH")
    if image_width is not None:
        env_vars["IMAGE_WIDTH"] = image_width
    ray_address = os.getenv("RAY_ADDRESS", "auto").strip()
    try:
        ray.init(
            address=ray_address,
            ignore_reinit_error=True,
            runtime_env={"env_vars": env_vars},
        )
    except Exception:
        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": env_vars})


USE_RAY = (os.getenv("VANE_RUNNER", "").strip().lower() or "ray") == "ray"
USE_PROCESS_UDF = os.getenv("UDF_USE_PROCESS", "0").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)
UDF_USE_RAY = USE_RAY and not USE_PROCESS_UDF
CPU_UDF_USE_PROCESS = _read_bool_env("CPU_UDF_USE_PROCESS", USE_PROCESS_UDF)
GPU_UDF_USE_PROCESS = _read_bool_env("GPU_UDF_USE_PROCESS", USE_PROCESS_UDF)
CPU_UDF_USE_RAY = _read_bool_env(
    "CPU_UDF_USE_RAY",
    _read_bool_env("CPU_UDF_USE_RAY", UDF_USE_RAY and not CPU_UDF_USE_PROCESS),
)
GPU_UDF_USE_RAY = _read_bool_env(
    "GPU_UDF_USE_RAY",
    UDF_USE_RAY and not GPU_UDF_USE_PROCESS,
)

NUM_GPU_NODES = _read_int_env("NUM_GPU_NODES", 2, minimum=1)
BATCH_SIZE = _read_int_env("BATCH_SIZE", 32, minimum=1)
IMAGE_HEIGHT = _read_int_env("IMAGE_HEIGHT", 640, minimum=1)
IMAGE_WIDTH = _read_int_env("IMAGE_WIDTH", 640, minimum=1)
VIDEO_GPU_TRANSPORT = video_gpu_transport_config_from_env()
VIDEO_SOURCE_TARGET_BLOCK_BYTES = _video_source_target_block_bytes(VIDEO_GPU_TRANSPORT.target_max_block_bytes)
VIDEO_CPU_BATCH_SIZE = _read_int_env("VIDEO_CPU_BATCH_SIZE", 32, minimum=1)
VIDEO_CPU_OUTPUT_BATCH_SIZE = _read_int_env("VIDEO_CPU_OUTPUT_BATCH_SIZE", 256, minimum=1)
VIDEO_CPU_CPUS = _read_optional_positive_float_env("VIDEO_CPU_CPUS")
VIDEO_CPU_MEMORY_BYTES = _read_optional_positive_int_env("VIDEO_CPU_MEMORY_BYTES") or 512 * 1024**2
VIDEO_CROP_MODE = _read_crop_mode()
VIDEO_SCAN_TASK_BACKLOG = _read_int_env("VANE_RAY_MAX_TASK_BACKLOG", 2048, minimum=1)
VIDEO_MAX_CONCURRENT_DECODES = _read_int_env("VANE_MAX_CONCURRENT_DECODES", 256, minimum=1)
VIDEO_RESIZE_THREADS = _read_int_env("VANE_VIDEO_RESIZE_THREADS", 1, minimum=1)
VIDEO_SOURCE_UDF_CPUS = _read_optional_positive_float_env("VANE_VIDEO_SOURCE_UDF_CPUS") or float(VIDEO_RESIZE_THREADS)
VIDEO_SCAN_TASK_MIN_PARTITIONS = _read_int_env("VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM", 256, minimum=1)
VIDEO_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION = _read_int_env(
    "VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION",
    1,
    minimum=1,
)
os.environ["VANE_RAY_MAX_TASK_BACKLOG"] = str(VIDEO_SCAN_TASK_BACKLOG)
os.environ["VANE_MAX_CONCURRENT_DECODES"] = str(VIDEO_MAX_CONCURRENT_DECODES)
os.environ["VANE_RAY_SCAN_TASK_MIN_PARTITION_NUM"] = str(VIDEO_SCAN_TASK_MIN_PARTITIONS)
os.environ["VANE_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION"] = str(VIDEO_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION)
INPUT_LIMIT = _read_int_env("INPUT_LIMIT", 0, minimum=0)
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
CPU_UDF_TIMING_LOG = _read_bool_env("VIDEO_CPU_UDF_TIMING_LOG", _read_bool_env("VIDEO_UDF_TIMING_LOG", False))
CPU_UDF_TIMING_SAMPLE_RATE = _read_int_env("VIDEO_CPU_UDF_TIMING_SAMPLE_RATE", 1, minimum=1)
GPU_UDF_TIMING_LOG = _read_bool_env("VIDEO_GPU_UDF_TIMING_LOG", _read_bool_env("VIDEO_UDF_TIMING_LOG", False))
GPU_UDF_TIMING_SAMPLE_RATE = _read_int_env("VIDEO_GPU_UDF_TIMING_SAMPLE_RATE", 1, minimum=1)
BENCHMARK_TIMING_LOG = _read_bool_env("VIDEO_BENCHMARK_TIMING_LOG", False)
UDF_TIMING_LOG_PATH = _read_optional_text_env(("VIDEO_UDF_TIMING_LOG_PATH", "UDF_TIMING_LOG_PATH"))
BENCHMARK_RUN_ID = os.getenv("VIDEO_BENCHMARK_RUN_ID", "-").strip() or "-"

_DEFAULT_INPUT_PATH = "datasets/multimodal_inference_benchmarks/hollywood2/AVIClips"
INPUT_PATH = os.getenv("INPUT_PATH", _DEFAULT_INPUT_PATH)
INPUT_MANIFEST = os.getenv("INPUT_MANIFEST", "").strip() or None
_DEFAULT_LOCAL_OUTPUT = f"/tmp/raydata-video-write-benchmark/{uuid.uuid4().hex}"
OUTPUT_PATH = os.getenv(
    "OUTPUT_PATH",
    _DEFAULT_LOCAL_OUTPUT,
)
# Ray Data calls PIL resize with size=(IMAGE_HEIGHT, IMAGE_WIDTH), where PIL interprets that as (width, height).
FRAME_HEIGHT = IMAGE_WIDTH
FRAME_WIDTH = IMAGE_HEIGHT

CPU_UDF_STREAMING_BREAKER = True
GPU_UDF_STREAMING_BREAKER = True
LOCAL_CPU_UDF_BACKEND = _read_subprocess_backend(
    (
        "VIDEO_CPU_SUBPROCESS_BACKEND",
        "CPU_SUBPROCESS_BACKEND",
    ),
    "subprocess_task",
)
LOCAL_GPU_UDF_BACKEND = _read_subprocess_backend(
    (
        "VIDEO_GPU_SUBPROCESS_BACKEND",
        "GPU_SUBPROCESS_BACKEND",
    ),
    "subprocess_actor",
)

FEATURE_ARROW_TYPE = pa.struct(
    [
        ("label", pa.int64()),
        ("confidence", pa.float64()),
        ("bbox", pa.list_(pa.float64())),
    ]
)
FEATURE_LIST_ARROW_TYPE = pa.list_(FEATURE_ARROW_TYPE)
FRAME_SQL_TYPE = vane.tensor_type(vane.sqltypes.UTINYINT, (FRAME_HEIGHT, FRAME_WIDTH, 3))
FEATURE_SQL_TYPE = vane.type("STRUCT(label BIGINT, confidence DOUBLE, bbox DOUBLE[])")
FEATURE_LIST_SQL_TYPE = vane.type("STRUCT(label BIGINT, confidence DOUBLE, bbox DOUBLE[])[]")
_CPU_UDF_TIMING_CALLS = 0


def _format_seconds_and_ms(name: str, seconds: float) -> str:
    value = max(0.0, float(seconds))
    return f"{name}_s={value:.6f} {name}_ms={value * 1000.0:.3f}"


def _format_video_timing_fields(*, total_s: float, rows_per_s: float, **stage_seconds: float) -> str:
    fields = [
        _format_seconds_and_ms(name[:-2] if name.endswith("_s") else name, seconds)
        for name, seconds in stage_seconds.items()
    ]
    fields.append(_format_seconds_and_ms("total", total_s))
    fields.append(f"rows_per_s={float(rows_per_s):.2f}")
    return " ".join(fields)


def _emit_video_timing_line(line: str) -> None:
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
            "[vane_video][udf_timing_log_error] "
            f"run_id={BENCHMARK_RUN_ID} pid={os.getpid()} path={UDF_TIMING_LOG_PATH!r} "
            f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )


def _emit_benchmark_timing(stage: str, elapsed_s: float, **fields: object) -> None:
    if not BENCHMARK_TIMING_LOG:
        return
    extra_fields = " ".join(f"{key}={value}" for key, value in fields.items())
    timing_fields = _format_seconds_and_ms("elapsed", elapsed_s)
    suffix = f" {extra_fields}" if extra_fields else ""
    _emit_video_timing_line(
        "[vane_video][benchmark_timing] "
        f"run_id={BENCHMARK_RUN_ID} pid={os.getpid()} stage={stage} {timing_fields}{suffix}"
    )


def _int64_array(values):
    array = np.ascontiguousarray(values, dtype=np.int64)
    return pa.Array.from_buffers(pa.int64(), len(array), [None, pa.py_buffer(array)])


def _int32_array(values):
    array = np.ascontiguousarray(values, dtype=np.int32)
    return pa.Array.from_buffers(pa.int32(), len(array), [None, pa.py_buffer(array)])


def _float64_array(values):
    array = np.ascontiguousarray(values, dtype=np.float64)
    return pa.Array.from_buffers(pa.float64(), len(array), [None, pa.py_buffer(array)])


def _binary_array(values):
    offsets = np.empty(len(values) + 1, dtype=np.int32)
    offsets[0] = 0
    chunks = []
    total_size = 0
    for idx, value in enumerate(values):
        chunk = bytes(value)
        chunks.append(chunk)
        total_size += len(chunk)
        offsets[idx + 1] = total_size
    data = b"".join(chunks)
    return pa.Array.from_buffers(pa.binary(), len(values), [None, pa.py_buffer(offsets), pa.py_buffer(data)])


def _feature_field(feature, name: str):
    candidates = []
    if name in feature:
        candidates.append(feature[name])
    quoted_name = f'"{name}"'
    if quoted_name in feature:
        candidates.append(feature[quoted_name])
    for key, value in feature.items():
        if isinstance(key, str) and key.strip('"') == name:
            candidates.append(value)
    for value in candidates:
        if value is not None:
            return value
    if candidates:
        return None
    raise KeyError(name)


def _feature_array(features):
    labels = np.empty(len(features), dtype=np.int64)
    confidences = np.empty(len(features), dtype=np.float64)
    bbox_lengths = np.empty(len(features), dtype=np.int32)
    bbox_values = []
    for idx, feature in enumerate(features):
        labels[idx] = int(_feature_field(feature, "label"))
        confidences[idx] = float(_feature_field(feature, "confidence"))
        bbox = _feature_field(feature, "bbox")
        bbox_lengths[idx] = len(bbox)
        bbox_values.extend(float(value) for value in bbox)

    bbox_offsets = np.empty(len(features) + 1, dtype=np.int32)
    bbox_offsets[0] = 0
    if len(features):
        bbox_offsets[1:] = np.cumsum(bbox_lengths, dtype=np.int32)
    bbox_array = pa.ListArray.from_arrays(_int32_array(bbox_offsets), _float64_array(bbox_values))
    return pa.StructArray.from_arrays(
        [_int64_array(labels), _float64_array(confidences), bbox_array],
        names=["label", "confidence", "bbox"],
    )


def _features_array(features):
    offsets = np.empty(len(features) + 1, dtype=np.int32)
    offsets[0] = 0
    flat_features = []
    total_features = 0
    for idx, frame_features in enumerate(features):
        total_features += len(frame_features)
        offsets[idx + 1] = total_features
        flat_features.extend(frame_features)
    return pa.ListArray.from_arrays(_int32_array(offsets), _feature_array(flat_features))


def _cpu_udf_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if CPU_UDF_STREAMING_BREAKER:
        kwargs["streaming_breaker"] = True
    if VIDEO_CPU_CPUS is not None:
        kwargs["cpus"] = VIDEO_CPU_CPUS
    if CPU_UDF_USE_RAY:
        kwargs["memory_bytes"] = VIDEO_CPU_MEMORY_BYTES
    return kwargs


def _gpu_udf_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {
        # Ray Data keeps complete upstream blocks. A block with fewer than one
        # compute batch is coalesced with following blocks until the bundle has
        # at least ``batch_size`` rows; larger blocks are never split merely to
        # align the transport task. This is its native min_rows_per_bundle
        # policy and is distinct from both per-block flushing and a continuous
        # global row batcher.
        "min_task_batch_size": BATCH_SIZE,
        "target_max_batch_bytes": VIDEO_GPU_TRANSPORT.target_max_block_bytes,
        "task_input_max_bytes": VIDEO_GPU_TRANSPORT.input_hard_max_bytes,
        "output_target_max_bytes": VIDEO_GPU_TRANSPORT.output_hard_max_bytes,
    }
    if GPU_UDF_STREAMING_BREAKER:
        kwargs["streaming_breaker"] = True
    return kwargs


def _aligned_ray_data_read_task_count(video_files: list[str], *, filesystem=None) -> tuple[int, int, int]:
    input_size_bytes = estimate_video_input_size_bytes(video_files, filesystem=filesystem)
    if USE_RAY:
        import ray
        import ray.data

        available_cpus = max(1, int(ray.cluster_resources().get("CPU", 1)))
        context = ray.data.DataContext.get_current()
        target_min_block_size = int(context.target_min_block_size)
        read_op_min_num_blocks = int(context.read_op_min_num_blocks)
    else:
        available_cpus = max(1, int(os.cpu_count() or 1))
        target_min_block_size = 1024 * 1024
        read_op_min_num_blocks = 200
    task_count = ray_data_read_task_count(
        video_files,
        input_size_bytes=input_size_bytes,
        available_cpus=available_cpus,
        target_max_block_size=VIDEO_GPU_TRANSPORT.target_max_block_bytes,
        target_min_block_size=target_min_block_size,
        read_op_min_num_blocks=read_op_min_num_blocks,
        input_limit=INPUT_LIMIT,
    )
    return task_count, input_size_bytes, available_cpus


def _empty_frame_batch() -> np.ndarray:
    return np.empty((0, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def _frame_shape() -> tuple[int, int, int]:
    return (FRAME_HEIGHT, FRAME_WIDTH, 3)


def _normalize_frame_column(frame_col):
    if isinstance(frame_col, pa.ChunkedArray):
        if frame_col.num_chunks == 1:
            frame_col = frame_col.chunk(0)
        else:
            frame_col = frame_col.combine_chunks()
    if frame_col.null_count:
        raise ValueError("Invalid null frame.")
    frame_type = frame_col.type
    if getattr(frame_type, "extension_name", "") != "arrow.fixed_shape_tensor":
        raise TypeError(f"frame must be arrow.fixed_shape_tensor, got {frame_type}")
    if frame_type.value_type != pa.uint8() or tuple(frame_type.shape) != _frame_shape():
        raise TypeError(f"frame must be uint8 tensor with shape {_frame_shape()}, got {frame_type}")
    return frame_col


def _frame_column_to_frame_batch(frame_col) -> np.ndarray:
    frame_col = _normalize_frame_column(frame_col)
    if len(frame_col) == 0:
        return _empty_frame_batch()
    frame_batch = frame_col.to_numpy_ndarray()
    expected_shape = (len(frame_col), *_frame_shape())
    if frame_batch.shape != expected_shape:
        raise ValueError(f"frame batch has shape {frame_batch.shape!r}, expected {expected_shape!r}")
    return frame_batch


def _frame_column_to_tensor_batch(frame_col, _device: torch.device) -> torch.Tensor:
    frames = _frame_column_to_frames(frame_col)
    if not frames:
        return torch.empty((0, 3, FRAME_HEIGHT, FRAME_WIDTH), dtype=torch.float32)
    return frames_to_torch_tensor(frames, None)


def _frame_column_to_frames(frame_col) -> list[np.ndarray]:
    if isinstance(frame_col, pa.ChunkedArray) and frame_col.num_chunks > 1:
        frames = []
        for chunk in frame_col.chunks:
            frame_batch = _frame_column_to_frame_batch(chunk)
            frames.extend(frame_batch[idx] for idx in range(len(frame_batch)))
        return frames
    frame_batch = _frame_column_to_frame_batch(frame_col)
    return [frame_batch[idx] for idx in range(len(frame_col))]


def _row_frame_to_frame(frame) -> np.ndarray:
    if hasattr(frame, "as_py"):
        frame = frame.as_py()
    array = np.asarray(frame, dtype=np.uint8)
    frame_shape = _frame_shape()
    if array.ndim == 1 and array.size == math.prod(frame_shape):
        array = array.reshape(frame_shape)
    if array.shape != frame_shape:
        raise ValueError(f"frame has shape {array.shape!r}, expected {frame_shape!r}")
    return array


class YOLODetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(YOLO_MODEL)
        if torch.cuda.is_available():
            self.model.to(self.device)
        self._timing_calls = 0

    def to_features(self, res):
        return yolo_result_to_features(res)

    def __call__(self, table):
        self._timing_calls += 1
        if GPU_UDF_TIMING_LOG and (self._timing_calls % GPU_UDF_TIMING_SAMPLE_RATE == 0):
            return self._call_timed(table, self._timing_calls)

        frame_indices = table.column("frame_index").to_pylist()
        frame_col = table.column("frame")
        if len(frame_col) == 0:
            features = []
        else:
            stack = _frame_column_to_tensor_batch(frame_col, self.device)
            results = self.model(stack, verbose=False)
            features = [self.to_features(res) for res in results]

        return pa.table(
            {
                "frame_index": _int64_array(frame_indices),
                "frame": frame_col,
                "features": _features_array(features),
            }
        )

    def _call_timed(self, table, call_number: int):
        total_start = time.perf_counter()

        stage_start = time.perf_counter()
        frame_indices = table.column("frame_index").to_pylist()
        frame_index_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        frame_col = table.column("frame")
        frame_arrow_s = time.perf_counter() - stage_start

        frame_s = 0.0
        tensor_s = 0.0
        model_s = 0.0
        feature_s = 0.0
        if len(frame_col) == 0:
            features = []
        else:
            stage_start = time.perf_counter()
            stack = _frame_column_to_tensor_batch(frame_col, self.device)
            tensor_s = time.perf_counter() - stage_start

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            stage_start = time.perf_counter()
            results = self.model(stack, verbose=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_s = time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            features = [self.to_features(res) for res in results]
            feature_s = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        result = pa.table(
            {
                "frame_index": _int64_array(frame_indices),
                "frame": frame_col,
                "features": _features_array(features),
            }
        )
        arrow_s = time.perf_counter() - stage_start
        total_s = time.perf_counter() - total_start

        rows = len(frame_indices)
        output_rows = sum(len(frame_features) for frame_features in features)
        rows_per_s = rows / total_s if total_s > 0 else 0.0
        timing_fields = _format_video_timing_fields(
            frame_index_s=frame_index_s,
            frame_arrow_s=frame_arrow_s,
            frame_s=frame_s,
            tensor_s=tensor_s,
            model_s=model_s,
            feature_s=feature_s,
            arrow_s=arrow_s,
            total_s=total_s,
            rows_per_s=rows_per_s,
        )
        _emit_video_timing_line(
            "[vane_video][yolo_udf_timing] "
            f"run_id={BENCHMARK_RUN_ID} pid={os.getpid()} call={call_number} "
            f"rows={rows} output_rows={output_rows} cuda={torch.cuda.is_available()} "
            f"{timing_fields}"
        )
        return result


def _crop_generator(table):
    global _CPU_UDF_TIMING_CALLS

    _CPU_UDF_TIMING_CALLS += 1
    call_number = _CPU_UDF_TIMING_CALLS
    should_time = CPU_UDF_TIMING_LOG and (call_number % CPU_UDF_TIMING_SAMPLE_RATE == 0)
    if not should_time:
        frame_indices = table.column("frame_index").to_pylist()
        features_list = table.column("features").to_pylist()
        frame_col = table.column("frame")
        frame_batch = _frame_column_to_frame_batch(frame_col)

        out_frame_index = []
        out_features = []
        out_object = []
        png_buffer = io.BytesIO()
        for idx, features in enumerate(features_list):
            if features:
                frame = frame_batch[idx]
                pil_image = Image.fromarray(frame)
                for feature in features:
                    bbox = _feature_field(feature, "bbox")
                    cropped_pil_png = crop_bbox_to_png(
                        frame,
                        bbox,
                        pil_image=pil_image,
                        png_buffer=png_buffer,
                    )

                    out_frame_index.append(frame_indices[idx])
                    out_features.append(feature)
                    out_object.append(cropped_pil_png)

        result = pa.table(
            {
                "frame_index": _int64_array(out_frame_index),
                "features": _feature_array(out_features),
                "object": _binary_array(out_object),
            }
        )
        yield result
        return

    total_start = time.perf_counter()
    stage_start = time.perf_counter()
    frame_indices = table.column("frame_index").to_pylist()
    frame_index_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    features_list = table.column("features").to_pylist()
    features_py_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    frame_col = table.column("frame")
    frame_batch = _frame_column_to_frame_batch(frame_col)
    frame_s = time.perf_counter() - stage_start

    out_frame_index = []
    out_features = []
    out_object = []
    crop_encode_s = 0.0
    crop_pil_s = 0.0
    png_encode_s = 0.0
    bbox_area = 0
    png_bytes = 0
    png_buffer = io.BytesIO()
    for idx, features in enumerate(features_list):
        if features:
            frame = frame_batch[idx]
            pil_image = Image.fromarray(frame)
            for feature in features:
                stage_start = time.perf_counter()
                bbox = _feature_field(feature, "bbox")
                x1, y1, x2, y2 = map(int, bbox)
                bbox_area += (x2 - x1) * (y2 - y1)
                cropped_pil = pil_image.crop((x1, y1, x2, y2))
                crop_pil_s += time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                png_buffer.seek(0)
                png_buffer.truncate(0)
                cropped_pil.save(png_buffer, format="PNG", compress_level=2)
                cropped_pil_png = png_buffer.getvalue()
                png_bytes += len(cropped_pil_png)
                png_encode_s += time.perf_counter() - stage_start

                out_frame_index.append(frame_indices[idx])
                out_features.append(feature)
                out_object.append(cropped_pil_png)
    crop_encode_s = crop_pil_s + png_encode_s

    arrow_start = time.perf_counter()

    stage_start = time.perf_counter()
    frame_index_array = _int64_array(out_frame_index)
    frame_index_array_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    feature_array = _feature_array(out_features)
    feature_array_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    object_array = _binary_array(out_object)
    object_array_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    result = pa.table(
        {
            "frame_index": frame_index_array,
            "features": feature_array,
            "object": object_array,
        }
    )
    table_build_s = time.perf_counter() - stage_start
    arrow_s = time.perf_counter() - arrow_start

    total_s = time.perf_counter() - total_start

    rows = len(frame_indices)
    output_rows = result.num_rows
    rows_per_s = rows / total_s if total_s > 0 else 0.0
    timing_fields = _format_video_timing_fields(
        frame_index_s=frame_index_s,
        features_py_s=features_py_s,
        frame_s=frame_s,
        crop_encode_s=crop_encode_s,
        crop_pil_s=crop_pil_s,
        png_encode_s=png_encode_s,
        frame_index_array_s=frame_index_array_s,
        feature_array_s=feature_array_s,
        object_array_s=object_array_s,
        table_build_s=table_build_s,
        arrow_s=arrow_s,
        total_s=total_s,
        rows_per_s=rows_per_s,
    )
    _emit_video_timing_line(
        "[vane_video][crop_udf_timing] "
        f"run_id={BENCHMARK_RUN_ID} pid={os.getpid()} call={call_number} "
        f"rows={rows} output_rows={output_rows} {timing_fields} "
        f"bbox_area={bbox_area} png_bytes={png_bytes}"
    )

    yield result


def _crop_flat_map(row):
    global _CPU_UDF_TIMING_CALLS

    _CPU_UDF_TIMING_CALLS += 1
    call_number = _CPU_UDF_TIMING_CALLS
    should_time = CPU_UDF_TIMING_LOG and (call_number % CPU_UDF_TIMING_SAMPLE_RATE == 0)

    if not should_time:
        frame_index = row["frame_index"]
        features = row["features"]
        if not features:
            return

        frame = _row_frame_to_frame(row["frame"])
        pil_image = Image.fromarray(frame)
        png_buffer = io.BytesIO()
        for feature in features:
            bbox = _feature_field(feature, "bbox")
            cropped_pil_png = crop_bbox_to_png(
                frame,
                bbox,
                pil_image=pil_image,
                png_buffer=png_buffer,
            )

            yield {
                "frame_index": frame_index,
                "features": feature,
                "object": cropped_pil_png,
            }
        return

    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    frame_index = row["frame_index"]
    frame_index_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    features = row["features"]
    features_py_s = time.perf_counter() - stage_start

    if not features:
        total_s = time.perf_counter() - total_start
        timing_fields = _format_video_timing_fields(
            frame_index_s=frame_index_s,
            features_py_s=features_py_s,
            frame_s=0.0,
            crop_encode_s=0.0,
            crop_pil_s=0.0,
            png_encode_s=0.0,
            arrow_s=0.0,
            total_s=total_s,
            rows_per_s=1.0 / total_s if total_s > 0 else 0.0,
        )
        _emit_video_timing_line(
            "[vane_video][crop_flat_map_timing] "
            f"run_id={BENCHMARK_RUN_ID} pid={os.getpid()} call={call_number} "
            f"rows=1 output_rows=0 {timing_fields} bbox_area=0 png_bytes=0"
        )
        return

    stage_start = time.perf_counter()
    frame = _row_frame_to_frame(row["frame"])
    pil_image = Image.fromarray(frame)
    frame_s = time.perf_counter() - stage_start

    crop_pil_s = 0.0
    png_encode_s = 0.0
    bbox_area = 0
    png_bytes = 0
    output_rows = 0
    png_buffer = io.BytesIO()
    for feature in features:
        stage_start = time.perf_counter()
        bbox = _feature_field(feature, "bbox")
        x1, y1, x2, y2 = map(int, bbox)
        bbox_area += (x2 - x1) * (y2 - y1)
        cropped_pil = pil_image.crop((x1, y1, x2, y2))
        crop_pil_s += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        png_buffer.seek(0)
        png_buffer.truncate(0)
        cropped_pil.save(png_buffer, format="PNG", compress_level=2)
        cropped_pil_png = png_buffer.getvalue()
        png_bytes += len(cropped_pil_png)
        png_encode_s += time.perf_counter() - stage_start

        output_rows += 1
        yield {
            "frame_index": frame_index,
            "features": feature,
            "object": cropped_pil_png,
        }

    total_s = time.perf_counter() - total_start
    timing_fields = _format_video_timing_fields(
        frame_index_s=frame_index_s,
        features_py_s=features_py_s,
        frame_s=frame_s,
        crop_encode_s=crop_pil_s + png_encode_s,
        crop_pil_s=crop_pil_s,
        png_encode_s=png_encode_s,
        arrow_s=0.0,
        total_s=total_s,
        rows_per_s=1.0 / total_s if total_s > 0 else 0.0,
    )
    _emit_video_timing_line(
        "[vane_video][crop_flat_map_timing] "
        f"run_id={BENCHMARK_RUN_ID} pid={os.getpid()} call={call_number} "
        f"rows=1 output_rows={output_rows} {timing_fields} "
        f"bbox_area={bbox_area} png_bytes={png_bytes}"
    )


def _crop_exploded_generator(table):
    global _CPU_UDF_TIMING_CALLS

    _CPU_UDF_TIMING_CALLS += 1
    call_number = _CPU_UDF_TIMING_CALLS
    should_time = CPU_UDF_TIMING_LOG and (call_number % CPU_UDF_TIMING_SAMPLE_RATE == 0)

    if not should_time:
        frame_indices = table.column("frame_index").to_pylist()
        features_list = table.column("features").to_pylist()
        frame_col = table.column("frame")
        frame_batch = _frame_column_to_frame_batch(frame_col)

        out_frame_index = []
        out_features = []
        out_object = []
        for idx, feature in enumerate(features_list):
            if feature is None:
                continue
            frame = frame_batch[idx]
            bbox = _feature_field(feature, "bbox")
            out_frame_index.append(frame_indices[idx])
            out_features.append(feature)
            out_object.append(crop_bbox_to_png(frame, bbox))

        yield pa.table(
            {
                "frame_index": _int64_array(out_frame_index),
                "features": _feature_array(out_features),
                "object": _binary_array(out_object),
            }
        )
        return

    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    frame_indices = table.column("frame_index").to_pylist()
    frame_index_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    features_list = table.column("features").to_pylist()
    features_py_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    frame_col = table.column("frame")
    frame_batch = _frame_column_to_frame_batch(frame_col)
    frame_s = time.perf_counter() - stage_start

    out_object = []
    crop_pil_s = 0.0
    png_encode_s = 0.0
    bbox_area = 0
    png_bytes = 0
    out_frame_index = []
    out_features = []
    for idx, feature in enumerate(features_list):
        if feature is None:
            continue
        frame = frame_batch[idx]
        pil_image = Image.fromarray(frame)

        stage_start = time.perf_counter()
        bbox = _feature_field(feature, "bbox")
        x1, y1, x2, y2 = map(int, bbox)
        bbox_area += (x2 - x1) * (y2 - y1)
        cropped_pil = pil_image.crop((x1, y1, x2, y2))
        crop_pil_s += time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        png_buffer = io.BytesIO()
        cropped_pil.save(png_buffer, format="PNG", compress_level=2)
        cropped_pil_png = png_buffer.getvalue()
        png_bytes += len(cropped_pil_png)
        png_encode_s += time.perf_counter() - stage_start
        out_frame_index.append(frame_indices[idx])
        out_features.append(feature)
        out_object.append(cropped_pil_png)

    arrow_start = time.perf_counter()

    stage_start = time.perf_counter()
    frame_index_array = _int64_array(out_frame_index)
    frame_index_array_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    feature_array = _feature_array(out_features)
    feature_array_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    object_array = _binary_array(out_object)
    object_array_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    result = pa.table(
        {
            "frame_index": frame_index_array,
            "features": feature_array,
            "object": object_array,
        }
    )
    table_build_s = time.perf_counter() - stage_start
    arrow_s = time.perf_counter() - arrow_start

    total_s = time.perf_counter() - total_start
    rows = len(frame_indices)
    rows_per_s = rows / total_s if total_s > 0 else 0.0
    timing_fields = _format_video_timing_fields(
        frame_index_s=frame_index_s,
        features_py_s=features_py_s,
        frame_s=frame_s,
        crop_encode_s=crop_pil_s + png_encode_s,
        crop_pil_s=crop_pil_s,
        png_encode_s=png_encode_s,
        frame_index_array_s=frame_index_array_s,
        feature_array_s=feature_array_s,
        object_array_s=object_array_s,
        table_build_s=table_build_s,
        arrow_s=arrow_s,
        total_s=total_s,
        rows_per_s=rows_per_s,
    )
    _emit_video_timing_line(
        "[vane_video][crop_exploded_udf_timing] "
        f"run_id={BENCHMARK_RUN_ID} pid={os.getpid()} call={call_number} "
        f"rows={rows} output_rows={result.num_rows} {timing_fields} "
        f"bbox_area={bbox_area} png_bytes={png_bytes}"
    )
    yield result


def main() -> None:
    _ensure_ray_initialized()
    con = vane.connect(config={"local_exchange_streaming": "true"})
    _configure_vane_s3(con, (INPUT_PATH, OUTPUT_PATH))
    con.execute("SET preserve_insertion_order=false")

    start_time = time.time()
    benchmark_start = time.perf_counter()
    print(f"[vane_video] input_path={INPUT_PATH}", file=sys.stderr, flush=True)
    print(f"[vane_video] output_path={OUTPUT_PATH}", file=sys.stderr, flush=True)
    print(
        "[vane_video] actor_counts yolo=%d" % (NUM_GPU_NODES),
        file=sys.stderr,
        flush=True,
    )
    print(
        "[vane_video] udf_backends yolo=%s explode_crop=%s cpu_workers=%s "
        "gpu_actors=%d "
        "gpu_batch_size=%d gpu_target_block_bytes=%d "
        "gpu_input_hard_max_bytes=%d gpu_output_hard_max_bytes=%d "
        "gpu_actor_runtime=engine-default tensor_mode=reference "
        "cpu_batch_size=%d cpu_output_batch_size=%d "
        "cpu_cpus=%s cpu_memory_bytes=%d crop_mode=%s"
        % (
            "ray_actor" if GPU_UDF_USE_RAY else LOCAL_GPU_UDF_BACKEND,
            "ray_task" if CPU_UDF_USE_RAY else LOCAL_CPU_UDF_BACKEND,
            "payload-default" if CPU_UDF_USE_RAY else "duckdb-native",
            NUM_GPU_NODES,
            BATCH_SIZE,
            VIDEO_GPU_TRANSPORT.target_max_block_bytes,
            VIDEO_GPU_TRANSPORT.input_hard_max_bytes,
            VIDEO_GPU_TRANSPORT.output_hard_max_bytes,
            VIDEO_CPU_BATCH_SIZE,
            VIDEO_CPU_OUTPUT_BATCH_SIZE,
            VIDEO_CPU_CPUS or "payload-default",
            VIDEO_CPU_MEMORY_BYTES,
            VIDEO_CROP_MODE,
        ),
        file=sys.stderr,
        flush=True,
    )
    print(
        "[vane_video] read_source scan_task_backlog=%d max_concurrent_decodes=%d "
        "resize_threads=%d source_task_cpus=%s source_target_block_bytes=%d "
        "scan_task_min_partitions=%d dynamic_scan_max_splits_per_partition=%d"
        % (
            VIDEO_SCAN_TASK_BACKLOG,
            VIDEO_MAX_CONCURRENT_DECODES,
            VIDEO_RESIZE_THREADS,
            VIDEO_SOURCE_UDF_CPUS,
            VIDEO_SOURCE_TARGET_BLOCK_BYTES,
            VIDEO_SCAN_TASK_MIN_PARTITIONS,
            VIDEO_FTE_DYNAMIC_SCAN_MAX_SPLITS_PER_PARTITION,
        ),
        file=sys.stderr,
        flush=True,
    )
    print(
        "[vane_video] streaming cpu_breaker=%s gpu_breaker=%s cpu_locality=%s gpu_locality=%s"
        % (
            CPU_UDF_STREAMING_BREAKER,
            GPU_UDF_STREAMING_BREAKER,
            "n/a",
            "ray-driver-default" if GPU_UDF_USE_RAY else "n/a",
        ),
        file=sys.stderr,
        flush=True,
    )
    source_filesystem = _s3_filesystem() if path_is_s3_like(INPUT_PATH) and INPUT_MANIFEST is None else None
    stage_start = time.perf_counter()
    video_files = resolve_video_files(
        INPUT_PATH,
        input_manifest=INPUT_MANIFEST,
        filesystem=source_filesystem,
    )
    if has_s3_video_files(video_files) and source_filesystem is None:
        source_filesystem = _s3_filesystem()
    ray_data_read_tasks, input_size_bytes, read_parallelism_cpus = _aligned_ray_data_read_task_count(
        video_files,
        filesystem=source_filesystem,
    )
    _emit_benchmark_timing("resolve_input", time.perf_counter() - stage_start, files=len(video_files))
    print(
        f"[vane_video] input_manifest={INPUT_MANIFEST or '<generated>'} files={len(video_files)}",
        file=sys.stderr,
        flush=True,
    )
    print(
        "[vane_video] ray_data_read_alignment "
        f"read_tasks={ray_data_read_tasks} input_size_bytes={input_size_bytes} "
        f"available_cpus={read_parallelism_cpus} files_per_task=ray-array-split",
        file=sys.stderr,
        flush=True,
    )

    from vane.datasource import read_datasource
    from vane.datasource.video_reader import VideoFrameSource

    stage_start = time.perf_counter()
    rel = read_datasource(
        VideoFrameSource(
            video_files,
            height=FRAME_HEIGHT,
            width=FRAME_WIDTH,
            max_partition_bytes=VIDEO_SOURCE_TARGET_BLOCK_BYTES,
            frame_limit=INPUT_LIMIT if INPUT_LIMIT > 0 else None,
            read_task_count=ray_data_read_tasks,
        ),
        con=con,
    )
    yolo_kwargs = {
        "schema": {
            "frame_index": vane.sqltypes.BIGINT,
            "frame": FRAME_SQL_TYPE,
            "features": FEATURE_LIST_SQL_TYPE,
        },
        "gpus": 1.0,
        "batch_size": BATCH_SIZE,
        "actor_number": NUM_GPU_NODES,
        **_gpu_udf_kwargs(),
    }
    if GPU_UDF_USE_RAY:
        yolo_kwargs.update(
            {
                "execution_backend": "ray_actor",
            }
        )
    else:
        yolo_kwargs["execution_backend"] = LOCAL_GPU_UDF_BACKEND
    rel = rel.map_batches(YOLODetector, **yolo_kwargs)
    crop_schema = {
        "frame_index": vane.sqltypes.BIGINT,
        "features": FEATURE_SQL_TYPE,
        "object": vane.sqltypes.BLOB,
    }
    crop_execution_backend = "ray_task" if CPU_UDF_USE_RAY else LOCAL_CPU_UDF_BACKEND
    if VIDEO_CROP_MODE == "flat_map":
        crop_kwargs = {
            "schema": crop_schema,
            "execution_backend": crop_execution_backend,
            "gpus": 0.0,
            "batch_size": VIDEO_CPU_BATCH_SIZE,
            "output_batch_size": VIDEO_CPU_OUTPUT_BATCH_SIZE,
            **_cpu_udf_kwargs(),
        }
        rel = rel.flat_map(_crop_flat_map, **crop_kwargs)
    elif VIDEO_CROP_MODE == "explode":
        rel = rel.explode("features")
        crop_kwargs = {
            "schema": crop_schema,
            "execution_backend": crop_execution_backend,
            "gpus": 0.0,
            **_cpu_udf_kwargs(),
        }
        rel = rel.map_batches(_crop_exploded_generator, **crop_kwargs)
    else:
        crop_kwargs = {
            "schema": crop_schema,
            "execution_backend": crop_execution_backend,
            "gpus": 0.0,
            **_cpu_udf_kwargs(),
        }
        rel = rel.map_batches(_crop_generator, **crop_kwargs)
    _emit_benchmark_timing(
        "build_plan",
        time.perf_counter() - stage_start,
        batch_size=BATCH_SIZE,
        source_target_block_bytes=VIDEO_SOURCE_TARGET_BLOCK_BYTES,
        gpu_target_block_bytes=VIDEO_GPU_TRANSPORT.target_max_block_bytes,
        gpu_input_hard_max_bytes=VIDEO_GPU_TRANSPORT.input_hard_max_bytes,
        gpu_output_hard_max_bytes=VIDEO_GPU_TRANSPORT.output_hard_max_bytes,
        gpu_actor_runtime="engine-default",
        tensor_mode="reference",
        cpu_batch_size=VIDEO_CPU_BATCH_SIZE,
        cpu_output_batch_size=VIDEO_CPU_OUTPUT_BATCH_SIZE,
        cpu_cpus=VIDEO_CPU_CPUS or "payload-default",
        cpu_memory_bytes=VIDEO_CPU_MEMORY_BYTES,
        crop_mode=VIDEO_CROP_MODE,
        input_limit=INPUT_LIMIT,
        gpu_actors=NUM_GPU_NODES,
    )

    _maybe_create_local_output_dir(OUTPUT_PATH)
    stage_start = time.perf_counter()
    rel.write_parquet(OUTPUT_PATH, per_thread_output=True)
    runtime = time.time() - start_time
    _emit_benchmark_timing(
        "write_parquet",
        time.perf_counter() - stage_start,
    )
    _emit_benchmark_timing("total", time.perf_counter() - benchmark_start)
    print("Runtime:", runtime)


if __name__ == "__main__":
    main()
