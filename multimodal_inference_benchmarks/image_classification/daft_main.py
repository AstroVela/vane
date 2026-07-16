# This file is adapted from https://github.com/Eventual-Inc/Daft/tree/9da265d8f1e5d5814ae871bed3cee1b0757285f5/benchmarking/ai/image_classification
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "0").strip())
os.environ.setdefault(
    "TRANSFORMERS_OFFLINE",
    os.getenv("TRANSFORMERS_OFFLINE", os.environ["HF_HUB_OFFLINE"]).strip(),
)
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

import daft
import numpy as np
import ray
import torch
from daft import col
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

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

NUM_GPU_NODES = int(os.getenv("NUM_GPU_NODES", os.getenv("DAFT_NUM_GPU_NODES", "8")))
_DATA_ROOT = os.path.expanduser(os.getenv("VANE_BENCHMARK_DATA_ROOT", "~/.cache/vane/benchmarks"))
INPUT_PATH = os.environ.get("INPUT_PATH", os.path.join(_DATA_ROOT, "imagenet", "metadata_file.parquet"))
LOCAL_IMAGE_ROOT = os.environ.get("LOCAL_IMAGE_ROOT", os.path.join(_DATA_ROOT, "imagenet", "train"))
S3_IMAGE_PREFIX = os.environ.get("S3_IMAGE_PREFIX", "s3://ray-example-data/imagenet/train/")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", f"/tmp/ray-data-write-benchmark/{uuid.uuid4().hex}")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))
IMAGE_DIM = (3, 224, 224)
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_TORCHVISION_WEIGHTS_SOURCE = None

daft.context.set_runner_ray()

DEBUG_LOCAL_PATHS = os.getenv("DAFT_DEBUG_LOCAL_PATHS", "").lower() not in ("", "0", "false", "no")
INPUT_LIMIT = int(os.getenv("INPUT_LIMIT", os.getenv("DAFT_INPUT_LIMIT", "803580")))
STAGE_TIMING = os.getenv("DAFT_STAGE_TIMING", "").lower() not in ("", "0", "false", "no")
STAGE_TIMING_SKIP_WRITE = os.getenv("DAFT_STAGE_TIMING_SKIP_WRITE", "").lower() not in (
    "",
    "0",
    "false",
    "no",
)
CPU_MONITOR = os.getenv("DAFT_CPU_MONITOR", "").lower() not in ("", "0", "false", "no")


def _parse_endpoint(endpoint_url: str) -> tuple[str, bool]:
    if "://" not in endpoint_url:
        endpoint_url = f"http://{endpoint_url}"
    parsed = urlparse(endpoint_url)
    use_ssl = parsed.scheme == "https"
    return endpoint_url, use_ssl


def _is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def _should_use_s3(path: str) -> bool:
    return _is_s3_uri(path) or not os.path.isabs(path)


def _normalize_s3_uri(path: str) -> tuple[str, bool]:
    force_anonymous = False
    if path.startswith("s3://anonymous@"):
        force_anonymous = True
        path = path.replace("s3://anonymous@", "s3://", 1)
    elif not path.startswith("s3://"):
        path = f"s3://{path.lstrip('/')}"
    return path, force_anonymous


def _normalize_local_parquet_path(path: str) -> str:
    return os.path.join(path, "**") if os.path.isdir(path) else path


def _make_s3_io_config(anonymous: bool = False):
    try:
        from daft.daft import IOConfig, S3Config
    except Exception:
        return None

    endpoint_url, use_ssl = _parse_endpoint(os.getenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9000").strip())
    key_id = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    session_token = os.getenv("AWS_SESSION_TOKEN", "").strip()
    profile = os.getenv("AWS_PROFILE", "").strip() or None
    has_creds = bool(key_id or access_key or session_token or profile)

    s3_cfg = S3Config(
        region_name=os.getenv("AWS_REGION", "us-east-1").strip(),
        endpoint_url=endpoint_url,
        key_id=key_id or None,
        access_key=access_key or None,
        session_token=session_token or None,
        profile_name=profile,
        anonymous=anonymous or not has_creds,
        use_ssl=use_ssl,
    )
    return IOConfig(s3=s3_cfg)


s3_io_config = _make_s3_io_config()


def timed_count(label: str, df: daft.DataFrame, cols: list[str] | None = None) -> None:
    if not STAGE_TIMING:
        return
    start = time.perf_counter()
    if cols:
        result = df.count(*cols).collect()
    else:
        result = df.count().collect()
    duration = time.perf_counter() - start
    result_dict = result.to_pydict()
    if len(result_dict) == 1:
        count_value = next(iter(result_dict.values()))[0]
    else:
        count_value = {key: values[0] for key, values in result_dict.items()}
    print(f"[stage] {label}: {duration:.2f}s, count={count_value}")


def start_cpu_monitor() -> tuple[threading.Event, threading.Thread, list[float], int] | None:
    if not CPU_MONITOR:
        return None
    try:
        import psutil
    except Exception as exc:
        print(f"[cpu] psutil unavailable: {exc}")
        return None

    stop_event = threading.Event()
    samples: list[float] = []
    cpu_count = psutil.cpu_count(logical=True) or 1

    def _monitor() -> None:
        psutil.cpu_percent(interval=None)
        while not stop_event.is_set():
            samples.append(psutil.cpu_percent(interval=0.5))

    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()
    return stop_event, thread, samples, cpu_count


@ray.remote
def warmup():
    pass


# NOTE: On a fresh Ray cluster, it can take a minute or longer to schedule the first
#       task. To ensure benchmarks compare data processing speed and not cluster startup
#       overhead, this code launches a several tasks as warmup.
ray.get([warmup.remote() for _ in range(64)])


weights = ResNet18_Weights.DEFAULT
transform = transforms.Compose([transforms.ToTensor(), weights.transforms()])


def _resolve_local_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(SCRIPT_DIR, expanded))


def _resolve_torchvision_weights_source() -> str:
    global _TORCHVISION_WEIGHTS_SOURCE
    explicit_path = os.getenv("TORCHVISION_WEIGHTS_PATH", "").strip()
    if explicit_path:
        resolved = _resolve_local_path(explicit_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"TORCHVISION_WEIGHTS_PATH does not exist: {resolved}")
        return resolved
    if _TORCHVISION_WEIGHTS_SOURCE is not None:
        return _TORCHVISION_WEIGHTS_SOURCE

    filename = weights.url.rsplit("/", 1)[-1]
    candidates = []
    try:
        candidates.append(Path(torch.hub.get_dir()) / "checkpoints" / filename)
    except Exception:
        pass
    candidates.append(Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename)
    for candidate in candidates:
        resolved = os.path.abspath(os.path.expanduser(str(candidate)))
        if os.path.exists(resolved):
            _TORCHVISION_WEIGHTS_SOURCE = resolved
            return resolved
    raise RuntimeError(
        f"Torchvision weights {filename} are not available locally; set "
        "TORCHVISION_WEIGHTS_PATH to a local checkpoint file."
    )


def _load_torch_state_dict(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


@daft.udf(
    return_dtype=daft.DataType.string(),
    concurrency=NUM_GPU_NODES,
    num_gpus=1.0,
    batch_size=BATCH_SIZE,
)
class ResNetModel:
    def __init__(self):
        self.weights = weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(weights=None)
        self.model.load_state_dict(_load_torch_state_dict(_resolve_torchvision_weights_source()))
        self.model = self.model.to(self.device)
        self.model.eval()

    def __call__(self, images):
        if len(images) == 0:
            return []
        torch_batch = torch.from_numpy(np.array(images.to_pylist())).to(self.device)
        with torch.inference_mode():
            prediction = self.model(torch_batch)
            predicted_classes = prediction.argmax(dim=1).detach().cpu()
            return [self.weights.meta["categories"][i] for i in predicted_classes]


start_time = time.time()
cpu_monitor = start_cpu_monitor()

if _should_use_s3(INPUT_PATH):
    input_path, force_anonymous = _normalize_s3_uri(INPUT_PATH)
    df = daft.read_parquet(input_path, io_config=_make_s3_io_config(force_anonymous))
else:
    df = daft.read_parquet(_normalize_local_parquet_path(INPUT_PATH))
# NOTE: Limit to the 803,580 images Daft uses in their benchmark.
if INPUT_LIMIT > 0:
    df = df.limit(INPUT_LIMIT)
# Map S3 image paths in metadata to local filesystem paths (only when prefixed).
image_url_col = col("image_url")
df = df.with_column(
    "image_url",
    image_url_col.startswith(S3_IMAGE_PREFIX).if_else(
        image_url_col.str.replace(S3_IMAGE_PREFIX, f"{LOCAL_IMAGE_ROOT}/"),
        image_url_col,
    ),
)
timed_count("metadata+limit+path_map", df)
if DEBUG_LOCAL_PATHS:
    print("Sample image_url after mapping:")
    print(df.select("image_url").limit(10).collect())
# NOTE: We need to manually repartition the DataFrame to achieve good performance. This
# code isn't in Daft's benchmark, possibly because their Parquet metadata is
# pre-partitioned. Note we're using `repartition(NUM_GPUS)` instead of
# `into_partitions(NUM_CPUS * 2)` as suggested in Daft's documentation. In our
# experiments, the recommended approach led to OOMs, crashes, and slower performance.
df = df.repartition(NUM_GPU_NODES)
df = df.with_column(
    "decoded_image",
    df["image_url"].url.download(io_config=s3_io_config).image.decode(on_error="null", mode=daft.ImageMode.RGB),
)
# NOTE: At least one image encounters this error: https://github.com/etemesi254/zune-image/issues/244.
# So, we need to return "null" for errored files and filter them out.
df = df.where(df["decoded_image"].not_null())
timed_count("decode+filter", df)
if DEBUG_LOCAL_PATHS:
    print("Decoded image count:")
    print(df.count().collect())
df = df.with_column(
    "norm_image",
    df["decoded_image"].apply(
        func=lambda image: transform(image),
        return_dtype=daft.DataType.tensor(dtype=daft.DataType.float32(), shape=IMAGE_DIM),
    ),
)
timed_count("norm_image", df, cols=["norm_image"])
df = df.with_column("label", ResNetModel(col("norm_image")))
df = df.select("image_url", "label")
timed_count("label_udf", df, cols=["label"])
if STAGE_TIMING and STAGE_TIMING_SKIP_WRITE:
    print("[stage] skipping write_parquet because DAFT_STAGE_TIMING_SKIP_WRITE=1")
else:
    write_start = time.perf_counter()
    if _should_use_s3(OUTPUT_PATH):
        output_path, force_anonymous = _normalize_s3_uri(OUTPUT_PATH)
        df.write_parquet(output_path, io_config=_make_s3_io_config(force_anonymous))
    else:
        df.write_parquet(OUTPUT_PATH)
    if STAGE_TIMING:
        write_duration = time.perf_counter() - write_start
        print(f"[stage] write_parquet (includes UDF): {write_duration:.2f}s")

if cpu_monitor is not None:
    stop_event, thread, samples, cpu_count = cpu_monitor
    stop_event.set()
    thread.join(timeout=5.0)
    if samples:
        max_total = max(samples)
        max_htop = max_total * cpu_count / 100.0
        print(f"[cpu] max_total={max_total:.1f}% (approx htop {max_htop:.0f}% over {cpu_count} cores)")

print("Runtime:", time.time() - start_time)
