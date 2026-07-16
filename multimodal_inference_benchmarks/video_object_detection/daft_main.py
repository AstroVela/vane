# This file is adapted from https://github.com/Eventual-Inc/Daft/tree/9da265d8f1e5d5814ae871bed3cee1b0757285f5/benchmarking/ai/video_object_detection
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import daft
import torch
import torchvision
from daft.expressions import col
from daft.io.av._read_video_frames import _VideoFramesSource
from PIL import Image
from ultralytics import YOLO

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

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
NUM_GPU_NODES = int(os.getenv("NUM_GPU_NODES", "8"))
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
_YOLO_MODEL_SOURCE = None
_DEFAULT_S3_VIDEO_DIR = "datasets/multimodal_inference_benchmarks/hollywood2/AVIClips"
INPUT_PATH = os.getenv("INPUT_PATH", _DEFAULT_S3_VIDEO_DIR)
_DEFAULT_S3_OUTPUT = (
    f"datasets/multimodal_inference_benchmarks/video_object_detection_output/raydata_{uuid.uuid4().hex}"
)
_DEFAULT_LOCAL_OUTPUT = f"/tmp/raydata-video-write-benchmark/{uuid.uuid4().hex}"
WRITE_S3 = os.getenv("WRITE_S3", "false").lower() in ("true", "1", "yes")
OUTPUT_PATH = os.getenv(
    "OUTPUT_PATH",
    _DEFAULT_S3_OUTPUT if WRITE_S3 else _DEFAULT_LOCAL_OUTPUT,
)
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "640"))
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
BATCH_SIZE = max(1, int(os.getenv("BATCH_SIZE", "9")))
INPUT_LIMIT = int(os.getenv("INPUT_LIMIT", "0"))
VIDEO_PARTITION_SIZE = BATCH_SIZE * (IMAGE_HEIGHT * IMAGE_WIDTH * 3 + 64)


@dataclass
class _BatchedVideoFramesSource(_VideoFramesSource):
    """Daft video source whose micro-partitions match the GPU batch size."""

    max_partition_size: int = VIDEO_PARTITION_SIZE

    def get_tasks(self, pushdowns):
        for task in super().get_tasks(pushdowns):
            # Store this on each task instance so Ray workers receive the tuned
            # value instead of falling back to Daft's fixed 10 MiB class value.
            task._max_partition_size = self.max_partition_size
            yield task


def _read_video_frames(path: str, io_config=None):
    return _BatchedVideoFramesSource(
        paths=[path],
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
        io_config=io_config,
        max_partition_size=VIDEO_PARTITION_SIZE,
    ).read()


def _parse_endpoint(endpoint_url: str) -> tuple[str, bool]:
    if "://" not in endpoint_url:
        endpoint_url = f"http://{endpoint_url}"
    parsed = urlparse(endpoint_url)
    use_ssl = parsed.scheme == "https"
    return endpoint_url, use_ssl


def _normalize_s3_uri(path: str) -> tuple[str, bool]:
    force_anonymous = False
    if path.startswith("s3://anonymous@"):
        force_anonymous = True
        path = path.replace("s3://anonymous@", "s3://", 1)
    elif not path.startswith("s3://"):
        path = f"s3://{path.lstrip('/')}"
    return path, force_anonymous


def _should_use_s3(path: str) -> bool:
    return path.startswith("s3://") or not (os.path.isabs(path) or path.startswith("file://"))


def _resolve_local_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(SCRIPT_DIR, expanded))


def _resolve_yolo_model_source() -> str:
    global _YOLO_MODEL_SOURCE
    explicit_path = os.getenv("YOLO_MODEL_PATH", "").strip()
    if explicit_path:
        resolved = _resolve_local_path(explicit_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"YOLO_MODEL_PATH does not exist: {resolved}")
        return resolved
    if _YOLO_MODEL_SOURCE is not None:
        return _YOLO_MODEL_SOURCE

    candidates = [
        Path(_resolve_local_path(YOLO_MODEL)),
        Path(os.path.abspath(os.path.expanduser(YOLO_MODEL))),
        Path(SCRIPT_DIR) / YOLO_MODEL,
        Path.home() / ".cache" / "ultralytics" / YOLO_MODEL,
        Path.home() / ".cache" / "Ultralytics" / YOLO_MODEL,
    ]
    for candidate in candidates:
        resolved = os.path.abspath(os.path.expanduser(str(candidate)))
        if os.path.exists(resolved):
            _YOLO_MODEL_SOURCE = resolved
            return resolved
    raise RuntimeError(f"YOLO weights {YOLO_MODEL} are not available locally; set YOLO_MODEL_PATH to a local .pt file.")


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


@daft.udf(
    return_dtype=daft.DataType.list(
        daft.DataType.struct(
            {
                "label": daft.DataType.string(),
                "confidence": daft.DataType.float32(),
                "bbox": daft.DataType.list(daft.DataType.int32()),
            }
        )
    ),
    concurrency=NUM_GPU_NODES,
    num_gpus=1.0,
    batch_size=BATCH_SIZE,
)
class ExtractImageFeatures:
    def __init__(self):
        self.model = YOLO(_resolve_yolo_model_source())
        if torch.cuda.is_available():
            self.model.to("cuda")

    def to_features(self, res):
        return [
            {
                "label": label,
                "confidence": confidence.item(),
                "bbox": bbox.tolist(),
            }
            for label, confidence, bbox in zip(res.names, res.boxes.conf, res.boxes.xyxy, strict=False)
        ]

    def __call__(self, images):
        if len(images) == 0:
            return []
        batch = [torchvision.transforms.functional.to_tensor(Image.fromarray(image)) for image in images]
        stack = torch.stack(batch, dim=0)
        return daft.Series.from_pylist([self.to_features(res) for res in self.model(stack)])


daft.context.set_runner_ray()

start_time = time.time()
print(
    f"[daft] model={YOLO_MODEL} batch_size={BATCH_SIZE} "
    f"image={IMAGE_WIDTH}x{IMAGE_HEIGHT} num_gpu_nodes={NUM_GPU_NODES}"
)

READ_S3 = not (os.path.isabs(INPUT_PATH) or INPUT_PATH.startswith("file://"))
if READ_S3:
    input_path, force_anonymous = _normalize_s3_uri(INPUT_PATH)
    df = _read_video_frames(
        input_path,
        io_config=_make_s3_io_config(force_anonymous),
    )
else:
    df = _read_video_frames(INPUT_PATH)
if INPUT_LIMIT > 0:
    df = df.limit(INPUT_LIMIT)
df = df.with_column("features", ExtractImageFeatures(col("data")))
df = df.explode("features")
df = df.with_column(
    "object",
    daft.col("data").image.crop(daft.col("features")["bbox"]).image.encode("png"),
)
df = df.exclude("data")
if _should_use_s3(OUTPUT_PATH):
    output_path, force_anonymous = _normalize_s3_uri(OUTPUT_PATH)
    df.write_parquet(output_path, io_config=_make_s3_io_config(force_anonymous))
else:
    df.write_parquet(OUTPUT_PATH)

print("Runtime:", time.time() - start_time)
