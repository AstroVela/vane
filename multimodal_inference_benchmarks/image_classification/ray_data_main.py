from __future__ import annotations

import io
import os
import time
import uuid
from urllib.parse import urlparse

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

import numpy as np
import pyarrow.fs as pafs
import ray
import ray.autoscaler._private.util  # Avoid Ray 2.55 driver log/DataContext import race.
import torch
from packaging import version
from PIL import Image
from ray.data.expressions import download
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "8"))
_DATA_ROOT = os.path.expanduser(os.getenv("VANE_BENCHMARK_DATA_ROOT", "~/.cache/vane/benchmarks"))
INPUT_PATH = os.environ.get("INPUT_PATH", os.path.join(_DATA_ROOT, "imagenet", "metadata_file.parquet"))
LOCAL_IMAGE_ROOT = os.environ.get("LOCAL_IMAGE_ROOT", os.path.join(_DATA_ROOT, "imagenet", "train"))
S3_IMAGE_PREFIX = os.environ.get("S3_IMAGE_PREFIX", "s3://ray-example-data/imagenet/train/")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", f"/tmp/ray-data-write-benchmark/{uuid.uuid4().hex}")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))
INPUT_LIMIT = int(os.environ.get("INPUT_LIMIT", "803580"))  # 0 = no limit
GPU_NOOP = os.environ.get("RAY_DATA_GPU_NOOP", "0").strip().lower() not in ("", "0", "false", "no", "off")
GPU_NOOP_OUTPUT_MODE = os.environ.get("RAY_DATA_GPU_NOOP_OUTPUT_MODE", "full").strip().lower()
if GPU_NOOP_OUTPUT_MODE not in {"full", "metadata_only"}:
    raise ValueError(
        "Unsupported RAY_DATA_GPU_NOOP_OUTPUT_MODE=%r; expected 'full' or 'metadata_only'" % GPU_NOOP_OUTPUT_MODE
    )
S3_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9000").strip()
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
S3_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", "").strip()
S3_REGION = os.getenv("AWS_REGION", "us-east-1").strip()

weights = ResNet18_Weights.DEFAULT
transform = transforms.Compose([transforms.ToTensor(), weights.transforms()])


def _parse_endpoint(endpoint_url: str) -> tuple[str, str]:
    if "://" not in endpoint_url:
        endpoint_url = f"http://{endpoint_url}"
    parsed = urlparse(endpoint_url)
    return parsed.netloc or parsed.path, "https" if parsed.scheme == "https" else "http"


def _is_s3_uri(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def _should_use_s3(path: str) -> bool:
    return _is_s3_uri(path) or not os.path.isabs(path)


def _normalize_s3_path(path: str) -> str:
    return path[len("s3://") :] if _is_s3_uri(path) else path


def _ray_runtime_env() -> dict[str, dict[str, str]]:
    return {
        "env_vars": {
            "http_proxy": "",
            "HTTP_PROXY": "",
            "https_proxy": "",
            "HTTPS_PROXY": "",
            "all_proxy": "",
            "ALL_PROXY": "",
            "no_proxy": "*",
            "NO_PROXY": "*",
            "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL", ""),
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_SESSION_TOKEN": os.environ.get("AWS_SESSION_TOKEN", ""),
            "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
            "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        }
    }


endpoint_override, s3_scheme = _parse_endpoint(S3_ENDPOINT)
s3_kwargs = {
    "region": S3_REGION,
    "endpoint_override": endpoint_override,
    "scheme": s3_scheme,
}
if S3_ACCESS_KEY or S3_SECRET_KEY:
    s3_kwargs["access_key"] = S3_ACCESS_KEY
    s3_kwargs["secret_key"] = S3_SECRET_KEY
    if S3_SESSION_TOKEN:
        s3_kwargs["session_token"] = S3_SESSION_TOKEN
    s3_kwargs["anonymous"] = False
else:
    s3_kwargs["anonymous"] = True
s3_fs = pafs.S3FileSystem(**s3_kwargs)

ray.init(ignore_reinit_error=True, runtime_env=_ray_runtime_env())


@ray.remote
def warmup():
    pass


# NOTE: On a fresh Ray cluster, it can take a minute or longer to schedule the first
#       task. To ensure benchmarks compare data processing speed and not cluster startup
#       overhead, this code launches a several tasks as warmup.
ray.get([warmup.remote() for _ in range(64)])


def deserialize_image(row):
    image = Image.open(io.BytesIO(row["bytes"])).convert("RGB")
    # NOTE: Remove the `bytes` column since we don't need it anymore. This is done by
    # the system automatically on Ray Data 2.51+ with the `with_column` API.
    del row["bytes"]
    row["image"] = np.array(image)
    return row


def transform_image(row):
    row["norm_image"] = transform(row["image"]).numpy()
    # NOTE: Remove the `image` column since we don't need it anymore. This is done by
    # the system automatically on Ray Data 2.51+ with the `with_column` API.
    del row["image"]
    return row


def preprocess_metadata(row):
    norm_image = row.get("norm_image")
    row["preprocess_ok"] = norm_image is not None
    row["norm_image_bytes"] = int(norm_image.nbytes) if norm_image is not None else 0
    row.pop("norm_image", None)
    return row


def to_local_image_path(row):
    path = row.get("image_url")
    if not path:
        return row
    if path.startswith(S3_IMAGE_PREFIX):
        row["image_url"] = path.replace(S3_IMAGE_PREFIX, f"{LOCAL_IMAGE_ROOT}/", 1)
    return row


class ResNetActor:
    def __init__(self):
        self.weights = weights
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet18(weights=self.weights).to(self.device)
        self.model.eval()

    def __call__(self, batch):
        torch_batch = torch.from_numpy(batch["norm_image"]).to(self.device)
        # NOTE: Remove the `norm_image` column since we don't need it anymore. This is
        # done by the system automatically on Ray Data 2.51+ with the `with_column`
        # API.
        del batch["norm_image"]
        with torch.inference_mode():
            prediction = self.model(torch_batch)
            predicted_classes = prediction.argmax(dim=1).detach().cpu()
            predicted_labels = [self.weights.meta["categories"][i] for i in predicted_classes]
            batch["label"] = predicted_labels
            return batch


def _read_input_dataset():
    if _should_use_s3(INPUT_PATH):
        return ray.data.read_parquet(_normalize_s3_path(INPUT_PATH), filesystem=s3_fs)
    return ray.data.read_parquet(INPUT_PATH)


def _write_output_dataset(ds):
    if _should_use_s3(OUTPUT_PATH):
        ds.write_parquet(_normalize_s3_path(OUTPUT_PATH), filesystem=s3_fs)
    else:
        ds.write_parquet(OUTPUT_PATH)


start_time = time.time()

print(
    f"[ray_data] input_path={INPUT_PATH} local_image_root={LOCAL_IMAGE_ROOT} output_path={OUTPUT_PATH}",
    flush=True,
)
print(
    f"[ray_data] input_limit={INPUT_LIMIT} batch_size={BATCH_SIZE} num_gpu_nodes={NUM_GPU_NODES} "
    f"gpu_noop={GPU_NOOP} gpu_noop_output_mode={GPU_NOOP_OUTPUT_MODE}",
    flush=True,
)


# You can use `download` on Ray 2.50+.
if version.parse(ray.__version__) > version.parse("2.49.2"):
    ds = _read_input_dataset()
    if INPUT_LIMIT > 0:
        ds = ds.limit(INPUT_LIMIT)
    ds = (
        ds.map(to_local_image_path)
        .with_column("bytes", download("image_url"))
        .map(fn=deserialize_image)
        .map(fn=transform_image)
    )
    if GPU_NOOP:
        if GPU_NOOP_OUTPUT_MODE == "metadata_only":
            ds = ds.map(fn=preprocess_metadata).select_columns(["image_url", "preprocess_ok", "norm_image_bytes"])
        else:
            ds = ds.select_columns(["image_url", "norm_image"])
    else:
        ds = ds.map_batches(
            fn=ResNetActor,
            batch_size=BATCH_SIZE,
            num_gpus=1.0,
            concurrency=NUM_GPU_NODES,
        ).select_columns(["image_url", "label"])
    _write_output_dataset(ds)

else:
    ds_paths = _read_input_dataset()
    if INPUT_LIMIT > 0:
        ds_paths = ds_paths.limit(INPUT_LIMIT)
    paths = ds_paths.map(to_local_image_path).take_all()
    paths = [row["image_url"] for row in paths]
    ds = ray.data.read_images(paths, include_paths=True, ignore_missing_paths=True, mode="RGB").map(fn=transform_image)
    if GPU_NOOP:
        if GPU_NOOP_OUTPUT_MODE == "metadata_only":
            ds = ds.map(fn=preprocess_metadata).select_columns(["path", "preprocess_ok", "norm_image_bytes"])
        else:
            ds = ds.select_columns(["path", "norm_image"])
    else:
        ds = ds.map_batches(
            fn=ResNetActor,
            batch_size=BATCH_SIZE,
            num_gpus=1.0,
            concurrency=NUM_GPU_NODES,
        ).select_columns(["path", "label"])
    _write_output_dataset(ds)


print("Runtime:", time.time() - start_time)
