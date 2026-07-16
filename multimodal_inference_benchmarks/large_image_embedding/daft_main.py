import os
import time
import uuid

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
from daft import DataType, udf
from pybase64 import b64decode
from transformers import ViTForImageClassification, ViTImageProcessor

BATCH_SIZE = 1024
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
VIT_MODEL = os.getenv("VIT_MODEL", "google/vit-base-patch16-224")
_VIT_MODEL_SOURCE = None

INPUT_PREFIX = os.getenv(
    "INPUT_PATH",
    "s3://anonymous@ray-example-data/image-datasets/10TiB-b64encoded-images-in-parquet-v3/",
)
OUTPUT_PREFIX = os.getenv("OUTPUT_PATH", f"s3://ray-data-write-benchmark/{uuid.uuid4().hex}")


def _normalize_local_parquet_path(path: str) -> str:
    return os.path.join(path, "**") if os.path.isdir(path) else path


def _resolve_local_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(SCRIPT_DIR, expanded))


def _resolve_vit_model_source() -> str:
    global _VIT_MODEL_SOURCE
    explicit_path = os.getenv("VIT_MODEL_PATH", "").strip() or os.getenv("LARGE_IMAGE_MODEL_PATH", "").strip()
    if explicit_path:
        resolved = _resolve_local_path(explicit_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"VIT_MODEL_PATH does not exist: {resolved}")
        return resolved
    if _VIT_MODEL_SOURCE is not None:
        return _VIT_MODEL_SOURCE
    try:
        from huggingface_hub import snapshot_download

        _VIT_MODEL_SOURCE = snapshot_download(VIT_MODEL, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "ViT model is not available locally. Pre-download "
            f"{VIT_MODEL} on every worker node or set "
            "VIT_MODEL_PATH to a local model directory."
        ) from exc
    return _VIT_MODEL_SOURCE


PROCESSOR = ViTImageProcessor(
    do_convert_rgb=None,
    do_normalize=True,
    do_rescale=True,
    do_resize=True,
    image_mean=[0.5, 0.5, 0.5],
    image_std=[0.5, 0.5, 0.5],
    resample=2,
    rescale_factor=0.00392156862745098,
    size={"height": 224, "width": 224},
)


daft.context.set_runner_ray()


@ray.remote
def warmup():
    pass


# NOTE: On a fresh Ray cluster, it can take a minute or longer to schedule the first
#       task. To ensure benchmarks compare data processing speed and not cluster startup
#       overhead, this code launches a several tasks as warmup.
ray.get([warmup.remote() for _ in range(64)])


def decode(data: bytes) -> bytes:
    return b64decode(data, None, True)


def preprocess(image):
    outputs = PROCESSOR(images=image)["pixel_values"]
    assert len(outputs) == 1, type(outputs)
    return outputs[0]


@udf(
    return_dtype=DataType.tensor(DataType.float32()),
    batch_size=BATCH_SIZE,
    num_gpus=1,
    concurrency=40,
)
class Infer:
    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = ViTForImageClassification.from_pretrained(
            _resolve_vit_model_source(),
            local_files_only=True,
        ).to(self._device)

    def __call__(self, image_column) -> np.ndarray:
        image_ndarray = np.array(image_column.to_pylist())
        with torch.inference_mode():
            next_tensor = torch.from_numpy(image_ndarray).to(
                dtype=torch.float32, device=self._device, non_blocking=True
            )
            output = self._model(next_tensor).logits
            return output.cpu().detach().numpy()


start_time = time.time()

df = daft.read_parquet(_normalize_local_parquet_path(INPUT_PREFIX))
df = df.with_column("image", df["image"].apply(decode, return_dtype=DataType.binary()))
df = df.with_column("image", df["image"].image.decode(mode=daft.ImageMode.RGB))
df = df.with_column("height", df["image"].image_height())
df = df.with_column("width", df["image"].image.width())
df = df.with_column(
    "image",
    df["image"].apply(preprocess, return_dtype=DataType.tensor(DataType.float32())),
)
df = df.with_column("embeddings", Infer(df["image"]))
df = df.select("embeddings")
df.write_parquet(OUTPUT_PREFIX)

print("Runtime", time.time() - start_time)
