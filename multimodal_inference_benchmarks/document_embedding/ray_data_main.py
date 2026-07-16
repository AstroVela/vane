from __future__ import annotations

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


import pyarrow.fs as pafs
import pymupdf
import ray
import ray.data
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
NUM_GPU_NODES = int(os.getenv("NUM_GPU_NODES", "8"))

S3_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://127.0.0.1:9000").strip()
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
S3_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN", "").strip()
S3_REGION = os.getenv("AWS_REGION", "us-east-1").strip()

_DEFAULT_S3_INPUT_PATH = "datasets/multimodal_inference_benchmarks/digitalcorpora/metadata"
_DEFAULT_S3_PDF_ROOT = "s3://datasets/multimodal_inference_benchmarks/digitalcorpora/pdf_dump"
WRITE_S3 = os.getenv("WRITE_S3", "true").lower() in ("true", "1", "yes")
_DEFAULT_S3_OUTPUT = f"datasets/multimodal_inference_benchmarks/document_embedding_output/raydata_{uuid.uuid4().hex}"
_DEFAULT_LOCAL_OUTPUT = f"/tmp/ray-data-write-benchmark/{uuid.uuid4().hex}"

INPUT_PATH = os.getenv("INPUT_PATH", _DEFAULT_S3_INPUT_PATH).strip()
PDF_ROOT = os.getenv("PDF_ROOT", os.getenv("LOCAL_PDF_ROOT", _DEFAULT_S3_PDF_ROOT)).strip()
OUTPUT_PATH = os.getenv(
    "OUTPUT_PATH",
    _DEFAULT_S3_OUTPUT if WRITE_S3 else _DEFAULT_LOCAL_OUTPUT,
).strip()

MAX_PDF_PAGES = 100
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 200
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "320"))
INPUT_LIMIT = int(os.getenv("INPUT_LIMIT", "0"))

_EMBED_MODEL_SOURCE = None


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


def _resolve_local_path(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(SCRIPT_DIR, path))


def _resolve_embed_model_source() -> str:
    global _EMBED_MODEL_SOURCE

    explicit_path = os.getenv("EMBED_MODEL_PATH", "").strip()
    if explicit_path:
        resolved = _resolve_local_path(explicit_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"EMBED_MODEL_PATH does not exist: {resolved}")
        return resolved

    if _EMBED_MODEL_SOURCE is not None:
        return _EMBED_MODEL_SOURCE

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to resolve the local embedding model cache") from exc

    try:
        _EMBED_MODEL_SOURCE = snapshot_download(
            EMBED_MODEL_ID,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Embedding model is not available locally on this node. "
            f"Pre-download {EMBED_MODEL_ID} on every worker node or set "
            "EMBED_MODEL_PATH to a local model directory."
        ) from exc
    return _EMBED_MODEL_SOURCE


def _ray_runtime_env() -> dict[str, dict[str, str]]:
    env_vars = {
        "http_proxy": "",
        "HTTP_PROXY": "",
        "https_proxy": "",
        "HTTPS_PROXY": "",
        "all_proxy": "",
        "ALL_PROXY": "",
        "no_proxy": "*",
        "NO_PROXY": "*",
        "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL", "http://127.0.0.1:9000"),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        "AWS_SESSION_TOKEN": os.environ.get("AWS_SESSION_TOKEN", ""),
        "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
        "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "0"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "0"),
    }

    resolved_model_path = os.environ.get("EMBED_MODEL_PATH", "").strip()
    if not resolved_model_path:
        try:
            resolved_model_path = _resolve_embed_model_source()
        except Exception:
            resolved_model_path = ""
    if resolved_model_path:
        env_vars["EMBED_MODEL_PATH"] = resolved_model_path

    for key in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
    ):
        value = os.environ.get(key, "").strip()
        if value:
            env_vars[key] = value

    return {"env_vars": env_vars}


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

print(f"[ray_data] input_path={INPUT_PATH} (s3={_should_use_s3(INPUT_PATH)})")
print(f"[ray_data] pdf_root={PDF_ROOT}")
print(f"[ray_data] output_path={OUTPUT_PATH} (s3={_should_use_s3(OUTPUT_PATH)})")
print(
    f"[ray_data] model={EMBED_MODEL_ID} batch_size={EMBEDDING_BATCH_SIZE} "
    f"num_gpu_nodes={NUM_GPU_NODES} input_limit={INPUT_LIMIT}"
)


@ray.remote
def warmup():
    pass


# NOTE: On a fresh Ray cluster, it can take a minute or longer to schedule the first
#       task. To ensure benchmarks compare data processing speed and not cluster startup
#       overhead, this code launches several tasks as warmup.
ray.get([warmup.remote() for _ in range(64)])


def _read_pdf_bytes(path: str) -> bytes:
    if _is_s3_uri(path):
        with s3_fs.open_input_file(_normalize_s3_path(path)) as f:
            return f.read()
    if os.path.isabs(path) and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    with s3_fs.open_input_file(_normalize_s3_path(path)) as f:
        return f.read()


def extract_text_from_pdf(row):
    path = row["uploaded_pdf_path"]
    doc = None
    try:
        doc = pymupdf.Document(stream=_read_pdf_bytes(path), filetype="pdf")
        if len(doc) > MAX_PDF_PAGES:
            print(f"Skipping PDF {path} because it has {len(doc)} pages")
            return
        for page in doc:
            row["page_text"] = page.get_text()
            row["page_number"] = page.number
            yield row
    except Exception as exc:
        print(f"Error extracting text from PDF {path}: {exc}")
        return
    finally:
        if doc is not None:
            doc.close()


def resolve_pdf_path(row):
    path = row.get("uploaded_pdf_path")
    if not path:
        return row
    if path.startswith("s3://ray-example-data/pdf_dump/"):
        row["uploaded_pdf_path"] = path.replace(
            "s3://ray-example-data/pdf_dump",
            PDF_ROOT,
            1,
        )
    return row


def chunker(row):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    for chunk_index, text in enumerate(splitter.split_text(row.pop("page_text"))):
        row["chunk"] = text
        row["chunk_id"] = chunk_index
        yield row


class Embedder:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(_resolve_embed_model_source(), device=device)
        self.model.compile()

    def __call__(self, batch):
        batch["embedding"] = self.model.encode(
            batch["chunk"],
            show_progress_bar=False,
        )
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


t_start = time.time()

ds = _read_input_dataset()
if INPUT_LIMIT > 0:
    ds = ds.limit(INPUT_LIMIT)
ds = ds.filter(lambda row: row["file_name"].endswith(".pdf"))
ds = ds.map(resolve_pdf_path)
ds = ds.flat_map(extract_text_from_pdf)
ds = ds.flat_map(chunker)
ds = ds.map_batches(
    Embedder,
    batch_size=EMBEDDING_BATCH_SIZE,
    num_gpus=1.0,
    concurrency=NUM_GPU_NODES,
)
ds = ds.select_columns(["uploaded_pdf_path", "page_number", "chunk_id", "chunk", "embedding"])
_write_output_dataset(ds)

print(f"Runtime: {time.time() - t_start}")
