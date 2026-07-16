# This file is adapted from https://github.com/Eventual-Inc/Daft/tree/9da265d8f1e5d5814ae871bed3cee1b0757285f5/benchmarking/ai/document_embedding
from __future__ import annotations

import os
import time
import uuid
from urllib.parse import urlparse

import daft
import pymupdf
import ray
import torch
from daft import col
from langchain.text_splitter import RecursiveCharacterTextSplitter

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

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
NUM_GPU_NODES = 1
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
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
INPUT_LIMIT = int(os.getenv("INPUT_LIMIT", "0"))
_EMBED_MODEL_SOURCE = None


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


def _normalize_pdf_root(path: str) -> str:
    if _is_s3_uri(path) or os.path.isabs(path) or path.startswith(("http://", "https://", "file://")):
        return path.rstrip("/")
    return os.path.abspath(os.path.join(SCRIPT_DIR, path)).rstrip("/")


def _normalize_local_parquet_path(path: str) -> str:
    return os.path.join(path, "**") if os.path.isdir(path) else path


def _resolve_local_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(SCRIPT_DIR, expanded))


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

        _EMBED_MODEL_SOURCE = snapshot_download(
            EMBED_MODEL_ID,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Embedding model is not available locally. Pre-download "
            f"{EMBED_MODEL_ID} on every worker node or set "
            "EMBED_MODEL_PATH to a local model directory."
        ) from exc
    return _EMBED_MODEL_SOURCE


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


PDF_ROOT = _normalize_pdf_root(PDF_ROOT)
s3_io_config = _make_s3_io_config()


daft.context.set_runner_ray()


@ray.remote
def warmup():
    pass


# NOTE: On a fresh Ray cluster, it can take a minute or longer to schedule the first
#       task. To ensure benchmarks compare data processing speed and not cluster startup
#       overhead, this code launches a several tasks as warmup.
ray.get([warmup.remote() for _ in range(64)])


def extract_text_from_parsed_pdf(pdf_bytes):
    try:
        doc = pymupdf.Document(stream=pdf_bytes, filetype="pdf")
        if len(doc) > MAX_PDF_PAGES:
            print(f"Skipping PDF because it has {len(doc)} pages")
            return None
        return [{"text": page.get_text(), "page_number": page.number} for page in doc]
    except Exception as e:
        print(f"Error extracting text from PDF {e}")
        return None


def chunk(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunk_iter = splitter.split_text(text)
    chunks = []
    for chunk_index, text in enumerate(chunk_iter):
        chunks.append(
            {
                "text": text,
                "chunk_id": chunk_index,
            }
        )
    return chunks


@daft.udf(
    return_dtype=daft.DataType.fixed_size_list(daft.DataType.float32(), EMBEDDING_DIM),
    concurrency=NUM_GPU_NODES,
    num_gpus=1.0,
    batch_size=EMBEDDING_BATCH_SIZE,
)
class Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(_resolve_embed_model_source(), device=device)
        self.model.compile()

    def __call__(self, text_col):
        if len(text_col) == 0:
            return []
        embeddings = self.model.encode(
            text_col.to_pylist(),
            convert_to_tensor=True,
        )
        return embeddings.cpu().numpy()


start_time = time.time()

if _should_use_s3(INPUT_PATH):
    input_path, force_anonymous = _normalize_s3_uri(INPUT_PATH)
    df = daft.read_parquet(input_path, io_config=_make_s3_io_config(force_anonymous))
else:
    df = daft.read_parquet(_normalize_local_parquet_path(INPUT_PATH))
if INPUT_LIMIT > 0:
    df = df.limit(INPUT_LIMIT)
df = df.where(daft.col("file_name").str.endswith(".pdf"))
df = df.with_column(
    "uploaded_pdf_path",
    df["uploaded_pdf_path"].str.replace("s3://ray-example-data/pdf_dump", PDF_ROOT),
)
df = df.with_column("pdf_bytes", df["uploaded_pdf_path"].url.download(io_config=s3_io_config))
pages_struct_type = daft.DataType.struct(fields={"text": daft.DataType.string(), "page_number": daft.DataType.int32()})
df = df.with_column(
    "pages",
    df["pdf_bytes"].apply(
        extract_text_from_parsed_pdf,
        return_dtype=daft.DataType.list(pages_struct_type),
    ),
)
df = df.explode("pages")
df = df.with_columns({"page_text": col("pages")["text"], "page_number": col("pages")["page_number"]})
df = df.where(daft.col("page_text").not_null())
chunks_struct_type = daft.DataType.struct(fields={"text": daft.DataType.string(), "chunk_id": daft.DataType.int32()})
df = df.with_column(
    "chunks",
    df["page_text"].apply(chunk, return_dtype=daft.DataType.list(chunks_struct_type)),
)
df = df.explode("chunks")
df = df.with_columns({"chunk": col("chunks")["text"], "chunk_id": col("chunks")["chunk_id"]})
df = df.where(daft.col("chunk").not_null())
df = df.with_column("embedding", Embedder(df["chunk"]))
df = df.select("uploaded_pdf_path", "page_number", "chunk_id", "chunk", "embedding")
if _should_use_s3(OUTPUT_PATH):
    output_path, force_anonymous = _normalize_s3_uri(OUTPUT_PATH)
    df.write_parquet(output_path, io_config=_make_s3_io_config(force_anonymous))
else:
    df.write_parquet(OUTPUT_PATH)

print("Runtime:", time.time() - start_time)
