from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from urllib.parse import urlparse

import pyarrow as pa

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


def _read_first_optional_positive_int_env(names: tuple[str, ...]) -> int | None:
    for name in names:
        value = os.getenv(name)
        if value is None or value.strip() == "":
            continue
        result = int(value)
        if result <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}")
        return result
    return None


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
            if key.startswith(("VANE_", "DUCKDB_", "HF_", "TRANSFORMERS_"))
        }
    )
    embedding_batch_size = os.environ.get("EMBEDDING_BATCH_SIZE")
    if embedding_batch_size is not None:
        env_vars["EMBEDDING_BATCH_SIZE"] = embedding_batch_size
    debug_progress = os.environ.get("DOCUMENT_DEBUG_PROGRESS")
    if debug_progress is not None:
        env_vars["DOCUMENT_DEBUG_PROGRESS"] = debug_progress
    try:
        embed_model_source = _resolve_embed_model_source()
    except Exception:
        embed_model_source = ""
    if embed_model_source:
        env_vars["EMBED_MODEL_PATH"] = embed_model_source
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
    if not _read_bool_env("SHUTDOWN_RAY", _RAY_STARTED_BY_SCRIPT):
        return
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
MAX_PDF_PAGES = 100
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 200
EMBEDDING_BATCH_SIZE = _read_int_env("EMBEDDING_BATCH_SIZE", 2560, minimum=1)
DEBUG_PROGRESS = _read_bool_env("DOCUMENT_DEBUG_PROGRESS", False)
INPUT_LIMIT = _read_int_env("INPUT_LIMIT", 0, minimum=0)
CPU_UDF_STREAMING_BREAKER = True
GPU_UDF_STREAMING_BREAKER = True
LOCAL_CPU_UDF_BACKEND = _read_subprocess_backend(
    ("DOCUMENT_CPU_SUBPROCESS_BACKEND", "CPU_SUBPROCESS_BACKEND"),
    "subprocess_task",
)
LOCAL_GPU_UDF_BACKEND = _read_subprocess_backend(
    ("DOCUMENT_GPU_SUBPROCESS_BACKEND", "GPU_SUBPROCESS_BACKEND"),
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


_DEFAULT_INPUT_PATH = "s3://datasets/multimodal_inference_benchmarks/digitalcorpora/metadata/**/*.parquet"
_DEFAULT_PDF_ROOT = "s3://datasets/multimodal_inference_benchmarks/digitalcorpora/pdf_dump"
INPUT_PATH = _resolve_path(
    (os.getenv("INPUT_PATH") or _DEFAULT_INPUT_PATH).strip(),
    SCRIPT_DIR,
)
PDF_ROOT = _resolve_path(
    (os.getenv("LOCAL_PDF_ROOT") or _DEFAULT_PDF_ROOT).strip(),
    SCRIPT_DIR,
)
OUTPUT_PATH = _resolve_path(
    (
        os.getenv("OUTPUT_PATH")
        or f"s3://datasets/multimodal_inference_benchmarks/document_embedding_output/{uuid.uuid4().hex}"
    ).strip(),
    SCRIPT_DIR,
)
if "://" not in OUTPUT_PATH and not OUTPUT_PATH.lower().endswith(".parquet"):
    OUTPUT_PATH = os.path.join(OUTPUT_PATH.rstrip("/"), "result.parquet")

_EMBED_MODEL_SOURCE: str | None = None


def _resolve_embed_model_source() -> str:
    global _EMBED_MODEL_SOURCE
    explicit_path = os.getenv("EMBED_MODEL_PATH", "").strip()
    if explicit_path:
        resolved = _resolve_path(os.path.expanduser(explicit_path), SCRIPT_DIR)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"EMBED_MODEL_PATH does not exist: {resolved}")
        return resolved
    if _EMBED_MODEL_SOURCE is not None:
        return _EMBED_MODEL_SOURCE
    try:
        from huggingface_hub import snapshot_download

        _EMBED_MODEL_SOURCE = snapshot_download(EMBED_MODEL_ID, local_files_only=True)
    except Exception as exc:
        raise RuntimeError("Embedding model is not available locally; set EMBED_MODEL_PATH.") from exc
    return _EMBED_MODEL_SOURCE


def _read_pdf_bytes(path: str) -> bytes:
    if _is_s3_path(path):
        return _read_s3_bytes(path)
    with open(path, "rb") as handle:
        return handle.read()


def _localize_pdf_path_sql() -> str:
    prefix = "s3://ray-example-data/pdf_dump"
    return (
        "CASE WHEN uploaded_pdf_path LIKE '"
        + prefix
        + "/%' THEN replace(uploaded_pdf_path, '"
        + prefix
        + "', '"
        + PDF_ROOT
        + "') ELSE uploaded_pdf_path END AS uploaded_pdf_path"
    )


def extract_text_from_pdf(row):
    import pymupdf

    path = row["uploaded_pdf_path"]
    doc = None
    try:
        doc = pymupdf.Document(stream=_read_pdf_bytes(path), filetype="pdf")
        if len(doc) > MAX_PDF_PAGES:
            if DEBUG_PROGRESS:
                print(f"Skipping PDF {path} because it has {len(doc)} pages", file=sys.stderr, flush=True)
            return
        base_row = dict(row)
        for page in doc:
            out_row = dict(base_row)
            out_row["page_text"] = page.get_text()
            out_row["page_number"] = int(page.number)
            yield out_row
    except Exception as exc:
        if DEBUG_PROGRESS:
            print(f"Error extracting text from PDF {path}: {exc}", file=sys.stderr, flush=True)
        return
    finally:
        if doc is not None:
            doc.close()


def chunker(row):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    page_text = row.pop("page_text")
    for chunk_index, text in enumerate(splitter.split_text(page_text)):
        out_row = dict(row)
        out_row["chunk"] = text
        out_row["chunk_id"] = chunk_index
        yield out_row


class Embedder:
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_source = _resolve_embed_model_source()
        print(
            f"[Embedder] loading model pid={os.getpid()} source={model_source} device={device}",
            file=sys.stderr,
            flush=True,
        )
        self.model = SentenceTransformer(model_source, device=device)
        self.model.compile()
        self._encode_lock = threading.Lock()

    def _encode(self, samples):
        with self._encode_lock:
            return self.model.encode(
                samples,
                show_progress_bar=False,
                convert_to_tensor=True,
            )

    def __repr__(self) -> str:
        return "Embedder(model_in_init=True)"

    @staticmethod
    def _empty_embedding_array() -> pa.Array:
        return pa.array([], type=pa.list_(pa.float32(), EMBEDDING_DIM))

    @staticmethod
    def _embedding_arrow_array(encoded, row_count: int) -> pa.Array:
        import torch

        if row_count == 0:
            return Embedder._empty_embedding_array()
        if not isinstance(encoded, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor embedding output, got {type(encoded)!r}")
        encoded = encoded.detach()
        if encoded.ndim == 1:
            encoded = encoded.reshape(1, -1)
        if encoded.ndim != 2:
            raise ValueError(f"Embedding output must be 2D, got {tuple(encoded.shape)}")
        if encoded.shape[0] != row_count:
            raise ValueError(f"Embedding row count mismatch: expected {row_count}, got {encoded.shape[0]}")
        if encoded.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {encoded.shape[1]}")
        if encoded.dtype != torch.float32:
            encoded = encoded.to(dtype=torch.float32)
        if encoded.device.type != "cpu":
            encoded = encoded.cpu()
        encoded = encoded.contiguous()
        values = pa.array(encoded.reshape(-1).numpy(), type=pa.float32())
        return pa.FixedSizeListArray.from_arrays(values, EMBEDDING_DIM)

    def __call__(self, table):
        start = time.time()
        chunks = table.column("chunk").to_pylist()
        embedding_array = self._empty_embedding_array()
        if chunks:
            encode_start = time.time()
            encoded = self._encode(chunks)
            encode_s = time.time() - encode_start
            embedding_array = self._embedding_arrow_array(encoded, len(chunks))
        else:
            encode_s = 0.0
        if DEBUG_PROGRESS:
            print(
                "[Embedder] "
                f"pid={os.getpid()} input_rows={table.num_rows} valid_rows={len(chunks)} "
                f"encode_s={encode_s:.3f} total_s={time.time() - start:.3f}",
                file=sys.stderr,
                flush=True,
            )
        return pa.table(
            {
                "uploaded_pdf_path": table.column("uploaded_pdf_path"),
                "page_number": table.column("page_number"),
                "chunk_id": table.column("chunk_id"),
                "chunk": table.column("chunk"),
                "embedding": embedding_array,
            }
        )


def _write_parquet(rel) -> None:
    _maybe_create_local_output_dir(OUTPUT_PATH)
    if _is_s3_path(OUTPUT_PATH):
        rel.write_parquet(OUTPUT_PATH, use_tmp_file=False)
    else:
        rel.write_parquet(OUTPUT_PATH)


def main() -> None:
    _ensure_ray_initialized()
    num_gpu_nodes = _read_first_int_env(
        ("NUM_GPU_NODES",),
        2,
        minimum=1,
    )
    embed_actor_number = _read_first_int_env(
        ("EMBED_ACTOR_NUMBER", "EMBED_ACTOR_COUNT"),
        num_gpu_nodes,
        minimum=1,
    )
    con = None
    try:
        con = vane.connect(config={"local_exchange_streaming": "true"})
        _configure_vane_s3(con, (INPUT_PATH, PDF_ROOT, OUTPUT_PATH))

        start_time = time.time()
        print(
            f"[vane_document] input_path={INPUT_PATH}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[vane_document] pdf_root={PDF_ROOT}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[vane_document] output_path={OUTPUT_PATH}",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"[vane_document] ray_address={os.getenv('RAY_ADDRESS') or 'auto'}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[vane_document] "
            "task_parallelism chunk_workers=%s embed_actor_number=%d"
            % (
                "per-local-ray-task" if USE_RAY else "duckdb-native",
                embed_actor_number,
            ),
            file=sys.stderr,
            flush=True,
        )
        print(
            "[vane_document] "
            "udf_backends chunk=%s embed=%s embed_actors=%d"
            % (
                "ray_task" if USE_RAY else LOCAL_CPU_UDF_BACKEND,
                "ray_actor" if USE_RAY else LOCAL_GPU_UDF_BACKEND,
                embed_actor_number,
            ),
            file=sys.stderr,
            flush=True,
        )
        print(
            "[vane_document] "
            f"cpu_udf_streaming_breaker={CPU_UDF_STREAMING_BREAKER} "
            f"gpu_udf_streaming_breaker={GPU_UDF_STREAMING_BREAKER}",
            file=sys.stderr,
            flush=True,
        )

        input_path = _normalize_parquet_input(INPUT_PATH)
        rel = con.read_parquet(input_path)
        if INPUT_LIMIT > 0:
            rel = rel.limit(INPUT_LIMIT)
        rel = rel.project("file_name", "uploaded_pdf_path")
        rel = rel.filter("ends_with(file_name, '.pdf')")
        rel = rel.project(_localize_pdf_path_sql())
        page_kwargs = {
            "schema": {
                "uploaded_pdf_path": vane.sqltypes.VARCHAR,
                "page_number": vane.sqltypes.INTEGER,
                "page_text": vane.sqltypes.VARCHAR,
            },
            "cpus": 1.0,
            "gpus": 0.0,
            "execution_backend": "ray_task" if USE_RAY else LOCAL_CPU_UDF_BACKEND,
            **_cpu_udf_streaming_kwargs(),
        }
        rel = rel.flat_map(extract_text_from_pdf, **page_kwargs)
        chunk_kwargs = {
            "schema": {
                "uploaded_pdf_path": vane.sqltypes.VARCHAR,
                "page_number": vane.sqltypes.INTEGER,
                "chunk_id": vane.sqltypes.INTEGER,
                "chunk": vane.sqltypes.VARCHAR,
            },
            "cpus": 1.0,
            "gpus": 0.0,
            "execution_backend": "ray_task" if USE_RAY else LOCAL_CPU_UDF_BACKEND,
            **_cpu_udf_streaming_kwargs(),
        }
        rel = rel.flat_map(chunker, **chunk_kwargs)
        rel = rel.map_batches(
            Embedder,
            schema={
                "uploaded_pdf_path": vane.sqltypes.VARCHAR,
                "page_number": vane.sqltypes.INTEGER,
                "chunk_id": vane.sqltypes.INTEGER,
                "chunk": vane.sqltypes.VARCHAR,
                "embedding": vane.array_type(vane.sqltypes.FLOAT, EMBEDDING_DIM),
            },
            batch_size=EMBEDDING_BATCH_SIZE,
            **_gpu_udf_streaming_kwargs(),
            **(
                {
                    "execution_backend": "ray_actor",
                    "actor_number": embed_actor_number,
                    "gpus": 1.0,
                }
                if USE_RAY
                else {
                    "execution_backend": LOCAL_GPU_UDF_BACKEND,
                    "actor_number": embed_actor_number,
                    "gpus": 1.0,
                }
            ),
        )
        rel = rel.project(
            "uploaded_pdf_path",
            "page_number",
            "chunk_id",
            "chunk",
            "embedding",
        )

        _write_parquet(rel)
        print("Runtime:", time.time() - start_time)
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
        _shutdown_ray()


if __name__ == "__main__":
    main()
