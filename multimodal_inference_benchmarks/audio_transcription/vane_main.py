from __future__ import annotations

import io
import os
import sys
import time
import uuid
from urllib.parse import urlparse

import numpy as np
import pyarrow as pa
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

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

import vane
from vane import ColumnExpression, ConstantExpression, FunctionExpression

USE_RAY = (os.getenv("VANE_RUNNER", "").strip().lower() or "ray") == "ray"
_RAY_STARTED_BY_SCRIPT = False


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


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in ("0", "false", "no")


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


def _default_local_gpu_actor_count() -> int:
    try:
        gpu_count = int(torch.cuda.device_count())
        if gpu_count > 0:
            return gpu_count
    except Exception:
        pass
    return 1


def _resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return path
    if "://" in path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _is_s3_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


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
    transcription_model = os.environ.get("TRANSCRIPTION_MODEL", "").strip()
    if transcription_model:
        env_vars["TRANSCRIPTION_MODEL"] = transcription_model
    sampling_rate = os.environ.get("SAMPLING_RATE")
    if sampling_rate is not None:
        env_vars["SAMPLING_RATE"] = sampling_rate
    try:
        model_source = _resolve_model_source()
    except Exception:
        model_source = ""
    if model_source:
        env_vars["TRANSCRIPTION_MODEL_PATH"] = model_source
    ray_address = os.getenv("RAY_ADDRESS", "auto").strip()
    try:
        ray.init(
            address=ray_address,
            ignore_reinit_error=True,
            runtime_env={"env_vars": env_vars},
        )
    except Exception:
        local_init_kwargs = {
            "ignore_reinit_error": True,
            "runtime_env": {"env_vars": env_vars},
        }
        ray_num_cpus = _read_int_env("RAY_NUM_CPUS", 0, minimum=0)
        if ray_num_cpus > 0:
            local_init_kwargs["num_cpus"] = ray_num_cpus
        ray.init(**local_init_kwargs)
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


TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "openai/whisper-tiny")
SAMPLING_RATE = _read_int_env("SAMPLING_RATE", 16000, minimum=1)
BATCH_SIZE = _read_int_env("BATCH_SIZE", 128, minimum=1)
INPUT_LIMIT = _read_int_env("INPUT_LIMIT", 0, minimum=0)

_DEFAULT_INPUT_PATH = "s3://datasets/multimodal_inference_benchmarks/common_voice_17/parquet"
INPUT_PATH = (os.getenv("INPUT_PATH") or _DEFAULT_INPUT_PATH).strip()
OUTPUT_PATH = (
    os.getenv("OUTPUT_PATH")
    or f"s3://datasets/multimodal_inference_benchmarks/audio_transcription_output/{uuid.uuid4().hex}"
).strip()
if "://" not in OUTPUT_PATH and not OUTPUT_PATH.lower().endswith(".parquet"):
    OUTPUT_PATH = os.path.join(OUTPUT_PATH.rstrip("/"), "result.parquet")
CPU_UDF_STREAMING_BREAKER = True
GPU_UDF_STREAMING_BREAKER = True
LOCAL_CPU_UDF_BACKEND = _read_subprocess_backend(
    (
        "AUDIO_CPU_SUBPROCESS_BACKEND",
        "AUDIO_PREPROCESS_SUBPROCESS_BACKEND",
        "CPU_SUBPROCESS_BACKEND",
    ),
    "subprocess_task",
)
LOCAL_GPU_UDF_BACKEND = _read_subprocess_backend(
    (
        "AUDIO_GPU_SUBPROCESS_BACKEND",
        "AUDIO_TRANSCRIBER_SUBPROCESS_BACKEND",
        "GPU_SUBPROCESS_BACKEND",
    ),
    "subprocess_actor",
)

FEATURE_MELS = 80
FEATURE_FRAMES = 3000
RESAMPLED_AUDIO_TYPE = vane.list_type(vane.sqltypes.FLOAT)
INPUT_FEATURES_TYPE = vane.tensor_type(vane.sqltypes.FLOAT, (FEATURE_MELS, FEATURE_FRAMES))
TOKEN_IDS_TYPE = vane.list_type(vane.sqltypes.INTEGER)
_TRANSCRIPTION_MODEL_SOURCE: str | None = None
_PROCESSOR = None
_RESAMPLERS: dict[int, T.Resample] = {}


def _resolve_model_source() -> str:
    global _TRANSCRIPTION_MODEL_SOURCE
    explicit_path = os.getenv("TRANSCRIPTION_MODEL_PATH", "").strip()
    if explicit_path:
        resolved = _resolve_path(os.path.expanduser(explicit_path), SCRIPT_DIR)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"TRANSCRIPTION_MODEL_PATH does not exist: {resolved}")
        return resolved
    if _TRANSCRIPTION_MODEL_SOURCE is not None:
        return _TRANSCRIPTION_MODEL_SOURCE
    try:
        from huggingface_hub import snapshot_download

        _TRANSCRIPTION_MODEL_SOURCE = snapshot_download(
            TRANSCRIPTION_MODEL,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError("Whisper model is not available locally; set TRANSCRIPTION_MODEL_PATH.") from exc
    return _TRANSCRIPTION_MODEL_SOURCE


def _get_processor():
    global _PROCESSOR
    if _PROCESSOR is None:
        _PROCESSOR = AutoProcessor.from_pretrained(_resolve_model_source())
    return _PROCESSOR


def _get_resampler(sample_rate: int):
    if sample_rate not in _RESAMPLERS:
        _RESAMPLERS[sample_rate] = T.Resample(sample_rate, SAMPLING_RATE)
    return _RESAMPLERS[sample_rate]


def _to_bytes(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, memoryview):
        return bytes(value)
    if isinstance(value, bytearray):
        return bytes(value)
    return value


def _decode_audio_bytes(payload: bytes):
    for fmt in ("flac", None, "wav", "mp3"):
        try:
            if fmt is None:
                return torchaudio.load(io.BytesIO(payload))
            return torchaudio.load(io.BytesIO(payload), format=fmt)
        except Exception:
            continue
    raise ValueError("Failed to decode audio bytes.")


def _stream_resample(table):
    audio_col = table.column("audio_bytes")
    passthrough_names = [name for name in table.column_names if name != "audio_bytes"]
    resampled_values: list[list[float]] = []

    for idx, value in enumerate(audio_col.to_pylist()):
        payload = _to_bytes(value)
        if not payload:
            raise ValueError(f"Missing audio bytes at row {idx}.")
        waveform, sample_rate = _decode_audio_bytes(payload)
        resampled = _get_resampler(int(sample_rate))(waveform).squeeze()
        arr = resampled.detach().cpu().numpy().astype(np.float32, copy=False).ravel()
        if arr.size == 0:
            raise ValueError(f"Decoded audio is empty at row {idx}.")
        resampled_values.append(arr.tolist())

    arrays = [table.column(name) for name in passthrough_names]
    names = list(passthrough_names)
    arrays.append(pa.array(resampled_values, type=pa.list_(pa.float32())))
    names.append("arr")
    return pa.table(arrays, names=names)


def _stream_whisper_preprocess(table):
    audio_col = table.column("arr")
    passthrough_names = [name for name in table.column_names if name not in ("arr", "audio_bytes")]
    audio_values: list[list[float]] = []
    for idx, value in enumerate(audio_col.to_pylist()):
        if value is None:
            raise ValueError(f"Missing resampled audio at row {idx}.")
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            raise ValueError(f"Resampled audio is empty at row {idx}.")
        audio_values.append(arr.tolist())

    extracted = _get_processor()(
        audio_values,
        sampling_rate=SAMPLING_RATE,
        return_tensors="np",
        device="cpu",
    ).input_features
    features = np.asarray(extracted, dtype=np.float32)
    expected_shape = (table.num_rows, FEATURE_MELS, FEATURE_FRAMES)
    if features.shape != expected_shape:
        raise ValueError(f"Whisper features have shape {features.shape}, expected {expected_shape}.")

    arrays = [table.column(name) for name in passthrough_names]
    names = list(passthrough_names)
    arrays.append(pa.FixedShapeTensorArray.from_numpy_ndarray(np.ascontiguousarray(features)))
    names.append("input_features")
    return pa.table(arrays, names=names)


class Transcriber:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            _resolve_model_source(),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, table):
        feature_col = table.column("input_features")
        passthrough_names = [name for name in table.column_names if name != "input_features"]
        if isinstance(feature_col, pa.ChunkedArray):
            feature_col = feature_col.combine_chunks()
        spectrograms_np = np.asarray(feature_col.to_numpy_ndarray(), dtype=np.float32)
        expected_shape = (table.num_rows, FEATURE_MELS, FEATURE_FRAMES)
        if spectrograms_np.shape != expected_shape:
            raise ValueError(f"Whisper features have shape {spectrograms_np.shape}, expected {expected_shape}.")

        spectrograms = torch.from_numpy(spectrograms_np).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            generated = self.model.generate(spectrograms)
        token_values = [[int(token) for token in token_ids] for token_ids in generated.detach().cpu().numpy().tolist()]

        arrays = [table.column(name) for name in passthrough_names]
        names = list(passthrough_names)
        arrays.append(pa.array(token_values, type=pa.list_(pa.int32())))
        names.append("token_ids")
        return pa.table(arrays, names=names)


def _stream_decode_tokens(table):
    token_col = table.column("token_ids")
    passthrough_names = [name for name in table.column_names if name != "token_ids"]
    values = token_col.to_pylist()
    for idx, value in enumerate(values):
        if value is None:
            raise ValueError(f"Missing token IDs at row {idx}.")
    transcriptions = _get_processor().batch_decode(values, skip_special_tokens=True)

    arrays = [table.column(name) for name in passthrough_names]
    names = list(passthrough_names)
    arrays.append(pa.array(transcriptions, type=pa.string()))
    names.append("transcription")
    return pa.table(arrays, names=names)


def _select_audio_bytes(rel):
    audio_bytes_expr = FunctionExpression(
        "struct_extract",
        ColumnExpression("audio"),
        ConstantExpression("bytes"),
    )
    rel = rel.select(*rel.columns, audio_bytes_expr.alias("audio_bytes"))
    if "audio" in rel.columns:
        rel = rel.select(*[name for name in rel.columns if name != "audio"])
    return rel


def _normalize_audio_parquet_input(path: str) -> str:
    if any(ch in path for ch in ("*", "?", "[")):
        return path
    lower = path.lower()
    if lower.endswith((".parquet", ".parquet.gz")):
        return path
    if _is_s3_path(path):
        return path.rstrip("/") + "/*.parquet"
    if os.path.isdir(path):
        return os.path.join(path.rstrip("/"), "**", "*.parquet")
    return path


def main() -> None:
    if os.getenv("AUDIO_PREPROCESS_IMPL", "python").strip().lower() != "python":
        raise RuntimeError("vane_main.py only implements the default python preprocess path.")

    _ensure_ray_initialized()
    _get_processor()

    gpu_actor_env_names = (
        ("AUDIO_GPU_ACTOR_NUMBER", "AUDIO_NUM_GPU_NODES", "GPU_ACTOR_NUMBER", "NUM_GPU_NODES")
        if USE_RAY
        else (
            "AUDIO_GPU_ACTOR_NUMBER",
            "AUDIO_TRANSCRIBER_ACTOR_NUMBER",
            "GPU_ACTOR_NUMBER",
            "AUDIO_NUM_GPU_NODES",
            "NUM_GPU_NODES",
        )
    )
    num_gpu_nodes = _read_first_int_env(
        gpu_actor_env_names,
        2 if USE_RAY else _default_local_gpu_actor_count(),
        minimum=1,
    )
    cpu_passthrough_kwargs = {"streaming_breaker": CPU_UDF_STREAMING_BREAKER}
    gpu_streaming_kwargs = {"streaming_breaker": GPU_UDF_STREAMING_BREAKER}
    con = None
    rel = None
    completed = False
    try:
        con = vane.connect(config={"local_exchange_streaming": "true"})
        _configure_vane_s3(con, (INPUT_PATH, OUTPUT_PATH))
        con.execute("SET preserve_insertion_order=false")

        start_time = time.time()
        print(f"[vane_audio] input_path={INPUT_PATH}", file=sys.stderr, flush=True)
        print(f"[vane_audio] output_path={OUTPUT_PATH}", file=sys.stderr, flush=True)
        print(
            "[vane_audio] udf_backends resample=%s preprocess=%s "
            "transcribe=%s decoder=%s gpu_actors=%d"
            % (
                "ray_task" if USE_RAY else LOCAL_CPU_UDF_BACKEND,
                "ray_task" if USE_RAY else LOCAL_CPU_UDF_BACKEND,
                "ray_actor" if USE_RAY else LOCAL_GPU_UDF_BACKEND,
                "ray_task" if USE_RAY else LOCAL_CPU_UDF_BACKEND,
                num_gpu_nodes,
            ),
            file=sys.stderr,
            flush=True,
        )
        print(
            "[vane_audio] ref_passthrough cpu_breaker=%s gpu_breaker=%s"
            % (
                CPU_UDF_STREAMING_BREAKER,
                GPU_UDF_STREAMING_BREAKER,
            ),
            file=sys.stderr,
            flush=True,
        )
        print(
            "[vane_audio] udf_batching batch_size=%d" % BATCH_SIZE,
            file=sys.stderr,
            flush=True,
        )

        rel = con.read_parquet(_normalize_audio_parquet_input(INPUT_PATH))
        if INPUT_LIMIT > 0:
            rel = rel.limit(INPUT_LIMIT)
        rel = _select_audio_bytes(rel)
        rel = rel.map_batches(
            _stream_resample,
            schema={
                **{name: rel.dtypes[i] for i, name in enumerate(rel.columns) if name != "audio_bytes"},
                "arr": RESAMPLED_AUDIO_TYPE,
            },
            execution_backend="ray_task" if USE_RAY else LOCAL_CPU_UDF_BACKEND,
            gpus=0.0,
            **cpu_passthrough_kwargs,
        )
        rel = rel.map_batches(
            _stream_whisper_preprocess,
            schema={
                **{name: rel.dtypes[i] for i, name in enumerate(rel.columns) if name != "arr"},
                "input_features": INPUT_FEATURES_TYPE,
            },
            execution_backend="ray_task" if USE_RAY else LOCAL_CPU_UDF_BACKEND,
            gpus=0.0,
            batch_size=BATCH_SIZE,
            **cpu_passthrough_kwargs,
        )
        rel = rel.map_batches(
            Transcriber,
            schema={
                **{name: rel.dtypes[i] for i, name in enumerate(rel.columns) if name != "input_features"},
                "token_ids": TOKEN_IDS_TYPE,
            },
            execution_backend="ray_actor" if USE_RAY else LOCAL_GPU_UDF_BACKEND,
            actor_number=num_gpu_nodes,
            gpus=1.0,
            batch_size=BATCH_SIZE,
            **gpu_streaming_kwargs,
        )
        rel = rel.map_batches(
            _stream_decode_tokens,
            schema={
                **{name: rel.dtypes[i] for i, name in enumerate(rel.columns) if name != "token_ids"},
                "transcription": vane.sqltypes.VARCHAR,
            },
            execution_backend="ray_task" if USE_RAY else LOCAL_CPU_UDF_BACKEND,
            gpus=0.0,
            batch_size=BATCH_SIZE,
            **cpu_passthrough_kwargs,
        )
        rel = rel.select(
            *rel.columns,
            FunctionExpression("length", ColumnExpression("transcription")).alias("transcription_length"),
        )
        final_cols = [name for name in rel.columns if name not in ("audio_bytes", "arr", "input_features", "token_ids")]
        rel = rel.select(*final_cols)

        _maybe_create_local_output_dir(OUTPUT_PATH)
        rel.write_parquet(OUTPUT_PATH)
        completed = True
        print("Runtime:", time.time() - start_time, flush=True)
    finally:
        rel = None
        global _PROCESSOR
        _PROCESSOR = None
        _RESAMPLERS.clear()
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
        if completed and _read_bool_env("AUDIO_OS_EXIT_AFTER_SUCCESS", True):
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
