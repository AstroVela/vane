# This file is adapted from https://github.com/Eventual-Inc/Daft/tree/9da265d8f1e5d5814ae871bed3cee1b0757285f5/benchmarking/ai/audio_transcription
from __future__ import annotations

import io
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
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def _has_local_parquet_dir(path: str) -> bool:
    return os.path.isdir(path) and any(name.endswith(".parquet") for name in os.listdir(path))


def _normalize_local_parquet_path(path: str) -> str:
    return os.path.join(path, "**") if os.path.isdir(path) else path


TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "openai/whisper-tiny")
NUM_GPUS = int(os.getenv("NUM_GPUS", "8"))
NEW_SAMPLING_RATE = int(os.getenv("NEW_SAMPLING_RATE", "16000"))
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
_DEFAULT_S3_INPUT_PATH = "s3://anonymous@ray-example-data/common_voice_17/parquet/"
_DATA_ROOT = os.path.expanduser(os.getenv("VANE_BENCHMARK_DATA_ROOT", "~/.cache/vane/benchmarks"))
_DEFAULT_LOCAL_INPUT_PATH = os.path.join(_DATA_ROOT, "common_voice_17", "parquet")
INPUT_PATH = os.getenv(
    "INPUT_PATH",
    _DEFAULT_LOCAL_INPUT_PATH if _has_local_parquet_dir(_DEFAULT_LOCAL_INPUT_PATH) else _DEFAULT_S3_INPUT_PATH,
)
OUTPUT_PATH = os.getenv("OUTPUT_PATH", f"/tmp/ray-data-audio-write-benchmark/{uuid.uuid4().hex}")
INPUT_LIMIT = int(os.getenv("INPUT_LIMIT", os.getenv("DAFT_INPUT_LIMIT", "0")))
_TRANSCRIPTION_MODEL_SOURCE = None

daft.context.set_runner_ray()


@ray.remote
def warmup():
    pass


# NOTE: On a fresh Ray cluster, it can take a minute or longer to schedule the first
#       task. To ensure benchmarks compare data processing speed and not cluster startup
#       overhead, this code launches a several tasks as warmup.
ray.get([warmup.remote() for _ in range(64)])


def _decode_audio_bytes(payload: bytes):
    # Common Voice parquet embeds MP3 bytes, but some sources use FLAC/WAV.
    for fmt in ("flac", None, "wav", "mp3"):
        try:
            if fmt is None:
                return torchaudio.load(io.BytesIO(payload))
            return torchaudio.load(io.BytesIO(payload), format=fmt)
        except Exception:
            continue
    raise ValueError("Failed to decode audio bytes (tried flac/auto/wav/mp3).")


def resample(audio_bytes):
    waveform, sampling_rate = _decode_audio_bytes(audio_bytes)
    waveform = T.Resample(sampling_rate, NEW_SAMPLING_RATE)(waveform).squeeze()
    return np.array(waveform)


def _resolve_local_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(SCRIPT_DIR, expanded))


def _resolve_transcription_model_source() -> str:
    global _TRANSCRIPTION_MODEL_SOURCE
    explicit_path = os.getenv("TRANSCRIPTION_MODEL_PATH", "").strip()
    if explicit_path:
        resolved = _resolve_local_path(explicit_path)
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
        raise RuntimeError(
            "Whisper model is not available locally. Pre-download "
            f"{TRANSCRIPTION_MODEL} on every worker node or set "
            "TRANSCRIPTION_MODEL_PATH to a local model directory."
        ) from exc
    return _TRANSCRIPTION_MODEL_SOURCE


processor = AutoProcessor.from_pretrained(
    _resolve_transcription_model_source(),
    local_files_only=True,
)


@daft.udf(return_dtype=daft.DataType.tensor(daft.DataType.float32()))
def whisper_preprocess(resampled):
    return processor(
        resampled.to_arrow().to_numpy(zero_copy_only=False).tolist(),
        sampling_rate=NEW_SAMPLING_RATE,
        device="cpu",
    ).input_features


@daft.udf(
    return_dtype=daft.DataType.list(daft.DataType.int32()),
    batch_size=64,
    concurrency=NUM_GPUS,
    num_gpus=1,
)
class Transcriber:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            _resolve_transcription_model_source(),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=True,
        )
        self.model.to(self.device)

    def __call__(self, extracted_features):
        spectrograms = np.array(extracted_features)
        spectrograms = torch.tensor(spectrograms).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            token_ids = self.model.generate(spectrograms)

        return token_ids.cpu().numpy()


@daft.udf(return_dtype=daft.DataType.string())
def decoder(token_ids):
    return processor.batch_decode(token_ids, skip_special_tokens=True)


def _resolve_parquet_source(path: str):
    # Daft uses IOConfig for S3. For the public ray-example-data bucket we default to
    # anonymous access.
    force_anonymous = False
    if path.startswith("s3://anonymous@"):  # legacy form
        force_anonymous = True
        path = path.replace("s3://anonymous@", "s3://", 1)
    if not path.startswith("s3://"):
        return path, None

    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    profile = os.getenv("AWS_PROFILE")

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")
    has_creds = bool(access_key or secret_key or session_token or profile)
    anonymous = force_anonymous or not has_creds

    try:
        from daft.daft import IOConfig, S3Config

        s3_cfg = S3Config(region_name=region, profile_name=profile, anonymous=anonymous)
        return path, IOConfig(s3=s3_cfg)
    except Exception:
        return path, None


start_time = time.time()

input_path, io_config = _resolve_parquet_source(INPUT_PATH)
if io_config is None:
    input_path = _normalize_local_parquet_path(input_path)
    df = daft.read_parquet(input_path)
else:
    df = daft.read_parquet(input_path, io_config=io_config)
if INPUT_LIMIT > 0:
    df = df.limit(INPUT_LIMIT)
df = df.with_column(
    "resampled",
    df["audio"]["bytes"].apply(resample, return_dtype=daft.DataType.list(daft.DataType.float32())),
)
df = df.with_column("extracted_features", whisper_preprocess(df["resampled"]))
df = df.with_column("token_ids", Transcriber(df["extracted_features"]))
df = df.with_column("transcription", decoder(df["token_ids"]))
df = df.with_column("transcription_length", df["transcription"].str.length())
df = df.exclude("token_ids", "extracted_features", "resampled")
df.write_parquet(OUTPUT_PATH)

print("Runtime:", time.time() - start_time)
