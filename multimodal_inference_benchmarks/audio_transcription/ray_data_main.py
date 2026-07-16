from __future__ import annotations

import io
import os
import time
import uuid

import numpy as np
import pyarrow.fs as pa_fs
import ray
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "openai/whisper-tiny")
NUM_GPUS = int(os.getenv("NUM_GPUS", "8"))
SAMPLING_RATE = int(os.getenv("SAMPLING_RATE", "16000"))
_DEFAULT_S3_INPUT_PATH = "s3://anonymous@ray-example-data/common_voice_17/parquet/"
_DATA_ROOT = os.path.expanduser(os.getenv("VANE_BENCHMARK_DATA_ROOT", "~/.cache/vane/benchmarks"))
_DEFAULT_LOCAL_INPUT_PATH = os.path.join(_DATA_ROOT, "common_voice_17", "parquet")
INPUT_PATH = os.getenv(
    "INPUT_PATH",
    _DEFAULT_LOCAL_INPUT_PATH
    if (
        os.path.isdir(_DEFAULT_LOCAL_INPUT_PATH)
        and any(name.endswith(".parquet") for name in os.listdir(_DEFAULT_LOCAL_INPUT_PATH))
    )
    else _DEFAULT_S3_INPUT_PATH,
)
OUTPUT_PATH = os.getenv("OUTPUT_PATH", f"/tmp/ray-data-audio-write-benchmark/{uuid.uuid4().hex}")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
INPUT_LIMIT = int(os.getenv("INPUT_LIMIT", "0"))

ray.init(ignore_reinit_error=True)


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


def resample(item):
    # NOTE: Remove the `audio` column since we don't need it anymore. This is done by
    # the system automatically on Ray Data 2.51+ with the `with_column` API.
    audio = item.pop("audio")
    audio_bytes = audio["bytes"]
    waveform, sampling_rate = _decode_audio_bytes(audio_bytes)
    waveform = T.Resample(sampling_rate, SAMPLING_RATE)(waveform).squeeze()
    item["arr"] = np.array(waveform)
    return item


processor = AutoProcessor.from_pretrained(TRANSCRIPTION_MODEL)


def whisper_preprocess(batch):
    array = batch.pop("arr")
    extracted_features = processor(
        array.tolist(),
        sampling_rate=SAMPLING_RATE,
        return_tensors="np",
        device="cpu",
    ).input_features
    batch["input_features"] = list(extracted_features)
    return batch


class Transcriber:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.model_id = TRANSCRIPTION_MODEL
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        self.model.to(self.device)

    def __call__(self, batch):
        input_features = batch.pop("input_features")
        spectrograms = np.array(input_features)
        spectrograms = torch.tensor(spectrograms).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            token_ids = self.model.generate(spectrograms)
        batch["token_ids"] = token_ids.cpu().numpy()
        return batch


def decoder(batch):
    # NOTE: Remove the `token_ids` column since we don't need it anymore. This is done by
    # the system automatically on Ray Data 2.51+ with the `with_column` API.
    token_ids = batch.pop("token_ids")
    transcription = processor.batch_decode(token_ids, skip_special_tokens=True)
    batch["transcription"] = transcription
    batch["transcription_length"] = np.array([len(t) for t in transcription])
    return batch


def _resolve_parquet_source(path: str):
    # Ray Data accepts a pyarrow.fs.FileSystem for S3.
    # For the public ray-example-data bucket we default to anonymous access.
    force_anonymous = False
    if path.startswith("s3://anonymous@"):  # legacy form
        force_anonymous = True
        path = path.replace("s3://anonymous@", "s3://", 1)
    if not path.startswith("s3://"):
        return path, None

    # Parse bucket for region resolution.
    bucket = path[len("s3://") :].split("/", 1)[0]
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        try:
            region = pa_fs.resolve_s3_region(bucket)
        except Exception:
            region = None

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = os.getenv("AWS_SESSION_TOKEN")
    profile = os.getenv("AWS_PROFILE")
    has_creds = bool(access_key or secret_key or session_token or profile)
    anonymous = force_anonymous or not has_creds

    try:
        fs = pa_fs.S3FileSystem(anonymous=anonymous, region=region)
    except Exception:
        fs = None
    return path, fs


start_time = time.time()

input_path, input_fs = _resolve_parquet_source(INPUT_PATH)

ds = ray.data.read_parquet(input_path, filesystem=input_fs)
if INPUT_LIMIT > 0:
    ds = ds.limit(INPUT_LIMIT)
ds = ds.repartition(target_num_rows_per_block=BATCH_SIZE)
ds = ds.map(resample)
ds = ds.map_batches(whisper_preprocess, batch_size=BATCH_SIZE)
ds = ds.map_batches(
    Transcriber,
    batch_size=BATCH_SIZE,
    concurrency=NUM_GPUS,
    num_gpus=1,
)
ds = ds.map_batches(decoder)
ds.write_parquet(OUTPUT_PATH)

print("Runtime:", time.time() - start_time)
