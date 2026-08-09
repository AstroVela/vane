# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

import vane
from vane import ColumnExpression, ConstantExpression, FunctionExpression

DATA_ROOT = Path("/data/multimodal_inference_benchmarks")
INPUT_PATH = Path(os.environ.get("INPUT_PATH", DATA_ROOT / "common_voice_17/parquet")).expanduser()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_PATH", f"/tmp/vane_audio_{uuid.uuid4().hex}")).expanduser()
MODEL_ID = "openai/whisper-tiny"
SAMPLING_RATE = 16000
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
NUM_GPU_NODES = int(os.environ.get("NUM_GPU_NODES", "1"))

FEATURE_MELS = 80
FEATURE_FRAMES = 3000
RESAMPLED_AUDIO_TYPE = vane.list_type(vane.sqltypes.FLOAT)
INPUT_FEATURES_TYPE = vane.tensor_type(vane.sqltypes.FLOAT, (FEATURE_MELS, FEATURE_FRAMES))
TOKEN_IDS_TYPE = vane.list_type(vane.sqltypes.INTEGER)

if BATCH_SIZE <= 0 or NUM_GPU_NODES <= 0:
    raise ValueError("BATCH_SIZE and NUM_GPU_NODES must be positive")


def _parquet_input(path: Path) -> str:
    return str(path / "**/*.parquet") if path.is_dir() else str(path)


def _model_path() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(MODEL_ID, local_files_only=True)


processor = AutoProcessor.from_pretrained(_model_path())


def _resampler(sample_rate: int) -> T.Resample:
    return T.Resample(sample_rate, SAMPLING_RATE)


def _decode_audio_bytes(payload: bytes):
    for fmt in ("flac", None, "wav", "mp3"):
        try:
            if fmt is None:
                return torchaudio.load(io.BytesIO(payload))
            return torchaudio.load(io.BytesIO(payload), format=fmt)
        except Exception:
            continue
    raise ValueError("Failed to decode audio bytes")


def _stream_resample(table):
    passthrough = [name for name in table.column_names if name != "audio_bytes"]
    values = []
    for index, value in enumerate(table.column("audio_bytes").to_pylist()):
        payload = bytes(value) if value is not None else b""
        if not payload:
            raise ValueError(f"Missing audio bytes at row {index}")
        waveform, sample_rate = _decode_audio_bytes(payload)
        audio = _resampler(int(sample_rate))(waveform).squeeze().cpu().numpy()
        audio = np.asarray(audio, dtype=np.float32).ravel()
        if not audio.size:
            raise ValueError(f"Decoded audio is empty at row {index}")
        values.append(audio.tolist())

    return pa.table(
        [*[table.column(name) for name in passthrough], pa.array(values, type=pa.list_(pa.float32()))],
        names=[*passthrough, "arr"],
    )


def _stream_whisper_preprocess(table):
    passthrough = [name for name in table.column_names if name not in {"arr", "audio_bytes"}]
    audio = [np.asarray(value, dtype=np.float32) for value in table.column("arr").to_pylist()]
    features = np.asarray(
        processor(audio, sampling_rate=SAMPLING_RATE, return_tensors="np").input_features,
        dtype=np.float32,
    )
    expected = (table.num_rows, FEATURE_MELS, FEATURE_FRAMES)
    if features.shape != expected:
        raise ValueError(f"Whisper features have shape {features.shape}, expected {expected}")

    return pa.table(
        [
            *[table.column(name) for name in passthrough],
            pa.FixedShapeTensorArray.from_numpy_ndarray(np.ascontiguousarray(features)),
        ],
        names=[*passthrough, "input_features"],
    )


class Transcriber:
    def __init__(self):
        self.device = torch.device("cuda")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            _model_path(),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)
        self.model.eval()

    def __call__(self, table):
        passthrough = [name for name in table.column_names if name != "input_features"]
        column = table.column("input_features")
        if isinstance(column, pa.ChunkedArray):
            column = column.combine_chunks()
        features = np.asarray(column.to_numpy_ndarray(), dtype=np.float32)
        with torch.inference_mode():
            generated = self.model.generate(torch.from_numpy(features).to(self.device, dtype=torch.float16))
        token_ids = generated.cpu().numpy().tolist()

        return pa.table(
            [
                *[table.column(name) for name in passthrough],
                pa.array(token_ids, type=pa.list_(pa.int32())),
            ],
            names=[*passthrough, "token_ids"],
        )


def _stream_decode_tokens(table):
    passthrough = [name for name in table.column_names if name != "token_ids"]
    transcriptions = processor.batch_decode(
        table.column("token_ids").to_pylist(),
        skip_special_tokens=True,
    )
    return pa.table(
        [
            *[table.column(name) for name in passthrough],
            pa.array(transcriptions, type=pa.string()),
        ],
        names=[*passthrough, "transcription"],
    )


def _select_audio_bytes(rel):
    audio_bytes = FunctionExpression(
        "struct_extract",
        ColumnExpression("audio"),
        ConstantExpression("bytes"),
    ).alias("audio_bytes")
    return rel.select(*[name for name in rel.columns if name != "audio"], audio_bytes)


def main() -> None:
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = vane.connect()
    try:
        rel = _select_audio_bytes(con.read_parquet(_parquet_input(INPUT_PATH)))
        rel = rel.map_batches(
            _stream_resample,
            schema={
                **{name: rel.dtypes[index] for index, name in enumerate(rel.columns) if name != "audio_bytes"},
                "arr": RESAMPLED_AUDIO_TYPE,
            },
        )
        rel = rel.map_batches(
            _stream_whisper_preprocess,
            schema={
                **{name: rel.dtypes[index] for index, name in enumerate(rel.columns) if name != "arr"},
                "input_features": INPUT_FEATURES_TYPE,
            },
            batch_size=BATCH_SIZE,
        )
        rel = rel.map_batches(
            Transcriber,
            schema={
                **{name: rel.dtypes[index] for index, name in enumerate(rel.columns) if name != "input_features"},
                "token_ids": TOKEN_IDS_TYPE,
            },
            batch_size=BATCH_SIZE,
            actor_number=NUM_GPU_NODES,
            gpus=1.0,
        )
        rel = rel.map_batches(
            _stream_decode_tokens,
            schema={
                **{name: rel.dtypes[index] for index, name in enumerate(rel.columns) if name != "token_ids"},
                "transcription": vane.sqltypes.VARCHAR,
            },
            batch_size=BATCH_SIZE,
        )
        rel = rel.select(
            *rel.columns,
            FunctionExpression("length", ColumnExpression("transcription")).alias("transcription_length"),
        )
        rel = rel.select(
            *[name for name in rel.columns if name not in {"audio_bytes", "arr", "input_features", "token_ids"}]
        )
        rel.write_parquet(str(OUTPUT_DIR / "result.parquet"))
    finally:
        con.close()

    print(f"Runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
