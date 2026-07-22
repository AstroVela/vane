#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Voice AI analytics with Vane.

This example adapts Daft's voice AI analytics tutorial:
https://docs.daft.ai/en/stable/examples/voice-ai-analytics/

The Daft tutorial reads audio files, transcribes them with Faster-Whisper,
summarizes and translates the transcripts, generates subtitles, then embeds
subtitle segments for later retrieval. This Vane version keeps the same shape:

1. Build a relation of audio bytes from generated samples or local files.
2. Apply a batch UDF that returns transcripts and timestamped segments.
3. Produce summaries and subtitle rows.
4. Embed subtitle text with ``vane.ai.embed_text``.

The default transcription backend is a deterministic placeholder so the
example can run without downloading a speech model. Use
``--transcription-backend faster-whisper`` for real transcription.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import math
import os
import re
import tempfile
import wave
from pathlib import Path
from typing import Any

import pyarrow as pa

import vane
from vane.ai import embed_text

DEFAULT_WHISPER_MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
DEFAULT_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUTPUT_DIR = Path("examples/output/voice_ai_analytics")

SAMPLE_ROWS = [
    {
        "path": "sample://support-call.wav",
        "frequency": 392.0,
        "transcript": (
            "The support call discussed a spike in query latency after a new "
            "batching policy was enabled. The team agreed to add GPU queue "
            "metrics, replay two failed customer workloads, and prepare a "
            "follow up note for the account owner."
        ),
    },
    {
        "path": "sample://release-review.wav",
        "frequency": 523.25,
        "transcript": (
            "The release review covered packaging work for Vane, the TestPyPI "
            "upload, and the next documentation pass. The main action items "
            "were to simplify install instructions, keep examples runnable "
            "offline, and separate experimental integrations from stable APIs."
        ),
    },
    {
        "path": "sample://research-demo.wav",
        "frequency": 659.25,
        "transcript": (
            "The research demo showed an audio analytics workflow that turns "
            "voice recordings into searchable transcript segments. The team "
            "highlighted structured metadata, multilingual summaries, and "
            "embedding based retrieval as the important downstream pieces."
        ),
    },
]


def make_wav_bytes(*, seconds: float, sample_rate: int, frequency: float) -> bytes:
    buffer = io.BytesIO()
    frames = int(seconds * sample_rate)
    amplitude = 0.22
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(frames):
            t = i / sample_rate
            sample = int(amplitude * math.sin(math.tau * frequency * t) * 32767)
            wav_file.writeframesraw(sample.to_bytes(2, byteorder="little", signed=True))
    return buffer.getvalue()


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def relation_from_dicts(
    conn: Any,
    rows: list[dict[str, Any]],
    schema: dict[str, str] | None = None,
) -> Any:
    """Build a VALUES relation whose data can travel with a Ray plan."""
    if not rows:
        raise RuntimeError("Cannot create a relation from zero rows.")
    columns = list(schema or rows[0])
    constant = vane.ConstantExpression
    raw = conn.values(
        *(tuple(constant(row[column]) for column in columns) for row in rows),
    )
    projections = []
    for source, column in zip(raw.columns, columns, strict=True):
        expression = quote_ident(source)
        if schema is not None:
            expression += f"::{schema[column]}"
        projections.append(f"{expression} AS {quote_ident(column)}")
    return raw.query("input_rows", f"select {', '.join(projections)} from input_rows")


def collect_relation(rel: Any) -> pa.Table:
    """Materialize a relation through the configured default runner."""
    tables = list(vane.runners.get_or_create_runner().run_iter_tables(rel))
    if not tables:
        return pa.table({column: pa.array([]) for column in rel.columns})
    table = pa.concat_tables(tables)
    expected_columns = list(rel.columns)
    if table.column_names != expected_columns:
        table = table.rename_columns(expected_columns)
    return table


def sample_relation(conn: Any, limit: int) -> Any:
    rows = [SAMPLE_ROWS[i % len(SAMPLE_ROWS)] for i in range(limit)]
    input_rows = [
        {
            "id": i,
            "path": row["path"],
            "audio_format": "wav",
            "audio_bytes": make_wav_bytes(
                seconds=2.0,
                sample_rate=16_000,
                frequency=float(row["frequency"]),
            ),
            "fallback_transcript": row["transcript"],
        }
        for i, row in enumerate(rows)
    ]
    return relation_from_dicts(
        conn,
        input_rows,
        {
            "id": "BIGINT",
            "path": "VARCHAR",
            "audio_format": "VARCHAR",
            "audio_bytes": "BLOB",
            "fallback_transcript": "VARCHAR",
        },
    )


def glob_relation(conn: Any, audio_glob: str, limit: int) -> Any:
    paths = sorted(glob.glob(audio_glob))[:limit]
    if not paths:
        raise RuntimeError(f"No audio files matched --audio-glob={audio_glob!r}.")

    rows = [
        {
            "id": i,
            "path": path,
            "audio_format": Path(path).suffix.lstrip(".").lower() or "audio",
            "audio_bytes": Path(path).read_bytes(),
            "fallback_transcript": f"Audio file {Path(path).name} is ready for transcription.",
        }
        for i, path in enumerate(paths)
    ]
    return relation_from_dicts(
        conn,
        rows,
        {
            "id": "BIGINT",
            "path": "VARCHAR",
            "audio_format": "VARCHAR",
            "audio_bytes": "BLOB",
            "fallback_transcript": "VARCHAR",
        },
    )


def split_segments(text: str, *, duration_seconds: float = 12.0) -> list[dict[str, Any]]:
    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", text.strip()) if piece.strip()]
    if not pieces and text.strip():
        pieces = [text.strip()]
    if not pieces:
        pieces = ["No speech was detected."]

    step = duration_seconds / max(1, len(pieces))
    return [
        {
            "id": i,
            "start": round(i * step, 2),
            "end": round((i + 1) * step, 2),
            "text": piece,
        }
        for i, piece in enumerate(pieces)
    ]


class TranscribeAudioBatch:
    """Batch UDF that returns transcript text plus JSON-encoded segments."""

    def __init__(
        self,
        *,
        backend: str,
        model_id: str,
        compute_type: str,
        device: str,
        language: str | None,
        whisper_batch_size: int,
        vad_filter: bool,
        local_files_only: bool,
    ):
        self.backend = backend
        self.model_id = model_id
        self.compute_type = compute_type
        self.device = device
        self.language = language
        self.whisper_batch_size = whisper_batch_size
        self.vad_filter = vad_filter
        self.local_files_only = local_files_only
        self._pipe = None

    def _load_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        if self.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install real transcription dependencies first: pip install faster-whisper av") from exc

        try:
            try:
                model = WhisperModel(
                    self.model_id,
                    compute_type=self.compute_type,
                    device=self.device,
                    local_files_only=self.local_files_only,
                )
            except TypeError:
                model = WhisperModel(
                    self.model_id,
                    compute_type=self.compute_type,
                    device=self.device,
                )
        except Exception as exc:
            raise RuntimeError(
                "Could not load the Faster-Whisper model. Download it first, then "
                "pass the local directory with --whisper-model-id. For example:\n\n"
                "  hf download "
                f"{DEFAULT_WHISPER_MODEL_ID} --local-dir ~/.cache/vane/models/faster-whisper-large-v3-turbo-ct2\n"
                "  python "
                "examples/voice_ai_analytics.py --source glob "
                "--audio-glob '/path/to/audio/*.wav' "
                "--transcription-backend faster-whisper "
                "--whisper-model-id ~/.cache/vane/models/faster-whisper-large-v3-turbo-ct2 "
                "--device cuda --compute-type float16\n\n"
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        self._pipe = BatchedInferencePipeline(model)
        return self._pipe

    def _transcribe_placeholder(self, fallback_transcript: str) -> dict[str, Any]:
        text = fallback_transcript.strip() or "No placeholder transcript was provided."
        segments = split_segments(text)
        return {
            "transcript": text,
            "language": "en",
            "duration_seconds": float(segments[-1]["end"]),
            "segments_json": json.dumps(segments, ensure_ascii=False),
        }

    def _transcribe_with_whisper(
        self,
        audio_bytes: bytes,
        audio_format: str,
    ) -> dict[str, Any]:
        pipe = self._load_pipe()
        suffix = "." + re.sub(r"[^A-Za-z0-9]+", "", audio_format or "wav")
        with tempfile.NamedTemporaryFile(suffix=suffix) as audio_file:
            audio_file.write(audio_bytes)
            audio_file.flush()
            segments_iter, info = pipe.transcribe(
                audio_file.name,
                language=self.language,
                vad_filter=self.vad_filter,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                },
                word_timestamps=True,
                without_timestamps=False,
                temperature=0,
                batch_size=self.whisper_batch_size,
            )

            segments = []
            for i, segment in enumerate(segments_iter):
                segments.append(
                    {
                        "id": int(getattr(segment, "id", i)),
                        "start": float(getattr(segment, "start", 0.0)),
                        "end": float(getattr(segment, "end", 0.0)),
                        "text": str(getattr(segment, "text", "")).strip(),
                    }
                )
            transcript = " ".join(segment["text"] for segment in segments).strip()
            duration = float(getattr(info, "duration", 0.0) or 0.0)
            language = str(getattr(info, "language", self.language or "") or "")

        if not segments:
            transcript = "No speech detected."
            segments = [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": round(duration, 2),
                    "text": transcript,
                }
            ]

        return {
            "transcript": transcript,
            "language": language,
            "duration_seconds": duration,
            "segments_json": json.dumps(segments, ensure_ascii=False),
        }

    def __call__(self, batch: pa.Table) -> pa.Table:
        ids = batch["id"].to_pylist()
        paths = batch["path"].to_pylist()
        formats = batch["audio_format"].to_pylist()
        audio_values = batch["audio_bytes"].to_pylist()
        fallbacks = batch["fallback_transcript"].to_pylist()

        results = []
        for audio_bytes, audio_format, fallback in zip(
            audio_values,
            formats,
            fallbacks,
            strict=True,
        ):
            if self.backend == "placeholder":
                results.append(self._transcribe_placeholder(str(fallback or "")))
            elif self.backend == "faster-whisper":
                results.append(
                    self._transcribe_with_whisper(
                        bytes(audio_bytes or b""),
                        str(audio_format or "wav"),
                    )
                )
            else:
                raise ValueError(f"Unsupported transcription backend: {self.backend}")

        return pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "path": pa.array(paths, type=pa.string()),
                "transcript": pa.array(
                    [result["transcript"] for result in results],
                    type=pa.string(),
                ),
                "language": pa.array(
                    [result["language"] for result in results],
                    type=pa.string(),
                ),
                "duration_seconds": pa.array(
                    [result["duration_seconds"] for result in results],
                    type=pa.float64(),
                ),
                "segments_json": pa.array(
                    [result["segments_json"] for result in results],
                    type=pa.string(),
                ),
            }
        )


def local_summary(text: str, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:")
    return truncated + "..."


def openai_summary(
    text: str,
    *,
    model: str,
    translated_language: str,
) -> tuple[str, str]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install OpenAI dependencies first: pip install openai") from exc

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the transcript and translate the summary. "
                    "Return compact JSON with keys summary and translated_summary."
                ),
            },
            {
                "role": "user",
                "content": (f"Target translation language: {translated_language}\n\nTranscript:\n{text}"),
            },
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or "{}"
    try:
        parsed = json.loads(content)
        return str(parsed["summary"]), str(parsed["translated_summary"])
    except Exception:
        summary = local_summary(content, max_chars=420)
        return summary, summary


def summarize_rows(table: pa.Table, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = table.to_pylist()
    output = []
    for row in rows:
        transcript = str(row["transcript"] or "")
        if args.summary_backend == "local":
            summary = local_summary(transcript, max_chars=args.summary_max_chars)
            translated = f"[{args.translated_language} translation not generated] {summary}"
        elif args.summary_backend == "openai":
            summary, translated = openai_summary(
                transcript,
                model=args.openai_model,
                translated_language=args.translated_language,
            )
        else:
            raise ValueError(f"Unsupported summary backend: {args.summary_backend}")

        output.append(
            {
                "id": row["id"],
                "path": row["path"],
                "language": row["language"],
                "transcript": transcript,
                "summary": summary,
                "translated_summary": translated,
            }
        )
    return output


def subtitle_rows(table: pa.Table, *, translated_language: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parent in table.to_pylist():
        segments = json.loads(parent["segments_json"] or "[]")
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            rows.append(
                {
                    "id": int(parent["id"]),
                    "path": str(parent["path"]),
                    "segment_id": int(segment.get("id", len(rows))),
                    "start": float(segment.get("start", 0.0)),
                    "end": float(segment.get("end", 0.0)),
                    "text": text,
                    "translated_text": (f"[{translated_language} translation not generated] {text}"),
                }
            )
    if not rows:
        raise RuntimeError("No subtitle segments were produced.")
    return rows


def append_embedding(base: pa.Table, embedding_rel: Any) -> pa.Table:
    embeddings = collect_relation(embedding_rel)
    if base.num_rows != embeddings.num_rows:
        raise RuntimeError(f"Embedding row count mismatch: {embeddings.num_rows} vs {base.num_rows}")
    if "embedding" not in embeddings.column_names:
        raise RuntimeError("Embedding output column was not returned.")
    return base.append_column("embedding", embeddings["embedding"])


def embedding_dims(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(
    *,
    summaries: list[dict[str, Any]],
    subtitles: list[dict[str, Any]],
    embedded_table: pa.Table,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(summaries, output_dir / "summaries.csv")
    save_csv(subtitles, output_dir / "subtitles.csv")

    embedding_rows = []
    for row in embedded_table.to_pylist():
        embedding_rows.append(
            {
                "id": row["id"],
                "path": row["path"],
                "segment_id": row["segment_id"],
                "start": row["start"],
                "end": row["end"],
                "text": row["text"],
                "embedding_dim": embedding_dims(row["embedding"]),
            }
        )
    save_csv(embedding_rows, output_dir / "segment_embeddings.csv")


def run(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")

    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    conn = vane.connect()
    rel = (
        sample_relation(conn, args.limit)
        if args.source == "sample"
        else glob_relation(conn, args.audio_glob, args.limit)
    )

    transcriber = TranscribeAudioBatch(
        backend=args.transcription_backend,
        model_id=args.whisper_model_id,
        compute_type=args.compute_type,
        device=args.device,
        language=args.language or None,
        whisper_batch_size=args.whisper_batch_size,
        vad_filter=not args.no_vad,
        local_files_only=args.local_files_only,
    )
    map_kwargs: dict[str, Any] = {
        "schema": {
            "id": vane.sqltypes.BIGINT,
            "path": vane.sqltypes.VARCHAR,
            "transcript": vane.sqltypes.VARCHAR,
            "language": vane.sqltypes.VARCHAR,
            "duration_seconds": vane.sqltypes.DOUBLE,
            "segments_json": vane.sqltypes.VARCHAR,
        },
        "batch_size": args.batch_size,
    }
    if args.gpus is not None:
        map_kwargs["gpus"] = args.gpus

    transcripts = rel.map_batches(transcriber.__call__, **map_kwargs)
    transcript_table = collect_relation(transcripts)

    summaries = summarize_rows(transcript_table, args)
    subtitles = subtitle_rows(
        transcript_table,
        translated_language=args.translated_language,
    )
    subtitle_rel = relation_from_dicts(conn, subtitles)
    embedded_only = embed_text(
        subtitle_rel,
        "text",
        provider="transformers",
        model=args.embedding_model_id,
        output_column="embedding",
        batch_size=args.embedding_batch_size,
    )
    embedded_table = append_embedding(collect_relation(subtitle_rel), embedded_only)

    output_dir = Path(args.output_dir)
    save_outputs(
        summaries=summaries,
        subtitles=subtitles,
        embedded_table=embedded_table,
        output_dir=output_dir,
    )

    print(f"\nTranscribed rows: {transcript_table.num_rows}")
    print(f"Subtitle rows: {embedded_table.num_rows}")
    print(f"Output directory: {output_dir}")

    relation_from_dicts(conn, summaries).query(
        "summaries",
        """
        select
            id,
            path,
            language,
            left(transcript, 72) as transcript,
            left(summary, 72) as summary
        from summaries
        order by id
        """,
    ).show(max_width=160)

    embedded_preview = relation_from_dicts(
        conn,
        [
            {
                "id": row["id"],
                "segment_id": row["segment_id"],
                "start": row["start"],
                "end": row["end"],
                "text": row["text"],
                "embedding_dim": embedding_dims(row["embedding"]),
            }
            for row in embedded_table.to_pylist()
        ],
        {
            "id": "BIGINT",
            "segment_id": "BIGINT",
            "start": "DOUBLE",
            "end": "DOUBLE",
            "text": "VARCHAR",
            "embedding_dim": "BIGINT",
        },
    )
    embedded_preview.query(
        "segments",
        """
        select
            id,
            segment_id,
            round(start, 2) as start,
            round("end", 2) as "end",
            left(text, 72) as text,
            embedding_dim
        from segments
        order by id, segment_id
        """,
    ).show(max_width=160)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe, summarize, subtitle, and embed audio with Vane.",
    )
    parser.add_argument("--source", choices=["sample", "glob"], default="sample")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--audio-glob",
        default="examples/data/audio/*",
        help="Local audio glob used when --source glob.",
    )
    parser.add_argument(
        "--transcription-backend",
        choices=["placeholder", "faster-whisper"],
        default="placeholder",
    )
    parser.add_argument("--whisper-model-id", default=DEFAULT_WHISPER_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="float32")
    parser.add_argument("--language", default="")
    parser.add_argument("--whisper-batch-size", type=int, default=16)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument(
        "--summary-backend",
        choices=["local", "openai"],
        default="local",
    )
    parser.add_argument("--openai-model", default="gpt-4o-mini")
    parser.add_argument("--summary-max-chars", type=int, default=320)
    parser.add_argument("--translated-language", default="Simplified Chinese")
    parser.add_argument("--embedding-model-id", default=DEFAULT_EMBEDDING_MODEL_ID)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1 so models load only from local cache.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--gpus", type=float, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
