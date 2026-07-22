#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate images from text prompts with Vane.

This example adapts the workflow from Daft's image generation tutorial:
https://docs.daft.ai/en/stable/examples/image-generation/

The Daft version reads LAION prompt metadata, previews source images, and
uses a Stable Diffusion UDF with GPU resources. This Vane version keeps the
same shape:

1. Build a relation of text prompts.
2. Apply a batch UDF that generates PNG image bytes.
3. Save generated images plus a metadata CSV.

The default backend is a deterministic placeholder generator, so the example
can run without downloading a diffusion model. Use ``--backend diffusers`` for
real Stable Diffusion generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import re
import struct
import zlib
from pathlib import Path
from typing import Any

import pyarrow as pa

import vane

DEFAULT_PARQUET_PATH = "s3://daft-oss-public-data/tutorials/laion-parquet/train-00000-of-00001-6f24a7497df494ae.parquet"
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_OUTPUT_DIR = Path("examples/output/image_generation")


SAMPLE_PROMPTS = [
    "A watercolor sketch of a compact data warehouse engine on a desk",
    "A cinematic photo of neon query plans floating above a city street",
    "A clean product render of a tiny robot sorting parquet files",
    "An isometric illustration of GPU workers generating images in parallel",
]


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def relation_from_rows(
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


def png_chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def placeholder_png(width: int, height: int, prompt: str) -> bytes:
    """Create a deterministic PNG thumbnail from prompt text."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    top = digest[0], digest[1], digest[2]
    bottom = digest[3], digest[4], digest[5]
    accent = digest[6], digest[7], digest[8]
    stripe_width = max(8, width // 12)

    rows: list[bytes] = []
    for y in range(height):
        t = y / max(1, height - 1)
        base = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        row = bytearray(b"\x00")
        for x in range(width):
            wave = 0.5 + 0.5 * math.sin((x + digest[9]) / max(1, width) * math.tau * 3)
            stripe = 1 if (x // stripe_width + y // stripe_width) % 2 == 0 else 0
            rgb = []
            for i in range(3):
                value = base[i] * 0.78 + accent[i] * 0.22 * wave
                if stripe:
                    value += 22
                rgb.append(max(0, min(255, int(value))))
            row.extend(rgb)
        rows.append(bytes(row))

    raw = b"".join(rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + png_chunk(b"IEND", b"")
    )


def pil_to_png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class GenerateImageFromTextBatch:
    """Batch UDF that generates one PNG image per prompt."""

    def __init__(
        self,
        *,
        backend: str,
        model_id: str,
        prompt_column: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        device: str,
        dtype: str,
        seed: int | None,
        revision: str | None,
        local_files_only: bool,
    ):
        self.backend = backend
        self.model_id = model_id
        self.prompt_column = prompt_column
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = device
        self.dtype = dtype
        self.seed = seed
        self.revision = revision
        self.local_files_only = local_files_only
        self._pipe = None

    def _load_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Install image generation dependencies first: "
                "pip install diffusers transformers accelerate torch Pillow"
            ) from exc

        torch_dtype = getattr(torch, self.dtype)
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load the diffusion model. Download it first, then pass "
                "the local directory with --model-id. For example:\n\n"
                "  hf download "
                f"{DEFAULT_MODEL_ID} --local-dir ~/.cache/vane/models/stable-diffusion-v1-5\n"
                "  python "
                "examples/image_generation.py --backend diffusers "
                "--model-id ~/.cache/vane/models/stable-diffusion-v1-5 "
                "--source sample --limit 2 --device cuda --dtype float16\n\n"
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        pipe.enable_attention_slicing(1)
        if self.device:
            pipe = pipe.to(self.device)
        self._pipe = pipe
        return pipe

    def _generate_with_diffusers(self, prompt: str, index: int) -> bytes:
        import torch

        pipe = self._load_pipe()
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=self.device or "cpu").manual_seed(self.seed + index)
        image = pipe(
            prompt,
            num_inference_steps=self.num_inference_steps,
            height=self.height,
            width=self.width,
            guidance_scale=self.guidance_scale,
            generator=generator,
        ).images[0]
        return pil_to_png_bytes(image)

    def _generate_one(self, prompt: str, index: int) -> bytes:
        if self.backend == "placeholder":
            return placeholder_png(self.width, self.height, prompt)
        if self.backend == "diffusers":
            return self._generate_with_diffusers(prompt, index)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def __call__(self, batch: pa.Table) -> pa.Table:
        prompts = [str(value or "") for value in batch[self.prompt_column].to_pylist()]
        ids = batch["id"].to_pylist()
        source_urls = batch["source_url"].to_pylist()
        aesthetic_scores = batch["aesthetic_score"].to_pylist()
        images = [
            self._generate_one(prompt, int(row_id) if isinstance(row_id, int) else i)
            for i, (prompt, row_id) in enumerate(zip(prompts, ids, strict=True))
        ]
        return pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "prompt": pa.array(prompts, type=pa.string()),
                "source_url": pa.array(source_urls, type=pa.string()),
                "aesthetic_score": pa.array(aesthetic_scores, type=pa.float64()),
                "generated_image": pa.array(images, type=pa.binary()),
            }
        )


def sample_relation(conn: Any, limit: int) -> Any:
    prompts = [
        SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
        + (f" variation {i // len(SAMPLE_PROMPTS)}" if i >= len(SAMPLE_PROMPTS) else "")
        for i in range(limit)
    ]
    rows = [
        {
            "id": i,
            "prompt": prompt,
            "source_url": None,
            "aesthetic_score": None,
        }
        for i, prompt in enumerate(prompts)
    ]
    return relation_from_rows(
        conn,
        rows,
        {
            "id": "BIGINT",
            "prompt": "VARCHAR",
            "source_url": "VARCHAR",
            "aesthetic_score": "DOUBLE",
        },
    )


def load_laion_relation(conn: Any, parquet_path: str, limit: int) -> Any:
    try:
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")
    except Exception:
        # httpfs may already be installed/loaded, or the path may be local.
        pass
    try:
        conn.execute("SET s3_region='us-west-2'")
        conn.execute("SET s3_url_style='path'")
    except Exception:
        pass

    return conn.sql(
        f"""
        select
            row_number() over () - 1 as id,
            TEXT as prompt,
            URL as source_url,
            cast(AESTHETIC_SCORE as double) as aesthetic_score
        from read_parquet({sql_literal(parquet_path)})
        where TEXT is not null
          and length(TEXT) > 50
        limit {int(limit)}
        """
    )


def sanitize_file_stem(value: Any) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return stem[:80] or "image"


def save_generated_images(table: pa.Table, output_dir: Path) -> pa.Table:
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = table["id"].to_pylist()
    prompts = table["prompt"].to_pylist()
    images = table["generated_image"].to_pylist()
    paths: list[str] = []

    for ordinal, (row_id, image_bytes) in enumerate(zip(ids, images, strict=True)):
        file_name = f"{ordinal:04d}-{sanitize_file_stem(row_id)}.png"
        path = output_dir / file_name
        path.write_bytes(image_bytes)
        paths.append(str(path))

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=["id", "prompt", "generated_path"],
        )
        writer.writeheader()
        for row_id, prompt, path in zip(ids, prompts, paths, strict=True):
            writer.writerow({"id": row_id, "prompt": prompt, "generated_path": path})

    return table.append_column("generated_path", pa.array(paths, type=pa.string()))


def run(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")
    if args.width < 1 or args.height < 1:
        raise SystemExit("--width and --height must be positive.")

    conn = vane.connect()
    rel = (
        sample_relation(conn, args.limit)
        if args.source == "sample"
        else load_laion_relation(conn, args.parquet_path, args.limit)
    )

    generator = GenerateImageFromTextBatch(
        backend=args.backend,
        model_id=args.model_id,
        prompt_column="prompt",
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )

    map_kwargs: dict[str, Any] = {
        "schema": {
            "id": vane.sqltypes.BIGINT,
            "prompt": vane.sqltypes.VARCHAR,
            "source_url": vane.sqltypes.VARCHAR,
            "aesthetic_score": vane.sqltypes.DOUBLE,
            "generated_image": vane.sqltypes.BLOB,
        },
        "batch_size": args.batch_size,
    }
    if args.gpus is not None:
        map_kwargs["gpus"] = args.gpus

    generated = rel.map_batches(generator.__call__, **map_kwargs)
    generated_table = collect_relation(generated)
    written_table = save_generated_images(generated_table, Path(args.output_dir))
    written = relation_from_rows(
        conn,
        [
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "aesthetic_score": row["aesthetic_score"],
                "generated_path": row["generated_path"],
            }
            for row in written_table.to_pylist()
        ],
        {
            "id": "BIGINT",
            "prompt": "VARCHAR",
            "aesthetic_score": "DOUBLE",
            "generated_path": "VARCHAR",
        },
    )

    print(f"\nGenerated rows: {written_table.num_rows}")
    print(f"Output directory: {args.output_dir}")
    written.query(
        "generated",
        """
        select
            id,
            left(prompt, 72) as prompt,
            aesthetic_score,
            generated_path
        from generated
        order by id
        """,
    ).show(max_width=140)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate images from text prompts with Vane map_batches.",
    )
    parser.add_argument("--source", choices=["sample", "laion"], default="sample")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--parquet-path", default=DEFAULT_PARQUET_PATH)
    parser.add_argument(
        "--backend",
        choices=["placeholder", "diffusers"],
        default="placeholder",
        help="Use placeholder for a dependency-light dry run; diffusers for Stable Diffusion.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load model files from the local Hugging Face cache or local path.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpus", type=float, default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
