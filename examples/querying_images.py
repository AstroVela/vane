#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Query image data with Vane.

This example adapts Daft's querying-images tutorial:
https://docs.daft.ai/en/stable/examples/querying-images/#working-with-complex-data

The Daft tutorial reads OpenImages files from S3, filters by file size,
downloads and decodes image bytes, then applies a custom ``magic_red_detector``
function to find images with large red regions.

This Vane version keeps the same shape:

1. Load image bytes from built-in samples, local files, or the OpenImages S3 prefix.
2. Apply a batch UDF that decodes each image and produces a red-region mask.
3. Sort by red pixel count and save image/mask previews plus metadata.

The default source is built-in sample images, so the example runs without S3.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from PIL import Image, ImageDraw, ImageFilter

import vane

DEFAULT_OUTPUT_DIR = Path("examples/output/querying_images")
DEFAULT_OPEN_IMAGES_LIST_URLS = [
    (
        "https://daft-oss-public-data.s3.us-west-2.amazonaws.com/"
        "?list-type=2&prefix=open-images/validation-images/&max-keys={max_keys}"
    ),
    (
        "https://s3.us-west-2.amazonaws.com/daft-oss-public-data/"
        "?list-type=2&prefix=open-images/validation-images/&max-keys={max_keys}"
    ),
    (
        "https://daft-oss-public-data.s3.amazonaws.com/"
        "?list-type=2&prefix=open-images/validation-images/&max-keys={max_keys}"
    ),
]
DEFAULT_OPEN_IMAGES_OBJECT_URL = "https://daft-oss-public-data.s3.us-west-2.amazonaws.com/{key}"


SAMPLE_SPECS = [
    {
        "path": "sample://red-wall.png",
        "description": "large red wall",
        "red_rects": [(18, 24, 238, 156)],
        "blue_rects": [(24, 164, 110, 184)],
        "green_rects": [],
    },
    {
        "path": "sample://red-sign.png",
        "description": "red sign",
        "red_rects": [(96, 46, 184, 120), (30, 135, 72, 170)],
        "blue_rects": [(180, 144, 240, 184)],
        "green_rects": [],
    },
    {
        "path": "sample://traffic-light.png",
        "description": "small red light",
        "red_rects": [(110, 32, 145, 67)],
        "blue_rects": [],
        "green_rects": [(110, 118, 145, 153)],
    },
    {
        "path": "sample://blue-water.png",
        "description": "mostly blue",
        "red_rects": [],
        "blue_rects": [(0, 0, 256, 192)],
        "green_rects": [],
    },
    {
        "path": "sample://red-border.png",
        "description": "thin red border",
        "red_rects": [(0, 0, 256, 16), (0, 176, 256, 192), (0, 0, 16, 192), (240, 0, 256, 192)],
        "blue_rects": [],
        "green_rects": [(58, 50, 198, 142)],
    },
]


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def sample_image_bytes(spec: dict[str, Any], width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), (232, 229, 220))
    draw = ImageDraw.Draw(image)
    for i in range(0, width, 16):
        shade = 222 + (i // 16) % 2 * 12
        draw.line((i, 0, i, height), fill=(shade, shade, 224))

    sx = width / 256.0
    sy = height / 192.0

    def scale_rect(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = rect
        return int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)

    for rect in spec["blue_rects"]:
        draw.rectangle(scale_rect(rect), fill=(35, 94, 184))
    for rect in spec["green_rects"]:
        draw.rectangle(scale_rect(rect), fill=(54, 154, 94))
    for rect in spec["red_rects"]:
        draw.rectangle(scale_rect(rect), fill=(224, 38, 42))

    draw.text((12, 10), str(spec["description"]), fill=(25, 25, 25))
    return pil_to_png_bytes(image)


def relation_from_rows(conn: Any, rows: list[dict[str, Any]]) -> Any:
    """Build a VALUES relation whose data can travel with a Ray plan."""
    if not rows:
        raise RuntimeError("Cannot create a relation from zero rows.")
    columns = list(rows[0])
    constant = vane.ConstantExpression
    raw = conn.values(
        *(tuple(constant(row[column]) for column in columns) for row in rows),
    )
    projections = [
        f"{quote_ident(source)} AS {quote_ident(column)}" for source, column in zip(raw.columns, columns, strict=True)
    ]
    return raw.query("input_rows", f"select {', '.join(projections)} from input_rows")


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


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


def sample_relation(conn: Any, limit: int, width: int, height: int) -> Any:
    rows = []
    for i in range(limit):
        spec = SAMPLE_SPECS[i % len(SAMPLE_SPECS)]
        image_bytes = sample_image_bytes(spec, width, height)
        rows.append(
            {
                "id": i,
                "path": spec["path"] if i < len(SAMPLE_SPECS) else f"{spec['path']}?copy={i // len(SAMPLE_SPECS)}",
                "size": len(image_bytes),
                "image_bytes": image_bytes,
            }
        )
    return relation_from_rows(conn, rows)


def glob_relation(
    conn: Any,
    *,
    image_glob: str,
    limit: int,
    min_size: int,
    max_size: int,
) -> Any:
    rows = []
    for path in sorted(glob.glob(image_glob)):
        image_bytes = Path(path).read_bytes()
        size = len(image_bytes)
        if min_size and size < min_size:
            continue
        if max_size and size > max_size:
            continue
        rows.append(
            {
                "id": len(rows),
                "path": path,
                "size": size,
                "image_bytes": image_bytes,
            }
        )
        if len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError(f"No images matched --image-glob={image_glob!r}.")
    return relation_from_rows(conn, rows)


def parse_s3_list(xml_bytes: bytes) -> list[tuple[str, int]]:
    root = ET.fromstring(xml_bytes)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    objects = []
    contents = root.findall(".//s3:Contents", namespace)
    if not contents:
        contents = root.findall(".//Contents")
    for item in contents:
        key_node = item.find("s3:Key", namespace)
        if key_node is None:
            key_node = item.find("Key")
        size_node = item.find("s3:Size", namespace)
        if size_node is None:
            size_node = item.find("Size")
        if key_node is None or size_node is None:
            continue
        objects.append((str(key_node.text or ""), int(size_node.text or "0")))
    return objects


def download_url(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "vane-example/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def open_images_relation(
    conn: Any,
    *,
    limit: int,
    min_size: int,
    max_size: int,
    max_scan: int,
) -> Any:
    objects: list[tuple[str, int]] = []
    last_error: Exception | None = None
    for template in DEFAULT_OPEN_IMAGES_LIST_URLS:
        try:
            xml_bytes = download_url(template.format(max_keys=max_scan))
            objects = parse_s3_list(xml_bytes)
            if objects:
                break
        except Exception as exc:
            last_error = exc

    if not objects:
        raise RuntimeError(
            f"Could not list the public OpenImages S3 prefix. Last error: {type(last_error).__name__}: {last_error}"
        )

    rows = []
    for key, size in objects:
        if min_size and size < min_size:
            continue
        if max_size and size > max_size:
            continue
        url = DEFAULT_OPEN_IMAGES_OBJECT_URL.format(
            key=urllib.parse.quote(key, safe="/"),
        )
        image_bytes = download_url(url)
        rows.append(
            {
                "id": len(rows),
                "path": "s3://daft-oss-public-data/" + key,
                "size": size,
                "image_bytes": image_bytes,
            }
        )
        if len(rows) >= limit:
            break

    if not rows:
        raise RuntimeError(
            "No OpenImages objects matched the requested size range. "
            "Try lowering --min-size or increasing --open-images-max-scan."
        )
    return relation_from_rows(conn, rows)


def magic_red_detector(image: Image.Image) -> Image.Image:
    """Return a mask covering red regions in an RGB image."""
    hsv = np.asarray(image.convert("HSV"))
    lower = np.array([245, 100, 100], dtype=np.uint8)
    upper = np.array([10, 255, 255], dtype=np.uint8)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hue_mask = (hue >= lower[0]) | (hue <= upper[0])
    saturation_mask = (saturation >= lower[1]) & (saturation <= upper[1])
    value_mask = (value >= lower[2]) & (value <= upper[2])
    mask = hue_mask & saturation_mask & value_mask

    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    return mask_image.filter(ImageFilter.ModeFilter(size=5))


class AnalyzeRedRegionsBatch:
    """Batch UDF that decodes images and computes red-region masks."""

    def __call__(self, batch: pa.Table) -> pa.Table:
        ids = batch["id"].to_pylist()
        paths = batch["path"].to_pylist()
        sizes = batch["size"].to_pylist()
        image_values = batch["image_bytes"].to_pylist()

        widths = []
        heights = []
        red_pixels = []
        red_fractions = []
        preview_values = []
        mask_values = []

        for image_bytes in image_values:
            image = Image.open(io.BytesIO(bytes(image_bytes or b""))).convert("RGB")
            mask = magic_red_detector(image)
            mask_array = np.asarray(mask)
            red_count = int(np.count_nonzero(mask_array))
            total_pixels = image.width * image.height

            widths.append(int(image.width))
            heights.append(int(image.height))
            red_pixels.append(red_count)
            red_fractions.append(red_count / max(1, total_pixels))
            preview_values.append(pil_to_png_bytes(image))
            mask_values.append(pil_to_png_bytes(mask.convert("RGB")))

        return pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "path": pa.array(paths, type=pa.string()),
                "size": pa.array(sizes, type=pa.int64()),
                "width": pa.array(widths, type=pa.int64()),
                "height": pa.array(heights, type=pa.int64()),
                "red_pixels": pa.array(red_pixels, type=pa.int64()),
                "red_fraction": pa.array(red_fractions, type=pa.float64()),
                "preview_png": pa.array(preview_values, type=pa.binary()),
                "red_mask_png": pa.array(mask_values, type=pa.binary()),
            }
        )


def sanitize_file_stem(value: Any) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return stem[:80] or "image"


def save_outputs(table: pa.Table, output_dir: Path, top_k: int) -> pa.Table:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = sorted(
        table.to_pylist(),
        key=lambda row: (int(row["red_pixels"]), float(row["red_fraction"])),
        reverse=True,
    )[:top_k]
    output_rows = []
    for rank, row in enumerate(rows, start=1):
        stem = f"{rank:03d}-{sanitize_file_stem(row['id'])}"
        image_path = image_dir / f"{stem}.png"
        mask_path = mask_dir / f"{stem}-red-mask.png"
        image_path.write_bytes(row["preview_png"])
        mask_path.write_bytes(row["red_mask_png"])
        output_rows.append(
            {
                "rank": rank,
                "id": row["id"],
                "path": row["path"],
                "size": row["size"],
                "width": row["width"],
                "height": row["height"],
                "red_pixels": row["red_pixels"],
                "red_fraction": row["red_fraction"],
                "image_path": str(image_path),
                "red_mask_path": str(mask_path),
            }
        )

    metadata_path = output_dir / "top_red_images.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as metadata_file:
        writer = csv.DictWriter(metadata_file, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    return pa.table({key: pa.array([row[key] for row in output_rows]) for key in output_rows[0]})


def run(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")

    conn = vane.connect()
    if args.source == "sample":
        rel = sample_relation(conn, args.limit, args.sample_width, args.sample_height)
    elif args.source == "glob":
        rel = glob_relation(
            conn,
            image_glob=args.image_glob,
            limit=args.limit,
            min_size=args.min_size,
            max_size=args.max_size,
        )
    else:
        rel = open_images_relation(
            conn,
            limit=args.limit,
            min_size=args.min_size,
            max_size=args.max_size,
            max_scan=args.open_images_max_scan,
        )

    analyzer = AnalyzeRedRegionsBatch()
    analyzed = rel.map_batches(
        analyzer.__call__,
        schema={
            "id": vane.sqltypes.BIGINT,
            "path": vane.sqltypes.VARCHAR,
            "size": vane.sqltypes.BIGINT,
            "width": vane.sqltypes.BIGINT,
            "height": vane.sqltypes.BIGINT,
            "red_pixels": vane.sqltypes.BIGINT,
            "red_fraction": vane.sqltypes.DOUBLE,
            "preview_png": vane.sqltypes.BLOB,
            "red_mask_png": vane.sqltypes.BLOB,
        },
        batch_size=args.batch_size,
    )
    analyzed_table = collect_relation(analyzed)
    top_table = save_outputs(
        analyzed_table,
        Path(args.output_dir),
        min(args.top_k, analyzed_table.num_rows),
    )
    top_rel = relation_from_rows(conn, top_table.to_pylist())

    print(f"\nAnalyzed images: {analyzed_table.num_rows}")
    print(f"Top rows saved: {top_table.num_rows}")
    print(f"Output directory: {args.output_dir}")
    top_rel.query(
        "top_images",
        """
        select
            rank,
            id,
            left(path, 72) as path,
            width,
            height,
            red_pixels,
            round(red_fraction, 4) as red_fraction,
            image_path,
            red_mask_path
        from top_images
        order by rank
        """,
    ).show(max_width=180)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find images with large red regions using Vane map_batches.",
    )
    parser.add_argument(
        "--source",
        choices=["sample", "glob", "open-images"],
        default="sample",
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--image-glob", default="examples/data/images/*")
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--max-size", type=int, default=0)
    parser.add_argument("--open-images-max-scan", type=int, default=1000)
    parser.add_argument("--sample-width", type=int, default=256)
    parser.add_argument("--sample-height", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
