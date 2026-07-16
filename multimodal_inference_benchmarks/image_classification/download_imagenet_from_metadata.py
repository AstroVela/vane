#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

S3_METADATA = "s3://ray-example-data/imagenet/metadata_file.parquet"
S3_PREFIX = "s3://ray-example-data/imagenet/"


def download_metadata(local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["s5cmd", "--no-sign-request", "cp", S3_METADATA, str(local_path)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def iter_image_urls(metadata_path: Path):
    pf = pq.ParquetFile(metadata_path)
    for batch in pf.iter_batches(columns=["image_url"]):
        for url in batch.column(0).to_pylist():
            if url:
                yield url


def to_relative_path(url: str, prefix: str) -> str:
    if url.startswith(prefix):
        return url[len(prefix) :]
    # Fallback: keep only filename if prefix doesn't match.
    return url.rsplit("/", 1)[-1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Generate an s5cmd batch file from ImageNet metadata and optionally run it.")
    )
    parser.add_argument(
        "--metadata",
        default="metadata_file.parquet",
        help="Local path for the metadata parquet.",
    )
    parser.add_argument(
        "--out-dir",
        default="./imagenet",
        help="Local directory to download images into.",
    )
    parser.add_argument(
        "--batch-file",
        default="download_imagenet.s5cmd",
        help="Output s5cmd batch file path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images (0 means no limit).",
    )
    parser.add_argument(
        "--skip-metadata-download",
        action="store_true",
        help="Assume metadata parquet is already present locally.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run s5cmd after generating the batch file.",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        if args.skip_metadata_download:
            print(f"Metadata not found at {metadata_path} and --skip-metadata-download set.")
            return 1
        download_metadata(metadata_path)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_path = Path(args.batch_file).resolve()

    written = 0
    with batch_path.open("w", encoding="utf-8") as handle:
        for url in iter_image_urls(metadata_path):
            if args.limit and written >= args.limit:
                break
            rel = to_relative_path(url, S3_PREFIX)
            dest = out_dir / rel
            handle.write(f'cp "{url}" "{dest}"\n')
            written += 1

    print(f"Wrote {written} commands to {batch_path}")
    print(f"Example: s5cmd --no-sign-request run {batch_path}")

    if args.run:
        cmd = ["s5cmd", "--no-sign-request", "run", str(batch_path)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
