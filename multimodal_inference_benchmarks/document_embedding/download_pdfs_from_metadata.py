#!/usr/bin/env python3
"""Generate an s5cmd batch file from digitalcorpora metadata and optionally run it.

Usage examples:
    # Generate batch file only
    python download_pdfs_from_metadata.py

    # Generate and run download
    python download_pdfs_from_metadata.py --run

    # Limit to first 100 PDFs
    python download_pdfs_from_metadata.py --limit 100 --run

    # Use a custom metadata path already on disk
    python download_pdfs_from_metadata.py --metadata /path/to/metadata --skip-metadata-download --run
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pyarrow.parquet as pq

S3_METADATA = "s3://ray-example-data/pdf_dump_metadata/"
S3_PREFIX = "s3://ray-example-data/pdf_dump/"


def download_metadata(local_path: Path) -> None:
    """Download metadata parquet from S3 using s5cmd."""
    local_path.mkdir(parents=True, exist_ok=True)
    cmd = ["s5cmd", "--no-sign-request", "cp", "--flatten", f"{S3_METADATA}*", str(local_path) + "/"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def iter_pdf_urls(metadata_path: Path):
    """Yield uploaded_pdf_path values from all parquet files under metadata_path."""
    metadata_path = metadata_path.resolve()
    if metadata_path.is_file():
        parquet_files = [metadata_path]
    else:
        parquet_files = sorted(metadata_path.rglob("*.parquet"))
    for pf_path in parquet_files:
        pf = pq.ParquetFile(pf_path)
        for batch in pf.iter_batches(columns=["uploaded_pdf_path"]):
            for url in batch.column(0).to_pylist():
                if url:
                    yield url


def to_relative_path(url: str, prefix: str) -> str:
    """Convert an S3 URL to a relative path by stripping the prefix."""
    if url.startswith(prefix):
        return url[len(prefix) :]
    # Fallback: keep only filename if prefix doesn't match.
    return url.rsplit("/", 1)[-1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Generate an s5cmd batch file from digitalcorpora PDF metadata and optionally run it.")
    )
    parser.add_argument(
        "--metadata",
        default="digitalcorpora/metadata",
        help="Local path for the metadata parquet directory (default: digitalcorpora/metadata).",
    )
    parser.add_argument(
        "--out-dir",
        default="./digitalcorpora/pdf_dump",
        help="Local directory to download PDFs into.",
    )
    parser.add_argument(
        "--batch-file",
        default="download_pdfs.s5cmd",
        help="Output s5cmd batch file path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of PDFs (0 means no limit).",
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
        for url in iter_pdf_urls(metadata_path):
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
