#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

S3_PARQUET_GLOB = "s3://ray-example-data/common_voice_17/parquet/*.parquet"
S3_PREFIX = "s3://ray-example-data/common_voice_17/parquet/"


def sanitize_s3_uri_for_s5cmd(uri: str) -> str:
    # s5cmd expects bucket path form (s3://bucket/key), not userinfo form (s3://anonymous@bucket/key).
    return uri.replace("s3://anonymous@", "s3://", 1)


def list_remote_parquet(s3_glob: str) -> list[str]:
    s3_glob = sanitize_s3_uri_for_s5cmd(s3_glob)
    cmd = ["s5cmd", "--no-sign-request", "ls", s3_glob]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"failed to list parquet files: exit code {proc.returncode}")

    glob_prefix = s3_glob.split("*", 1)[0]
    urls: list[str] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        entry = parts[-1]
        if entry.startswith("s3://"):
            url = entry
        else:
            url = glob_prefix + entry.lstrip("/")
        if not url.lower().endswith(".parquet"):
            continue
        urls.append(url)
    return sorted(set(urls))


def to_relative_path(url: str, prefix: str) -> str:
    if url.startswith(prefix):
        return url[len(prefix) :]
    return url.rsplit("/", 1)[-1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=("Generate an s5cmd batch file for Common Voice parquet download and optionally run it.")
    )
    parser.add_argument(
        "--s3-glob",
        default=S3_PARQUET_GLOB,
        help="S3 parquet glob to list via s5cmd ls.",
    )
    parser.add_argument(
        "--s3-prefix",
        default=S3_PREFIX,
        help="S3 prefix used to keep relative paths in output dir.",
    )
    parser.add_argument(
        "--out-dir",
        default="./common_voice_17/parquet",
        help="Local directory to download parquet files into.",
    )
    parser.add_argument(
        "--batch-file",
        default="download_common_voice_parquet.s5cmd",
        help="Output s5cmd batch file path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of parquet files (0 means no limit).",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run s5cmd after generating the batch file.",
    )
    args = parser.parse_args()

    urls = list_remote_parquet(args.s3_glob)
    if args.limit > 0:
        urls = urls[: args.limit]

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_path = Path(args.batch_file).resolve()

    written = 0
    with batch_path.open("w", encoding="utf-8") as handle:
        for url in urls:
            rel = to_relative_path(url, sanitize_s3_uri_for_s5cmd(args.s3_prefix))
            dest = out_dir / rel
            handle.write(f'cp "{sanitize_s3_uri_for_s5cmd(url)}" "{dest}"\n')
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
