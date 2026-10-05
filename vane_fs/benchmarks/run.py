# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run native benchmarks, retain measurements/hashes, and remove owned data."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    with path.open("rb") as source:
        value = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
        return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    binary, output = args.binary.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    component = Path(__file__).resolve().parents[1]
    profiles = [("small-1k", 1000, 1), ("small-10k", 10000, 1), ("large-64mib", 100, 64)]
    if args.quick:
        profiles = [("smoke", 10, 1)]
    source_paths = [
        component / "CMakeLists.txt",
        component / "vcpkg.json",
        Path(__file__).resolve(),
        *component.glob("src/*"),
        *component.glob("include/**/*.hpp"),
    ]
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "binary_sha256": digest(binary),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=component, text=True).strip(),
        "source_sha256": {
            str(path.relative_to(component)): digest(path) for path in sorted(source_paths) if path.is_file()
        },
        "cache_policy": "warm OS cache; no cache eviction; synchronous=FULL; WAL autocheckpoint defaults",
        "runs": [],
    }
    cleanup = []
    try:
        for name, files, large_mib in profiles:
            for repetition in range(args.repeat):
                free = shutil.disk_usage(output).free
                required = 4 * (files * 8192 + large_mib * 1024 * 1024) + 128 * 1024 * 1024
                if free < required:
                    raise RuntimeError(f"Insufficient free disk space: {free} < {required}")
                run_name = f"{name}-{repetition}"
                temporary = Path(tempfile.mkdtemp(prefix=f"{run_name}-data-", dir=output))
                database = temporary / "workspace.sqlite"
                command = [
                    str(binary),
                    str(database),
                    str(files),
                    str(large_mib),
                    *(["8", "4", "10"] if args.quick else ["64", "16", "100"]),
                ]
                process = None
                record = {"name": run_name, "command": command, "free_bytes_before": free}
                report["runs"].append(record)
                try:
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    stdout, stderr = process.communicate(timeout=1200)
                    (output / f"{run_name}.log").write_text(stderr)
                    record["returncode"] = process.returncode
                    if process.returncode:
                        raise RuntimeError(f"Native benchmark failed: {stderr}")
                    record["measurements"] = json.loads(stdout)
                    if not record["measurements"]["validated"]:
                        raise RuntimeError("Native benchmark validation failed")
                    if record["measurements"]["build_type"] != "Release":
                        raise RuntimeError("Performance measurements require a Release build")
                    print(f"{run_name}: validated", flush=True)
                except BaseException as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    if process is not None and process.poll() is None:
                        process.kill()
                        _, stderr = process.communicate()
                        (output / f"{run_name}.log").write_text(stderr)
                    removed = []
                    try:
                        for path in sorted(temporary.rglob("*")):
                            if path.is_file():
                                removed.append(
                                    {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
                                )
                    finally:
                        # All native workers have exited. Hashing and cleanup are
                        # outside every measured comparison window, even on error.
                        try:
                            shutil.rmtree(temporary)
                        finally:
                            cleanup.append(
                                {"directory": str(temporary), "files": removed, "removed": not temporary.exists()}
                            )
                            record["free_bytes_after_cleanup"] = shutil.disk_usage(output).free
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")


if __name__ == "__main__":
    main()
