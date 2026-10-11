# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bracket the ext4 window with the existing ordinary-file sync reproducer."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    args = parser.parse_args()
    out = args.artifacts.resolve() / ("control-" + args.phase)
    out.mkdir()
    binary = args.binary.resolve()
    report = {
        "status": "RUNNING",
        "started_epoch": time.time(),
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "samples": [],
        "cleanup": [],
    }

    def save():
        (out / "results.json").write_text(json.dumps(report, indent=2) + "\n")

    try:
        for rep in range(3):
            for framed, pace in ((0, 0), (1, 100)):
                assert shutil.disk_usage(out).free > 4 * 1024**3
                work = Path(tempfile.mkdtemp(prefix="owned-control-", dir=out))
                command = [str(binary), str(work), str(framed), "0", str(pace), "64"]
                row = {"command": command, "rep": rep, "started_epoch": time.time()}
                report["samples"].append(row)
                try:
                    result = subprocess.run(command, text=True, capture_output=True, timeout=90)
                    name = f"{rep}-{framed}"
                    (out / (name + ".stdout")).write_text(result.stdout)
                    (out / (name + ".stderr")).write_text(result.stderr)
                    result.check_returncode()
                    row.update(metrics=json.loads(result.stdout), ended_epoch=time.time())
                    assert row["metrics"]["status"] == "PASS"
                    trace = out / (name + "-timeline.json")
                    shutil.copy2(work / "io.timeline.json", trace)
                    row["timeline_file"] = str(trace)
                    print(args.phase, rep, framed, row["metrics"]["seconds"], flush=True)
                finally:
                    files = []
                    for path in work.rglob("*"):
                        if path.is_file():
                            digest = hashlib.sha256()
                            with path.open("rb") as stream:
                                for block in iter(lambda: stream.read(1024**2), b""):
                                    digest.update(block)
                            files.append(
                                {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
                            )
                    shutil.rmtree(work)
                    report["cleanup"].append({"path": str(work), "files": files, "removed": not work.exists()})
                    save()
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save()


if __name__ == "__main__":
    main()
