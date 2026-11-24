# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Probe block high-index VM/page costs on tmpfs; separate from adoption throughput."""

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
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--sqlite", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir()
    report = dict(
        status="RUNNING",
        scope=__doc__,
        orders=[["full", "primary"], ["primary", "full"]] * 3,
        runs=[],
        cleanup=[],
        inputs={
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), args.binary, args.source, args.library, args.sqlite)
        },
    )
    try:
        for pair, order in enumerate(report["orders"]):
            for mode in order:
                work = Path(tempfile.mkdtemp(prefix="owned-vane-index-rca-", dir="/dev/shm"))
                start = time.time()
                try:
                    log = out / f"{pair}-{mode}.log"
                    with log.open("w") as stream:
                        result = subprocess.run(
                            [str(args.binary.resolve()), str(work / "db.sqlite"), mode],
                            stdout=stream,
                            stderr=stream,
                            timeout=60,
                        )
                    report["runs"].append(
                        dict(
                            pair=pair,
                            mode=mode,
                            returncode=result.returncode,
                            started_epoch=start,
                            ended_epoch=time.time(),
                            log=str(log),
                        )
                    )
                    print(pair, mode, result.returncode, log.read_text().splitlines()[0], flush=True)
                    result.check_returncode()
                finally:
                    files = [
                        dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                        for p in work.rglob("*")
                        if p.is_file()
                    ]
                    shutil.rmtree(work)
                    report["cleanup"].append(dict(path=str(work), removed=not work.exists(), files=files))
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
