# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run frozen extent FUSE windows serially, without owned builds or tests."""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from prepare import digest, save


def interrupted(signum, frame):
    raise InterruptedError(signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    assert all(digest(Path(p)) == sha for p, sha in build["inputs"].items())
    report = dict(status="RUNNING", started_epoch=time.time(), commands=[])
    assert not (out / "run.json").exists()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        for window, config in build["windows"].items():
            root = Path(config["root"])
            log = out / (window + ".log")
            argv = [sys.executable, "-u", str(root / "run_fuse.py"), str(root), "--medium", "ext4"]
            start = time.time()
            with log.open("w") as stream:
                process = subprocess.Popen(argv, stdout=stream, stderr=stream)
                try:
                    returncode = process.wait(timeout=3600)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=180)
            report["commands"].append(
                dict(
                    window=window,
                    argv=argv,
                    started_epoch=start,
                    ended_epoch=time.time(),
                    returncode=returncode,
                    log=str(log),
                    log_sha256=digest(log),
                )
            )
            save(out / "run.json", report)
            assert returncode == 0, log
            result = json.loads((root / "ext4-plain/results.json").read_text())
            assert result["status"] == "PASS" and result["monitor_stopped"]
            assert len(result["fuse"]) + len(result["aged_fuse"]) == config["mounts"]
            assert all(digest(Path(p)) == sha for p, sha in build["inputs"].items())
            print("PASS", window, config["mounts"], "mounts", flush=True)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save(out / "run.json", report)


if __name__ == "__main__":
    main()
