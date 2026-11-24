# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Count allocation/syscall bytes separately from the plain timing windows."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    report = dict(status="RUNNING", samples=[], cleanup=[])
    try:
        for variant in ("baseline", "updated"):
            binary = out / variant / "allocation"
            assert digest(binary) == build["binaries"][str(binary.relative_to(out))]
            for mode in ("aligned", "split"):
                owned = Path(tempfile.mkdtemp(prefix="vane-external-allocation-", dir="/dev/shm"))
                try:
                    label = variant + "-" + mode + "-allocation"
                    with (out / (label + ".stderr")).open("w") as log:
                        result = subprocess.run(
                            [str(binary), str(owned / "db.sqlite"), mode],
                            stdout=subprocess.PIPE,
                            stderr=log,
                            text=True,
                            timeout=120,
                        )
                    (out / (label + ".stdout")).write_text(result.stdout)
                    result.check_returncode()
                    sample = json.loads(result.stdout)
                    assert sample["content_verified"] and sample["logical_bytes"] == 32 * 1024**2
                    report["samples"].append(dict(variant=variant, **sample))
                finally:
                    # The synchronous child has stopped before inspecting and
                    # deleting only this run's generated database and sidecars.
                    assert str(owned) not in Path("/proc/mounts").read_text()
                    files = [
                        dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p))
                        for p in owned.rglob("*")
                        if p.is_file()
                    ]
                    shutil.rmtree(owned)
                    report["cleanup"].append(dict(path=str(owned), removed=not owned.exists(), files=files))
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        (out / "allocation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
