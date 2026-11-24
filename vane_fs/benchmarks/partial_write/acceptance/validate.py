# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

out = Path(sys.argv[1]).resolve()
assert json.loads((out / "probe-build.json").read_text())["status"] == "PASS"
r = {"status": "RUNNING", "started_epoch": time.time(), "runs": [], "cleanup": []}


def save():
    (out / "tests.json").write_text(json.dumps(r, indent=2) + "\n")


def run(label, cmd, env=None, expected=0):
    t = time.time()
    with (out / (label + ".log")).open("w") as log:
        p = subprocess.run(cmd, env=env, cwd="/tmp", stdout=log, stderr=log, timeout=600)
    r["runs"].append(
        {
            "label": label,
            "argv": cmd,
            "returncode": p.returncode,
            "expected": expected,
            "started_epoch": t,
            "ended_epoch": time.time(),
        }
    )
    save()
    assert p.returncode == expected, (label, p.returncode)
    print(label, "PASS", flush=True)


def clean(path):
    if not path.exists():
        return
    assert str(path) not in Path("/proc/mounts").read_text()
    logs = [p for p in path.rglob("*.log") if p.is_file()]
    for p in logs:
        q = out / "test-logs" / path.name / p.relative_to(path)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
    shutil.rmtree(path)
    r["cleanup"].append({"path": str(path), "removed": not path.exists(), "mounts_absent": True})
    save()


try:
    for v in ["baseline", "updated"]:
        run(v + "-native", ["ctest", "--test-dir", str(out / v / "build/python-release"), "--output-on-failure"])
    target = out / "updated"
    python = target / "venv/bin/python"
    temp = out / "owned-updated-python"
    env = dict(os.environ, VANE_FS_MOUNT_BINARY=str(target / "build/python-release/vane-fs-mount"))
    env.pop("PYTHONPATH", None)
    run(
        "module-identity",
        [
            str(python),
            "-I",
            "-c",
            "import vane_fs._native as n;print(n.__file__);assert " + repr(str(target / "venv")) + " in n.__file__",
        ],
        env,
    )
    try:
        run(
            "updated-python",
            [
                str(python),
                "-I",
                "-m",
                "pytest",
                "--rootdir",
                str(target),
                "--confcutdir",
                str(target),
                "--basetemp",
                str(temp),
                "-q",
                str(target / "tests"),
            ],
            env,
        )
    finally:
        clean(temp)
    r["status"] = "PASS"
except BaseException as e:
    r.update(status="FAIL", error=repr(e))
    raise
finally:
    r["ended_epoch"] = time.time()
    save()
