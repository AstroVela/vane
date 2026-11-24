# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run only the component tests, plus the bounded version-scan regression."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    assert json.loads((out / "build.json").read_text())["status"] == "PASS"
    here = Path(__file__).resolve().parent
    report = {"status": "RUNNING", "runs": [], "cleanup": [], "started_epoch": time.time()}

    def save():
        (out / "tests.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, command, env=None, expected=0):
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(command, cwd="/tmp", env=env, stdout=log, stderr=log, timeout=600)
        report["runs"].append(
            dict(
                label=label,
                argv=command,
                returncode=result.returncode,
                expected=expected,
                started_epoch=start,
                ended_epoch=time.time(),
            )
        )
        save()
        assert result.returncode == expected, (label, result.returncode)
        print(label, "PASS", flush=True)

    def cleanup(path):
        if path.exists():
            assert str(path) not in Path("/proc/mounts").read_text()
            for log in path.rglob("*.log"):
                target = out / "test-logs" / path.name / log.relative_to(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log, target)
            shutil.rmtree(path)
            report["cleanup"].append(dict(path=str(path), removed=not path.exists()))
            save()

    try:
        for name in ("blob", "blob_point", "external", "external_point"):
            component = out / name
            run(
                name + "-native",
                ["ctest", "--test-dir", str(component / "build/python-release"), "--output-on-failure"],
            )
            owned = out / ("owned-regression-" + name)
            try:
                run(
                    name + "-bounded-scan",
                    [str(component / "regression"), str(owned)],
                    expected=0 if name.endswith("_point") else 1,
                )
            finally:
                cleanup(owned)
            if not name.endswith("_point"):
                continue
            python = component / "venv/bin/python"
            env = dict(
                os.environ,
                VANE_FS_MOUNT_BINARY=str(component / "build/python-release/vane-fs-mount"),
                VANE_FS_PRODUCTION_PYTHON=str(here.parents[2] / ".venv/bin/python"),
            )
            env.pop("PYTHONPATH", None)
            run(
                name + "-identity",
                [
                    str(python),
                    "-I",
                    "-c",
                    "import vane_fs._native as n;print(n.__file__);assert "
                    + repr(str(component / "venv"))
                    + " in n.__file__",
                ],
                env,
            )
            owned = out / ("owned-python-" + name)
            try:
                run(
                    name + "-python",
                    [
                        str(python),
                        "-I",
                        "-m",
                        "pytest",
                        "--rootdir",
                        str(component),
                        "--confcutdir",
                        str(component),
                        "--basetemp",
                        str(owned),
                        "-q",
                        str(component / "tests"),
                        str(here / "test_corruption.py"),
                    ],
                    env,
                )
            finally:
                cleanup(owned)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save()


if __name__ == "__main__":
    main()
