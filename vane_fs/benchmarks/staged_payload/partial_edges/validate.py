# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run related component tests against the installed candidate and sanitizers."""

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
    repo = Path(__file__).resolve().parents[4]
    build = json.loads((out / "build.json").read_text())
    assert build["status"] == "PASS"
    report = dict(status="RUNNING", runs=[], cleanup=[])

    def save():
        (out / "tests.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, argv, env=None):
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(argv, cwd="/tmp", env=env, stdout=log, stderr=log, timeout=600)
        report["runs"].append(
            dict(label=label, argv=argv, returncode=result.returncode, started_epoch=start, ended_epoch=time.time())
        )
        save()
        result.check_returncode()
        print(label, "PASS", flush=True)

    try:
        for name in ("baseline", "updated"):
            run(
                name + "-native",
                ["ctest", "--test-dir", str(out / name / "build/python-release"), "--output-on-failure"],
            )
        target = out / "updated"
        python = target / "venv/bin/python"
        env = dict(
            os.environ,
            VANE_FS_MOUNT_BINARY=str(target / "build/python-release/vane-fs-mount"),
            VANE_FS_PRODUCTION_PYTHON=str(repo / ".venv/bin/python"),
        )
        env.pop("PYTHONPATH", None)
        run(
            "updated-module-identity",
            [
                str(python),
                "-I",
                "-c",
                "import vane_fs._native as n;print(n.__file__);assert " + repr(str(target / "venv")) + " in n.__file__",
            ],
            env,
        )
        owned = out / "owned-python"
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
                    str(owned),
                    "-q",
                    str(target / "tests"),
                ],
                env,
            )
        finally:
            if owned.exists():
                assert str(owned) not in Path("/proc/mounts").read_text()
                for log in owned.rglob("*.log"):
                    dest = out / "test-logs" / log.relative_to(owned)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(log, dest)
                shutil.rmtree(owned)
                report["cleanup"].append(dict(path=str(owned), removed=not owned.exists()))
                save()
        sanitize = target / "sanitize"
        run(
            "sanitize-configure",
            [
                "cmake",
                "-S",
                str(target),
                "-B",
                str(sanitize),
                "-G",
                "Ninja",
                "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                "-DCMAKE_PREFIX_PATH=" + build["sdk"]["sqlite"],
                "-DVANE_FS_BUILD_PYTHON=OFF",
                "-DVANE_FS_BUILD_TOOLS=OFF",
                "-DVANE_FS_BUILD_FUSE=OFF",
                "-DVANE_FS_BUILD_TESTS=ON",
                "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer",
                "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined",
                "-DVANE_FS_TEST_SQLITE_SYNC=fsync",
            ],
        )
        run("sanitize-build", ["cmake", "--build", str(sanitize), "-j", "2"])
        first = len(report["runs"])
        run(
            "sanitize-tests",
            ["ctest", "--test-dir", str(sanitize), "--output-on-failure"],
            dict(os.environ, ASAN_OPTIONS="abort_on_error=1", UBSAN_OPTIONS="halt_on_error=1"),
        )
        (out / "sanitize.json").write_text(
            json.dumps(dict(status="PASS", runs=report["runs"][first:]), indent=2) + "\n"
        )
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save()


if __name__ == "__main__":
    main()
