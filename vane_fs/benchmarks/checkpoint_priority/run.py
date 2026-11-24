# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate scheduling, then run the frozen serial storage comparison."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--control-binary", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Resume before performance windows after validation is repaired"
    )
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    acceptance = here.parent / "partial_write/acceptance"
    python = str(repo / ".venv/bin/python")
    assert json.loads((out / "build.json").read_text())["status"] == "PASS"
    previous = None
    if args.resume:
        previous = json.loads((out / "windows.json").read_text())
        assert previous["status"] == "FAIL"
        assert json.loads((out / "tests.json").read_text())["status"] == "PASS"
        assert not any(
            (out / name).exists()
            for name in ("tmpfs-plain", "ext4-plain", "ext4-diagnostic", "control-before", "control-after")
        )
        scheduling = [row for row in previous["runs"] if "-scheduling-" in row["label"]]
        assert len(scheduling) == 6 and all(row["returncode"] == row["expected_exit"] for row in scheduling)
        shutil.copy2(out / "windows.json", out / "previous-windows.json")
        for variant, files in json.loads((out / "build.json").read_text())["source_inputs"].items():
            assert all(digest(out / variant / path) == value for path, value in files.items())
    else:
        assert not (out / "windows.json").exists()
    sources = [
        *here.glob("*.py"),
        here / "plan.json",
        here / "priority.patch",
        here / "test_priority.cpp",
        *[acceptance / name for name in ("run_fuse.py", "run_controls.py", "workload.py", "validate.py")],
        here.parent / "io_interference/monitor.py",
    ]
    frozen = out / "frozen/runners"
    for path in sources:
        target = frozen / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(here / "plan.json", out / "plan.json")
    inputs = {str(path): digest(path) for path in sources}
    report = {
        "status": "RUNNING",
        "started_epoch": time.time(),
        "runs": [],
        "inputs": inputs,
        "control_binary": str(args.control_binary.resolve()),
        "control_sha256": digest(args.control_binary),
    }
    if previous:
        report["prior_validation"] = str(out / "previous-windows.json")

    def save():
        (out / "windows.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, command, expected=0):
        row = {"label": label, "argv": command, "started_epoch": time.time(), "expected_exit": expected}
        report["runs"].append(row)
        save()
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=log, timeout=2400)
        row.update(returncode=result.returncode, ended_epoch=time.time())
        save()
        assert result.returncode == expected, label
        print(label, "PASS (expected exit", expected, ")", flush=True)

    try:
        if not previous:
            for name in ("baseline", "updated"):
                for scenario in ("success", "busy", "restart"):
                    run(
                        name + "-scheduling-" + scenario,
                        [str(out / "scheduling" / name / "priority-test"), scenario],
                        expected=1 if name == "baseline" else 0,
                    )
            run("component-validation", [python, str(acceptance / "validate.py"), str(out)])
        for label, command in (
            ("tmpfs-plain", [python, str(acceptance / "run_fuse.py"), str(out), "--medium", "tmpfs"]),
            (
                "control-before",
                [python, str(acceptance / "run_controls.py"), str(out), str(args.control_binary), "--phase", "before"],
            ),
            ("ext4-plain", [python, str(acceptance / "run_fuse.py"), str(out), "--medium", "ext4"]),
            (
                "control-after",
                [python, str(acceptance / "run_controls.py"), str(out), str(args.control_binary), "--phase", "after"],
            ),
            (
                "ext4-diagnostic",
                [python, str(acceptance / "run_fuse.py"), str(out), "--medium", "ext4", "--diagnostic"],
            ),
        ):
            assert all(digest(Path(path)) == value for path, value in inputs.items())
            print("Starting", label, flush=True)
            run(label, command)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        report["ended_epoch"] = time.time()
        save()


if __name__ == "__main__":
    main()
