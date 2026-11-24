# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse verified diagnostic binaries to isolate the commit-wake latch."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def verify(artifact):
    build = read(artifact / "build.json")
    assert build["status"] == "PASS"
    assert all(digest(Path(p)) == h for p, h in build["inputs"].items())
    assert all(digest(artifact / p) == h for p, h in build["binaries"].items())
    for name, source in build["original_sources"].items():
        assert all(digest(Path(source["path"]) / p) == h for p, h in source["sha256"].items())
        assert all(digest(artifact / "diagnostic" / name / p) == h for p, h in build["source_inputs"][name].items())
    assert read(artifact / "sync-tests.json")["status"] == "PASS"
    return build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--preceding-diagnostic", type=Path, required=True)
    args = parser.parse_args()
    parent, preceding = args.artifacts.resolve(), args.preceding_diagnostic.resolve()
    here = Path(__file__).resolve().parent
    before, after = verify(preceding), verify(parent)
    native_before = Path(before["original_sources"]["updated"]["path"])
    native_after = Path(after["original_sources"]["updated"]["path"])
    expected_before = before["original_sources"]["updated"]["sha256"]
    expected_after = after["original_sources"]["updated"]["sha256"]
    assert expected_before.keys() == expected_after.keys()
    assert {p for p in expected_before if expected_before[p] != expected_after[p]} == {"src/checkpoint.hpp"}
    progress = here.parent / "checkpoint_progress/progress.patch"
    restored = subprocess.check_output(
        [
            "patch",
            "--quiet",
            "--reverse",
            "--fuzz=0",
            "--output=-",
            str(native_after / "src/checkpoint.hpp"),
            str(progress),
        ],
        stderr=subprocess.PIPE,
    )
    assert restored == (native_before / "src/checkpoint.hpp").read_bytes()
    old_generator = Path(before["parent"]) / "diagnostic-generator"
    new_generator = parent.with_name(parent.name + "-generator")
    for name in ("trace.hpp", "trace.cpp", "test_trace.cpp", "validate.py"):
        assert digest(old_generator / name) == digest(new_generator / name)
    out = parent / "causal"
    out.mkdir()
    (out / "diagnostic").mkdir()
    variants = {"baseline": (preceding, before, native_before), "updated": (parent, after, native_after)}
    report = dict(
        status="PASS",
        parent=after["parent"],
        baseline_role="Preceding restart control; only the warm commit-wake latch differs",
        original_sources={},
        source_inputs={},
        binaries={},
        inputs={},
        python_modules={},
        reused_artifacts={"baseline": str(preceding), "updated": str(parent)},
        native_source_difference=["src/checkpoint.hpp"],
        reverse_progress_patch_fuzz=0,
        recorder_byte_identical=True,
        no_runtime_rebuild=True,
    )
    for name, (artifact, build, component) in variants.items():
        (out / "diagnostic" / name).symlink_to(artifact / "diagnostic/updated", target_is_directory=True)
        (out / name).mkdir()
        (out / name / "venv").symlink_to(component / "venv", target_is_directory=True)
        report["original_sources"][name] = build["original_sources"]["updated"]
        report["source_inputs"][name] = build["source_inputs"]["updated"]
        binary = "diagnostic/updated/build/vane-fs-mount"
        report["binaries"][binary.replace("updated", name)] = build["binaries"][binary]
        report["inputs"].update(build["inputs"])
        native_build = read(component.parent / "build.json")
        report["python_modules"][name] = native_build["python_modules"]["updated"]
        for filename in ("build.json", "tests.json", "sync-tests.json"):
            path = artifact / filename
            report["inputs"][str(path)] = digest(path)
    for filename in ("tests.json", "sync-tests.json"):
        (out / filename).write_bytes((parent / filename).read_bytes())
    runner = (here / "run.py").read_text()
    runner = runner.replace("here = Path(__file__).resolve().parent", "here = Path(" + repr(str(here)) + ")")
    runner = runner.replace(
        'baseline_role="Original partial-write control; identical baseline to checkpoint-progress ext4 FUSE rejection",',
        'baseline_role=build["baseline_role"],',
    )
    for name in variants:
        runner = runner.replace(
            'json.loads((parent / "build.json").read_text())["python_modules"]["' + name + '"]',
            'build["python_modules"]["' + name + '"]',
        )
    assert 'baseline_role=build["baseline_role"]' in runner
    assert runner.count('build["python_modules"]') == 2
    generated = out / "run-generator.py"
    generated.write_text(runner)
    for path in (Path(__file__).resolve(), here / "run.py", progress, generated):
        report["inputs"][str(path)] = digest(path)
    write(out / "build.json", report)
    subprocess.run([sys.executable, str(generated), str(out)], check=True)
    verify(out)
    print("commit-wake causal diagnostic PASS", flush=True)


if __name__ == "__main__":
    main()
