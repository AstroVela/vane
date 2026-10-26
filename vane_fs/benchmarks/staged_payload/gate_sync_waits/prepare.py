# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build reversible SQL/sync diagnostics for the matched paced parent and local gate."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "sqlite_step_waits/prepare.py"
spec = importlib.util.spec_from_file_location("gate_sql_recorder", PARENT)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
builder, digest, save = prior.builder, prior.digest, prior.save
original_instrument, original_recorder = builder["instrument"], prior.recorder


def instrument(source, name):
    text, pairs = original_instrument(source, name)
    if name == "src/checkpoint.hpp" and "BeginWrite()" in source:
        for before, after in (
            (
                "return std::unique_lock<std::mutex>(begin_mutex);",
                "return fuse_waits::Lock(fuse_waits::BeginRestartWait, begin_mutex);",
            ),
            (
                "if (begin.owns_lock()) {",
                "if (begin.owns_lock()) {\n\t\t\t\t\t\t\t\tfuse_waits::Span gate_hold(fuse_waits::BeginRestartHold);",
            ),
        ):
            assert text.count(before) == 1
            text = text.replace(before, after)
            pairs.append((before, after, 1))
    reverse = text
    for before, after, count in reversed(pairs):
        assert reverse.count(after) == count
        reverse = reverse.replace(after, before)
    assert reverse == source
    return text, pairs


def recorder(file):
    text, pairs = original_recorder(file)
    extra = (
        [("\tSleep,\n\tCount", "\tSleep,\n\tBeginRestartWait,\n\tBeginRestartHold,\n\tCount")]
        if file.endswith(".hpp")
        else [
            (
                '"range_writeback", "sleep"};',
                '"range_writeback", "sleep", "begin_restart_wait", "begin_restart_hold"};',
            ),
            (
                "if (stage == PublicationHold || stage == CheckpointHold ||",
                "if (stage == BeginRestartHold || stage == PublicationHold || stage == CheckpointHold ||",
            ),
            ("((stage == DatabaseWait ||", "((stage == BeginRestartWait || stage == DatabaseWait ||"),
        ]
    )
    for before, after in extra:
        assert text.count(before) == 1, (file, before)
        text = text.replace(before, after)
        pairs.append((before, after))
    return text, pairs


def validate(trace):
    result = prior.validate(trace)
    assert trace["stages"][-2:] == ["begin_restart_wait", "begin_restart_hold"]
    assert len(trace["stages"]) == 31
    return result


builder.update(HERE=HERE, __file__=__file__, instrument=instrument, recorder=recorder, validate=validate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--comparison", type=Path, required=True)
    args = parser.parse_args()
    out, comparison = args.artifacts.resolve(), args.comparison.resolve()
    matched = json.loads((comparison / "comparison.json").read_text())
    frozen = json.loads((comparison / "build.json").read_text())
    assert matched["status"] == frozen["status"] == "PASS"
    assert all(digest(Path(p)) == sha for p, sha in frozen["inputs"].items())
    component = Path(matched["component"])
    original = json.loads((component / "build.json").read_text())
    adapted = json.loads(json.dumps(original))
    old_control = adapted["source_manifest"]["original"]
    adapted["source_manifest"]["original"] = matched["roots"]["baseline"]
    assert adapted["source_manifest"]["candidate"] == matched["roots"]["updated"]
    out.mkdir()
    binding = out / "binding"
    (binding / "component").mkdir(parents=True)
    (binding / "comparison").mkdir()
    save(binding / "component/build.json", adapted)
    (binding / "comparison/full").symlink_to(comparison / "full", target_is_directory=True)
    inputs = dict(frozen["inputs"])
    for p in (
        comparison / "comparison.json",
        comparison / "build.json",
        component / "build.json",
        binding / "component/build.json",
    ):
        inputs[str(p)] = digest(p)
    save(binding / "comparison/build.json", dict(status="PASS", component=str(binding / "component"), inputs=inputs))
    save(
        out / "setup.json",
        dict(
            status="PASS",
            comparison=str(comparison),
            native=str(out / "native"),
            original_control_pointer=old_control,
            actual_matched_roots=matched["roots"],
            override_only="source_manifest.original selects the already frozen paced control; ordinary source files and binaries are untouched",
            changed_native_files=matched["native_changed_files"],
            diagnostic_stages=31,
        ),
    )
    saved = sys.argv
    sys.argv = [__file__, str(out / "native"), "--comparison", str(binding / "comparison")]
    try:
        builder["main"]()
    finally:
        sys.argv = saved
    report = json.loads((out / "native/build.json").read_text())
    assert report["status"] == "PASS" and len(report["reverse_proof"]) == 12
    for p in [
        PARENT,
        *PARENT.parent.glob("*.hpp"),
        *HERE.glob("*.py"),
        *HERE.glob("*.cpp"),
        out / "setup.json",
        binding / "comparison/build.json",
        binding / "component/build.json",
    ]:
        report["inputs"][str(p)] = digest(p)
    report["matched_parent"] = str(comparison)
    report["gate_trace"] = dict(
        only_instrumentation=True,
        gate_wait_stage="begin_restart_wait",
        successful_worker_gate_hold_stage="begin_restart_hold",
        passive_unchanged=True,
        ordinary_failed_checks_retained=9,
    )
    save(out / "native/build.json", report)
    print("PASS matched sources, 12 exact reversals, 31 native stages", flush=True)


if __name__ == "__main__":
    main()
