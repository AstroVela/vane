# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run one unchanged diagnostic window between two frozen implementations."""

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

from prepare import digest, replace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--baseline", choices=("progress", "lookup"), required=True)
    parser.add_argument("--updated", choices=("lookup", "coalesced"), required=True)
    args = parser.parse_args()
    parent = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    build = json.loads((parent / "build.json").read_text())
    assert build["status"] == "PASS"
    assert json.loads((parent / "sync-tests.json").read_text())["status"] == "PASS"
    verifier_path = here.parent / "checkpoint_request_costs/analyze.py"
    spec = importlib.util.spec_from_file_location("old_analysis", verifier_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    out = parent / (args.baseline + "-" + args.updated)
    out.mkdir()
    (out / "diagnostic").mkdir()
    report = dict(
        status="PASS",
        parent=build["parent"],
        original_sources={},
        source_inputs={},
        binaries={},
        inputs=dict(build["inputs"]),
        python_modules={},
        reused_artifacts=str(parent),
        baseline_role=args.baseline + " vs " + args.updated + "; isolated mechanism comparison",
    )
    proof = dict(status="PASS", changed_runtime_logic=False, variants={})
    for alias, name in (("baseline", args.baseline), ("updated", args.updated)):
        original = build["original_sources"][name]
        component = Path(original["path"])
        target = parent / "diagnostic" / name
        entries = {}
        for relative, expected in original["sha256"].items():
            before, after = component / relative, target / relative
            assert digest(before) == expected
            assert digest(after) == build["source_inputs"][name].get(relative, expected)
            source = after.read_text()
            if relative.startswith("src/"):
                source = re.sub(r"\n[\t ]*request_trace::Decision[^\n]*;", "", source)
                source = verifier.original_text(source)
            assert source.encode() == before.read_bytes(), (name, relative)
            entries[relative] = dict(original_sha256=expected, diagnostic_sha256=digest(after))
        proof["variants"][alias] = entries
        (out / "diagnostic" / alias).symlink_to(target, target_is_directory=True)
        (out / alias).mkdir()
        (out / alias / "venv").symlink_to(component / "venv", target_is_directory=True)
        report["original_sources"][alias] = original
        report["source_inputs"][alias] = build["source_inputs"][name]
        binary = "diagnostic/" + alias + "/build/vane-fs-mount"
        report["binaries"][binary] = build["binaries"][binary.replace(alias, name)]
        native = component / "venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so"
        report["python_modules"][alias] = dict(path=str(native.resolve()), sha256=digest(native))
        report["inputs"][str(native.resolve())] = digest(native)
    for filename in ("reused-tests.json", "sync-tests.json"):
        (out / filename).write_bytes((parent / filename).read_bytes())
        report["inputs"][str(parent / filename)] = digest(parent / filename)
    (out / "tests.json").write_text(
        json.dumps(
            dict(
                status="PASS",
                runtime_tests="Reused from unchanged source builds",
                recorder_tests="New Release and ASAN/UBSAN",
                source_reversal="New exact check",
            ),
            indent=2,
        )
        + "\n"
    )
    (out / "source-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    old = here.parent / "checkpoint_request_costs/run.py"
    runner = old.read_text()
    runner = replace(runner, "here = Path(__file__).resolve().parent", "here = Path(" + repr(str(old.parent)) + ")")
    runner = replace(
        runner,
        'baseline_role="Original partial-write control; identical baseline to checkpoint-progress ext4 FUSE rejection",',
        'baseline_role=build["baseline_role"],',
    )
    for alias in ("baseline", "updated"):
        runner = replace(
            runner,
            'json.loads((parent / "build.json").read_text())["python_modules"]["' + alias + '"]',
            'build["python_modules"]["' + alias + '"]',
        )
    generated = out / "run-generator.py"
    generated.write_text(runner)
    for path in (Path(__file__), here / "prepare.py", old, verifier_path, generated):
        report["inputs"][str(path)] = digest(path)
    (out / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    subprocess.run([sys.executable, str(generated), str(out)], check=True)


if __name__ == "__main__":
    main()
