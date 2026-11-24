# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse the verified typed-sync recorder and prove exact reversal of instrumentation."""

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--recorder", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--mode", choices=("build", "run"), required=True)
    args = parser.parse_args()
    root, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    repo, out = here.parents[3], root / "trace"
    candidate = json.loads((root / "candidate.json").read_text())
    assert candidate["status"] == json.loads((root / "tests.json").read_text())["status"] == "PASS"
    originals = {"baseline": Path(candidate["parent"]) / "coalesced", "updated": root / "updated"}
    expected = {"updated": candidate["sources"]}
    expected["baseline"] = json.loads((Path(candidate["parent"]) / "candidates.json").read_text())["variants"][
        "coalesced"
    ]["sources"]
    recorder, template = args.recorder.resolve(), args.template.resolve()
    preceding = json.loads((recorder / "build.json").read_text())
    tests_path = recorder / "sync-tests.json"
    assert preceding["status"] == json.loads(tests_path.read_text())["status"] == "PASS"
    old = recorder.with_name(recorder.name + "-generator")
    for filename in ("prepare.py", "trace.hpp", "trace.cpp"):
        assert digest(old / filename) == preceding["inputs"][str(old / filename)]

    if args.mode == "build":
        generator = root / "trace-generator"
        generator.mkdir()
        source = (old / "prepare.py").read_text()
        source = replace(
            source,
            'identity = json.loads((parent / "build.json").read_text())',
            "identity = json.loads(Path(" + repr(str(root / "candidate.json")) + ").read_text())",
        )
        source = replace(
            source,
            'assert json.loads((parent / "progress-tests.json").read_text())["status"] == "PASS"',
            "assert json.loads(Path(" + repr(str(root / "tests.json")) + ').read_text())["status"] == "PASS"',
        )
        source = replace(
            source, 'for name in ("progress", "lookup", "coalesced"):', 'for name in ("baseline", "updated"):'
        )
        lines = source.splitlines(True)
        for i, line in enumerate(lines):
            if line.startswith("            component = Path("):
                lines[i] = (
                    "            component = Path(" + repr({n: str(p) for n, p in originals.items()}) + "[name])\n"
                )
            elif line.startswith("            expected = {"):
                lines[i] = "            expected = " + repr(expected) + "[name]\n"
        source = "".join(lines)
        source = replace(
            source,
            '("void BeforeWrite(sqlite3 *writer) {", "Admission"),',
            '("void BeforeWrite(sqlite3 *writer, bool external_payload = false) {" if name == "updated" else "void BeforeWrite(sqlite3 *writer) {", "Admission"),',
        )
        source = replace(source, 'if name == "coalesced":', "if True:")
        marker = '            (target / "src/checkpoint.hpp").write_text(checkpoint)'
        source = replace(
            source,
            marker,
            """            if name == "updated":
                checkpoint = replace_once(checkpoint, "void PrepareExternal() {", 'void PrepareExternal() {\\n\\t\\trequest_trace::SyncSite prepare_site("clean_prepare");')
"""
            + marker,
        )
        source = replace(
            source,
            'shutil.copy2(parent / "tests.json", out / "reused-tests.json")',
            "shutil.copy2(Path(" + repr(str(root / "tests.json")) + '), out / "reused-tests.json")',
        )
        (generator / "prepare.py").write_text(source)
        for filename in ("trace.hpp", "trace.cpp"):
            shutil.copy2(old / filename, generator / filename)
        subprocess.run(
            [
                sys.executable,
                str(generator / "prepare.py"),
                str(out),
                "--parent",
                str(template),
                "--sqlite-prefix",
                str(repo / "vane_fs/vcpkg_installed/x64-linux-release"),
                "--fuse-prefix",
                str(repo / ".cache/fuse-sdk/root/usr"),
            ],
            check=True,
        )
        build = json.loads((out / "build.json").read_text())
        build["python_modules"] = {}
        for name, component in originals.items():
            module = component / "venv/lib/python3.12/site-packages/vane_fs/_native.cpython-312-x86_64-linux-gnu.so"
            build["python_modules"][name] = dict(path=str(module.resolve()), sha256=digest(module))
            build["inputs"][str(module)] = digest(module)
        for path in (Path(__file__), root / "candidate.json", root / "tests.json", tests_path, old / "prepare.py"):
            build["inputs"][str(path)] = digest(path)
        build["baseline_role"] = "Original 5 ms CLEAN policy versus selective pre-OPEN backfill"
        (out / "build.json").write_text(json.dumps(build, indent=2) + "\n")
        shutil.copy2(tests_path, out / "reused-recorder-tests.json")

    verifier_path = here.parent / "checkpoint_request_costs/analyze.py"
    spec = importlib.util.spec_from_file_location("source_verifier", verifier_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    build = json.loads((out / "build.json").read_text())
    proof = dict(status="PASS", changed_runtime_logic=False, recorder_code_identical=True, variants={})
    for name, record in build["original_sources"].items():
        entries = {}
        for path, sha in record["sha256"].items():
            original, instrumented = Path(record["path"]) / path, out / "diagnostic" / name / path
            assert digest(original) == sha
            assert digest(instrumented) == build["source_inputs"][name].get(path, sha)
            source = instrumented.read_text()
            if path.startswith("src/"):
                source = re.sub(r"\n[\t ]*request_trace::Decision[^\n]*;", "", source)
                source = verifier.original_text(source)
            assert source.encode() == original.read_bytes(), (name, path)
            entries[path] = dict(original_sha256=sha, diagnostic_sha256=digest(instrumented))
        proof["variants"][name] = entries
    for path, sha in build["inputs"].items():
        assert digest(Path(path)) == sha
    for path, sha in build["binaries"].items():
        assert digest(out / path) == sha
    proof["inputs"] = {str(p): digest(p) for p in (Path(__file__), verifier_path, out / "build.json")}
    (out / "source-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    if args.mode == "run":
        old_runner = here.parent / "checkpoint_request_costs/run.py"
        source = old_runner.read_text()
        source = replace(
            source, "here = Path(__file__).resolve().parent", "here = Path(" + repr(str(old_runner.parent)) + ")"
        )
        source = replace(
            source,
            'baseline_role="Original partial-write control; identical baseline to checkpoint-progress ext4 FUSE rejection",',
            'baseline_role=build["baseline_role"],',
        )
        for name in ("baseline", "updated"):
            source = replace(
                source,
                'json.loads((parent / "build.json").read_text())["python_modules"]["' + name + '"]',
                'build["python_modules"]["' + name + '"]',
            )
        runner = out / "run-generator.py"
        runner.write_text(source)
        subprocess.run([sys.executable, str(runner), str(out)], check=True)


if __name__ == "__main__":
    main()
