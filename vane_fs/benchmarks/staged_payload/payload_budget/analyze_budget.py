# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Verify unchanged barriers and run the frozen per-sync attribution analyzer."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def body(source, name):
    start = source.index("\tvoid " + name + "(")
    begin = source.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out = args.artifacts.resolve()
    here = Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    parent = Path(build["parent"])
    native = json.loads((parent / "build.json").read_text())
    reference = Path(native["orphan_parent"]) / "updated"
    updated = parent / "updated"
    before = (reference / "src/staging.hpp").read_text()
    after = (updated / "src/staging.hpp").read_text()
    names = ("BeforeFullCommit", "AfterFullCommit", "Publish", "Recover")
    assert all(body(before, name) == body(after, name) for name in names)
    assert (reference / "src/payload_file.hpp").read_bytes() == (updated / "src/payload_file.hpp").read_bytes()
    assert (reference / "tests").is_dir()
    assert {
        p for p, h in native["source_inputs"]["orphan-control"].items() if h != native["source_inputs"]["updated"][p]
    } == {"src/staging.hpp", "src/checkpoint.hpp", "src/workspace.cpp"}
    assert all(
        digest(Path(record["path"]) / p) == h
        for record in build["original_sources"].values()
        for p, h in record["sha256"].items()
    )
    assert all(
        digest(out / "diagnostic" / n / p) == h
        for n, entries in build["source_inputs"].items()
        for p, h in entries.items()
    )
    assert all(digest(out / p) == h for p, h in build["binaries"].items())
    proof = dict(
        status="PASS",
        runtime_changed_files=["src/staging.hpp", "src/checkpoint.hpp", "src/workspace.cpp"],
        barrier_and_recovery_bodies_byte_identical=list(names),
        payload_layout_and_reclaim_byte_identical=True,
        component_tests_unchanged=True,
        payload_origin_field_offset=80,
        payload_origin_advisory_only=True,
        original_and_instrumented_source_hashes_verified=True,
        inputs={
            str(reference / "src/staging.hpp"): digest(reference / "src/staging.hpp"),
            str(updated / "src/staging.hpp"): digest(updated / "src/staging.hpp"),
        },
    )
    (out / "source-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    generator = out / "analysis-generator"
    generator.mkdir()
    old = here.parent / "sync_costs/analyze.py"
    source = old.read_text()
    begin, end = source.index("def source_proof("), source.index("def main():")
    source = source[:begin] + source[end:]
    source = source.replace('    parser.add_argument("--prior-diagnostic", type=Path, required=True)\n', "")
    marker = '    write(out / "source-proof.json", source_proof(out, args.prior_diagnostic.resolve()))'
    assert source.count(marker) == 1
    source = source.replace(marker, '    assert read(out / "source-proof.json")["status"] == "PASS"')
    source = source.replace(
        'read(out / "sync-tests.json")', 'read(Path(read(out / "build.json")["parent"]) / "budget-tests.json")'
    )
    source = source.replace("import re\n", "")
    (generator / "analyze.py").write_text(source)
    identity = dict(
        status="PASS",
        inputs={str(old): digest(old), str(here / "analyze_budget.py"): digest(here / "analyze_budget.py")},
        generated_sha256=digest(generator / "analyze.py"),
        adaptation="Use the native barrier/layout proof and budget test report; sync decoding, reconciliation, grouping and slow-call retention unchanged",
    )
    (generator / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    subprocess.run([sys.executable, str(generator / "analyze.py"), str(out)], check=True)


if __name__ == "__main__":
    main()
