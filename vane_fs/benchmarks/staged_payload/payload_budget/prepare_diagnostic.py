# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build typed sync diagnostics for the orphan control and payload byte budget."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    parser.add_argument("--fuse-prefix", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    old = here.parent / "request_costs"
    typed = here.parent / "sync_costs"
    generator = args.parent.resolve() / "diagnostic-generator"
    generator.mkdir()
    source = (old / "prepare.py").read_text()
    replacements = {
        'parent / "writeback-tests.json"': 'parent / "budget-tests.json"',
        'Path(identity["parent"]) / "updated"': 'Path(identity["orphan_parent"]) / "updated"',
        'expected = identity["source_inputs"][name]': 'expected = identity["source_inputs"]["orphan-control" if name == "baseline" else name]',
    }
    for before, after in replacements.items():
        assert source.count(before) == 1
        source = source.replace(before, after)
    start = source.index('            if name == "updated":')
    end = source.index('            (target / "src/staging.hpp").write_text(stage)', start)
    source = source[:start] + source[end:]
    stage_marker = '            (target / "src/staging.hpp").write_text(stage)'
    sites = """            for signature, label in [
                ("void Load() {", '"fence_load"'),
                ("void Save(bool pending) {", 'pending ? "fence_open" : "fence_clean"'),
                ("void Recover() {", '"wal_recovery"'),
                ("void BeginBatch(sqlite3 *writer) {", '"wal_prefix"'),
                ("void BeforeFullCommit() {", '"payload_final"'),
            ]:
                stage = replace_once(stage, signature, signature + "\\n\\t\\trequest_trace::SyncSite sync_site(" + label + ");")
            if name == "updated":
                stage = replace_once(stage, "uint64_t PendingPayloadBytes() {", "uint64_t PendingPayloadBytes() {" + span("PayloadBudgetCheck"))
            payload = (target / "src/payload_file.hpp").read_text()
            payload = '#include "request_trace.hpp"\\n' + payload
            payload = replace_once(payload, "void Flush() {", 'void Flush() {\\n\\t\\trequest_trace::SyncSite sync_site("payload_flush");')
            (target / "src/payload_file.hpp").write_text(payload)
"""
    source = source.replace(stage_marker, sites + stage_marker)
    workspace_marker = '            (target / "src/workspace.cpp").write_text(source)'
    assert source.count(workspace_marker) == 1
    source = source.replace(
        workspace_marker,
        """            if name == "updated":
                signature = "\\t\\t    database.staging->PendingPayloadBytes() >= Staging::PAYLOAD_BUDGET) {"
                source = replace_once(source, signature, signature + "\\n\\t\\t\\trequest_trace::Span budget_admission(request_trace::Stage::PayloadAdmission);")
"""
        + workspace_marker,
    )
    (generator / "prepare.py").write_text(source)
    header = (typed / "trace.hpp").read_text()
    assert header.count("\tCount\n") == 1
    (generator / "trace.hpp").write_text(
        header.replace("\tCount\n", "\tPayloadBudgetCheck,\n\tPayloadAdmission,\n\tCount\n")
    )
    recorder = (typed / "trace.cpp").read_text()
    assert recorder.count('"presync"};') == 1
    (generator / "trace.cpp").write_text(
        recorder.replace('"presync"};', '"presync", "payload_budget_check", "payload_admission"};')
    )
    identity = dict(
        status="PASS",
        inputs={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (here / "prepare_diagnostic.py", old / "prepare.py", typed / "trace.cpp", typed / "trace.hpp")
        },
        adaptation="Orphan control binding, typed per-sync events, payload budget check/admission spans; unchanged runtime logic",
    )
    (generator / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    subprocess.run(
        [
            sys.executable,
            str(generator / "prepare.py"),
            str(args.output),
            "--parent",
            str(args.parent),
            "--sqlite-prefix",
            str(args.sqlite_prefix),
            "--fuse-prefix",
            str(args.fuse_prefix),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
