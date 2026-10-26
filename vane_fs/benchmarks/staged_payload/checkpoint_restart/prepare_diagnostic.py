# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Build the frozen typed recorder with one optional-restart timing stage."""

import argparse
import hashlib
import json
import shutil
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
    old, typed = here.parent / "request_costs", here.parent / "sync_costs"
    generator = args.parent.resolve() / "diagnostic-generator"
    generator.mkdir()
    source = (old / "prepare.py").read_text()
    for before, after in {
        'parent / "writeback-tests.json"': 'parent / "restart-tests.json"',
        'Path(identity["parent"]) / "updated"': 'Path(identity["orphan_parent"]) / "updated"',
        'expected = identity["source_inputs"][name]': 'expected = identity["source_inputs"]["orphan-control" if name == "baseline" else name]',
        '(alias / "venv").symlink_to(parent / "updated/venv", target_is_directory=True)': '(alias / "venv").symlink_to(component / "venv", target_is_directory=True)',
    }.items():
        assert source.count(before) == 1
        source = source.replace(before, after)
    start = source.index('            if name == "updated":')
    end = source.index('            (target / "src/staging.hpp").write_text(stage)', start)
    source = source[:start] + source[end:]
    marker = '            (target / "src/staging.hpp").write_text(stage)'
    sites = """            for signature, label in [
                ("void Load() {", '"fence_load"'),
                ("void Save(bool pending) {", 'pending ? "fence_open" : "fence_clean"'),
                ("void Recover() {", '"wal_recovery"'),
                ("void BeginBatch(sqlite3 *writer) {", '"wal_prefix"'),
                ("void BeforeFullCommit() {", '"payload_final"'),
            ]:
                stage = replace_once(stage, signature, signature + "\\n\\t\\trequest_trace::SyncSite sync_site(" + label + ");")
            payload = (target / "src/payload_file.hpp").read_text()
            payload = '#include "request_trace.hpp"\\n' + payload
            payload = replace_once(payload, "void Flush() {", 'void Flush() {\\n\\t\\trequest_trace::SyncSite sync_site("payload_flush");')
            (target / "src/payload_file.hpp").write_text(payload)
"""
    source = source.replace(marker, sites + marker)
    marker = '            (target / "src/checkpoint.hpp").write_text(checkpoint)'
    extra = """            if name == "updated":
                signature = "Check(sqlite3_busy_timeout(db, 0));"
                checkpoint = replace_once(checkpoint, signature, 'request_trace::Span restart_trace(request_trace::Stage::Restart);\\n\\t\\t\\t\\t\\t\\t\\trequest_trace::SyncSite restart_site("wal_restart");\\n\\t\\t\\t\\t\\t\\t\\t' + signature)
"""
    assert source.count(marker) == 1
    source = source.replace(marker, extra + marker)
    (generator / "prepare.py").write_text(source)
    header = (typed / "trace.hpp").read_text()
    recorder = (typed / "trace.cpp").read_text()
    assert header.count("\tCount\n") == 1 and recorder.count('"presync"};') == 1
    (generator / "trace.hpp").write_text(header.replace("\tCount\n", "\tRestart,\n\tCount\n"))
    (generator / "trace.cpp").write_text(recorder.replace('"presync"};', '"presync", "restart"};'))
    shutil.copy2(typed / "test_trace.cpp", generator / "test_trace.cpp")
    shutil.copy2(typed / "validate.py", generator / "validate.py")
    identity = dict(
        status="PASS",
        adaptation="Orphan-control binding, independent GC modules, unchanged typed recorder plus restart span/site; unchanged runtime logic",
        inputs={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__).resolve(),
                old / "prepare.py",
                typed / "trace.hpp",
                typed / "trace.cpp",
                typed / "validate.py",
                typed / "test_trace.cpp",
            )
        },
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
    subprocess.run(
        [sys.executable, str(generator / "validate.py"), str(args.output), "--sqlite-prefix", str(args.sqlite_prefix)],
        check=True,
    )


if __name__ == "__main__":
    main()
