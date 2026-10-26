# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Add descriptor-attributed sync recording to two frozen diagnostic variants."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    parser.add_argument("--fuse-prefix", type=Path, required=True)
    args = parser.parse_args()
    out, parent = args.output.resolve(), args.parent.resolve()
    here = Path(__file__).resolve().parent
    prior = parent / "diagnostic-generator"
    assert json.loads((parent / "presync-tests.json").read_text())["status"] == "PASS"
    out.mkdir()
    generator = out / "generator"
    generator.mkdir()
    source = (prior / "prepare.py").read_text()
    assert source.count("out.mkdir()") == 1
    marker = '            (target / "src/staging.hpp").write_text(stage)'
    assert source.count(marker) == 1
    sites = """            for signature, label in [
                ("void Load() {", '"fence_load"'),
                ("void Save(bool pending) {", 'pending ? "fence_open" : "fence_clean"'),
                ("void Recover() {", '"wal_recovery"'),
                ("void BeginBatch(sqlite3 *writer) {", '"wal_prefix"'),
                ("void BeforeFullCommit() {", '"payload_final"'),
            ]:
                stage = replace_once(stage, signature, signature + "\\n\\t\\trequest_trace::SyncSite sync_site(" + label + ");")
            if name == "updated":
                stage = replace_once(stage, "void Presync() {", 'void Presync() {\\n\\t\\trequest_trace::SyncSite sync_site("payload_presync");')
            payload = (target / "src/payload_file.hpp").read_text()
            payload = '#include "request_trace.hpp"\\n' + payload
            payload = replace_once(payload, "void Flush() {", 'void Flush() {\\n\\t\\trequest_trace::SyncSite sync_site("payload_flush");')
            (target / "src/payload_file.hpp").write_text(payload)
"""
    source = source.replace(marker, sites + marker)
    (generator / "prepare.py").write_text(source.replace("out.mkdir()", "out.mkdir(exist_ok=True)"))
    for name in ("trace.cpp", "trace.hpp"):
        shutil.copy2(here / name, generator / name)
    identity = dict(
        status="PASS",
        inputs={
            str(p): digest(p)
            for p in (here / "prepare.py", prior / "prepare.py", here / "trace.cpp", here / "trace.hpp")
        },
        adaptation="Same frozen orphan/presync logic; add per-sync descriptor classification, call-site tags, size, timings and errno",
        generated_inputs={str(p.relative_to(out)): digest(p) for p in generator.iterdir()},
        installed_production_unchanged=True,
    )
    (out / "generator-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    subprocess.run(
        [
            sys.executable,
            str(generator / "prepare.py"),
            str(out),
            "--parent",
            str(parent),
            "--sqlite-prefix",
            str(args.sqlite_prefix.resolve()),
            "--fuse-prefix",
            str(args.fuse_prefix.resolve()),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
