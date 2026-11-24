# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Check authoritative source invariants and summarize separate presync costs."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    out, diagnostic = args.artifacts.resolve(), args.diagnostic.resolve()
    build = json.loads((out / "build.json").read_text())
    result = dict(status="PASS", inputs={}, authoritative_bodies_unchanged={}, groups=[])
    before = Path(build["orphan_parent"]) / "updated/src/staging.hpp"
    after = out / "updated/src/staging.hpp"
    result["inputs"].update({str(p): digest(p) for p in (Path(__file__).resolve(), before, after)})
    old, new = before.read_text(), after.read_text()
    for start, end in (
        ("void BeforeFullCommit() {", "void AfterFullCommit() {"),
        ("void AfterFullCommit() {", "void Publish(sqlite3 *db) {"),
        ("void Publish(sqlite3 *db) {", "\nprivate:"),
    ):
        a = old[old.index(start) : old.index(end, old.index(start))]
        b = new[new.index(start) : new.index(end, new.index(start))]
        assert a == b, start
        result["authoritative_bodies_unchanged"][start] = True
    path = diagnostic / "maintenance-analysis.json"
    result["inputs"][str(path)] = digest(path)
    maintenance = json.loads(path.read_text())
    assert maintenance["status"] == "PASS"
    for group in maintenance["groups"]:
        if group["workload"] not in ("stream_256", "write_64", "random_write_4k", "mkdir", "rename", "rmdir"):
            continue
        result["groups"].append(group)
    for directory, role in ((out, "adoption"), (out / "causal", "causal")):
        path = directory / "summary.json"
        result["inputs"][str(path)] = digest(path)
        summary = json.loads(path.read_text())
        result[role] = dict(
            pass_all=summary["pass_all"],
            large_writes=summary["large_writes"],
            failed_checks=[c for c in summary["checks"] if not c["pass_gate"]],
        )
    (out / "presync-analysis.json").write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print("presync analysis PASS; final sync, FULL publication and CLEAN fence bodies are byte-identical")


if __name__ == "__main__":
    main()
