# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain indexed synchronization/hold traces, causal links and recorder preflight history."""

import json
import sys
from pathlib import Path

from prepare import TRACE, digest, save


def main():
    out, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    for name in ("wrapper-proof.json", "analysis-command.json"):
        path = here / "validation" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        save(path, json.loads((out / name).read_text()))
    parent = TRACE / "publish.py"
    exec(compile(parent.read_text(), str(parent), "exec"), dict(__name__="__main__", __file__=__file__))
    audit = json.loads((here / "audit.json").read_text())
    case = out / "case-final-wait.json"
    save(here / "case-final-wait.json", json.loads(case.read_text()))
    audit["inputs"][str(case)] = digest(case)
    audit["original_files"][str(case)] = dict(
        sha256=digest(case), retained_as="case-final-wait.json", values_unchanged=True
    )
    value = json.loads((out / "links.json").read_text())
    rows, names, chunk, size = value["links"], [], [], 2

    def flush():
        path = here / "links" / f"{len(names):03}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(chunk, separators=(",", ":")) + "\n")
        assert path.stat().st_size < 500 * 1024
        names.append(str(path.relative_to(here)))

    for row in rows:
        count = len(json.dumps(row, separators=(",", ":"))) + 1
        if chunk and size + count > 300000:
            flush()
            chunk, size = [], 2
        chunk.append(row)
        size += count
    if chunk:
        flush()
    assert [row for name in names for row in json.loads((here / name).read_text())] == rows
    value["links"] = names
    save(here / "links/index.json", value)
    audit["original_files"][str(out / "links.json")] = dict(
        sha256=digest(out / "links.json"),
        retained_as="links/index.json",
        values_unchanged=True,
    )
    preflight = out.parent / "payload-waits-20261018"
    record = json.loads((preflight / "preflight.json").read_text())
    assert record["status"] == "PASS" and not record["measurement_started"]
    assert all(row["removed"] and not Path(row["path"]).exists() for row in record["cleanup"])
    for path in [
        preflight / "preflight.json",
        *preflight.glob("*.frozen.txt"),
        *preflight.glob("*.log"),
        preflight / "release-trace.json",
        preflight / "asan-trace.json",
    ]:
        target = here / "validation/preflight" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        assert target.stat().st_size < 500 * 1024 and digest(target) == digest(path)
        audit["original_files"][str(path)] = dict(
            sha256=digest(path), retained_as=str(target.relative_to(here)), bytes_unchanged=True
        )
    audit.update(
        format_versions=dict(baseline=1005, updated=1005),
        original_plain_failed_checks=11,
        both_diagnostics_built_with_identical_recorder=True,
        baseline_diagnostic_reused=False,
        additional_preflight_cleanup_directories=len(record["cleanup"]),
    )
    for path in [Path(__file__), parent, out / "links.json", out / "wrapper-proof.json", out / "analysis-command.json"]:
        audit["inputs"][str(path)] = digest(path)
    for path in sorted(here.rglob("*")):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            assert path.stat().st_size < 500 * 1024, path
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    save(here / "audit.json", audit)
    print(
        "PASS: retained all write wait/holder links and preflight history; 18 diagnostic + 2 preflight directories cleaned"
    )


if __name__ == "__main__":
    main()
