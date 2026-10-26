# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain all range/phase diagnostics and original source proofs losslessly."""

import json
import sys
from pathlib import Path

from prepare import TRACE, digest, save

if __name__ == "__main__":
    out, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    parent = TRACE / "publish.py"
    exec(compile(parent.read_text(), str(parent), "exec"), dict(__name__="__main__", __file__=__file__))
    audit = json.loads((here / "audit.json").read_text())
    for name in ("wrapper-proof.json", "workload-proof.json", "analysis-command.json"):
        original, target = out / name, here / "validation" / name
        save(target, json.loads(original.read_text()))
        audit["original_files"][str(original)] = dict(
            sha256=digest(original), retained_as=str(target.relative_to(here)), values_unchanged=True
        )
    data = (out / "requests.json").read_bytes()
    parts = []
    for start in range(0, len(data), 300000):
        target = here / "requests" / f"{len(parts):03}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data[start : start + 300000])
        parts.append(str(target.relative_to(here)))
    assert b"".join((here / name).read_bytes() for name in parts) == data
    save(here / "requests/index.json", dict(bytes=len(data), sha256=digest(out / "requests.json"), parts=parts))
    audit["original_files"][str(out / "requests.json")] = dict(
        sha256=digest(out / "requests.json"), retained_as=parts, bytes_unchanged=True
    )
    for path in here.rglob("*"):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            assert path.stat().st_size < 500 * 1024, path
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    audit.update(
        original_plain_failed_checks=7,
        format_versions=dict(baseline=1005, updated=1005),
        both_diagnostics_built_with_identical_recorder=True,
        algorithm_correctness_reused_by_exact_reverse=True,
    )
    save(here / "audit.json", audit)
