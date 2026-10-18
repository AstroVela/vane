# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain every matched FULL/read diagnostic and its reversible source proofs."""

import json
import sys
from pathlib import Path

from prepare import TRACE, digest, save

if __name__ == "__main__":
    out, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    parent = TRACE / "publish.py"
    exec(compile(parent.read_text(), str(parent), "exec"), dict(__name__="__main__", __file__=__file__))
    audit = json.loads((here / "audit.json").read_text())
    for name in ("wrapper-proof.json", "read-workload-proof.json", "analysis-command.json", "preflight.json"):
        original, target = out / name, here / "validation" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        save(target, json.loads(original.read_text()))
        assert target.stat().st_size < 500 * 1024
        audit["original_files"][str(original)] = dict(
            sha256=digest(original), retained_as=str(target.relative_to(here)), values_unchanged=True
        )
    for path in sorted(here.rglob("*")):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            assert path.stat().st_size < 500 * 1024, path
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    audit.update(
        format_versions=dict(baseline=1005, updated=1005),
        original_plain_failed_checks=4,
        both_diagnostics_built_with_identical_recorder=True,
        algorithm_correctness_reused_by_exact_reverse=True,
    )
    save(here / "audit.json", audit)
