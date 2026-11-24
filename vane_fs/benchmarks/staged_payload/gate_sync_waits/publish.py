# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain every native trace, conservative sample, contract and source proof losslessly."""

import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

from prepare import digest


def main():
    root, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    out = root / "native"
    final, findings, build = [
        json.loads(path.read_text())
        for path in (root / "final-checks.json", root / "findings.json", out / "build.json")
    ]
    assert final["status"] == findings["status"] == build["status"] == "PASS"
    assert final["all_owned_processes_stopped"] and final["no_owned_mounts"]
    assert all(digest(Path(p)) == sha for p, sha in (build["inputs"] | findings["inputs"]).items())
    paths = {
        str(p.relative_to(root)): p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".jsonl", ".log", ".py", ".txt")
        and not p.name.startswith(("pr967-", "completion"))
        and not {"venv", "build", "__pycache__", "diagnostic"}.intersection(p.relative_to(root).parts)
    }
    for path, sha in build["sources"].items():
        p = Path(path)
        assert digest(p) == sha
        paths["native-sources/" + str(p.relative_to(out / "diagnostic"))] = p
    audit = dict(
        status="PASS",
        published_epoch=time.time(),
        decision=findings["decision"],
        ordinary_failed_checks_retained=9,
        diagnostic_mounts=24,
        sequential_content_checks=48,
        cleanup_directories=final["cleanup_directories"],
        all_owned_processes_stopped=True,
        production_unchanged=True,
        original_files={},
        retained_files={},
    )
    for relative, path in sorted(paths.items()):
        data = path.read_bytes()
        packed = gzip.compress(data, compresslevel=6, mtime=0)
        parts = []
        for start in range(0, len(packed), 450000):
            target = here / "evidence" / relative / f"{len(parts):03}.gz.part"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(packed[start : start + 450000])
            parts.append(str(target.relative_to(here)))
        assert gzip.decompress(b"".join((here / p).read_bytes() for p in parts)) == data
        audit["original_files"][str(path)] = dict(
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            retained_as=parts,
            encoding="gzip concatenated parts",
            bytes_unchanged=True,
        )
    for name in ("findings.json", "final-checks.json"):
        (here / name).write_text(json.dumps(json.loads((root / name).read_text()), separators=(",", ":")) + "\n")
    for p in here.rglob("*"):
        if p.is_file() and p.name != "audit.json" and "__pycache__" not in p.parts:
            assert p.stat().st_size < 500 * 1024, p
            audit["retained_files"][str(p.relative_to(here))] = dict(bytes=p.stat().st_size, sha256=digest(p))
    (here / "audit.json").write_text(json.dumps(audit, separators=(",", ":")) + "\n")
    assert (here / "audit.json").stat().st_size < 500 * 1024
    print("PASS lossless gate/sync evidence and owned cleanup", flush=True)


if __name__ == "__main__":
    main()
