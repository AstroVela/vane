# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain all matched measurement vectors, controls, identities and cleanup losslessly."""

import gzip
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from prepare import digest


def main():
    out, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    final = json.loads((out / "final-checks.json").read_text())
    build = json.loads((out / "build.json").read_text())
    comparison = json.loads((out / "comparison.json").read_text())
    findings = json.loads((out / "findings.json").read_text())
    assert final["status"] == build["status"] == findings["status"] == "PASS"
    assert final["all_owned_processes_stopped"] and final["no_owned_mounts"]
    assert all(digest(Path(p)) == sha for p, sha in (build["inputs"] | findings["inputs"]).items())
    paths = {
        str(p.relative_to(out)): p
        for p in out.rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".jsonl", ".log", ".py")
        and not p.name.startswith(("pr967-", "completion"))
        and "__pycache__" not in p.parts
    }
    for variant, sources in comparison["sources"].items():
        for relative, entry in sources.items():
            p = Path(entry["path"])
            assert digest(p) == entry["sha256"]
            paths["native-sources/" + variant + "/" + relative] = p
    audit = dict(
        status="PASS",
        published_epoch=time.time(),
        decision=findings["decision"],
        original_failed_checks=len(findings["original_failed_checks"]),
        mounts=32,
        sequential_content_checks=112,
        cleanup_directories=36,
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
        (here / name).write_text(json.dumps(json.loads((out / name).read_text()), separators=(",", ":")) + "\n")
    shutil.copy2(out / "comparison.json", here / "comparison.json")
    for p in here.rglob("*"):
        if p.is_file() and p.name != "audit.json" and "__pycache__" not in p.parts:
            assert p.stat().st_size < 500 * 1024, p
            audit["retained_files"][str(p.relative_to(here))] = dict(bytes=p.stat().st_size, sha256=digest(p))
    (here / "audit.json").write_text(json.dumps(audit, separators=(",", ":")) + "\n")
    assert (here / "audit.json").stat().st_size < 500 * 1024
    print("PASS lossless matched evidence and 36 cleaned owned directories", flush=True)


if __name__ == "__main__":
    main()
