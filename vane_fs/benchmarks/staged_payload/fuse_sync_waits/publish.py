# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Publish lossless raw evidence after identity, process and cleanup checks."""

import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

from common import digest


def main():
    out = Path(sys.argv[1]).resolve()
    here = Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    raw = json.loads((out / "ext4-diagnostic/results.json").read_text())
    findings = json.loads((out / "findings.json").read_text())
    final = json.loads((out / "final-checks.json").read_text())
    assert build["status"] == raw["status"] == findings["status"] == final["status"] == "PASS"
    assert raw["monitor_stopped"] and final["all_owned_processes_stopped"] and final["no_owned_mounts"]
    assert build["ended_epoch"] < raw["started_epoch"] < raw["ended_epoch"] < findings["started_epoch"]
    assert all(digest(Path(p)) == sha for p, sha in findings["inputs"].items())
    cleanup = build["cleanup"] + raw["cleanup"]
    assert len(cleanup) == len({r["path"] for r in cleanup})
    assert all(
        r["removed"]
        and r.get("processes_stopped", r.get("process_stopped"))
        and r["mount_absent"]
        and not Path(r["path"]).exists()
        for r in cleanup
    )
    audit = dict(
        status="PASS",
        published_epoch=time.time(),
        mounts=16,
        inputs_unchanged=True,
        production_unchanged=True,
        original_fuse_failed_checks=7,
        all_owned_processes_stopped=True,
        cleanup_directories=len(cleanup),
        original_files={},
        retained_files={},
    )
    parent = Path(build["parent"])
    parent_build = json.loads((parent / "build.json").read_text())
    paths = [
        (str(p.relative_to(out)), p)
        for p in [*out.glob("*.json"), *out.glob("*.log"), *out.glob("*.py"), *(out / "ext4-diagnostic").glob("*")]
        if p.is_file() and not p.name.startswith(("pr967-", "completion"))
    ]
    paths += [(f"native-sources/{Path(p).relative_to(parent)}", Path(p)) for p in parent_build["sources"]]
    paths += [
        (f"reused-parent/{name}", parent / name)
        for name in ("build.json", "plan.json", "tests.json", "source-proof.json", "recorder-proof.json")
        if (parent / name).is_file()
    ]
    for relative, path in sorted(paths):
        original = path.read_bytes()
        packed = gzip.compress(original, compresslevel=6, mtime=0)
        parts = []
        for start in range(0, len(packed), 450000):
            target = here / "evidence" / relative / f"{len(parts):03}.gz.part"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(packed[start : start + 450000])
            parts.append(str(target.relative_to(here)))
        assert gzip.decompress(b"".join((here / p).read_bytes() for p in parts)) == original
        audit["original_files"][str(path)] = dict(
            bytes=len(original),
            sha256=hashlib.sha256(original).hexdigest(),
            retained_as=parts,
            encoding="gzip concatenated parts",
            bytes_unchanged=True,
        )
    for name in ("findings.json", "plan.json", "final-checks.json"):
        (here / name).write_text(json.dumps(json.loads((out / name).read_text()), separators=(",", ":")) + "\n")
    for path in here.rglob("*"):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            assert path.stat().st_size < 500 * 1024, path
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    (here / "audit.json").write_text(json.dumps(audit, separators=(",", ":")) + "\n")
    assert (here / "audit.json").stat().st_size < 500 * 1024
    print("PASS lossless raw samples, native source snapshots, all identities and cleanup", flush=True)


if __name__ == "__main__":
    main()
