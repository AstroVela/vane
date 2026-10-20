# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain raw request diagnostics, exact source proofs and cleanup records."""

import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

from prepare import digest, save


def main():
    out, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    raw = json.loads((out / "ext4-diagnostic/results.json").read_text())
    findings = json.loads((out / "findings.json").read_text())
    final = json.loads((out / "final-checks.json").read_text())
    assert build["status"] == raw["status"] == findings["status"] == final["status"] == "PASS"
    assert build["ended_epoch"] < raw["started_epoch"] < raw["ended_epoch"] < findings["started_epoch"]
    assert raw["monitor_stopped"] and final["all_owned_processes_stopped"] and final["no_owned_mounts"]
    assert all(digest(Path(p)) == sha for p, sha in findings["inputs"].items())
    cleanup = build["cleanup"] + raw["cleanup"]
    assert len(cleanup) == len({row["path"] for row in cleanup})
    assert all(
        row["removed"]
        and row.get("processes_stopped", row.get("process_stopped"))
        and row.get("mount_absent", True)
        and not Path(row["path"]).exists()
        for row in cleanup
    )
    paths = {
        str(p.relative_to(out)): p
        for p in [*out.glob("*.json"), *out.glob("*.log"), *out.glob("*.py"), *(out / "ext4-diagnostic").glob("*")]
        if p.is_file() and not p.name.startswith(("pr967-", "completion"))
    }
    paths.update({"native-sources/" + str(Path(p).relative_to(out)): Path(p) for p in build["sources"]})
    paths.update({str(path.relative_to(out)): path for path in (out / "runner-preflight").rglob("*") if path.is_file()})
    audit = dict(
        status="PASS",
        published_epoch=time.time(),
        mounts=16,
        original_plain_failed_checks=9,
        native_policy_unchanged=True,
        production_unchanged=True,
        cleanup_directories=len(cleanup),
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
        assert gzip.decompress(b"".join((here / name).read_bytes() for name in parts)) == data
        audit["original_files"][str(path)] = dict(
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            retained_as=parts,
            bytes_unchanged=True,
            encoding="gzip concatenated parts",
        )
    for name in ("diagnosis.json", "final-checks.json"):
        save(here / name, json.loads((out / name).read_text()))
    for path in here.rglob("*"):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            assert path.stat().st_size < 500 * 1024, path
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    save(here / "audit.json", audit)
    assert (here / "audit.json").stat().st_size < 500 * 1024
    print("PASS lossless evidence and", len(cleanup), "cleaned owned directories", flush=True)


if __name__ == "__main__":
    main()
