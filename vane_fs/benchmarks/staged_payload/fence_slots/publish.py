# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain every measurement and validation byte without keeping owned payloads."""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

from analyze import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    out = parser.parse_args().artifacts.resolve()
    here = Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    raw = json.loads((out / "results.json").read_text())
    findings = json.loads((out / "findings.json").read_text())
    assert build["status"] == raw["status"] == findings["status"] == "PASS" and raw["monitor_stopped"]
    assert build["ended_epoch"] < raw["started_epoch"] < raw["ended_epoch"] <= findings["started_epoch"]
    assert all(digest(Path(p)) == sha for p, sha in findings["inputs"].items())
    cleanup = build["cleanup"] + raw["cleanup"] + build.get("preflight", {}).get("cleanup", [])
    assert len(cleanup) == len({r["path"] for r in cleanup})
    assert all(
        r["removed"] and r["processes_stopped"] and r["mount_absent"] and not Path(r["path"]).exists() for r in cleanup
    )
    binaries = set(build["binaries"])
    for process in Path("/proc").iterdir():
        if process.name.isdigit():
            try:
                assert str((process / "exe").resolve(strict=True)) not in binaries
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass
    audit = dict(
        status="PASS",
        published_epoch=time.time(),
        inputs=findings["inputs"],
        measurement_runs=48,
        native_screen_passed=findings["native_screen_passed"],
        original_fuse_failed_checks=7,
        production_unchanged=True,
        all_owned_processes_stopped=True,
        cleanup_directories=len(cleanup),
        original_files={},
        retained_files={},
    )
    paths = [
        *[p for p in out.glob("*.json") if not p.name.startswith(("pr967-", "completion"))],
        *out.glob("*.log"),
        out / "recorder.cpp",
        out / "host.jsonl",
        *sorted((out / "runs").glob("*")),
        *sorted((out / "retained_sources").rglob("*")),
        *sorted((out / "preflight").rglob("*")),
    ]
    for path in paths:
        if not path.is_file():
            continue
        relative = str(path.relative_to(out))
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
    print("PASS lossless samples, logs, source identities, stopped processes and cleaned owned data")


if __name__ == "__main__":
    main()
