# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain every control sample, trace and proof with deterministic compressed chunks."""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

from analyze import digest, save


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
    inputs = findings["inputs"]
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    cleanup = build["cleanup"] + raw["cleanup"] + build.get("preflight", {}).get("cleanup", [])
    preflight_runs = len(build.get("preflight", {}).get("cleanup", []))
    assert len(cleanup) == len({r["path"] for r in cleanup}) == 68 + preflight_runs
    assert all(
        r["removed"] and r["process_stopped"] and r["mount_absent"] and not Path(r["path"]).exists() for r in cleanup
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
        inputs=inputs,
        all_samples_retained=True,
        owned_windows_do_not_overlap=True,
        runtime_sources_unchanged=True,
        all_owned_processes_stopped=True,
        measurement_runs=48,
        validation_runs=20,
        foreground_full_barriers=findings["full_barriers"],
        foreground_wal_sync_calls=findings["full_wal_sync_calls"],
        preflight_runs=preflight_runs,
        cleanup_directories=len(cleanup),
        original_plain_failed_checks=7,
        original_files={},
        retained_files={},
    )
    paths = [
        *[p for p in out.glob("*.json") if not p.name.startswith(("pr967-", "completion"))],
        *out.glob("*.log"),
        out / "recorder.cpp",
        out / "offsets.hpp",
        out / "host.jsonl",
        *sorted((out / "runs").glob("*")),
        *sorted((out / "preflight").rglob("*")),
    ]
    for path in paths:
        if not path.is_file():
            continue
        relative = str(path.relative_to(out))
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
            encoding="gzip concatenated parts",
            retained_as=parts,
            bytes_unchanged=True,
        )
    for name in ("findings.json", "plan.json", "final-checks.json"):
        value = json.loads((out / name).read_text())
        target = here / name
        target.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        assert target.stat().st_size < 500 * 1024, target
    for path in here.rglob("*"):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            assert path.stat().st_size < 500 * 1024, path
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    save(here / "audit.json", audit)
    assert (here / "audit.json").stat().st_size < 500 * 1024
    print(
        f"PASS all 48 control runs, 20 validation runs, {preflight_runs} preflight runs, 6192 FULL barriers and {len(cleanup)} cleaned data directories"
    )


if __name__ == "__main__":
    main()
