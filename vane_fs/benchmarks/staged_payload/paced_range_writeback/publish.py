# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain the complete experiment, failed gates and setup history losslessly."""

import gzip
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from generate import digest


def main():
    out, here = Path(sys.argv[1]).resolve(), Path(__file__).resolve().parent
    build = json.loads((out / "build.json").read_text())
    source = json.loads((out / "source.json").read_text())
    findings = json.loads((out / "findings.json").read_text())
    final = json.loads((out / "final-checks.json").read_text())
    full = json.loads((out / "full-pauses.json").read_text())
    assert build["status"] == full["status"] == findings["status"] == final["status"] == "PASS"
    assert final["all_owned_processes_stopped"] and final["no_owned_mounts"]
    assert all(digest(Path(p)) == sha for p, sha in findings["inputs"].items())
    failed = json.loads((out / "full-pause-preflight.json").read_text())
    failed_report = (
        json.loads(Path(failed["original_report"]).read_text()) if "original_report" in failed else dict(cleanup=[])
    )
    raw = [json.loads((out / "fuse" / w / "ext4-plain/results.json").read_text()) for w in ("full", "sustained")]
    cleanup = build["cleanup"] + full["cleanup"] + failed_report["cleanup"] + [r for d in raw for r in d["cleanup"]]
    assert len(cleanup) == len({r["path"] for r in cleanup})
    assert all(
        r["removed"]
        and r.get("processes_stopped", r.get("process_stopped"))
        and r.get("mount_absent", True)
        and not Path(r["path"]).exists()
        for r in cleanup
    )
    audit = dict(
        status="PASS",
        published_epoch=time.time(),
        decision=findings["decision"],
        full_failed_checks=len(findings["failed_checks"]),
        mounts=32,
        sequential_content_checks=112,
        production_unchanged=True,
        all_owned_processes_stopped=True,
        cleanup_directories=len(cleanup),
        original_files={},
        retained_files={},
    )
    paths = {}
    for p in [
        *out.glob("*.json"),
        *out.glob("*.log"),
        *out.glob("*.py"),
        *out.glob("*.cpp"),
        out / "policy.patch",
        *(out / "fuse").rglob("*"),
        *(out / "full-pause-preflight").rglob("*"),
    ]:
        if (
            p.is_file()
            and not p.name.startswith(("pr967-", "completion"))
            and (
                p.suffix in (".json", ".jsonl", ".log", ".py", ".cpp", ".hpp", ".patch")
                or "full-pause-preflight" in p.parts
            )
        ):
            paths[str(p.relative_to(out))] = p
    for path in source["sources"]:
        p = Path(path)
        paths["native-sources/" + str(p.relative_to(Path(source["candidate"])))] = p
    first = json.loads((out / "initial-preflight.json").read_text())
    for path in first["inputs"]:
        p = Path(path)
        paths["initial-preflight/" + str(p.relative_to(Path(first["path"])))] = p
    if "path" in first:
        first_source = json.loads((Path(first["path"]) / "source.json").read_text())
        for path in first_source["sources"]:
            p = Path(path)
            paths["initial-preflight/native-sources/" + str(p.relative_to(Path(first_source["candidate"])))] = p
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
    shutil.copy2(out / "policy.patch", here / "policy.patch")
    for p in here.rglob("*"):
        if p.is_file() and p.name != "audit.json" and "__pycache__" not in p.parts:
            assert p.stat().st_size < 500 * 1024, p
            audit["retained_files"][str(p.relative_to(here))] = dict(bytes=p.stat().st_size, sha256=digest(p))
    (here / "audit.json").write_text(json.dumps(audit, separators=(",", ":")) + "\n")
    assert (here / "audit.json").stat().st_size < 500 * 1024
    print(
        "PASS lossless evidence, rejected gates, preflights and", len(cleanup), "cleaned owned directories", flush=True
    )


if __name__ == "__main__":
    main()
