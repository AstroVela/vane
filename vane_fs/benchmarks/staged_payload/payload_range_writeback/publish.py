# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Retain trial evidence, component tests and source proofs without large binaries."""

import json
import sys
from pathlib import Path

from generate import digest, save

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parent = here.parent / "extent_fuse/publish.py"
    exec(compile(parent.read_text(), str(parent), "exec"), {"__name__": "__main__", "__file__": __file__})
    out = Path(sys.argv[1]).resolve()
    component = out.parent
    audit = json.loads((here / "audit.json").read_text())

    def retain(path, relative):
        # Lossless byte chunks permit large manifests/logs with small Git blobs.
        data = path.read_bytes()
        parts = []
        for start in range(0, len(data), 300000):
            target = here / relative / f"{len(parts):03}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data[start : start + 300000])
            parts.append(str(target.relative_to(here)))
        assert b"".join((here / name).read_bytes() for name in parts) == data
        audit["original_files"][str(path)] = dict(
            sha256=digest(path), bytes=len(data), retained_as=parts, bytes_unchanged=True
        )

    preflight = sorted(component.glob("*regression-before*.json"))
    history_path = component / "initial-preflight.json"
    history = json.loads(history_path.read_text())
    assert history["status"] == "PASS" and not history["measurement_started"]
    assert all(digest(Path(path)) == sha for path, sha in history["inputs"].items())
    for path in (
        [component / name for name in ("source.json", "build.json", "build-adapter.json")] + preflight + [history_path]
    ):
        retain(path, "validation/component/" + path.name)
    for path in sorted(component.glob("*.log")):
        retain(path, "validation/component/logs/" + path.name)
    for name in history["retained_inputs"]:
        path = Path(name)
        retain(path, "validation/initial-preflight/" + path.name)
    for path in sorted(here.rglob("*")):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            assert path.stat().st_size < 500 * 1024
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    cleanup = json.loads((component / "build.json").read_text())["cleanup"]
    cleanup += [json.loads(path.read_text())["cleanup"] for path in preflight]
    cleanup += history["cleanup"]
    assert len(cleanup) == len({row["path"] for row in cleanup})
    for row in cleanup:
        assert row["removed"] and not Path(row["path"]).exists()
    audit["component_and_preflight_cleanup_directories"] = len(cleanup)
    audit["total_cleanup_directories"] = audit["cleanup_directories"] + len(cleanup)
    audit["initial_preflight_had_no_performance_measurements"] = True
    save(here / "audit.json", audit)
    print(
        "PASS: component source proofs, related tests, failed-baseline regression and complete FUSE evidence retained"
    )
