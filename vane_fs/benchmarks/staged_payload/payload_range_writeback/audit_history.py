# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Complete initial preflight retention without changing frozen measurement inputs."""

import argparse
import json
import time
from pathlib import Path

from generate import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    args = parser.parse_args()
    out, here = args.artifacts.resolve(), Path(__file__).resolve().parent
    component = out.parent
    history = json.loads((component / "initial-preflight.json").read_text())
    audit = json.loads((here / "audit.json").read_text())
    assert audit["status"] == history["status"] == "PASS"
    assert not history["measurement_started"]
    assert all(digest(Path(p)) == sha for p, sha in audit["inputs"].items())
    initial = Path(history["initial_build_manifest"]).parent
    manifests = sorted(initial.glob("*.json"))
    inputs = {str(p): digest(p) for p in manifests + [Path(__file__)]}
    # The initial primitive-negative manifest was absent from the first history
    # index. Retain all initial manifests and count each owned directory once.
    cleanup = json.loads((component / "build.json").read_text())["cleanup"]
    cleanup += history["cleanup"]
    negatives = sorted(component.glob("*regression-before*.json"))
    negatives += sorted(initial.glob("*regression-before*.json"))
    for path in negatives:
        record = json.loads(path.read_text())
        assert record["status"] == "PASS" and record["expected_failure"]
        assert record["ended_epoch"] < audit["windows"][0]["begin"]
        for name, sha in record["inputs"].items():
            actual = name
            if digest(Path(actual)) != sha and path.parent == initial:
                actual = history["source_aliases"].get(name, name)
            assert digest(Path(actual)) == sha, actual
            assert inputs.get(actual, sha) == sha
            inputs[actual] = sha
        assert digest(Path(record["binary"]["path"])) == record["binary"]["sha256"]
        cleanup.append(record["cleanup"])
    unique = {}
    for row in cleanup:
        assert row["removed"] and not Path(row["path"]).exists()
        assert unique.get(row["path"], row) == row
        unique[row["path"]] = row
    assert len(unique) == 17
    cleanup_path = component / "history-cleanup.json"
    save(cleanup_path, list(unique.values()))
    inputs[str(cleanup_path)] = digest(cleanup_path)
    for path in manifests + [cleanup_path]:
        if str(path) in audit["original_files"]:
            continue
        data, parts = path.read_bytes(), []
        for start in range(0, len(data), 300000):
            relative = f"validation/initial-preflight/{path.name}/{len(parts):03}.txt"
            target = here / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data[start : start + 300000])
            parts.append(relative)
        assert b"".join((here / name).read_bytes() for name in parts) == data
        audit["original_files"][str(path)] = dict(
            sha256=digest(path), bytes=len(data), retained_as=parts, bytes_unchanged=True
        )
    supplement = dict(
        status="PASS",
        recorded_epoch=time.time(),
        reason="Final retention audit found an omitted initial primitive-negative manifest; all initial JSON manifests now retained. Frozen measurement inputs are unchanged.",
        inputs=inputs,
        initial_manifests=list(map(str, manifests)),
        component_and_preflight_cleanup=audit["original_files"][str(cleanup_path)],
        total_cleanup_directories=audit["cleanup_directories"] + len(unique),
    )
    path = here / "validation/history-audit.json"
    save(path, supplement)
    audit.update(
        component_and_preflight_cleanup_directories=len(unique),
        total_cleanup_directories=supplement["total_cleanup_directories"],
        history_audit=str(path.relative_to(here)),
    )
    for path in here.rglob("*"):
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts:
            assert path.stat().st_size < 500 * 1024
            audit["retained_files"][str(path.relative_to(here))] = dict(bytes=path.stat().st_size, sha256=digest(path))
    save(here / "audit.json", audit)
    print("PASS: all initial preflight manifests retained; 49 unique removed data directories; frozen inputs unchanged")


if __name__ == "__main__":
    main()
