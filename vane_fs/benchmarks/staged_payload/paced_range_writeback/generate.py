# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Derive bounded paced claims and balanced FULL pauses from frozen format 1005."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    out, parent, here = args.artifacts.resolve(), args.parent.resolve(), Path(__file__).resolve().parent
    base = json.loads((parent / "source.json").read_text())
    build = json.loads((parent / "build.json").read_text())
    assert base["status"] == build["status"] == "PASS" and base["format"] == 1005
    inputs = base["inputs"] | base["sources"] | build["inputs"] | build["binaries"] | base["default_unchanged"]
    sites = [
        Path(base["candidate"]) / "venv/lib/python3.12/site-packages",
        Path(base["dependency_support"]) / "venv/lib/python3.12/site-packages",
        here.parents[3] / ".venv/lib/python3.12/site-packages",
    ]
    for site in sites:
        support = site / "support.pth"
        if support.is_file():
            inputs[str(support)] = digest(support)
            for line in support.read_text().splitlines():
                path = Path(line)
                if path.is_absolute() and path.is_dir() and path not in sites:
                    sites.append(path)
    for path in [
        parent / "source.json",
        parent / "build.json",
        *here.glob("*.py"),
        here / "prewriteback.hpp",
        here / "test_pacing.cpp",
    ]:
        inputs[str(path)] = digest(path)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    previous, target = Path(base["candidate"]), out / "candidate"
    out.mkdir()
    target.mkdir()
    for name in (
        "src",
        "include",
        "python",
        "tests",
        "LICENSES",
        "triplets",
        "CMakeLists.txt",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "vcpkg.json",
    ):
        path = previous / name
        if path.is_dir():
            shutil.copytree(path, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif path.is_file():
            shutil.copy2(path, target / name)
    proof = {}
    for relative in ("src/prewriteback.hpp", "src/payload_file.hpp", "src/staging.hpp", "src/workspace.cpp"):
        path = target / relative
        original = text = path.read_text()
        changes = []

        def replace(before, after, count=1):
            nonlocal text
            assert text.count(before) == count, (relative, before)
            text = text.replace(before, after)
            changes.append([before, after, count])

        if relative.endswith("prewriteback.hpp"):
            replace(text, (here / "prewriteback.hpp").read_text())
        elif relative.endswith("payload_file.hpp"):
            replace(
                "\tvoid WritebackRange(std::pair<int64_t, int64_t> range) {",
                "\t// Called under the worker's independent shared GC lease.\n\tint64_t WritebackSize() { return Size(); }\n\tvoid WritebackRange(std::pair<int64_t, int64_t> range) {",
            )
        elif relative.endswith("staging.hpp"):
            replace('#include "payload_file.hpp"', '#include "payload_file.hpp"\n#include "prewriteback.hpp"')
            replace(
                "\tvoid Publish(sqlite3 *db) {",
                "\tvoid AttachPrewriteback(Prewriteback *worker) { prewriteback = worker; }\n\tvoid Publish(sqlite3 *db) {",
            )
            replace(
                '\t\tExec(db, "PRAGMA synchronous=FULL; BEGIN IMMEDIATE");',
                '\t\tPrewriteback::Pause pause(prewriteback);\n\t\tExec(db, "PRAGMA synchronous=FULL; BEGIN IMMEDIATE");',
            )
            replace(
                '\t\t\tAfterFullCommit();\n\t\t\tExec(db, "PRAGMA synchronous=NORMAL");',
                '\t\t\tAfterFullCommit();\n\t\t\tif (prewriteback) prewriteback->Retire();\n\t\t\tExec(db, "PRAGMA synchronous=NORMAL");',
            )
            replace("\tconst int timeout;", "\tPrewriteback *prewriteback = nullptr;\n\tconst int timeout;")
        else:
            replace(
                '\n\t\t\t\tprewriteback = std::make_unique<Prewriteback>(sqlite3_db_filename(db, "main"), timeout_ms);',
                '\n\t\t\t\tprewriteback = std::make_unique<Prewriteback>(sqlite3_db_filename(db, "main"), timeout_ms);\n\t\t\t\tstaging->AttachPrewriteback(prewriteback.get());',
            )
            for indent, count in (("\t" * 3, 2), ("\t" * 2, 1)):
                replace(
                    "\n" + indent + "prewriteback.reset();",
                    "\n" + indent + "staging->AttachPrewriteback(nullptr);\n" + indent + "prewriteback.reset();",
                    count,
                )
            replace(
                "\tvoid Commit() {\n\t\t// OPEN keeps",
                "\tvoid Commit() {\n\t\tPrewriteback::Pause pause(database.prewriteback.get(), full);\n\t\t// OPEN keeps",
            )
        reverse = text
        for before, after, count in reversed(changes):
            assert reverse.count(after) == count
            reverse = reverse.replace(after, before)
        assert reverse == original
        path.write_text(text)
        proof[relative] = dict(
            original_sha256=digest(previous / relative),
            candidate_sha256=digest(path),
            exact_reverse=True,
            changes=changes,
        )
    sources = {str(p): digest(p) for p in target.rglob("*") if p.is_file()}
    for path in target.rglob("*"):
        if path.is_file() and str(path.relative_to(target)) not in proof:
            assert path.read_bytes() == (previous / path.relative_to(target)).read_bytes()
    save(
        out / "source.json",
        dict(
            status="PASS",
            created_epoch=time.time(),
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            parent=str(parent),
            original=base["original"],
            mechanism_parent=str(previous),
            candidate=str(target),
            sdk=base["sdk"],
            dependency_support=base["dependency_support"],
            dependency_site_paths=list(map(str, sites)),
            inputs=inputs,
            sources=sources,
            default_unchanged=base["default_unchanged"],
            transformations=proof,
            format=1005,
            original_format=1005,
            payload_start_bytes=64 * 1024**2,
            batch_bytes=4 * 1024**2,
            completion_delay_ms=16,
            unchanged=base["unchanged"],
            accounting=base["accounting"],
            publication_ordering_unchanged=True,
            checkpoint_source_unchanged=True,
            retained_payload_barrier_descriptor=True,
            description="At most 4 MiB per range claim, 16 ms after completion, original 64 MiB trigger, balanced per-coordinator pauses around foreground FULL and background Publish; no cancellation or wait for active calls. Original extent control and unchanged full gates.",
        ),
    )
    print("PASS isolated format-1005 candidate, four changed sources with exact reversal", flush=True)


if __name__ == "__main__":
    main()
