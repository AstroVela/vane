# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Derive isolated advisory payload prewriteback from the frozen extent component."""

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


def transform(relative, source):
    changes = []

    def replace(before, after, count=1):
        nonlocal source
        assert source.count(before) == count, (relative, before)
        source = source.replace(before, after)
        changes.append([before, after, count])

    if relative == "src/payload_file.hpp":
        replace(
            "\t\tWrite(data.data(), data.size(), offset);\n\t\treturn offset;",
            "\t\tWrite(data.data(), data.size(), offset);\n\t\tappended += data.size();\n\t\treturn offset;",
        )
        replace(
            "\tvoid Flush() {\n\t\tif (dirty) {",
            "\tuint64_t Appended() const { return appended; }\n\tvoid Flush(bool force = false) {\n\t\tif (dirty || force) {",
        )
        replace("\tbool dirty = false;", "\tbool dirty = false;\n\tuint64_t appended = 0;")
    if relative == "src/workspace.cpp":
        replace('#include "checkpoint.hpp"', '#include "checkpoint.hpp"\n#include "prewriteback.hpp"')
        replace(
            "\tstd::unique_ptr<Checkpointer> checkpointer;",
            "\tstd::unique_ptr<Checkpointer> checkpointer;\n\tstd::unique_ptr<Prewriteback> prewriteback;",
        )
        replace(
            "\t\t\t\tcheckpointer = std::make_unique<Checkpointer>(db, int(pages.Integer(0)), timeout_ms, *staging);",
            '\t\t\t\tprewriteback = std::make_unique<Prewriteback>(sqlite3_db_filename(db, "main"), timeout_ms);\n'
            "\t\t\t\tcheckpointer = std::make_unique<Checkpointer>(db, int(pages.Integer(0)), timeout_ms, *staging);",
        )
        replace(
            "\t\t\tcheckpointer.reset();\n\t\t\tif (db) {",
            "\t\t\tcheckpointer.reset();\n\t\t\tprewriteback.reset();\n\t\t\tif (db) {",
            2,
        )
        replace(
            "\t      full(write && (synchronous || database.durability == Durability::Strict)) {",
            "\t      full(write && (synchronous || database.durability == Durability::Strict)), external(external_payload) {",
        )
        replace(
            "\t\tif (write && database.checkpointer) {",
            "\t\tif (write && database.prewriteback)\n\t\t\tdatabase.prewriteback->CheckError();\n"
            "\t\tif (write && database.checkpointer) {",
        )
        replace(
            "\t\t// Strict connections retain the former 16 MiB checkpoint threshold,",
            "\t\tif (database.prewriteback && (full || external))\n"
            "\t\t\tdatabase.prewriteback->OnCommit(database.payload_file->Appended(), full);\n"
            "\t\t// Strict connections retain the former 16 MiB checkpoint threshold,",
        )
        replace("\tbool restore_normal, write, full;", "\tbool restore_normal, write, full, external;")
        replace(
            "\t\tcheckpointer.release();",
            "\t\tcheckpointer.release();\n\t\tif (prewriteback) {\n"
            "\t\t\tprewriteback->AbandonInherited();\n\t\t\tprewriteback.release();\n\t\t}",
        )
        replace(
            "\tif (checkpointer)\n\t\tcheckpointer->Stop();",
            "\tif (prewriteback)\n\t\tprewriteback->Stop();\n\tif (checkpointer)\n\t\tcheckpointer->Stop();",
        )
        replace(
            "\ttry {\n\t\tif (checkpointer)\n\t\t\tcheckpointer->CheckError();",
            "\ttry {\n\t\tif (prewriteback)\n\t\t\tprewriteback->CheckError();\n"
            "\t\tif (checkpointer)\n\t\t\tcheckpointer->CheckError();",
        )
        replace(
            "\t\tcheckpointer.reset();\n\t\tCheck(db, sqlite3_close(db));",
            "\t\tcheckpointer.reset();\n\t\tprewriteback.reset();\n\t\tCheck(db, sqlite3_close(db));",
        )
        replace(
            "\t\tif (checkpointer)\n\t\t\tcheckpointer->Start();\n\t\tthrow;",
            "\t\tif (checkpointer)\n\t\t\tcheckpointer->Start();\n"
            "\t\tif (prewriteback)\n\t\t\tprewriteback->Start();\n\t\tthrow;",
        )
    return source, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    out, parent, here = args.artifacts.resolve(), args.parent.resolve(), Path(__file__).resolve().parent
    repo = here.parents[3]
    base = json.loads((parent / "source.json").read_text())
    build = json.loads((parent / "build.json").read_text())
    assert base["status"] == build["status"] == "PASS"
    inputs = dict(base["inputs"], **base["sources"], **build["binaries"], **build["inputs"])
    for path in [parent / "source.json", parent / "build.json", Path(__file__)]:
        inputs[str(path)] = digest(path)
    assert all(digest(Path(p)) == sha for p, sha in inputs.items())
    original, target = Path(base["candidate"]), out / "candidate"
    out.mkdir()
    target.mkdir()
    proof = {}
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
        src = original / name
        if src.is_dir():
            shutil.copytree(src, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif src.is_file():
            shutil.copy2(src, target / name)
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(target))
        old = path.read_text()
        text, changes = transform(relative, old)
        if not changes:
            continue
        reverse = text
        for before, after, count in reversed(changes):
            assert reverse.count(after) == count
            reverse = reverse.replace(after, before)
        assert reverse == old
        path.write_text(text)
        proof[relative] = dict(
            original_sha256=digest(original / relative),
            candidate_sha256=digest(path),
            changes=changes,
            exact_reverse=True,
        )
    assert set(proof) == {"src/payload_file.hpp", "src/workspace.cpp"}
    for name in ("staging.hpp", "checkpoint.hpp"):
        assert (target / "src" / name).read_bytes() == (original / "src" / name).read_bytes()
    shutil.copy2(here / "prewriteback.hpp", target / "src/prewriteback.hpp")
    inputs[str(here / "prewriteback.hpp")] = digest(here / "prewriteback.hpp")
    save(
        out / "source.json",
        dict(
            status="PASS",
            created_epoch=time.time(),
            head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            parent=str(parent),
            original=str(original),
            candidate=str(target),
            sdk=base["sdk"],
            dependency_support=base["original"],
            inputs=inputs,
            sources={str(p): digest(p) for p in target.rglob("*") if p.is_file()},
            default_unchanged=base["default_unchanged"],
            transformations=proof,
            format=1005,
            original_format=1005,
            payload_start_bytes=64 * 1024**2,
            accounting="Connection-local appended bytes; rollback appends count conservatively, notifications follow successful commits; FULL retires requests.",
            fence_and_checkpoint_sources_unchanged=True,
            unchanged=[
                "WAL start/admission thresholds",
                "FULL barriers",
                "OPEN/CLEAN recovery ordering",
                "SQLite SDK",
                "production",
            ],
        ),
    )
    print("PASS: two reversible C++ changes plus advisory prewriteback header; fixed 64 MiB threshold")


if __name__ == "__main__":
    main()
