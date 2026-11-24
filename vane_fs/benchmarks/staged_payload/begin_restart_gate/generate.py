# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Isolate BEGIN/RESTART coordination from the frozen paced format-1005 parent."""

import argparse
import difflib
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
    for p in [parent / "source.json", parent / "build.json", *here.glob("*.py"), *here.glob("*.cpp")]:
        inputs[str(p)] = digest(p)
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
        p = previous / name
        if p.is_dir():
            shutil.copytree(p, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif p.is_file():
            shutil.copy2(p, target / name)
    proof, patch = {}, []
    for relative in ("src/checkpoint.hpp", "src/workspace.cpp"):
        path = target / relative
        original = text = path.read_text()
        changes = []

        def replace(before, after):
            nonlocal text
            assert text.count(before) == 1, (relative, before)
            text = text.replace(before, after)
            changes.append([before, after, 1])

        if relative.endswith("checkpoint.hpp"):
            replace(
                "\tvoid BeforeWrite(sqlite3 *writer, bool external_payload = false) {",
                "\tstd::unique_lock<std::mutex> BeginWrite() {\n\t\treturn std::unique_lock<std::mutex>(begin_mutex);\n\t}\n\tvoid BeforeWrite(sqlite3 *writer, bool external_payload = false) {",
            )
            begin = text.index("\t\t\t\t\t\t\t// Finish a concurrently appended tail")
            end = text.index("\n\t\t\t\t\t\t}\n", begin)
            old = text[begin:end]
            new = (
                "\t\t\t\t\t\t\t// PASSIVE remains concurrent. Only RESTART competes with BEGIN\n"
                "\t\t\t\t\t\t\t// for SQLite's writer lock. Never wait for the begin gate while\n"
                "\t\t\t\t\t\t\t// holding a checkpoint lease: an external writer can need it.\n"
                "\t\t\t\t\t\t\tstd::unique_lock<std::mutex> begin(begin_mutex, std::try_to_lock);\n"
                "\t\t\t\t\t\t\tif (begin.owns_lock()) {\n"
                + "\n".join("\t" + line for line in old.splitlines())
                + "\n\t\t\t\t\t\t\t}"
            )
            replace(old, new)
            replace("\tstd::mutex checkpoint_mutex;", "\tstd::mutex checkpoint_mutex;\n\tstd::mutex begin_mutex;")
        else:
            replace(
                "\t\t// Inline NORMAL transactions cannot introduce unpublished external bytes.",
                "\t\t// Acquire before publication/payload guards, and release immediately\n"
                "\t\t// after BEGIN, before external writes can request a checkpoint lease.\n"
                "\t\tstd::unique_lock<std::mutex> begin;\n"
                "\t\tif (write && database.checkpointer)\n"
                "\t\t\tbegin = database.checkpointer->BeginWrite();\n"
                "\t\t// Inline NORMAL transactions cannot introduce unpublished external bytes.",
            )
            replace(
                '\t\tControl(database.db, write ? "BEGIN IMMEDIATE" : "BEGIN");',
                '\t\tControl(database.db, write ? "BEGIN IMMEDIATE" : "BEGIN");\n\t\tif (begin.owns_lock())\n\t\t\tbegin.unlock();',
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
        patch.extend(
            difflib.unified_diff(
                original.splitlines(True),
                text.splitlines(True),
                fromfile="paced-parent/" + relative,
                tofile="begin-gate/" + relative,
            )
        )
    for p in target.rglob("*"):
        if p.is_file() and str(p.relative_to(target)) not in proof:
            assert p.read_bytes() == (previous / p.relative_to(target)).read_bytes()
    (out / "policy.patch").write_text("".join(patch))
    source = base | dict(
        status="PASS",
        created_epoch=time.time(),
        head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        parent=str(parent),
        mechanism_parent=str(previous),
        candidate=str(target),
        inputs=inputs,
        sources={str(p): digest(p) for p in target.rglob("*") if p.is_file()},
        transformations=proof,
        checkpoint_source_unchanged=False,
        passive_concurrency_unchanged=True,
        description="Paced format 1005 plus a local gate only around foreground BEGIN and opportunistic background RESTART; PASSIVE, publication, durability, SDK and original extent control/22 gates unchanged.",
    )
    save(out / "source.json", source)
    print("PASS isolated BEGIN/RESTART candidate, two exact-reversible source changes", flush=True)


if __name__ == "__main__":
    main()
