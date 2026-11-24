# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Derive one isolated 64 MiB payload-progress policy from frozen format 1005."""

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

    def replace(before, after):
        nonlocal source
        assert source.count(before) == 1, (relative, before)
        source = source.replace(before, after)
        changes.append([before, after])

    if relative == "src/staging.hpp":
        replace(
            "\tvoid BeginBatch(sqlite3 *writer) {",
            """\t// Scheduling hint only: append distance includes alignment and abandoned
\t// rollback bytes. The shared OPEN record gives every owner the same origin.
\t// Old format-1005 OPEN records have a zero hint, causing an earlier flush.
\tuint64_t PayloadBytes() const {
\t\tif (!Pending())
\t\t\treturn 0;
\t\tauto end = std::filesystem::file_size(path + ".payload");
\t\tauto start = Decode(record.data() + 80);
\t\treturn end > start ? end - start : 0;
\t}
\tvoid BeginBatch(sqlite3 *writer) {""",
        )
        replace(
            "\t\tSync(wal.fd);\n\t\tSave(true);",
            """\t\t// The reserved eight-byte hint is covered by the existing checksum.
\t\t// Recovery ignores it; OPEN/CLEAN ordering and WAL-prefix capture do not change.
\t\tEncode(record.data() + 80, std::filesystem::file_size(path + ".payload"));
\t\tSync(wal.fd);
\t\tSave(true);""",
        )
    if relative == "src/checkpoint.hpp":
        replace(
            "\tstatic constexpr int64_t LIMIT_BYTES = 64 * 1024 * 1024;",
            "\tstatic constexpr int64_t LIMIT_BYTES = 64 * 1024 * 1024;\n"
            "\tstatic constexpr uint64_t PAYLOAD_START_BYTES = 64 * 1024 * 1024;",
        )
        replace(
            "\tvoid CheckError() {",
            """\t// Called only after a successful external COMMIT, while its publication
\t// guard is still held. Maintenance cannot turn that committed write into
\t// an API failure; latch errors for the next mutation or explicit barrier.
\tvoid OnPayloadCommit() noexcept {
\t\ttry {
\t\t\tif (staging.PayloadBytes() < PAYLOAD_START_BYTES)
\t\t\t\treturn;
\t\t\t{
\t\t\t\tstd::lock_guard<std::mutex> lock(mutex);
\t\t\t\trequested = true;
\t\t\t}
\t\t\twake.notify_one();
\t\t} catch (...) {
\t\t\tint expected = SQLITE_OK;
\t\t\terror.compare_exchange_strong(expected, SQLITE_IOERR);
\t\t}
\t}
\tvoid CheckError() {""",
        )
        replace(
            "\t\t\t\t\t\t\t\t// Preserve the existing OPEN publication budget.\n\t\t\t\t\t\t\t\tdeferred = small;",
            "\t\t\t\t\t\t\t\t// Either independent budget can publish an OPEN batch.\n"
            "\t\t\t\t\t\t\t\tdeferred = small && staging.PayloadBytes() < PAYLOAD_START_BYTES;",
        )
    if relative == "src/workspace.cpp":
        replace(
            "\t      full(write && (synchronous || database.durability == Durability::Strict)) {",
            "\t      full(write && (synchronous || database.durability == Durability::Strict)), external(external_payload) {",
        )
        replace(
            "\t\t// Strict connections retain the former 16 MiB checkpoint threshold,",
            "\t\tif (external && database.checkpointer)\n"
            "\t\t\tdatabase.checkpointer->OnPayloadCommit();\n"
            "\t\t// Strict connections retain the former 16 MiB checkpoint threshold,",
        )
        replace("\tbool restore_normal, write, full;", "\tbool restore_normal, write, full, external;")
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
        for before, after in reversed(changes):
            assert reverse.count(after) == 1
            reverse = reverse.replace(after, before)
        assert reverse == old
        path.write_text(text)
        proof[relative] = dict(
            original_sha256=digest(original / relative),
            candidate_sha256=digest(path),
            changes=changes,
            exact_reverse=True,
        )
    assert set(proof) == {"src/staging.hpp", "src/checkpoint.hpp", "src/workspace.cpp"}
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
            accounting="Shared OPEN-fence append origin; includes alignment and rollback garbage; scheduling only.",
            fence_hint="Reserved bytes 80..87, covered by existing CRC64; zero in older OPEN records is conservative.",
            unchanged=[
                "WAL start/admission thresholds",
                "FULL barriers",
                "OPEN/CLEAN recovery ordering",
                "SQLite SDK",
                "production",
            ],
        ),
    )
    print("PASS: three reversibly changed C++ files; fixed 64 MiB payload threshold")


if __name__ == "__main__":
    main()
