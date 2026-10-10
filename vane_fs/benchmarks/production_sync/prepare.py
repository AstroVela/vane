# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare production-format SQLite sync controls without changing installed packages."""

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--amalgamation", type=Path, required=True)
    parser.add_argument("--sqlite-prefix", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("These controls require Linux")
    if args.output.exists():
        parser.error("Output must not exist")
    component = Path(__file__).resolve().parents[2]
    helper = component / "benchmarks/staged_payload/wal_sync/prepare.py"
    spec = importlib.util.spec_from_file_location("sqlite_sync_controls", helper)
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    controls.check(args.amalgamation, controls.SOURCE_SHA256)
    controls.check(args.sqlite_prefix / "include/sqlite3-vcpkg-config.h", controls.CONFIG_SHA256)
    controls.check(args.sqlite_prefix / "include/sqlite3.h", controls.HEADER_SHA256)
    output = args.output.resolve()
    output.mkdir()
    sqlite = output / "sqlite"
    sqlite.mkdir()
    shutil.copy2(args.amalgamation, sqlite / "sqlite3.c")
    (sqlite / "CMakeLists.txt").write_text(controls.SQLITE_CMAKE)
    sources = {}
    for variant in ("sdk", "fsync", "fdatasync"):
        sdk = sqlite / variant
        for name in ("include", "share/unofficial-sqlite3"):
            shutil.copytree(args.sqlite_prefix / name, sdk / name)
        if variant == "sdk":
            (sdk / "lib").mkdir()
            shutil.copy2(args.sqlite_prefix / "lib/libsqlite3.a", sdk / "lib/libsqlite3.a")
        target = output / "components" / variant
        target.mkdir(parents=True)
        for name in ("src", "include", "python", "tests", "LICENSES"):
            shutil.copytree(component / name, target / name, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("CMakeLists.txt", "pyproject.toml", "README.md", "LICENSE"):
            shutil.copy2(component / name, target / name)
        # Reuse the actual-WAL-syscall regression without applying storage patches.
        shutil.copy2(helper.with_name("test_sqlite_sync.cpp"), target / "tests/test_sqlite_sync.cpp")
        shutil.copy2(Path(__file__).with_name("seed.cpp"), target / "tests/seed_sync_compat.cpp")
        with (target / "CMakeLists.txt").open("a") as file:
            file.write(controls.TEST_CMAKE.replace("@VARIANT@", "fsync" if variant == "sdk" else variant))
            file.write("""
if(VANE_FS_BUILD_TESTS)
  add_executable(vane_fs_sync_seed tests/seed_sync_compat.cpp)
  target_link_libraries(vane_fs_sync_seed PRIVATE vane_fs)
endif()
""")
        sources[variant] = {
            str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ("src", "include", "python", "tests")
            for p in (target / directory).rglob("*")
            if p.is_file()
        }
    assert sources["sdk"] == sources["fsync"] == sources["fdatasync"]
    (output / "source-inputs.json").write_text(
        json.dumps(
            {
                "scope": "Unmodified production v2 BLOB sources; no experimental storage patches",
                "sources": sources,
                "amalgamation_sha256": controls.SOURCE_SHA256,
                "config_sha256": controls.CONFIG_SHA256,
                "header_sha256": controls.HEADER_SHA256,
                "sdk_library_sha256": hashlib.sha256(
                    (args.sqlite_prefix / "lib/libsqlite3.a").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
