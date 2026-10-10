# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare Linux SQLite sync controls without changing the installed SDK."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE_SHA256 = "0a409f1633283fa31a9126b11fbfd64a1991c5d30defad07e5745d4667f5e23d"
CONFIG_SHA256 = "058b15e09bc22c009800609a90aaf302ee06d013cf160cb5af08d9c692676dee"
HEADER_SHA256 = "a95def1c32bbe6007ff029a15ddb9682496a087efef761520730eb71897d0477"

SQLITE_CMAKE = """cmake_minimum_required(VERSION 3.29)
project(vane_fs_sqlite_sync_controls LANGUAGES C)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "These sync controls have only been validated on Linux")
endif()
include(CheckSymbolExists)
check_symbol_exists(fdatasync "unistd.h" HAVE_FDATASYNC)
if(NOT HAVE_FDATASYNC)
  message(FATAL_ERROR "fdatasync is required for this experiment")
endif()
foreach(variant IN ITEMS fsync fdatasync)
  add_library(${variant} STATIC sqlite3.c)
  set_target_properties(${variant} PROPERTIES POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME sqlite3
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${variant}/lib")
  target_compile_options(${variant} PRIVATE
    "-include" "${CMAKE_CURRENT_SOURCE_DIR}/${variant}/include/sqlite3-vcpkg-config.h")
  if(variant STREQUAL "fdatasync")
    target_compile_definitions(${variant} PRIVATE HAVE_FDATASYNC=1)
  else()
    target_compile_definitions(${variant} PRIVATE HAVE_FDATASYNC=0)
  endif()
endforeach()
"""

TEST_CMAKE = """
if(VANE_FS_BUILD_TESTS AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(vane_fs_sqlite_sync_test tests/test_sqlite_sync.cpp)
  target_link_libraries(vane_fs_sqlite_sync_test PRIVATE
    vane_fs unofficial::sqlite3::sqlite3)
  target_link_options(vane_fs_sqlite_sync_test PRIVATE
    "LINKER:--wrap=fsync" "LINKER:--wrap=fdatasync")
  add_test(NAME vane_fs_sqlite_sync COMMAND vane_fs_sqlite_sync_test
    "${CMAKE_CURRENT_BINARY_DIR}/owned-sqlite-sync" "@VARIANT@")
  set_tests_properties(vane_fs_sqlite_sync PROPERTIES TIMEOUT 30)
endif()
"""


def check(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Unvalidated input {path}: expected {expected}, found {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New experiment directory")
    parser.add_argument("--amalgamation", type=Path, required=True, help="Unmodified SQLite 3.53.2 sqlite3.c")
    parser.add_argument("--sqlite-prefix", type=Path, required=True, help="Pinned core-only vcpkg SQLite SDK")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This experiment requires Linux")
    if args.output.exists():
        parser.error("Output must not exist")
    check(args.amalgamation, SOURCE_SHA256)
    check(args.sqlite_prefix / "include/sqlite3-vcpkg-config.h", CONFIG_SHA256)
    check(args.sqlite_prefix / "include/sqlite3.h", HEADER_SHA256)
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here.parent / "aged_write/prepare.py"), str(args.output)], check=True)
    sqlite = args.output / "sqlite"
    sqlite.mkdir()
    shutil.copy2(args.amalgamation, sqlite / "sqlite3.c")
    (sqlite / "CMakeLists.txt").write_text(SQLITE_CMAKE)
    for variant in ("fsync", "fdatasync"):
        sdk = sqlite / variant
        shutil.copytree(args.sqlite_prefix / "include", sdk / "include")
        shutil.copytree(args.sqlite_prefix / "share/unofficial-sqlite3", sdk / "share/unofficial-sqlite3")
        target = args.output / variant
        shutil.copytree(args.output / "updated", target)
        subprocess.run(
            ["patch", "--batch", "-p1", "-i", str(here / "durability-fixture.patch")], cwd=target, check=True
        )
        shutil.copy2(here / "test_sqlite_sync.cpp", target / "tests/test_sqlite_sync.cpp")
        with (target / "CMakeLists.txt").open("a") as file:
            file.write(TEST_CMAKE.replace("@VARIANT@", variant))
        print(target)
    (args.output / "sync-inputs.json").write_text(
        json.dumps(
            {"amalgamation_sha256": SOURCE_SHA256, "config_sha256": CONFIG_SHA256, "header_sha256": HEADER_SHA256},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
