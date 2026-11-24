# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Keep native dependencies from mixing pinned and host OpenSSL ABIs."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _run(*arguments, **kwargs):
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=60, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.skipif(sys.platform == "win32", reason="Windows native builds already use the vcpkg toolchain")
def test_native_launcher_links_pinned_openssl_despite_a_host_root(tmp_path):
    if any(shutil.which(tool) is None for tool in ("bash", "cmake", "cc", "ninja")):
        pytest.skip("requires Bash, CMake, a C compiler and Ninja")
    checkout = tmp_path / "checkout with spaces"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    for name in ("run_native_tests.sh", "vcpkg_triplet.sh"):
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    (scripts / "resolve_duckdb_fork_version.py").write_text("print('v1.5.5-vane.fixture')\n")
    (scripts / "sync_duckdb_source_id.py").write_text("print('a' * 40)\n")
    (checkout / "DUCKDB_UPSTREAM_VERSION").write_text("v1.5.5\n")
    prefix = checkout / "vcpkg_installed/fixture-triplet"
    host = tmp_path / "host-openssl"
    for directory, version in ((prefix, "0x30603000L"), (host, "0x40100000L")):
        (directory / "include/openssl").mkdir(parents=True)
        (directory / "include/openssl/ssl.h").write_text("/* SSL fixture */\n")
        (directory / "include/openssl/opensslv.h").write_text(f"#define OPENSSL_VERSION_NUMBER {version}\n")
    (prefix / "share/arrow").mkdir(parents=True)
    (prefix / "share/arrow/ArrowConfig.cmake").write_text("# Launcher dependency preflight fixture\n")

    libraries = tmp_path / "libraries"
    libraries.mkdir()
    (libraries / "legacy.c").write_text("void *ENGINE_by_id(const char *name) { static int engine; return &engine; }\n")
    (libraries / "modern.c").write_text("int modern_ssl(void) { return 4; }\n")
    (libraries / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.29)
project(openssl_fixture C)
foreach(kind IN ITEMS pinned host)
  if(kind STREQUAL "pinned")
    set(prefix "{prefix.as_posix()}")
    set(source legacy.c)
  else()
    set(prefix "{host.as_posix()}")
    set(source modern.c)
  endif()
  foreach(library IN ITEMS crypto ssl)
    add_library(${{kind}}_${{library}} STATIC ${{source}})
    set_target_properties(${{kind}}_${{library}} PROPERTIES
      OUTPUT_NAME ${{library}} ARCHIVE_OUTPUT_DIRECTORY "${{prefix}}/lib")
  endforeach()
endforeach()
""")
    _run("cmake", "-S", str(libraries), "-B", str(libraries / "build"), "-G", "Ninja")
    _run("cmake", "--build", str(libraries / "build"))

    project = checkout / "external/duckdb"
    project.mkdir(parents=True)
    # Model a gRPC object compiled against the legacy ENGINE API, which a
    # newer host OpenSSL cannot satisfy when the final consumer is linked.
    (project / "consumer.c").write_text(
        'extern void *ENGINE_by_id(const char *name);\nint main(void) { return ENGINE_by_id("fixture") == 0; }\n'
    )
    (project / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.29)
project(openssl_consumer C)
set(OPENSSL_USE_STATIC_LIBS TRUE)
find_package(OpenSSL REQUIRED)
file(WRITE "${CMAKE_BINARY_DIR}/selected-openssl.txt"
  "${OPENSSL_CRYPTO_LIBRARY}\n${OPENSSL_SSL_LIBRARY}\n${OPENSSL_INCLUDE_DIR}\n")
add_executable(unittest consumer.c)
target_link_libraries(unittest PRIVATE OpenSSL::SSL OpenSSL::Crypto)
set_target_properties(unittest PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test")
foreach(name IN ITEMS demo batch_demo optimizer_demo)
  add_custom_target(loadable_extension_${name}_loadable_extension)
endforeach()
""")
    environment = os.environ.copy()
    environment.update(
        OPENSSL_ROOT_DIR=str(host),
        VCPKG_TARGET_TRIPLET="fixture-triplet",
        VCPKG_INSTALLED_DIR=str(checkout / "vcpkg_installed"),
        VANE_NATIVE_PYTHON=sys.executable,
        VANE_NATIVE_CMAKE_GENERATOR="Ninja",
        VANE_NATIVE_BUILD_DIR=str(checkout / "native-build"),
    )
    _run("bash", str(scripts / "run_native_tests.sh"), env=environment)
    selected = (checkout / "native-build/selected-openssl.txt").read_text().splitlines()
    assert [Path(path) for path in selected] == [
        prefix / "lib/libcrypto.a",
        prefix / "lib/libssl.a",
        prefix / "include",
    ]
