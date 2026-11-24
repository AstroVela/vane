# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("generator", ["Ninja", "Ninja Multi-Config"])
def test_external_static_extension_is_linked_but_loadable_extension_is_not(tmp_path: Path, generator):
    duckdb_source = tmp_path / "duckdb"
    duckdb_source.mkdir()
    (duckdb_source / "builtin.cpp").write_text("int builtin_value() { return 40; }\n", encoding="ascii")
    (duckdb_source / "external.cpp").write_text("int external_value() { return 2; }\n", encoding="ascii")
    (duckdb_source / "optional.cpp").write_text("int optional_value() { return 0; }\n", encoding="ascii")
    (duckdb_source / "CMakeLists.txt").write_text(
        """
add_library(builtin_extension STATIC builtin.cpp)
add_library(external_extension STATIC external.cpp)
add_library(optional_extension MODULE optional.cpp)
set(DUCKDB_EXTENSION_NAMES "builtin;external;optional")
set(DUCKDB_EXTENSION_BUILTIN_SHOULD_LINK TRUE)
set(DUCKDB_EXTENSION_EXTERNAL_SHOULD_LINK TRUE)
set(DUCKDB_EXTENSION_OPTIONAL_SHOULD_LINK FALSE)
""".lstrip(),
        encoding="ascii",
    )
    (tmp_path / "loader.cpp").write_text("int loader_value() { return 0; }\n", encoding="ascii")
    (tmp_path / "main.cpp").write_text(
        "int builtin_value(); int external_value();\n"
        "int main() { return builtin_value() + external_value() == 42 ? 0 : 1; }\n",
        encoding="ascii",
    )
    (tmp_path / "CMakeLists.txt").write_text(
        """
cmake_minimum_required(VERSION 3.29)
project(static_extension_probe LANGUAGES CXX)
set(DUCKDB_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/duckdb")
set(BUILD_EXTENSIONS builtin)
set(VANE_LOADABLE_EXTENSION_NAMES optional)
include("${VANE_TEST_REPOSITORY}/cmake/duckdb_loader.cmake")
add_subdirectory(duckdb)
add_library(duckdb_generated_extension_loader STATIC loader.cpp)
add_executable(probe main.cpp)
set_target_properties(probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/bin")
endforeach()
duckdb_link_extensions(probe)
""".lstrip(),
        encoding="ascii",
    )
    build = tmp_path / "build"
    for command in (
        (
            "cmake",
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            f"-DVANE_TEST_REPOSITORY={REPOSITORY_ROOT.as_posix()}",
            "-G",
            generator,
        ),
        ("cmake", "--build", str(build), "--config", "Release", "--target", "probe", "--parallel", "2"),
    ):
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run((str(build / "bin" / ("probe.exe" if os.name == "nt" else "probe")),), check=True)


def test_external_loadable_extension_preserves_duckdb_config(tmp_path: Path):
    duckdb_source = tmp_path / "duckdb"
    extension_config_directory = duckdb_source / ".github" / "config" / "extensions"
    extension_config_directory.mkdir(parents=True)
    (extension_config_directory / "httpfs.cmake").write_text(
        """
duckdb_extension_load(httpfs
    LOAD_TESTS
    GIT_URL https://example.invalid/duckdb-httpfs
    GIT_TAG 0123456789abcdef
    SOURCE_DIR extension-source
)
""".lstrip(),
        encoding="ascii",
    )

    probe = tmp_path / "probe.cmake"
    probe.write_text(
        """
set(PROJECT_SOURCE_DIR "${VANE_TEST_REPOSITORY}")
set(DUCKDB_SOURCE_PATH "${VANE_TEST_DUCKDB_SOURCE}")
set(BUILD_EXTENSIONS "core_functions;httpfs;json")
set(VANE_LOADABLE_EXTENSIONS "HTTPFS;tpch;httpfs")
set(DUCKDB_EXTENSION_CONFIGS "existing-config.cmake")
include("${PROJECT_SOURCE_DIR}/cmake/duckdb_loader.cmake")
_duckdb_configure_loadable_extensions()

if(NOT "${VANE_LOADABLE_EXTENSION_NAMES}" STREQUAL "httpfs;tpch")
  message(FATAL_ERROR
          "unexpected loadable extensions: ${VANE_LOADABLE_EXTENSION_NAMES}")
endif()
if(NOT "${BUILD_EXTENSIONS}" STREQUAL "core_functions;json")
  message(FATAL_ERROR "unexpected static extensions: ${BUILD_EXTENSIONS}")
endif()
list(LENGTH DUCKDB_EXTENSION_CONFIGS _config_count)
if(NOT _config_count EQUAL 2)
  message(FATAL_ERROR "unexpected extension config count: ${_config_count}")
endif()
list(GET DUCKDB_EXTENSION_CONFIGS 0 _generated_config)

set(EXTENSION_CONFIG_BASE_DIR
    "${DUCKDB_SOURCE_PATH}/.github/config/extensions")
function(duckdb_extension_load name)
  set("VANE_TEST_${name}_ARGS" "${ARGN}" PARENT_SCOPE)
endfunction()
include("${_generated_config}")

set(_expected_httpfs_args
    "LOAD_TESTS;GIT_URL;https://example.invalid/duckdb-httpfs;GIT_TAG;0123456789abcdef;SOURCE_DIR;extension-source")
if(NOT "${VANE_TEST_httpfs_ARGS}" STREQUAL "${_expected_httpfs_args}")
  message(FATAL_ERROR
          "external extension config was not preserved: ${VANE_TEST_httpfs_ARGS}")
endif()
if(NOT DEFINED DUCKDB_EXTENSION_HTTPFS_SHOULD_LINK)
  message(FATAL_ERROR "external extension link setting was not overridden")
endif()
if(DUCKDB_EXTENSION_HTTPFS_SHOULD_LINK)
  message(FATAL_ERROR "external extension remained statically linked")
endif()
if(NOT "${VANE_TEST_tpch_ARGS}" STREQUAL "DONT_LINK")
  message(FATAL_ERROR
          "in-tree extension was not registered as loadable: ${VANE_TEST_tpch_ARGS}")
endif()
""".lstrip(),
        encoding="ascii",
    )

    result = subprocess.run(
        (
            "cmake",
            f"-DVANE_TEST_REPOSITORY={REPOSITORY_ROOT}",
            f"-DVANE_TEST_DUCKDB_SOURCE={duckdb_source}",
            "-P",
            str(probe),
        ),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("build_extensions", "loadable_extensions", "expected_success"),
    (
        ("core_functions;file;json", "", True),
        ("core_functions;json", "", False),
        ("core_functions;file;json", "file", False),
    ),
    ids=("static", "omitted", "loadable"),
)
def test_python_file_api_requires_static_file_extension(
    tmp_path: Path,
    build_extensions: str,
    loadable_extensions: str,
    expected_success: bool,
):
    probe = tmp_path / "probe.cmake"
    probe.write_text(
        f"""
cmake_minimum_required(VERSION 3.29)
set(PROJECT_SOURCE_DIR "${{VANE_TEST_REPOSITORY}}")
set(DUCKDB_SOURCE_PATH "${{VANE_TEST_REPOSITORY}}/external/duckdb")
set(BUILD_EXTENSIONS "{build_extensions}")
set(VANE_LOADABLE_EXTENSIONS "{loadable_extensions}")
include("${{PROJECT_SOURCE_DIR}}/cmake/duckdb_loader.cmake")
_duckdb_configure_loadable_extensions()
duckdb_require_static_extension(file "Vane Python FILE API")
""".lstrip(),
        encoding="ascii",
    )

    result = subprocess.run(
        (
            "cmake",
            f"-DVANE_TEST_REPOSITORY={REPOSITORY_ROOT}",
            "-P",
            str(probe),
        ),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    normalized_output = " ".join(output.split())

    if expected_success:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0
        assert "Vane Python FILE API requires DuckDB extension 'file' to be statically linked" in normalized_output
