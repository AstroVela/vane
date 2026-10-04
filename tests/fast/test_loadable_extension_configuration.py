# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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
