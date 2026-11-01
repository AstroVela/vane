# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Both header orders compile and link with consistent class/struct tags."""

import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_distributed_callback_headers_link_in_both_include_orders(tmp_path: Path):
    distributed = '#include "duckdb/function/distributed_table_function.hpp"\n'
    function = '#include "duckdb/function/function.hpp"\n'
    signature = "void CheckWorkerBind(duckdb::table_function_create_distributed_worker_bind_t)"
    (tmp_path / "implementation.cpp").write_text(distributed + function + signature + " {}\n", encoding="ascii")
    (tmp_path / "caller.cpp").write_text(
        function + distributed + signature + ";\nint main() { CheckWorkerBind(nullptr); }\n", encoding="ascii"
    )
    (tmp_path / "CMakeLists.txt").write_text(
        """
cmake_minimum_required(VERSION 3.29)
project(distributed_callback_headers LANGUAGES CXX)
add_executable(probe implementation.cpp caller.cpp)
target_compile_features(probe PRIVATE cxx_std_20)
target_compile_definitions(probe PRIVATE DUCKDB_STATIC_BUILD)
if(MSVC)
  target_compile_options(probe PRIVATE /utf-8 /we4099)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(probe PRIVATE -Werror=mismatched-tags)
endif()
target_include_directories(probe PRIVATE
    "${VANE_TEST_REPOSITORY}/external/duckdb/src/include"
    "${VANE_TEST_REPOSITORY}/external/duckdb/third_party/fmt/include")
set_target_properties(probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/bin")
endforeach()
""".lstrip(),
        encoding="ascii",
    )
    build = tmp_path / "build"
    for command in (
        ("cmake", "-S", str(tmp_path), "-B", str(build), f"-DVANE_TEST_REPOSITORY={REPOSITORY_ROOT.as_posix()}"),
        ("cmake", "--build", str(build), "--config", "Release", "--parallel", "2"),
    ):
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(build / "bin" / ("probe.exe" if os.name == "nt" else "probe"))], check=True)
