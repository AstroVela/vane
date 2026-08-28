# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

duckdb_require_static_extension(file "Vane Python FILE API")

# this is used for clang-tidy checks
add_library(python_native OBJECT file.cpp file_reader.cpp python_objects.cpp
                                 python_conversion.cpp)

target_link_libraries(python_native PRIVATE vane_python_dependencies)
target_include_directories(
  python_native PRIVATE "${DUCKDB_SOURCE_PATH}/extension/file/include")
