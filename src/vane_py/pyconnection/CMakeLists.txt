# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

# this is used for clang-tidy checks
add_library(
  python_connection OBJECT
  bindings.cpp
  execution.cpp
  filesystem.cpp
  read_csv.cpp
  read_json.cpp
  relations.cpp
  result.cpp
  type_creation.cpp
  udf.cpp)

target_link_libraries(python_connection PRIVATE vane_python_dependencies)
