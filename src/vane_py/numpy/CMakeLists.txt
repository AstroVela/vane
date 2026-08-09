# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

# this is used for clang-tidy checks
add_library(
  python_numpy OBJECT
  type.cpp numpy_scan.cpp array_wrapper.cpp raw_array_wrapper.cpp
  numpy_bind.cpp numpy_result_conversion.cpp)

target_link_libraries(python_numpy PRIVATE vane_python_dependencies)
