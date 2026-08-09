# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

# this is used for clang-tidy checks
add_library(
  python_arrow OBJECT arrow_array_stream.cpp arrow_export_utils.cpp
                      polars_filter_pushdown.cpp pyarrow_filter_pushdown.cpp)

target_link_libraries(python_arrow PRIVATE vane_python_dependencies)
