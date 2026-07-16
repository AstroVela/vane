# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT
#
# Modified by Vane contributors.

add_library_unity(
  duckdb_operator_projection
  OBJECT
  physical_projection.cpp
  physical_tableinout_function.cpp
  physical_pivot.cpp
  physical_unnest.cpp
  physical_vllm.cpp
  physical_udf_inout.cpp)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_operator_projection>
    PARENT_SCOPE)
