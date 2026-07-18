# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT
#
# Modified by Vane contributors.

add_subdirectory(expression_executor)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/nested_loop_join)
  add_subdirectory(nested_loop_join)
endif()
add_subdirectory(operator)
add_subdirectory(physical_plan)
add_subdirectory(index)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/sample)
  add_subdirectory(sample)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/streaming)
  add_subdirectory(streaming)
endif()
option(BUILD_DISTRIBUTED "Build the distributed execution module" ON)
if(BUILD_DISTRIBUTED)
  add_subdirectory(distributed)
  if(DUCKDB_DISTRIBUTED_LINK_LIBS)
    set(DUCKDB_DISTRIBUTED_LINK_LIBS
        ${DUCKDB_DISTRIBUTED_LINK_LIBS}
        PARENT_SCOPE)
  endif()
endif()

add_library_unity(
  duckdb_execution
  OBJECT
  adaptive_filter.cpp
  aggregate_hashtable.cpp
  base_aggregate_hashtable.cpp
  column_binding_resolver.cpp
  expression_executor.cpp
  expression_executor_state.cpp
  external_block.cpp
  join_hashtable.cpp
  perfect_aggregate_hashtable.cpp
  physical_operator.cpp
  physical_plan_generator.cpp
  physical_plan_serialization.cpp
  physical_operator_visitor.cpp
  radix_partitioned_hashtable.cpp
  udf_executor.cpp
  vllm_executor.cpp)

# Include duckdb_execution and optionally the distributed execution module
# object files
if(BUILD_DISTRIBUTED)
  set(ALL_OBJECT_FILES
      ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_execution>
      $<TARGET_OBJECTS:duckdb_execution_distributed>
      PARENT_SCOPE)
else()
  set(ALL_OBJECT_FILES
      ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_execution>
      PARENT_SCOPE)
endif()
