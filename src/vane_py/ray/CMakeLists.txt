# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: Apache-2.0

# Build object library for ray Python glue
add_library(
  python_ray OBJECT
  query_lifecycle_coordinator.cpp
  ray_module.cpp
  native_fragment_executor.cpp
  result_stream_bindings.cpp
  logical_plan_bindings.cpp
  distributed_plan_bindings.cpp
  ray_binding_support.cpp
  ray_register_core.cpp
  ray_register_plans.cpp
  ray_register_runner.cpp
  ray_register_execution_tests.cpp
  ray_register_exchange_tests.cpp
  ray_register_shuffle_tests.cpp
  ray_register_recovery_tests.cpp
  ray_register_descriptor_tests.cpp
  ray_register_misc_tests.cpp
  task.cpp
  worker.cpp
  worker_manager.cpp)

target_include_directories(
  python_ray
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..
          ${CMAKE_SOURCE_DIR}/external/duckdb/third_party/yyjson/include)

target_link_libraries(python_ray PRIVATE vane_python_dependencies)

# No installation target; this is built into the vane._native module via
# $<TARGET_OBJECTS:python_ray>
