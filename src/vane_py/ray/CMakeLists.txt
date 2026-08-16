# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: Apache-2.0

# Build object library for ray Python glue
add_library(python_ray OBJECT ray_module.cpp task.cpp worker.cpp
                              worker_manager.cpp)

target_include_directories(
  python_ray
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/..
          ${CMAKE_SOURCE_DIR}/external/duckdb/third_party/yyjson/include
          ${DUCKDB_SOURCE_PATH}/third_party/mbedtls/include)

target_link_libraries(python_ray PRIVATE vane_python_dependencies)

# No installation target; this is built into the vane._native module via
# $<TARGET_OBJECTS:python_ray>
