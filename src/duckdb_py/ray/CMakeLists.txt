# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: Apache-2.0

# Build object library for ray Python glue
add_library(python_ray OBJECT ray_module.cpp task.cpp worker.cpp
                              worker_manager.cpp)

target_include_directories(
  python_ray
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
          ${CMAKE_SOURCE_DIR}/external/duckdb/third_party/yyjson/include)

target_link_libraries(python_ray PRIVATE _duckdb_dependencies)

# No installation target; this is built into the _vane_duckdb module via
# $<TARGET_OBJECTS:python_ray>
