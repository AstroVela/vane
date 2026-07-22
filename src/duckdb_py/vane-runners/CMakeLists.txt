# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: Apache-2.0

# Build object library for vane-runners C++ translation
add_library(python_vane_runners OBJECT vane_runners.cpp)

target_include_directories(python_vane_runners
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(python_vane_runners PRIVATE _duckdb_dependencies)

# No installation target; this is built into the _vane_duckdb module via
# $<TARGET_OBJECTS:python_vane_runners>
