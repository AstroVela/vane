# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: Apache-2.0

# Build object library for vane-runners C++ translation
add_library(python_vane_runners OBJECT vane_runners.cpp)

target_include_directories(python_vane_runners
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(python_vane_runners PRIVATE vane_python_dependencies)

# No installation target; this is built into the vane._native module via
# $<TARGET_OBJECTS:python_vane_runners>
