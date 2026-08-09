# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

# this is used for clang-tidy checks
add_library(python_type OBJECT pytype.cpp typing.cpp)

target_link_libraries(python_type PRIVATE vane_python_dependencies)
