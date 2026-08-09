# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

# this is used for clang-tidy checks
add_library(python_jupyter OBJECT jupyter_progress_bar_display.cpp)

target_link_libraries(python_jupyter PRIVATE vane_python_dependencies)
