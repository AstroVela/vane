# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT
#
# Modified by Vane contributors.

add_subdirectory(aggregate)
add_subdirectory(csv_scanner)
add_subdirectory(filter)
add_subdirectory(helper)
add_subdirectory(join)
add_subdirectory(order)
add_subdirectory(persistent)
add_subdirectory(projection)
add_subdirectory(scan)
add_subdirectory(schema)
add_subdirectory(set)
add_subdirectory(exchange)

set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES}
    PARENT_SCOPE)
