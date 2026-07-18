# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT

add_library_unity(
  duckdb_operator_exchange OBJECT physical_remote_exchange_sink.cpp
  physical_remote_exchange_source.cpp physical_local_exchange.cpp
  repartition.cpp)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_operator_exchange>
    PARENT_SCOPE)
