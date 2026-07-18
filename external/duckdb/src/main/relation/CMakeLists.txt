# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT
#
# Modified by Vane contributors.

add_library_unity(
  duckdb_main_relation
  OBJECT
  aggregate_relation.cpp
  create_table_relation.cpp
  create_view_relation.cpp
  cross_product_relation.cpp
  delete_relation.cpp
  delim_get_relation.cpp
  distinct_relation.cpp
  explain_relation.cpp
  filter_relation.cpp
  insert_relation.cpp
  join_relation.cpp
  limit_relation.cpp
  order_relation.cpp
  repartition_relation.cpp
  local_exchange_relation.cpp
  update_relation.cpp
  projection_relation.cpp
  query_relation.cpp
  read_csv_relation.cpp
  read_json_relation.cpp
  setop_relation.cpp
  subquery_relation.cpp
  table_function_relation.cpp
  table_relation.cpp
  value_relation.cpp
  materialized_relation.cpp
  view_relation.cpp
  write_parquet_relation.cpp
  write_csv_relation.cpp
  unnest_relation.cpp)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_main_relation>
    PARENT_SCOPE)
