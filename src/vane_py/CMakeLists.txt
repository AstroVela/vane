# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

add_subdirectory(pyrelation)
add_subdirectory(pyexpression)
add_subdirectory(pybind11)
add_subdirectory(numpy)
add_subdirectory(native)
add_subdirectory(jupyter)
add_subdirectory(typing)
add_subdirectory(functional)
add_subdirectory(pyconnection)
add_subdirectory(common)
add_subdirectory(pandas)
add_subdirectory(arrow)
add_subdirectory(ray)
add_subdirectory(vane-runners)

add_library(
  python_src OBJECT
  dataframe.cpp
  dynamic_extension.cpp
  vane_python.cpp
  importer.cpp
  merge_relation.cpp
  ai_sql_functions.cpp
  image_file_functions.cpp
  path_like.cpp
  pyconnection.cpp
  datasource_function.cpp
  pyexpression.cpp
  pyfilesystem.cpp
  pyrelation.cpp
  pyresult.cpp
  pyresult_source.cpp
  pystatement.cpp
  python_dependency.cpp
  python_import_cache.cpp
  python_replacement_scan.cpp
  python_udf_actor_resources.cpp
  python_udf_utils.cpp
  python_udf.cpp
  udf_executor.cpp
  vllm_executor.cpp)

target_link_libraries(python_src PRIVATE vane_python_dependencies)
target_include_directories(
  python_src PRIVATE "${DUCKDB_SOURCE_PATH}/extension/file/include")
