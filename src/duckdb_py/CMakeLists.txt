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
  duckdb_python.cpp
  importer.cpp
  ai_sql_functions.cpp
  map.cpp
  path_like.cpp
  pyconnection.cpp
  datasource_function.cpp
  pyexpression.cpp
  pyfilesystem.cpp
  pyrelation.cpp
  pyresult.cpp
  pystatement.cpp
  python_dependency.cpp
  python_import_cache.cpp
  python_replacement_scan.cpp
  python_udf_actor_resources.cpp
  python_udf_utils.cpp
  python_udf.cpp
  udf_executor.cpp
  vllm_executor.cpp)

target_link_libraries(python_src PRIVATE _duckdb_dependencies)
