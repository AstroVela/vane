# this is used for clang-tidy checks
add_library(
  python_numpy OBJECT
  type.cpp
  numpy_scan.cpp
  array_wrapper.cpp
  raw_array_wrapper.cpp
  numpy_bind.cpp
  numpy_result_conversion.cpp
  numpy_array.cpp)

target_link_libraries(python_numpy PRIVATE _duckdb_dependencies)

# numpy_array.cpp is the single TU that uses the numpy C API (PyArray_Empty), so
# it needs numpy's headers. Resolved by find_package(Python ... COMPONENTS ...
# NumPy) in the top-level CMakeLists. Scoped to this object library only.
target_include_directories(python_numpy PRIVATE ${Python_NumPy_INCLUDE_DIRS})
