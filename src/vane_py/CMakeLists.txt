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
  bound_plan.cpp
  query_parameters.cpp
  dynamic_extension.cpp
  execution/fragment_plan.cpp
  execution/fragment_plan_bindings.cpp
  execution/local_query.cpp
  vane_python.cpp
  importer.cpp
  merge_relation.cpp
  ai_sql_functions.cpp
  audio_file_functions.cpp
  image_file_functions.cpp
  image_functions.cpp
  image_compute_functions.cpp
  video_file_functions.cpp
  video_clip_functions.cpp
  video_frame_functions.cpp
  video_frame_macros.cpp
  read_video_frames.cpp
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
  python_input_callback.cpp
  python_replacement_scan.cpp
  python_udf_actor_resources.cpp
  python_udf_utils.cpp
  python_udf.cpp
  udf_executor.cpp
  vllm_executor.cpp)

target_link_libraries(python_src PRIVATE vane_python_dependencies)

# The fragment wire identity includes the compiler/loader implementation as well
# as DuckDB's SourceID. Vane-only native edits must invalidate old plan bytes
# too.
set(_fragment_identity "")
foreach(_fragment_file execution/fragment_plan.hpp execution/fragment_plan.cpp
                       execution/fragment_plan_bindings.cpp)
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${_fragment_file}")
  file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/${_fragment_file}" _fragment_hash)
  string(APPEND _fragment_identity "${_fragment_hash}")
endforeach()
string(SHA256 _fragment_identity "${_fragment_identity}")
set_property(
  SOURCE execution/fragment_plan.cpp
  APPEND
  PROPERTY COMPILE_DEFINITIONS
           "VANE_FRAGMENT_BUILD_ID=\"${_fragment_identity}\"")
target_include_directories(
  python_src PRIVATE "${DUCKDB_SOURCE_PATH}/extension/file/include")
