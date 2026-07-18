# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT
#
# Modified by Vane contributors.

include_directories(../third_party/catch)
include_directories(../third_party/sqlite/include)
include_directories(include)
include_directories(../src)
include_directories(../src/execution/distributed)

# Allows loading tests from registered extensions
include_directories("${PROJECT_BINARY_DIR}/codegen/include/")
add_definitions(-DGENERATED_EXTENSION_HEADERS=1)
add_definitions(-DDUCKDB_MAJOR_VERSION=${DUCKDB_MAJOR_VERSION})
add_definitions(-DDUCKDB_MINOR_VERSION=${DUCKDB_MINOR_VERSION})
add_definitions(-DDUCKDB_PATCH_VERSION=${DUCKDB_PATCH_VERSION})

if(${ENABLE_UNITTEST_CPP_TESTS})
  add_subdirectory(api)
  add_subdirectory(appender)
  add_subdirectory(arrow)
  add_subdirectory(catalog)
  add_subdirectory(common)
  add_subdirectory(logging)
  add_subdirectory(memoryleak)
  add_subdirectory(parallel_csv)
  add_subdirectory(secrets)
  add_subdirectory(optimizer)
  add_subdirectory(serialize)
  add_subdirectory(sql)
  add_subdirectory(ossfuzz)
  add_subdirectory(mbedtls)
  add_subdirectory(encryption)
  add_subdirectory(execution)

  # Ensure unittest target uses the repository C++ standard.
  if(TARGET unittest)
    target_compile_features(unittest PRIVATE cxx_std_11)
  endif()
endif()
add_subdirectory(extension)
add_subdirectory(helpers)
add_subdirectory(sqlite)

if(NOT WIN32 AND NOT SUN)
  if(${ENABLE_UNITTEST_CPP_TESTS})
    add_subdirectory(persistence)
  endif()
endif()

set(UNITTEST_ROOT_DIRECTORY
    ""
    CACHE
      STRING
      "The unittest root directory (defaults to the project source directory)")
if(DEFINED UNITTEST_ROOT_DIRECTORY AND NOT UNITTEST_ROOT_DIRECTORY STREQUAL "")
  add_definitions(-DDUCKDB_ROOT_DIRECTORY="${UNITTEST_ROOT_DIRECTORY}")
else()
  add_definitions(-DDUCKDB_ROOT_DIRECTORY="${PROJECT_SOURCE_DIR}")
endif()

add_executable(unittest unittest.cpp ${ALL_OBJECT_FILES})
target_compile_features(unittest PRIVATE cxx_std_11)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(unittest PRIVATE -Wno-error=maybe-uninitialized)
endif()

target_link_libraries(unittest duckdb_static test_helpers)

link_extension_libraries(unittest "")

# Streaming AI inference tests
add_subdirectory(streaming)
