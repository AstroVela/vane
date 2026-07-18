# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT

# Streaming tests CMakeLists.txt

# Only add streaming tests if the streaming header exists
if(EXISTS
   ${CMAKE_SOURCE_DIR}/src/include/duckdb/execution/streaming/streaming.hpp)
  # Add test executables
  add_executable(test_streaming test_streaming.cpp)
  add_executable(test_sql_functions test_sql_functions.cpp)
  add_executable(test_benchmark test_benchmark.cpp)
  add_executable(test_true_streaming test_true_streaming.cpp)
  add_executable(test_streaming_demo test_streaming_demo.cpp)

  # Link with DuckDB
  target_link_libraries(test_streaming duckdb)
  target_link_libraries(test_sql_functions duckdb)
  target_link_libraries(test_benchmark duckdb)
  target_link_libraries(test_true_streaming duckdb)
  target_link_libraries(test_streaming_demo duckdb)

  # Set include directories
  target_include_directories(test_streaming
                             PRIVATE ${CMAKE_SOURCE_DIR}/src/include)
  target_include_directories(test_sql_functions
                             PRIVATE ${CMAKE_SOURCE_DIR}/src/include)
  target_include_directories(test_benchmark
                             PRIVATE ${CMAKE_SOURCE_DIR}/src/include)
  target_include_directories(test_true_streaming
                             PRIVATE ${CMAKE_SOURCE_DIR}/src/include)
  target_include_directories(test_streaming_demo
                             PRIVATE ${CMAKE_SOURCE_DIR}/src/include)
else()
  message(
    STATUS "Skipping streaming tests because streaming header is not present")
endif()
