# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.29)
project(
  vane_fs
  VERSION 0.1.0
  LANGUAGES CXX)

option(VANE_FS_BUILD_PYTHON "Build the Python bindings" ON)
option(VANE_FS_BUILD_TESTS "Build the native tests" OFF)
option(VANE_FS_BUILD_TOOLS "Build the native command-line tools" OFF)
option(VANE_FS_BUILD_FUSE "Build the Linux libfuse3 mount adapter" OFF)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_library(vane_fs STATIC src/workspace.cpp)
target_include_directories(vane_fs PUBLIC include)
target_compile_features(vane_fs PUBLIC cxx_std_17)
target_link_libraries(vane_fs PRIVATE unofficial::sqlite3::sqlite3
                                      Threads::Threads)
set_target_properties(vane_fs PROPERTIES POSITION_INDEPENDENT_CODE ON
                                         CXX_VISIBILITY_PRESET hidden)

if(VANE_FS_BUILD_TOOLS)
  add_executable(vane-fs src/cli.cpp)
  target_link_libraries(vane-fs PRIVATE vane_fs)
  add_executable(vane-fs-benchmark src/benchmark.cpp)
  target_link_libraries(
    vane-fs-benchmark PRIVATE vane_fs unofficial::sqlite3::sqlite3
                              Threads::Threads)
  target_compile_definitions(
    vane-fs-benchmark
    PRIVATE
      VANE_FS_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
      VANE_FS_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
endif()

if(VANE_FS_BUILD_FUSE)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "The FUSE adapter currently supports Linux only")
  endif()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FUSE3 REQUIRED IMPORTED_TARGET fuse3>=3.10)
  add_executable(vane-fs-mount src/fuse.cpp)
  target_link_libraries(vane-fs-mount PRIVATE vane_fs PkgConfig::FUSE3)
endif()

if(VANE_FS_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native src/python.cpp src/native_reader.cpp)
  target_link_libraries(_native PRIVATE vane_fs)
  target_compile_definitions(_native PRIVATE PYBIND11_COMPILER_TYPE="vane_fs")
  if(UNIX AND NOT APPLE)
    target_link_options(_native PRIVATE "LINKER:--exclude-libs,ALL")
  endif()
  install(
    TARGETS _native
    LIBRARY DESTINATION vane_fs
    RUNTIME DESTINATION vane_fs)
endif()

if(VANE_FS_BUILD_TESTS)
  enable_testing()
  add_executable(vane_fs_test tests/test_core.cpp)
  target_link_libraries(
    vane_fs_test PRIVATE vane_fs unofficial::sqlite3::sqlite3 Threads::Threads)
  add_test(NAME vane_fs_core COMMAND vane_fs_test)
  set_tests_properties(vane_fs_core PROPERTIES TIMEOUT 60)
  add_executable(vane_fs_inode_refs_test tests/test_inode_refs.cpp)
  target_link_libraries(vane_fs_inode_refs_test
                        PRIVATE vane_fs unofficial::sqlite3::sqlite3)
  add_test(NAME vane_fs_inode_refs COMMAND vane_fs_inode_refs_test)
  set_tests_properties(vane_fs_inode_refs PROPERTIES TIMEOUT 60)
  if(UNIX)
    add_executable(vane_fs_durability_test tests/test_durability.cpp)
    target_link_libraries(vane_fs_durability_test
                          PRIVATE vane_fs unofficial::sqlite3::sqlite3)
    add_test(NAME vane_fs_durability COMMAND vane_fs_durability_test)
    set_tests_properties(vane_fs_durability PROPERTIES TIMEOUT 60)
    add_executable(vane_fs_crash_test tests/test_crash.cpp)
    target_link_libraries(vane_fs_crash_test
                          PRIVATE vane_fs unofficial::sqlite3::sqlite3)
    add_test(NAME vane_fs_crash COMMAND vane_fs_crash_test)
    set_tests_properties(vane_fs_crash PROPERTIES TIMEOUT 60)
  endif()
endif()
