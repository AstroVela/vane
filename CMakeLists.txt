# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

cmake_minimum_required(VERSION 3.29)

project(duckdb_py LANGUAGES CXX)

# Arrow 24 requires C++20. Use the same standard for the complete Vane native
# build so Python bindings, distributed exchange, and their consumers cannot
# silently diverge based on transitive CMake requirements.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Vane does not use C++ modules. CMake's automatic C++20 module scan emits GCC
# dependency flags that are not understood by the ccache version in
# manylinux_2_28, so keep the scan disabled for all targets.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Set the library name
set(DUCKDB_PYTHON_LIB_NAME "_vane_duckdb")

# Detect CCache
include(cmake/compiler_launcher.cmake)
setup_compiler_launcher_if_available()

# ────────────────────────────────────────────
# IDE support
# ────────────────────────────────────────────
# Create compile_commands.json for IntelliSense and clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ────────────────────────────────────────────
# Policy hygiene
# ────────────────────────────────────────────
if(POLICY CMP0148) # Disallow FindPythonLibs
  cmake_policy(SET CMP0148 NEW)
endif()

if(POLICY CMP0003) # No implicit link directories
  cmake_policy(SET CMP0003 NEW)
endif()

if(POLICY CMP0169) # DuckDB's external extension loader still uses
                   # FetchContent_Populate.
  set(CMAKE_POLICY_DEFAULT_CMP0169 OLD)
endif()

# ────────────────────────────────────────────
# Dependencies
# ────────────────────────────────────────────
# Prefer repo-local vcpkg packages when present. This keeps local PEP 517 builds
# working after `vcpkg install` without requiring callers to export
# CMAKE_PREFIX_PATH before running `pip install .`.
set(_VANE_VCPKG_INSTALLED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg_installed")
if(EXISTS "${_VANE_VCPKG_INSTALLED_DIR}")
  set(_VANE_VCPKG_PREFIX_CANDIDATES)

  if(DEFINED VCPKG_TARGET_TRIPLET)
    list(APPEND _VANE_VCPKG_PREFIX_CANDIDATES
         "${_VANE_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
  endif()

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
      list(APPEND _VANE_VCPKG_PREFIX_CANDIDATES
           "${_VANE_VCPKG_INSTALLED_DIR}/x64-linux")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
      list(APPEND _VANE_VCPKG_PREFIX_CANDIDATES
           "${_VANE_VCPKG_INSTALLED_DIR}/arm64-linux")
    endif()
  elseif(APPLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
      list(APPEND _VANE_VCPKG_PREFIX_CANDIDATES
           "${_VANE_VCPKG_INSTALLED_DIR}/arm64-osx")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
      list(APPEND _VANE_VCPKG_PREFIX_CANDIDATES
           "${_VANE_VCPKG_INSTALLED_DIR}/x64-osx")
    endif()
  elseif(WIN32)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$")
      list(APPEND _VANE_VCPKG_PREFIX_CANDIDATES
           "${_VANE_VCPKG_INSTALLED_DIR}/arm64-windows")
    else()
      list(APPEND _VANE_VCPKG_PREFIX_CANDIDATES
           "${_VANE_VCPKG_INSTALLED_DIR}/x64-windows")
    endif()
  endif()

  file(
    GLOB _VANE_VCPKG_INSTALLED_PREFIXES
    LIST_DIRECTORIES true
    "${_VANE_VCPKG_INSTALLED_DIR}/*")
  list(APPEND _VANE_VCPKG_PREFIX_CANDIDATES ${_VANE_VCPKG_INSTALLED_PREFIXES})
  list(REMOVE_DUPLICATES _VANE_VCPKG_PREFIX_CANDIDATES)

  foreach(_VANE_VCPKG_PREFIX IN LISTS _VANE_VCPKG_PREFIX_CANDIDATES)
    if(EXISTS "${_VANE_VCPKG_PREFIX}/share/arrow/ArrowConfig.cmake")
      list(FIND CMAKE_PREFIX_PATH "${_VANE_VCPKG_PREFIX}" _VANE_PREFIX_INDEX)
      if(_VANE_PREFIX_INDEX EQUAL -1)
        list(APPEND CMAKE_PREFIX_PATH "${_VANE_VCPKG_PREFIX}")
      endif()
      set(VCPKG_BUILD TRUE)
      message(
        STATUS "Vane: using repo-local vcpkg prefix: ${_VANE_VCPKG_PREFIX}")
      break()
    endif()
  endforeach()
endif()

# PyBind11
find_package(pybind11 REQUIRED CONFIG)

# DuckDB
include(cmake/duckdb_loader.cmake)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  duckdb_configure_for_debug()
else()
  duckdb_configure_for_release()
endif()

# When a vcpkg toolchain is used, resolve distributed exchange dependencies
# through vcpkg packages instead of the legacy external/duckdb/installed layout.
if(DEFINED CMAKE_TOOLCHAIN_FILE
   AND CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg[.]cmake$"
   AND NOT DEFINED DUCKDB_DISTRIBUTED_EXCHANGE_USE_INSTALLED_LIBS)
  set(DUCKDB_DISTRIBUTED_EXCHANGE_USE_INSTALLED_LIBS
      OFF
      CACHE BOOL "Use Arrow/Flight static libs from external/duckdb/installed")
endif()

duckdb_add_library(duckdb_target)

# Bundle in INTERFACE library
add_library(_duckdb_dependencies INTERFACE)
target_link_libraries(_duckdb_dependencies INTERFACE pybind11::pybind11
                                                     duckdb_target)
# Keep Vane's pybind11 type registry private from the official `_duckdb`
# extension. The two engines intentionally do not exchange native Python wrapper
# objects, and sharing a registry would reject duplicate enum and exception
# registrations when both modules are imported in one process. A private
# compiler tag changes pybind11's ABI-domain key without opting into a future
# internals layout.
target_compile_definitions(_duckdb_dependencies
                           INTERFACE PYBIND11_COMPILER_TYPE="vane")
# Also add include directory
target_include_directories(
  _duckdb_dependencies
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/duckdb_py/include>
)

# ────────────────────────────────────────────
# Descend into the real DuckDB‑Python sources
# ────────────────────────────────────────────
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_subdirectory(src/duckdb_py)

pybind11_add_module(
  _vane_duckdb
  $<TARGET_OBJECTS:python_src>
  $<TARGET_OBJECTS:python_arrow>
  $<TARGET_OBJECTS:python_common>
  $<TARGET_OBJECTS:python_functional>
  $<TARGET_OBJECTS:python_jupyter>
  $<TARGET_OBJECTS:python_native>
  $<TARGET_OBJECTS:python_numpy>
  $<TARGET_OBJECTS:python_pandas>
  $<TARGET_OBJECTS:python_pybind11>
  $<TARGET_OBJECTS:python_connection>
  $<TARGET_OBJECTS:python_expression>
  $<TARGET_OBJECTS:python_relation>
  $<TARGET_OBJECTS:python_type>
  $<TARGET_OBJECTS:python_ray>
  $<TARGET_OBJECTS:python_vane_runners>)
# add _duckdb_dependencies
target_link_libraries(_vane_duckdb PRIVATE _duckdb_dependencies)
duckdb_link_extensions(_vane_duckdb)

# ────────────────────────────────────────────
# Controlling symbol export
#
# We want to export exactly two symbols: - PyInit__vane_duckdb: this allows
# CPython to load the module - duckdb_adbc_init: the DuckDB ADBC driver
#
# The export of symbols on OSX and Linux is controlled by: - Visibility
# annotations in the code (for this lib we use the PYBIND11_EXPORT macro) -
# Telling the linker which symbols we want exported, which we do below
#
# For Windows, we rely on just the visbility annotations.
# ────────────────────────────────────────────
set_target_properties(
  _vane_duckdb
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             C_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON)

if(APPLE)
  target_link_options(
    _vane_duckdb PRIVATE "LINKER:-exported_symbol,_duckdb_adbc_init"
    "LINKER:-exported_symbol,_PyInit__vane_duckdb")
elseif(UNIX AND NOT APPLE)
  # Use a version script to export only the two required symbols. This hides all
  # internal symbols (including statically-linked OpenSSL from httpfs) from the
  # dynamic symbol table, preventing SIGSEGV crashes caused by symbol conflicts
  # with system OpenSSL loaded by Ray/Python/gRPC in the same process.
  target_link_options(
    _vane_duckdb
    PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/cmake/exported_symbols.version"
  )
elseif(WIN32)
  target_link_options(_vane_duckdb PRIVATE "/EXPORT:duckdb_adbc_init"
                      "/EXPORT:PyInit__vane_duckdb")
endif()

# ────────────────────────────────────────────
# Put the object file in the correct place
# ────────────────────────────────────────────

# If we're not building through scikit-build-core then we have to set a
# different dest dir
include(GNUInstallDirs)
if(DEFINED SKBUILD_PLATLIB_DIR)
  set(_DUCKDB_PY_INSTALL_DIR "${SKBUILD_PLATLIB_DIR}")
elseif(DEFINED Python_SITEARCH)
  set(_DUCKDB_PY_INSTALL_DIR "${Python_SITEARCH}")
else()
  message(
    WARNING
      "Could not determine Python install dir. Falling back to CMAKE_INSTALL_LIBDIR."
  )
  set(_DUCKDB_PY_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}")
endif()

install(TARGETS _vane_duckdb LIBRARY DESTINATION "${_DUCKDB_PY_INSTALL_DIR}")
