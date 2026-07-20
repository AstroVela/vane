# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
#
# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

# cmake/duckdb_loader.cmake
#
# Simple DuckDB Build Configuration Module
#
# Sets sensible defaults for DuckDB Python extension builds and provides a clean
# interface for adding DuckDB as a library target. Adds jemalloc option for
# debugging but will never allow jemalloc in a release build if not on Linux.
#
# Usage: include(cmake/duckdb_loader.cmake) # Optionally load extensions
# set(BUILD_EXTENSIONS "json;parquet;icu")
#
# # set sensible defaults for a debug build: duckdb_configure_for_debug()
#
# # ...or, set sensible defaults for a release build:
# duckdb_configure_for_release()
#
# # Link to your target duckdb_add_library(duckdb_target)
# target_link_libraries(my_lib PRIVATE ${duckdb_target})

include_guard(GLOBAL)

# ════════════════════════════════════════════════════════════════════════════════
# Configuration Defaults - Optimized for Python Extension Builds
# ════════════════════════════════════════════════════════════════════════════════

# Helper macro to set default values that can be overridden from command line
macro(_duckdb_set_default var_name default_value)
  if(NOT DEFINED ${var_name})
    set(${var_name} ${default_value})
  endif()
endmacro()

# Source configuration
_duckdb_set_default(DUCKDB_SOURCE_PATH
                    "${CMAKE_CURRENT_SOURCE_DIR}/external/duckdb")

# Extension list - commonly used extensions for Python
_duckdb_set_default(BUILD_EXTENSIONS "core_functions;parquet;icu;json;httpfs")

# Core build options - disable unnecessary components for Python builds
_duckdb_set_default(BUILD_SHELL OFF)
_duckdb_set_default(BUILD_UNITTESTS OFF)
_duckdb_set_default(BUILD_BENCHMARKS OFF)
_duckdb_set_default(DISABLE_UNITY OFF)

# Extension configuration
_duckdb_set_default(DISABLE_BUILTIN_EXTENSIONS OFF)
_duckdb_set_default(ENABLE_EXTENSION_AUTOINSTALL ON)
_duckdb_set_default(ENABLE_EXTENSION_AUTOLOADING ON)

# Performance options - enable optimizations by default
_duckdb_set_default(NATIVE_ARCH OFF)

# Sanitizers are off for Python by default. Enabling might result in "symbol not
# found" for  '___ubsan_vptr_type_cache'
_duckdb_set_default(ENABLE_SANITIZER OFF)
_duckdb_set_default(ENABLE_UBSAN OFF)

# Debug options - off by default for release builds
_duckdb_set_default(FORCE_ASSERT OFF)
_duckdb_set_default(DEBUG_STACKTRACE OFF)

# Convert to cache variables for CMake GUI/ccmake compatibility
set(DUCKDB_SOURCE_PATH
    "${DUCKDB_SOURCE_PATH}"
    CACHE PATH "Path to DuckDB source directory")
set(BUILD_EXTENSIONS
    "${BUILD_EXTENSIONS}"
    CACHE STRING "Semicolon-separated list of extensions to enable")
set(BUILD_SHELL
    "${BUILD_SHELL}"
    CACHE BOOL "Build the DuckDB shell executable")
set(BUILD_UNITTESTS
    "${BUILD_UNITTESTS}"
    CACHE BOOL "Build DuckDB unit tests")
set(BUILD_BENCHMARKS
    "${BUILD_BENCHMARKS}"
    CACHE BOOL "Build DuckDB benchmarks")
set(DISABLE_UNITY
    "${DISABLE_UNITY}"
    CACHE BOOL "Disable unity builds (slower compilation)")
set(DISABLE_BUILTIN_EXTENSIONS
    "${DISABLE_BUILTIN_EXTENSIONS}"
    CACHE BOOL "Disable all built-in extensions")
set(ENABLE_EXTENSION_AUTOINSTALL
    "${ENABLE_EXTENSION_AUTOINSTALL}"
    CACHE BOOL "Enable extension auto-installing by default.")
set(ENABLE_EXTENSION_AUTOLOADING
    "${ENABLE_EXTENSION_AUTOLOADING}"
    CACHE BOOL "Enable extension auto-loading by default.")
set(NATIVE_ARCH
    "${NATIVE_ARCH}"
    CACHE BOOL "Optimize for native architecture")
set(ENABLE_SANITIZER
    "${ENABLE_SANITIZER}"
    CACHE BOOL "Enable address sanitizer")
set(ENABLE_UBSAN
    "${ENABLE_UBSAN}"
    CACHE BOOL "Enable undefined behavior sanitizer")
set(FORCE_ASSERT
    "${FORCE_ASSERT}"
    CACHE BOOL "Enable assertions in release builds")
set(DEBUG_STACKTRACE
    "${DEBUG_STACKTRACE}"
    CACHE BOOL "Print a stracktrace on asserts and when testing crashes")

# ════════════════════════════════════════════════════════════════════════════════
# Internal Functions
# ════════════════════════════════════════════════════════════════════════════════

function(_duckdb_validate_jemalloc_config)
  # Check if jemalloc is in the extension list
  if(NOT BUILD_EXTENSIONS MATCHES "jemalloc")
    return()
  endif()

  # jemalloc is only enabled on 64bit x86 linux builds
  if(CMAKE_SIZEOF_VOID_P EQUAL 8
     AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
     AND NOT BSD)
    set(jemalloc_allowed TRUE)
  else()
    set(jemalloc_allowed FALSE)
  endif()

  if(NOT jemalloc_allowed)
    message(WARNING "jemalloc extension is only supported on Linux.\n"
                    "Removing jemalloc from extension list.")
    # Remove jemalloc from the extension list
    string(REPLACE "jemalloc" "" BUILD_EXTENSIONS_FILTERED
                   "${BUILD_EXTENSIONS}")
    string(REGEX REPLACE ";+" ";" BUILD_EXTENSIONS_FILTERED
                         "${BUILD_EXTENSIONS_FILTERED}")
    string(REGEX REPLACE "^;|;$" "" BUILD_EXTENSIONS_FILTERED
                         "${BUILD_EXTENSIONS_FILTERED}")
    set(BUILD_EXTENSIONS
        "${BUILD_EXTENSIONS_FILTERED}"
        PARENT_SCOPE)
  endif()
endfunction()

function(_duckdb_validate_source_path)
  if(NOT EXISTS "${DUCKDB_SOURCE_PATH}")
    message(
      FATAL_ERROR "DuckDB source path does not exist: ${DUCKDB_SOURCE_PATH}\n"
                  "Please set DUCKDB_SOURCE_PATH to the correct location.")
  endif()

  if(NOT EXISTS "${DUCKDB_SOURCE_PATH}/CMakeLists.txt")
    message(
      FATAL_ERROR
        "DuckDB source path does not contain CMakeLists.txt: ${DUCKDB_SOURCE_PATH}\n"
        "Please ensure this points to the root of DuckDB source tree.")
  endif()
endfunction()

function(_duckdb_resolve_source_id)
  set(_VANE_DUCKDB_SOURCE_ID_FILE "${PROJECT_SOURCE_DIR}/DUCKDB_SOURCE_ID")
  set(_VANE_DUCKDB_SOURCE_ID_SCRIPT
      "${PROJECT_SOURCE_DIR}/scripts/sync_duckdb_source_id.py")
  set(_VANE_DUCKDB_DEFAULT_SOURCE_PATH "${PROJECT_SOURCE_DIR}/external/duckdb")
  set(_VANE_DUCKDB_SOURCE_ID_DYNAMIC FALSE)

  if(DEFINED VANE_DUCKDB_SOURCE_ID)
    set(_VANE_DUCKDB_SOURCE_ID "${VANE_DUCKDB_SOURCE_ID}")
  elseif(NOT DUCKDB_SOURCE_PATH STREQUAL _VANE_DUCKDB_DEFAULT_SOURCE_PATH)
    message(FATAL_ERROR "A custom DUCKDB_SOURCE_PATH requires an explicit "
                        "VANE_DUCKDB_SOURCE_ID.")
  elseif(EXISTS "${_VANE_DUCKDB_SOURCE_ID_FILE}"
         AND NOT EXISTS "${PROJECT_SOURCE_DIR}/.git")
    file(READ "${_VANE_DUCKDB_SOURCE_ID_FILE}" _VANE_DUCKDB_SOURCE_ID)
    string(STRIP "${_VANE_DUCKDB_SOURCE_ID}" _VANE_DUCKDB_SOURCE_ID)
  elseif(EXISTS "${_VANE_DUCKDB_SOURCE_ID_SCRIPT}")
    find_package(Python REQUIRED COMPONENTS Interpreter)
    execute_process(
      COMMAND "${Python_EXECUTABLE}" "${_VANE_DUCKDB_SOURCE_ID_SCRIPT}" --print
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      RESULT_VARIABLE _VANE_DUCKDB_SOURCE_ID_RESULT
      OUTPUT_VARIABLE _VANE_DUCKDB_SOURCE_ID
      ERROR_VARIABLE _VANE_DUCKDB_SOURCE_ID_ERROR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_VANE_DUCKDB_SOURCE_ID_RESULT)
      message(FATAL_ERROR "Unable to compute the DuckDB source tree ID: "
                          "${_VANE_DUCKDB_SOURCE_ID_ERROR}")
    endif()
    set(_VANE_DUCKDB_SOURCE_ID_DYNAMIC TRUE)
  else()
    message(
      FATAL_ERROR
        "DUCKDB_SOURCE_ID is unavailable. Provide VANE_DUCKDB_SOURCE_ID or "
        "build from a source tree containing DUCKDB_SOURCE_ID or "
        "scripts/sync_duckdb_source_id.py.")
  endif()

  string(LENGTH "${_VANE_DUCKDB_SOURCE_ID}" _VANE_DUCKDB_SOURCE_ID_LENGTH)
  if(NOT _VANE_DUCKDB_SOURCE_ID MATCHES "^[0-9a-f]+$"
     OR (NOT _VANE_DUCKDB_SOURCE_ID_LENGTH EQUAL 40
         AND NOT _VANE_DUCKDB_SOURCE_ID_LENGTH EQUAL 64))
    message(
      FATAL_ERROR
        "Invalid DuckDB source tree ID '${_VANE_DUCKDB_SOURCE_ID}'. Expected "
        "a 40- or 64-character lowercase hexadecimal Git object ID.")
  endif()

  string(SUBSTRING "${_VANE_DUCKDB_SOURCE_ID}" 0 10
                   _VANE_DUCKDB_SHORT_SOURCE_ID)
  set(VANE_DUCKDB_SOURCE_TREE
      "${_VANE_DUCKDB_SOURCE_ID}"
      PARENT_SCOPE)
  set(GIT_COMMIT_HASH
      "${_VANE_DUCKDB_SHORT_SOURCE_ID}"
      PARENT_SCOPE)
  set(VANE_DUCKDB_SOURCE_ID_DYNAMIC
      "${_VANE_DUCKDB_SOURCE_ID_DYNAMIC}"
      PARENT_SCOPE)
endfunction()

function(_duckdb_collect_buildsystem_targets directory output_variable)
  get_property(
    _VANE_DUCKDB_DIRECTORY_TARGETS
    DIRECTORY "${directory}"
    PROPERTY BUILDSYSTEM_TARGETS)
  get_property(
    _VANE_DUCKDB_SUBDIRECTORIES
    DIRECTORY "${directory}"
    PROPERTY SUBDIRECTORIES)
  foreach(_VANE_DUCKDB_SUBDIRECTORY IN LISTS _VANE_DUCKDB_SUBDIRECTORIES)
    _duckdb_collect_buildsystem_targets("${_VANE_DUCKDB_SUBDIRECTORY}"
                                        _VANE_DUCKDB_CHILD_TARGETS)
    list(APPEND _VANE_DUCKDB_DIRECTORY_TARGETS ${_VANE_DUCKDB_CHILD_TARGETS})
  endforeach()
  set(${output_variable}
      "${_VANE_DUCKDB_DIRECTORY_TARGETS}"
      PARENT_SCOPE)
endfunction()

function(_duckdb_enable_source_id_refresh)
  set(_VANE_DUCKDB_SOURCE_ID_SCRIPT
      "${PROJECT_SOURCE_DIR}/scripts/sync_duckdb_source_id.py")
  set(_VANE_DUCKDB_SOURCE_ID_HEADER
      "${PROJECT_BINARY_DIR}/generated/vane_duckdb_source_id.hpp")

  if(NOT EXISTS "${_VANE_DUCKDB_SOURCE_ID_SCRIPT}")
    message(FATAL_ERROR "Missing ${_VANE_DUCKDB_SOURCE_ID_SCRIPT}")
  endif()
  if(NOT TARGET duckdb_func_table_version)
    message(FATAL_ERROR "DuckDB version target is unavailable")
  endif()

  find_package(Python REQUIRED COMPONENTS Interpreter)
  set(_VANE_DUCKDB_SOURCE_ID_ARGUMENTS --header
                                       "${_VANE_DUCKDB_SOURCE_ID_HEADER}")

  # DuckDB supplies the configured source hash as the default version of each
  # in-tree extension. Record those entry-point targets so the generated header
  # can override their configure-time definitions when a mode-only change is
  # invisible to build-system timestamp checks.
  set(_VANE_DUCKDB_EXTENSION_ROOT "${DUCKDB_SOURCE_PATH}/extension")
  set(_VANE_DUCKDB_IDENTITY_EXTENSION_TARGETS)
  set(_VANE_DUCKDB_IDENTITY_EXTENSION_SOURCES)
  _duckdb_collect_buildsystem_targets("${DUCKDB_SOURCE_PATH}"
                                      _VANE_DUCKDB_EXTENSION_TARGETS)
  list(SORT _VANE_DUCKDB_EXTENSION_TARGETS)
  foreach(_VANE_DUCKDB_EXTENSION_TARGET IN LISTS _VANE_DUCKDB_EXTENSION_TARGETS)
    if(NOT _VANE_DUCKDB_EXTENSION_TARGET MATCHES "^(.+)_extension$")
      continue()
    endif()
    set(_VANE_DUCKDB_EXTENSION_NAME "${CMAKE_MATCH_1}")
    if(_VANE_DUCKDB_EXTENSION_NAME MATCHES "_loadable$")
      continue()
    endif()

    get_target_property(_VANE_DUCKDB_EXTENSION_SOURCE_DIRECTORY
                        "${_VANE_DUCKDB_EXTENSION_TARGET}" SOURCE_DIR)
    if(NOT _VANE_DUCKDB_EXTENSION_SOURCE_DIRECTORY)
      continue()
    endif()
    cmake_path(
      IS_PREFIX _VANE_DUCKDB_EXTENSION_ROOT
      "${_VANE_DUCKDB_EXTENSION_SOURCE_DIRECTORY}" NORMALIZE
      _VANE_DUCKDB_IS_IN_TREE_EXTENSION)
    if(NOT _VANE_DUCKDB_IS_IN_TREE_EXTENSION)
      continue()
    endif()

    string(TOUPPER "${_VANE_DUCKDB_EXTENSION_NAME}"
                   _VANE_DUCKDB_EXTENSION_NAME_UPPERCASE)
    set(_VANE_DUCKDB_EXTENSION_DEFINITION
        "EXT_VERSION_${_VANE_DUCKDB_EXTENSION_NAME_UPPERCASE}=\"${GIT_COMMIT_HASH}\""
    )
    get_property(
      _VANE_DUCKDB_EXTENSION_DEFINITIONS
      DIRECTORY "${_VANE_DUCKDB_EXTENSION_SOURCE_DIRECTORY}"
      PROPERTY COMPILE_DEFINITIONS)
    list(FIND _VANE_DUCKDB_EXTENSION_DEFINITIONS
         "${_VANE_DUCKDB_EXTENSION_DEFINITION}"
         _VANE_DUCKDB_EXTENSION_DEFINITION_INDEX)
    if(_VANE_DUCKDB_EXTENSION_DEFINITION_INDEX EQUAL -1)
      continue()
    endif()

    set(_VANE_DUCKDB_EXTENSION_ENTRY_SOURCE
        "${_VANE_DUCKDB_EXTENSION_SOURCE_DIRECTORY}/${_VANE_DUCKDB_EXTENSION_NAME}_extension.cpp"
    )
    if(NOT EXISTS "${_VANE_DUCKDB_EXTENSION_ENTRY_SOURCE}")
      message(
        FATAL_ERROR
          "Unable to apply the dynamic SourceID to ${_VANE_DUCKDB_EXTENSION_TARGET}: "
          "missing ${_VANE_DUCKDB_EXTENSION_ENTRY_SOURCE}")
    endif()

    list(APPEND _VANE_DUCKDB_SOURCE_ID_ARGUMENTS --define
         "EXT_VERSION_${_VANE_DUCKDB_EXTENSION_NAME_UPPERCASE}")
    list(APPEND _VANE_DUCKDB_IDENTITY_EXTENSION_TARGETS
         "${_VANE_DUCKDB_EXTENSION_TARGET}")
    list(APPEND _VANE_DUCKDB_IDENTITY_EXTENSION_SOURCES
         "${_VANE_DUCKDB_EXTENSION_ENTRY_SOURCE}")
  endforeach()

  if(NOT VANE_DUCKDB_SOURCE_ID_DYNAMIC)
    list(APPEND _VANE_DUCKDB_SOURCE_ID_ARGUMENTS --source-id
         "${VANE_DUCKDB_SOURCE_TREE}")
  endif()

  execute_process(
    COMMAND "${Python_EXECUTABLE}" "${_VANE_DUCKDB_SOURCE_ID_SCRIPT}"
            ${_VANE_DUCKDB_SOURCE_ID_ARGUMENTS}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    RESULT_VARIABLE _VANE_DUCKDB_HEADER_RESULT
    ERROR_VARIABLE _VANE_DUCKDB_HEADER_ERROR)
  if(_VANE_DUCKDB_HEADER_RESULT)
    message(FATAL_ERROR "Unable to generate the DuckDB SourceID header: "
                        "${_VANE_DUCKDB_HEADER_ERROR}")
  endif()

  if(VANE_DUCKDB_SOURCE_ID_DYNAMIC)
    # Makefile generators check whether CMake must rerun before executing ALL
    # targets. Watch the external tree itself so configure-time version values
    # are refreshed on the first incremental build for every generator.
    file(
      GLOB_RECURSE _VANE_DUCKDB_SOURCE_DEPENDENCIES
      LIST_DIRECTORIES FALSE
      CONFIGURE_DEPENDS "${DUCKDB_SOURCE_PATH}/*")
    set_property(
      DIRECTORY "${PROJECT_SOURCE_DIR}"
      APPEND
      PROPERTY CMAKE_CONFIGURE_DEPENDS ${_VANE_DUCKDB_SOURCE_DEPENDENCIES})

    # This target intentionally runs on every native build. The script computes
    # the identity without writing the source tree and rewrites the generated
    # header only when the DuckDB tree ID changes, so direct incremental builds
    # cannot retain an earlier SourceID.
    add_custom_target(
      vane_duckdb_source_id_refresh ALL
      COMMAND "${Python_EXECUTABLE}" "${_VANE_DUCKDB_SOURCE_ID_SCRIPT}"
              ${_VANE_DUCKDB_SOURCE_ID_ARGUMENTS}
      BYPRODUCTS "${_VANE_DUCKDB_SOURCE_ID_HEADER}"
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      COMMENT "Refreshing DuckDB SourceID"
      VERBATIM)
  endif()

  set_source_files_properties("${_VANE_DUCKDB_SOURCE_ID_HEADER}"
                              PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
  target_sources(duckdb_func_table_version
                 PRIVATE "${_VANE_DUCKDB_SOURCE_ID_HEADER}")
  if(MSVC)
    target_compile_options(duckdb_func_table_version
                           PRIVATE "/FI${_VANE_DUCKDB_SOURCE_ID_HEADER}")
  else()
    target_compile_options(duckdb_func_table_version
                           PRIVATE -include "${_VANE_DUCKDB_SOURCE_ID_HEADER}")
  endif()
  if(VANE_DUCKDB_SOURCE_ID_DYNAMIC)
    add_dependencies(duckdb_func_table_version vane_duckdb_source_id_refresh)
  endif()

  list(LENGTH _VANE_DUCKDB_IDENTITY_EXTENSION_TARGETS
       _VANE_DUCKDB_IDENTITY_EXTENSION_COUNT)
  if(_VANE_DUCKDB_IDENTITY_EXTENSION_COUNT GREATER 0)
    math(EXPR _VANE_DUCKDB_IDENTITY_EXTENSION_LAST
         "${_VANE_DUCKDB_IDENTITY_EXTENSION_COUNT} - 1")
    foreach(_VANE_DUCKDB_IDENTITY_EXTENSION_INDEX
            RANGE ${_VANE_DUCKDB_IDENTITY_EXTENSION_LAST})
      list(GET _VANE_DUCKDB_IDENTITY_EXTENSION_TARGETS
           ${_VANE_DUCKDB_IDENTITY_EXTENSION_INDEX}
           _VANE_DUCKDB_IDENTITY_EXTENSION_TARGET)
      list(GET _VANE_DUCKDB_IDENTITY_EXTENSION_SOURCES
           ${_VANE_DUCKDB_IDENTITY_EXTENSION_INDEX}
           _VANE_DUCKDB_IDENTITY_EXTENSION_SOURCE)
      target_sources("${_VANE_DUCKDB_IDENTITY_EXTENSION_TARGET}"
                     PRIVATE "${_VANE_DUCKDB_SOURCE_ID_HEADER}")
      set_property(
        SOURCE "${_VANE_DUCKDB_IDENTITY_EXTENSION_SOURCE}" TARGET_DIRECTORY
               "${_VANE_DUCKDB_IDENTITY_EXTENSION_TARGET}"
        APPEND
        PROPERTY OBJECT_DEPENDS "${_VANE_DUCKDB_SOURCE_ID_HEADER}")
      if(MSVC)
        set_property(
          SOURCE "${_VANE_DUCKDB_IDENTITY_EXTENSION_SOURCE}" TARGET_DIRECTORY
                 "${_VANE_DUCKDB_IDENTITY_EXTENSION_TARGET}"
          APPEND
          PROPERTY COMPILE_OPTIONS "/FI${_VANE_DUCKDB_SOURCE_ID_HEADER}")
      else()
        set_property(
          SOURCE "${_VANE_DUCKDB_IDENTITY_EXTENSION_SOURCE}" TARGET_DIRECTORY
                 "${_VANE_DUCKDB_IDENTITY_EXTENSION_TARGET}"
          APPEND
          PROPERTY COMPILE_OPTIONS -include "${_VANE_DUCKDB_SOURCE_ID_HEADER}")
      endif()
      if(VANE_DUCKDB_SOURCE_ID_DYNAMIC)
        add_dependencies("${_VANE_DUCKDB_IDENTITY_EXTENSION_TARGET}"
                         vane_duckdb_source_id_refresh)
      endif()

      # Static and loadable variants share the same entry-point source and
      # therefore its directory-scoped source properties. Give the loadable
      # target the matching generated-file dependency when DuckDB created one.
      string(
        REGEX
        REPLACE "_extension$" "_loadable_extension"
                _VANE_DUCKDB_IDENTITY_LOADABLE_TARGET
                "${_VANE_DUCKDB_IDENTITY_EXTENSION_TARGET}")
      if(TARGET "${_VANE_DUCKDB_IDENTITY_LOADABLE_TARGET}")
        target_sources("${_VANE_DUCKDB_IDENTITY_LOADABLE_TARGET}"
                       PRIVATE "${_VANE_DUCKDB_SOURCE_ID_HEADER}")
        if(VANE_DUCKDB_SOURCE_ID_DYNAMIC)
          add_dependencies("${_VANE_DUCKDB_IDENTITY_LOADABLE_TARGET}"
                           vane_duckdb_source_id_refresh)
        endif()
      endif()
    endforeach()
  endif()
endfunction()

function(_duckdb_create_interface_target target_name)
  add_library(${target_name} INTERFACE)

  # Include directories to deal with leaking third-party headers in DuckDB
  # headers.
  target_include_directories(
    ${target_name}
    INTERFACE
      # Main DuckDB headers
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/src/include>
      # Third-party headers that leak through DuckDB's API
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/third_party>
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/third_party/re2>
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/third_party/fast_float>
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/third_party/utf8proc/include>
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/third_party/libpg_query/include>
      $<BUILD_INTERFACE:${DUCKDB_SOURCE_PATH}/third_party/fmt/include>)

  # Compile definitions based on configuration
  target_compile_definitions(
    ${target_name} INTERFACE $<$<BOOL:${FORCE_ASSERT}>:DUCKDB_FORCE_ASSERT>
                             $<$<CONFIG:Debug>:DUCKDB_DEBUG_MODE>)

  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_options(
      ${target_name}
      INTERFACE /wd4244 # suppress Conversion from 'type1' to 'type2', possible
                        # loss of data
                /wd4267 # suppress Conversion from ‘size_t’ to ‘type’, possible
                        # loss of data
                /wd4200 # suppress Nonstandard extension used: zero-sized array
                        # in struct/union
                /wd26451
                /wd26495 # suppress Code Analysis
                /D_CRT_SECURE_NO_WARNINGS # suppress warnings about unsafe
                                          # functions
                /utf-8 # treat source files as UTF-8 encoded
    )
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_compile_options(
      ${target_name}
      INTERFACE -stdlib=libc++ # for libc++ in favor of older libstdc++
                -mmacosx-version-min=10.7 # minimum osx version compatibility
    )
  endif()

  # Link to the DuckDB static library
  target_link_libraries(${target_name} INTERFACE duckdb_static)

  # Enable position independent code for shared library builds
  set_target_properties(${target_name}
                        PROPERTIES INTERFACE_POSITION_INDEPENDENT_CODE ON)
endfunction()

function(_duckdb_print_summary)
  message(STATUS "DuckDB Configuration:")
  message(STATUS "  Source: ${DUCKDB_SOURCE_PATH}")
  message(STATUS "  Source tree: ${VANE_DUCKDB_SOURCE_TREE}")
  message(STATUS "  Source ID: ${GIT_COMMIT_HASH}")
  message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
  message(STATUS "  Native Arch: ${NATIVE_ARCH}")
  message(STATUS "  Unity Build Disabled: ${DISABLE_UNITY}")

  set(debug_opts)
  if(FORCE_ASSERT)
    list(APPEND debug_opts "FORCE_ASSERT")
  endif()
  if(DEBUG_STACKTRACE)
    list(APPEND debug_opts "DEBUG_STACKTRACE")
  endif()

  if(debug_opts)
    message(STATUS "  Debug Options: ${debug_opts}")
  endif()
endfunction()

# ════════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════════

function(duckdb_add_library target_name)
  _duckdb_validate_source_path()
  _duckdb_validate_jemalloc_config()
  _duckdb_resolve_source_id()
  _duckdb_print_summary()

  # Add DuckDB subdirectory - it will use our variables
  add_subdirectory("${DUCKDB_SOURCE_PATH}" duckdb EXCLUDE_FROM_ALL)
  _duckdb_enable_source_id_refresh()

  # Create clean interface target
  _duckdb_create_interface_target(${target_name})

  # Propagate BUILD_EXTENSIONS back to caller scope in case it was modified
  set(BUILD_EXTENSIONS
      "${BUILD_EXTENSIONS}"
      PARENT_SCOPE)
endfunction()

function(duckdb_link_extensions target_name)
  # Link to the DuckDB static library and extensions We use WHOLE_ARCHIVE
  # because duckdb_static calls LoadAllExtensions which is defined in the
  # extension loader. Without this, linkers (especially on Linux with
  # --as-needed) may drop the extension loader before seeing the reference.
  target_link_libraries(
    ${target_name}
    PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,duckdb_generated_extension_loader>")
  if(BUILD_EXTENSIONS)
    message(STATUS "Linking DuckDB extensions:")
    foreach(ext IN LISTS BUILD_EXTENSIONS)
      message(STATUS "- ${ext}")
      target_link_libraries(${target_name} PRIVATE ${ext}_extension)
    endforeach()
  else()
    message(STATUS "No DuckDB extensions linked in")
  endif()
endfunction()

# ════════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ════════════════════════════════════════════════════════════════════════════════

function(duckdb_configure_for_debug)
  # Only set if not already defined (allows override from command line)
  if(NOT DEFINED FORCE_ASSERT)
    set(FORCE_ASSERT
        ON
        PARENT_SCOPE)
  endif()
  if(NOT DEFINED DEBUG_STACKTRACE)
    set(DEBUG_STACKTRACE
        ON
        PARENT_SCOPE)
  endif()
  message(STATUS "DuckDB: Configured for debug build")
endfunction()

function(duckdb_configure_for_release)
  message(STATUS "DuckDB: Configured for release build")
endfunction()
