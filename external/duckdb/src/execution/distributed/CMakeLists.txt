# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT

# CMakeLists.txt for DuckDB Distributed Execution Module
#
# This module provides distributed execution capabilities translated from
# DuckDB's duckdb-distributed framework.
#
# Requirements: - C++11 - Thread support

# Option to build pipeline_node sources. Turn OFF to skip compiling
# distributed/pipeline_node.
option(BUILD_DISTRIBUTED_PIPELINE_NODE
       "Build pipeline_node components of the distributed module" ON)
# Option to build Arrow Flight-based exchange support.
option(BUILD_DISTRIBUTED_EXCHANGE "Build Arrow Flight exchange support" ON)

set(DISTRIBUTED_SOURCES_EXCHANGE)
if(BUILD_DISTRIBUTED_EXCHANGE)
  set(DISTRIBUTED_SOURCES_EXCHANGE
      exchange/flight_ticket.cpp exchange/shuffle_cache.cpp
      exchange/flight_client.cpp exchange/flight_server.cpp
      exchange/flight_exchange_manager.cpp)
else()
  set(DISTRIBUTED_SOURCES_EXCHANGE exchange/flight_exchange_manager_stub.cpp)
endif()

# Source files for the distributed module
set(DISTRIBUTED_SOURCES_COMMON
    plan/distributed_physical_plan.cpp
    plan/exchange_sink_instance_task.cpp
    plan/exchange_source_task.cpp
    plan/fte_split_queue.cpp
    plan/scan_task.cpp
    ${DISTRIBUTED_SOURCES_EXCHANGE}
    # python-related sources are optional and appended below when enabled
)

set(DISTRIBUTED_SOURCES_PIPELINE_NODE
    pipeline_node/aggregate.cpp
    pipeline_node/expression_scan.cpp
    pipeline_node/join/broadcast_join.cpp
    pipeline_node/join/delim_join.cpp
    pipeline_node/join/hash_join.cpp
    pipeline_node/limit.cpp
    pipeline_node/materialized_coordinator.cpp
    pipeline_node/translate.cpp
    pipeline_node/translator_aggregate.cpp
    pipeline_node/translator_copy.cpp
    pipeline_node/translator_join.cpp
    pipeline_node/translator_misc_unary.cpp
    pipeline_node/translator_order_limit.cpp
    pipeline_node/translator_scan.cpp
    pipeline_node/translator_source.cpp
    pipeline_node/translator_unary.cpp
    pipeline_node/scan_source.cpp
    pipeline_node/sample.cpp
    pipeline_node/pivot.cpp
    pipeline_node/unnest.cpp
    pipeline_node/shuffles/repartition.cpp
    pipeline_node/sink.cpp
    pipeline_node/sort.cpp
    pipeline_node/window.cpp
    pipeline_node/pipeline_node.cpp
    pipeline_node/pipeline_node_task_builder.cpp
    pipeline_node/pipeline_node_visualization.cpp
    pipeline_node/submittable_task_stream.cpp)

if(BUILD_DISTRIBUTED_PIPELINE_NODE)
  set(DISTRIBUTED_SOURCES ${DISTRIBUTED_SOURCES_COMMON}
                          ${DISTRIBUTED_SOURCES_PIPELINE_NODE})
else()
  set(DISTRIBUTED_SOURCES ${DISTRIBUTED_SOURCES_COMMON})
endif()

# Create the distributed execution library using DuckDB's unity build helper
add_library_unity(duckdb_execution_distributed OBJECT ${DISTRIBUTED_SOURCES})

target_compile_features(duckdb_execution_distributed PRIVATE cxx_std_11)

# Set include directories
target_include_directories(
  duckdb_execution_distributed
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/plan
          ${PROJECT_SOURCE_DIR}/third_party/parquet
          ${PROJECT_SOURCE_DIR}/third_party/thrift)

if(BUILD_DISTRIBUTED_PIPELINE_NODE)
  target_include_directories(
    duckdb_execution_distributed
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_node
            ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_node/join
            ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_node/shuffles)
endif()

if(BUILD_DISTRIBUTED_EXCHANGE)
  target_include_directories(duckdb_execution_distributed
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/exchange)
endif()

set(DUCKDB_DISTRIBUTED_LINK_LIBS "")
if(BUILD_DISTRIBUTED_EXCHANGE)
  option(DUCKDB_DISTRIBUTED_EXCHANGE_USE_INSTALLED_LIBS
         "Use Arrow/Flight static libs from external/duckdb/installed" ON)

  set(_duckdb_use_installed_libs
      ${DUCKDB_DISTRIBUTED_EXCHANGE_USE_INSTALLED_LIBS})
  if(_duckdb_use_installed_libs)
    get_filename_component(_duckdb_root "${CMAKE_CURRENT_LIST_DIR}/../../.."
                           ABSOLUTE)
    set(_duckdb_installed_prefix "${_duckdb_root}/installed")
    if(EXISTS "${_duckdb_installed_prefix}/lib64/libarrow.a")
      set(_duckdb_installed_lib "${_duckdb_installed_prefix}/lib64")
    elseif(EXISTS "${_duckdb_installed_prefix}/lib/libarrow.a")
      set(_duckdb_installed_lib "${_duckdb_installed_prefix}/lib")
    else()
      set(_duckdb_use_installed_libs OFF)
    endif()
  endif()

  if(_duckdb_use_installed_libs)
    find_package(Threads REQUIRED)

    function(duckdb_append_if_exists list_var)
      foreach(path IN LISTS ARGN)
        if(EXISTS "${path}")
          list(APPEND ${list_var} "${path}")
        endif()
      endforeach()
      set(${list_var}
          "${${list_var}}"
          PARENT_SCOPE)
    endfunction()

    if(NOT EXISTS "${_duckdb_installed_lib}/libarrow.a"
       OR NOT EXISTS "${_duckdb_installed_lib}/libarrow_flight.a"
       OR NOT EXISTS "${_duckdb_installed_lib}/libgrpc.a"
       OR NOT EXISTS "${_duckdb_installed_lib}/libgrpc++.a")
      message(
        FATAL_ERROR
          "Installed Arrow/Flight static libs not found under ${_duckdb_installed_lib}"
      )
    endif()

    target_include_directories(duckdb_execution_distributed
                               PRIVATE "${_duckdb_installed_prefix}/include")

    file(GLOB _duckdb_absl_libs "${_duckdb_installed_lib}/libabsl_*.a")
    list(SORT _duckdb_absl_libs)
    file(GLOB _duckdb_grpc_libs "${_duckdb_installed_lib}/libgrpc*.a")
    list(SORT _duckdb_grpc_libs)

    set(_duckdb_exchange_libs "")
    duckdb_append_if_exists(
      _duckdb_exchange_libs
      "${_duckdb_installed_lib}/libarrow_flight.a"
      "${_duckdb_installed_lib}/libarrow.a"
      "${_duckdb_installed_lib}/libarrow_flight_sql.a"
      "${_duckdb_installed_lib}/libarrow_bundled_dependencies.a"
      "${_duckdb_installed_lib}/libglog.a"
      "${_duckdb_installed_lib}/libgflags.a"
      "${_duckdb_installed_lib}/libgflags_nothreads.a"
      "${_duckdb_installed_lib}/libprotobuf.a"
      "${_duckdb_installed_lib}/libprotobuf-lite.a"
      "${_duckdb_installed_lib}/libgpr.a"
      "${_duckdb_installed_lib}/libaddress_sorting.a"
      "${_duckdb_installed_lib}/libcares.a"
      "${_duckdb_installed_lib}/libssl.a"
      "${_duckdb_installed_lib}/libcrypto.a"
      "${_duckdb_installed_lib}/libz.a"
      "${_duckdb_installed_lib}/libsnappy.a"
      "${_duckdb_installed_lib}/libzstd.a"
      "${_duckdb_installed_lib}/liblz4.a"
      "${_duckdb_installed_lib}/libbz2.a"
      "${_duckdb_installed_lib}/libbrotlicommon.a"
      "${_duckdb_installed_lib}/libbrotlidec.a"
      "${_duckdb_installed_lib}/libbrotlienc.a"
      "${_duckdb_installed_lib}/libre2.a"
      "${_duckdb_installed_lib}/libupb.a")

    list(APPEND _duckdb_exchange_libs ${_duckdb_grpc_libs} ${_duckdb_absl_libs}
         Threads::Threads)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU"
       AND UNIX
       AND NOT APPLE)
      # Some prebuilt static exchange dependencies reference libsupc++ ABI
      # helpers that are not exported by GCC 11's shared libstdc++.
      list(APPEND _duckdb_exchange_libs supc++)
    endif()

    if(UNIX AND NOT APPLE)
      set(_duckdb_link_group_start "-Wl,--start-group")
      set(_duckdb_link_group_end "-Wl,--end-group")
    endif()

    target_link_libraries(
      duckdb_execution_distributed
      PRIVATE ${_duckdb_link_group_start} ${_duckdb_exchange_libs}
              ${_duckdb_link_group_end})
    target_compile_definitions(duckdb_execution_distributed
                               PRIVATE DUCKDB_DISTRIBUTED_EXCHANGE=1)
    list(APPEND DUCKDB_DISTRIBUTED_LINK_LIBS ${_duckdb_link_group_start}
         ${_duckdb_exchange_libs} ${_duckdb_link_group_end})
  else()
    find_package(Arrow CONFIG REQUIRED)
    find_package(ArrowFlight CONFIG REQUIRED)
    find_package(gRPC CONFIG REQUIRED)
    find_package(glog CONFIG REQUIRED)
    set(GFLAGS_USE_TARGET_NAMESPACE TRUE)
    find_package(gflags CONFIG REQUIRED)

    function(duckdb_set_imported_global target_name)
      if(TARGET ${target_name})
        set_target_properties(${target_name} PROPERTIES IMPORTED_GLOBAL TRUE)
      endif()
    endfunction()

    duckdb_set_imported_global(Arrow::arrow)
    duckdb_set_imported_global(Arrow::arrow_shared)
    duckdb_set_imported_global(Arrow::arrow_static)
    duckdb_set_imported_global(ArrowFlight::arrow_flight)
    duckdb_set_imported_global(ArrowFlight::arrow_flight_shared)
    duckdb_set_imported_global(ArrowFlight::arrow_flight_static)
    duckdb_set_imported_global(gRPC::grpc++)
    duckdb_set_imported_global(gRPC::grpc)
    duckdb_set_imported_global(glog::glog)
    duckdb_set_imported_global(gflags::gflags_static)

    set(ARROW_TARGET Arrow::arrow)
    if(NOT TARGET ${ARROW_TARGET})
      if(TARGET Arrow::arrow_shared)
        set(ARROW_TARGET Arrow::arrow_shared)
      elseif(TARGET Arrow::arrow_static)
        set(ARROW_TARGET Arrow::arrow_static)
      else()
        message(FATAL_ERROR "Arrow target not found after find_package(Arrow)")
      endif()
    endif()

    set(ARROW_FLIGHT_TARGET ArrowFlight::arrow_flight)
    if(NOT TARGET ${ARROW_FLIGHT_TARGET})
      if(TARGET ArrowFlight::arrow_flight_shared)
        set(ARROW_FLIGHT_TARGET ArrowFlight::arrow_flight_shared)
      elseif(TARGET ArrowFlight::arrow_flight_static)
        set(ARROW_FLIGHT_TARGET ArrowFlight::arrow_flight_static)
      else()
        message(
          FATAL_ERROR
            "Arrow Flight target not found after find_package(ArrowFlight)")
      endif()
    endif()

    target_link_libraries(
      duckdb_execution_distributed
      PRIVATE ${ARROW_TARGET} ${ARROW_FLIGHT_TARGET} gRPC::grpc++ gRPC::grpc
              glog::glog gflags::gflags_static)
    target_compile_definitions(duckdb_execution_distributed
                               PRIVATE DUCKDB_DISTRIBUTED_EXCHANGE=1)
    list(
      APPEND
      DUCKDB_DISTRIBUTED_LINK_LIBS
      ${ARROW_TARGET}
      ${ARROW_FLIGHT_TARGET}
      gRPC::grpc++
      gRPC::grpc
      glog::glog
      gflags::gflags_static)
  endif()
else()
  target_compile_definitions(duckdb_execution_distributed
                             PRIVATE DUCKDB_DISTRIBUTED_EXCHANGE=0)
endif()

set(DUCKDB_DISTRIBUTED_LINK_LIBS
    ${DUCKDB_DISTRIBUTED_LINK_LIBS}
    PARENT_SCOPE)

# Export object files to parent scope for linking
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_execution_distributed>
    PARENT_SCOPE)

# Propagate BUILD_DISTRIBUTED_PIPELINE_NODE into compile-time macro for the
# target
if(BUILD_DISTRIBUTED_PIPELINE_NODE)
  target_compile_definitions(duckdb_execution_distributed
                             PRIVATE BUILD_DISTRIBUTED_PIPELINE_NODE=1)
else()
  target_compile_definitions(duckdb_execution_distributed
                             PRIVATE BUILD_DISTRIBUTED_PIPELINE_NODE=0)
endif()
