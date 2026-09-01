# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT

set(TEST_EXECUTION_DISTRIBUTED_SOURCES
    test_aggregate_node.cpp
    test_udf_per_thread.cpp
    test_udf_serialization.cpp
    test_clustering_spec.cpp
    test_copy_finalize.cpp
    test_data_sink.cpp
    test_distributed_extension_manager.cpp
    test_distributed_csv.cpp
    test_distributed_portable_sources.cpp
    test_distributed_range.cpp
    test_distributed_test_extension.cpp
    test_execute_plan.cpp
    test_physical_operator_visitor.cpp
    test_physical_plan_translator_visitor.cpp
    test_physical_plan_translator.cpp
    test_physical_plan_serialization.cpp
    test_physical_repartition.cpp
    test_pipeline_nodes.cpp
    test_planrunner_local.cpp
    test_repartition_node.cpp
    test_stream.cpp
    test_streaming_channel.cpp
    test_executor_initialize.cpp
    test_executor_pipeline.cpp
    test_external_block.cpp
    test_source_id_routing.cpp)

add_library_unity(test_execution_distributed OBJECT
                  ${TEST_EXECUTION_DISTRIBUTED_SOURCES})
set_target_properties(
  test_execution_distributed
  PROPERTIES CXX_STANDARD 11
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS OFF)

if(BUILD_DISTRIBUTED_EXCHANGE)
  add_library_unity(test_execution_distributed_exchange OBJECT
                    test_exchange.cpp)
  set_target_properties(
    test_execution_distributed_exchange
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_EXTENSIONS OFF)
  if(TARGET Arrow::arrow)
    target_link_libraries(test_execution_distributed_exchange
                          PRIVATE Arrow::arrow)
  elseif(TARGET Arrow::arrow_shared)
    target_link_libraries(test_execution_distributed_exchange
                          PRIVATE Arrow::arrow_shared)
  elseif(TARGET Arrow::arrow_static)
    target_link_libraries(test_execution_distributed_exchange
                          PRIVATE Arrow::arrow_static)
  else()
    target_include_directories(
      test_execution_distributed_exchange
      PRIVATE "${PROJECT_SOURCE_DIR}/installed/include")
  endif()
  set(ALL_OBJECT_FILES
      ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:test_execution_distributed>
      $<TARGET_OBJECTS:test_execution_distributed_exchange>
      PARENT_SCOPE)
else()
  set(ALL_OBJECT_FILES
      ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:test_execution_distributed>
      PARENT_SCOPE)
endif()
