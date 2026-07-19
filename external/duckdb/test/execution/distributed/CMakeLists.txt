# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT

set(TEST_EXECUTION_DISTRIBUTED_SOURCES
    test_aggregate_node.cpp
    test_udf_per_thread.cpp
    test_udf_serialization.cpp
    test_clustering_spec.cpp
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

if(BUILD_DISTRIBUTED_EXCHANGE)
  list(APPEND TEST_EXECUTION_DISTRIBUTED_SOURCES test_exchange.cpp)
endif()

add_library_unity(test_execution_distributed OBJECT
                  ${TEST_EXECUTION_DISTRIBUTED_SOURCES})

set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:test_execution_distributed>
    PARENT_SCOPE)

target_compile_features(test_execution_distributed PRIVATE cxx_std_11)

if(BUILD_DISTRIBUTED_EXCHANGE)
  # test_exchange.cpp includes Arrow C++ headers directly. Consume the imported
  # target when available so its include path and C++20 requirement reach this
  # object library instead of relying on the final unittest link target.
  target_compile_features(test_execution_distributed PRIVATE cxx_std_20)
  if(TARGET Arrow::arrow)
    target_link_libraries(test_execution_distributed PRIVATE Arrow::arrow)
  elseif(TARGET Arrow::arrow_shared)
    target_link_libraries(test_execution_distributed
                          PRIVATE Arrow::arrow_shared)
  elseif(TARGET Arrow::arrow_static)
    target_link_libraries(test_execution_distributed
                          PRIVATE Arrow::arrow_static)
  else()
    target_include_directories(
      test_execution_distributed
      PRIVATE "${PROJECT_SOURCE_DIR}/installed/include")
  endif()
endif()
