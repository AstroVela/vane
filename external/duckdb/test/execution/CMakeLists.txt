# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT

option(BUILD_DISTRIBUTED "Build the distributed execution tests" ON)

add_library_unity(test_execution OBJECT test_execution_batch.cpp
                  test_vllm_inflight_counter.cpp)
set(ALL_OBJECT_FILES ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:test_execution>)

if(BUILD_DISTRIBUTED)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/distributed)
    add_subdirectory(distributed)
  endif()
endif()

# If any subdirectory appended to ALL_OBJECT_FILES in this scope, propagate it
# up so top-level `test/CMakeLists.txt` picks up object libraries from nested
# dirs
if(DEFINED ALL_OBJECT_FILES)
  set(ALL_OBJECT_FILES
      ${ALL_OBJECT_FILES}
      PARENT_SCOPE)
endif()
