set(PARAMETERS "-warnings")

build_loadable_extension_directory(
  loadable_extension_demo
  "CPP"
  test/extension
  "default-version"
  ""
  ${PARAMETERS}
  loadable_extension_demo.cpp)

# Exercise a private, self-contained engine just like production C++ providers.
if(ENABLE_UNITTEST_CPP_TESTS AND EXTENSION_STATIC_BUILD)
  build_loadable_extension_directory(
    loadable_extension_batch_demo
    "CPP"
    test/extension
    "default-version"
    ""
    ${PARAMETERS}
    loadable_extension_batch_demo.cpp)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(loadable_extension_batch_demo_loadable_extension
                        PRIVATE -Wl,-Bsymbolic -Wl,-z,defs)
  elseif(APPLE)
    # Mach-O's two-level namespace already keeps the private engine local. The
    # extension builder exports only its entry point unless asked for the two
    # additional test factories used by the native adapter regression.
    target_link_options(
      loadable_extension_batch_demo_loadable_extension PRIVATE
      -Wl,-exported_symbol,_batch_demo_has_private_reader
      -Wl,-exported_symbol,_batch_demo_make_operator)
  endif()
  target_compile_definitions(
    test_execution
    PRIVATE
      VANE_BATCH_DEMO_PATH="$<TARGET_FILE:loadable_extension_batch_demo_loadable_extension>"
  )
  if(VANE_ENABLE_TEST_EXTENSION_SIGNING_KEY)
    find_package(
      Python3
      COMPONENTS Interpreter
      REQUIRED)
    get_filename_component(
      VANE_TEST_SIGNER
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../../scripts/sign_test_dynamic_extension.py"
      ABSOLUTE)
    add_custom_command(
      TARGET loadable_extension_batch_demo_loadable_extension
      POST_BUILD
      COMMAND
        "${Python3_EXECUTABLE}" "${VANE_TEST_SIGNER}"
        "$<TARGET_FILE:loadable_extension_batch_demo_loadable_extension>"
        "$<TARGET_FILE:loadable_extension_batch_demo_loadable_extension>"
        --private-key "${PROJECT_SOURCE_DIR}/test/mbedtls/private.pem"
      VERBATIM)
    target_compile_definitions(test_execution PRIVATE VANE_BATCH_DEMO_SIGNED=1)
  endif()
endif()

if(NOT WIN32 AND NOT SUN)
  add_definitions(-DDUCKDB_BUILD_DIRECTORY="${PROJECT_BINARY_DIR}")

  build_loadable_extension_directory(
    loadable_extension_optimizer_demo
    "CPP"
    test/extension
    "default-version"
    ""
    ${PARAMETERS}
    ../extension/loadable_extension_optimizer_demo.cpp)

  if(${ENABLE_UNITTEST_CPP_TESTS} AND NOT (${ENABLE_THREAD_SANITIZER}))
    set(TEST_EXT_OBJECTS test_remote_optimizer.cpp)

    add_library_unity(test_extensions OBJECT ${TEST_EXT_OBJECTS})
    set(ALL_OBJECT_FILES
        ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:test_extensions>
        PARENT_SCOPE)
  endif()
endif()
