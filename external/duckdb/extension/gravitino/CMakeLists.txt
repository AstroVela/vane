cmake_minimum_required(VERSION 3.5...3.29)
project(GravitinoExtension)

find_package(CURL REQUIRED)
include_directories(include ../../third_party/yyjson/include)
set(GRAVITINO_SOURCES
    gravitino_extension.cpp gravitino_client.cpp gravitino_catalog.cpp
    gravitino_filesystem.cpp gravitino_functions.cpp)
build_static_extension(gravitino ${GRAVITINO_SOURCES})
build_loadable_extension(gravitino "-warnings" ${GRAVITINO_SOURCES})
target_link_libraries(gravitino_extension CURL::libcurl)
target_link_libraries(gravitino_loadable_extension CURL::libcurl duckdb_yyjson)
install(
  TARGETS gravitino_extension
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
