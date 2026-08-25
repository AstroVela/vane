cmake_minimum_required(VERSION 3.5...3.29)

project(FileExtension)

include_directories(include)

set(FILE_EXTENSION_FILES file_extension.cpp file_functions.cpp)

build_static_extension(file ${FILE_EXTENSION_FILES})
set(PARAMETERS "-warnings")
build_loadable_extension(file ${PARAMETERS} ${FILE_EXTENSION_FILES})

install(
  TARGETS file_extension
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
