# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: MIT

add_library_unity(duckdb_func_image OBJECT image_functions.cpp)
target_link_libraries(duckdb_func_image PRIVATE ZLIB::ZLIB)
set(ALL_OBJECT_FILES
    ${ALL_OBJECT_FILES} $<TARGET_OBJECTS:duckdb_func_image>
    PARENT_SCOPE)
