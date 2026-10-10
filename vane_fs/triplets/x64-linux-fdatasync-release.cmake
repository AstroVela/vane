# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: Apache-2.0

set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)
set(VCPKG_BUILD_TYPE release)

# This must affect SQLite itself, before its static library is built.
if(PORT STREQUAL "sqlite3")
  string(APPEND VCPKG_C_FLAGS " -DHAVE_FDATASYNC=1")
  string(APPEND VCPKG_CXX_FLAGS " -DHAVE_FDATASYNC=1")
endif()
