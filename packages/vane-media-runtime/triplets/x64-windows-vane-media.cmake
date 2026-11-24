# SPDX-FileCopyrightText: 2026 Vane contributors
#
# SPDX-License-Identifier: Apache-2.0

set(VCPKG_TARGET_ARCHITECTURE x64)
# Keep independently replaceable codec DLLs self-contained with respect to the
# MSVC runtime, matching the base wheel's static CRT policy.
set(VCPKG_CRT_LINKAGE static)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_BUILD_TYPE release)
