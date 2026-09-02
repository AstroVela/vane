#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

vane_default_vcpkg_triplet() {
  local system_name="${1:-$(uname -s)}"
  local machine_name="${2:-$(uname -m)}"

  case "${system_name}:${machine_name}" in
    Linux:x86_64 | Linux:amd64)
      printf '%s\n' x64-linux-release
      ;;
    Linux:aarch64 | Linux:arm64)
      printf '%s\n' arm64-linux-release
      ;;
    Darwin:x86_64 | Darwin:amd64)
      printf '%s\n' x64-osx-release
      ;;
    Darwin:arm64 | Darwin:aarch64)
      printf '%s\n' arm64-osx-release
      ;;
    MINGW*_NT*:x86_64 | MSYS*_NT*:x86_64 | CYGWIN*_NT*:x86_64)
      printf '%s\n' x64-windows-static-release
      ;;
    MINGW*_NT*:arm64 | MINGW*_NT*:aarch64 | MSYS*_NT*:arm64 | MSYS*_NT*:aarch64 | CYGWIN*_NT*:arm64 | CYGWIN*_NT*:aarch64)
      printf '%s\n' arm64-windows-static-release
      ;;
    *)
      printf 'No default vcpkg triplet for %s on %s; set VCPKG_TARGET_TRIPLET and VCPKG_HOST_TRIPLET.\n' \
        "$system_name" "$machine_name" >&2
      return 1
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  vane_default_vcpkg_triplet "$@"
fi
