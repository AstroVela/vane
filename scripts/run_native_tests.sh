#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  echo "Usage: scripts/run_native_tests.sh [unittest arguments...]"
  echo
  echo "Build Vane's Arrow/Flight native tests with pinned dependencies and C++20."
}

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${1:-}" in
  -h | --help)
    usage
    exit 0
    ;;
esac

build_jobs="${VANE_NATIVE_BUILD_JOBS:-2}"
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "VANE_NATIVE_BUILD_JOBS must be a positive integer: $build_jobs" >&2
  exit 2
fi

generator="${VANE_NATIVE_CMAKE_GENERATOR:-Ninja}"
case "$generator" in
  *"Multi-Config"* | "Green Hills MULTI" | Xcode | Visual\ Studio*)
    echo "Multi-config CMake generators are not supported by this script (Release single-config expected): $generator" >&2
    echo "Set VANE_NATIVE_CMAKE_GENERATOR to a single-config generator such as 'Ninja'." >&2
    exit 2
    ;;
esac

cmake_args=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CXX_STANDARD=20
  -DCMAKE_CXX_STANDARD_REQUIRED=ON
  -DCMAKE_CXX_EXTENSIONS=OFF
  -DBUILD_UNITTESTS=ON
  -DBUILD_BENCHMARKS=OFF
  -DBUILD_DISTRIBUTED=ON
  -DBUILD_DISTRIBUTED_EXCHANGE=ON
  -DDUCKDB_DISTRIBUTED_EXCHANGE_USE_INSTALLED_LIBS=OFF
)

triplet="${VCPKG_TARGET_TRIPLET:-x64-linux}"
install_root="${VCPKG_INSTALLED_DIR:-$project_root/vcpkg_installed}"
if [[ "$install_root" != /* ]]; then
  install_root="$project_root/$install_root"
fi
vcpkg_prefix="$install_root/$triplet"
arrow_config="$vcpkg_prefix/share/arrow/ArrowConfig.cmake"
if [[ ! -f "$arrow_config" ]]; then
  echo "Pinned Arrow package not found at $arrow_config" >&2
  echo "Run 'bash scripts/bootstrap_vcpkg.sh' from the repository root first." >&2
  exit 1
fi
cmake_args+=("-DCMAKE_PREFIX_PATH=$vcpkg_prefix")

override_git_describe="$(
  awk -F'"' '/^OVERRIDE_GIT_DESCRIBE = "/ { print $2; exit }' \
    "$project_root/pyproject.toml"
)"
if [[ -n "$override_git_describe" ]]; then
  cmake_args+=("-DOVERRIDE_GIT_DESCRIBE=$override_git_describe")
fi

build_dir="${VANE_NATIVE_BUILD_DIR:-$project_root/build/native-cxx20}"
if [[ "$build_dir" != /* ]]; then
  build_dir="$project_root/$build_dir"
fi

cmake --fresh \
  -S "$project_root/external/duckdb" \
  -B "$build_dir" \
  -G "$generator" \
  "${cmake_args[@]}"
cmake --build "$build_dir" --target unittest --parallel "$build_jobs"

test_binary="$build_dir/test/unittest"
if [[ ! -x "$test_binary" ]]; then
  echo "Native test binary was not generated at $test_binary" >&2
  exit 1
fi

exec "$test_binary" "$@"
