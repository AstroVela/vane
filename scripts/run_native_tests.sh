#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  echo "Usage: scripts/run_native_tests.sh [unittest arguments...]"
  echo
  echo "Build and run Vane's complete DuckDB unit test suite with pinned Arrow/Flight dependencies and C++20."
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

python_cmd="${VANE_NATIVE_PYTHON:-}"
if [[ -z "$python_cmd" ]]; then
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      python_cmd="$(command -v "$candidate")"
      break
    fi
  done
fi
if [[ -z "$python_cmd" ]]; then
  echo "Python is required to configure native tests." >&2
  exit 1
fi

generator="${VANE_NATIVE_CMAKE_GENERATOR:-Ninja}"
generator_platform="${VANE_NATIVE_CMAKE_GENERATOR_PLATFORM:-}"
kernel_name="$(uname -s)"
windows_host=false
case "$kernel_name" in
  MINGW*_NT* | MSYS*_NT* | CYGWIN*_NT*)
    windows_host=true
    ;;
esac
multi_config=false
case "$generator" in
  *"Multi-Config"* | Xcode | Visual\ Studio*)
    multi_config=true
    ;;
  "Green Hills MULTI")
    echo "Unsupported CMake generator: $generator" >&2
    exit 2
    ;;
esac

cmake_args=(
  -DCMAKE_CXX_STANDARD=20
  -DCMAKE_CXX_STANDARD_REQUIRED=ON
  -DCMAKE_CXX_EXTENSIONS=OFF
  -DBUILD_UNITTESTS=ON
  -DBUILD_BENCHMARKS=OFF
  -DBUILD_DISTRIBUTED=ON
  -DBUILD_DISTRIBUTED_EXCHANGE=ON
  -DDUCKDB_DISTRIBUTED_EXCHANGE_USE_INSTALLED_LIBS=OFF
)
if [[ "$multi_config" == true ]]; then
  cmake_args+=("-DCMAKE_CONFIGURATION_TYPES=Release")
else
  cmake_args+=("-DCMAKE_BUILD_TYPE=Release")
fi

source "$project_root/scripts/vcpkg_triplet.sh"
triplet="${VCPKG_TARGET_TRIPLET:-}"
if [[ -z "$triplet" ]]; then
  triplet="$(vane_default_vcpkg_triplet)"
fi
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
if [[ "$windows_host" == true ]]; then
  # vcpkg's Windows package wrappers map names such as the static zlib `zs`
  # library for CMake's generic FindZLIB module. A prefix alone cannot provide
  # that mapping, so use the pinned toolchain in classic (already installed)
  # mode without triggering a second manifest install.
  vcpkg_root="${VCPKG_ROOT:-${RUNNER_TEMP:-$project_root/.cache}/vcpkg}"
  vcpkg_toolchain="$vcpkg_root/scripts/buildsystems/vcpkg.cmake"
  if [[ ! -f "$vcpkg_toolchain" ]]; then
    echo "Pinned vcpkg toolchain not found at $vcpkg_toolchain" >&2
    echo "Run 'bash scripts/bootstrap_vcpkg.sh' from the repository root first." >&2
    exit 1
  fi
  cmake_args+=(
    "-DCMAKE_TOOLCHAIN_FILE=$vcpkg_toolchain"
    "-DVCPKG_INSTALLED_DIR=$install_root"
    "-DVCPKG_TARGET_TRIPLET=$triplet"
    "-DVCPKG_MANIFEST_MODE=OFF"
  )
fi

duckdb_upstream_version="$(<"$project_root/DUCKDB_UPSTREAM_VERSION")"
duckdb_fork_version="$(
  "$python_cmd" "$project_root/scripts/resolve_duckdb_fork_version.py" --print-version
)"
duckdb_source_id="$(
  "$python_cmd" "$project_root/scripts/sync_duckdb_source_id.py" --print
)"
cmake_args+=(
  "-DOVERRIDE_GIT_DESCRIBE=${duckdb_upstream_version}-0-g${duckdb_source_id:0:10}"
  "-DDUCKDB_EXPLICIT_VERSION=$duckdb_fork_version"
  "-DGIT_COMMIT_HASH=${duckdb_source_id:0:10}"
)

build_dir="${VANE_NATIVE_BUILD_DIR:-$project_root/build/native-cxx20}"
if [[ "$build_dir" != /* ]]; then
  build_dir="$project_root/$build_dir"
fi

generator_args=(-G "$generator")
if [[ -n "$generator_platform" ]]; then
  generator_args+=(-A "$generator_platform")
fi
cmake --fresh \
  -S "$project_root/external/duckdb" \
  -B "$build_dir" \
  "${generator_args[@]}" \
  "${cmake_args[@]}"

build_targets=(
  unittest
  loadable_extension_demo_loadable_extension
)
case "$kernel_name" in
  MINGW*_NT* | MSYS*_NT* | CYGWIN*_NT* | SunOS)
    # DuckDB does not define the optimizer demo extension target on Windows or
    # Sun; keep the launcher aligned with test/extension/CMakeLists.txt.
    ;;
  *)
    build_targets+=(loadable_extension_optimizer_demo_loadable_extension)
    ;;
esac
build_args=(
  --target
  "${build_targets[@]}"
  --parallel "$build_jobs"
)
if [[ "$multi_config" == true ]]; then
  build_args=(--config Release "${build_args[@]}")
fi
cmake --build "$build_dir" "${build_args[@]}"

test_binary="$build_dir/test/unittest"
if [[ "$multi_config" == true ]]; then
  test_binary="$build_dir/test/Release/unittest"
fi
if [[ "$windows_host" == true ]]; then
  test_binary="${test_binary}.exe"
fi
if [[ ! -x "$test_binary" ]]; then
  echo "Native test binary was not generated at $test_binary" >&2
  exit 1
fi

exec "$test_binary" "$@"
