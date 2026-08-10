#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest_path="$project_root/tests/iceberg/upstream_persistent_read_tests.txt"
test_extension_config="$project_root/tests/iceberg/upstream_extension_config.cmake"

build_jobs="${VANE_ICEBERG_COMPAT_BUILD_JOBS:-2}"
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo "VANE_ICEBERG_COMPAT_BUILD_JOBS must be a positive integer: $build_jobs" >&2
  exit 2
fi

generator="${VANE_ICEBERG_COMPAT_CMAKE_GENERATOR:-Ninja}"
case "$generator" in
  *"Multi-Config"* | "Green Hills MULTI" | Xcode | Visual\ Studio*)
    echo "A single-config CMake generator is required: $generator" >&2
    exit 2
    ;;
esac

python_bin="${VANE_ICEBERG_COMPAT_PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python interpreter not found: $python_bin" >&2
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required" >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

find_extension_source() {
  local extension_name="$1"
  local override="$2"
  local candidate

  if [[ -n "$override" ]]; then
    printf '%s\n' "$override"
    return
  fi
  for candidate in \
    "$project_root/build/python-release/_deps/${extension_name}_extension_fc-src" \
    "$project_root/build/_deps/${extension_name}_extension_fc-src"; do
    if [[ -d "$candidate/.git" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

httpfs_source="$(
  find_extension_source httpfs "${VANE_HTTPFS_EXTENSION_SOURCE_DIR:-}"
)" || {
  echo "Pinned httpfs source was not found; run the incremental package build first." >&2
  exit 1
}
avro_source="$(
  find_extension_source avro "${VANE_AVRO_EXTENSION_SOURCE_DIR:-}"
)" || {
  echo "Pinned Avro source was not found; run the incremental package build first." >&2
  exit 1
}
iceberg_source="$(
  find_extension_source iceberg "${VANE_ICEBERG_EXTENSION_SOURCE_DIR:-}"
)" || {
  echo "Pinned Iceberg source was not found; run the incremental package build first." >&2
  exit 1
}

verify_extension_revision() {
  local extension_name="$1"
  local source_dir="$2"
  local config_path="$project_root/external/duckdb/.github/config/extensions/${extension_name}.cmake"
  local expected_revision
  local actual_revision

  expected_revision="$(awk '$1 == "GIT_TAG" { print $2; exit }' "$config_path")"
  actual_revision="$(git -C "$source_dir" rev-parse HEAD)"
  if [[ ! "$expected_revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Could not read the pinned $extension_name revision from $config_path" >&2
    exit 1
  fi
  if [[ "$actual_revision" != "$expected_revision" ]]; then
    echo "$extension_name source revision mismatch: expected $expected_revision, found $actual_revision" >&2
    exit 1
  fi
}

verify_extension_revision httpfs "$httpfs_source"
verify_extension_revision avro "$avro_source"
verify_extension_revision iceberg "$iceberg_source"

iceberg_patch_dir="$project_root/external/duckdb/.github/patches/extensions/iceberg"
for iceberg_patch in \
  "$iceberg_patch_dir/vane-distributed-scan.patch" \
  "$iceberg_patch_dir/vane-distributed-write.patch"; do
  if ! git -C "$iceberg_source" apply --reverse --check "$iceberg_patch"; then
    echo "The reviewed Vane Iceberg patch is not applied cleanly to $iceberg_source: $iceberg_patch" >&2
    exit 1
  fi
done

mapfile -t manifest_tests < <(sed -e '/^[[:space:]]*#/d' -e '/^[[:space:]]*$/d' "$manifest_path")
if ((${#manifest_tests[@]} != 34)); then
  echo "Expected 34 reviewed upstream persistent tests, found ${#manifest_tests[@]} in $manifest_path" >&2
  exit 1
fi
duplicate_tests="$(printf '%s\n' "${manifest_tests[@]}" | sort | uniq -d)"
if [[ -n "$duplicate_tests" ]]; then
  echo "Duplicate tests in $manifest_path:" >&2
  echo "$duplicate_tests" >&2
  exit 1
fi
mapfile -t manifest_tests < <(printf '%s\n' "${manifest_tests[@]}" | sort)

mapfile -t discovered_tests < <(
  cd "$iceberg_source/test/sql"
  while IFS= read -r test_path; do
    if [[ "$test_path" == local/irc/* ]]; then
      continue
    fi
    if ! grep -Eq '^require-env[[:space:]]' "$test_path"; then
      printf '%s\n' "$test_path"
    fi
  done < <(find local -type f \( -name '*.test' -o -name '*.test_slow' \) -print | sort)
)

if [[ "$(printf '%s\n' "${manifest_tests[@]}")" != "$(printf '%s\n' "${discovered_tests[@]}")" ]]; then
  echo "The reviewed upstream persistent-test manifest no longer matches the pinned source:" >&2
  diff -u \
    <(printf '%s\n' "${manifest_tests[@]}") \
    <(printf '%s\n' "${discovered_tests[@]}") || true
  exit 1
fi

triplet="${VCPKG_TARGET_TRIPLET:-x64-linux}"
vcpkg_install_root="${VCPKG_INSTALLED_DIR:-$project_root/vcpkg_installed}"
if [[ "$vcpkg_install_root" != /* ]]; then
  vcpkg_install_root="$project_root/$vcpkg_install_root"
fi
vcpkg_prefix="$vcpkg_install_root/$triplet"
if [[ ! -f "$vcpkg_prefix/share/arrow/ArrowConfig.cmake" || ! -f "$vcpkg_prefix/include/avro.h" ]]; then
  echo "Pinned Arrow and Avro dependencies were not found under $vcpkg_prefix" >&2
  echo "Run scripts/bootstrap_vcpkg.sh or set VCPKG_INSTALLED_DIR." >&2
  exit 1
fi

site_packages="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
if ! "$python_bin" -c 'import pybind11' >/dev/null 2>&1; then
  echo "pybind11 is required in $python_bin" >&2
  exit 1
fi

build_dir="${VANE_ICEBERG_COMPAT_BUILD_DIR:-$project_root/build/iceberg-compat}"
if [[ "$build_dir" != /* ]]; then
  build_dir="$project_root/$build_dir"
fi

export DUCKDB_HTTPFS_DIRECTORY="$httpfs_source"
export DUCKDB_AVRO_DIRECTORY="$avro_source"
export DUCKDB_ICEBERG_DIRECTORY="$iceberg_source"

cmake --fresh \
  -S "$project_root" \
  -B "$build_dir" \
  -G "$generator" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_EXTENSIONS=OFF \
  -DBUILD_UNITTESTS=ON \
  -DBUILD_BENCHMARKS=OFF \
  '-DBUILD_EXTENSIONS=core_functions;parquet;icu;json;httpfs;avro' \
  "-DDUCKDB_EXTENSION_CONFIGS=$test_extension_config" \
  "-DCMAKE_PREFIX_PATH=$vcpkg_prefix;$site_packages" \
  -DVCPKG_BUILD=TRUE \
  -DDUCKDB_DISTRIBUTED_EXCHANGE_USE_INSTALLED_LIBS=OFF

cmake --build "$build_dir" --target unittest --parallel "$build_jobs"

test_binary="$build_dir/duckdb/test/unittest"
if [[ ! -x "$test_binary" ]]; then
  test_binary="$build_dir/test/unittest"
fi
if [[ ! -x "$test_binary" ]]; then
  echo "Iceberg compatibility test runner was not generated under $build_dir" >&2
  exit 1
fi

test_spec=""
for test_path in "${manifest_tests[@]}"; do
  absolute_test_path="$iceberg_source/test/sql/$test_path"
  if [[ ! -f "$absolute_test_path" ]]; then
    echo "Upstream Iceberg test is missing: $absolute_test_path" >&2
    exit 1
  fi
  if [[ -n "$test_spec" ]]; then
    test_spec+=","
  fi
  test_spec+="$absolute_test_path"
done

registered_count="$(
  "$test_binary" --skip-compiled --list-test-names-only "$test_spec" \
    | grep -F -c "$iceberg_source/test/sql/" || true
)"
if [[ "$registered_count" != "${#manifest_tests[@]}" ]]; then
  echo "Expected ${#manifest_tests[@]} registered Iceberg tests, found $registered_count" >&2
  exit 1
fi

exec "$test_binary" --skip-compiled "$test_spec"
