#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_fast_tests.sh [all|non-ray|shared-ray|owner-ray]

Run the complete fast suite or one process-isolated phase.

Environment variables:
  VANE_FAST_TEST_ARTIFACT_MODE
      Set to 1 to test installed packages while retaining checkout support modules.
  VANE_FAST_TEST_EXCLUDE_GPU
      Set to 1 to exclude tests marked gpu on CPU-only hosts.
  VANE_FAST_TEST_EXTERNAL_RAY_CLEANUP
      Set to 1 on an isolated host to stop Ray between owner-test processes.
  VANE_FAST_TEST_JUNIT_DIR
      Write one or more JUnit XML reports to this directory.
  VANE_FAST_TEST_NON_RAY_SHARD_COUNT
      Split the non-Ray phase into this many deterministic shards.
  VANE_FAST_TEST_NON_RAY_SHARD_INDEX
      Zero-based non-Ray shard index.
  VANE_FAST_TEST_PROCESS_TIMEOUT_SECONDS
      Apply a hard deadline to every pytest process (requires GNU timeout).
EOF
}

phase="${1:-all}"
if (($# > 1)); then
  usage >&2
  exit 2
fi
case "$phase" in
  all | non-ray | shared-ray | owner-ray) ;;
  -h | --help)
    usage
    exit
    ;;
  *)
    printf 'Unknown fast-test phase: %s\n' "$phase" >&2
    usage >&2
    exit 2
    ;;
esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

artifact_mode="${VANE_FAST_TEST_ARTIFACT_MODE:-0}"
pytest_mode_args=()
case "$artifact_mode" in
  0) ;;
  1)
    site_packages="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
    export PYTHONSAFEPATH=1
    export PYTHONPATH="${site_packages}:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
    pytest_mode_args+=("--import-mode=importlib")
    ;;
  *)
    printf 'VANE_FAST_TEST_ARTIFACT_MODE must be 0 or 1: %s\n' "$artifact_mode" >&2
    exit 2
    ;;
esac

exclude_gpu="${VANE_FAST_TEST_EXCLUDE_GPU:-0}"
case "$exclude_gpu" in
  0) gpu_marker="" ;;
  1) gpu_marker=" and not gpu" ;;
  *)
    printf 'VANE_FAST_TEST_EXCLUDE_GPU must be 0 or 1: %s\n' "$exclude_gpu" >&2
    exit 2
    ;;
esac

external_ray_cleanup="${VANE_FAST_TEST_EXTERNAL_RAY_CLEANUP:-0}"
case "$external_ray_cleanup" in
  0 | 1) ;;
  *)
    printf 'VANE_FAST_TEST_EXTERNAL_RAY_CLEANUP must be 0 or 1: %s\n' \
      "$external_ray_cleanup" >&2
    exit 2
    ;;
esac

non_ray_shard_count="${VANE_FAST_TEST_NON_RAY_SHARD_COUNT:-1}"
non_ray_shard_index="${VANE_FAST_TEST_NON_RAY_SHARD_INDEX:-0}"
if [[ ! "$non_ray_shard_count" =~ ^[1-9][0-9]*$ ]]; then
  printf 'VANE_FAST_TEST_NON_RAY_SHARD_COUNT must be a positive integer: %s\n' "$non_ray_shard_count" >&2
  exit 2
fi
if [[ ! "$non_ray_shard_index" =~ ^[0-9]+$ ]] || ((non_ray_shard_index >= non_ray_shard_count)); then
  printf 'VANE_FAST_TEST_NON_RAY_SHARD_INDEX must be in [0, %d): %s\n' \
    "$non_ray_shard_count" "$non_ray_shard_index" >&2
  exit 2
fi

process_timeout_seconds="${VANE_FAST_TEST_PROCESS_TIMEOUT_SECONDS:-0}"
if [[ ! "$process_timeout_seconds" =~ ^[0-9]+$ ]]; then
  printf 'VANE_FAST_TEST_PROCESS_TIMEOUT_SECONDS must be a non-negative integer: %s\n' \
    "$process_timeout_seconds" >&2
  exit 2
fi
if ((process_timeout_seconds > 0)) && ! command -v timeout >/dev/null 2>&1; then
  printf '%s\n' "VANE_FAST_TEST_PROCESS_TIMEOUT_SECONDS requires the GNU timeout command" >&2
  exit 2
fi

junit_dir="${VANE_FAST_TEST_JUNIT_DIR:-}"
if [[ -n "$junit_dir" ]]; then
  mkdir -p "$junit_dir"
  junit_dir="$(cd "$junit_dir" && pwd)"
fi

ray_object_store_bytes="$(
  PYTHONPATH="tests${PYTHONPATH:+:${PYTHONPATH}}" \
    python -c "from ray_test_profile import ray_test_object_store_bytes; print(ray_test_object_store_bytes())"
)"

run_pytest() {
  if ((process_timeout_seconds > 0)); then
    timeout \
      --signal=INT \
      --kill-after=30s \
      "${process_timeout_seconds}s" \
      python -m pytest "${pytest_mode_args[@]}" "$@"
  else
    python -m pytest "${pytest_mode_args[@]}" "$@"
  fi
}

run_reported_pytest() {
  local report_name="$1"
  shift

  local report_args=()
  if [[ -n "$junit_dir" ]]; then
    report_args+=("--junitxml=$junit_dir/$report_name.xml")
  fi
  run_pytest "${report_args[@]}" "$@"
}

run_ray_pytest() {
  VANE_TEST_RAY_OBJECT_STORE_BYTES="${ray_object_store_bytes}" \
    run_reported_pytest "$@"
}

run_owner_ray_pytest() {
  if ((external_ray_cleanup)); then
    VANE_TEST_EXTERNAL_RAY_CLUSTER_CLEANUP=1 run_ray_pytest "$@"
  else
    run_ray_pytest "$@"
  fi
}

stop_owner_ray_processes() {
  local cleanup_command=(ray stop --force)
  local cleanup_output
  if command -v timeout >/dev/null 2>&1; then
    cleanup_command=(
      timeout
      --signal=TERM
      --kill-after=10s
      30s
      "${cleanup_command[@]}"
    )
  fi
  if cleanup_output="$("${cleanup_command[@]}" 2>&1)"; then
    return 0
  else
    local cleanup_status=$?
    printf '%s\n' "$cleanup_output" >&2
    return "$cleanup_status"
  fi
}

collect_nodeids() {
  local marker="$1"
  local collection
  collection="$({
    PYTEST_ADDOPTS= run_pytest \
      -o addopts= \
      --collect-only \
      -q \
      -m "$marker" \
      tests/fast
  } 2>&1)" || {
    printf '%s\n' "$collection" >&2
    return 1
  }
  printf '%s\n' "$collection" | sed -n '/^tests\/fast\/.*::/p'
}

run_non_ray_tests() {
  local marker="not external_service and not real_ray${gpu_marker}"
  if ((non_ray_shard_count == 1)); then
    run_reported_pytest \
      "non-ray" \
      -m "$marker" \
      tests/fast
    return
  fi

  local collection
  collection="$(collect_nodeids "$marker")"
  if [[ -z "$collection" ]]; then
    printf '%s\n' "No non-Ray fast tests were collected" >&2
    return 1
  fi
  local nodeids=()
  mapfile -t nodeids <<<"$collection"

  local selected_nodeids=()
  local position
  for ((position = non_ray_shard_index; position < ${#nodeids[@]}; position += non_ray_shard_count)); do
    selected_nodeids+=("${nodeids[position]}")
  done
  if ((${#selected_nodeids[@]} == 0)); then
    printf 'Non-Ray shard %d of %d selected no tests; reduce the shard count\n' \
      "$((non_ray_shard_index + 1))" "$non_ray_shard_count" >&2
    return 1
  fi

  local report_name
  printf -v report_name \
    'non-ray-%02d-of-%02d' \
    "$((non_ray_shard_index + 1))" \
    "$non_ray_shard_count"
  run_reported_pytest \
    "$report_name" \
    -m "$marker" \
    "${selected_nodeids[@]}"
}

run_shared_ray_tests() {
  run_ray_pytest \
    "shared-ray" \
    -m "not external_service and real_ray and not ray_cluster_owner${gpu_marker}" \
    tests/fast
}

run_owner_ray_tests() {
  local marker="not external_service and real_ray and ray_cluster_owner${gpu_marker}"
  local collection
  collection="$(collect_nodeids "$marker")"
  if [[ -z "$collection" ]]; then
    printf '%s\n' "No real-Ray cluster-owner tests were collected" >&2
    return 1
  fi
  local owner_nodeids=()
  mapfile -t owner_nodeids <<<"$collection"

  local owner_status=0
  local owner_index=0
  local nodeid
  for nodeid in "${owner_nodeids[@]}"; do
    owner_index=$((owner_index + 1))
    local report_name
    printf -v report_name 'owner-ray-%02d' "$owner_index"
    if ! run_owner_ray_pytest \
      "$report_name" \
      -m "$marker" \
      "$nodeid"; then
      owner_status=1
    fi
    if ((external_ray_cleanup)); then
      if ! stop_owner_ray_processes; then
        printf 'Ray cleanup failed after owner test %s\n' "$nodeid" >&2
        owner_status=1
      fi
    fi
  done
  return "$owner_status"
}

case "$phase" in
  all)
    # Keep the real Ray runtime out of the long-lived non-Ray pytest process.
    run_non_ray_tests
    run_shared_ray_tests
    run_owner_ray_tests
    ;;
  non-ray)
    run_non_ray_tests
    ;;
  shared-ray)
    run_shared_ray_tests
    ;;
  owner-ray)
    run_owner_ray_tests
    ;;
esac
