#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ray_object_store_bytes="$(
  PYTHONPATH="tests${PYTHONPATH:+:${PYTHONPATH}}" \
    python -c "from ray_test_profile import ray_test_object_store_bytes; print(ray_test_object_store_bytes())"
)"

run_ray_pytest() {
  VANE_TEST_RAY_OBJECT_STORE_BYTES="${ray_object_store_bytes}" \
    python -m pytest "$@"
}

# The pure suite gets a full process lifetime, so its Python/native heap is
# returned to the OS before any Ray control-plane process is created.
python -m pytest \
  -m "not external_service and not real_ray" \
  tests/fast

# All tests in this shard connect to one session-owned Ray cluster.
run_ray_pytest \
  -m "not external_service and real_ray and not ray_cluster_owner" \
  tests/fast

# A cluster owner may start, stop, or kill its own Ray control plane. Collect
# the checked markers once, then give every owner a fresh pytest OS process.
owner_collection="$({
  python -m pytest \
    -o addopts= \
    --collect-only \
    -q \
    -m "not external_service and real_ray and ray_cluster_owner" \
    tests/fast
} 2>&1)" || {
  printf '%s\n' "${owner_collection}" >&2
  exit 1
}

mapfile -t owner_nodeids < <(printf '%s\n' "${owner_collection}" | sed -n '/^tests\/fast\/.*::/p')
if ((${#owner_nodeids[@]} == 0)); then
  printf '%s\n' "No real-Ray cluster-owner tests were collected" >&2
  exit 1
fi

owner_status=0
for nodeid in "${owner_nodeids[@]}"; do
  if ! run_ray_pytest \
    -m "not external_service and real_ray and ray_cluster_owner" \
    "${nodeid}"; then
    owner_status=1
  fi
done

exit "${owner_status}"
