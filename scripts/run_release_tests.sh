#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Keep this gate limited to tests that exercise the supported base installation.
# Optional provider, benchmark, compatibility, and external-service suites run
# separately because they need additional dependencies or infrastructure.
release_tests=(
  tests/fast/test_package_metadata.py
  tests/fast/test_ray_test_profile.py
  tests/fast/test_transformers_provider_security.py
  tests/fast/test_vane_config.py
  tests/fast/test_expression_udf_contracts.py
  tests/fast/test_local_e2e.py
  tests/fast/test_ray_cpp_bindings.py
  tests/fast/test_ray_remote_exceptions.py
  tests/fast/test_ray_result_contract.py
  tests/fast/test_fte_production_readiness.py
)

pytest_args=(
  --import-mode=importlib
  -o pythonpath=tests
)

ray_object_store_bytes="$(
  PYTHONPATH="tests${PYTHONPATH:+:${PYTHONPATH}}" \
    python -c "from ray_test_profile import ray_test_object_store_bytes; print(ray_test_object_store_bytes())"
)"

# Let the non-Ray process release all Python/native state before a fresh pytest
# process starts the real Ray runtime.
python -m pytest \
  "${pytest_args[@]}" \
  -m "not external_service and not real_ray" \
  "${release_tests[@]}"

VANE_TEST_RAY_OBJECT_STORE_BYTES="${ray_object_store_bytes}" \
  python -m pytest \
  "${pytest_args[@]}" \
  -m "not external_service and real_ray" \
  "${release_tests[@]}"
