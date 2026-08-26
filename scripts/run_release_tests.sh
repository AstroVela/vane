#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
site_packages="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
test_workdir="$(mktemp -d "${TMPDIR:-/tmp}/vane-release-tests.XXXXXX")"
cleanup_test_workdir() {
  rm -rf -- "$test_workdir"
}
trap cleanup_test_workdir EXIT
cd "$test_workdir"

export PYTHONSAFEPATH=1
export PYTHONPATH="${site_packages}:${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export VANE_FAST_TEST_ARTIFACT_MODE=1

# Keep this gate limited to tests that exercise the supported base installation.
# Optional provider, benchmark, compatibility, and external-service suites run
# separately because they need additional dependencies or infrastructure.
release_tests=(
  "$project_root/tests/fast/test_ai_release_contracts.py"
  "$project_root/tests/fast/test_datasink.py"
  "$project_root/tests/fast/test_milvus_datasink.py"
  "$project_root/tests/fast/test_turbopuffer_datasink.py"
  "$project_root/tests/fast/test_package_metadata.py"
  "$project_root/tests/fast/test_ray_test_profile.py"
  "$project_root/tests/fast/test_transformers_provider_security.py"
  "$project_root/tests/fast/test_vane_config.py"
  "$project_root/tests/fast/test_expression_udf_contracts.py"
  "$project_root/tests/fast/test_local_e2e.py"
  "$project_root/tests/fast/test_ray_cpp_bindings.py"
  "$project_root/tests/fast/test_ray_remote_exceptions.py"
  "$project_root/tests/fast/test_ray_result_contract.py"
)

pytest_args=(
  -c "$project_root/pyproject.toml"
  --rootdir="$project_root"
  --import-mode=importlib
  -o "pythonpath=$project_root/tests"
)

# Let the non-Ray process release all Python/native state before a fresh pytest
# process starts the real Ray runtime.
python -m pytest \
  "${pytest_args[@]}" \
  -m "not external_service and not real_ray" \
  "${release_tests[@]}"

python -m pytest \
  "${pytest_args[@]}" \
  -m "not external_service and real_ray" \
  "${release_tests[@]}"
