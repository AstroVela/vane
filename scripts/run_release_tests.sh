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
  "$project_root/tests/fast/test_doris_datasink.py"
  "$project_root/tests/fast/test_extension_catalog.py"
  "$project_root/tests/fast/test_milvus_datasink.py"
  "$project_root/tests/fast/test_qdrant_datasink.py"
  "$project_root/tests/fast/test_package_metadata.py"
  "$project_root/tests/fast/test_python_filesystem_concurrency.py"
  "$project_root/tests/fast/test_python_callback_entry.py"
  "$project_root/tests/fast/test_ray_test_profile.py"
  "$project_root/tests/fast/test_transformers_provider_security.py"
  "$project_root/tests/fast/test_vane_config.py"
  "$project_root/tests/fast/test_expression_udf_contracts.py"
  "$project_root/tests/fast/test_local_e2e.py"
  "$project_root/tests/fast/test_ray_cpp_bindings.py"
  "$project_root/tests/fast/test_ray_diagnostics.py"
  "$project_root/tests/fast/test_ray_remote_exceptions.py"
  "$project_root/tests/fast/test_ray_result_contract.py"
)

# Keep the local backpressure acceptance cases in the non-Ray process. These
# use real native plans/subprocesses and small shared-memory budgets, without
# optional models or GPUs. The full matrices remain in the fast-test shards.
local_runtime_tests=(
  "$project_root/tests/fast/test_local_query_runtime.py"
  "$project_root/tests/fast/test_local_runtime_baseline.py"
  "$project_root/tests/fast/test_udf_data_wait_native.py"
  "$project_root/tests/fast/test_udf_data_wait_progress.py"
  "$project_root/tests/fast/test_udf_data_wait_scans.py::test_native_scans_execute_with_byte_wait"
  "$project_root/tests/fast/test_udf_data_wait.py::test_older_byte_waiter_keeps_its_turn_among_newer_ordinary_work"
  "$project_root/tests/fast/test_udf_local_request_cancellation.py::test_cancel_mixed_native_pipeline_and_reuse_registered_model"
  "$project_root/tests/fast/test_udf_local_request_cancellation.py::test_failed_native_start_retires_query_before_releasing_its_plan"
  "$project_root/tests/fast/test_udf_process.py::test_native_dispatcher_notification_cannot_cross_the_wait_boundary"
  "$project_root/tests/fast/test_ray_udf_plan_replay.py::test_execute_native_subprocess_udf_reports_admission_task_stats"
  "$project_root/tests/fast/test_local_result_delivery_native.py::test_native_managed_results_release_request_slots_and_preserve_exported_views"
  "$project_root/tests/fast/test_local_result_delivery_native.py::test_queued_managed_request_cannot_reserve_the_ready_requests_result_slot"
  "$project_root/tests/fast/test_udf_task_admission.py::test_common_admission_reentrant_wakeup_and_exact_input_handoff"
  "$project_root/tests/fast/test_udf_task_admission.py::test_common_admission_callback_removal_keeps_the_ready_lease"
  "$project_root/tests/fast/test_udf_task_admission.py::test_common_admission_close_fences_a_late_grant"
  "$project_root/tests/fast/test_udf_data_lease.py::test_shared_owner_keeps_output_through_task_completion_and_forward_transitions"
  "$project_root/tests/fast/test_udf_data_lease.py::test_shared_owner_concurrent_release_returns_capacity_once"
  "$project_root/tests/fast/test_udf_data_lease.py::test_shared_owner_keeps_accounting_until_a_failed_release_is_retried"
  "$project_root/tests/fast/test_operator_byte_budget.py"
  "$project_root/tests/fast/test_query_resource_manager.py::test_local_envelopes_and_ray_use_the_same_full_reservation_partition"
  "$project_root/tests/fast/test_query_resource_manager.py::test_common_byte_accounting_preserves_backend_reservation_baselines"
)
for shape in unnest_expression sort_payload topn hash_min join_build join_probe; do
  local_runtime_tests+=(
    "$project_root/tests/fast/test_udf_data_wait_native_owners.py::test_consumed_native_inputs_release_byte_capacity[$shape-True-True]"
  )
done
for limited in False True; do
  local_runtime_tests+=(
    "$project_root/tests/fast/test_udf_data_wait_batching.py::test_byte_pressure_drains_a_short_downstream_batch[2-2-False-$limited]"
    "$project_root/tests/fast/test_udf_local_request_native.py::test_failed_output_grant_cleanup_retains_native_request[False-request-$limited-default]"
    "$project_root/tests/fast/test_udf_local_request_native.py::test_failed_output_grant_cleanup_retains_native_request[False-runtime-$limited-byte_wait]"
  )
done

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
  "${release_tests[@]}" \
  "${local_runtime_tests[@]}"

python -m pytest \
  "${pytest_args[@]}" \
  -m "not external_service and real_ray" \
  "${release_tests[@]}"
