#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Anchor diagnostics to the caller before entering a disposable test directory.
if [[ -n "${VANE_TEST_DIAGNOSTICS_DIR:-}" ]]; then
  VANE_TEST_DIAGNOSTICS_DIR="$(python -c 'import os; print(os.path.abspath(os.environ["VANE_TEST_DIAGNOSTICS_DIR"]))')"
  export VANE_TEST_DIAGNOSTICS_DIR
fi
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
  "$project_root/tests/fast/test_ai_client_config.py"
  "$project_root/tests/fast/test_ai_embedding_requests.py"
  "$project_root/tests/fast/test_ai_image_embedding.py"
  "$project_root/tests/fast/test_ai_video_embedding.py"
  "$project_root/tests/fast/test_ai_audio_embedding.py"
  "$project_root/tests/fast/test_clap.py"
  "$project_root/tests/fast/test_cosmos_embed1.py"
  "$project_root/tests/fast/test_datasink.py"
  "$project_root/tests/fast/test_doris_datasink.py"
  "$project_root/tests/fast/test_extension_catalog.py"
  "$project_root/tests/fast/test_execution_fragment_graph.py"
  "$project_root/tests/fast/test_execution_submission.py"
  "$project_root/tests/fast/test_execution_cutover.py"
  "$project_root/tests/fast/test_execution_acceptance.py"
  "$project_root/tests/fast/test_ray_execution_acceptance.py"
  "$project_root/tests/fast/test_execution_benchmark.py"
  "$project_root/tests/fast/test_ray_execution_benchmark.py"
  "$project_root/tests/fast/test_milvus_datasink.py"
  "$project_root/tests/fast/test_native_fragment_compiler.py"
  "$project_root/tests/fast/test_analytical_fragment_compiler.py"
  "$project_root/tests/fast/test_analytical_exchange.py"
  "$project_root/tests/fast/test_worker_resources.py"
  "$project_root/tests/fast/test_ray_analytical_execution.py"
  "$project_root/tests/fast/test_qdrant_datasink.py"
  "$project_root/tests/fast/test_package_metadata.py"
  "$project_root/tests/fast/test_python_filesystem_concurrency.py"
  "$project_root/tests/fast/test_python_callback_entry.py"
  "$project_root/tests/fast/test_python_parameter_callbacks.py"
  "$project_root/tests/fast/test_python_binding_callbacks.py"
  "$project_root/tests/fast/test_query_execution_options.py"
  "$project_root/tests/fast/test_query_result_runtime.py"
  "$project_root/tests/fast/test_direct_exchange.py"
  "$project_root/tests/fast/test_direct_flight.py"
  "$project_root/tests/fast/test_materialized_exchange.py"
  "$project_root/tests/fast/test_fte_sources.py"
  "$project_root/tests/fast/test_fte_store.py"
  "$project_root/tests/fast/test_ray_recovery_runtime.py"
  "$project_root/tests/fast/test_pipelined_plan.py"
  "$project_root/tests/fast/test_ray_pipelined.py"
  "$project_root/tests/fast/test_ray_query_service.py"
  "$project_root/tests/fast/test_server_sessions.py"
  "$project_root/tests/fast/test_flight_server.py"
  "$project_root/tests/fast/test_flight_proxy_isolation.py"
  "$project_root/tests/fast/test_ray_server_sessions.py"
  "$project_root/tests/fast/test_server_queries.py"
  "$project_root/tests/fast/test_ray_server_queries.py"
  "$project_root/tests/fast/test_ray_server_acceptance.py"
  "$project_root/tests/fast/test_server_cli.py"
  "$project_root/tests/fast/test_result_service.py"
  "$project_root/tests/fast/test_ray_test_profile.py"
  "$project_root/tests/fast/test_transformers_provider_security.py"
  "$project_root/tests/fast/test_vane_config.py"
  "$project_root/tests/fast/test_expression_udf_contracts.py"
  "$project_root/tests/fast/test_ray_remote_exceptions.py"
)

# Keep the local backpressure acceptance cases in the non-Ray process. These
# use real native plans/subprocesses and small shared-memory budgets, without
# optional models or GPUs. The full matrices remain in the fast-test shards.
local_runtime_tests=(
  "$project_root/tests/fast/test_local_query_models.py"
  "$project_root/tests/fast/test_local_query_runtime.py"
  "$project_root/tests/fast/test_local_query_streaming.py"
  "$project_root/tests/fast/test_result_delivery_capacity.py"
  "$project_root/tests/fast/test_local_query_results.py::test_full_result_slot_refuses_before_udf_execution_without_leaking_request"
  "$project_root/tests/fast/test_local_query_results.py::test_delivery_byte_refusal_never_replays_a_model_call"
  "$project_root/tests/fast/test_local_query_results.py::test_managed_native_queries_preserve_parameters_and_nested_schema"
  "$project_root/tests/fast/test_local_query_results.py::test_queued_native_query_does_not_occupy_result_capacity"
  "$project_root/tests/fast/test_local_query_results.py::test_exported_views_keep_delivery_bytes_after_connection_close"
  "$project_root/tests/fast/test_local_query_results.py::test_result_cleanup_failure_keeps_runtime_retry_owner"
  "$project_root/tests/fast/test_local_serving_acceptance.py::test_cpu_serving_acceptance_uses_one_runtime_and_returns_to_baseline[False]"
  "$project_root/tests/fast/test_local_serving_soak.py"
  "$project_root/tests/fast/test_udf_worker_metrics.py"
  "$project_root/tests/fast/test_udf_model_resources.py"
  "$project_root/tests/fast/test_udf_local_gpu.py"
  "$project_root/tests/fast/test_udf_local_gpu_admission.py"
  "$project_root/tests/fast/test_local_query_gpu.py"
  "$project_root/tests/fast/test_udf_data_wait_progress.py"
  "$project_root/tests/fast/test_udf_data_wait.py::test_older_byte_waiter_keeps_its_turn_among_newer_ordinary_work"
  "$project_root/tests/fast/test_udf_process.py::test_native_dispatcher_notification_cannot_cross_the_wait_boundary"
  "$project_root/tests/fast/test_udf_task_admission.py::test_common_admission_reentrant_wakeup_and_exact_input_handoff"
  "$project_root/tests/fast/test_udf_task_admission.py::test_common_admission_callback_removal_keeps_the_ready_lease"
  "$project_root/tests/fast/test_udf_task_admission.py::test_common_admission_close_fences_a_late_grant"
  "$project_root/tests/fast/test_udf_data_lease.py::test_shared_owner_keeps_output_through_task_completion_and_forward_transitions"
  "$project_root/tests/fast/test_udf_data_lease.py::test_shared_owner_concurrent_release_returns_capacity_once"
  "$project_root/tests/fast/test_udf_data_lease.py::test_shared_owner_keeps_accounting_until_a_failed_release_is_retried"
  "$project_root/tests/fast/test_operator_byte_budget.py"
)
for shape in unnest_expression sort_payload topn hash_min join_build join_probe; do
  local_runtime_tests+=(
    "$project_root/tests/fast/test_udf_data_wait_native_owners.py::test_consumed_native_inputs_release_byte_capacity[$shape-True-True]"
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
  -m "not external_service and real_ray and not ray_cluster_owner" \
  "${release_tests[@]}"

# Cluster-owner media/provider checks require optional signed artifacts and run
# separately in their CI jobs and the fast-test owner-ray phase.
