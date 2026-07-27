# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from multimodal_inference_benchmarks import check_fte_production_readiness as readiness


def _ready_env(shuffle_dir):
    return {
        "VANE_FTE_SPLIT_QUEUE_MAX_BUFFERED_SPLITS": "256",
        "VANE_FTE_TASK_UPDATE_MAX_SPLITS": "512",
        "VANE_FTE_TASK_UPDATE_MAX_PAYLOAD_BYTES": "1048576",
        "VANE_SHUFFLE_LOCAL_DIRS": str(shuffle_dir),
    }


def test_fte_production_readiness_passes_ready_environment(tmp_path):
    shuffle_dir = tmp_path / "shuffle"
    shuffle_dir.mkdir()

    checks = readiness.evaluate_readiness(
        env=_ready_env(shuffle_dir),
        min_shuffle_free_bytes=0,
    )

    assert readiness.overall_status(checks) == "PASS"
    assert {check.status for check in checks} == {"PASS"}


def test_fte_production_readiness_warns_for_recommended_settings():
    checks = readiness.evaluate_readiness(
        env={},
        min_shuffle_free_bytes=0,
    )

    assert readiness.overall_status(checks) == "WARN"
    warnings = {check.name for check in checks if check.status == "WARN"}
    assert warnings == {
        "backpressure.vane_fte_split_queue_max_buffered_splits",
        "backpressure.vane_fte_task_update_max_splits",
        "backpressure.vane_fte_task_update_max_payload_bytes",
        "shuffle.path",
    }


def test_fte_production_readiness_rejects_invalid_retry_configuration(tmp_path):
    shuffle_dir = tmp_path / "shuffle"
    shuffle_dir.mkdir()
    env = _ready_env(shuffle_dir)
    env["VANE_FTE_RETRY_INITIAL_DELAY_S"] = "0"
    env["VANE_FTE_RETRY_MAX_DELAY_S"] = "invalid"

    checks = readiness.evaluate_readiness(
        env=env,
        min_shuffle_free_bytes=0,
    )

    assert readiness.overall_status(checks) == "FAIL"
    failures = {check.name for check in checks if check.status == "FAIL"}
    assert failures == {
        "retry.vane_fte_retry_initial_delay_s",
        "retry.vane_fte_retry_max_delay_s",
    }


def test_fte_production_readiness_rejects_missing_shuffle_parent(tmp_path):
    checks = readiness.evaluate_readiness(
        env=_ready_env(tmp_path / "missing" / "shuffle"),
        min_shuffle_free_bytes=0,
    )

    assert readiness.overall_status(checks) == "FAIL"
    failures = [check for check in checks if check.status == "FAIL"]
    assert [check.name for check in failures] == ["shuffle.path.0"]
    assert failures[0].message == "shuffle path parent does not exist"
