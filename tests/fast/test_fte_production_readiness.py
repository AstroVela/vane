import json

from multimodal_inference_benchmarks import check_fte_production_readiness as readiness


def _ready_env(shuffle_dir):
    return {
        "VANE_FTE_SPLIT_QUEUE_MAX_BUFFERED_SPLITS": "256",
        "VANE_FTE_TASK_UPDATE_MAX_SPLITS": "512",
        "VANE_FTE_TASK_UPDATE_MAX_PAYLOAD_BYTES": "1048576",
        "VANE_SHUFFLE_LOCAL_DIRS": str(shuffle_dir),
    }


def _write_full_matrix_manifest(path, suites=readiness.DEFAULT_SUITES, *, full=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for suite in suites:
            handle.write(
                json.dumps(
                    {
                        "event": "comparison_success",
                        "suite": suite,
                        "full": full,
                        "reference": {"row_count": 1, "hash_sum": "1", "hash_xor": "1"},
                        "chaos": {"row_count": 1, "hash_sum": "1", "hash_xor": "1"},
                    }
                )
                + "\n"
            )


def test_fte_production_readiness_passes_ready_environment(tmp_path):
    shuffle_dir = tmp_path / "shuffle"
    shuffle_dir.mkdir()
    manifest_path = tmp_path / "manifest.jsonl"
    _write_full_matrix_manifest(manifest_path)

    checks = readiness.evaluate_readiness(
        env=_ready_env(shuffle_dir),
        manifest_path=manifest_path,
        require_full_matrix=True,
        min_shuffle_free_bytes=0,
    )

    assert readiness.overall_status(checks) == "PASS"
    assert {check.status for check in checks} == {"PASS"}


def test_fte_production_readiness_reports_required_failures():
    checks = readiness.evaluate_readiness(
        env={},
        manifest_path=None,
        require_full_matrix=True,
        min_shuffle_free_bytes=0,
    )
    failures = {check.name for check in checks if check.status == "FAIL"}

    assert readiness.overall_status(checks) == "FAIL"
    assert "benchmark.full_matrix" in failures


def test_fte_production_readiness_warns_for_optional_manifest_without_requirement(tmp_path):
    shuffle_dir = tmp_path / "shuffle"
    shuffle_dir.mkdir()

    checks = readiness.evaluate_readiness(
        env=_ready_env(shuffle_dir),
        manifest_path=None,
        require_full_matrix=False,
        min_shuffle_free_bytes=0,
    )

    assert readiness.overall_status(checks) == "WARN"
    assert [check.status for check in checks if check.name == "benchmark.full_matrix"] == ["WARN"]


def test_fte_production_readiness_rejects_smoke_manifest_when_full_required(tmp_path):
    shuffle_dir = tmp_path / "shuffle"
    shuffle_dir.mkdir()
    manifest_path = tmp_path / "manifest.jsonl"
    _write_full_matrix_manifest(manifest_path, full=False)

    checks = readiness.evaluate_readiness(
        env=_ready_env(shuffle_dir),
        manifest_path=manifest_path,
        require_full_matrix=True,
        min_shuffle_free_bytes=0,
    )

    assert readiness.overall_status(checks) == "FAIL"
    assert [check.details["smoke_suites"] for check in checks if check.name == "benchmark.full_matrix"] == [
        list(readiness.DEFAULT_SUITES)
    ]
