import json
import sys
import types

import pytest

from multimodal_inference_benchmarks import run_fte_release_gate as release_gate


def _read_report(run_root):
    return json.loads((run_root / "release_gate_report.json").read_text(encoding="utf-8"))


def _write_matrix_manifest(manifest_path, suites=("image", "document")):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for suite in suites:
            handle.write(
                json.dumps(
                    {
                        "event": "comparison_success",
                        "suite": suite,
                        "full": False,
                        "reference": {"row_count": 1, "hash_sum": "1", "hash_xor": "1"},
                        "chaos": {"row_count": 1, "hash_sum": "1", "hash_xor": "1"},
                    }
                )
                + "\n"
            )


def test_fte_release_gate_dry_run_writes_plan(monkeypatch, tmp_path):
    run_root = tmp_path / "gate"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fte_release_gate.py",
            "--run-root",
            str(run_root),
            "--dry-run",
            "--skip-readiness",
            "--suite",
            "image",
            "--suite",
            "document",
        ],
    )
    monkeypatch.setattr(
        release_gate.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dry-run must not execute subprocesses")),
    )

    release_gate.main()

    report = _read_report(run_root)
    assert report["overall"] == "PASS"
    assert [step["name"] for step in report["steps"]] == [
        "fast-regression",
        "full-matrix-smoke",
        "matrix-manifest",
    ]
    assert {step["status"] for step in report["steps"]} == {"SKIPPED"}
    assert (run_root / "release_gate_report.md").exists()


def test_fte_release_gate_runs_fast_tests_and_matrix(monkeypatch, tmp_path):
    run_root = tmp_path / "gate"
    calls = []
    monkeypatch.setenv("INPUT_PATH", "/tmp/should-not-leak")

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        if any("run_fte_full_matrix.py" in item for item in command):
            run_root_index = command.index("--run-root") + 1
            _write_matrix_manifest(tmp_path / "gate" / "full_matrix" / "manifest.jsonl")
            assert command[run_root_index] == str(run_root / "full_matrix")
            assert "--smoke-limit" in command
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(release_gate.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fte_release_gate.py",
            "--run-root",
            str(run_root),
            "--skip-readiness",
            "--suite",
            "image",
            "--suite",
            "document",
            "--fast-test",
            "tests/fast/test_fte_release_gate.py",
        ],
    )

    release_gate.main()

    assert len(calls) == 2
    fast_command, fast_env = calls[0]
    matrix_command, _matrix_env = calls[1]
    assert fast_command[:4] == [sys.executable, "-m", "pytest", "-q"]
    assert "tests/fast/test_fte_release_gate.py" in fast_command
    assert "INPUT_PATH" not in fast_env
    assert matrix_command[1] == "multimodal_inference_benchmarks/run_fte_full_matrix.py"
    report = _read_report(run_root)
    assert report["overall"] == "PASS"
    assert [step["status"] for step in report["steps"]] == ["PASS", "PASS", "PASS"]


def test_fte_release_gate_preflight_ignores_missing_manifest(monkeypatch, tmp_path):
    shuffle_dir = tmp_path / "shuffle"
    shuffle_dir.mkdir()
    monkeypatch.setenv("VANE_FTE_SPLIT_QUEUE_MAX_BUFFERED_SPLITS", "256")
    monkeypatch.setenv("VANE_FTE_TASK_UPDATE_MAX_SPLITS", "512")
    monkeypatch.setenv("VANE_FTE_TASK_UPDATE_MAX_PAYLOAD_BYTES", "1048576")
    monkeypatch.setenv("VANE_SHUFFLE_LOCAL_DIRS", str(shuffle_dir))

    step = release_gate._readiness_step(
        name="readiness-preflight",
        manifest_path=None,
        require_full_matrix=False,
        strict=False,
        dry_run=False,
    )

    assert step.status == "PASS"
    assert all(check["name"] != "benchmark.full_matrix" for check in step.details["checks"])


def test_fte_release_gate_records_fast_failure(monkeypatch, tmp_path):
    run_root = tmp_path / "gate"

    monkeypatch.setattr(
        release_gate.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(returncode=7),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_fte_release_gate.py",
            "--run-root",
            str(run_root),
            "--skip-readiness",
            "--skip-matrix",
            "--fast-test",
            "tests/fast/test_fte_release_gate.py",
        ],
    )

    with pytest.raises(RuntimeError, match="fast regression failed"):
        release_gate.main()

    report = _read_report(run_root)
    assert report["overall"] == "FAIL"
    assert report["steps"][0]["name"] == "fast-regression"
    assert report["steps"][0]["returncode"] == 7
    assert report["steps"][0]["status"] == "FAIL"
