# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_audio_parity.py"


def _write_result(root, label, result):
    directory = root / label
    directory.mkdir()
    (directory / "results.json").write_text(json.dumps(result))


def _manifest_digest(manifest):
    data = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    "left_engine,right_engine", [("daft", "vane"), ("vane", "vane"), ("daft", "daft"), (None, "daft")]
)
def test_audio_audit_rejects_unsupported_engine_order(tmp_path, left_engine, right_engine):
    _write_result(tmp_path, "left-run", {"engine": left_engine})
    _write_result(tmp_path, "right-run", {"engine": right_engine})
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "compare", str(tmp_path), "--left", "left-run", "--right", "right-run"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--left to name a Vane run and --right to name a Daft run" in result.stderr
    assert "Traceback" not in result.stderr
    assert not list(tmp_path.glob("comparison-*.json"))


@pytest.fixture
def audit_inputs(tmp_path):
    manifest = {
        "seed": 20260908,
        "cases": [
            {
                "id": "input",
                "file": "input.wav",
                "sha256": hashlib.sha256(b"original input").hexdigest(),
                "frames": 2,
                "channels": 1,
                "sample_rate": 8000,
                "targets": [8000],
            }
        ],
    }
    identity = {"input_manifest": manifest, "input_manifest_sha256": _manifest_digest(manifest), "complete": True}

    def save(name, value):
        path = tmp_path / name
        np.save(path, value, allow_pickle=False)
        return {
            "array": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    python_array = save("python.npy", np.array([[1.0], [2.0]], dtype=np.float64))
    daft_array = save("daft.npy", np.array([1.0, 2.0], dtype=np.float64))
    metadata = {
        "value": {
            "sample_rate": 8000,
            "channels": 1,
            "frames": 2,
            "duration": 2 / 8000,
            "format": "WAV",
            "subtype": "PCM_16",
        }
    }
    _write_result(
        tmp_path,
        "candidate",
        {
            "engine": "vane",
            **identity,
            "resamples": {
                "input_8000": {
                    name: python_array
                    for name in (
                        "value",
                        "sql_python",
                        "value_native_connection",
                        "sql_native",
                        "sql_python_expression",
                        "sql_native_expression",
                        "librosa_axis0_reference",
                    )
                }
            },
            "files": {
                "input": {
                    "input_sha256": manifest["cases"][0]["sha256"],
                    "decode": python_array,
                    **{
                        name: metadata
                        for name in (
                            "metadata_value",
                            "metadata_sql_native",
                            "metadata_sql_python",
                            "metadata_expression_python",
                            "metadata_expression_native",
                        )
                    },
                }
            },
        },
    )
    _write_result(
        tmp_path,
        "reference",
        {
            "engine": "daft",
            **identity,
            "resamples": {"input_8000": {"value": daft_array, "expression": daft_array}},
            "files": {
                "input": {
                    "input_sha256": manifest["cases"][0]["sha256"],
                    "decode": daft_array,
                    "metadata_value": metadata,
                    "metadata_expression": metadata,
                }
            },
        },
    )
    return tmp_path


def _compare(root):
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "compare", str(root), "--left", "candidate", "--right", "reference"],
        capture_output=True,
        text=True,
    )


def test_audio_audit_retains_custom_labels_and_array_sides(audit_inputs):
    result = _compare(audit_inputs)
    assert result.returncode == 0, result.stderr
    comparison = json.loads((audit_inputs / "comparison-candidate-reference.json").read_text())
    assert (comparison["left"], comparison["right"]) == ("candidate", "reference")
    wave = comparison["resamples"]["input_8000"]["python_vs_daft"]
    assert wave["left_shape"] == [2, 1] and wave["right_shape"] == [2]
    assert wave["equal_after_mono_normalization"]
    assert comparison["metadata"]["input"]["shared_python_daft_fields_equal"]


@pytest.mark.parametrize("filename", ["python.npy", "daft.npy"])
def test_audio_audit_rejects_arrays_changed_after_capture(audit_inputs, filename):
    np.save(audit_inputs / filename, np.array([42.0, 99.0], dtype=np.float64))
    result = _compare(audit_inputs)
    assert result.returncode == 1
    assert "array SHA-256 does not match the recorded run" in result.stderr
    assert filename in result.stderr and "Traceback" in result.stderr
    assert not list(audit_inputs.glob("comparison-*.json"))


@pytest.mark.parametrize("field,value", [("sha256", None), ("shape", [2]), ("dtype", "float32")])
def test_audio_audit_rejects_inconsistent_array_records(audit_inputs, field, value):
    path = audit_inputs / "candidate/results.json"
    record = json.loads(path.read_text())
    record["resamples"]["input_8000"]["value"][field] = value
    path.write_text(json.dumps(record))
    result = _compare(audit_inputs)
    assert result.returncode == 1
    assert "does not match the recorded run" in result.stderr and "Traceback" in result.stderr
    assert not list(audit_inputs.glob("comparison-*.json"))


@pytest.mark.parametrize("kind", ["corrupt", "object"])
def test_audio_audit_preserves_array_data_error_tracebacks(audit_inputs, kind):
    array = audit_inputs / "python.npy"
    if kind == "corrupt":
        array.write_bytes(b"invalid NumPy file")
    else:
        np.save(array, np.array([{"unexpected": "object"}], dtype=object), allow_pickle=True)
    path = audit_inputs / "candidate/results.json"
    record = json.loads(path.read_text())
    record["resamples"]["input_8000"]["value"]["sha256"] = hashlib.sha256(array.read_bytes()).hexdigest()
    path.write_text(json.dumps(record))
    result = _compare(audit_inputs)
    assert result.returncode == 1
    assert "ValueError:" in result.stderr and "Traceback" in result.stderr
    assert "usage:" not in result.stderr and "array SHA-256" not in result.stderr
    assert not list(audit_inputs.glob("comparison-*.json"))


@pytest.mark.parametrize("change", ["encoded_input", "generation_plan"])
def test_audio_audit_rejects_different_recorded_corpora(audit_inputs, change):
    path = audit_inputs / "reference/results.json"
    record = json.loads(path.read_text())
    if change == "encoded_input":
        checksum = hashlib.sha256(b"regenerated input").hexdigest()
        record["input_manifest"]["cases"][0]["sha256"] = checksum
        record["files"]["input"]["input_sha256"] = checksum
    else:
        record["input_manifest"]["seed"] += 1
    record["input_manifest_sha256"] = _manifest_digest(record["input_manifest"])
    path.write_text(json.dumps(record))
    result = _compare(audit_inputs)
    assert result.returncode == 1 and "Input manifests differ" in result.stderr
    assert not list(audit_inputs.glob("comparison-*.json"))


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("input_manifest", None, "no recorded input manifest"),
        ("input_manifest_sha256", "invalid", "input manifest SHA-256 mismatch"),
        ("complete", False, "incomplete audio audit run"),
    ],
)
def test_audio_audit_rejects_missing_or_invalid_run_identity(audit_inputs, field, value, message):
    path = audit_inputs / "candidate/results.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    result = _compare(audit_inputs)
    assert result.returncode == 1 and message in result.stderr
    assert not list(audit_inputs.glob("comparison-*.json"))


@pytest.mark.parametrize("section", ["files", "resamples"])
def test_audio_audit_rejects_partial_result_coverage(audit_inputs, section):
    path = audit_inputs / "candidate/results.json"
    record = json.loads(path.read_text())
    record[section].clear()
    path.write_text(json.dumps(record))
    result = _compare(audit_inputs)
    assert result.returncode == 1 and "results do not cover the recorded input manifest" in result.stderr
    assert not list(audit_inputs.glob("comparison-*.json"))


def test_audio_audit_checks_each_recorded_input_digest(audit_inputs):
    path = audit_inputs / "candidate/results.json"
    record = json.loads(path.read_text())
    record["files"]["input"]["input_sha256"] = "incorrect"
    path.write_text(json.dumps(record))
    result = _compare(audit_inputs)
    assert result.returncode == 1 and "recorded input digest mismatch for input" in result.stderr


def test_audio_audit_keeps_old_run_identity_after_shared_corpus_changes(audit_inputs):
    (audit_inputs / "manifest.json").write_text('{"cases": []}')
    (audit_inputs / "input.wav").write_bytes(b"regenerated input")
    result = _compare(audit_inputs)
    assert result.returncode == 0, result.stderr
    comparison = json.loads((audit_inputs / "comparison-candidate-reference.json").read_text())
    left_path = audit_inputs / "candidate/results.json"
    left = json.loads(left_path.read_text())
    assert comparison["input_manifest_sha256"] == left["input_manifest_sha256"]
    assert comparison["runs"]["left"]["results_sha256"] == hashlib.sha256(left_path.read_bytes()).hexdigest()


@pytest.fixture
def audit_module():
    spec = importlib.util.spec_from_file_location("audio_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audio_audit_snapshots_verified_input_bytes(tmp_path, audit_module):
    source = tmp_path / "input.wav"
    original = b"original encoded input"
    source.write_bytes(original)
    case = {"file": source.name, "sha256": hashlib.sha256(original).hexdigest()}
    snapshot = audit_module.snapshot_input(tmp_path, tmp_path / "run", case)
    source.write_bytes(b"regenerated input")
    assert snapshot != source and snapshot.read_bytes() == original
    assert snapshot.suffix == ".wav"


def test_audio_audit_rejects_input_changed_since_manifest(tmp_path, audit_module):
    source = tmp_path / "input.wav"
    source.write_bytes(b"regenerated input")
    case = {"file": source.name, "sha256": hashlib.sha256(b"original input").hexdigest()}
    with pytest.raises(ValueError, match="Input changed"):
        audit_module.snapshot_input(tmp_path, tmp_path / "run", case)
    assert not (tmp_path / "run").exists()
