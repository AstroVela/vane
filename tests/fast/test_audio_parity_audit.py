# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import hashlib
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
            "resamples": {
                "wave": {
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
            "resamples": {"wave": {"value": daft_array, "expression": daft_array}},
            "files": {"input": {"decode": daft_array, "metadata_value": metadata, "metadata_expression": metadata}},
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
    wave = comparison["resamples"]["wave"]["python_vs_daft"]
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
    record["resamples"]["wave"]["value"][field] = value
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
    record["resamples"]["wave"]["value"]["sha256"] = hashlib.sha256(array.read_bytes()).hexdigest()
    path.write_text(json.dumps(record))
    result = _compare(audit_inputs)
    assert result.returncode == 1
    assert "ValueError:" in result.stderr and "Traceback" in result.stderr
    assert "usage:" not in result.stderr and "array SHA-256" not in result.stderr
    assert not list(audit_inputs.glob("comparison-*.json"))
