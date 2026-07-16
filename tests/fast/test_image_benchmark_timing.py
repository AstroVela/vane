from __future__ import annotations

import pytest


def _timing_tokens(text: str) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for item in text.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        tokens[key] = value
    return tokens


def test_image_cpu_udf_timing_fields_include_ms_aliases():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
    pytest.importorskip("pyarrow")

    from multimodal_inference_benchmarks.image_classification import vane_main

    fields = vane_main._format_cpu_udf_timing_fields(
        read_s=0.001,
        decode_s=0.002,
        transform_s=0.0035,
        stack_s=0.004,
        arrow_s=0.005,
        total_s=0.020,
        rows_per_s=123.456,
        transform_to_tensor_s=0.006,
        transform_resize_s=0.007,
        transform_center_crop_s=0.008,
        transform_convert_dtype_s=0.009,
        transform_normalize_s=0.010,
        transform_numpy_s=0.011,
    )

    tokens = _timing_tokens(fields)
    assert tokens["decode_s"] == "0.002000"
    assert tokens["decode_ms"] == "2.000"
    assert tokens["transform_s"] == "0.003500"
    assert tokens["transform_ms"] == "3.500"
    assert tokens["transform_resize_ms"] == "7.000"
    assert tokens["total_ms"] == "20.000"
    assert tokens["rows_per_s"] == "123.46"


def test_image_udf_timing_line_appends_to_configured_file(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
    pytest.importorskip("pyarrow")

    from multimodal_inference_benchmarks.image_classification import vane_main

    log_path = tmp_path / "timing" / "udf.log"
    monkeypatch.setattr(vane_main, "UDF_TIMING_LOG_PATH", str(log_path))

    vane_main._emit_udf_timing_line("[vane_image][cpu_udf_timing] run_id=test rows=1 decode_ms=2.000")

    captured = capsys.readouterr()
    assert "decode_ms=2.000" in captured.err
    assert log_path.read_text(encoding="utf-8").strip().endswith("decode_ms=2.000")
