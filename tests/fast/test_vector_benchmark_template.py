from __future__ import annotations

from pathlib import Path

import pytest

from benchmarking.vector.parity_template import (
    BenchmarkCaseSpec,
    BenchmarkTemplate,
    create_default_parity_template,
    load_template_json,
    render_comparison_report_template,
    save_template_json,
)


def test_default_parity_template_is_valid():
    template = create_default_parity_template()
    template.validate()
    assert template.version == "vector-parity-v1"
    assert "duckdb_track_b" in template.engines
    assert "milvus" in template.engines
    assert "qdrant" in template.engines
    assert "recall_at_k" in template.required_metrics
    assert "latency_p95_ms" in template.required_metrics


def test_template_json_roundtrip(tmp_path: Path):
    template = create_default_parity_template()
    output_path = tmp_path / "vector_parity_template.json"

    saved_path = save_template_json(template, output_path)
    assert saved_path == output_path
    assert output_path.exists()

    loaded = load_template_json(output_path)
    loaded.validate()
    assert loaded.version == template.version
    assert loaded.engines == template.engines
    assert [dataset.name for dataset in loaded.datasets] == [dataset.name for dataset in template.datasets]
    assert [case.name for case in loaded.cases] == [case.name for case in template.cases]


def test_template_validation_rejects_unknown_dataset():
    template = create_default_parity_template()
    template.cases.append(
        BenchmarkCaseSpec(
            name="bad_case",
            dataset_name="unknown_dataset",
            k=10,
            query_count=10,
            filter_selectivity=None,
            include_hybrid=False,
        )
    )
    with pytest.raises(ValueError, match="references unknown dataset"):
        template.validate()


def test_render_comparison_report_template_contains_all_cases():
    template = create_default_parity_template()
    report = render_comparison_report_template(template)
    assert report.startswith("# Vector Engine Parity Report Template")
    for case in template.cases:
        assert f"### {case.name}" in report
    for engine in template.engines:
        assert f"`{engine}`" in report
    assert "TODO" in report


def test_template_requires_core_metrics():
    template = create_default_parity_template()
    template.required_metrics = ["latency_p50_ms"]
    with pytest.raises(ValueError, match="requires metric 'recall_at_k'"):
        template.validate()

    template = create_default_parity_template()
    template.required_metrics = ["recall_at_k"]
    with pytest.raises(ValueError, match="requires metric 'latency_p95_ms'"):
        template.validate()


def test_load_template_json_requires_object(tmp_path: Path):
    path = tmp_path / "bad_template.json"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON must be an object"):
        load_template_json(path)


def test_duplicate_engine_rejected():
    template = BenchmarkTemplate(
        version="x",
        engines=["duckdb_track_b", "duckdb_track_b"],
        required_metrics=["recall_at_k", "latency_p95_ms"],
        datasets=create_default_parity_template().datasets,
        cases=create_default_parity_template().cases,
    )
    with pytest.raises(ValueError, match="engines must be unique"):
        template.validate()
