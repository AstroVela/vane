from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_METRICS = (
    "recall_at_k",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "qps",
    "index_build_seconds",
    "ingest_rows_per_second",
    "peak_memory_mb",
    "index_storage_mb",
    "fallback_rate",
)

DEFAULT_ENGINES = (
    "duckdb_track_b",
    "milvus",
    "qdrant",
)


@dataclass
class BenchmarkDatasetSpec:
    name: str
    source: str
    row_count: int
    vector_dim: int
    has_text: bool
    has_payload_filter: bool
    ground_truth_path: str
    notes: str = ""


@dataclass
class BenchmarkCaseSpec:
    name: str
    dataset_name: str
    k: int
    query_count: int
    filter_selectivity: float | None
    include_hybrid: bool
    notes: str = ""


@dataclass
class BenchmarkTemplate:
    version: str
    engines: list[str]
    required_metrics: list[str]
    datasets: list[BenchmarkDatasetSpec]
    cases: list[BenchmarkCaseSpec]
    notes: str = ""

    def validate(self) -> None:
        if not self.version.strip():
            raise ValueError("benchmark template requires non-empty version")
        if not self.engines:
            raise ValueError("benchmark template requires at least one engine")
        if len(set(self.engines)) != len(self.engines):
            raise ValueError("benchmark template engines must be unique")
        if not self.required_metrics:
            raise ValueError("benchmark template requires at least one metric")
        if len(set(self.required_metrics)) != len(self.required_metrics):
            raise ValueError("benchmark template metrics must be unique")

        metric_set = set(self.required_metrics)
        for required in ("recall_at_k", "latency_p95_ms"):
            if required not in metric_set:
                raise ValueError(f"benchmark template requires metric '{required}'")

        dataset_names = [dataset.name for dataset in self.datasets]
        if not dataset_names:
            raise ValueError("benchmark template requires at least one dataset")
        if len(set(dataset_names)) != len(dataset_names):
            raise ValueError("benchmark template dataset names must be unique")
        dataset_set = set(dataset_names)

        case_names = [case.name for case in self.cases]
        if not case_names:
            raise ValueError("benchmark template requires at least one case")
        if len(set(case_names)) != len(case_names):
            raise ValueError("benchmark template case names must be unique")

        for dataset in self.datasets:
            if dataset.row_count <= 0:
                raise ValueError(f"dataset '{dataset.name}' requires row_count > 0")
            if dataset.vector_dim <= 0:
                raise ValueError(f"dataset '{dataset.name}' requires vector_dim > 0")
            if not dataset.ground_truth_path.strip():
                raise ValueError(f"dataset '{dataset.name}' requires non-empty ground_truth_path")

        for case in self.cases:
            if case.dataset_name not in dataset_set:
                raise ValueError(f"case '{case.name}' references unknown dataset '{case.dataset_name}'")
            if case.k <= 0:
                raise ValueError(f"case '{case.name}' requires k > 0")
            if case.query_count <= 0:
                raise ValueError(f"case '{case.name}' requires query_count > 0")
            if case.filter_selectivity is not None and not (0.0 < case.filter_selectivity <= 1.0):
                raise ValueError(f"case '{case.name}' requires filter_selectivity in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": self.version,
            "engines": list(self.engines),
            "required_metrics": list(self.required_metrics),
            "datasets": [asdict(dataset) for dataset in self.datasets],
            "cases": [asdict(case) for case in self.cases],
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> BenchmarkTemplate:
        template = BenchmarkTemplate(
            version=str(payload.get("version", "")).strip(),
            engines=[str(engine) for engine in payload.get("engines", [])],
            required_metrics=[str(metric) for metric in payload.get("required_metrics", [])],
            datasets=[
                BenchmarkDatasetSpec(
                    name=str(dataset.get("name", "")).strip(),
                    source=str(dataset.get("source", "")).strip(),
                    row_count=int(dataset.get("row_count", 0)),
                    vector_dim=int(dataset.get("vector_dim", 0)),
                    has_text=bool(dataset.get("has_text", False)),
                    has_payload_filter=bool(dataset.get("has_payload_filter", False)),
                    ground_truth_path=str(dataset.get("ground_truth_path", "")).strip(),
                    notes=str(dataset.get("notes", "")).strip(),
                )
                for dataset in payload.get("datasets", [])
            ],
            cases=[
                BenchmarkCaseSpec(
                    name=str(case.get("name", "")).strip(),
                    dataset_name=str(case.get("dataset_name", "")).strip(),
                    k=int(case.get("k", 0)),
                    query_count=int(case.get("query_count", 0)),
                    filter_selectivity=(
                        None if case.get("filter_selectivity", None) is None else float(case.get("filter_selectivity"))
                    ),
                    include_hybrid=bool(case.get("include_hybrid", False)),
                    notes=str(case.get("notes", "")).strip(),
                )
                for case in payload.get("cases", [])
            ],
            notes=str(payload.get("notes", "")).strip(),
        )
        template.validate()
        return template


def create_default_parity_template() -> BenchmarkTemplate:
    return BenchmarkTemplate(
        version="vector-parity-v1",
        engines=list(DEFAULT_ENGINES),
        required_metrics=list(DEFAULT_METRICS),
        datasets=[
            BenchmarkDatasetSpec(
                name="rag_corpus_1m_768",
                source="s3://datasets/rag_corpus_1m_768",
                row_count=1_000_000,
                vector_dim=768,
                has_text=True,
                has_payload_filter=True,
                ground_truth_path="ground_truth/rag_corpus_1m_768_top100.parquet",
                notes="RAG baseline; English text + metadata filter",
            ),
            BenchmarkDatasetSpec(
                name="recommendation_10m_256",
                source="s3://datasets/recommendation_10m_256",
                row_count=10_000_000,
                vector_dim=256,
                has_text=False,
                has_payload_filter=True,
                ground_truth_path="ground_truth/recommendation_10m_256_top100.parquet",
                notes="Recommendation-like ANN + filter workload",
            ),
            BenchmarkDatasetSpec(
                name="hybrid_news_3m_1024",
                source="s3://datasets/hybrid_news_3m_1024",
                row_count=3_000_000,
                vector_dim=1024,
                has_text=True,
                has_payload_filter=True,
                ground_truth_path="ground_truth/hybrid_news_3m_1024_top100.parquet",
                notes="Hybrid retrieval (vector + BM25-like text matching)",
            ),
        ],
        cases=[
            BenchmarkCaseSpec(
                name="ann_no_filter_k10",
                dataset_name="recommendation_10m_256",
                k=10,
                query_count=1000,
                filter_selectivity=None,
                include_hybrid=False,
                notes="Pure ANN latency/recall baseline",
            ),
            BenchmarkCaseSpec(
                name="ann_filter_1pct_k10",
                dataset_name="rag_corpus_1m_768",
                k=10,
                query_count=1000,
                filter_selectivity=0.01,
                include_hybrid=False,
                notes="Filtered ANN with 1% selectivity",
            ),
            BenchmarkCaseSpec(
                name="ann_filter_01pct_k10",
                dataset_name="rag_corpus_1m_768",
                k=10,
                query_count=1000,
                filter_selectivity=0.001,
                include_hybrid=False,
                notes="Filtered ANN with 0.1% selectivity",
            ),
            BenchmarkCaseSpec(
                name="hybrid_filter_10pct_k20",
                dataset_name="hybrid_news_3m_1024",
                k=20,
                query_count=1000,
                filter_selectivity=0.1,
                include_hybrid=True,
                notes="Hybrid retrieval with text + vector fusion",
            ),
        ],
        notes=(
            "Default template for DuckDB Track B parity benchmark. "
            "Use the same dataset snapshot and query set across engines."
        ),
    )


def save_template_json(template: BenchmarkTemplate, output_path: str | Path) -> Path:
    path = Path(output_path)
    payload = template.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_template_json(path: str | Path) -> BenchmarkTemplate:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark template JSON must be an object")
    return BenchmarkTemplate.from_dict(payload)


def render_comparison_report_template(template: BenchmarkTemplate) -> str:
    template.validate()
    lines: list[str] = []
    lines.append("# Vector Engine Parity Report Template")
    lines.append("")
    lines.append(f"- Version: `{template.version}`")
    lines.append("- Engines: " + ", ".join(f"`{engine}`" for engine in template.engines))
    lines.append("- Required Metrics: " + ", ".join(f"`{metric}`" for metric in template.required_metrics))
    lines.append("")
    lines.append("## Dataset Matrix")
    lines.append("")
    lines.append("| Dataset | Rows | Dim | Text | Filter | Ground Truth |")
    lines.append("| --- | ---: | ---: | :---: | :---: | --- |")
    for dataset in template.datasets:
        lines.append(
            f"| `{dataset.name}` | {dataset.row_count} | {dataset.vector_dim} | "
            f"{'Y' if dataset.has_text else 'N'} | {'Y' if dataset.has_payload_filter else 'N'} | "
            f"`{dataset.ground_truth_path}` |"
        )
    lines.append("")
    lines.append("## Case Matrix")
    lines.append("")
    lines.append("| Case | Dataset | K | Query Count | Filter Selectivity | Hybrid |")
    lines.append("| --- | --- | ---: | ---: | ---: | :---: |")
    for case in template.cases:
        selectivity = "-" if case.filter_selectivity is None else f"{case.filter_selectivity:.4f}"
        lines.append(
            f"| `{case.name}` | `{case.dataset_name}` | {case.k} | {case.query_count} | "
            f"{selectivity} | {'Y' if case.include_hybrid else 'N'} |"
        )
    lines.append("")
    lines.append("## Engine Comparison")
    lines.append("")
    for case in template.cases:
        lines.append(f"### {case.name}")
        lines.append("")
        header_metrics = " | ".join(template.required_metrics)
        lines.append(f"| Engine | {header_metrics} |")
        lines.append("| --- | " + " | ".join("---" for _ in template.required_metrics) + " |")
        for engine in template.engines:
            lines.append(f"| `{engine}` | " + " | ".join("TODO" for _ in template.required_metrics) + " |")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Keep dataset snapshot/checkpoint identical across engines.")
    lines.append("- For ANN quality, compute recall@k against exact ground truth.")
    lines.append("- Record fallback ratio for strategy stability analysis.")
    lines.append("")
    return "\n".join(lines)
