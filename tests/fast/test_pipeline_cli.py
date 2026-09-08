# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

import vane
from vane import cli
from vane.pipeline import PipelineConfigError, execute_pipeline, load_pipeline, validate_pipeline


def _pipeline() -> dict:
    return {
        "version": 1,
        "name": "test",
        "runner": {"type": "local"},
        "source": {"type": "parquet", "path": "input.parquet"},
        "steps": [
            {"type": "filter", "expression": "score < 0.9"},
            {"type": "select", "expressions": ["id", "score"]},
            {"type": "limit", "count": 10},
        ],
        "sink": {"type": "json", "path": "output.json"},
    }


def test_load_pipeline_renders_parameters(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        """\
version: 1
source: {type: parquet, path: '{{ input }}'}
steps: []
sink: {type: parquet, path: '{{ output }}'}
""",
        encoding="utf-8",
    )

    config = load_pipeline(path, {"input": "in.parquet", "output": "out.parquet"})

    assert config["source"]["path"] == "in.parquet"
    assert config["sink"]["path"] == "out.parquet"
    assert config["runner"] == {"type": "ray"}


def test_load_pipeline_rejects_missing_parameter(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "version: 1\nsource: {type: parquet, path: '{{ input }}'}\nsink: {type: csv, path: out.csv}\n",
        encoding="utf-8",
    )

    with pytest.raises(PipelineConfigError, match="missing pipeline parameter: input"):
        load_pipeline(path)


def test_load_pipeline_reads_only_up_to_size_limit(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_bytes(b" " * (1024 * 1024 + 100))

    with pytest.raises(PipelineConfigError, match="pipeline file exceeds"):
        load_pipeline(path)


def test_load_pipeline_rejects_yaml_aliases(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text("version: 1\nsteps: &steps [*steps]\n", encoding="utf-8")

    with pytest.raises(PipelineConfigError, match="YAML aliases are not supported"):
        load_pipeline(path)


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        ({"type": "ray", "typo": 1}, "unknown keys: typo"),
        ({"type": "ray", "ray_max_task_backlog": "many"}, "ray_max_task_backlog must be int"),
    ],
)
def test_validate_pipeline_rejects_invalid_runner_options(runner, message):
    config = _pipeline()
    config["runner"] = runner

    with pytest.raises(PipelineConfigError, match=message):
        validate_pipeline(config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config.update(version=2),
        lambda config: config.update(unknown=True),
        lambda config: config["steps"].append({"type": "python", "code": "do_anything()"}),
        lambda config: config["sink"].update(type="shell"),
    ],
)
def test_validate_pipeline_rejects_unsupported_configuration(mutation):
    config = _pipeline()
    mutation(config)

    with pytest.raises(PipelineConfigError):
        validate_pipeline(config)


def test_execute_pipeline_builds_expected_relation_chain():
    events = []

    class Relation:
        def filter(self, expression):
            events.append(("filter", expression))
            return self

        def project(self, expression):
            events.append(("project", expression))
            return self

        def limit(self, count):
            events.append(("limit", count))
            return self

        def write_file(self, path, *, format):
            events.append(("write", path, format))

    class Connection:
        def read_parquet(self, path):
            events.append(("read_parquet", path))
            return Relation()

        def close(self):
            events.append(("close",))

    class Engine:
        @staticmethod
        def configure(**options):
            events.append(("configure", options))

        @staticmethod
        def connect():
            events.append(("connect",))
            return Connection()

        @staticmethod
        def teardown_runner():
            events.append(("teardown",))

    execute_pipeline(_pipeline(), engine=Engine())

    assert events == [
        ("configure", {"runner": "local"}),
        ("connect",),
        ("read_parquet", "input.parquet"),
        ("filter", "score < 0.9"),
        ("project", "id, score"),
        ("limit", 10),
        ("write", "output.json", "json"),
        ("close",),
        ("teardown",),
    ]


def test_cli_check_prints_rendered_configuration(tmp_path, capsys):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "version: 1\nsource: {type: sql, query: 'select {{ value }} as value'}\nsink: {type: csv, path: out.csv}\n",
        encoding="utf-8",
    )

    result = cli.main(["run", str(path), "--param", "value=42", "--check"])

    assert result == 0
    assert json.loads(capsys.readouterr().out)["source"]["query"] == "select 42 as value"


def test_cli_handles_engine_errors(tmp_path, monkeypatch, capsys):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "version: 1\nsource: {type: sql, query: 'select 1'}\nsink: {type: csv, path: out.csv}\n",
        encoding="utf-8",
    )

    def fail(_config):
        raise vane.IOException("input is unavailable")

    monkeypatch.setattr(cli, "execute_pipeline", fail)

    assert cli.main(["run", str(path)]) == 2
    assert "input is unavailable" in capsys.readouterr().err
