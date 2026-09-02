# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Declarative YAML pipelines for the Vane data engine."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_MAX_PIPELINE_BYTES = 1024 * 1024
_PARAMETER_PATTERN = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*}}")
_TOP_LEVEL_KEYS = frozenset({"version", "name", "runner", "source", "steps", "sink"})
_SOURCE_KEYS = frozenset({"type", "path", "query"})
_SINK_KEYS = frozenset({"type", "path"})
_STEP_KEYS = {
    "filter": frozenset({"type", "expression"}),
    "select": frozenset({"type", "expressions"}),
    "sql": frozenset({"type", "query", "alias"}),
    "limit": frozenset({"type", "count"}),
    "order": frozenset({"type", "expression"}),
}


class PipelineConfigError(ValueError):
    """Raised when a declarative pipeline is invalid."""


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineConfigError(f"{location} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise PipelineConfigError(f"{location} keys must be strings")
    return dict(value)


def _reject_unknown_keys(value: Mapping[str, Any], allowed: frozenset[str], location: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise PipelineConfigError(f"{location} contains unknown keys: {', '.join(unknown)}")


def _required_string(value: Mapping[str, Any], key: str, location: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise PipelineConfigError(f"{location}.{key} must be a non-empty string")
    return result


def _render_parameters(value: Any, parameters: Mapping[str, str]) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            try:
                return parameters[name]
            except KeyError:
                raise PipelineConfigError(f"missing pipeline parameter: {name}") from None

        return _PARAMETER_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_render_parameters(item, parameters) for item in value]
    if isinstance(value, Mapping):
        return {key: _render_parameters(item, parameters) for key, item in value.items()}
    return value


def validate_pipeline(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one parsed pipeline document."""
    normalized = _mapping(config, "pipeline")
    _reject_unknown_keys(normalized, _TOP_LEVEL_KEYS, "pipeline")
    if normalized.get("version") != 1:
        raise PipelineConfigError("pipeline.version must be 1")
    if "name" in normalized:
        _required_string(normalized, "name", "pipeline")

    runner = normalized.get("runner", {"type": "ray"})
    if isinstance(runner, str):
        runner = {"type": runner}
    runner = _mapping(runner, "pipeline.runner")
    runner_type = _required_string(runner, "type", "pipeline.runner").lower()
    if runner_type not in {"local", "ray"}:
        raise PipelineConfigError("pipeline.runner.type must be 'local' or 'ray'")
    runner["type"] = runner_type
    normalized["runner"] = runner

    source = _mapping(normalized.get("source"), "pipeline.source")
    _reject_unknown_keys(source, _SOURCE_KEYS, "pipeline.source")
    source_type = _required_string(source, "type", "pipeline.source").lower()
    if source_type in {"parquet", "csv", "json"}:
        _required_string(source, "path", "pipeline.source")
    elif source_type == "sql":
        _required_string(source, "query", "pipeline.source")
    else:
        raise PipelineConfigError("pipeline.source.type must be parquet, csv, json, or sql")
    source["type"] = source_type
    normalized["source"] = source

    raw_steps = normalized.get("steps", [])
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise PipelineConfigError("pipeline.steps must be a list")
    steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_steps):
        location = f"pipeline.steps[{index}]"
        step = _mapping(raw_step, location)
        step_type = _required_string(step, "type", location).lower()
        allowed = _STEP_KEYS.get(step_type)
        if allowed is None:
            raise PipelineConfigError(f"{location}.type is unsupported: {step_type}")
        _reject_unknown_keys(step, allowed, location)
        if step_type in {"filter", "order"}:
            _required_string(step, "expression", location)
        elif step_type == "sql":
            _required_string(step, "query", location)
            if "alias" in step:
                _required_string(step, "alias", location)
        elif step_type == "select":
            expressions = step.get("expressions")
            if (
                not isinstance(expressions, Sequence)
                or isinstance(expressions, (str, bytes))
                or not expressions
                or any(not isinstance(item, str) or not item.strip() for item in expressions)
            ):
                raise PipelineConfigError(f"{location}.expressions must be a non-empty list of strings")
            step["expressions"] = list(expressions)
        elif step_type == "limit":
            count = step.get("count")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise PipelineConfigError(f"{location}.count must be a non-negative integer")
        step["type"] = step_type
        steps.append(step)
    normalized["steps"] = steps

    sink = _mapping(normalized.get("sink"), "pipeline.sink")
    _reject_unknown_keys(sink, _SINK_KEYS, "pipeline.sink")
    sink_type = _required_string(sink, "type", "pipeline.sink").lower()
    if sink_type not in {"parquet", "csv", "json"}:
        raise PipelineConfigError("pipeline.sink.type must be parquet, csv, or json")
    _required_string(sink, "path", "pipeline.sink")
    sink["type"] = sink_type
    normalized["sink"] = sink
    return normalized


def load_pipeline(path: str | Path, parameters: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Load, parameterize, and validate a YAML pipeline file."""
    pipeline_path = Path(path)
    raw = pipeline_path.read_bytes()
    if len(raw) > _MAX_PIPELINE_BYTES:
        raise PipelineConfigError(f"pipeline file exceeds {_MAX_PIPELINE_BYTES} bytes")
    try:
        document = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PipelineConfigError(f"could not parse pipeline YAML: {error}") from error
    rendered = _render_parameters(document, parameters or {})
    return validate_pipeline(_mapping(rendered, "pipeline"))


def execute_pipeline(config: Mapping[str, Any], *, engine: Any = None) -> None:
    """Build and execute a previously loaded declarative pipeline."""
    normalized = validate_pipeline(config)
    if engine is None:
        import vane as engine

    runner = normalized["runner"]
    runner_options = {key: value for key, value in runner.items() if key != "type"}
    engine.configure(runner=runner["type"], **runner_options)
    connection = engine.connect()
    try:
        source = normalized["source"]
        source_type = source["type"]
        if source_type == "sql":
            relation = connection.sql(source["query"])
        elif source_type == "parquet":
            relation = connection.read_parquet(source["path"])
        elif source_type == "csv":
            relation = connection.read_csv(source["path"])
        else:
            relation = connection.read_json(source["path"])

        for step in normalized["steps"]:
            step_type = step["type"]
            if step_type == "filter":
                relation = relation.filter(step["expression"])
            elif step_type == "select":
                relation = relation.project(", ".join(step["expressions"]))
            elif step_type == "sql":
                relation = relation.query(step.get("alias", "input"), step["query"])
            elif step_type == "limit":
                relation = relation.limit(step["count"])
            else:
                relation = relation.order(step["expression"])

        sink = normalized["sink"]
        relation.write_file(sink["path"], format=sink["type"])
    finally:
        connection.close()
        teardown = getattr(engine, "teardown_runner", None)
        if callable(teardown):
            teardown()
