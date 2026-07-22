# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from collections.abc import Callable

_FALSE_VALUES = {"", "0", "false", "no", "off", "none"}
_LOG_VALUES = {"log", "raylog", "text"}
_PIPELINE_TOPOLOGY_SCHEMA = "pipeline_topology"


def progress_enabled(runner_type: str | None = None) -> bool:
    value = os.getenv("VANE_PROGRESS", "auto").strip().lower()
    if value in _FALSE_VALUES:
        return False
    runner = (runner_type or os.getenv("VANE_RUNNER", "")).strip().lower() or "ray"
    return runner in {"local", "ray"} or value not in ("", "auto")


def _progress_interval_s() -> float:
    try:
        return max(0.1, float(os.getenv("VANE_PROGRESS_INTERVAL_SEC", "0.5")))
    except ValueError:
        return 0.5


def _int_value(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def validate_pipeline_topology(
    topology: Any,
    *,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Validate and detach the exact immutable pipeline-topology contract."""
    if type(topology) is not dict:
        raise TypeError("pipeline topology must be a dict")
    if set(topology) != {"schema", "pipelines"}:
        raise ValueError("pipeline topology must contain exactly schema and pipelines")
    if topology["schema"] != _PIPELINE_TOPOLOGY_SCHEMA:
        raise ValueError(f"pipeline topology schema must be {_PIPELINE_TOPOLOGY_SCHEMA!r}")
    raw_pipelines = topology["pipelines"]
    if type(raw_pipelines) is not list:
        raise TypeError("pipeline topology pipelines must be a list")
    if not allow_empty and not raw_pipelines:
        raise ValueError("pipeline topology must contain at least one pipeline")

    pipeline_ids: set[int] = set()
    pipelines: list[dict[str, Any]] = []
    required_fields = {"pipeline_id", "operators", "operator_details", "stage_ids"}
    for raw_pipeline in raw_pipelines:
        if type(raw_pipeline) is not dict or set(raw_pipeline) != required_fields:
            raise ValueError(
                "pipeline topology entries must contain exactly pipeline_id, operators, operator_details, and stage_ids"
            )
        pipeline_id = raw_pipeline["pipeline_id"]
        if type(pipeline_id) is not int or pipeline_id <= 0 or pipeline_id in pipeline_ids:
            raise ValueError("pipeline topology requires unique positive integer pipeline_id values")
        pipeline_ids.add(pipeline_id)

        operators = raw_pipeline["operators"]
        if type(operators) is not list or not operators or any(type(op) is not str or not op for op in operators):
            raise ValueError("pipeline topology operators must be a non-empty list of strings")
        operator_details = raw_pipeline["operator_details"]
        if type(operator_details) is not list or len(operator_details) != len(operators):
            raise ValueError("pipeline topology operator_details must align with operators")
        if any(type(details) is not dict for details in operator_details):
            raise TypeError("pipeline topology operator_details entries must be dicts")
        stage_ids = raw_pipeline["stage_ids"]
        if type(stage_ids) is not list or any(type(stage_id) is not int or stage_id < 0 for stage_id in stage_ids):
            raise ValueError("pipeline topology stage_ids must be non-negative integers")
        pipelines.append(
            {
                "pipeline_id": pipeline_id,
                "operators": list(operators),
                "operator_details": copy.deepcopy(operator_details),
                "stage_ids": list(stage_ids),
            }
        )
    return {"schema": _PIPELINE_TOPOLOGY_SCHEMA, "pipelines": pipelines}


def _format_count(value: float) -> str:
    value = float(value or 0)
    sign = "-" if value < 0 else ""
    value = abs(value)
    units = ("", "K", "M", "B", "T")
    unit = ""
    for unit in units:
        if value < 1000 or unit == units[-1]:
            break
        value /= 1000.0
    if unit == "":
        return f"{sign}{int(value)}"
    if value >= 100:
        text = f"{value:.0f}"
    elif value >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{sign}{text}{unit}"


def _format_bytes(value: float) -> str:
    value = float(value or 0)
    sign = "-" if value < 0 else ""
    value = abs(value)
    units = ("B", "KB", "MB", "GB", "TB", "PB")
    unit = "B"
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{sign}{int(value)}B"
    if value >= 100:
        text = f"{value:.0f}"
    elif value >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return f"{sign}{text}{unit}"


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def _rows_from_stats(stats: Any) -> int:
    if not isinstance(stats, dict):
        return 0
    return max(0, _int_value(stats.get("processed_input_rows")))


def _bytes_from_stats(stats: Any) -> int:
    if not isinstance(stats, dict):
        return 0
    return max(0, _int_value(stats.get("processed_input_bytes")))


def _statement_bytes_from_stats(stats: Any) -> int:
    if not isinstance(stats, dict):
        return 0
    statement_bytes = (
        _int_value(stats.get("physical_input_bytes"))
        + _int_value(stats.get("internal_network_input_bytes"))
        + _int_value(stats.get("memory_input_bytes"))
    )
    return max(0, statement_bytes)


def _selected_stats_from_partition(partition: dict[str, Any]) -> dict[str, Any]:
    stats = partition.get("selected_output_stats")
    if isinstance(stats, dict):
        return stats
    return {}


def _stats_from_partition(partition: dict[str, Any]) -> list[dict[str, Any]]:
    stats_items: list[dict[str, Any]] = []
    selected = _selected_stats_from_partition(partition)
    if selected:
        stats_items.append(selected)
    if str(partition.get("state") or "").upper() != "FINISHED":
        for attempt in partition.get("running_attempts") or []:
            if not isinstance(attempt, dict):
                continue
            attempt_stats = attempt.get("task_stats")
            if isinstance(attempt_stats, dict):
                stats_items.append(attempt_stats)
    return stats_items


_OPERATOR_LABELS = {
    "TABLE_SCAN": "ScanSource",
    "COLUMN_DATA_SCAN": "ColumnDataScan",
    "EXCHANGE_SOURCE": "ExchangeSource",
    "EXCHANGE_SINK": "ExchangeSink",
    "REPARTITION": "Repartition",
    "LOCAL_EXCHANGE": "LocalExchange",
    "PROJECTION": "Projection",
    "FILTER": "Filter",
    "STREAMING_LIMIT": "Limit",
    "COPY_TO_FILE": "CopySink",
    "BATCH_COPY_TO_FILE": "CopySink",
    "STREAMING_UDF": "StreamingUDF",
    "INOUT_FUNCTION": "StreamingUDF",
    "TABLE_IN_OUT_FUNCTION": "StreamingUDF",
}


def _operator_detail_at(operator_details: Any, index: int) -> dict[str, Any]:
    if not isinstance(operator_details, list) or index >= len(operator_details):
        return {}
    details = operator_details[index]
    return details if isinstance(details, dict) else {}


def _display_operator_label(operator_name: str, details: dict[str, Any]) -> str:
    udf_name = str(details.get("udf_name") or "").strip()
    label = udf_name if udf_name and udf_name != "udf" else _OPERATOR_LABELS.get(operator_name, operator_name)
    role = str(details.get("pipeline_role") or "").strip()
    return f"{label}({role})" if role else label


def _display_pipeline_name(pipeline: dict[str, Any]) -> str:
    name = str(pipeline.get("name") or "")
    operators = pipeline.get("operators")
    if isinstance(operators, list) and operators:
        operator_details = pipeline.get("operator_details")
        parts = [
            _display_operator_label(str(op), _operator_detail_at(operator_details, index))
            for index, op in enumerate(operators)
        ]
        return "->".join(part for part in parts if part) or name or "Pipeline"
    if name:
        return "->".join(_OPERATOR_LABELS.get(part, part) for part in name.split("->"))
    return "Pipeline"


def _aggregate_pipeline_stats(
    stats_items: list[dict[str, Any]],
    *,
    fragment_display_index: int,
) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for stats in stats_items:
        for pipeline in stats.get("pipelines") or []:
            if type(pipeline) is not dict:
                raise TypeError("pipeline stats entries must be dicts")
            pipeline_id = pipeline.get("pipeline_id")
            if type(pipeline_id) is not int or pipeline_id <= 0:
                raise ValueError("live pipeline stats require a positive integer pipeline_id")
            name = _display_pipeline_name(pipeline)
            operators = tuple(str(operator) for operator in pipeline.get("operators") or [])
            item = by_id.get(pipeline_id)
            if item is None:
                item = {
                    "pipeline_id": pipeline_id,
                    "name": name,
                    "operators": operators,
                    "processed_rows": 0,
                    "processed_bytes": 0,
                    "output_rows": 0,
                    "output_bytes": 0,
                    "total_pipeline_tasks": 0,
                    "queued_pipeline_tasks": 0,
                    "running_pipeline_tasks": 0,
                    "completed_pipeline_tasks": 0,
                }
                by_id[pipeline_id] = item
            elif item["operators"] != operators or item["name"] != name:
                raise RuntimeError(f"pipeline {pipeline_id} changed structure while aggregating progress")
            item["processed_rows"] += max(0, _int_value(pipeline.get("input_rows")))
            item["processed_bytes"] += max(0, _int_value(pipeline.get("input_bytes")))
            item["output_rows"] += max(0, _int_value(pipeline.get("output_rows")))
            item["output_bytes"] += max(0, _int_value(pipeline.get("output_bytes")))
            pipeline_task_counts = _pipeline_task_counts_from_stats(pipeline)
            if pipeline_task_counts is not None:
                part_total, part_queued, part_running, part_done = pipeline_task_counts
                item["total_pipeline_tasks"] += part_total
                item["queued_pipeline_tasks"] += part_queued
                item["running_pipeline_tasks"] += part_running
                item["completed_pipeline_tasks"] += part_done

    pipelines: list[dict[str, Any]] = []
    ordered_items = sorted(by_id.items(), key=lambda entry: -entry[0])
    for local_index, (_, item) in enumerate(ordered_items, start=1):
        display_id = f"{fragment_display_index}.{local_index}"
        total = item["total_pipeline_tasks"]
        queued = item["queued_pipeline_tasks"]
        running = item["running_pipeline_tasks"]
        done = item["completed_pipeline_tasks"]
        pipelines.append(
            {
                "id": display_id,
                "display_id": display_id,
                "name": item["name"],
                "state": "D" if done >= total and total > 0 else ("R" if running > 0 else ("Q" if queued > 0 else "P")),
                "processed_rows": item["processed_rows"],
                "processed_bytes": item["processed_bytes"],
                "output_rows": item["output_rows"],
                "output_bytes": item["output_bytes"],
                "queued_pipeline_tasks": queued,
                "running_pipeline_tasks": running,
                "completed_pipeline_tasks": done,
                "total_pipeline_tasks": total,
            }
        )
    return pipelines


def _pipeline_stats_with_progress_topology(
    progress_topology: Any,
    live_stats_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep native topology stable while overlaying matching live counters."""
    topology = validate_pipeline_topology(progress_topology, allow_empty=True)
    topology_by_id = {pipeline["pipeline_id"]: pipeline for pipeline in topology["pipelines"]}
    zero_pipelines = [
        {
            **copy.deepcopy(pipeline),
            "input_rows": 0,
            "input_bytes": 0,
            "output_rows": 0,
            "output_bytes": 0,
            "total_pipeline_tasks": 0,
            "queued_pipeline_tasks": 0,
            "running_pipeline_tasks": 0,
            "completed_pipeline_tasks": 0,
        }
        for pipeline in topology["pipelines"]
    ]
    aligned_stats: list[dict[str, Any]] = [{"pipelines": zero_pipelines}]
    for stats in live_stats_items:
        aligned = dict(stats)
        aligned_pipelines: list[dict[str, Any]] = []
        for pipeline in stats.get("pipelines") or []:
            if type(pipeline) is not dict:
                raise TypeError("live pipeline stats entries must be dicts")
            pipeline_id = pipeline.get("pipeline_id")
            if type(pipeline_id) is not int or pipeline_id <= 0:
                raise ValueError("live pipeline stats require a positive integer pipeline_id")
            planned = topology_by_id.get(pipeline_id)
            if planned is None:
                raise RuntimeError(f"live progress reported unknown pipeline_id {pipeline_id}")
            operators = pipeline.get("operators")
            if operators != planned["operators"]:
                raise RuntimeError(f"live progress pipeline {pipeline_id} operators do not match topology")
            aligned_pipelines.append(pipeline)
        aligned["pipelines"] = aligned_pipelines
        aligned_stats.append(aligned)
    return aligned_stats


def _pipeline_task_counts_from_stats(stats: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(stats, dict):
        return None
    total = max(0, _int_value(stats.get("total_pipeline_tasks")))
    if total == 0:
        return None
    completed = min(total, max(0, _int_value(stats.get("completed_pipeline_tasks"))))
    running = min(total - completed, max(0, _int_value(stats.get("running_pipeline_tasks"))))
    queued = min(total - completed - running, max(0, _int_value(stats.get("queued_pipeline_tasks"))))
    queued += total - completed - running - queued
    return total, queued, running, completed


def _stage_native_pipeline_task_counts(stats_items: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    total = queued = running = done = 0
    for stats in stats_items:
        counts = _pipeline_task_counts_from_stats(stats)
        if counts is None:
            continue
        part_total, part_queued, part_running, part_done = counts
        total += part_total
        queued += part_queued
        running += part_running
        done += part_done
    return total, queued, running, done


def _logical_partition_counts(partition_values: list[dict[str, Any]]) -> tuple[int, int, int, int, int]:
    """Return total/queued/running/completed/failed logical partitions."""
    states = [str(partition.get("state") or "").upper() for partition in partition_values]
    total = len(states)
    completed = states.count("FINISHED")
    failed = states.count("FAILED")
    running = states.count("RUNNING")
    queued = max(0, total - completed - failed - running)
    return total, queued, running, completed, failed


def build_progress_snapshot(
    registry_stats: dict[str, Any] | None,
    query_id: str,
    *,
    started_at: float | None = None,
) -> dict[str, Any]:
    registry_stats = registry_stats or {}
    requested_query_id = str(query_id)
    queries = registry_stats.get("queries") or {}
    query = queries.get(requested_query_id) or {}
    if isinstance(query, dict):
        query_payload = {"query_id": str(query.get("query_id") or requested_query_id)}
        for key in ("failed", "finished"):
            if key in query:
                query_payload[key] = query[key]
        manager_snapshot = query.get("query_resource_manager")
        if manager_snapshot:
            query_payload["query_resource_manager"] = copy.deepcopy(manager_snapshot)
    else:
        query = {}
        query_payload = {"query_id": requested_query_id}
    if not query_payload.get("query_resource_manager"):
        try:
            from vane.runners.ray.query_resource_runtime import query_resource_manager_snapshot

            manager_snapshot = query_resource_manager_snapshot(requested_query_id)
        except Exception:
            manager_snapshot = {}
        if manager_snapshot:
            query_payload["query_resource_manager"] = manager_snapshot
    raw_fragment_executions = query.get("fragment_executions") or {}
    fragments: list[dict[str, Any]] = []
    total_pipeline_tasks = 0
    queued_pipeline_tasks = 0
    running_pipeline_tasks = 0
    completed_pipeline_tasks = 0
    total_partitions = 0
    queued_partitions = 0
    running_partitions = 0
    completed_partitions = 0
    failed_partitions = 0
    pending_partitions = 0
    processed_rows_total = 0
    processed_bytes = 0
    sorted_fragment_executions = sorted(
        raw_fragment_executions.values(),
        key=lambda item: (
            _int_value(item.get("fragment_execution_id")),
            str(item.get("fragment_id", "")),
        ),
    )
    for display_index, fragment_execution in enumerate(sorted_fragment_executions, start=1):
        partitions = fragment_execution.get("partitions") or {}
        partition_values = [value for value in partitions.values() if isinstance(value, dict)]
        stats_items: list[dict[str, Any]] = []
        for partition in partition_values:
            stats_items.extend(_stats_from_partition(partition))
        stage_rows = sum(_rows_from_stats(stats) for stats in stats_items)
        stage_bytes = sum(_statement_bytes_from_stats(stats) for stats in stats_items)
        native_total, native_queued, native_running, native_done = _stage_native_pipeline_task_counts(stats_items)
        partition_counts = _logical_partition_counts(partition_values)
        partition_total, partition_queued, partition_running, partition_done, partition_failed = partition_counts
        fragment_pending_partitions = max(
            0,
            _int_value(fragment_execution["pending_submission_count"]),
        )
        total_pipeline_tasks += native_total
        queued_pipeline_tasks += native_queued
        running_pipeline_tasks += native_running
        completed_pipeline_tasks += native_done
        total_partitions += partition_total
        queued_partitions += partition_queued
        running_partitions += partition_running
        completed_partitions += partition_done
        failed_partitions += partition_failed
        pending_partitions += fragment_pending_partitions
        processed_rows_total += stage_rows
        processed_bytes += stage_bytes
        pipelines = _aggregate_pipeline_stats(
            _pipeline_stats_with_progress_topology(
                fragment_execution["progress_topology"],
                stats_items,
            ),
            fragment_display_index=display_index,
        )
        fragments.append(
            {
                "id": str(fragment_execution.get("fragment_id") or display_index),
                "display_id": str(display_index),
                "name": f"Fragment {display_index}",
                "pending_partitions": fragment_pending_partitions,
                "pipelines": pipelines,
            }
        )

    query_failed = bool(query.get("failed"))
    all_fragment_partitions_sealed = all(
        bool(fragment_execution.get("no_more_partitions", True)) for fragment_execution in sorted_fragment_executions
    )
    query_finished = bool(query.get("finished")) and all_fragment_partitions_sealed
    query_state = "FAILED" if query_failed else ("FINISHED" if query_finished else "RUNNING")
    now = time.time()
    elapsed_s = max(0.0, now - float(started_at or now))
    return {
        "schema": "progress",
        "query_id": requested_query_id,
        "requested_query_id": requested_query_id,
        "state": query_state,
        "processed_rows": processed_rows_total,
        "processed_bytes": processed_bytes,
        "total_pipeline_tasks": total_pipeline_tasks,
        "queued_pipeline_tasks": queued_pipeline_tasks,
        "running_pipeline_tasks": running_pipeline_tasks,
        "completed_pipeline_tasks": completed_pipeline_tasks,
        "total_partitions": total_partitions,
        "queued_partitions": queued_partitions,
        "running_partitions": running_partitions,
        "completed_partitions": completed_partitions,
        "failed_partitions": failed_partitions,
        "pending_partitions": pending_partitions,
        "elapsed_s": elapsed_s,
        "query": query_payload,
        "fragments": fragments,
    }


def _stats_progress_bytes(stats: dict[str, Any]) -> int:
    return max(_bytes_from_stats(stats), _statement_bytes_from_stats(stats))


def build_local_progress_snapshot(
    task_stats: dict[str, Any] | None,
    query_id: str,
    *,
    started_at: float | None = None,
    state: str = "RUNNING",
) -> dict[str, Any]:
    stats = task_stats if isinstance(task_stats, dict) else {}
    state = state.upper()
    finished = state == "FINISHED"
    failed = state == "FAILED"
    pipeline_task_counts = _pipeline_task_counts_from_stats(stats)
    total, queued, running, done = pipeline_task_counts if pipeline_task_counts is not None else (0, 0, 0, 0)
    if finished:
        queued = 0
        running = 0
        done = total
    elif failed:
        queued = 0
        running = 0

    rows = _rows_from_stats(stats)
    bytes_value = _stats_progress_bytes(stats)
    pipelines = _aggregate_pipeline_stats(
        [stats] if stats else [],
        fragment_display_index=1,
    )
    if not pipelines:
        pipelines = [
            {
                "id": "1.1",
                "display_id": "1.1",
                "name": "LocalWrite",
                "state": "D" if finished else ("F" if failed else "R"),
                "processed_rows": rows,
                "processed_bytes": bytes_value,
                "queued_pipeline_tasks": queued,
                "running_pipeline_tasks": running,
                "completed_pipeline_tasks": done,
                "total_pipeline_tasks": total,
            }
        ]

    now = time.time()
    elapsed_s = max(0.0, now - float(started_at or now))
    return {
        "schema": "progress",
        "query_id": str(query_id),
        "requested_query_id": str(query_id),
        "state": state,
        "processed_rows": rows,
        "processed_bytes": bytes_value,
        "total_pipeline_tasks": total,
        "queued_pipeline_tasks": queued,
        "running_pipeline_tasks": running,
        "completed_pipeline_tasks": done,
        "pending_partitions": 0,
        "elapsed_s": elapsed_s,
        "fragments": [
            {
                "id": "local",
                "display_id": "1",
                "name": "Local",
                "pending_partitions": 0,
                "pipelines": pipelines,
            }
        ],
    }


class LocalProgressSnapshotStore:
    def __init__(self, query_id: str, *, started_at: float | None = None) -> None:
        self.query_id = str(query_id)
        self.started_at = time.time() if started_at is None else float(started_at)
        self._lock = threading.Lock()
        self._task_stats: dict[str, Any] = {}
        self._state = "RUNNING"

    def record(self, task_stats: dict[str, Any] | None) -> None:
        if not isinstance(task_stats, dict):
            return
        with self._lock:
            self._task_stats = copy.deepcopy(task_stats)

    def finish(self, task_stats: dict[str, Any] | None = None, *, failed: bool = False) -> None:
        with self._lock:
            if isinstance(task_stats, dict):
                self._task_stats = copy.deepcopy(task_stats)
            self._state = "FAILED" if failed else "FINISHED"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            task_stats = copy.deepcopy(self._task_stats)
            state = self._state
        return build_local_progress_snapshot(
            task_stats,
            self.query_id,
            started_at=self.started_at,
            state=state,
        )


def format_progress_snapshot(snapshot: dict[str, Any], *, width: int = 100) -> list[str]:
    elapsed_s = _float_value(snapshot.get("elapsed_s"))
    rows = _int_value(snapshot.get("processed_rows"))
    bytes_value = _int_value(snapshot.get("processed_bytes"))
    rows_rate = rows / elapsed_s if elapsed_s > 0 else 0.0
    bytes_rate = bytes_value / elapsed_s if elapsed_s > 0 else 0.0
    first = (
        f"{_format_elapsed(elapsed_s)} [{_format_count(rows):>5} rows, {_format_bytes(bytes_value):>6}] "
        f"[{_format_count(rows_rate):>5} rows/s, {_format_bytes(bytes_rate) + '/s':>8}]"
    )

    fragments = [fragment for fragment in snapshot.get("fragments") or [] if isinstance(fragment, dict)]
    if not fragments:
        return [first]

    label_width = max(32, min(92, width - 62))
    lines = [first, _format_tree_header(label_width=label_width)]
    for fragment in fragments:
        lines.append(f"Fragment {fragment['display_id']} [PENDING {_format_count(fragment['pending_partitions'])}]")
        lines.extend(
            _format_pipeline_item(pipeline, label_width=label_width, elapsed_s=elapsed_s)
            for pipeline in fragment.get("pipelines") or []
        )
    return lines


def _format_pipeline_item(item: dict[str, Any], *, label_width: int, elapsed_s: float) -> str:
    rows = _int_value(item.get("processed_rows"))
    bytes_value = _int_value(item.get("processed_bytes"))
    rate_elapsed_s = _float_value(item.get("rate_elapsed_s"), elapsed_s)
    if rate_elapsed_s <= 0:
        rate_elapsed_s = elapsed_s
    rows_rate = rows / rate_elapsed_s if rate_elapsed_s > 0 else 0.0
    bytes_rate = bytes_value / rate_elapsed_s if rate_elapsed_s > 0 else 0.0
    rows_rate_text = f"{_format_count(rows_rate)}/s"
    bytes_rate_text = f"{_format_bytes(bytes_rate)}/s"
    queued = _int_value(item.get("queued_pipeline_tasks"))
    running = _int_value(item.get("running_pipeline_tasks"))
    done = _int_value(item.get("completed_pipeline_tasks"))
    state = str(item.get("state") or "P")[:1]
    name = str(item.get("name") or "")
    if item.get("display_id") and name.startswith("Pipeline"):
        label = f"{name}"
    else:
        label = f"Pipeline {item.get('display_id')} | {name}"
    label = "  " + label
    if len(label) > label_width:
        label = label[: max(0, label_width - 1)] + "."
    dotted = label + "." * max(0, label_width - len(label))
    return (
        f"{dotted}{state:>2} "
        f"{_format_count(rows):>5} "
        f"{rows_rate_text:>8} "
        f"{_format_bytes(bytes_value):>6} "
        f"{bytes_rate_text:>9} "
        f"{queued:>6} {running:>3} {done:>4}"
    )


def _format_tree_header(*, label_width: int) -> str:
    return (
        f"{'FRAGMENTS':<{label_width}}   "
        f"{'ROWS':>5} "
        f"{'ROWS/s':>8} "
        f"{'BYTES':>6} "
        f"{'BYTES/s':>9} "
        f"{'QUEUED':>6} {'RUN':>3} {'DONE':>4}"
    )


class ProgressRenderer:
    def __init__(
        self,
        snapshot_getter: Callable[[], dict[str, Any] | None],
        *,
        stream: TextIO | None = None,
        interval_s: float | None = None,
    ) -> None:
        self.snapshot_getter = snapshot_getter
        self.stream = stream or sys.stderr
        self.interval_s = _progress_interval_s() if interval_s is None else max(0.1, interval_s)
        self.started_at = time.time()
        self._last_render_at = 0.0
        self._last_line_count = 0
        self._completed_rate_elapsed_s: dict[tuple[str, ...], float] = {}
        self._last_fragments: list[dict[str, Any]] = []
        self._last_snapshot: dict[str, Any] | None = None
        mode = os.getenv("VANE_PROGRESS", "auto").strip().lower()
        if mode in _LOG_VALUES:
            self._dynamic = False
        elif mode in ("", "auto"):
            isatty = getattr(self.stream, "isatty", None)
            self._dynamic = bool(isatty and isatty())
        else:
            self._dynamic = True

    def _annotate_completed_rate_windows(self, snapshot: dict[str, Any]) -> None:
        elapsed_s = max(0.0, _float_value(snapshot.get("elapsed_s")))

        def annotate(item: dict[str, Any], key: tuple[str, ...]) -> None:
            state = str(item.get("state") or "")[:1].upper()
            if state == "D":
                rate_elapsed_s = self._completed_rate_elapsed_s.setdefault(key, elapsed_s)
                item["rate_elapsed_s"] = rate_elapsed_s
            else:
                self._completed_rate_elapsed_s.pop(key, None)

        live_keys: set[tuple[str, ...]] = set()
        for fragment in snapshot.get("fragments") or []:
            if not isinstance(fragment, dict):
                continue
            fragment_key = ("fragment", str(fragment.get("id") or fragment.get("display_id") or ""))
            for pipeline in fragment.get("pipelines") or []:
                if not isinstance(pipeline, dict):
                    continue
                pipeline_key = (
                    "pipeline",
                    fragment_key[1],
                    str(pipeline.get("id") or pipeline.get("display_id") or ""),
                    str(pipeline.get("name") or ""),
                )
                live_keys.add(pipeline_key)
                annotate(pipeline, pipeline_key)

        for key in list(self._completed_rate_elapsed_s):
            if key not in live_keys:
                self._completed_rate_elapsed_s.pop(key, None)

    @staticmethod
    def _is_empty_progress_snapshot(snapshot: dict[str, Any]) -> bool:
        return (
            not snapshot.get("fragments")
            and _int_value(snapshot.get("processed_rows")) == 0
            and _int_value(snapshot.get("processed_bytes")) == 0
        )

    @staticmethod
    def _force_finished_snapshot(snapshot: dict[str, Any]) -> dict[str, Any] | None:
        if ProgressRenderer._is_empty_progress_snapshot(snapshot):
            return None
        partition_total = _int_value(snapshot.get("total_partitions"))
        pipeline_task_total = _int_value(snapshot.get("total_pipeline_tasks"))
        work_total = partition_total if partition_total > 0 else pipeline_task_total
        if work_total <= 0:
            return None
        no_active_work = (
            _int_value(snapshot.get("running_partitions")) == 0
            and _int_value(snapshot.get("queued_partitions")) == 0
            and _int_value(snapshot.get("running_pipeline_tasks")) == 0
            and _int_value(snapshot.get("queued_pipeline_tasks")) == 0
        )
        completed_work = (
            _int_value(snapshot.get("completed_partitions")) >= partition_total
            if partition_total > 0
            else _int_value(snapshot.get("completed_pipeline_tasks")) >= pipeline_task_total
        )
        already_finished = str(snapshot.get("state") or "").upper() == "FINISHED" and no_active_work and completed_work
        if already_finished:
            return None

        forced = copy.deepcopy(snapshot)
        forced["state"] = "FINISHED"
        forced["queued_pipeline_tasks"] = 0
        forced["running_pipeline_tasks"] = 0
        forced["completed_pipeline_tasks"] = pipeline_task_total
        forced["pending_partitions"] = 0
        if partition_total > 0:
            forced["queued_partitions"] = 0
            forced["running_partitions"] = 0
            forced["completed_partitions"] = partition_total
            forced["failed_partitions"] = 0

        def force_pipeline(item: dict[str, Any]) -> None:
            item_total = _int_value(item.get("total_pipeline_tasks"))
            if item_total > 0:
                item["state"] = "D"
                item["queued_pipeline_tasks"] = 0
                item["running_pipeline_tasks"] = 0
                item["completed_pipeline_tasks"] = item_total

        for fragment in forced.get("fragments") or []:
            if isinstance(fragment, dict):
                fragment["pending_partitions"] = 0
                for pipeline in fragment.get("pipelines") or []:
                    if isinstance(pipeline, dict):
                        force_pipeline(pipeline)
        return forced

    def _render(self, snapshot: dict[str, Any], *, now: float) -> None:
        self._annotate_completed_rate_windows(snapshot)
        width = 100
        try:
            width = os.get_terminal_size().columns
        except OSError:
            pass
        lines = format_progress_snapshot(snapshot, width=width)
        if self._dynamic:
            if self._last_line_count:
                self.stream.write("\r")
                if self._last_line_count > 1:
                    self.stream.write(f"\x1b[{self._last_line_count - 1}A")
                self.stream.write("\x1b[J")
            self.stream.write("\r" + "\n".join("\x1b[2K" + line for line in lines))
        else:
            self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()
        self._last_line_count = len(lines)
        self._last_render_at = now

    def _update_snapshot(self, snapshot: dict[str, Any], *, now: float) -> None:
        snapshot = dict(snapshot)
        snapshot["elapsed_s"] = max(0.0, now - self.started_at)
        if self._last_snapshot is not None and self._is_empty_progress_snapshot(snapshot):
            current_state = str(snapshot.get("state") or "").upper()
            snapshot = copy.deepcopy(self._last_snapshot)
            snapshot["elapsed_s"] = max(0.0, now - self.started_at)
            if current_state in {"FINISHED", "FAILED"}:
                snapshot["state"] = current_state
        elif snapshot.get("fragments"):
            self._last_fragments = copy.deepcopy(snapshot["fragments"])
        elif self._last_fragments and str(snapshot.get("state") or "").upper() not in {"FINISHED", "FAILED"}:
            snapshot = dict(snapshot)
            snapshot["fragments"] = copy.deepcopy(self._last_fragments)
        if not self._is_empty_progress_snapshot(snapshot):
            self._last_snapshot = copy.deepcopy(snapshot)
        self._render(snapshot, now=now)

    def update(self, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_render_at < self.interval_s:
            return
        snapshot = self.snapshot_getter()
        if not snapshot:
            return
        self._update_snapshot(snapshot, now=now)

    def finish(
        self,
        *,
        final_state: str | None = None,
        final_snapshot: dict[str, Any] | None = None,
    ) -> None:
        if final_snapshot is None:
            self.update(force=True)
        elif final_snapshot:
            self._update_snapshot(final_snapshot, now=time.time())
        if str(final_state or "").upper() == "FINISHED" and self._last_snapshot is not None:
            now = time.time()
            forced = self._force_finished_snapshot(self._last_snapshot)
            if forced is not None:
                forced["elapsed_s"] = max(0.0, now - self.started_at)
                self._last_snapshot = copy.deepcopy(forced)
                self._render(forced, now=now)
        if self._dynamic and self._last_line_count:
            self.stream.write("\n")
            self.stream.flush()
            self._last_line_count = 0
