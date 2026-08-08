# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from vane.runners.ray.fragment_registry import (
    _FTE_CLOSING_QUERIES,
    _FTE_FRAGMENT_EXECUTIONS,
    _FTE_FRAGMENT_STATES,
    _FTE_REGISTRY_LOCK,
    _FTE_SCHEDULERS,
    _FteFragmentState,
)
from vane.runners.ray.fragment_worker_assignment import make_fte_assigner


def fte_fragment_execution_items(
    query_id_filter: str | None = None,
) -> list[tuple[tuple[str, str], Any]]:
    if query_id_filter is not None:
        query_id_filter = str(query_id_filter)
    with _FTE_REGISTRY_LOCK:
        return [
            item
            for item in _FTE_FRAGMENT_EXECUTIONS.items()
            if query_id_filter is None or item[0][0] == query_id_filter
        ]


def fte_fragment_execution_query_ids() -> set[str]:
    with _FTE_REGISTRY_LOCK:
        return {query_id for query_id, _ in _FTE_FRAGMENT_EXECUTIONS}


def get_or_create_fte_fragment_state(
    query_id: str,
    fragment_id: str,
    *,
    dynamic_scan_sources: set[str],
    dynamic_exchange_sources: set[str],
    exchange_source_partition_ids: set[int],
    replicated_exchange_sources: set[str] | None = None,
    exchange_source_partition_count: int = 0,
    exchange_source_task_count: int = 0,
) -> _FteFragmentState:
    key = (str(query_id), str(fragment_id))
    with _FTE_REGISTRY_LOCK:
        if key[0] in _FTE_CLOSING_QUERIES:
            raise RuntimeError(f"FTE query registry is closing: {key[0]}")
        scheduler = _FTE_SCHEDULERS.get_or_create(key[0])
        state = scheduler.fragment_state(key[1])
        if state is None:
            state = _FTE_FRAGMENT_STATES.get(key)
        if state is None:
            state = _FteFragmentState()
            scheduler.put_fragment_state(key[1], state)
            _FTE_FRAGMENT_STATES[key] = state
        elif key not in _FTE_FRAGMENT_STATES:
            _FTE_FRAGMENT_STATES[key] = state
        had_assigner = state.assigner is not None
        had_exchange_sources = bool(state.dynamic_exchange_source_node_ids)
        previous_exchange_partition_count = state.exchange_source_partition_count
        previous_exchange_task_count = state.exchange_source_task_count
        state.source_node_ids.update(dynamic_scan_sources)
        state.source_node_ids.update(dynamic_exchange_sources)
        state.dynamic_scan_source_node_ids.update(dynamic_scan_sources)
        state.dynamic_exchange_source_node_ids.update(dynamic_exchange_sources)
        state.replicated_exchange_source_node_ids.update(replicated_exchange_sources or set())
        state.exchange_source_partition_ids.update(exchange_source_partition_ids)
        state.exchange_source_partition_count = max(
            state.exchange_source_partition_count,
            int(exchange_source_partition_count),
        )
        state.exchange_source_task_count = max(
            state.exchange_source_task_count,
            int(exchange_source_task_count),
        )
        if had_assigner and dynamic_exchange_sources and not had_exchange_sources:
            raise RuntimeError("FTE cannot switch an existing arbitrary split assigner to hash distribution")
        if (
            had_assigner
            and had_exchange_sources
            and state.exchange_source_partition_count > previous_exchange_partition_count
        ):
            raise RuntimeError("FTE hash split assigner cannot grow source partition count")
        if had_assigner and had_exchange_sources and state.exchange_source_task_count > previous_exchange_task_count:
            raise RuntimeError("FTE hash split assigner cannot grow source task count")
        if state.assigner is None:
            state.assigner = make_fte_assigner(state)
        return state
