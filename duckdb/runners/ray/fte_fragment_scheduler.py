# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import math
import threading
import time
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from duckdb.runners.fte import (
    FteSplit,
    FteTaskAttemptId,
    FteTaskExecutionClass,
    FteWorkerReservationUnavailable,
    validate_fte_status_identity,
)
from duckdb.runners.progress import validate_pipeline_topology
from duckdb.runners.ray.fragment_submission_window import (
    admit_fte_partition_submission,
    fte_submission_window_snapshot,
    release_fte_partition_submission,
    release_fte_query_submissions,
    resolve_fte_partition_submission,
)
from duckdb.runners.ray.fragment_worker_context import resource_identity_from_context
from duckdb.runners.ray.fragment_worker_waiters import (
    release_fte_partition_task_waiter,
    release_fte_query_task_waiters,
)
from duckdb.runners.ray.fte_scheduler_config import (
    _chaos_host_loss_worker_indices,
    _fte_control_rpc_timeout_s,
    _fte_exhausted_node_wait_period_s,
    _fte_retry_delay_scale_factor,
    _fte_retry_initial_delay_s,
    _fte_retry_max_delay_s,
    _ray_fragment_plan_cache_session_key,
)
from duckdb.runners.ray.ray_env import collect_vane_env_overrides

if TYPE_CHECKING:
    from collections.abc import Callable

    from duckdb.runners.fte import FteFragmentExecution, NodeRequirements
    from duckdb.runners.ray.fragment_worker_client import RayWorkerActorHandle

import ray

from duckdb.runners.ray.fragment_registry import (
    _FRAGMENT_PLAN_REF_CACHE,
    _FRAGMENT_PLAN_REF_CACHE_LOCK,
    _FTE_ACTIVE_OPERATIONS_BY_QUERY,
    _FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY,
    _FTE_CLOSING_QUERIES,
    _FTE_FRAGMENT_EXECUTION_IDS,
    _FTE_FRAGMENT_EXECUTIONS,
    _FTE_FRAGMENT_PROGRESS_TOPOLOGIES,
    _FTE_FRAGMENT_PROGRESS_TOPOLOGY_BUILDS,
    _FTE_FRAGMENT_STATES,
    _FTE_PARTITION_OWNERS,
    _FTE_PARTITION_TASK_LEASES,
    _FTE_PARTITION_TASK_WAITERS,
    _FTE_PENDING_WORKER_RESERVATIONS,
    _FTE_QUERY_NEXT_FRAGMENT_EXECUTION_ID,
    _FTE_REGISTRY_CONDITION,
    _FTE_REGISTRY_LOCK,
    _FTE_RESULT_HANDLES_BY_QUERY,
    _FTE_RETRY_DELAYS,
    _FTE_SCHEDULERS,
    _FTE_SEQUENCES,
    _FTE_STAGE_SUBMISSION_BLOCKS,
    _FTE_STAGE_SUBMISSION_PROBES,
    _FTE_STATUS_WATCHERS,
    _FTE_WORKER_HANDLES,
    _FTE_WORKER_RESERVATION_GENERATIONS,
    _FteExchangeSourceOutputSelectorSnapshot,
    _FteSchedulingDelayer,
)

if TYPE_CHECKING:
    from duckdb.runners.ray.fragment_registry import _FteFragmentState


def _stable_fte_split_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ("bytes", hashlib.sha256(bytes(value)).hexdigest())
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (str(key), _stable_fte_split_value(value[key]))
                for key in sorted(value.keys(), key=lambda item: str(item))
            ),
        )
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_stable_fte_split_value(item) for item in value))
    if isinstance(value, set):
        return (
            "set",
            tuple(sorted(repr(_stable_fte_split_value(item)) for item in value)),
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _exchange_source_split_key(split: FteSplit) -> str:
    payload = (
        split.kind,
        int(split.source_partition_id),
        _stable_fte_split_value(split.data),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _exchange_selector_int_value(payload: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        return int(value)
    return None


def _exchange_selector_bool_value(payload: Mapping[str, Any], *keys: str) -> bool:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, str):
            return value.strip().lower() not in ("0", "false", "no", "off", "")
        return bool(value)
    return False


def _exchange_selector_attempt_from_split(split_payload: Mapping[str, Any]) -> int | None:
    for key in ("attempt_id", "attemptId"):
        if split_payload.get(key) is not None:
            return int(split_payload[key])
    data = split_payload.get("data")
    if not isinstance(data, Mapping):
        return None
    for key in ("attempt_id", "attemptId"):
        if data.get(key) is not None:
            return int(data[key])
    source_handles = data.get("source_handles") or data.get("sourceHandles") or data.get("handles")
    if not source_handles:
        return None
    for handle in source_handles:
        if isinstance(handle, Mapping):
            for key in ("attempt_id", "attemptId"):
                if handle.get(key) is not None:
                    return int(handle[key])
    return None


def _normalize_exchange_selector_split(
    source_node_id: str,
    partition_id: int,
    split_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], int | None, str]:
    payload = dict(split_payload)
    payload.setdefault("source_node_id", source_node_id)
    payload.setdefault("source_partition_id", int(partition_id))
    split = FteSplit.from_dict(source_node_id, payload)
    if split.kind != "exchange_source_task":
        raise ValueError(f"exchange selector split must be exchange_source_task, got {split.kind!r}")
    if int(split.source_partition_id) != int(partition_id):
        raise ValueError(
            "exchange selector split source_partition_id "
            f"{split.source_partition_id} does not match selected partition {partition_id}"
        )
    return split.to_dict(), _exchange_selector_attempt_from_split(payload), _exchange_source_split_key(split)


def _normalize_exchange_selector_selected_entry(
    source_node_id: str,
    partition_id: int,
    raw_entry: Any,
) -> dict[str, Any]:
    if raw_entry is None:
        return {
            "attempt_id": None,
            "split": None,
            "split_key": None,
        }
    if not isinstance(raw_entry, Mapping):
        raw_entry = {"attempt_id": raw_entry}
    entry = dict(raw_entry)
    split_payload = entry.get("split")
    if split_payload is None and entry.get("kind") == "exchange_source_task":
        split_payload = entry
    attempt_id = entry.get("attempt_id") if "attempt_id" in entry else entry.get("attemptId")
    split_key = entry.get("split_key") or entry.get("splitKey")
    normalized_split = None
    split_attempt_id = None
    if split_payload is not None:
        if not isinstance(split_payload, Mapping):
            raise ValueError("exchange selector selected split must be a mapping")
        normalized_split, split_attempt_id, split_key = _normalize_exchange_selector_split(
            source_node_id,
            partition_id,
            split_payload,
        )
    if attempt_id is None:
        attempt_id = split_attempt_id
    return {
        "attempt_id": None if attempt_id is None else int(attempt_id),
        "split": normalized_split,
        "split_key": None if split_key is None else str(split_key),
    }


def _normalize_exchange_selector_selected(
    source_node_id: str,
    selected_payload: Any,
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    if not selected_payload:
        return selected
    items: Iterable[tuple[Any, Any]]
    if isinstance(selected_payload, Mapping):
        items = selected_payload.items()
    elif isinstance(selected_payload, (list, tuple)):
        normalized_items: list[tuple[Any, Any]] = []
        for entry in selected_payload:
            if not isinstance(entry, Mapping):
                raise ValueError("exchange selector selected list entries must be mappings")
            partition_id = _exchange_selector_int_value(
                entry,
                "partition_id",
                "partitionId",
                "source_partition_id",
                "sourcePartitionId",
            )
            if partition_id is None:
                raise ValueError("exchange selector selected entry is missing partition_id")
            normalized_items.append((partition_id, entry))
        items = normalized_items
    else:
        raise ValueError("exchange selector selected must be a mapping or list")
    for raw_partition_id, raw_entry in items:
        partition_id = int(raw_partition_id)
        if partition_id < 0:
            raise ValueError("exchange selector partition id must be non-negative")
        entry = _normalize_exchange_selector_selected_entry(source_node_id, partition_id, raw_entry)
        existing = selected.get(partition_id)
        if existing is not None and (
            existing.get("attempt_id"),
            existing.get("split_key"),
        ) != (
            entry.get("attempt_id"),
            entry.get("split_key"),
        ):
            raise ValueError(f"conflicting exchange selector decision for partition {partition_id}")
        selected[partition_id] = entry
    return selected


def _normalize_exchange_selector_snapshot(
    fragment_state: _FteFragmentState,
    source_node_id: str,
    selector_payload: Mapping[str, Any],
) -> _FteExchangeSourceOutputSelectorSnapshot:
    selector = dict(selector_payload)
    source_node_id = str(
        selector.get("source_node_id") or selector.get("sourceNodeId") or selector.get("source") or source_node_id
    )
    version = _exchange_selector_int_value(selector, "version")
    if version is None:
        version = fragment_state.exchange_source_selector_next_version_by_source.get(source_node_id, 0)
    partition_count = _exchange_selector_int_value(
        selector,
        "partition_count",
        "partitionCount",
        "source_partition_count",
        "sourcePartitionCount",
    )
    selected_payload = (
        selector.get("selected")
        or selector.get("selections")
        or selector.get("selected_partitions")
        or selector.get("selectedPartitions")
        or {}
    )
    selected = _normalize_exchange_selector_selected(source_node_id, selected_payload)
    for split_payload in selector.get("splits") or ():
        if not isinstance(split_payload, Mapping):
            raise ValueError("exchange selector splits entries must be mappings")
        split = FteSplit.from_dict(source_node_id, split_payload)
        partition_id = int(split.source_partition_id)
        if partition_count is None:
            partition_count = 0
        partition_count = max(partition_count, partition_id + 1)
        selected[partition_id] = _normalize_exchange_selector_selected_entry(
            source_node_id,
            partition_id,
            split_payload,
        )
    return _FteExchangeSourceOutputSelectorSnapshot(
        version=int(version),
        source_node_id=source_node_id,
        final=_exchange_selector_bool_value(selector, "final", "finalSelector"),
        partition_count=partition_count,
        selected=selected,
    )


def _validate_exchange_selector_snapshot(snapshot: _FteExchangeSourceOutputSelectorSnapshot) -> None:
    if snapshot.version < 0:
        raise ValueError("exchange selector version must be non-negative")
    if snapshot.final:
        if snapshot.partition_count is None or snapshot.partition_count < 0:
            raise ValueError("final exchange selector requires partition_count")
        missing = [
            partition_id
            for partition_id in range(int(snapshot.partition_count))
            if partition_id not in snapshot.selected
        ]
        if missing:
            raise ValueError(f"final exchange selector is missing partitions: {missing}")


def _advance_exchange_selector_next_version(
    fragment_state: _FteFragmentState,
    source_node_id: str,
    version: int,
) -> None:
    fragment_state.exchange_source_selector_next_version_by_source[source_node_id] = max(
        fragment_state.exchange_source_selector_next_version_by_source.get(source_node_id, 0),
        int(version) + 1,
    )


def _merge_exchange_selector_snapshot(
    current: _FteExchangeSourceOutputSelectorSnapshot,
    snapshot: _FteExchangeSourceOutputSelectorSnapshot,
) -> tuple[_FteExchangeSourceOutputSelectorSnapshot, set[int]]:
    selected = {int(partition_id): dict(entry) for partition_id, entry in snapshot.selected.items()}
    materialized_partitions: set[int] = set()
    for partition_id, current_entry in current.selected.items():
        partition_id = int(partition_id)
        new_entry = selected.get(partition_id)
        if new_entry is None:
            selected[partition_id] = dict(current_entry)
            continue
        current_attempt = current_entry.get("attempt_id")
        new_attempt = new_entry.get("attempt_id")
        if current_attempt is not None and new_attempt is not None and current_attempt != new_attempt:
            raise ValueError(
                f"exchange selector cannot change selected attempt for partition {partition_id}: "
                f"{current_attempt} -> {new_attempt}"
            )
        current_split_key = current_entry.get("split_key")
        new_split_key = new_entry.get("split_key")
        if current_split_key is not None and new_split_key is not None and current_split_key != new_split_key:
            raise ValueError(f"exchange selector cannot change selected split for partition {partition_id}")
        if current_split_key is None and new_split_key is not None:
            materialized_partitions.add(partition_id)
        selected[partition_id] = {
            "attempt_id": new_attempt if new_attempt is not None else current_attempt,
            "split": new_entry.get("split") if new_split_key is not None else current_entry.get("split"),
            "split_key": new_split_key if new_split_key is not None else current_split_key,
        }
    for partition_id, entry in selected.items():
        if partition_id not in current.selected and entry.get("split_key") is not None:
            materialized_partitions.add(int(partition_id))
    return (
        _FteExchangeSourceOutputSelectorSnapshot(
            version=snapshot.version,
            source_node_id=snapshot.source_node_id,
            final=snapshot.final,
            partition_count=snapshot.partition_count,
            selected=selected,
        ),
        materialized_partitions,
    )


def _apply_exchange_selector_snapshot(
    fragment_state: _FteFragmentState,
    snapshot: _FteExchangeSourceOutputSelectorSnapshot,
) -> tuple[bool, set[int]]:
    current = fragment_state.exchange_source_selectors_by_source.get(snapshot.source_node_id)
    if current is None:
        _validate_exchange_selector_snapshot(snapshot)
        fragment_state.exchange_source_selectors_by_source[snapshot.source_node_id] = snapshot
        _advance_exchange_selector_next_version(fragment_state, snapshot.source_node_id, snapshot.version)
        return True, {
            int(partition_id) for partition_id, entry in snapshot.selected.items() if entry.get("split_key") is not None
        }
    if snapshot.version < current.version:
        return False, set()
    if snapshot.version == current.version:
        try:
            merged, _ = _merge_exchange_selector_snapshot(current, snapshot)
            _validate_exchange_selector_snapshot(merged)
            if merged.semantic_key() == current.semantic_key():
                return False, set()
        except ValueError as exc:
            raise ValueError(
                f"conflicting exchange selector update for source {snapshot.source_node_id} "
                f"at version {snapshot.version}"
            ) from exc
        raise ValueError(
            f"conflicting exchange selector update for source {snapshot.source_node_id} at version {snapshot.version}"
        )
    merged, new_partition_ids = _merge_exchange_selector_snapshot(current, snapshot)
    _validate_exchange_selector_snapshot(merged)
    if current.final:
        if merged.semantic_key() == current.semantic_key():
            return False, set()
        raise ValueError(f"cannot update final exchange selector for source {snapshot.source_node_id}")
    fragment_state.exchange_source_selectors_by_source[snapshot.source_node_id] = merged
    _advance_exchange_selector_next_version(fragment_state, snapshot.source_node_id, snapshot.version)
    return True, new_partition_ids


def _fragment_plan_ref(query_id: str, fragment_id: str, plan: Any) -> Any:
    query_key = str(query_id or "").strip()
    if not query_key:
        raise ValueError("fragment plan cache requires a non-empty query_id")
    if plan is None or isinstance(plan, ray.ObjectRef):
        return plan
    cache_key = (
        _ray_fragment_plan_cache_session_key(),
        query_key,
        str(fragment_id),
    )
    with _FRAGMENT_PLAN_REF_CACHE_LOCK:
        existing = _FRAGMENT_PLAN_REF_CACHE.get(cache_key)
        if existing is not None:
            return existing
        ref = ray.put(plan)
        _FRAGMENT_PLAN_REF_CACHE[cache_key] = ref
        return ref


def _drop_fragment_plan_refs_for_query(query_id: str) -> int:
    query_key = str(query_id or "").strip()
    if not query_key:
        return 0
    removed = 0
    with _FRAGMENT_PLAN_REF_CACHE_LOCK:
        for cache_key in list(_FRAGMENT_PLAN_REF_CACHE):
            _, owner_query_id, _ = cache_key
            if owner_query_id == query_key:
                _FRAGMENT_PLAN_REF_CACHE.pop(cache_key, None)
                removed += 1
    return removed


def _stop_fte_status_watchers(query_id: str | None = None) -> None:
    query_key = None if query_id is None else str(query_id)
    while True:
        with _FTE_REGISTRY_LOCK:
            watcher_items = [
                (attempt_key, watcher)
                for attempt_key, watcher in _FTE_STATUS_WATCHERS.items()
                if query_key is None or FteTaskAttemptId.parse(attempt_key).query_id == query_key
            ]
        if not watcher_items:
            return
        for _, watcher in watcher_items:
            watcher.stop()
        alive: list[str] = []
        for attempt_key, watcher in watcher_items:
            watcher.join(watcher.shutdown_timeout_s())
            if watcher.is_alive():
                alive.append(attempt_key)
                continue
            with _FTE_REGISTRY_LOCK:
                if _FTE_STATUS_WATCHERS.get(attempt_key) is watcher:
                    _FTE_STATUS_WATCHERS.pop(attempt_key, None)
        if alive:
            raise RuntimeError(
                "FTE status watcher shutdown timed out without losing ownership: " + ", ".join(sorted(alive))
            )


def close_fte_registry_for_query(query_id: str) -> None:
    query_key = str(query_id or "").strip()
    if not query_key:
        return
    with _FTE_REGISTRY_CONDITION:
        _FTE_CLOSING_QUERIES.add(query_key)
        _FTE_REGISTRY_CONDITION.notify_all()


def begin_fte_registry_operation(query_id: str) -> bool:
    """Acquire one query lifecycle read-side ownership token."""
    query_key = str(query_id or "").strip()
    if not query_key:
        raise ValueError("query_id must be non-empty")
    with _FTE_REGISTRY_CONDITION:
        if query_key in _FTE_CLOSING_QUERIES:
            return False
        _FTE_ACTIVE_OPERATIONS_BY_QUERY[query_key] = _FTE_ACTIVE_OPERATIONS_BY_QUERY.get(query_key, 0) + 1
        return True


def begin_fte_registry_teardown_operation(query_id: str) -> None:
    """Own one remote teardown mutation while the query is closing."""
    query_key = str(query_id or "").strip()
    if not query_key:
        raise ValueError("query_id must be non-empty")
    with _FTE_REGISTRY_CONDITION:
        if query_key not in _FTE_CLOSING_QUERIES:
            raise RuntimeError(f"FTE teardown operation requires a closing query: {query_key}")
        _FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY[query_key] = (
            _FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY.get(query_key, 0) + 1
        )


def end_fte_registry_operation(query_id: str) -> None:
    query_key = str(query_id or "").strip()
    with _FTE_REGISTRY_CONDITION:
        active = _FTE_ACTIVE_OPERATIONS_BY_QUERY.get(query_key, 0)
        if active <= 0:
            raise RuntimeError(f"FTE registry operation ownership underflow: {query_key}")
        if active == 1:
            _FTE_ACTIVE_OPERATIONS_BY_QUERY.pop(query_key, None)
        else:
            _FTE_ACTIVE_OPERATIONS_BY_QUERY[query_key] = active - 1
        _FTE_REGISTRY_CONDITION.notify_all()


def end_fte_registry_teardown_operation(query_id: str) -> None:
    query_key = str(query_id or "").strip()
    with _FTE_REGISTRY_CONDITION:
        active = _FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY.get(query_key, 0)
        if active <= 0:
            raise RuntimeError(f"FTE registry teardown ownership underflow: {query_key}")
        if active == 1:
            _FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY.pop(query_key, None)
        else:
            _FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY[query_key] = active - 1
        _FTE_REGISTRY_CONDITION.notify_all()


def _transfer_fte_registry_operations_to_ref(
    query_ids: list[str] | tuple[str, ...] | set[str],
    object_ref: Any,
    *,
    on_success: Any = None,
    on_failure: Any = None,
    end_operation: Any,
) -> None:
    """Keep acquired lifecycle tokens until the remote mutation is terminal."""
    owned_queries = tuple(dict.fromkeys(str(query_id) for query_id in query_ids))
    future_method = getattr(object_ref, "future", None)
    if not callable(future_method):
        raise TypeError("tracked FTE registry mutation must return an ObjectRef with future()")
    completion = future_method()
    add_done_callback = getattr(completion, "add_done_callback", None)
    if not callable(add_done_callback):
        raise TypeError("tracked FTE registry mutation future must provide add_done_callback()")

    def release_operations(_completed: Any) -> None:
        failed = False
        try:
            _completed.result()
        except BaseException:
            failed = True
        try:
            callback = on_failure if failed else on_success
            if callable(callback):
                callback()
        finally:
            for query_id in owned_queries:
                end_operation(query_id)

    add_done_callback(release_operations)


def transfer_fte_registry_operations_to_ref(
    query_ids: list[str] | tuple[str, ...] | set[str],
    object_ref: Any,
    *,
    on_success: Any = None,
    on_failure: Any = None,
) -> None:
    _transfer_fte_registry_operations_to_ref(
        query_ids,
        object_ref,
        on_success=on_success,
        on_failure=on_failure,
        end_operation=end_fte_registry_operation,
    )


def transfer_fte_registry_teardown_operations_to_ref(
    query_ids: list[str] | tuple[str, ...] | set[str],
    object_ref: Any,
    *,
    on_success: Any = None,
    on_failure: Any = None,
) -> None:
    _transfer_fte_registry_operations_to_ref(
        query_ids,
        object_ref,
        on_success=on_success,
        on_failure=on_failure,
        end_operation=end_fte_registry_teardown_operation,
    )


def quiesce_fte_registry_for_query(query_id: str) -> None:
    """Join watcher ownership and every in-flight remote mutation before drop."""
    query_key = str(query_id or "").strip()
    if not query_key:
        return
    close_fte_registry_for_query(query_key)
    _stop_fte_status_watchers(query_key)
    deadline = time.monotonic() + _fte_control_rpc_timeout_s()
    with _FTE_REGISTRY_CONDITION:
        while _FTE_ACTIVE_OPERATIONS_BY_QUERY.get(query_key, 0) > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "FTE registry operation shutdown timed out without losing ownership: "
                    f"query={query_key} active="
                    f"{_FTE_ACTIVE_OPERATIONS_BY_QUERY.get(query_key, 0)}"
                )
            _FTE_REGISTRY_CONDITION.wait(remaining)
    # An operation admitted before close may publish its watcher while the
    # first join is running.  No new operation can start after close, so this
    # second pass is the final watcher ownership cut.
    _stop_fte_status_watchers(query_key)


def open_fte_registry_for_query(query_id: str) -> None:
    query_key = str(query_id or "").strip()
    if not query_key:
        raise ValueError("query_id must be non-empty")
    with _FTE_REGISTRY_LOCK:
        stale_owners = [
            name
            for name, present in (
                ("fragment_executions", any(key[0] == query_key for key in _FTE_FRAGMENT_EXECUTIONS)),
                ("fragment_execution_ids", any(key[0] == query_key for key in _FTE_FRAGMENT_EXECUTION_IDS)),
                (
                    "fragment_progress_topologies",
                    any(key[0] == query_key for key in _FTE_FRAGMENT_PROGRESS_TOPOLOGIES),
                ),
                (
                    "fragment_progress_topology_builds",
                    any(key[0] == query_key for key in _FTE_FRAGMENT_PROGRESS_TOPOLOGY_BUILDS),
                ),
                ("fragment_states", any(key[0] == query_key for key in _FTE_FRAGMENT_STATES)),
                ("partition_owners", any(key[0] == query_key for key in _FTE_PARTITION_OWNERS)),
                ("partition_task_leases", any(key[0] == query_key for key in _FTE_PARTITION_TASK_LEASES)),
                ("partition_task_waiters", any(key[0] == query_key for key in _FTE_PARTITION_TASK_WAITERS)),
                (
                    "stage_submission_probes",
                    any(partition_key[0] == query_key for partition_key in _FTE_STAGE_SUBMISSION_PROBES.values()),
                ),
                (
                    "stage_submission_blocks",
                    any(stage_key[0] == query_key for stage_key in _FTE_STAGE_SUBMISSION_BLOCKS),
                ),
                ("pending_worker_reservations", any(key[0] == query_key for key in _FTE_PENDING_WORKER_RESERVATIONS)),
                (
                    "worker_reservation_generations",
                    any(key[0] == query_key for key in _FTE_WORKER_RESERVATION_GENERATIONS),
                ),
                ("sequences", any(key[0] == query_key for key in _FTE_SEQUENCES)),
                ("result_handles", query_key in _FTE_RESULT_HANDLES_BY_QUERY),
                (
                    "status_watchers",
                    any(FteTaskAttemptId.parse(key).query_id == query_key for key in _FTE_STATUS_WATCHERS),
                ),
                ("active_operations", _FTE_ACTIVE_OPERATIONS_BY_QUERY.get(query_key, 0) > 0),
                (
                    "active_teardown_operations",
                    _FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY.get(query_key, 0) > 0,
                ),
                ("retry_delay", query_key in _FTE_RETRY_DELAYS),
                ("fragment_execution_counter", query_key in _FTE_QUERY_NEXT_FRAGMENT_EXECUTION_ID),
                ("event_scheduler", _FTE_SCHEDULERS.get(query_key) is not None),
                (
                    "fragment_plan_refs",
                    _fragment_plan_cache_has_query(query_key),
                ),
                (
                    "worker_fragment_registrations",
                    any(
                        _worker_has_fragment_registration_state_for_query(
                            handle,
                            query_key,
                        )
                        for handle in _FTE_WORKER_HANDLES.values()
                        if handle is not None
                    ),
                ),
                (
                    "worker_controls",
                    any(
                        _worker_has_fte_control_state_for_query(handle, query_key)
                        for handle in _FTE_WORKER_HANDLES.values()
                        if handle is not None
                    ),
                ),
                (
                    "worker_teardowns",
                    any(
                        _worker_has_fte_teardown_state_for_query(
                            handle,
                            query_key,
                        )
                        for handle in _FTE_WORKER_HANDLES.values()
                        if handle is not None
                    ),
                ),
            )
            if present
        ]
        if stale_owners:
            raise RuntimeError(
                f"cannot reopen FTE registry with old generation state: {query_key}; " + ", ".join(stale_owners)
            )
        _FTE_CLOSING_QUERIES.discard(query_key)


def fte_registry_query_is_closing(query_id: str) -> bool:
    with _FTE_REGISTRY_LOCK:
        return str(query_id) in _FTE_CLOSING_QUERIES


def _worker_has_fte_control_state_for_query(handle: Any, query_id: str) -> bool:
    has_state = getattr(handle, "_has_fte_control_state_for_query", None)
    return callable(has_state) and bool(has_state(query_id))


def _worker_has_fragment_registration_state_for_query(
    handle: Any,
    query_id: str,
) -> bool:
    has_state = getattr(
        handle,
        "_has_fragment_registration_state_for_query",
        None,
    )
    return callable(has_state) and bool(has_state(query_id))


def _worker_has_fte_teardown_state_for_query(handle: Any, query_id: str) -> bool:
    has_state = getattr(handle, "_has_fte_teardown_state_for_query", None)
    return callable(has_state) and bool(has_state(query_id))


def _fragment_plan_cache_has_query(query_id: str) -> bool:
    with _FRAGMENT_PLAN_REF_CACHE_LOCK:
        return any(owner_query_id == query_id for _, owner_query_id, _ in _FRAGMENT_PLAN_REF_CACHE)


def fte_query_remote_teardown_blockers(query_id: str) -> tuple[str, ...]:
    """Return remote/future owners that make local teardown non-final."""
    query_key = str(query_id or "").strip()
    if not query_key:
        return ()
    with _FTE_REGISTRY_LOCK:
        blockers: list[str] = []
        ingress_count = _FTE_ACTIVE_OPERATIONS_BY_QUERY.get(query_key, 0)
        if ingress_count:
            blockers.append(f"active_ingress={ingress_count}")
        teardown_count = _FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY.get(
            query_key,
            0,
        )
        if teardown_count:
            blockers.append(f"active_teardown={teardown_count}")
        for worker_id, handle in _FTE_WORKER_HANDLES.items():
            if handle is None:
                continue
            if _worker_has_fte_control_state_for_query(handle, query_key):
                blockers.append(f"worker_control={worker_id}")
            if _worker_has_fte_teardown_state_for_query(handle, query_key):
                blockers.append(f"worker_teardown={worker_id}")
        return tuple(blockers)


def _drop_fte_registry_for_query(query_id: str) -> None:
    query_id = str(query_id or "")
    if not query_id:
        return
    quiesce_fte_registry_for_query(query_id)
    FteWorkerPlacementManager.release_query(query_id)
    # Query-resource and reservation capacity may already be visible.  Signal
    # before the remaining local cleanup so a later teardown error cannot
    # strand unrelated queries.
    request_fte_pending_task_drain()
    pending_worker_reservation_futures: list[FteWorkerReservationFuture] = []
    worker_handles: list[Any] = []
    with _FTE_REGISTRY_LOCK:
        for key in list(_FTE_FRAGMENT_EXECUTIONS):
            if key[0] == query_id:
                _FTE_FRAGMENT_EXECUTIONS.pop(key, None)
        for key in list(_FTE_FRAGMENT_EXECUTION_IDS):
            if key[0] == query_id:
                _FTE_FRAGMENT_EXECUTION_IDS.pop(key, None)
        for key in list(_FTE_FRAGMENT_PROGRESS_TOPOLOGIES):
            if key[0] == query_id:
                _FTE_FRAGMENT_PROGRESS_TOPOLOGIES.pop(key, None)
        _FTE_FRAGMENT_PROGRESS_TOPOLOGY_BUILDS.difference_update(
            {key for key in _FTE_FRAGMENT_PROGRESS_TOPOLOGY_BUILDS if key[0] == query_id}
        )
        for key in list(_FTE_WORKER_RESERVATION_GENERATIONS):
            if key[0] == query_id:
                _FTE_WORKER_RESERVATION_GENERATIONS.pop(key, None)
        for key in list(_FTE_PENDING_WORKER_RESERVATIONS):
            if key[0] == query_id:
                future = _FTE_PENDING_WORKER_RESERVATIONS.pop(key, None)
                if future is not None:
                    pending_worker_reservation_futures.append(future)
        _FTE_RESULT_HANDLES_BY_QUERY.pop(query_id, None)
        for key in list(_FTE_SEQUENCES):
            if key[0] == query_id:
                _FTE_SEQUENCES.pop(key, None)
        for key in list(_FTE_FRAGMENT_STATES):
            if key[0] == query_id:
                _FTE_FRAGMENT_STATES.pop(key, None)
        _FTE_RETRY_DELAYS.pop(query_id, None)
        _FTE_QUERY_NEXT_FRAGMENT_EXECUTION_ID.pop(query_id, None)
        _FTE_ACTIVE_OPERATIONS_BY_QUERY.pop(query_id, None)
        worker_handles = [handle for handle in _FTE_WORKER_HANDLES.values() if handle is not None]
        _FTE_REGISTRY_CONDITION.notify_all()
    _FTE_SCHEDULERS.drop_query(query_id)
    try:
        for handle in worker_handles:
            handle._drop_fte_state_for_query(query_id)
    finally:
        # Worker-local running pressure is released by the cleanup above, not
        # by release_query().  Keep this wake in a finally block so partial
        # cleanup cannot strand capacity either.
        request_fte_pending_task_drain()
    for future in pending_worker_reservation_futures:
        future.cancel()


def _store_fte_result_handles(
    query_id: str,
    handles: list[Any],
    *,
    registry_operation_owned: bool = False,
) -> None:
    if not handles:
        return
    with _FTE_REGISTRY_LOCK:
        if not registry_operation_owned and str(query_id) in _FTE_CLOSING_QUERIES:
            raise RuntimeError(f"FTE query registry is closing: {query_id}")
        _FTE_RESULT_HANDLES_BY_QUERY.setdefault(str(query_id), []).extend(handles)


def request_fte_pending_task_drain() -> list[Any]:
    """Signal and, when elected leader, run the single-flight admission pump."""
    from duckdb.runners.fte.fte_events import ResourceAdmissionChanged

    def drain_query(
        query_id: str,
        execution_class: FteTaskExecutionClass,
    ) -> list[Any]:
        if not begin_fte_registry_operation(query_id):
            return []
        try:
            scheduler = _FTE_SCHEDULERS.get(query_id)
            if scheduler is None:
                return []
            scheduler.enqueue(
                ResourceAdmissionChanged(
                    query_id,
                    execution_class=execution_class.value,
                    internal=True,
                )
            )
            return scheduler.drain()
        finally:
            end_fte_registry_operation(query_id)

    def drain_round(query_ids: list[str]) -> list[Any]:
        handles: list[Any] = []

        def drain_execution_class(execution_class: FteTaskExecutionClass) -> None:
            for query_id in query_ids:
                scheduler = _FTE_SCHEDULERS.get(query_id)
                if scheduler is None or scheduler.stats().state != "RUNNING":
                    continue
                try:
                    handles.extend(drain_query(query_id, execution_class))
                except Exception as exc:
                    scheduler.fail(f"FTE admission failed: {exc}")

        drain_execution_class(FteTaskExecutionClass.EAGER_SPECULATIVE)
        drain_execution_class(FteTaskExecutionClass.STANDARD)
        with _FTE_REGISTRY_LOCK:
            candidate_fragment_execution_items = [
                (key, fragment_execution)
                for key, fragment_execution in _FTE_FRAGMENT_EXECUTIONS.items()
                if key[0] not in _FTE_CLOSING_QUERIES
                and callable(getattr(fragment_execution, "has_pending_partitions", None))
            ]
        if not _has_fte_pending_standard_partitions_for_active_queries(candidate_fragment_execution_items):
            drain_execution_class(FteTaskExecutionClass.SPECULATIVE)
        return handles

    return _FTE_SCHEDULERS.run_pending_drain(drain_round)


def _fte_execution_queries_waiting_for_resource(
    resource_query_id: str,
) -> tuple[str, ...]:
    resource_query_key = str(resource_query_id)
    with _FTE_REGISTRY_LOCK:
        closing_queries = set(_FTE_CLOSING_QUERIES)
        fragment_execution_items = tuple(_FTE_FRAGMENT_EXECUTIONS.items())
        execution_query_ids = {
            execution_query_id
            for (
                execution_query_id,
                _fragment_id,
                _partition_id,
            ), waiter_identity in _FTE_PARTITION_TASK_WAITERS.items()
            if waiter_identity[0] == resource_query_key and execution_query_id not in closing_queries
        }
    # Do not hold the registry lock while taking a fragment state lock.  FTE
    # admission callbacks take those locks in the opposite order.
    for (execution_query_id, _fragment_id), fragment_execution in fragment_execution_items:
        if execution_query_id in closing_queries:
            continue
        scheduler = _FTE_SCHEDULERS.get(execution_query_id)
        if scheduler is None or scheduler.stats().state != "RUNNING":
            continue
        try:
            owner_query_id, _stage_id = resource_identity_from_context(fragment_execution.context)
            if owner_query_id != resource_query_key:
                continue
            has_pending_partitions = fragment_execution.has_pending_partitions()
        except Exception as exc:
            scheduler.fail(f"FTE resource admission scan failed: {exc}")
            continue
        if has_pending_partitions:
            execution_query_ids.add(execution_query_id)
    return tuple(
        query_id
        for query_id in sorted(execution_query_ids)
        if (scheduler := _FTE_SCHEDULERS.get(query_id)) is not None and scheduler.stats().state == "RUNNING"
    )


def drain_fte_resource_admission_change(query_id: str) -> list[Any]:
    """Event-drive pending FTE descriptors after a QRM mutation.

    The driver calls this outside the QRM lock on an owned background task.
    Scheduler draining is serialized by ``FteQueryScheduler`` itself, so a
    concurrent status/event drain either handles this event or leaves it queued
    for the active drainer.
    """
    if not _fte_execution_queries_waiting_for_resource(query_id):
        return []
    return request_fte_pending_task_drain()


def has_fte_resource_admission_waiter(query_id: str) -> bool:
    # Kept as a compatibility name for the driver bridge.  Demand now includes
    # passive FTE descriptors; FTE task waiters are intentionally absent from
    # QRM when the execution window is full.
    return bool(_fte_execution_queries_waiting_for_resource(query_id))


def fte_execution_query_ids_for_resource(
    resource_query_id: str,
) -> tuple[str, ...]:
    """Return every FTE lifecycle query owned by one resource query."""
    resource_query_key = str(resource_query_id or "").strip()
    if not resource_query_key:
        raise ValueError("resource_query_id must be non-empty")
    with _FTE_REGISTRY_LOCK:
        fragment_executions = tuple(_FTE_FRAGMENT_EXECUTIONS.items())
    execution_query_ids = {
        execution_query_id
        for (
            execution_query_id,
            _fragment_id,
        ), fragment_execution in fragment_executions
        if resource_identity_from_context(fragment_execution.context)[0] == resource_query_key
    }
    return tuple(sorted(execution_query_ids))


def _worker_id_host_and_index(worker_id: str) -> tuple[str, str]:
    worker_id = str(worker_id or "")
    if "#" not in worker_id:
        return worker_id, ""
    host_id, worker_index = worker_id.rsplit("#", 1)
    return host_id, worker_index


def _node_requirements_host(node_requirements: NodeRequirements | Mapping[str, Any] | None) -> str | None:
    if node_requirements is None:
        return None
    if isinstance(node_requirements, Mapping):
        value = node_requirements.get("host", node_requirements.get("address"))
    else:
        value = node_requirements.host
    if value is None:
        return None
    host = str(value).strip()
    return host or None


def _node_requirements_remotely_accessible(
    node_requirements: NodeRequirements | Mapping[str, Any] | None,
) -> bool:
    if node_requirements is None:
        return True
    if isinstance(node_requirements, Mapping):
        return bool(node_requirements.get("remotely_accessible", True))
    return bool(node_requirements.remotely_accessible)


def _fte_retry_delayer_for_query(query_id: str) -> _FteSchedulingDelayer:
    query_id = str(query_id)
    with _FTE_REGISTRY_LOCK:
        delayer = _FTE_RETRY_DELAYS.get(query_id)
        if delayer is None:
            delayer = _FteSchedulingDelayer(
                initial_delay_s=_fte_retry_initial_delay_s(),
                max_delay_s=_fte_retry_max_delay_s(),
                scale_factor=_fte_retry_delay_scale_factor(),
            )
            _FTE_RETRY_DELAYS[query_id] = delayer
        return delayer


def _start_or_prolong_fte_retry_delay(query_id: str) -> None:
    with _FTE_REGISTRY_LOCK:
        _fte_retry_delayer_for_query(query_id).start_or_prolong_delay_if_necessary()


def _fte_retry_remaining_delay_s(query_id: str) -> float:
    with _FTE_REGISTRY_LOCK:
        delayer = _FTE_RETRY_DELAYS.get(str(query_id))
        if delayer is None:
            return 0.0
        return delayer.remaining_delay_s()


def _worker_matches_node_requirements(
    handle: RayWorkerActorHandle,
    node_requirements: NodeRequirements | Mapping[str, Any] | None,
) -> bool:
    host = _node_requirements_host(node_requirements)
    if host is None:
        return True
    worker_host, _ = _worker_id_host_and_index(str(handle.worker_id))
    return worker_host == host


def _filter_workers_for_node_requirements(
    workers: list[RayWorkerActorHandle],
    node_requirements: NodeRequirements | Mapping[str, Any] | None,
    *,
    node_requirements_wait_started_at: float | None = None,
) -> list[RayWorkerActorHandle]:
    host = _node_requirements_host(node_requirements)
    if host is None:
        return workers
    matching = [worker for worker in workers if _worker_matches_node_requirements(worker, node_requirements)]
    if not _node_requirements_remotely_accessible(node_requirements):
        return matching
    if not matching:
        return workers
    wait_started = (
        time.time() if node_requirements_wait_started_at is None else float(node_requirements_wait_started_at)
    )
    if time.time() - wait_started < _fte_exhausted_node_wait_period_s():
        return matching
    return workers


def _node_requirements_preference_penalty(
    handle: RayWorkerActorHandle,
    node_requirements: NodeRequirements | Mapping[str, Any] | None,
) -> int:
    host = _node_requirements_host(node_requirements)
    if host is None:
        return 0
    return 0 if _worker_matches_node_requirements(handle, node_requirements) else 1


def _node_requirements_have_candidates(
    workers: list[RayWorkerActorHandle],
    node_requirements: NodeRequirements | Mapping[str, Any] | None,
    *,
    node_requirements_wait_started_at: float | None = None,
) -> bool:
    return bool(
        _filter_workers_for_node_requirements(
            workers,
            node_requirements,
            node_requirements_wait_started_at=node_requirements_wait_started_at,
        )
    )


def _memory_requirement_bytes(value: Any) -> int:
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _is_write_sink_fragment(fragment_execution: Any) -> bool:
    sink_keys = {
        "copy_output_base",
        "copy_output_run_id",
        "copy_output_remote_base",
        "sink_node_id",
        "copy_sink_node_id",
    }
    for payload in (
        getattr(fragment_execution, "context", None),
        getattr(fragment_execution, "task_context_info", None),
    ):
        if isinstance(payload, Mapping) and any(key in payload for key in sink_keys):
            return True
    return False


def _write_sink_has_input(fragment_execution: Any) -> tuple[bool, bool]:
    partitions = getattr(fragment_execution, "partitions", {}) or {}
    if not partitions:
        return False, False
    all_terminal = True
    for partition in partitions.values():
        finished = bool(getattr(partition, "finished", False))
        failed = bool(getattr(partition, "failed", False))
        all_terminal = all_terminal and (finished or failed)
        if finished or failed:
            continue
        if (
            bool(getattr(partition, "sealed", False))
            or bool(getattr(partition, "ready_for_scheduling", False))
            or bool(getattr(partition, "running_attempts", {}))
            or bool(getattr(partition, "execution_ready_deferred", False))
            or getattr(partition, "node_wait_started_at", None) is not None
        ):
            return True, False
    return False, all_terminal


def _sync_write_sink_stage_for_fragment(fragment_execution: Any) -> str | None:
    if not _is_write_sink_fragment(fragment_execution):
        return None

    from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

    resource_query_id, stage_id = resource_identity_from_context(fragment_execution.context)
    has_input, all_terminal = _write_sink_has_input(fragment_execution)
    get_query_resource_manager(resource_query_id).update_stage_state(
        stage_id,
        runnable=has_input,
        completed=all_terminal,
    )
    return stage_id


def _fte_partition_resource_key(query_id: str, fragment_id: str, partition_id: int) -> tuple[str, str, int]:
    return (str(query_id), str(fragment_id), int(partition_id))


def _fte_fragment_resource_identity(
    query_id: str,
    fragment_id: str,
) -> tuple[str, str]:
    with _FTE_REGISTRY_LOCK:
        fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((str(query_id), str(fragment_id)))
    if fragment_execution is None:
        raise KeyError(f"FTE fragment execution {query_id}/{fragment_id} is not registered")
    return resource_identity_from_context(fragment_execution.context)


def _fte_partition_fragment_execution_id(query_id: str, fragment_id: str, partition_id: int) -> int:
    key = _fte_partition_resource_key(query_id, fragment_id, partition_id)
    with _FTE_REGISTRY_LOCK:
        existing_lease = _FTE_PARTITION_TASK_LEASES.get(key)
        if existing_lease is not None:
            return int(existing_lease[0])
        pending = _FTE_PENDING_WORKER_RESERVATIONS.get(key)
        if pending is not None:
            return int(pending.fragment_execution_id)
        fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((key[0], key[1]))
        if fragment_execution is not None:
            return int(fragment_execution.fragment_execution_id)
    raise KeyError(f"FTE fragment execution {key[0]}/{key[1]} is not registered")


def _fte_partition_attempt_identity(
    query_id: str,
    fragment_execution_id: int,
    fragment_id: str,
    partition_id: int,
) -> tuple[str, str]:
    fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((str(query_id), str(fragment_id)))
    if fragment_execution is None:
        raise KeyError(f"FTE fragment execution {query_id}/{fragment_id} is not registered")
    if int(fragment_execution.fragment_execution_id) != int(fragment_execution_id):
        raise RuntimeError(
            f"FTE fragment execution id mismatch for {query_id}/{fragment_id}: "
            f"registered={fragment_execution.fragment_execution_id} requested={fragment_execution_id}"
        )
    partition = fragment_execution.partitions.get(int(partition_id))
    if partition is None:
        raise KeyError(f"FTE partition {query_id}/{fragment_id}/{partition_id} is not registered")
    attempt = FteTaskAttemptId(partition.task_id, partition.next_attempt_number())
    return str(partition.task_id), str(attempt)


def _set_fte_pending_reservation_blocked_reason(
    query_id: str,
    fragment_id: str,
    partition_id: int,
    blocked_reason: str,
) -> None:
    key = _fte_partition_resource_key(query_id, fragment_id, partition_id)
    future = _FTE_PENDING_WORKER_RESERVATIONS.get(key)
    if future is not None:
        future.set_blocked_reason(blocked_reason)


def _acquire_fte_partition_task_lease(
    *,
    query_id: str,
    fragment_execution_id: int,
    fragment_id: str,
    partition_id: int,
    node_id: str,
) -> Any:
    from duckdb.runners.ray.query_resource_manager import TaskRequest
    from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

    with _FTE_REGISTRY_LOCK:
        if str(query_id) in _FTE_CLOSING_QUERIES:
            raise RuntimeError(f"FTE query registry is closing: {query_id}")
    resource_query_id, stage_id = _fte_fragment_resource_identity(
        query_id,
        fragment_id,
    )
    if not admit_fte_partition_submission(
        query_id,
        fragment_id,
        partition_id,
    ):
        blocked_reason = "submission_window"
        _set_fte_pending_reservation_blocked_reason(
            query_id,
            fragment_id,
            partition_id,
            blocked_reason,
        )
        raise FteWorkerReservationUnavailable(
            query_id=str(query_id),
            fragment_id=str(fragment_id),
            partition_id=int(partition_id),
            memory_requirement_bytes=0,
            blocked_reason=blocked_reason,
        )
    task_id, attempt_id = _fte_partition_attempt_identity(
        query_id,
        fragment_execution_id,
        fragment_id,
        partition_id,
    )
    manager = get_query_resource_manager(resource_query_id)
    request = TaskRequest(
        query_id=resource_query_id,
        stage_id=stage_id,
        task_id=task_id,
        attempt_id=attempt_id,
        node_id=str(node_id),
    )
    try:
        # This is a non-persistent descriptor probe.  QRM atomically arbitrates
        # it against persistent driver waiters, but a denial leaves ownership
        # with the FTE descriptor queue instead of creating one QRM waiter per
        # logical partition.
        grant = manager.try_acquire_task_descriptor(request)
    except BaseException:
        release_fte_partition_submission(
            query_id,
            fragment_id,
            partition_id,
        )
        raise
    if grant.granted:
        resolve_fte_partition_submission(
            query_id,
            fragment_id,
            partition_id,
            granted=True,
        )
        with _FTE_REGISTRY_LOCK:
            query_closing = str(query_id) in _FTE_CLOSING_QUERIES
        if query_closing:
            manager.abandon_task_lease(
                grant.lease.lease_id,
                attempt_id=grant.lease.attempt_id,
            )
            raise RuntimeError(f"FTE query registry is closing: {query_id}")
        return grant.lease
    _set_fte_pending_reservation_blocked_reason(
        query_id,
        fragment_id,
        partition_id,
        grant.blocked_reason,
    )
    resolve_fte_partition_submission(
        query_id,
        fragment_id,
        partition_id,
        granted=False,
        blocked_reason=grant.blocked_reason,
        fatal=grant.fatal,
        admission_epoch=getattr(grant, "admission_epoch", None),
    )
    if grant.fatal:
        raise RuntimeError(
            f"fatal FTE task lease rejection for {query_id}/{fragment_id}/{partition_id}: {grant.blocked_reason}"
        )
    raise FteWorkerReservationUnavailable(
        query_id=str(query_id),
        fragment_id=str(fragment_id),
        partition_id=int(partition_id),
        memory_requirement_bytes=0,
        blocked_reason=grant.blocked_reason,
    )


def _pop_fte_partition_task_lease(
    query_id: str,
    fragment_id: str,
    partition_id: int,
) -> tuple[int, Any] | None:
    with _FTE_REGISTRY_LOCK:
        return _FTE_PARTITION_TASK_LEASES.pop(
            _fte_partition_resource_key(query_id, fragment_id, partition_id),
            None,
        )


def _release_fte_partition_task_lease(lease: Any, *, terminal: bool) -> None:
    from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

    try:
        manager = get_query_resource_manager(lease.query_id)
        if terminal:
            manager.release_task_lease(lease.lease_id, attempt_id=lease.attempt_id)
        else:
            manager.abandon_task_lease(lease.lease_id, attempt_id=lease.attempt_id)
    except KeyError:
        # Query teardown cancels every live lease before it clears FTE registry state.
        return


def _release_fte_partition_task_lease_for_key(query_id: str, fragment_id: str, partition_id: int) -> None:
    release_fte_partition_submission(
        query_id,
        fragment_id,
        partition_id,
    )
    release_fte_partition_task_waiter(_fte_partition_resource_key(query_id, fragment_id, partition_id))
    lease_record = _pop_fte_partition_task_lease(query_id, fragment_id, partition_id)
    if lease_record is None:
        return
    _, lease = lease_record
    _release_fte_partition_task_lease(lease, terminal=False)


def _release_fte_partition_owner_for_attempt(attempt_id: Any) -> None:
    try:
        attempt = FteTaskAttemptId.coerce(attempt_id)
    except Exception:
        return
    query_id = attempt.task_id.query_id
    fragment_execution_id = int(attempt.task_id.fragment_execution_id)
    partition_id = int(attempt.task_id.partition_id)
    attempt_key = str(attempt)
    with _FTE_REGISTRY_LOCK:
        candidates = [
            fragment_id
            for (
                candidate_query_id,
                fragment_id,
                candidate_partition_id,
            ), lease_record in _FTE_PARTITION_TASK_LEASES.items()
            if candidate_query_id == query_id
            and int(candidate_partition_id) == partition_id
            and int(lease_record[0]) == fragment_execution_id
            and str(lease_record[1].attempt_id) == attempt_key
        ]
    for fragment_id in candidates:
        FteWorkerPlacementManager.release_owner(
            query_id=query_id,
            fragment_id=fragment_id,
            partition_id=partition_id,
            terminal=True,
        )


def fte_partition_task_lease_payload(
    query_id: str,
    fragment_id: str,
    partition_id: int,
    attempt_id: Any,
) -> dict[str, Any]:
    key = _fte_partition_resource_key(query_id, fragment_id, partition_id)
    with _FTE_REGISTRY_LOCK:
        lease_record = _FTE_PARTITION_TASK_LEASES.get(key)
    if lease_record is None:
        raise RuntimeError(f"FTE task {query_id}/{fragment_id}/{partition_id} has no task lease")
    _, lease = lease_record
    actual_attempt_id = str(FteTaskAttemptId.coerce(attempt_id))
    if lease.attempt_id != actual_attempt_id:
        raise RuntimeError(f"FTE task lease attempt mismatch: lease={lease.attempt_id} command={actual_attempt_id}")
    from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

    stage = get_query_resource_manager(lease.query_id).graph.stage_by_id(lease.stage_id)
    return {
        "lease_id": lease.lease_id,
        "query_id": lease.query_id,
        "execution_query_id": str(query_id),
        "stage_id": lease.stage_id,
        "task_id": lease.task_id,
        "attempt_id": lease.attempt_id,
        "node_id": lease.node_id,
        "resources": lease.resources.to_dict(),
        "output_window_bytes": lease.output_window_bytes,
        "target_output_block_bytes": stage.target_output_block_bytes,
        "liveness": lease.liveness,
        "allocation_generation": lease.allocation_generation,
    }


def _install_fte_partition_terminal_lease_release_hook(owner: Any) -> None:
    record_terminal = getattr(owner, "record_fte_task_terminal", None)
    if not callable(record_terminal):
        return
    if getattr(owner, "_fte_partition_task_lease_release_hook_installed", False):
        return

    def record_terminal_with_lease_release(attempt_id: Any, *args: Any, **kwargs: Any) -> Any:
        _release_fte_partition_owner_for_attempt(attempt_id)
        return record_terminal(attempt_id, *args, **kwargs)

    owner.record_fte_task_terminal = record_terminal_with_lease_release
    owner._fte_partition_task_lease_release_hook_installed = True


def _required_fte_pressure_stats(handle: RayWorkerActorHandle) -> Mapping[str, Any]:
    stats = handle.fte_pressure_stats()
    if not isinstance(stats, Mapping):
        raise TypeError("worker fte_pressure_stats must return a mapping")
    return stats


def _fte_effective_worker_memory_budget_bytes(
    handle: RayWorkerActorHandle,
    execution_class: FteTaskExecutionClass | str | None,
) -> int:
    del execution_class
    budget_bytes = int(handle.memory_capacity_bytes)
    if budget_bytes <= 0:
        raise RuntimeError(
            f"Ray worker {handle.worker_id or '<unknown>'} has invalid logical memory capacity {budget_bytes}"
        )
    return budget_bytes


def _fte_pressure_total_memory_bytes(stats: Mapping[str, Any]) -> int:
    return int(
        stats.get(
            "total_memory_bytes",
            int(stats.get("assigned_memory_bytes", 0)) + int(stats.get("reserved_memory_bytes", 0)),
        )
    )


def _fte_pressure_capacity_memory_bytes(
    stats: Mapping[str, Any],
    execution_class: FteTaskExecutionClass | str | None,
) -> int:
    del execution_class
    return _fte_pressure_total_memory_bytes(stats)


def _fte_worker_has_memory_capacity(
    handle: RayWorkerActorHandle,
    *,
    memory_requirement_bytes: Any = None,
    execution_class: FteTaskExecutionClass | str | None = None,
) -> bool:
    budget_bytes = _fte_effective_worker_memory_budget_bytes(handle, execution_class)
    stats = _required_fte_pressure_stats(handle)
    current_memory_bytes = _fte_pressure_capacity_memory_bytes(
        stats,
        execution_class,
    )
    memory_after_bytes = current_memory_bytes + _memory_requirement_bytes(memory_requirement_bytes)
    return memory_after_bytes <= budget_bytes


def _fte_worker_selection_key(
    handle: RayWorkerActorHandle,
    *,
    memory_requirement_bytes: Any = None,
    execution_class: FteTaskExecutionClass | str | None = None,
    node_requirements: NodeRequirements | Mapping[str, Any] | None = None,
) -> tuple[int, int, int, int, int, int, str]:
    stats = _required_fte_pressure_stats(handle)
    running = int(stats.get("running_attempt_count", 0)) + int(stats.get("reserved_partition_count", 0))
    current_memory_bytes = _fte_pressure_capacity_memory_bytes(
        stats,
        execution_class,
    )
    required_memory_bytes = _memory_requirement_bytes(memory_requirement_bytes)
    memory_after_bytes = current_memory_bytes + required_memory_bytes
    budget_bytes = _fte_effective_worker_memory_budget_bytes(handle, execution_class)
    over_budget = 1 if memory_after_bytes > budget_bytes else 0
    split_bytes = int(stats.get("assigned_split_bytes", 0))
    split_count = int(stats.get("assigned_split_count", 0))
    return (
        over_budget,
        _node_requirements_preference_penalty(handle, node_requirements),
        running,
        memory_after_bytes,
        split_bytes,
        split_count,
        str(handle.worker_id),
    )


def _fte_partition_memory_requirement_bytes(owner_key: tuple[str, str, int]) -> int | None:
    fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((owner_key[0], owner_key[1]))
    if fragment_execution is None:
        return None
    partition = fragment_execution.partitions.get(owner_key[2])
    if partition is None:
        return None
    memory_requirement = partition.memory_requirement_bytes
    if memory_requirement is None:
        return None
    return _memory_requirement_bytes(memory_requirement)


def _fte_partition_execution_class(owner_key: tuple[str, str, int]) -> FteTaskExecutionClass:
    fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((owner_key[0], owner_key[1]))
    if fragment_execution is None:
        return FteTaskExecutionClass.STANDARD
    partition = fragment_execution.partitions.get(owner_key[2])
    if partition is None:
        return FteTaskExecutionClass.STANDARD
    return FteTaskExecutionClass.coerce(partition.execution_class)


def _fte_partition_node_requirements(owner_key: tuple[str, str, int]) -> NodeRequirements | None:
    fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((owner_key[0], owner_key[1]))
    if fragment_execution is None:
        return None
    partition = fragment_execution.partitions.get(owner_key[2])
    if partition is None:
        return None
    return partition.node_requirements


def _select_replacement_fte_worker(
    exclude_worker_id: str | set[str],
    *,
    memory_requirement_bytes: Any = None,
    execution_class: FteTaskExecutionClass | str | None = None,
    node_requirements: NodeRequirements | Mapping[str, Any] | None = None,
    node_requirements_wait_started_at: float | None = None,
) -> RayWorkerActorHandle | None:
    exclude_worker_ids = (
        {str(exclude_worker_id)}
        if isinstance(exclude_worker_id, str)
        else {str(worker_id) for worker_id in exclude_worker_id}
    )
    candidates = [
        handle
        for worker_id, handle in sorted(_FTE_WORKER_HANDLES.items())
        if str(worker_id) not in exclude_worker_ids and handle is not None and handle._fte_healthy
    ]
    if not candidates:
        return None
    candidates = _filter_workers_for_node_requirements(
        candidates,
        node_requirements,
        node_requirements_wait_started_at=node_requirements_wait_started_at,
    )
    if not candidates:
        return None
    candidates = [
        handle
        for handle in candidates
        if _fte_worker_has_memory_capacity(
            handle,
            memory_requirement_bytes=memory_requirement_bytes,
            execution_class=execution_class,
        )
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda handle: _fte_worker_selection_key(
            handle,
            memory_requirement_bytes=memory_requirement_bytes,
            execution_class=execution_class,
            node_requirements=node_requirements,
        ),
    )


def _has_replacement_fte_worker(
    exclude_worker_id: str | set[str],
    *,
    node_requirements: NodeRequirements | Mapping[str, Any] | None = None,
    node_requirements_wait_started_at: float | None = None,
) -> bool:
    exclude_worker_ids = (
        {str(exclude_worker_id)}
        if isinstance(exclude_worker_id, str)
        else {str(worker_id) for worker_id in exclude_worker_id}
    )
    candidates = [
        handle
        for worker_id, handle in _FTE_WORKER_HANDLES.items()
        if str(worker_id) not in exclude_worker_ids and handle is not None and handle._fte_healthy
    ]
    return bool(
        _filter_workers_for_node_requirements(
            candidates,
            node_requirements,
            node_requirements_wait_started_at=node_requirements_wait_started_at,
        )
    )


# FTE lock hierarchy:
# 1. Scheduler and registry locks protect queues/snapshots only and are released
#    before fragment state is inspected.
# 2. A thread may hold one fragment state lock at a time.
# 3. Global fragment traversal runs with no fragment state lock held.
def _assert_no_fte_fragment_state_lock_held(
    fragment_execution_items: list[tuple[tuple[str, str], FteFragmentExecution]],
) -> None:
    if not __debug__:
        return
    held_keys = [
        key
        for key, fragment_execution in fragment_execution_items
        if fragment_execution._state_lock_owned_by_current_thread()
    ]
    assert not held_keys, (
        f"FTE lock hierarchy violation: global fragment traversal while holding a fragment state lock: {held_keys}"
    )


def _ordered_fte_fragment_execution_items_for_pending_drain(
    fragment_execution_items: list[tuple[tuple[str, str], FteFragmentExecution]],
    *,
    execution_class: FteTaskExecutionClass | str | None = None,
) -> list[tuple[tuple[str, str], FteFragmentExecution]]:
    _assert_no_fte_fragment_state_lock_held(fragment_execution_items)
    if not fragment_execution_items:
        return []
    if execution_class is not None:
        fragment_execution_items = [
            item for item in fragment_execution_items if item[1].has_pending_partitions(execution_class)
        ]
        if not fragment_execution_items:
            return []
    grouped: dict[str, list[tuple[tuple[str, str], FteFragmentExecution]]] = {}
    for key, fragment_execution in sorted(fragment_execution_items, key=lambda item: item[0]):
        grouped.setdefault(str(key[0]), []).append((key, fragment_execution))
    query_ids = _FTE_SCHEDULERS.ordered_pending_drain_query_ids(list(grouped))
    if not query_ids:
        return []
    max_fragment_execution_count = max(len(fragment_executions) for fragment_executions in grouped.values())
    ordered: list[tuple[tuple[str, str], FteFragmentExecution]] = []
    for fragment_execution_index in range(max_fragment_execution_count):
        for query_id in query_ids:
            fragment_executions = grouped[query_id]
            if fragment_execution_index < len(fragment_executions):
                ordered.append(fragment_executions[fragment_execution_index])
    return ordered


def _has_fte_pending_standard_partitions(
    fragment_execution_items: list[tuple[tuple[str, str], FteFragmentExecution]],
) -> bool:
    _assert_no_fte_fragment_state_lock_held(fragment_execution_items)
    return any(
        fragment_execution.has_pending_partitions(FteTaskExecutionClass.STANDARD)
        for _, fragment_execution in fragment_execution_items
    )


def _has_fte_pending_standard_partitions_for_active_queries(
    fragment_execution_items: list[tuple[tuple[str, str], FteFragmentExecution]],
) -> bool:
    """Keep another query's admission scan failure local to that query."""
    _assert_no_fte_fragment_state_lock_held(fragment_execution_items)
    for key, fragment_execution in fragment_execution_items:
        scheduler = _FTE_SCHEDULERS.get(key[0])
        if scheduler is None or scheduler.stats().state != "RUNNING":
            continue
        try:
            if fragment_execution.has_pending_partitions(FteTaskExecutionClass.STANDARD):
                return True
        except Exception as exc:
            scheduler.fail(f"FTE admission scan failed: {exc}")
    return False


def _admit_fte_partition_execution_ready(
    query_id: str,
    fragment_execution: FteFragmentExecution | None,
    partition: Any,
) -> bool:
    if fragment_execution is not None:
        _sync_write_sink_stage_for_fragment(fragment_execution)
    if fragment_execution is None:
        return False
    return admit_fte_partition_submission(
        query_id,
        fragment_execution.fragment_id,
        int(partition.task_id.partition_id),
    )


def _admit_fte_partition_node_wait(
    query_id: str,
    partition: Any,
    fragment_execution: FteFragmentExecution | None = None,
) -> bool:
    if fragment_execution is not None:
        _sync_write_sink_stage_for_fragment(fragment_execution)
    with _FTE_REGISTRY_LOCK:
        fragment_execution_items = [
            item for item in _FTE_FRAGMENT_EXECUTIONS.items() if item[0][0] not in _FTE_CLOSING_QUERIES
        ]
    if partition.node_wait_started_at is None:
        execution_class = FteTaskExecutionClass.coerce(partition.execution_class)
        if (
            execution_class == FteTaskExecutionClass.SPECULATIVE
            and _has_fte_pending_standard_partitions_for_active_queries(fragment_execution_items)
        ):
            return False
    if fragment_execution is None:
        return False
    return admit_fte_partition_submission(
        query_id,
        fragment_execution.fragment_id,
        int(partition.task_id.partition_id),
    )


class FteWorkerReservation:
    def __init__(
        self,
        *,
        query_id: str,
        fragment_execution_id: int,
        fragment_id: str,
        partition_id: int,
        worker: RayWorkerActorHandle,
        stage_id: str,
        task_lease_id: str,
        attempt_id: str,
    ) -> None:
        self.query_id = str(query_id)
        self.fragment_execution_id = int(fragment_execution_id)
        self.fragment_id = str(fragment_id)
        self.partition_id = int(partition_id)
        self.worker = worker
        self.worker_id = str(worker.worker_id)
        self.stage_id = str(stage_id)
        self.task_lease_id = str(task_lease_id)
        self.attempt_id = str(attempt_id)


class FteWorkerReservationFuture:
    def __init__(
        self,
        *,
        query_id: str,
        fragment_execution_id: int,
        fragment_id: str,
        partition_id: int,
        reservation_generation: int,
        memory_requirement_bytes: Any = None,
        execution_class: FteTaskExecutionClass | str | None = None,
        node_requirements: NodeRequirements | Mapping[str, Any] | None = None,
        node_requirements_wait_started_at: float | None = None,
    ) -> None:
        self.query_id = str(query_id)
        self.fragment_execution_id = int(fragment_execution_id)
        self.fragment_id = str(fragment_id)
        self.partition_id = int(partition_id)
        self.key = (self.query_id, self.fragment_id, self.partition_id)
        self.reservation_generation = int(reservation_generation)
        self.memory_requirement_bytes = memory_requirement_bytes
        self.execution_class = FteTaskExecutionClass.coerce(execution_class)
        self.node_requirements = node_requirements
        self.node_requirements_wait_started_at = node_requirements_wait_started_at
        self.blocked_reason = ""
        self._lock = threading.Lock()
        self._done = False
        self._cancelled = False
        self._result: FteWorkerReservation | None = None
        self._exception: BaseException | None = None
        self._callbacks: list[Callable[[FteWorkerReservationFuture], None]] = []

    def done(self) -> bool:
        with self._lock:
            return self._done

    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    def result(self) -> FteWorkerReservation:
        with self._lock:
            if not self._done:
                raise RuntimeError("worker reservation future is not done")
            if self._cancelled:
                raise RuntimeError("worker reservation future was cancelled")
            if self._exception is not None:
                raise self._exception
            if self._result is None:
                raise RuntimeError("worker reservation future completed without a reservation")
            return self._result

    def exception(self) -> BaseException | None:
        with self._lock:
            return self._exception

    def set_blocked_reason(self, blocked_reason: str) -> None:
        with self._lock:
            self.blocked_reason = str(blocked_reason or "")

    def add_done_callback(self, callback: Callable[[FteWorkerReservationFuture], None]) -> None:
        run_now = False
        with self._lock:
            if self._done:
                run_now = True
            else:
                self._callbacks.append(callback)
        if run_now:
            self._run_callback(callback)

    def set_result(self, reservation: FteWorkerReservation) -> bool:
        callbacks = self._complete(result=reservation)
        self._run_callbacks(callbacks)
        return bool(callbacks)

    def set_exception(self, error: BaseException) -> bool:
        callbacks = self._complete(exception=error)
        self._run_callbacks(callbacks)
        return bool(callbacks)

    def cancel(self) -> bool:
        callbacks = self._complete(cancelled=True)
        self._run_callbacks(callbacks)
        return bool(callbacks)

    def _complete(
        self,
        *,
        result: FteWorkerReservation | None = None,
        exception: BaseException | None = None,
        cancelled: bool = False,
    ) -> list[Callable[[FteWorkerReservationFuture], None]]:
        with self._lock:
            if self._done:
                return []
            self._done = True
            self._cancelled = bool(cancelled)
            self._result = result
            self._exception = exception
            callbacks = list(self._callbacks)
            self._callbacks.clear()
        return callbacks

    def _run_callbacks(self, callbacks: list[Callable[[FteWorkerReservationFuture], None]]) -> None:
        for callback in callbacks:
            self._run_callback(callback)

    def _run_callback(self, callback: Callable[[FteWorkerReservationFuture], None]) -> None:
        try:
            callback(self)
        except Exception:
            pass


def _fte_partition_owner(query_id: str, fragment_id: str, partition_id: int) -> Any | None:
    with _FTE_REGISTRY_LOCK:
        return _FTE_PARTITION_OWNERS.get((str(query_id), str(fragment_id), int(partition_id)))


class FteWorkerPlacementManager:
    def __init__(self, coordinator: RayWorkerActorHandle) -> None:
        self.coordinator = coordinator

    def request_async(
        self,
        *,
        query_id: str,
        fragment_execution_id: int,
        fragment_id: str,
        partition_id: int,
        memory_requirement_bytes: Any = None,
        execution_class: FteTaskExecutionClass | str | None = None,
        node_requirements: NodeRequirements | Mapping[str, Any] | None = None,
        node_requirements_wait_started_at: float | None = None,
        on_done: Callable[[FteWorkerReservationFuture], None] | None = None,
    ) -> tuple[FteWorkerReservationFuture, bool]:
        query_id = str(query_id)
        fragment_id = str(fragment_id)
        partition_id = int(partition_id)
        key = (query_id, fragment_id, partition_id)
        with _FTE_REGISTRY_LOCK:
            if query_id in _FTE_CLOSING_QUERIES:
                raise RuntimeError(f"FTE query registry is closing: {query_id}")
            existing = _FTE_PENDING_WORKER_RESERVATIONS.get(key)
            if existing is not None:
                return existing, False
            generation = _FTE_WORKER_RESERVATION_GENERATIONS.get(key, 0) + 1
            _FTE_WORKER_RESERVATION_GENERATIONS[key] = generation
            future = FteWorkerReservationFuture(
                query_id=query_id,
                fragment_execution_id=fragment_execution_id,
                fragment_id=fragment_id,
                partition_id=partition_id,
                reservation_generation=generation,
                memory_requirement_bytes=memory_requirement_bytes,
                execution_class=execution_class,
                node_requirements=node_requirements,
                node_requirements_wait_started_at=node_requirements_wait_started_at,
            )
            _FTE_PENDING_WORKER_RESERVATIONS[key] = future
        if on_done is not None:
            future.add_done_callback(on_done)
        return future, True

    def acquire(
        self,
        *,
        query_id: str,
        fragment_id: str,
        partition_id: int,
        memory_requirement_bytes: Any = None,
        execution_class: FteTaskExecutionClass | str | None = None,
        node_requirements: NodeRequirements | Mapping[str, Any] | None = None,
        node_requirements_wait_started_at: float | None = None,
    ) -> FteWorkerReservation:
        query_id = str(query_id)
        fragment_id = str(fragment_id)
        partition_id = int(partition_id)
        owner_key = (query_id, fragment_id, partition_id)
        fragment_execution_id = _fte_partition_fragment_execution_id(query_id, fragment_id, partition_id)
        from duckdb.runners.ray.query_resource_runtime import get_query_resource_manager

        resource_query_id, stage_id = _fte_fragment_resource_identity(
            query_id,
            fragment_id,
        )
        eligible_node_ids = set(get_query_resource_manager(resource_query_id).task_eligible_node_ids(stage_id))
        if not eligible_node_ids:
            raise RuntimeError(f"FTE stage {stage_id} has no feasible node in its query allocation")
        with _FTE_REGISTRY_LOCK:
            if query_id in _FTE_CLOSING_QUERIES:
                raise RuntimeError(f"FTE query registry is closing: {query_id}")
            lease_record = _FTE_PARTITION_TASK_LEASES.get(owner_key)
            owner = _FTE_PARTITION_OWNERS.get(owner_key)
            if owner is not None and (not owner._fte_healthy or str(owner.node_id) not in eligible_node_ids):
                _FTE_PARTITION_OWNERS.pop(owner_key, None)
                _release_fte_partition_task_lease_for_key(
                    query_id,
                    fragment_id,
                    partition_id,
                )
                lease_record = None
                owner = None
            if owner is not None:
                owner.release_fte_partition_reservation(query_id, fragment_id, partition_id)
                if not _fte_worker_has_memory_capacity(
                    owner,
                    memory_requirement_bytes=memory_requirement_bytes,
                    execution_class=execution_class,
                ):
                    if _FTE_PARTITION_OWNERS.get(owner_key) is owner:
                        _FTE_PARTITION_OWNERS.pop(owner_key, None)
                    _release_fte_partition_task_lease_for_key(
                        query_id,
                        fragment_id,
                        partition_id,
                    )
                    lease_record = None
                    owner = None

            if owner is not None and lease_record is not None:
                task_lease = lease_record[1]
                if task_lease.node_id != owner.node_id:
                    _FTE_PARTITION_OWNERS.pop(owner_key, None)
                    _release_fte_partition_task_lease_for_key(
                        query_id,
                        fragment_id,
                        partition_id,
                    )
                    lease_record = None
                    owner = None

            excluded_worker_ids: set[str] = set()
            while owner is None:
                try:
                    owner = self.coordinator._select_fte_worker(
                        exclude=excluded_worker_ids,
                        allowed_node_ids=eligible_node_ids,
                        memory_requirement_bytes=memory_requirement_bytes,
                        execution_class=execution_class,
                        node_requirements=node_requirements,
                        node_requirements_wait_started_at=node_requirements_wait_started_at,
                    )
                    if owner is not None and str(owner.worker_id) in excluded_worker_ids:
                        owner = None
                except Exception:
                    _release_fte_partition_task_lease_for_key(
                        query_id,
                        fragment_id,
                        partition_id,
                    )
                    raise
                if owner is None:
                    _release_fte_partition_task_lease_for_key(
                        query_id,
                        fragment_id,
                        partition_id,
                    )
                    from duckdb.runners.ray.fragment_worker_selection import (
                        available_fte_workers,
                    )

                    available_workers = available_fte_workers(
                        self.coordinator,
                        getattr(self.coordinator, "worker_id", None),
                    )
                    live_worker_node_ids = {str(worker.node_id) for worker in available_workers}
                    if eligible_node_ids.isdisjoint(live_worker_node_ids):
                        raise RuntimeError(
                            f"FTE stage {stage_id} allocation has no live Ray worker: "
                            f"allocated_nodes={sorted(eligible_node_ids)} "
                            f"worker_nodes={sorted(live_worker_node_ids)}"
                        )
                    raise FteWorkerReservationUnavailable(
                        query_id=query_id,
                        fragment_id=fragment_id,
                        partition_id=partition_id,
                        memory_requirement_bytes=_memory_requirement_bytes(memory_requirement_bytes),
                    )
                try:
                    task_lease = _acquire_fte_partition_task_lease(
                        query_id=query_id,
                        fragment_execution_id=fragment_execution_id,
                        fragment_id=fragment_id,
                        partition_id=partition_id,
                        node_id=owner.node_id,
                    )
                except FteWorkerReservationUnavailable as exc:
                    if exc.blocked_reason != "node_capacity":
                        raise
                    excluded_worker_ids.add(str(owner.worker_id))
                    owner = None
                    continue
                lease_record = (fragment_execution_id, task_lease)
                _FTE_PARTITION_TASK_LEASES[owner_key] = lease_record

            if lease_record is None:
                task_lease = _acquire_fte_partition_task_lease(
                    query_id=query_id,
                    fragment_execution_id=fragment_execution_id,
                    fragment_id=fragment_id,
                    partition_id=partition_id,
                    node_id=owner.node_id,
                )
                _FTE_PARTITION_TASK_LEASES[owner_key] = (
                    fragment_execution_id,
                    task_lease,
                )
            else:
                task_lease = lease_record[1]
            _install_fte_partition_terminal_lease_release_hook(owner)
            try:
                owner.reserve_fte_partition(
                    query_id,
                    fragment_id,
                    partition_id,
                    memory_requirement_bytes=memory_requirement_bytes,
                    execution_class=execution_class,
                )
            except Exception:
                if _FTE_PARTITION_OWNERS.get(owner_key) is owner:
                    _FTE_PARTITION_OWNERS.pop(owner_key, None)
                _release_fte_partition_task_lease_for_key(query_id, fragment_id, partition_id)
                raise
            _FTE_PARTITION_OWNERS[owner_key] = owner
            _FTE_SCHEDULERS.record_pending_drain_progress(query_id)
        return FteWorkerReservation(
            query_id=query_id,
            fragment_execution_id=fragment_execution_id,
            fragment_id=fragment_id,
            partition_id=partition_id,
            worker=owner,
            stage_id=task_lease.stage_id,
            task_lease_id=task_lease.lease_id,
            attempt_id=task_lease.attempt_id,
        )

    def release(self, *, query_id: str, fragment_id: str, partition_id: int) -> None:
        self.release_owner(query_id=query_id, fragment_id=fragment_id, partition_id=partition_id)

    @staticmethod
    def release_owner(
        *,
        query_id: str,
        fragment_id: str,
        partition_id: int,
        terminal: bool | None = None,
    ) -> RayWorkerActorHandle | None:
        query_id = str(query_id)
        fragment_id = str(fragment_id)
        partition_id = int(partition_id)
        with _FTE_REGISTRY_LOCK:
            owner = _FTE_PARTITION_OWNERS.pop((query_id, fragment_id, partition_id), None)
            lease_record = _FTE_PARTITION_TASK_LEASES.pop((query_id, fragment_id, partition_id), None)
            if terminal is None and lease_record is not None:
                _, task_lease = lease_record
                fragment_execution = _FTE_FRAGMENT_EXECUTIONS.get((query_id, fragment_id))
                partition = None if fragment_execution is None else fragment_execution.partitions.get(partition_id)
                terminal = bool(
                    partition is not None
                    and any(
                        str(running.attempt_id) == task_lease.attempt_id
                        for running in partition.running_attempts.values()
                    )
                )
        release_fte_partition_task_waiter((query_id, fragment_id, partition_id))
        release_fte_partition_submission(
            query_id,
            fragment_id,
            partition_id,
        )
        try:
            if owner is not None:
                owner.release_fte_partition_reservation(query_id, fragment_id, partition_id)
        finally:
            if lease_record is not None:
                _, task_lease = lease_record
                _release_fte_partition_task_lease(
                    task_lease,
                    terminal=bool(terminal),
                )
        return owner

    @staticmethod
    def release_query(query_id: str) -> int:
        query_id = str(query_id or "").strip()
        if not query_id:
            return 0
        release_fte_query_submissions(query_id)
        release_fte_query_task_waiters(query_id)
        with _FTE_REGISTRY_LOCK:
            owner_items = [(key, owner) for key, owner in list(_FTE_PARTITION_OWNERS.items()) if key[0] == query_id]
            for key, _ in owner_items:
                _FTE_PARTITION_OWNERS.pop(key, None)
            lease_items = [
                (key, lease_record)
                for key, lease_record in list(_FTE_PARTITION_TASK_LEASES.items())
                if key[0] == query_id
            ]
            for key, _ in lease_items:
                _FTE_PARTITION_TASK_LEASES.pop(key, None)
        for (owner_query_id, fragment_id, partition_id), owner in owner_items:
            if owner is not None:
                owner.release_fte_partition_reservation(
                    owner_query_id,
                    fragment_id,
                    partition_id,
                )
        for (_lease_query_id, _fragment_id, _partition_id), lease_record in lease_items:
            _, task_lease = lease_record
            _release_fte_partition_task_lease(task_lease, terminal=False)
        return len(owner_items)

    def release_reservation(self, *, query_id: str, fragment_id: str, partition_id: int) -> None:
        owner = _fte_partition_owner(query_id, fragment_id, partition_id)
        if owner is not None:
            owner.release_fte_partition_reservation(
                str(query_id),
                str(fragment_id),
                int(partition_id),
            )

    def update_execution_class(
        self,
        *,
        query_id: str,
        fragment_id: str,
        partition_id: int,
        execution_class: FteTaskExecutionClass | str | None,
    ) -> bool:
        owner = _fte_partition_owner(query_id, fragment_id, partition_id)
        if owner is None:
            return False
        return bool(
            owner.set_fte_partition_reservation_execution_class(
                query_id,
                fragment_id,
                partition_id,
                execution_class,
            )
        )


def _mark_fte_worker_failed(
    worker_id: str,
    error: Any = None,
    *,
    query_id_filter: str | None = None,
    failed_worker_ids_override: set[str] | frozenset[str] | None = None,
) -> list[tuple[str, str, list[Any], list[Any]]]:
    worker_id = str(worker_id or "")
    if not worker_id:
        return []
    if query_id_filter is not None:
        query_id_filter = str(query_id_filter)
    failed_worker_ids = (
        {str(item) for item in failed_worker_ids_override}
        if failed_worker_ids_override is not None
        else _expanded_fte_failed_worker_ids(worker_id)
    )
    handles_to_kill: list[RayWorkerActorHandle] = []
    pending_worker_reservation_futures_to_cancel: list[FteWorkerReservationFuture] = []
    retryable_by_owner_key: dict[tuple[str, str, int], bool] = {}
    with _FTE_REGISTRY_LOCK:
        for failed_worker_id in sorted(failed_worker_ids):
            failed_handle = _FTE_WORKER_HANDLES.pop(failed_worker_id, None)
            if failed_handle is not None:
                failed_handle._fte_healthy = False
                if failed_worker_id != worker_id:
                    handles_to_kill.append(failed_handle)
        for owner_key, owner in list(_FTE_PARTITION_OWNERS.items()):
            if query_id_filter is not None and owner_key[0] != query_id_filter:
                continue
            if str(owner.worker_id) not in failed_worker_ids:
                continue
            memory_requirement_bytes = _fte_partition_memory_requirement_bytes(owner_key)
            execution_class = _fte_partition_execution_class(owner_key)
            node_requirements = _fte_partition_node_requirements(owner_key)
            wait_started_at = time.time()
            replacement = _select_replacement_fte_worker(
                failed_worker_ids,
                memory_requirement_bytes=memory_requirement_bytes,
                execution_class=execution_class,
                node_requirements=node_requirements,
                node_requirements_wait_started_at=wait_started_at,
            )
            if replacement is None:
                future = _FTE_PENDING_WORKER_RESERVATIONS.pop(owner_key, None)
                if future is not None:
                    pending_worker_reservation_futures_to_cancel.append(future)
                FteWorkerPlacementManager.release_owner(
                    query_id=owner_key[0],
                    fragment_id=owner_key[1],
                    partition_id=owner_key[2],
                )
                retryable_by_owner_key[owner_key] = _has_replacement_fte_worker(
                    failed_worker_ids,
                    node_requirements=node_requirements,
                    node_requirements_wait_started_at=wait_started_at,
                )
            else:
                retryable_by_owner_key[owner_key] = True
                future = _FTE_PENDING_WORKER_RESERVATIONS.pop(owner_key, None)
                if future is not None:
                    pending_worker_reservation_futures_to_cancel.append(future)
                FteWorkerPlacementManager.release_owner(
                    query_id=owner_key[0],
                    fragment_id=owner_key[1],
                    partition_id=owner_key[2],
                )
        fragment_execution_items = [
            item
            for item in _FTE_FRAGMENT_EXECUTIONS.items()
            if query_id_filter is None or item[0][0] == query_id_filter
        ]

    for handle in handles_to_kill:
        try:
            ray.kill(handle.actor_handle)
        except Exception:
            pass
    for future in pending_worker_reservation_futures_to_cancel:
        future.cancel()

    scheduled_by_stage: list[tuple[str, str, list[Any], list[Any]]] = []
    delayed_query_ids: set[str] = set()
    for (query_id, fragment_id), fragment_execution in fragment_execution_items:
        retryable_by_partition_id = {
            partition_id: retryable
            for (owner_query_id, owner_fragment_id, partition_id), retryable in retryable_by_owner_key.items()
            if owner_query_id == query_id and owner_fragment_id == fragment_id
        }
        fragment_execution_has_retryable_failed_worker = any(
            fragment_execution.has_retryable_running_attempt_on_worker(
                failed_worker_id,
                retryable_by_partition_id,
            )
            for failed_worker_id in failed_worker_ids
        )
        if fragment_execution_has_retryable_failed_worker and query_id not in delayed_query_ids:
            _start_or_prolong_fte_retry_delay(query_id)
            delayed_query_ids.add(query_id)
        for failed_worker_id in sorted(failed_worker_ids):
            scheduled = fragment_execution.mark_worker_failed(
                failed_worker_id,
                error or f"FTE worker lost: {failed_worker_id}",
                retryable=True,
                retryable_by_partition_id=retryable_by_partition_id,
                schedule_retries=False,
            )
            if scheduled:
                scheduled_by_stage.append((query_id, fragment_id, scheduled, []))
    return scheduled_by_stage


def _expanded_fte_failed_worker_ids(worker_id: str) -> set[str]:
    worker_id = str(worker_id or "")
    if not worker_id:
        return set()
    host_id, worker_index = _worker_id_host_and_index(worker_id)
    failed_worker_ids = {worker_id}
    chaos_host_indices = _chaos_host_loss_worker_indices()
    if chaos_host_indices and (not worker_index or worker_index in chaos_host_indices):
        with _FTE_REGISTRY_LOCK:
            for candidate_worker_id in _FTE_WORKER_HANDLES:
                candidate_host_id, candidate_index = _worker_id_host_and_index(candidate_worker_id)
                if candidate_host_id == host_id and candidate_index in chaos_host_indices:
                    failed_worker_ids.add(str(candidate_worker_id))
    return failed_worker_ids


def _collect_vane_env_overrides() -> dict[str, str]:
    return collect_vane_env_overrides()


def _fte_attempt_metrics(attempt: Any) -> dict[str, Any]:
    started_at = float(attempt.started_at)
    payload = {
        "attempt_id": str(attempt.attempt_id),
        "task_id": attempt.attempt_id.task_id.to_dict(),
        "attempt_number": int(attempt.attempt_id.attempt_id),
        "worker_id": attempt.worker_id,
        "started_at": started_at,
    }
    if started_at > 0:
        payload["age_s"] = max(0.0, time.time() - started_at)
    return payload


def _fte_partition_metrics(
    query_id: str,
    fragment_id: str,
    partition_id: int,
    partition: Any,
    *,
    partition_owners: Mapping[tuple[str, str, int], Any],
    pending_worker_reservations: Mapping[tuple[str, str, int], Any],
    partition_task_leases: Mapping[tuple[str, str, int], Any],
) -> dict[str, Any]:
    descriptor = partition.descriptor
    initial_splits = descriptor.initial_splits
    no_more_splits = descriptor.no_more_splits
    resource_key = (query_id, fragment_id, int(partition_id))
    owner = partition_owners.get(resource_key)
    pending_reservation = pending_worker_reservations.get(resource_key)
    task_lease_record = partition_task_leases.get(resource_key)
    task_lease = None if task_lease_record is None else task_lease_record[1]
    return {
        "task_id": str(partition.task_id),
        "task": partition.task_id.to_dict(),
        "partition_id": int(partition_id),
        "state": partition.state.value,
        "execution_class": partition.execution_class.value,
        "sealed": bool(partition.sealed),
        "ready_for_scheduling": bool(partition.ready_for_scheduling),
        "execution_ready_deferred": bool(partition.execution_ready_deferred),
        "waiting_for_node": (
            partition.node_wait_started_at is not None
            and not partition.running_attempts
            and not partition.finished
            and not partition.failed
        ),
        "waiting_for_execution": (
            bool(partition.ready_for_scheduling)
            and partition.node_wait_started_at is None
            and not partition.running_attempts
            and not partition.finished
            and not partition.failed
        ),
        "remaining_attempts": int(partition.remaining_attempts),
        "max_attempts": int(partition.max_attempts),
        "memory_requirement_bytes": partition.memory_requirement_bytes,
        "owner_worker_id": None if owner is None else owner.worker_id,
        "pending_worker_reservation": pending_reservation is not None,
        "pending_worker_reservation_done": bool(pending_reservation is not None and pending_reservation.done()),
        "pending_worker_reservation_generation": (
            None if pending_reservation is None else int(pending_reservation.reservation_generation)
        ),
        "pending_worker_reservation_blocked_reason": (
            "" if pending_reservation is None else str(pending_reservation.blocked_reason)
        ),
        "resource_stage_id": None if task_lease is None else task_lease.stage_id,
        "task_lease_id": None if task_lease is None else task_lease.lease_id,
        "task_lease_active": task_lease is not None,
        "running_attempts": [
            {
                **_fte_attempt_metrics(attempt),
                **(
                    {"task_stats": dict(partition.running_task_stats.get(attempt_id))}
                    if isinstance(partition.running_task_stats.get(attempt_id), dict)
                    else {}
                ),
            }
            for _, attempt in sorted(partition.running_attempts.items())
            for attempt_id in [attempt.attempt_id.attempt_id]
        ],
        "running_count": len(partition.running_attempts),
        "selected_attempt": partition.selected_attempt,
        "selected_output_stats": (
            dict(partition.selected_output_stats)
            if isinstance(partition.selected_output_stats, dict)
            else partition.selected_output_stats
        ),
        "finished_attempts": sorted(int(attempt_id) for attempt_id in partition.finished_attempts),
        "failure_observed": bool(partition.failure_observed),
        "failure_count": len(partition.failures),
        "initial_split_count_by_source": {
            str(source_id): len(splits) for source_id, splits in sorted(initial_splits.items())
        },
        "no_more_splits": sorted(str(source_id) for source_id in no_more_splits),
    }


def _fte_fragment_execution_metrics(
    fragment_id: str,
    fragment_execution: FteFragmentExecution,
    *,
    partition_owners: Mapping[tuple[str, str, int], Any],
    pending_worker_reservations: Mapping[tuple[str, str, int], Any],
    partition_task_leases: Mapping[tuple[str, str, int], Any],
    exchange_selectors: Mapping[str, Any],
) -> dict[str, Any]:
    with fragment_execution._state_lock:
        partitions = {
            str(partition_id): _fte_partition_metrics(
                fragment_execution.query_id,
                fragment_id,
                partition_id,
                partition,
                partition_owners=partition_owners,
                pending_worker_reservations=pending_worker_reservations,
                partition_task_leases=partition_task_leases,
            )
            for partition_id, partition in sorted(fragment_execution.partitions.items())
        }
        execution_class_counts: dict[str, int] = {}
        for partition in partitions.values():
            execution_class = str(partition["execution_class"])
            execution_class_counts[execution_class] = execution_class_counts.get(execution_class, 0) + 1
        partition_count = len(partitions)
        running_count = sum(int(partition["running_count"]) for partition in partitions.values())
        failed_count = sum(1 for partition in partitions.values() if partition["state"] == "FAILED")
        finished_count = sum(1 for partition in partitions.values() if partition["state"] == "FINISHED")
        waiting_for_node_count = sum(1 for partition in partitions.values() if partition["waiting_for_node"])
        waiting_for_execution_count = sum(1 for partition in partitions.values() if partition["waiting_for_execution"])
        deferred_count = sum(1 for partition in partitions.values() if partition["execution_ready_deferred"])
        pending_submission_count = fragment_execution.pending_submission_count()
        return {
            "query_id": fragment_execution.query_id,
            "fragment_id": fragment_id,
            "fragment_execution_id": int(fragment_execution.fragment_execution_id),
            "fragment_execution_class": fragment_execution.execution_class.value,
            "partition_count": partition_count,
            "running_count": running_count,
            "failed_count": failed_count,
            "finished_count": finished_count,
            "waiting_for_node_count": waiting_for_node_count,
            "waiting_for_execution_count": waiting_for_execution_count,
            "execution_deferred_count": deferred_count,
            "pending_submission_count": pending_submission_count,
            "execution_class_counts": execution_class_counts,
            "failed": bool(fragment_execution.failed or failed_count),
            "finished": bool(partition_count) and finished_count == partition_count,
            "no_more_partitions": bool(fragment_execution.no_more_partitions),
            "source_node_ids": sorted(fragment_execution.source_node_ids),
            "dynamic_scan_source_node_ids": sorted(fragment_execution.dynamic_scan_source_node_ids),
            "dynamic_exchange_source_node_ids": sorted(fragment_execution.dynamic_exchange_source_node_ids),
            "exchange_selectors": exchange_selectors,
            "partitions": partitions,
        }


def _fte_progress_partition_metrics(
    partition: Any,
) -> dict[str, Any]:
    """Copy only the immutable data consumed by the progress renderer.

    The caller holds the owning fragment's state lock.  This intentionally
    avoids registry-owned placement and lease maps: progress is an observation
    path and must not acquire or extend the lifetime of scheduler-global state.
    """
    return {
        "state": partition.state.value,
        "running_attempts": [
            {"task_stats": dict(task_stats)}
            for _, attempt in sorted(partition.running_attempts.items())
            for task_stats in [partition.running_task_stats.get(attempt.attempt_id.attempt_id)]
            if isinstance(task_stats, Mapping)
        ],
        "selected_output_stats": (
            dict(partition.selected_output_stats)
            if isinstance(partition.selected_output_stats, Mapping)
            else partition.selected_output_stats
        ),
    }


def ensure_fte_fragment_progress_topology(
    query_id: str,
    fragment_id: str,
    build_topology: Callable[[], dict[str, Any]],
) -> bool:
    """Build and publish one coordinator-owned immutable Fragment topology."""
    query_key = str(query_id or "").strip()
    fragment_key = str(fragment_id or "").strip()
    if not query_key or not fragment_key:
        raise ValueError("fragment progress topology requires query_id and fragment_id")
    if not callable(build_topology):
        raise TypeError("fragment progress topology builder must be callable")

    owner_key = (query_key, fragment_key)
    with _FTE_REGISTRY_CONDITION:
        while owner_key in _FTE_FRAGMENT_PROGRESS_TOPOLOGY_BUILDS:
            if query_key in _FTE_CLOSING_QUERIES:
                raise RuntimeError(f"FTE query registry is closing: {query_key}")
            _FTE_REGISTRY_CONDITION.wait()
        if query_key in _FTE_CLOSING_QUERIES:
            raise RuntimeError(f"FTE query registry is closing: {query_key}")
        if owner_key in _FTE_FRAGMENT_PROGRESS_TOPOLOGIES:
            return False
        _FTE_FRAGMENT_PROGRESS_TOPOLOGY_BUILDS.add(owner_key)

    try:
        topology = validate_pipeline_topology(build_topology())
    except BaseException:
        with _FTE_REGISTRY_CONDITION:
            _FTE_FRAGMENT_PROGRESS_TOPOLOGY_BUILDS.discard(owner_key)
            _FTE_REGISTRY_CONDITION.notify_all()
        raise

    with _FTE_REGISTRY_CONDITION:
        _FTE_FRAGMENT_PROGRESS_TOPOLOGY_BUILDS.discard(owner_key)
        if query_key in _FTE_CLOSING_QUERIES:
            _FTE_REGISTRY_CONDITION.notify_all()
            raise RuntimeError(f"FTE query registry closed during topology build: {query_key}")
        existing = _FTE_FRAGMENT_PROGRESS_TOPOLOGIES.get(owner_key)
        if existing is not None:
            if existing != topology:
                _FTE_REGISTRY_CONDITION.notify_all()
                raise RuntimeError(f"native fragment progress topology changed after publication: {fragment_key}")
            _FTE_REGISTRY_CONDITION.notify_all()
            return False
        _FTE_FRAGMENT_PROGRESS_TOPOLOGIES[owner_key] = topology
        _FTE_REGISTRY_CONDITION.notify_all()
        return True


def wait_for_fte_query_progress_topology(
    query_id: str,
    *,
    timeout_s: float,
) -> None:
    """Wait for the native COPY path to publish its first Fragment topology."""
    query_key = str(query_id or "").strip()
    if not query_key:
        raise ValueError("query_id must be non-empty")
    timeout_s = float(timeout_s)
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("topology wait timeout must be finite and positive")
    deadline = time.monotonic() + timeout_s
    with _FTE_REGISTRY_CONDITION:
        while not any(fragment_query_id == query_key for fragment_query_id, _ in _FTE_FRAGMENT_PROGRESS_TOPOLOGIES):
            if query_key in _FTE_CLOSING_QUERIES:
                raise RuntimeError(f"FTE query closed before topology initialization: {query_key}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"FTE query topology initialization timed out after {timeout_s:.3f}s: {query_key}")
            _FTE_REGISTRY_CONDITION.wait(remaining)


def _fte_fragment_progress_metrics(
    fragment_id: str,
    fragment_execution: FteFragmentExecution,
) -> dict[str, Any]:
    """Materialize one fragment view without touching registry-global state."""
    with fragment_execution._state_lock:
        pending_submission_count = fragment_execution.pending_submission_count()
        partitions = {
            str(partition_id): _fte_progress_partition_metrics(partition)
            for partition_id, partition in sorted(fragment_execution.partitions.items())
        }
        partition_count = len(partitions)
        failed_count = sum(1 for partition in partitions.values() if partition["state"] == "FAILED")
        finished_count = sum(1 for partition in partitions.values() if partition["state"] == "FINISHED")
        return {
            "fragment_id": fragment_id,
            "fragment_execution_id": int(fragment_execution.fragment_execution_id),
            "failed": bool(fragment_execution.failed or failed_count),
            "finished": bool(fragment_execution.no_more_partitions)
            and bool(partition_count)
            and finished_count == partition_count,
            "no_more_partitions": bool(fragment_execution.no_more_partitions),
            "pending_submission_count": pending_submission_count,
            "partitions": partitions,
        }


def fte_progress_registry_snapshot(query_id: str) -> dict[str, Any]:
    """Return a query-scoped, non-mutating progress snapshot.

    The global registry lock protects only the shallow reference capture.  All
    potentially proportional work happens after that lock is released, under
    independent per-fragment locks.  A retained fragment reference remains
    valid even if query teardown removes it from the live registry meanwhile.
    """
    query_id = str(query_id or "").strip()
    if not query_id:
        raise ValueError("query_id must be non-empty")
    with _FTE_REGISTRY_LOCK:
        fragment_execution_items = tuple(
            (fragment_id, fragment_execution)
            for (fragment_query_id, fragment_id), fragment_execution in sorted(_FTE_FRAGMENT_EXECUTIONS.items())
            if fragment_query_id == query_id
        )
        topologies_by_fragment = {
            fragment_id: copy.deepcopy(topology)
            for (fragment_query_id, fragment_id), topology in _FTE_FRAGMENT_PROGRESS_TOPOLOGIES.items()
            if fragment_query_id == query_id
        }

    fragment_executions = {
        fragment_id: _fte_fragment_progress_metrics(
            fragment_id,
            fragment_execution,
        )
        for fragment_id, fragment_execution in fragment_execution_items
    }
    missing_topologies = sorted(set(fragment_executions) - set(topologies_by_fragment))
    if missing_topologies:
        raise RuntimeError(
            "FTE fragment execution exists without coordinator topology: " + ", ".join(missing_topologies)
        )
    for fragment_id, topology in topologies_by_fragment.items():
        fragment_execution = fragment_executions.get(fragment_id)
        if fragment_execution is None:
            fragment_execution = {
                "fragment_id": fragment_id,
                "fragment_execution_id": 0,
                "failed": False,
                "finished": False,
                "no_more_partitions": False,
                "pending_submission_count": 0,
                "partitions": {},
            }
            fragment_executions[fragment_id] = fragment_execution
        fragment_execution["progress_topology"] = topology
    failed = any(fragment_execution["failed"] for fragment_execution in fragment_executions.values())
    scheduler = _FTE_SCHEDULERS.get(query_id)
    failed = failed or (scheduler is not None and scheduler.stats().state == "FAILED")
    query = {
        "query_id": query_id,
        "failed": failed,
        "finished": bool(fragment_executions)
        and all(fragment_execution["finished"] for fragment_execution in fragment_executions.values()),
        "fragment_executions": fragment_executions,
    }
    return {
        "fragment_execution_count": len(fragment_executions),
        "queries": {query_id: query},
    }


def _fte_query_metrics(
    event_scheduler_stats: Mapping[str, Any],
    *,
    fragment_execution_items: tuple[tuple[tuple[str, str], Any], ...],
    partition_owners: Mapping[tuple[str, str, int], Any],
    pending_worker_reservations: Mapping[tuple[str, str, int], Any],
    partition_task_leases: Mapping[tuple[str, str, int], Any],
    exchange_selectors_by_fragment: Mapping[tuple[str, str], Mapping[str, Any]],
    result_handle_counts: Mapping[str, int],
    retry_delays: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    query_ids = {query_id for (query_id, _), _ in fragment_execution_items}
    query_ids.update(query_id for query_id, _, _ in pending_worker_reservations)
    query_ids.update(result_handle_counts)
    query_ids.update(event_scheduler_stats)
    queries: dict[str, dict[str, Any]] = {}
    for query_id in sorted(query_ids):
        query_fragment_execution_items = [
            (fragment_id, fragment_execution)
            for (fragment_execution_query_id, fragment_id), fragment_execution in fragment_execution_items
            if fragment_execution_query_id == query_id
        ]
        fragment_executions = {
            fragment_id: _fte_fragment_execution_metrics(
                fragment_id,
                fragment_execution,
                partition_owners=partition_owners,
                pending_worker_reservations=pending_worker_reservations,
                partition_task_leases=partition_task_leases,
                exchange_selectors=exchange_selectors_by_fragment.get(
                    (query_id, fragment_id),
                    {},
                ),
            )
            for fragment_id, fragment_execution in query_fragment_execution_items
        }
        pending_reservation_count = sum(
            1 for reservation_query_id, _, _ in pending_worker_reservations if reservation_query_id == query_id
        )
        pending_reservation_done_count = sum(
            1
            for (reservation_query_id, _, _), future in pending_worker_reservations.items()
            if reservation_query_id == query_id and future.done()
        )
        partition_count = sum(
            fragment_execution["partition_count"] for fragment_execution in fragment_executions.values()
        )
        running_count = sum(fragment_execution["running_count"] for fragment_execution in fragment_executions.values())
        failed_count = sum(fragment_execution["failed_count"] for fragment_execution in fragment_executions.values())
        finished_count = sum(
            fragment_execution["finished_count"] for fragment_execution in fragment_executions.values()
        )
        waiting_for_node_count = sum(
            fragment_execution["waiting_for_node_count"] for fragment_execution in fragment_executions.values()
        )
        waiting_for_execution_count = sum(
            fragment_execution["waiting_for_execution_count"] for fragment_execution in fragment_executions.values()
        )
        pending_submission_count = sum(
            fragment_execution["pending_submission_count"] for fragment_execution in fragment_executions.values()
        )
        payload = {
            "query_id": query_id,
            "fragment_execution_count": len(fragment_executions),
            "partition_count": partition_count,
            "running_count": running_count,
            "failed_count": failed_count,
            "finished_count": finished_count,
            "waiting_for_node_count": waiting_for_node_count,
            "waiting_for_execution_count": waiting_for_execution_count,
            "pending_submission_count": pending_submission_count,
            "pending_worker_reservation_count": pending_reservation_count,
            "pending_worker_reservation_done_count": pending_reservation_done_count,
            "result_handle_count": int(result_handle_counts.get(query_id, 0)),
            "retry_delay_s": float(retry_delays.get(query_id, 0.0)),
            "event_scheduler": dict(event_scheduler_stats.get(query_id, {})),
            "failed": failed_count > 0
            or any(fragment_execution["failed"] for fragment_execution in fragment_executions.values())
            or event_scheduler_stats.get(query_id, {}).get("state") == "FAILED",
            "finished": bool(fragment_executions)
            and all(fragment_execution["finished"] for fragment_execution in fragment_executions.values()),
            "fragment_executions": fragment_executions,
        }
        from duckdb.runners.ray.query_resource_runtime import query_resource_manager_snapshot

        manager_snapshot = query_resource_manager_snapshot(query_id)
        if manager_snapshot:
            payload["query_resource_manager"] = manager_snapshot
        queries[query_id] = payload
    return queries


def fte_registry_stats() -> dict[str, Any]:
    with _FTE_REGISTRY_LOCK:
        fragment_execution_items = tuple(sorted(_FTE_FRAGMENT_EXECUTIONS.items()))
        partition_owners = dict(_FTE_PARTITION_OWNERS)
        pending_worker_reservations = dict(_FTE_PENDING_WORKER_RESERVATIONS)
        partition_task_leases = dict(_FTE_PARTITION_TASK_LEASES)
        worker_handles = tuple(sorted(_FTE_WORKER_HANDLES.items()))
        result_handle_counts = {query_id: len(handles) for query_id, handles in _FTE_RESULT_HANDLES_BY_QUERY.items()}
        retry_delays = {query_id: delayer.remaining_delay_s() for query_id, delayer in _FTE_RETRY_DELAYS.items()}
        closing_queries = sorted(_FTE_CLOSING_QUERIES)
        active_operations_by_query = dict(_FTE_ACTIVE_OPERATIONS_BY_QUERY)
        active_teardown_operations_by_query = dict(_FTE_ACTIVE_TEARDOWN_OPERATIONS_BY_QUERY)
        exchange_selectors_by_fragment = {
            key: {
                str(source_node_id): selector.to_metrics()
                for source_node_id, selector in sorted(fragment_state.exchange_source_selectors_by_source.items())
            }
            for key, fragment_state in _FTE_FRAGMENT_STATES.items()
        }
        registry_counts = {
            "fragment_execution_count": len(_FTE_FRAGMENT_EXECUTIONS),
            "partition_owner_count": len(_FTE_PARTITION_OWNERS),
            "fragment_state_count": len(_FTE_FRAGMENT_STATES),
            "worker_count": len(_FTE_WORKER_HANDLES),
            "pending_worker_reservation_count": len(_FTE_PENDING_WORKER_RESERVATIONS),
            "partition_task_lease_count": len(_FTE_PARTITION_TASK_LEASES),
            "partition_task_waiter_count": len(_FTE_PARTITION_TASK_WAITERS),
            "stage_submission_probe_count": len(_FTE_STAGE_SUBMISSION_PROBES),
            "stage_submission_block_count": len(_FTE_STAGE_SUBMISSION_BLOCKS),
            "pending_worker_reservation_done_count": sum(
                1 for future in _FTE_PENDING_WORKER_RESERVATIONS.values() if future.done()
            ),
            "status_watcher_count": len(_FTE_STATUS_WATCHERS),
            "result_handle_query_count": len(_FTE_RESULT_HANDLES_BY_QUERY),
            "closing_query_count": len(closing_queries),
            "active_registry_operation_count": sum(active_operations_by_query.values()),
            "active_registry_teardown_operation_count": sum(active_teardown_operations_by_query.values()),
        }

    event_scheduler_stats = _FTE_SCHEDULERS.stats()
    worker_stats = {
        str(worker_id): handle.fte_pressure_stats() for worker_id, handle in worker_handles if handle is not None
    }
    return {
        **registry_counts,
        "submission_window": fte_submission_window_snapshot(),
        "event_scheduler_count": len(event_scheduler_stats),
        "event_schedulers": event_scheduler_stats,
        "workers": worker_stats,
        "closing_queries": closing_queries,
        "active_registry_operations_by_query": active_operations_by_query,
        "active_registry_teardown_operations_by_query": (active_teardown_operations_by_query),
        "queries": _fte_query_metrics(
            event_scheduler_stats,
            fragment_execution_items=fragment_execution_items,
            partition_owners=partition_owners,
            pending_worker_reservations=pending_worker_reservations,
            partition_task_leases=partition_task_leases,
            exchange_selectors_by_fragment=exchange_selectors_by_fragment,
            result_handle_counts=result_handle_counts,
            retry_delays=retry_delays,
        ),
    }


def refresh_fte_running_task_stats(query_id: str | None = None) -> None:
    query_filter = None if query_id is None else str(query_id)
    running_items: list[tuple[FteFragmentExecution, Any]] = []
    timed_out_workers: set[str] = set()
    status_timeout_s = 0.05
    with _FTE_REGISTRY_LOCK:
        fragment_executions = list(_FTE_FRAGMENT_EXECUTIONS.items())
    for (fragment_execution_query_id, _fragment_id), fragment_execution in fragment_executions:
        if query_filter is not None and fragment_execution_query_id != query_filter:
            continue
        with fragment_execution._state_lock:
            for partition in fragment_execution.partitions.values():
                for running in partition.running_attempts.values():
                    if running.remote_handle is not None:
                        running_items.append((fragment_execution, running))
    for fragment_execution, running in running_items:
        if fte_registry_query_is_closing(fragment_execution.query_id):
            continue
        worker_key = str(getattr(running.remote_handle, "worker_id", "") or id(running.remote_handle))
        if worker_key in timed_out_workers:
            continue
        try:
            status = running.remote_handle.fte_get_task_status(
                running.attempt_id.to_dict(),
                timeout_s=status_timeout_s,
            )
        except TimeoutError:
            timed_out_workers.add(worker_key)
            continue
        except Exception:
            continue
        if isinstance(status, Mapping) and str(status.get("state", "")).upper() != "UNKNOWN":
            try:
                validate_fte_status_identity(status, running.attempt_id)
                if fte_registry_query_is_closing(fragment_execution.query_id):
                    continue
                running.remote_handle.handle_fte_task_status(status)
            except Exception:
                pass


__all__ = [name for name in globals() if not name.startswith("__")]
