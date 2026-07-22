# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import pickle
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vane.runners.fte.fte_config import (
    _FALSE_VALUES,
    _TRUE_VALUES,
    FTE_WORKER_RUNTIME,
)
from vane.runners.fte.fte_split_assigner import _normalize_sources
from vane.runners.fte.fte_types import FteSplit, FteTaskAttemptId, FteTaskId

if TYPE_CHECKING:
    from collections.abc import Sequence


def fte_descriptor_max_in_memory() -> int:
    raw = os.getenv("VANE_FTE_DESCRIPTOR_MAX_IN_MEMORY", "0")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, value)


def fte_descriptor_spill_dir() -> Path | None:
    raw = os.getenv("VANE_FTE_DESCRIPTOR_SPILL_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def _source_id_from_assignment(assignment: Mapping[str, Any]) -> str:
    for key in ("source_node_id", "plan_node_id", "node_id", "source"):
        value = assignment.get(key)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError("split assignment is missing source_node_id")


_OUTPUT_BUFFER_TYPE_KEYS = ("type", "@type", "kind", "buffer_type")
_OUTPUT_BUFFER_NO_MORE_KEYS = (
    "no_more_buffer_ids",
    "noMoreBufferIds",
    "no_more_buffers",
    "noMoreBuffers",
    "sealed",
)
_OUTPUT_BUFFER_PARTITION_COUNT_KEYS = (
    "output_partition_count",
    "outputPartitionCount",
    "partition_count",
    "partitionCount",
)
_OUTPUT_BUFFER_EXCHANGE_SINK_KEYS = (
    "exchange_sink_instance",
    "exchangeSinkInstanceHandle",
    "exchange_sink_instance_handle",
    "sink_instance",
)
_OUTPUT_BUFFER_ALIAS_KEYS = (
    set(_OUTPUT_BUFFER_TYPE_KEYS)
    | set(_OUTPUT_BUFFER_NO_MORE_KEYS)
    | set(_OUTPUT_BUFFER_PARTITION_COUNT_KEYS)
    | set(_OUTPUT_BUFFER_EXCHANGE_SINK_KEYS)
    | {"version", "buffers", "buffer_ids", "bufferIds"}
)


def _first_present(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload.get(key)
    return None


def _freeze_output_buffer_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_output_buffer_value(item)) for key, item in value.items()))
    if isinstance(value, set):
        return tuple(sorted((_freeze_output_buffer_value(item) for item in value), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_output_buffer_value(item) for item in value)
    return value


def _output_buffer_bool(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _FALSE_VALUES:
            return False
        if text in _TRUE_VALUES:
            return True
    return bool(value)


def _normalize_output_buffers(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("output_buffers must be a mapping")

    normalized: dict[str, Any] = {}
    version = int(payload.get("version", 0) or 0)
    if version < 0:
        raise ValueError("output_buffers version must be non-negative")
    normalized["version"] = version

    buffer_type = _first_present(payload, _OUTPUT_BUFFER_TYPE_KEYS)
    if buffer_type is not None and str(buffer_type).strip():
        normalized["type"] = str(buffer_type).strip().lower()

    buffers = _first_present(payload, ("buffers", "buffer_ids", "bufferIds"))
    if buffers is not None:
        if isinstance(buffers, Mapping):
            normalized["buffers"] = {str(buffer_id): value for buffer_id, value in buffers.items()}
        elif isinstance(buffers, set):
            normalized["buffers"] = [str(buffer_id) for buffer_id in sorted(buffers, key=str)]
        elif isinstance(buffers, (list, tuple)):
            normalized["buffers"] = [str(buffer_id) for buffer_id in buffers]
        else:
            raise ValueError("output_buffers buffers must be a mapping or sequence")

    no_more = _first_present(payload, _OUTPUT_BUFFER_NO_MORE_KEYS)
    if no_more is not None and _output_buffer_bool(no_more):
        normalized["no_more_buffer_ids"] = True

    partition_count = _first_present(payload, _OUTPUT_BUFFER_PARTITION_COUNT_KEYS)
    if partition_count is not None:
        partition_count = int(partition_count)
        if partition_count < 0:
            raise ValueError("output_buffers partition count must be non-negative")
        normalized["output_partition_count"] = partition_count

    exchange_sink_instance = _first_present(payload, _OUTPUT_BUFFER_EXCHANGE_SINK_KEYS)
    if exchange_sink_instance is not None:
        normalized["exchange_sink_instance"] = exchange_sink_instance

    for key, value in payload.items():
        if key not in _OUTPUT_BUFFER_ALIAS_KEYS:
            normalized[str(key)] = value
    return normalized


def _output_buffer_version(payload: Mapping[str, Any]) -> int:
    return int(payload.get("version", 0) or 0)


def _output_buffer_type(payload: Mapping[str, Any]) -> str | None:
    value = payload.get("type")
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _output_buffers_sealed(payload: Mapping[str, Any]) -> bool:
    return bool(payload.get("no_more_buffer_ids"))


def _output_buffer_partition_count(payload: Mapping[str, Any]) -> int | None:
    if "output_partition_count" not in payload:
        return None
    return int(payload["output_partition_count"])


def _output_buffer_assignments(payload: Mapping[str, Any]) -> dict[str, Any]:
    buffers = payload.get("buffers")
    if buffers is None:
        return {}
    if isinstance(buffers, Mapping):
        return {str(buffer_id): _freeze_output_buffer_value(value) for buffer_id, value in buffers.items()}
    if isinstance(buffers, (list, tuple, set)):
        return {str(buffer_id): None for buffer_id in buffers}
    raise ValueError("output_buffers buffers must be a mapping or sequence")


def _validate_output_buffer_transition(
    current: Mapping[str, Any],
    update: Mapping[str, Any],
) -> None:
    current_type = _output_buffer_type(current)
    update_type = _output_buffer_type(update)
    if current_type is not None and update_type is not None and current_type != update_type:
        raise ValueError(f"output_buffers type cannot change from {current_type!r} to {update_type!r}")

    if _output_buffers_sealed(current) and dict(current) != dict(update):
        raise ValueError("output_buffers are sealed and cannot be changed")

    current_partition_count = _output_buffer_partition_count(current)
    update_partition_count = _output_buffer_partition_count(update)
    if (
        current_partition_count is not None
        and update_partition_count is not None
        and current_partition_count != update_partition_count
    ):
        raise ValueError(
            "output_buffers output_partition_count cannot change "
            f"from {current_partition_count} to {update_partition_count}"
        )

    current_buffers = _output_buffer_assignments(current)
    update_buffers = _output_buffer_assignments(update)
    missing_buffers = sorted(set(current_buffers) - set(update_buffers))
    if missing_buffers:
        raise ValueError(f"output_buffers cannot remove buffers: {missing_buffers}")
    for buffer_id in sorted(set(current_buffers) & set(update_buffers)):
        if current_buffers[buffer_id] != update_buffers[buffer_id]:
            raise ValueError(f"output_buffers assignment cannot change for buffer {buffer_id}")


def _merge_output_buffers(
    current: Mapping[str, Any] | None,
    update: Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, bool]:
    update_normalized = _normalize_output_buffers(update)
    if update_normalized is None:
        return (_normalize_output_buffers(current), False)
    if current is None:
        return update_normalized, True

    current_normalized = _normalize_output_buffers(current)
    if current_normalized is None:
        return update_normalized, True

    current_version = _output_buffer_version(current_normalized)
    update_version = _output_buffer_version(update_normalized)
    if update_version < current_version:
        return current_normalized, False
    if update_version == current_version:
        if update_normalized == current_normalized:
            return current_normalized, False
        raise ValueError(f"output_buffers version {update_version} contains conflicting content")

    _validate_output_buffer_transition(current_normalized, update_normalized)
    return update_normalized, True


def _output_buffer_status(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    normalized = _normalize_output_buffers(payload)
    if normalized is None:
        return None
    status: dict[str, Any] = {
        "version": _output_buffer_version(normalized),
        "buffer_count": len(_output_buffer_assignments(normalized)),
        "sealed": _output_buffers_sealed(normalized),
    }
    buffer_type = _output_buffer_type(normalized)
    if buffer_type is not None:
        status["type"] = buffer_type
    partition_count = _output_buffer_partition_count(normalized)
    if partition_count is not None:
        status["output_partition_count"] = partition_count
    if "exchange_sink_instance" in normalized:
        status["has_exchange_sink_instance"] = True
    return status


def normalize_initial_splits(
    initial_splits: Mapping[str, list[Mapping[str, Any]]] | None,
) -> dict[str, list[FteSplit]]:
    normalized: dict[str, list[FteSplit]] = {}
    if not initial_splits:
        return normalized
    for source_node_id, splits in initial_splits.items():
        normalized[str(source_node_id)] = [
            split if isinstance(split, FteSplit) else FteSplit.from_dict(str(source_node_id), split) for split in splits
        ]
    return normalized


@dataclass
class FteTaskUpdateRequest:
    """FTE task update request used by control RPCs."""

    initial_splits: Mapping[str, list[FteSplit | Mapping[str, Any]]] | None = None
    no_more_splits: set[str] | list[str] | tuple[str, ...] | None = None
    output_buffers: Mapping[str, Any] | None = None
    dynamic_filter_domains: Mapping[str, Any] | None = None
    context: Mapping[str, Any] | None = None
    resource_request: Mapping[str, Any] | None = None
    fragment_plan: Any = None
    fragment_plan_present: bool = False
    source_node_ids: set[str] | list[str] | tuple[str, ...] | None = None
    dynamic_scan_source_node_ids: set[str] | list[str] | tuple[str, ...] | None = None
    dynamic_exchange_source_node_ids: set[str] | list[str] | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self.initial_splits = normalize_initial_splits(self.initial_splits)
        self.no_more_splits = {str(source) for source in (self.no_more_splits or [])}
        self.output_buffers = _normalize_output_buffers(self.output_buffers)
        self.dynamic_filter_domains = dict(self.dynamic_filter_domains or {})
        self.context = dict(self.context or {})
        self.resource_request = dict(self.resource_request or {})
        self.source_node_ids = _normalize_sources(self.source_node_ids)
        self.dynamic_scan_source_node_ids = _normalize_sources(self.dynamic_scan_source_node_ids)
        self.dynamic_exchange_source_node_ids = _normalize_sources(self.dynamic_exchange_source_node_ids)
        if self.fragment_plan is not None:
            self.fragment_plan_present = True

    @classmethod
    def coerce(cls, update: FteTaskUpdateRequest | Mapping[str, Any] | None) -> FteTaskUpdateRequest:
        if isinstance(update, FteTaskUpdateRequest):
            return update
        if update is None:
            return cls()
        return cls.from_dict(update)

    @classmethod
    def from_dict(cls, update: Mapping[str, Any]) -> FteTaskUpdateRequest:
        payload = dict(update or {})
        initial_splits = dict(payload.get("initial_splits") or {})
        no_more_splits = {str(source) for source in (payload.get("no_more_splits") or [])}
        for assignment in payload.get("split_assignments") or []:
            source_id = _source_id_from_assignment(assignment)
            assignment_splits = assignment.get("splits") or []
            if assignment_splits:
                initial_splits.setdefault(source_id, []).extend(assignment_splits)
            if assignment.get("no_more_splits") or assignment.get("noMoreSplits"):
                no_more_splits.add(source_id)

        output_buffers = None
        for key in ("output_buffers", "outputIds", "output_buffer_update"):
            if key in payload:
                output_buffers = _normalize_output_buffers(payload.get(key) or {})
                break

        dynamic_filter_domains = {}
        for key in ("dynamic_filter_domains", "dynamicFilterDomains", "dynamic_filters"):
            if key in payload:
                dynamic_filter_domains = dict(payload.get(key) or {})
                break

        return cls(
            initial_splits=initial_splits,
            no_more_splits=no_more_splits,
            output_buffers=output_buffers,
            dynamic_filter_domains=dynamic_filter_domains,
            context=payload.get("context"),
            resource_request=payload.get("resource_request"),
            fragment_plan=payload.get("fragment_plan"),
            fragment_plan_present="fragment_plan" in payload,
            source_node_ids=payload.get("source_node_ids"),
            dynamic_scan_source_node_ids=payload.get("dynamic_scan_source_node_ids"),
            dynamic_exchange_source_node_ids=payload.get("dynamic_exchange_source_node_ids"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.initial_splits:
            payload["initial_splits"] = {
                source_id: [split.to_dict() for split in splits] for source_id, splits in self.initial_splits.items()
            }
        if self.no_more_splits:
            payload["no_more_splits"] = sorted(self.no_more_splits)
        if self.output_buffers is not None:
            payload["output_buffers"] = dict(self.output_buffers)
        if self.dynamic_filter_domains:
            payload["dynamic_filter_domains"] = dict(self.dynamic_filter_domains)
        if self.context:
            payload["context"] = dict(self.context)
        if self.resource_request:
            payload["resource_request"] = dict(self.resource_request)
        if self.fragment_plan_present:
            payload["fragment_plan"] = self.fragment_plan
        if self.source_node_ids:
            payload["source_node_ids"] = sorted(self.source_node_ids)
        if self.dynamic_scan_source_node_ids:
            payload["dynamic_scan_source_node_ids"] = sorted(self.dynamic_scan_source_node_ids)
        if self.dynamic_exchange_source_node_ids:
            payload["dynamic_exchange_source_node_ids"] = sorted(self.dynamic_exchange_source_node_ids)
        return payload


@dataclass
class TaskDescriptor:
    task_id: FteTaskId
    fragment_id: str
    context: dict[str, Any] = field(default_factory=dict)
    initial_splits: dict[str, list[FteSplit]] = field(default_factory=dict)
    no_more_splits: set[str] = field(default_factory=set)
    resource_request: dict[str, Any] = field(default_factory=dict)
    fragment_plan: Any = None
    fragment_registration_result: Any = None
    exchange_sink_instance: Any = None
    task_context_info: dict[str, Any] | None = None
    output_buffers: dict[str, Any] | None = None
    dynamic_filter_domains: dict[str, Any] = field(default_factory=dict)
    descriptor_version: int = 0
    sealed: bool = False
    source_node_ids: set[str] = field(default_factory=set)
    dynamic_scan_source_node_ids: set[str] = field(default_factory=set)
    dynamic_exchange_source_node_ids: set[str] = field(default_factory=set)
    seen_sequences: dict[str, set[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.task_id = FteTaskId.coerce(self.task_id)
        self.fragment_id = str(self.fragment_id).strip()
        if not self.fragment_id:
            raise ValueError("fragment_id must be non-empty")
        self.context = dict(self.context or {})
        self.resource_request = dict(self.resource_request or {})
        self.task_context_info = dict(self.task_context_info or {})
        self.output_buffers = _normalize_output_buffers(self.output_buffers)
        self.dynamic_filter_domains = dict(self.dynamic_filter_domains or {})
        self.descriptor_version = int(self.descriptor_version)
        self.initial_splits = normalize_initial_splits(self.initial_splits)
        self.no_more_splits = {str(source_id) for source_id in self.no_more_splits}
        self.source_node_ids = {str(source_id) for source_id in (self.source_node_ids or set())}
        self.dynamic_scan_source_node_ids = {
            str(source_id) for source_id in (self.dynamic_scan_source_node_ids or set())
        }
        self.dynamic_exchange_source_node_ids = {
            str(source_id) for source_id in (self.dynamic_exchange_source_node_ids or set())
        }
        self.seen_sequences = {
            str(source_id): {int(sequence_id) for sequence_id in sequences}
            for source_id, sequences in (self.seen_sequences or {}).items()
        }
        for source_id, splits in self.initial_splits.items():
            self.source_node_ids.add(source_id)
            seen = self.seen_sequences.setdefault(source_id, set())
            seen.update(split.sequence_id for split in splits)
        self.source_node_ids.update(self.no_more_splits)

    def append_splits(
        self,
        source_node_id: str,
        splits: list[FteSplit | Mapping[str, Any]],
    ) -> list[FteSplit]:
        source_node_id = str(source_node_id)
        if source_node_id in self.no_more_splits and splits:
            raise RuntimeError(f"source {source_node_id} is already marked no_more_splits")
        normalized = normalize_initial_splits({source_node_id: splits}).get(source_node_id, [])
        if normalized:
            self.source_node_ids.add(source_node_id)
        target = self.initial_splits.setdefault(source_node_id, [])
        seen = self.seen_sequences.setdefault(source_node_id, set())
        added: list[FteSplit] = []
        for split in normalized:
            if split.source_node_id != source_node_id:
                raise ValueError("split source_node_id does not match descriptor update source")
            if split.sequence_id in seen:
                continue
            seen.add(split.sequence_id)
            target.append(split)
            added.append(split)
        return added

    def mark_no_more_splits(self, source_node_id: str) -> bool:
        source_node_id = str(source_node_id)
        self.source_node_ids.add(source_node_id)
        if source_node_id in self.no_more_splits:
            return False
        self.no_more_splits.add(source_node_id)
        return True

    def apply_task_update(self, update: FteTaskUpdateRequest | Mapping[str, Any] | None) -> bool:
        update_request = FteTaskUpdateRequest.coerce(update)
        changed = False
        for source_id, splits in update_request.initial_splits.items():
            changed = bool(self.append_splits(source_id, list(splits))) or changed
        for source_id in sorted(update_request.no_more_splits):
            changed = self.mark_no_more_splits(source_id) or changed
        if update_request.context:
            merged = dict(self.context)
            merged.update(update_request.context)
            if merged != self.context:
                self.context = merged
                changed = True
        if update_request.resource_request:
            merged = dict(self.resource_request)
            merged.update(update_request.resource_request)
            if merged != self.resource_request:
                self.resource_request = merged
                changed = True
        if update_request.fragment_plan_present and self.fragment_plan is not update_request.fragment_plan:
            self.fragment_plan = update_request.fragment_plan
            changed = True
        if update_request.output_buffers is not None:
            output_buffers, output_buffers_changed = _merge_output_buffers(
                self.output_buffers,
                update_request.output_buffers,
            )
            if output_buffers_changed:
                self.output_buffers = output_buffers
                changed = True
        if update_request.dynamic_filter_domains:
            merged = dict(self.dynamic_filter_domains)
            merged.update(update_request.dynamic_filter_domains)
            if merged != self.dynamic_filter_domains:
                self.dynamic_filter_domains = merged
                changed = True
        for attr in (
            "source_node_ids",
            "dynamic_scan_source_node_ids",
            "dynamic_exchange_source_node_ids",
        ):
            current = getattr(self, attr)
            merged = set(current) | set(getattr(update_request, attr))
            if merged != current:
                setattr(self, attr, merged)
                changed = True
        if changed:
            self.descriptor_version += 1
        return changed

    def to_create_task_request(
        self,
        attempt_id: int,
        *,
        exchange_sink_instance: Any = None,
    ) -> dict[str, Any]:
        payload = {
            "task_id": FteTaskAttemptId(self.task_id, attempt_id).to_dict(),
            "fragment_id": self.fragment_id,
            "context": dict(self.context),
            "initial_splits": {
                source_id: [split.to_dict() for split in splits] for source_id, splits in self.initial_splits.items()
            },
            "no_more_splits": sorted(self.no_more_splits),
            "resource_request": dict(self.resource_request),
            "fragment_plan": self.fragment_plan,
            "fragment_registration_result": self.fragment_registration_result,
            "descriptor_version": self.descriptor_version,
            "worker_runtime": FTE_WORKER_RUNTIME,
        }
        if self.output_buffers is not None:
            payload["output_buffers"] = dict(self.output_buffers)
        if self.dynamic_filter_domains:
            payload["dynamic_filter_domains"] = dict(self.dynamic_filter_domains)
        if self.source_node_ids:
            payload["source_node_ids"] = sorted(self.source_node_ids)
        if self.dynamic_scan_source_node_ids:
            payload["dynamic_scan_source_node_ids"] = sorted(self.dynamic_scan_source_node_ids)
        if self.dynamic_exchange_source_node_ids:
            payload["dynamic_exchange_source_node_ids"] = sorted(self.dynamic_exchange_source_node_ids)
        sink_instance = exchange_sink_instance if exchange_sink_instance is not None else self.exchange_sink_instance
        if sink_instance is not None:
            if hasattr(sink_instance, "to_dict"):
                payload["exchange_sink_instance"] = sink_instance.to_dict()
            else:
                payload["exchange_sink_instance"] = sink_instance
        return payload


class TaskDescriptorStorage:
    def __init__(
        self,
        *,
        max_in_memory_descriptors: int | None = None,
        spill_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self._descriptors: dict[FteTaskId, TaskDescriptor] = {}
        self._query_index: dict[str, set[FteTaskId]] = {}
        self._spill_paths: dict[FteTaskId, Path] = {}
        if max_in_memory_descriptors is None:
            max_in_memory_descriptors = fte_descriptor_max_in_memory()
        self._max_in_memory_descriptors = max(0, int(max_in_memory_descriptors))
        if spill_dir is None:
            spill_dir = fte_descriptor_spill_dir()
        self._spill_dir = Path(spill_dir) if spill_dir is not None else None
        if self._spill_dir is not None:
            self._spill_dir.mkdir(parents=True, exist_ok=True)

    def put(self, task_id: FteTaskId | str | Mapping[str, Any], descriptor: TaskDescriptor) -> None:
        key = FteTaskId.coerce(task_id)
        if descriptor.task_id != key:
            raise ValueError("descriptor task_id does not match storage key")
        self._descriptors[key] = descriptor
        self._query_index.setdefault(key.query_id, set()).add(key)
        self._remove_spill_file(key)
        self._enforce_memory_quota()

    def get(self, task_id: FteTaskId | str | Mapping[str, Any]) -> TaskDescriptor | None:
        key = FteTaskId.coerce(task_id)
        descriptor = self._descriptors.get(key)
        if descriptor is not None:
            return descriptor
        if key in self._spill_paths:
            return self._load_spilled_descriptor(key)
        return None

    def require(self, task_id: FteTaskId | str | Mapping[str, Any]) -> TaskDescriptor:
        key = FteTaskId.coerce(task_id)
        descriptor = self.get(key)
        if descriptor is None:
            raise KeyError(str(key))
        return descriptor

    def remove(self, task_id: FteTaskId | str | Mapping[str, Any]) -> TaskDescriptor | None:
        key = FteTaskId.coerce(task_id)
        descriptor = self._descriptors.pop(key, None)
        if descriptor is None and key in self._spill_paths:
            descriptor = self._load_spilled_descriptor(key, keep_in_memory=False)
        self._remove_spill_file(key)
        query_ids = self._query_index.get(key.query_id)
        if query_ids is not None:
            query_ids.discard(key)
            if not query_ids:
                self._query_index.pop(key.query_id, None)
        return descriptor

    def destroy_query(self, query_id: str) -> int:
        keys = self._query_index.pop(str(query_id), set())
        removed = 0
        for key in list(keys):
            if self._descriptors.pop(key, None) is not None or key in self._spill_paths:
                removed += 1
            self._remove_spill_file(key)
        return removed

    def __len__(self) -> int:
        return len(self._descriptors) + len(self._spill_paths)

    def stats(self) -> dict[str, int]:
        return {
            "in_memory": len(self._descriptors),
            "spilled": len(self._spill_paths),
            "total": len(self),
            "max_in_memory": self._max_in_memory_descriptors,
        }

    def _enforce_memory_quota(self) -> None:
        if self._max_in_memory_descriptors <= 0 or self._spill_dir is None:
            return
        while len(self._descriptors) > self._max_in_memory_descriptors:
            key = next(iter(self._descriptors))
            self._spill_descriptor(key)

    def _spill_descriptor(self, key: FteTaskId) -> None:
        if self._spill_dir is None:
            return
        descriptor = self._descriptors.get(key)
        if descriptor is None:
            return
        path = self._spill_dir / f"{key.query_id}_{key.fragment_execution_id}_{key.partition_id}.pickle"
        fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=self._spill_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                pickle.dump(descriptor, handle, protocol=pickle.HIGHEST_PROTOCOL)
            Path(tmp_name).replace(path)
            self._descriptors.pop(key, None)
            self._spill_paths[key] = path
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def _load_spilled_descriptor(
        self,
        key: FteTaskId,
        *,
        keep_in_memory: bool = True,
    ) -> TaskDescriptor | None:
        path = self._spill_paths.get(key)
        if path is None or not path.exists():
            return None
        with path.open("rb") as handle:
            descriptor = pickle.load(handle)
        if not isinstance(descriptor, TaskDescriptor):
            raise TypeError(f"spilled descriptor for {key} has invalid type")
        if keep_in_memory:
            self._descriptors[key] = descriptor
            self._remove_spill_file(key)
            self._query_index.setdefault(key.query_id, set()).add(key)
            self._enforce_memory_quota()
        return descriptor

    def _remove_spill_file(self, key: FteTaskId) -> None:
        path = self._spill_paths.pop(key, None)
        if path is not None:
            path.unlink(missing_ok=True)
