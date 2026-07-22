# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class TaskResultState(str, Enum):
    NOT_READY = "NOT_READY"
    NO_OUTPUT = "NO_OUTPUT"
    MATERIALIZED_OUTPUT = "MATERIALIZED_OUTPUT"
    ERROR = "ERROR"


@dataclass(frozen=True)
class TaskResultPoll:
    state: TaskResultState
    output: Any | None = None
    error: BaseException | None = None


class TaskResultHandle(Protocol):
    def task_context(self) -> Any: ...

    def fte_task_id(self) -> str: ...

    def worker_id(self) -> str: ...

    def poll(self) -> TaskResultPoll: ...

    def ack(self) -> None: ...

    def release_result_payload(self) -> None: ...


class WorkerHandle(Protocol):
    @property
    def worker_id(self) -> str: ...

    def fte_create_task(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def fte_add_splits(
        self,
        task_id: str | Mapping[str, Any],
        source_node_id: str,
        splits: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]: ...

    def fte_no_more_splits(
        self,
        task_id: str | Mapping[str, Any],
        source_node_id: str,
    ) -> Mapping[str, Any]: ...

    def fte_update_task(
        self,
        task_id: str | Mapping[str, Any],
        update: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def fte_wait_task_status(
        self,
        task_id: str | Mapping[str, Any],
        min_version: int | None = None,
        timeout_s: float | None = None,
    ) -> Mapping[str, Any]: ...

    def fte_cancel_task(self, task_id: str | Mapping[str, Any]) -> Mapping[str, Any]: ...


class FragmentTemplateStore(Protocol):
    def put(self, query_id: str, fragment_id: str, plan: Any) -> Any: ...

    def get(self, query_id: str, fragment_id: str) -> Any: ...

    def drop_query(self, query_id: str) -> None: ...


class ResourceSnapshotProvider(Protocol):
    def snapshots(self) -> Sequence[Mapping[str, Any]]: ...


class WorkerManagerBackend(Protocol):
    def worker_snapshots(self) -> Sequence[Mapping[str, Any]]: ...

    def submit_tasks(self, tasks: Sequence[Any]) -> Sequence[TaskResultHandle]: ...

    def task_input_stream_exhausted(
        self,
        query_id: str,
        source_node_ids: Sequence[str],
    ) -> None: ...

    def wait_query(
        self,
        query_id: str,
        timeout_s: float,
        task_context_filter: Sequence[Any] | None = None,
    ) -> Sequence[Any]: ...

    def drop_query(self, query_id: str) -> None: ...

    def shutdown(self) -> None: ...
