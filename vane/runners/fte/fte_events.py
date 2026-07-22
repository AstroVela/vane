# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vane.runners.fte import FteTaskAttemptId


@dataclass(frozen=True)
class FteEvent:
    query_id: str

    @property
    def event_type(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class SplitEventsSubmitted(FteEvent):
    events: tuple[dict[str, Any], ...]

    @classmethod
    def from_events(
        cls,
        query_id: str,
        events: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> SplitEventsSubmitted:
        return cls(str(query_id), tuple(dict(event) for event in events))


@dataclass(frozen=True)
class SourceInputExhausted(FteEvent):
    source_node_ids: frozenset[str]

    @classmethod
    def from_source_node_ids(
        cls,
        query_id: str,
        source_node_ids: set[str] | list[str] | tuple[str, ...] | frozenset[str],
    ) -> SourceInputExhausted:
        return cls(str(query_id), frozenset(str(source_node_id) for source_node_id in source_node_ids))


@dataclass(frozen=True)
class TaskStatusChanged(FteEvent):
    attempt_id: FteTaskAttemptId
    status: dict[str, Any]

    @classmethod
    def from_status(
        cls,
        query_id: str,
        attempt_id: FteTaskAttemptId | str | dict[str, Any],
        status: dict[str, Any],
    ) -> TaskStatusChanged:
        from vane.runners.fte import FteTaskAttemptId

        return cls(str(query_id), FteTaskAttemptId.coerce(attempt_id), dict(status))


@dataclass(frozen=True)
class WorkerFailed(FteEvent):
    worker_id: str
    error: Any = None
    failed_worker_ids: frozenset[str] | None = None


@dataclass(frozen=True)
class MemoryPressureDetected(FteEvent):
    max_count_per_worker: int | None = None


@dataclass(frozen=True)
class ResourceAdmissionChanged(FteEvent):
    """Wake pending FTE reservations after query resource ownership changes."""


@dataclass(frozen=True)
class WorkerReservationCompleted(FteEvent):
    fragment_execution_id: int
    fragment_id: str
    partition_id: int
    reservation_generation: int
    worker_id: str | None = None
    error: Any = None


@dataclass(frozen=True)
class RetryDelayExpired(FteEvent):
    generation: int


@dataclass(frozen=True)
class ExchangeSelectorUpdated(FteEvent):
    consumer_fragment_id: str
    source_node_id: str
    selector: dict[str, Any]

    @classmethod
    def from_selector(
        cls,
        query_id: str,
        consumer_fragment_id: str,
        source_node_id: str,
        *,
        selector: Mapping[str, Any] | None = None,
    ) -> ExchangeSelectorUpdated:
        if selector is None:
            raise ValueError("FTE exchange selector update requires selector payload")
        return cls(
            str(query_id),
            str(consumer_fragment_id),
            str(source_node_id),
            dict(selector),
        )


@dataclass(frozen=True)
class QueryAbort(FteEvent):
    reason: Any = None


@dataclass(frozen=True)
class FteWorkerCommand:
    query_id: str
    fragment_id: str
    worker_id: str | None
    worker: Any

    @property
    def command_type(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class FteCreateTaskCommand(FteWorkerCommand):
    attempt_id: FteTaskAttemptId
    partition_id: int
    request: dict[str, Any]


@dataclass(frozen=True)
class FteAddSplitsCommand(FteWorkerCommand):
    attempt_id: FteTaskAttemptId
    source_node_id: str
    splits: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FteNoMoreSplitsCommand(FteWorkerCommand):
    attempt_id: FteTaskAttemptId
    source_node_id: str


@dataclass(frozen=True)
class FteTaskUpdateCommand(FteWorkerCommand):
    attempt_id: FteTaskAttemptId
    update: dict[str, Any]
