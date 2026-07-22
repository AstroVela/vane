# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vane.runners.fte.fte_descriptor import TaskDescriptor
    from vane.runners.fte.fte_state import FteTaskExecutionClass
    from vane.runners.fte.fte_types import FteTaskAttemptId, FteTaskId


@dataclass(frozen=True)
class ReadyTask:
    task_id: FteTaskId
    reason: str = "ready"


@dataclass(frozen=True)
class RunningAttempt:
    attempt_id: FteTaskAttemptId
    worker_id: str | None = None
    remote_handle: Any = None
    sink_instance: Any = None
    started_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ScheduledAttempt:
    attempt_id: FteTaskAttemptId
    descriptor: TaskDescriptor
    request: dict[str, Any]
    sink_instance: Any = None
    worker_id: str | None = None


@dataclass(frozen=True)
class ExecutionClassTransition:
    task_id: FteTaskId
    new_execution_class: FteTaskExecutionClass
    running_attempts: tuple[RunningAttempt, ...] = ()


@dataclass(frozen=True)
class RevokedAttempt:
    attempt_id: FteTaskAttemptId
    worker_id: str | None = None
    remote_handle: Any = None
    retry_ready: bool = False


@dataclass(frozen=True)
class FinishedAttempt:
    attempt_id: FteTaskAttemptId
    sink_instance: Any = None
    output_stats: Any = None


@dataclass(frozen=True)
class FragmentExecutionMutationResult(Sequence[ScheduledAttempt]):
    scheduled_attempts: tuple[ScheduledAttempt, ...] = ()
    worker_commands: tuple[Any, ...] = ()

    def __iter__(self):
        return iter(self.scheduled_attempts)

    def __len__(self) -> int:
        return len(self.scheduled_attempts)

    def __getitem__(self, index):
        return self.scheduled_attempts[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FragmentExecutionMutationResult):
            return self.scheduled_attempts == other.scheduled_attempts and self.worker_commands == other.worker_commands
        if isinstance(other, list):
            return list(self.scheduled_attempts) == other
        if isinstance(other, tuple):
            return self.scheduled_attempts == other
        return False

    @classmethod
    def from_attempts(
        cls,
        scheduled_attempts: list[ScheduledAttempt],
        worker_commands: list[Any] | None = None,
    ) -> FragmentExecutionMutationResult:
        return cls(
            scheduled_attempts=tuple(scheduled_attempts),
            worker_commands=tuple(worker_commands or []),
        )
