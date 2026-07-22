# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


class FteTaskState(str, Enum):
    PLANNED = "PLANNED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"
    ABORTED = "ABORTED"


class FtePartitionState(str, Enum):
    OPEN = "OPEN"
    SEALED = "SEALED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"


class FteTaskExecutionClass(str, Enum):
    STANDARD = "STANDARD"
    SPECULATIVE = "SPECULATIVE"
    EAGER_SPECULATIVE = "EAGER_SPECULATIVE"

    @classmethod
    def coerce(cls, value: Any) -> FteTaskExecutionClass:
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.STANDARD
        text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        if not text:
            return cls.STANDARD
        aliases = {
            "NORMAL": cls.STANDARD,
            "NON_SPECULATIVE": cls.STANDARD,
            "EAGER": cls.EAGER_SPECULATIVE,
        }
        if text in aliases:
            return aliases[text]
        try:
            return cls(text)
        except ValueError as exc:
            raise ValueError(f"unknown FTE task execution class: {value!r}") from exc

    @property
    def is_speculative(self) -> bool:
        return self in (self.SPECULATIVE, self.EAGER_SPECULATIVE)

    def can_transition_to(self, target: Any) -> bool:
        target_class = self.coerce(target)
        if self == self.STANDARD:
            return target_class == self.STANDARD
        return target_class in (self, self.STANDARD)


_TASK_EXECUTION_CLASS_KEYS = (
    "fte_task_execution_class",
    "task_execution_class",
    "taskExecutionClass",
    "execution_class",
)


def fte_task_execution_class_metadata_present(
    *metadata_items: Mapping[str, Any] | None,
) -> bool:
    for metadata in metadata_items:
        if not metadata:
            continue
        for key in _TASK_EXECUTION_CLASS_KEYS:
            if key not in metadata:
                continue
            value = metadata.get(key)
            if value is not None and value != "":
                return True
    return False


def fte_task_execution_class_from_metadata(
    *metadata_items: Mapping[str, Any] | None,
) -> FteTaskExecutionClass:
    for metadata in metadata_items:
        if not metadata:
            continue
        for key in _TASK_EXECUTION_CLASS_KEYS:
            if key not in metadata:
                continue
            value = metadata.get(key)
            if value is None or value == "":
                continue
            return FteTaskExecutionClass.coerce(value)
    return FteTaskExecutionClass.STANDARD


_TERMINAL_STATES = {
    FteTaskState.FINISHED,
    FteTaskState.FAILED,
    FteTaskState.CANCELED,
    FteTaskState.ABORTED,
}
