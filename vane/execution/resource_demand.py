# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Resource declarations for submission; reservations belong to the scheduler."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vane.execution.plan import _fields


def _capacity(value: int, name: str, *, minimum: int = 1) -> None:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise ValueError(f"{name} must be an integer between {minimum} and 9223372036854775807")


@dataclass(frozen=True)
class MemoryDemand:
    operator_bytes: int
    result_bytes: int
    exchange_bytes: int = 0
    staging_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("operator_bytes", "result_bytes", "exchange_bytes", "staging_bytes"):
            _capacity(getattr(self, name), name, minimum=1 if name in {"operator_bytes", "result_bytes"} else 0)

    def to_dict(self) -> dict[str, int]:
        return {
            "operator_bytes": self.operator_bytes,
            "result_bytes": self.result_bytes,
            "exchange_bytes": self.exchange_bytes,
            "staging_bytes": self.staging_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MemoryDemand:
        _fields(value, {"operator_bytes", "result_bytes", "exchange_bytes", "staging_bytes"}, cls.__name__)
        return cls(**value)


@dataclass(frozen=True)
class ResourceDemand:
    """Query-level declared capacity, without claiming it has been reserved.

    The initial operator set is CPU-only. GPU and UDF ownership will extend
    this contract together with their executable capability, not in advance.
    """

    cpu_share: float
    task_contexts: int
    memory: MemoryDemand
    io_concurrency: int

    def __post_init__(self) -> None:
        if isinstance(self.cpu_share, bool) or not isinstance(self.cpu_share, (int, float)):
            raise ValueError("cpu_share must be a positive finite number")
        try:
            cpu = float(self.cpu_share)
        except OverflowError as exc:
            raise ValueError("cpu_share must be finite") from exc
        if not math.isfinite(cpu) or cpu <= 0:
            raise ValueError("cpu_share must be a positive finite number")
        object.__setattr__(self, "cpu_share", cpu)
        _capacity(self.task_contexts, "task_contexts")
        _capacity(self.io_concurrency, "io_concurrency")
        if not isinstance(self.memory, MemoryDemand):
            raise ValueError("memory must be MemoryDemand")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_share": self.cpu_share,
            "task_contexts": self.task_contexts,
            "memory": self.memory.to_dict(),
            "io_concurrency": self.io_concurrency,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResourceDemand:
        _fields(value, {"cpu_share", "task_contexts", "memory", "io_concurrency"}, cls.__name__)
        return cls(
            value["cpu_share"], value["task_contexts"], MemoryDemand.from_dict(value["memory"]), value["io_concurrency"]
        )
