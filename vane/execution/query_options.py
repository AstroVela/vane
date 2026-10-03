# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Immutable query configuration for native local and distributed Ray execution.

These contracts do not start queries or change the existing public connection
API. Local execution deliberately has no distributed execution mode.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DistributedMode(str, Enum):
    PIPELINED = "pipelined"
    FTE = "fte"


def _fields(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if set(value) != expected:
        raise ValueError(f"{name} must contain exactly these fields: {', '.join(sorted(expected))}")


def _seconds(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        comparison = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be finite and {comparison}")
    return result


@dataclass(frozen=True)
class FteOptions:
    """Recovery policy referencing a store registered with the query service.

    The store name is not a claim of durability. The service must resolve it
    and verify its failure domain before admitting a distributed query.
    """

    exchange_store: str
    max_attempts: int
    retry_backoff_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.exchange_store, str) or not self.exchange_store.strip():
            raise ValueError("exchange_store must be a non-empty registered store name")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 2**31 - 1:
            raise ValueError("max_attempts must be an integer between 1 and 2147483647")
        object.__setattr__(
            self,
            "retry_backoff_seconds",
            _seconds(self.retry_backoff_seconds, "retry_backoff_seconds", allow_zero=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_store": self.exchange_store,
            "max_attempts": self.max_attempts,
            "retry_backoff_seconds": self.retry_backoff_seconds,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FteOptions:
        _fields(value, {"exchange_store", "max_attempts", "retry_backoff_seconds"}, "FteOptions")
        return cls(**value)


@dataclass(frozen=True)
class LocalExecution:
    """Execute a native query without a distributed mode or task protocol."""

    def to_dict(self) -> dict[str, Any]:
        return {"backend": "local"}


@dataclass(frozen=True)
class RayExecution:
    mode: DistributedMode = DistributedMode.PIPELINED
    fte_options: FteOptions | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "mode", DistributedMode(self.mode))
        except (ValueError, TypeError) as exc:
            raise ValueError("Ray execution mode must be 'pipelined' or 'fte'") from exc
        if self.mode is DistributedMode.FTE:
            if not isinstance(self.fte_options, FteOptions):
                raise ValueError("Ray FTE requires FteOptions")
        elif self.fte_options is not None:
            raise ValueError("pipelined execution does not accept FTE options")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "ray",
            "execution": self.mode.value,
            "fte_options": self.fte_options.to_dict() if self.fte_options is not None else None,
        }


ExecutionTarget = LocalExecution | RayExecution


def select_execution(
    backend: str,
    *,
    execution: str | DistributedMode | None = None,
    fte_options: FteOptions | None = None,
) -> ExecutionTarget:
    """Validate an explicit target without consulting process environment."""
    if backend == "local":
        if execution is not None or fte_options is not None:
            raise ValueError("local execution does not accept a distributed mode or FTE options")
        return LocalExecution()
    if backend == "ray":
        return RayExecution(
            mode=DistributedMode.PIPELINED if execution is None else DistributedMode(execution),
            fte_options=fte_options,
        )
    raise ValueError("backend must be 'local' or 'ray'")


def execution_from_dict(value: Mapping[str, Any]) -> ExecutionTarget:
    if not isinstance(value, Mapping):
        raise ValueError("execution target must be an object")
    if value.get("backend") == "local":
        _fields(value, {"backend"}, "LocalExecution")
        return LocalExecution()
    if value.get("backend") == "ray":
        _fields(value, {"backend", "execution", "fte_options"}, "RayExecution")
        fte = value["fte_options"]
        return RayExecution(
            mode=value["execution"],
            fte_options=FteOptions.from_dict(fte) if fte is not None else None,
        )
    raise ValueError("backend must be 'local' or 'ray'")


@dataclass(frozen=True)
class QueryExecutionOptions:
    """A per-submission snapshot; timeout clocks start in the executor."""

    target: ExecutionTarget
    admission_timeout: float
    execution_timeout: float
    delivery_timeout: float

    def __post_init__(self) -> None:
        if not isinstance(self.target, (LocalExecution, RayExecution)):
            raise ValueError("target must be LocalExecution or RayExecution")
        for name in ("admission_timeout", "execution_timeout", "delivery_timeout"):
            object.__setattr__(self, name, _seconds(getattr(self, name), name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "admission_timeout": self.admission_timeout,
            "execution_timeout": self.execution_timeout,
            "delivery_timeout": self.delivery_timeout,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> QueryExecutionOptions:
        _fields(value, {"target", "admission_timeout", "execution_timeout", "delivery_timeout"}, cls.__name__)
        return cls(
            target=execution_from_dict(value["target"]),
            admission_timeout=value["admission_timeout"],
            execution_timeout=value["execution_timeout"],
            delivery_timeout=value["delivery_timeout"],
        )
