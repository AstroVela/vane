# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared logical resource declarations for local and Ray execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

_RESOURCE_ABS_TOLERANCE = 1e-12
_RESOURCE_REL_TOLERANCE = 1e-12


def _float_resource_leq(value: float, capacity: float) -> bool:
    return value <= capacity or math.isclose(
        value,
        capacity,
        rel_tol=_RESOURCE_REL_TOLERANCE,
        abs_tol=_RESOURCE_ABS_TOLERANCE,
    )


def _strict_fields(payload: Mapping[str, Any], expected: tuple[str, ...], type_name: str) -> None:
    actual = set(payload)
    expected_set = set(expected)
    unknown = sorted(actual - expected_set)
    missing = sorted(expected_set - actual)
    if unknown:
        raise ValueError(f"{type_name} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{type_name} is missing required fields: {', '.join(missing)}")


@dataclass(frozen=True)
class ResourceVector:
    """Logical resources requested by work or reserved by its owner.

    CPU and GPU are logical resources and may be fractional. Byte resources
    use integer accounting units. On a concrete task or actor, CPU, GPU, and
    declared heap become real Ray Core scheduling requests. On a
    ``QueryAllocation`` all four fields are soft admission/reservation shares:
    existing work is not revoked after an overage, and one bounded liveness
    path may escape a soft block. Object-store bytes additionally describe
    spillable flow rather than process memory. A vector is never allowed to
    carry negative capacity; subtraction that would underflow is a
    control-plane bug. Local resident model limits use the same arithmetic,
    but represent admission limits rather than OS memory or CPU enforcement.
    """

    cpu: float = 0.0
    gpu: float = 0.0
    heap_bytes: int = 0
    object_store_bytes: int = 0

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "cpu",
        "gpu",
        "heap_bytes",
        "object_store_bytes",
    )

    def __post_init__(self) -> None:
        for name in ("cpu", "gpu"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
            object.__setattr__(self, name, value)
        for name in ("heap_bytes", "object_store_bytes"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
            object.__setattr__(self, name, value)

    def __add__(self, other: ResourceVector) -> ResourceVector:
        if not isinstance(other, ResourceVector):
            return NotImplemented
        return ResourceVector(
            cpu=self.cpu + other.cpu,
            gpu=self.gpu + other.gpu,
            heap_bytes=self.heap_bytes + other.heap_bytes,
            object_store_bytes=self.object_store_bytes + other.object_store_bytes,
        )

    def __sub__(self, other: ResourceVector) -> ResourceVector:
        if not isinstance(other, ResourceVector):
            return NotImplemented
        values = {
            "cpu": self.cpu - other.cpu,
            "gpu": self.gpu - other.gpu,
            "heap_bytes": self.heap_bytes - other.heap_bytes,
            "object_store_bytes": self.object_store_bytes - other.object_store_bytes,
        }
        underflow = [name for name in ("heap_bytes", "object_store_bytes") if values[name] < 0]
        underflow.extend(
            name
            for name in ("cpu", "gpu")
            if not _float_resource_leq(float(getattr(other, name)), float(getattr(self, name)))
        )
        if underflow:
            raise ValueError(f"resource subtraction underflow: {', '.join(underflow)}")
        return ResourceVector(
            cpu=max(0.0, values["cpu"]),
            gpu=max(0.0, values["gpu"]),
            heap_bytes=int(values["heap_bytes"]),
            object_store_bytes=int(values["object_store_bytes"]),
        )

    def scale(self, factor: float) -> ResourceVector:
        factor = float(factor)
        if not math.isfinite(factor) or factor < 0:
            raise ValueError("resource scale factor must be finite and >= 0")
        return ResourceVector(
            cpu=self.cpu * factor,
            gpu=self.gpu * factor,
            heap_bytes=math.floor(self.heap_bytes * factor),
            object_store_bytes=math.floor(self.object_store_bytes * factor),
        )

    def fits_within(self, capacity: ResourceVector) -> bool:
        return (
            _float_resource_leq(self.cpu, capacity.cpu)
            and _float_resource_leq(self.gpu, capacity.gpu)
            and self.heap_bytes <= capacity.heap_bytes
            and self.object_store_bytes <= capacity.object_store_bytes
        )

    def exceeded_dimensions(self, capacity: ResourceVector) -> tuple[str, ...]:
        return tuple(
            name
            for name in self._FIELDS
            if (
                not _float_resource_leq(float(getattr(self, name)), float(getattr(capacity, name)))
                if name in {"cpu", "gpu"}
                else getattr(self, name) > getattr(capacity, name)
            )
        )

    def dominant_share(self, capacity: ResourceVector) -> float:
        shares: list[float] = []
        for name in self._FIELDS:
            demand = float(getattr(self, name))
            available = float(getattr(capacity, name))
            if demand <= 0:
                shares.append(0.0)
            elif available <= 0:
                shares.append(math.inf)
            else:
                shares.append(demand / available)
        return max(shares, default=0.0)

    def is_zero(self) -> bool:
        return all(getattr(self, name) == 0 for name in self._FIELDS)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "cpu": self.cpu,
            "gpu": self.gpu,
            "heap_bytes": self.heap_bytes,
            "object_store_bytes": self.object_store_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResourceVector:
        values = dict(payload)
        _strict_fields(values, cls._FIELDS, cls.__name__)
        return cls(
            cpu=float(values["cpu"]),
            gpu=float(values["gpu"]),
            heap_bytes=int(values["heap_bytes"]),
            object_store_bytes=int(values["object_store_bytes"]),
        )


def udf_process_resources(payload: Mapping[str, Any]) -> ResourceVector:
    """Parse one UDF task/actor's declared process resources.

    Actor invocations do not reserve these again: the pool owns the process
    reservation. Object-store/shared-memory flow has a separate lifetime.
    Missing memory declarations reserve zero bytes, as in Ray's resource graph.
    """
    values: dict[str, float] = {}
    for name, default in (("cpus", 1.0), ("gpus", 0.0)):
        try:
            value = float(payload.get(name, default))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite non-negative number") from error
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
        values[name] = value
    if values["cpus"] <= 0 and values["gpus"] <= 0:
        raise ValueError("UDF must request CPU or GPU resources")
    declared_heap = payload.get("memory_bytes")
    heap_bytes = 0
    if declared_heap is not None:
        if type(declared_heap) is not int or declared_heap <= 0:
            raise ValueError("memory_bytes must be a positive integer")
        heap_bytes = declared_heap
    return ResourceVector(cpu=values["cpus"], gpu=values["gpus"], heap_bytes=heap_bytes)
