# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared reservation arithmetic for operators, tasks, and output capacity.

Adapters supply validated demands, phase eligibility, and usage snapshots.
They retain synchronization, physical capacity checks, and execution ownership.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Set
from dataclasses import dataclass


def allocate_resource_reservations(
    baselines: Mapping[str, int | float],
    maxima: Mapping[str, int | float],
    *,
    limit: int | float,
    reservation_ratio: float,
    integral: bool,
    arithmetic_tolerance: float = 0.0,
) -> dict[str, int | float]:
    """Protect baselines plus equal surplus, or apportion a smaller budget.

    Maxima cap each reservation; unused and rounded capacity stays shared.
    The caller chooses its arithmetic tolerance. A soft reservation is not
    authorization to exceed a backend's hard resource capacity.
    """
    if not baselines:
        return {}
    baseline_total = sum(baselines.values())
    if baseline_total <= limit + arithmetic_tolerance:
        bonus_pool = max(0.0, limit - baseline_total) * reservation_ratio
        bonus = bonus_pool / len(baselines)
        reserved = {key: baseline + bonus for key, baseline in baselines.items()}
    else:
        reservation_pool = limit * reservation_ratio
        reserved = {key: reservation_pool * baseline / baseline_total for key, baseline in baselines.items()}
    reserved = {key: min(value, maxima[key]) for key, value in reserved.items()}
    if integral:
        reserved = {key: math.floor(value) for key, value in reserved.items()}
    return reserved


@dataclass(frozen=True)
class ByteBudgetUsage:
    task_reserved_bytes: int
    output_reserved_bytes: int
    task_internal_usage_bytes: int
    output_usage_bytes: int

    @property
    def task_budget_usage_bytes(self) -> int:
        return self.task_internal_usage_bytes + max(0, self.output_usage_bytes - self.output_reserved_bytes)

    @property
    def task_reserved_remaining_bytes(self) -> int:
        return max(0, self.task_reserved_bytes - self.task_budget_usage_bytes)

    @property
    def output_reserved_remaining_bytes(self) -> int:
        return max(0, self.output_reserved_bytes - self.output_usage_bytes)

    @property
    def shared_used_bytes(self) -> int:
        return max(0, self.task_budget_usage_bytes - self.task_reserved_bytes)


@dataclass(frozen=True)
class ByteBudgetState:
    limit_bytes: int
    ineligible_usage_bytes: int
    shared_pool_bytes: int
    shared_used_bytes: int
    query_usage_bytes: int
    reservation_unit_ids: tuple[str, ...]
    units: dict[str, ByteBudgetUsage]

    @property
    def shared_remaining_bytes(self) -> int:
        return max(0, self.shared_pool_bytes - self.shared_used_bytes)


def build_byte_budget_state(
    *,
    limit_bytes: int,
    usage_by_unit: Mapping[str, int],
    output_usage_by_unit: Mapping[str, int],
    reserved_by_unit: Mapping[str, int],
    streaming_units: Set[str],
) -> ByteBudgetState:
    """Separate protected task/output bytes, shared usage, and retired usage.

    The reservation mapping identifies eligible units, including zero shares.
    Callers calculate those shares after deducting ineligible retained bytes.
    Streaming units protect half their share for output, rounded upward.
    This describes soft debt as well; strict transports enforce their total
    capacity separately and cannot assume spill will make a grant fit.
    """
    if not reserved_by_unit.keys() <= usage_by_unit.keys():
        raise ValueError("byte reservations require usage for every eligible unit")
    units = {}
    ineligible_usage = shared_used = 0
    for key, usage in usage_by_unit.items():
        output = output_usage_by_unit[key]
        if output > usage:
            raise RuntimeError(f"unit {key} output usage exceeds total byte usage")
        reserved = reserved_by_unit.get(key, 0)
        output_reserved = (reserved + 1) // 2 if key in streaming_units else 0
        budget = ByteBudgetUsage(
            task_reserved_bytes=reserved - output_reserved,
            output_reserved_bytes=output_reserved,
            task_internal_usage_bytes=usage - output,
            output_usage_bytes=output,
        )
        units[key] = budget
        if key in reserved_by_unit:
            shared_used += budget.shared_used_bytes
        else:
            ineligible_usage += usage
    return ByteBudgetState(
        limit_bytes=limit_bytes,
        ineligible_usage_bytes=ineligible_usage,
        shared_pool_bytes=max(0, limit_bytes - ineligible_usage - sum(reserved_by_unit.values())),
        shared_used_bytes=shared_used,
        query_usage_bytes=sum(usage_by_unit.values()),
        reservation_unit_ids=tuple(reserved_by_unit),
        units=units,
    )


def byte_budget_block_reason(
    unit: ByteBudgetUsage,
    amount: int,
    *,
    request_kind: str,
    usage_bytes: int,
    limit_bytes: int,
    shared_used_bytes: int,
    shared_pool_bytes: int,
) -> str | None:
    """Protect output completion before considering shared-pool capacity.

    Owners supply their reservation policy and serialize this check with the
    corresponding acquire. Transport, query authorization and wakes stay with
    those owners. Ray may preserve protected shares while over its soft limit;
    a strict owner must never overcommit the supplied reservations.
    """
    if request_kind not in {"task", "output"}:
        raise ValueError(f"invalid byte request kind: {request_kind}")
    protected = unit.task_reserved_remaining_bytes
    if request_kind == "output":
        protected += unit.output_reserved_remaining_bytes
    shared_need = max(0, amount - protected)
    if shared_need == 0:
        return None
    if usage_bytes + amount > limit_bytes:
        return "total_bytes"
    if shared_used_bytes + shared_need > shared_pool_bytes:
        return "shared_bytes"
    return None
