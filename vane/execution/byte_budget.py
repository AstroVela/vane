# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared integer-byte policy for protected task and output capacity."""

from dataclasses import dataclass


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
