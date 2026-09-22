# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Local complete-task envelopes on the shared operator byte-budget policy.

The runtime ledger owns synchronization, allocation identity and transport.
These calculations acquire no capacity and never authorize a hard-limit escape.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import replace

from vane.execution.byte_budget import (
    ByteBudgetState,
    allocate_resource_reservations,
    build_byte_budget_state,
    byte_budget_block_reason,
)
from vane.execution.udf_data_admission import DataAdmissionLimits


def local_byte_budget_state(
    limits: DataAdmissionLimits,
    usage_by_unit: Mapping[str, int],
    output_usage_by_unit: Mapping[str, int],
    eligible_units: Set[str],
) -> ByteBudgetState:
    assert limits.unit_reservation_ratio is not None
    inactive_usage = sum(usage for key, usage in usage_by_unit.items() if key not in eligible_units)
    reserved = allocate_resource_reservations(
        {key: limits.task_bytes for key in sorted(eligible_units)},
        {key: limits.max_bytes for key in eligible_units},
        limit=max(0, limits.max_bytes - inactive_usage),
        reservation_ratio=limits.unit_reservation_ratio,
        integral=True,
    )
    shares = {key: int(value) for key, value in reserved.items()}
    return build_byte_budget_state(
        limit_bytes=limits.max_bytes,
        usage_by_unit=usage_by_unit,
        output_usage_by_unit=output_usage_by_unit,
        reserved_by_unit=shares,
        streaming_units=set(),
        output_reserved_by_unit={
            key: (share * limits.max_task_output_bytes + limits.task_bytes - 1) // limits.task_bytes
            for key, share in shares.items()
        },
    )


def local_task_budget_block_reason(
    limits: DataAdmissionLimits, budget: ByteBudgetState, resource_unit_id: str
) -> str | None:
    # Local shared memory cannot rely on Ray's spill or soft-limit escapes.
    if budget.query_usage_bytes + limits.task_bytes > limits.max_bytes:
        return "total_bytes"
    unit = budget.units[resource_unit_id]
    reason = byte_budget_block_reason(
        unit,
        limits.max_task_input_bytes,
        request_kind="task",
        usage_bytes=budget.query_usage_bytes,
        limit_bytes=limits.max_bytes,
        shared_used_bytes=budget.shared_used_bytes,
        shared_pool_bytes=budget.shared_pool_bytes,
    )
    if reason is not None:
        return f"unit_input_{reason}"
    # Evaluate output against the same locked state plus the candidate input.
    # Neither portion is published until both checks and transport admission pass.
    with_input = replace(unit, task_internal_usage_bytes=unit.task_internal_usage_bytes + limits.max_task_input_bytes)
    reason = byte_budget_block_reason(
        with_input,
        limits.max_task_output_bytes,
        request_kind="output",
        usage_bytes=budget.query_usage_bytes + limits.max_task_input_bytes,
        limit_bytes=limits.max_bytes,
        shared_used_bytes=budget.shared_used_bytes + with_input.shared_used_bytes - unit.shared_used_bytes,
        shared_pool_bytes=budget.shared_pool_bytes,
    )
    return None if reason is None else f"unit_output_{reason}"
