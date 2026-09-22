# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from vane.execution.byte_budget import (
    allocate_resource_reservations,
    build_byte_budget_state,
    byte_budget_block_reason,
)


@pytest.mark.parametrize(
    ("baselines", "maxima", "limit", "ratio", "integral", "expected"),
    [
        ({}, {}, 0, 0.5, False, {}),
        ({"actor": 4, "task": 1}, {"actor": 4, "task": 3}, 9, 0.5, False, {"actor": 4, "task": 2}),
        ({"a": 4, "b": 2}, {"a": 4, "b": 2}, 3, 0.5, False, {"a": 1, "b": 0.5}),
        ({"a": 0, "b": 0}, {"a": 7, "b": 7}, 7, 1.0, True, {"a": 3, "b": 3}),
        ({"a": 0, "b": 0}, {"a": 7, "b": 7}, 7, 0.5, True, {"a": 1, "b": 1}),
        ({"a": 0, "b": 0}, {"a": 7, "b": 7}, 7, 0.0, True, {"a": 0, "b": 0}),
        ({"a": 0, "b": 0}, {"a": 1, "b": 10}, 10, 1.0, True, {"a": 1, "b": 5}),
        ({"a": 1, "b": 3}, {"a": 1, "b": 3}, 0, 0.5, True, {"a": 0, "b": 0}),
    ],
)
def test_reservations_preserve_baselines_caps_and_unallocated_shared_capacity(
    baselines, maxima, limit, ratio, integral, expected
):
    assert (
        allocate_resource_reservations(baselines, maxima, limit=limit, reservation_ratio=ratio, integral=integral)
        == expected
    )


def test_zero_capacity_is_exact_unless_the_adapter_explicitly_requests_tolerance():
    options = {"limit": 0, "reservation_ratio": 0.5, "integral": False}
    baseline, maximum = {"task": 1e-13}, {"task": 1}
    assert allocate_resource_reservations(baseline, maximum, **options) == {"task": 0}
    assert allocate_resource_reservations(baseline, maximum, **options, arithmetic_tolerance=1e-9) == {"task": 1e-13}


def test_retired_output_remains_charged_while_active_units_share_remaining_bytes():
    state = build_byte_budget_state(
        limit_bytes=20,
        usage_by_unit={"a": 5, "b": 7, "retired": 6},
        output_usage_by_unit={"a": 1, "b": 5, "retired": 6},
        reserved_by_unit={"a": 3, "b": 3},
        streaming_units={"a", "b", "retired"},
    )
    assert state.query_usage_bytes == 18
    assert state.ineligible_usage_bytes == 6
    assert state.shared_pool_bytes == 8
    assert state.shared_used_bytes == 7
    assert state.shared_remaining_bytes == 1
    assert state.units["a"].task_reserved_bytes == 1
    assert state.units["a"].output_reserved_bytes == 2
    assert state.units["retired"].output_usage_bytes == 6
    assert state.units["retired"].output_reserved_bytes == 0


def test_zero_share_is_distinct_from_an_ineligible_operator():
    state = build_byte_budget_state(
        limit_bytes=10,
        usage_by_unit={"active": 5, "retired": 6},
        output_usage_by_unit={"active": 5, "retired": 6},
        reserved_by_unit={"active": 0},
        streaming_units={"active", "retired"},
    )
    assert state.reservation_unit_ids == ("active",)
    assert state.shared_used_bytes == 5
    assert state.ineligible_usage_bytes == 6
    assert state.shared_remaining_bytes == 0


def test_task_admission_preserves_output_completion_capacity():
    state = build_byte_budget_state(
        limit_bytes=4,
        usage_by_unit={"a": 0},
        output_usage_by_unit={"a": 0},
        reserved_by_unit={"a": 4},
        streaming_units={"a"},
    )
    options = {
        "usage_bytes": state.query_usage_bytes,
        "limit_bytes": state.limit_bytes,
        "shared_used_bytes": state.shared_used_bytes,
        "shared_pool_bytes": state.shared_pool_bytes,
    }
    assert byte_budget_block_reason(state.units["a"], 3, request_kind="task", **options) == "shared_bytes"
    assert byte_budget_block_reason(state.units["a"], 3, request_kind="output", **options) is None


def test_soft_debt_does_not_remove_another_operators_protected_share():
    state = build_byte_budget_state(
        limit_bytes=4,
        usage_by_unit={"busy": 10, "ready": 0},
        output_usage_by_unit={"busy": 10, "ready": 0},
        reserved_by_unit={"busy": 2, "ready": 2},
        streaming_units={"busy", "ready"},
    )
    assert state.query_usage_bytes > state.limit_bytes
    assert (
        byte_budget_block_reason(
            state.units["ready"],
            1,
            request_kind="output",
            usage_bytes=state.query_usage_bytes,
            limit_bytes=state.limit_bytes,
            shared_used_bytes=state.shared_used_bytes,
            shared_pool_bytes=state.shared_pool_bytes,
        )
        is None
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_byte_partitions_conserve_usage_across_output_and_task_ownership(streaming):
    for reserved in range(8):
        for internal in range(9):
            for output in range(9):
                state = build_byte_budget_state(
                    limit_bytes=20,
                    usage_by_unit={"a": internal + output, "retired": 3},
                    output_usage_by_unit={"a": output, "retired": 3},
                    reserved_by_unit={"a": reserved},
                    streaming_units={"a"} if streaming else set(),
                )
                unit = state.units["a"]
                protected_output = min(output, unit.output_reserved_bytes)
                protected_task = min(internal + output - protected_output, unit.task_reserved_bytes)
                assert state.query_usage_bytes == (
                    protected_output + protected_task + state.shared_used_bytes + state.ineligible_usage_bytes
                )
                assert state.shared_pool_bytes + reserved + state.ineligible_usage_bytes == state.limit_bytes


def test_byte_snapshot_requires_complete_accounting():
    options = {"limit_bytes": 10, "usage_by_unit": {"a": 1}, "streaming_units": set()}
    with pytest.raises(ValueError, match="every eligible unit"):
        build_byte_budget_state(**options, output_usage_by_unit={"a": 0}, reserved_by_unit={"missing": 1})
    with pytest.raises(RuntimeError, match="output usage exceeds"):
        build_byte_budget_state(**options, output_usage_by_unit={"a": 2}, reserved_by_unit={"a": 1})
    with pytest.raises(KeyError):
        build_byte_budget_state(**options, output_usage_by_unit={}, reserved_by_unit={"a": 1})


@pytest.mark.parametrize("output_share", [-1, 8, 1.5, True])
def test_explicit_output_protection_stays_within_integer_unit_share(output_share):
    with pytest.raises(ValueError, match="integer bytes"):
        build_byte_budget_state(
            limit_bytes=10,
            usage_by_unit={"a": 0},
            output_usage_by_unit={"a": 0},
            reserved_by_unit={"a": 7},
            streaming_units={"a"},
            output_reserved_by_unit={"a": output_share},
        )


def test_explicit_output_protection_requires_an_eligible_unit():
    with pytest.raises(ValueError, match="eligible unit"):
        build_byte_budget_state(
            limit_bytes=10,
            usage_by_unit={"retired": 3},
            output_usage_by_unit={"retired": 3},
            reserved_by_unit={},
            streaming_units={"retired"},
            output_reserved_by_unit={"retired": 1},
        )
