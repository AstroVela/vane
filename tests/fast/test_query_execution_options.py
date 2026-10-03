# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from vane.execution.query_options import (
    DistributedMode,
    FteOptions,
    LocalExecution,
    QueryExecutionOptions,
    RayExecution,
    execution_from_dict,
    select_execution,
)


def _fte():
    return FteOptions(exchange_store="durable-test-store", max_attempts=3, retry_backoff_seconds=0.25)


def test_local_has_no_distributed_mode_even_when_environment_selects_ray(monkeypatch):
    monkeypatch.setenv("VANE_RUNNER", "ray")
    monkeypatch.setenv("VANE_EXECUTION_POLICY", "fte")

    target = select_execution("local")

    assert target == LocalExecution()
    assert not hasattr(target, "mode")
    assert target.to_dict() == {"backend": "local"}
    assert execution_from_dict(target.to_dict()) == target


@pytest.mark.parametrize("execution", ["pipelined", "fte", "native", ""])
def test_local_rejects_any_explicit_distributed_strategy(execution):
    with pytest.raises(ValueError, match="local execution"):
        select_execution("local", execution=execution)
    with pytest.raises(ValueError, match="LocalExecution"):
        execution_from_dict({"backend": "local", "execution": execution})


def test_local_does_not_accept_retry_configuration():
    with pytest.raises(ValueError, match="local execution"):
        select_execution("local", fte_options=_fte())


def test_ray_modes_and_per_submission_overrides_are_independent():
    defaults = QueryExecutionOptions(
        target=select_execution("ray"), admission_timeout=10, execution_timeout=60, delivery_timeout=120
    )
    submission = replace(defaults, target=select_execution("ray", execution="fte", fte_options=_fte()))

    assert defaults.target == RayExecution(DistributedMode.PIPELINED)
    assert submission.target == RayExecution(DistributedMode.FTE, _fte())
    assert QueryExecutionOptions.from_dict(json.loads(json.dumps(submission.to_dict()))) == submission
    with pytest.raises(FrozenInstanceError):
        submission.target = defaults.target
    with pytest.raises(FrozenInstanceError):
        submission.target.fte_options.max_attempts = 4


def test_wire_payloads_cannot_mutate_the_submission_snapshot():
    original = QueryExecutionOptions(RayExecution(DistributedMode.FTE, _fte()), 10, 60, 120)
    payload = original.to_dict()
    restored = QueryExecutionOptions.from_dict(payload)
    payload["target"]["fte_options"]["exchange_store"] = "another-store"
    restored.to_dict()["target"]["fte_options"]["max_attempts"] = 99

    assert restored == original


@pytest.mark.parametrize(
    "payload",
    [
        {"backend": "local-fast"},
        {"backend": "unknown"},
        {"backend": "local", "execution": None},
        {"backend": "ray", "execution": "fte", "fte_options": None},
        {"backend": "ray", "execution": "pipelined", "fte_options": _fte().to_dict()},
        {"backend": "ray", "execution": "auto", "fte_options": None},
        {"backend": "ray", "execution": "pipelined"},
        {"backend": "ray", "execution": "pipelined", "fte_options": None, "legacy": True},
    ],
)
def test_execution_contract_rejects_incomplete_unknown_or_incompatible_payloads(payload):
    with pytest.raises(ValueError):
        execution_from_dict(payload)


@pytest.mark.parametrize("value", [True, "30", -1, 0, float("nan"), float("inf"), 10**400])
@pytest.mark.parametrize("field", ["admission_timeout", "execution_timeout", "delivery_timeout"])
def test_timeouts_do_not_accept_values_that_can_disable_a_deadline(field, value):
    fields = {"target": LocalExecution(), "admission_timeout": 10, "execution_timeout": 60, "delivery_timeout": 120}
    fields[field] = value
    with pytest.raises(ValueError, match=field):
        QueryExecutionOptions(**fields)


@pytest.mark.parametrize("attempts", [True, 1.5, "3", 0, -1, 2**31])
def test_attempt_limit_is_an_exact_bounded_integer(attempts):
    with pytest.raises(ValueError, match="max_attempts"):
        FteOptions("store", attempts, 0)


def test_retry_configuration_requires_a_store_and_finite_nonnegative_backoff():
    with pytest.raises(ValueError, match="exchange_store"):
        FteOptions(" ", 3, 0)
    for backoff in (True, -0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="retry_backoff_seconds"):
            FteOptions("store", 3, backoff)
    assert FteOptions("store", 1, 0).retry_backoff_seconds == 0


def test_options_reject_unknown_or_missing_fields():
    payload = QueryExecutionOptions(LocalExecution(), 10, 60, 120).to_dict()
    payload["retry_policy"] = "task"
    with pytest.raises(ValueError, match="exactly these fields"):
        QueryExecutionOptions.from_dict(payload)
    del payload["retry_policy"]
    del payload["delivery_timeout"]
    with pytest.raises(ValueError, match="exactly these fields"):
        QueryExecutionOptions.from_dict(payload)
