# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from vane.runners.exchange_sink import bind_exchange_sink_instance, normalize_exchange_sink_config


def test_bind_task_identity_exchange_sink_attempt():
    config = {
        "identity_source": "task",
        "query_id": "query-a",
        "output_location_prefix": "exchange-a",
        "output_partition_count": 8,
    }

    instance = bind_exchange_sink_instance(config, attempt_id=2, task_partition_id=9)
    retry = bind_exchange_sink_instance(config, attempt_id=3, task_partition_id=9)

    assert instance == {
        "sink_handle": {"task_partition_id": 9},
        "task_partition_id": 9,
        "attempt_id": 2,
        "query_id": "query-a",
        "output_partition_count": 8,
        "output_location": "exchange-a__sink_9__attempt_2",
    }
    assert retry["task_partition_id"] == 9
    assert retry["output_location"] == "exchange-a__sink_9__attempt_3"


def test_bind_plan_identity_exchange_sink_attempt_preserves_plan_id():
    config = {
        "identity_source": "plan",
        "plan_task_partition_id": 5,
        "query_id": "query-b",
        "output_location_prefix": "exchange-b",
        "output_partition_count": 4,
    }

    first = bind_exchange_sink_instance(config, attempt_id=0)
    retry = bind_exchange_sink_instance(config, attempt_id=3)

    assert first["task_partition_id"] == retry["task_partition_id"] == 5
    assert first["output_location"] == "exchange-b__sink_5__attempt_0"
    assert retry["output_location"] == "exchange-b__sink_5__attempt_3"


def test_exchange_sink_config_rejects_mixed_identity_sources():
    for plan_task_partition_id in (1, None):
        with pytest.raises(ValueError, match="cannot carry plan_task_partition_id"):
            normalize_exchange_sink_config(
                {
                    "identity_source": "task",
                    "plan_task_partition_id": plan_task_partition_id,
                    "query_id": "query-c",
                    "output_location_prefix": "exchange-c",
                    "output_partition_count": 1,
                }
            )

    with pytest.raises(ValueError, match="cannot override task_partition_id"):
        bind_exchange_sink_instance(
            {
                "identity_source": "plan",
                "plan_task_partition_id": 1,
                "query_id": "query-c",
                "output_location_prefix": "exchange-c",
                "output_partition_count": 1,
            },
            attempt_id=0,
            task_partition_id=2,
        )


def test_task_identity_exchange_sink_requires_runtime_task_id():
    with pytest.raises(ValueError, match="requires task_partition_id"):
        bind_exchange_sink_instance(
            {
                "identity_source": "task",
                "query_id": "query-d",
                "output_location_prefix": "exchange-d",
                "output_partition_count": 1,
            },
            attempt_id=0,
        )


def test_exchange_sink_config_rejects_legacy_instance_fields():
    with pytest.raises(ValueError, match="unexpected fields.*fte_task_identity"):
        normalize_exchange_sink_config(
            {
                "identity_source": "task",
                "query_id": "query-e",
                "output_location_prefix": "exchange-e",
                "output_partition_count": 1,
                "fte_task_identity": True,
            }
        )
