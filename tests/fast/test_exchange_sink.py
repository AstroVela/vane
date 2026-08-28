# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

from vane.runners.exchange_sink import bind_exchange_sink_instance, normalize_exchange_sink_config


def test_bind_scheduler_task_identity_exchange_sink_attempt():
    config = {
        "query_id": "query-a",
        "output_location_prefix": "exchange-a",
        "output_partition_count": 8,
    }

    instance = bind_exchange_sink_instance(config, attempt_id=2, task_partition_id=9)
    retry = bind_exchange_sink_instance(config, attempt_id=3, task_partition_id=9)

    assert instance == {
        "sink_handle": {"task_partition_id": 9},
        "attempt_id": 2,
        "query_id": "query-a",
        "output_partition_count": 8,
        "output_location": "exchange-a__sink_9__attempt_2",
    }
    assert retry["sink_handle"]["task_partition_id"] == 9
    assert retry["output_location"] == "exchange-a__sink_9__attempt_3"


def test_bind_ordered_exchange_sink_carries_independent_source_order():
    config = {
        "query_id": "query-ordered",
        "output_location_prefix": "exchange-ordered",
        "output_partition_count": 1,
        "preserve_order": True,
    }

    instance = bind_exchange_sink_instance(
        config,
        attempt_id=2,
        task_partition_id=991,
        source_task_order=4,
    )

    assert instance["sink_handle"]["task_partition_id"] == 991
    assert instance["source_task_order"] == 4
    assert instance["output_location"] == "exchange-ordered__sink_991__attempt_2"


def test_ordered_exchange_sink_requires_source_order():
    with pytest.raises(TypeError, match="source_task_order must be an integer"):
        bind_exchange_sink_instance(
            {
                "query_id": "query-ordered",
                "output_location_prefix": "exchange-ordered",
                "output_partition_count": 1,
                "preserve_order": True,
            },
            attempt_id=0,
            task_partition_id=1,
        )


def test_exchange_sink_config_rejects_non_boolean_preserve_order():
    with pytest.raises(TypeError, match="preserve_order must be a boolean"):
        normalize_exchange_sink_config(
            {
                "query_id": "query-ordered",
                "output_location_prefix": "exchange-ordered",
                "output_partition_count": 1,
                "preserve_order": 1,
            }
        )


@pytest.mark.parametrize("field", ["identity_source", "plan_task_partition_id"])
def test_exchange_sink_config_rejects_plan_owned_identity_fields(field):
    with pytest.raises(ValueError, match=f"unexpected fields.*{field}"):
        normalize_exchange_sink_config(
            {
                "query_id": "query-c",
                "output_location_prefix": "exchange-c",
                "output_partition_count": 1,
                field: "plan" if field == "identity_source" else 1,
            },
        )


def test_exchange_sink_binding_rejects_non_integer_scheduler_task_id():
    with pytest.raises(TypeError, match="task_partition_id must be an integer"):
        bind_exchange_sink_instance(
            {
                "query_id": "query-d",
                "output_location_prefix": "exchange-d",
                "output_partition_count": 1,
            },
            attempt_id=0,
            task_partition_id=True,
        )


def test_exchange_sink_config_rejects_legacy_instance_fields():
    with pytest.raises(ValueError, match="unexpected fields.*fte_task_identity"):
        normalize_exchange_sink_config(
            {
                "query_id": "query-e",
                "output_location_prefix": "exchange-e",
                "output_partition_count": 1,
                "fte_task_identity": True,
            }
        )
