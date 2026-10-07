# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from vane import configure, current_config, env


def test_configure_sets_registered_environment_variables(monkeypatch):
    monkeypatch.delenv("VANE_UDF_PARALLEL", raising=False)
    monkeypatch.delenv("VANE_NDJSON_MAX_SPLIT_BYTES", raising=False)
    cfg = configure(udf_parallel=True, ndjson_max_split_bytes=128 * 1024 * 1024)
    assert cfg.udf_parallel and env.udf_parallel and current_config().udf_parallel
    assert os.environ["VANE_NDJSON_MAX_SPLIT_BYTES"] == "134217728"
    assert current_config().ndjson_max_split_bytes == cfg.ndjson_max_split_bytes


@pytest.mark.parametrize(
    "field", ["runner", "ray_init_sql", "ray_max_task_backlog", "ray_scan_split_min_count", "unknown_option"]
)
def test_execution_configuration_is_not_process_global(field):
    with pytest.raises(AttributeError, match="Unknown config field"):
        configure(**{field: "ray"})
    assert field not in current_config().__dict__
