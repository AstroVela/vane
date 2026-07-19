# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"batch_size": 0}, "batch_size"),
        ({"batch_size": True}, "batch_size"),
        ({"batch_size": 1.5}, "batch_size"),
        ({"prefix_match_threshold": float("nan")}, "prefix_match_threshold"),
        ({"prefix_match_threshold": float("inf")}, "prefix_match_threshold"),
        ({"prefix_match_threshold": True}, "prefix_match_threshold"),
        ({"gpus_per_actor": 0}, "gpus_per_actor"),
        ({"gpus_per_actor": 1.5}, "gpus_per_actor"),
        ({"gpus_per_actor": True}, "gpus_per_actor"),
        ({"concurrency": True}, "concurrency"),
        ({"do_prefix_routing": "false"}, "do_prefix_routing"),
    ],
)
def test_vllm_numeric_options_are_strict(options, message):
    from duckdb.execution.vllm import normalize_options

    with pytest.raises(ValueError, match=message):
        normalize_options(options)


def test_vllm_fractional_gpu_option_is_preserved():
    from duckdb.execution.vllm import normalize_options

    assert normalize_options({"gpus_per_actor": 0.25})["gpus_per_actor"] == pytest.approx(0.25)
