# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import os

import pytest
from ray_test_profile import ray_test_object_store_bytes

pytestmark = [pytest.mark.real_ray, pytest.mark.ray_cluster_owner]


def _explain_text(con, sql):
    explain_rows = con.sql("EXPLAIN " + sql).fetchall()
    return "\n".join(
        row[1] if isinstance(row, tuple) and len(row) > 1 else row[0] if isinstance(row, tuple) else str(row)
        for row in explain_rows
    )


def _assert_explain_contains(explain_text, keyword):
    assert keyword in explain_text.upper(), f"expected {keyword} in EXPLAIN"


def test_vllm_e2e_basic():
    pytest.importorskip("pyarrow")

    try:
        import ray
    except Exception as exc:
        pytest.skip(f"ray unavailable: {exc}")
    try:
        importlib.import_module("vllm")
    except Exception as exc:
        pytest.skip(f"vllm unavailable: {exc}")
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"torch unavailable: {exc}")

    if not torch.cuda.is_available():
        pytest.skip("vllm requires CUDA")

    ray.init(
        address="local",
        include_dashboard=False,
        object_store_memory=ray_test_object_store_bytes(),
    )

    import vane

    timeout_s = int(os.getenv("VLLM_E2E_TIMEOUT", "300"))
    try:
        import signal

        signal.alarm(timeout_s)
    except Exception:
        pass

    model = os.getenv("VLLM_E2E_MODEL", "Qwen/Qwen3-1.7B")
    options = {
        "use_ray": True,
        "concurrency": 1,
        "gpus_per_actor": 1,
        "do_prefix_routing": False,
        "engine_args": {
            "trust_remote_code": True,
            "max_model_len": 512,
        },
        "generate_args": {
            "sampling_params": {
                "temperature": 0.8,
                "top_p": 0.95,
                "max_tokens": 100,
            }
        },
    }

    con = vane.connect()
    try:
        con.execute("CREATE TABLE prompts(id INTEGER, prompt VARCHAR)")
        con.execute(
            "INSERT INTO prompts VALUES "
            "(1, '中国的首都在哪里'), "
            "(2, '世界上最高的山峰是什么'), "
            "(3, 'DuckDB 是什么类型的数据库')"
        )

        sql = (
            "SELECT id, prompt, "
            "vllm('请用简洁中文回答：' || prompt, '" + model + "', '" + json.dumps(options) + "') AS out "
            "FROM prompts ORDER BY id"
        )
        explain_text = _explain_text(con, sql)
        _assert_explain_contains(explain_text, "VLLM")

        rows = con.execute(sql).fetchall()
        assert len(rows) == 3
        for row in rows:
            print(f"vllm_e2e: id={row[0]} prompt={row[1]!r} output={row[2]!r}")
        assert all(isinstance(row[2], str) and row[2] for row in rows)
    finally:
        con.close()
        try:
            ray.shutdown()
        except Exception:
            pass
