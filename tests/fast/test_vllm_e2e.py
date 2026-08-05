# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import importlib
import os

import pytest
from ray_test_profile import ray_test_object_store_options

pytestmark = [pytest.mark.real_ray, pytest.mark.ray_cluster_owner, pytest.mark.gpu]


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
        **ray_test_object_store_options(),
    )

    import duckdb
    from vane.ai.providers.vllm import _build_native_vllm_options_argument

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

    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE prompts(id INTEGER, prompt VARCHAR)")
        con.execute(
            "INSERT INTO prompts VALUES "
            "(1, '中国的首都在哪里'), "
            "(2, '世界上最高的山峰是什么'), "
            "(3, 'DuckDB 是什么类型的数据库')"
        )

        packed_options = _build_native_vllm_options_argument(options)
        input_prompt = duckdb.FunctionExpression(
            "concat",
            duckdb.ConstantExpression("请用简洁中文回答："),
            duckdb.ColumnExpression("prompt"),
        )
        generated = duckdb.FunctionExpression(
            "vllm",
            input_prompt,
            duckdb.ConstantExpression(model),
            duckdb.ConstantExpression(packed_options),
        ).alias("out")
        relation = con.table("prompts").select(
            duckdb.ColumnExpression("id"),
            duckdb.ColumnExpression("prompt"),
            generated,
        )
        relation = relation.order("id")
        explain_text = relation.explain()
        _assert_explain_contains(explain_text, "VLLM")

        rows = relation.fetchall()
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
