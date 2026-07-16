from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import vane

if TYPE_CHECKING:
    from collections.abc import Mapping


def _enabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _require_real_e2e(provider_flag: str) -> None:
    if not _enabled("VANE_REAL_PROVIDER_E2E"):
        pytest.skip("set VANE_REAL_PROVIDER_E2E=1 to run real provider E2E")
    if not _enabled(provider_flag):
        pytest.skip(f"set {provider_flag}=1 to run this provider E2E")


def _record(case: str, payload: Mapping[str, object]) -> None:
    artifact_dir = os.getenv("VANE_E2E_ARTIFACT_DIR")
    if not artifact_dir:
        return
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "sql_results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"case": case, **payload}, ensure_ascii=False, sort_keys=True) + "\n")


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def test_openai_prompt_sql_real_provider() -> None:
    _require_real_e2e("VANE_E2E_OPENAI")
    pytest.importorskip("openai")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for OpenAI-compatible E2E")

    model = os.getenv("VANE_E2E_OPENAI_PROMPT_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL")
    timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
    started = time.monotonic()

    conn = vane.connect()
    rows = conn.sql(f"""
        SELECT ai_prompt(
            chunk,
            struct_pack(
                provider := 'openai',
                model := {_sql_string(model)},
                base_url := {'NULL' if not base_url else _sql_string(base_url)},
                timeout := {timeout},
                concurrency := 1,
                max_api_concurrency := 2,
                max_tokens := 32,
                max_output_tokens := 32,
                temperature := 0,
                system_message := 'Return one concise English phrase.'
            )
        ) AS answer
        FROM (SELECT 'DuckDB analytical database' AS chunk)
    """).fetchall()

    assert len(rows) == 1
    answer = rows[0][0]
    assert isinstance(answer, str)
    assert answer.strip()
    _record(
        "openai_prompt_sql",
        {
            "model": model,
            "base_url_set": bool(base_url),
            "answer_preview": answer[:120],
            "duration_s": round(time.monotonic() - started, 3),
        },
    )


def test_openai_embed_sql_real_provider() -> None:
    _require_real_e2e("VANE_E2E_OPENAI")
    if not _enabled("VANE_E2E_OPENAI_EMBED"):
        pytest.skip("set VANE_E2E_OPENAI_EMBED=1 when endpoint supports embeddings")
    pytest.importorskip("openai")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for OpenAI-compatible E2E")

    model = os.getenv("VANE_E2E_OPENAI_EMBED_MODEL", "text-embedding-3-small")
    dimensions_env = os.getenv("VANE_E2E_OPENAI_EMBED_DIMENSIONS")
    dimensions_sql = "NULL" if not dimensions_env else str(int(dimensions_env))
    base_url = os.getenv("OPENAI_BASE_URL")
    started = time.monotonic()

    conn = vane.connect()
    rows = conn.sql(f"""
        SELECT ai_embed(
            chunk,
            struct_pack(
                provider := 'openai',
                model := {_sql_string(model)},
                base_url := {'NULL' if not base_url else _sql_string(base_url)},
                dimensions := {dimensions_sql},
                encoding_format := 'float',
                normalize := false,
                concurrency := 1
            )
        ) AS embedding
        FROM (SELECT 'vector database retrieval' AS chunk)
    """).fetchall()

    assert len(rows) == 1
    embedding = list(rows[0][0])
    assert embedding
    if dimensions_env:
        assert len(embedding) == int(dimensions_env)
    assert all(math.isfinite(float(value)) for value in embedding)
    _record(
        "openai_embed_sql",
        {
            "model": model,
            "dimensions": len(embedding),
            "base_url_set": bool(base_url),
            "duration_s": round(time.monotonic() - started, 3),
        },
    )


def test_vllm_prompt_sql_real_provider() -> None:
    _require_real_e2e("VANE_E2E_VLLM")
    pytest.importorskip("vllm")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("vLLM E2E requires CUDA")

    model = os.getenv("VANE_E2E_VLLM_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
    max_model_len = int(os.getenv("VANE_E2E_VLLM_MAX_MODEL_LEN", "512"))
    max_tokens = int(os.getenv("VANE_E2E_VLLM_MAX_TOKENS", "16"))
    gpu_memory_utilization = os.getenv("VANE_E2E_VLLM_GPU_MEMORY_UTILIZATION")
    engine_args = {
        "trust_remote_code": True,
        "max_model_len": max_model_len,
    }
    if gpu_memory_utilization:
        engine_args["gpu_memory_utilization"] = float(gpu_memory_utilization)
    generate_args = {
        "sampling_params": {
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": max_tokens,
        }
    }
    started = time.monotonic()

    conn = vane.connect()
    rows = conn.sql(f"""
        SELECT ai_prompt(
            chunk,
            struct_pack(
                provider := 'vllm',
                model := {_sql_string(model)},
                concurrency := 1,
                gpus_per_actor := 1,
                engine_args_json := {_sql_string(json.dumps(engine_args))},
                generate_args_json := {_sql_string(json.dumps(generate_args))},
                system_message := 'Answer briefly.'
            )
        ) AS answer
        FROM (SELECT 'What is DuckDB?' AS chunk)
    """).fetchall()

    assert len(rows) == 1
    answer = rows[0][0]
    assert isinstance(answer, str)
    assert answer.strip()
    _record(
        "vllm_prompt_sql",
        {
            "model": model,
            "answer_preview": answer[:120],
            "duration_s": round(time.monotonic() - started, 3),
        },
    )
