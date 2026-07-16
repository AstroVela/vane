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
    with (path / "results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"case": case, **payload}, ensure_ascii=False, sort_keys=True) + "\n")


def _openai_provider_options() -> vane.ai.OpenAIProviderOptions:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required for OpenAI-compatible E2E")
    timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
    return vane.ai.OpenAIProviderOptions(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        organization=os.getenv("OPENAI_ORGANIZATION") or None,
        timeout=timeout,
        concurrency=1,
        max_api_concurrency=2,
    )


def test_openai_prompt_expression_real_provider() -> None:
    _require_real_e2e("VANE_E2E_OPENAI")
    pytest.importorskip("openai")

    model = os.getenv("VANE_E2E_OPENAI_PROMPT_MODEL", "gpt-4o-mini")
    use_chat = os.getenv("VANE_E2E_OPENAI_USE_CHAT_COMPLETIONS", "1").lower() not in {"0", "false", "no"}
    started = time.monotonic()

    conn = vane.connect()
    try:
        rel = conn.sql("select 1 as id, 'DuckDB analytical database' as chunk")
        answer_expr = vane.ai.prompt(
            vane.col("chunk"),
            provider="openai",
            model=model,
            provider_options=_openai_provider_options(),
            prompt_options=vane.ai.OpenAIPromptOptions(
                use_chat_completions=use_chat,
                max_tokens=32,
                max_output_tokens=32,
                temperature=0,
            ),
            system_message="Return one concise English phrase.",
        ).alias("answer")

        rows = rel.select(vane.col("id"), answer_expr).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    answer = rows[0][1]
    assert isinstance(answer, str)
    assert answer.strip()
    _record(
        "openai_prompt_expression",
        {
            "model": model,
            "rows": len(rows),
            "answer_preview": answer[:120],
            "duration_s": round(time.monotonic() - started, 3),
        },
    )


def test_openai_embed_expression_real_provider() -> None:
    _require_real_e2e("VANE_E2E_OPENAI")
    if not _enabled("VANE_E2E_OPENAI_EMBED"):
        pytest.skip("set VANE_E2E_OPENAI_EMBED=1 when the endpoint supports embeddings")
    pytest.importorskip("openai")

    model = os.getenv("VANE_E2E_OPENAI_EMBED_MODEL", "text-embedding-3-small")
    dimensions_env = os.getenv("VANE_E2E_OPENAI_EMBED_DIMENSIONS")
    dimensions = int(dimensions_env) if dimensions_env else None
    started = time.monotonic()

    conn = vane.connect()
    try:
        rel = conn.sql("select 1 as id, 'vector database retrieval' as chunk")
        embedding_expr = vane.ai.embed(
            vane.col("chunk"),
            provider="openai",
            model=model,
            provider_options=_openai_provider_options(),
            embedding_options=vane.ai.OpenAIEmbeddingOptions(encoding_format="float"),
            dimensions=dimensions,
            normalize=False,
        ).alias("embedding")

        rows = rel.select(vane.col("id"), embedding_expr).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    embedding = list(rows[0][1])
    assert embedding
    if dimensions is not None:
        assert len(embedding) == dimensions
    assert all(math.isfinite(float(value)) for value in embedding)
    _record(
        "openai_embed_expression",
        {
            "model": model,
            "dimensions": len(embedding),
            "duration_s": round(time.monotonic() - started, 3),
        },
    )


def test_vllm_prompt_expression_real_provider() -> None:
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
    started = time.monotonic()

    conn = vane.connect()
    try:
        rel = conn.sql("select 1 as id, 'What is DuckDB?' as chunk")
        answer_expr = vane.ai.prompt(
            vane.col("chunk"),
            provider="vllm",
            model=model,
            provider_options=vane.ai.VLLMProviderOptions(
                engine_args=engine_args,
                concurrency=1,
                gpus_per_actor=1,
            ),
            prompt_options=vane.ai.VLLMPromptOptions(
                generate_args={
                    "sampling_params": {
                        "temperature": 0,
                        "top_p": 1.0,
                        "max_tokens": max_tokens,
                    }
                }
            ),
            system_message="Answer briefly.",
        ).alias("answer")

        rows = rel.select(vane.col("id"), answer_expr).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    answer = rows[0][1]
    assert isinstance(answer, str)
    assert answer.strip()
    _record(
        "vllm_prompt_expression",
        {
            "model": model,
            "rows": len(rows),
            "answer_preview": answer[:120],
            "duration_s": round(time.monotonic() - started, 3),
        },
    )
