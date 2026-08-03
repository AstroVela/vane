# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import pickle
import sys
import threading
import traceback
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pytest


def _drive(wrapper, table: pa.Table) -> pa.Table:
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        return wrapper(table)
    finally:
        loop.close()


def _install_fake_google(monkeypatch, calls: list[dict[str, object]]) -> None:
    class HttpRetryOptions:
        def __init__(self, *, attempts):
            self.attempts = attempts

    class HttpOptions:
        def __init__(self, *, retry_options):
            self.retry_options = retry_options

    def client(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace()

    fake_genai = ModuleType("google.genai")
    fake_genai.Client = client
    fake_genai.types = SimpleNamespace(HttpOptions=HttpOptions, HttpRetryOptions=HttpRetryOptions)
    fake_google = ModuleType("google")
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)


def test_google_embedding_metadata_and_retry_contract_without_optional_sdk(monkeypatch):
    from vane.ai.providers.google import GoogleProvider

    calls: list[dict[str, object]] = []
    _install_fake_google(monkeypatch, calls)
    provider = GoogleProvider(embedding_model="custom-fixed-model", embedding_dimensions=4)

    metadata_only = pickle.loads(pickle.dumps(provider.get_text_embedder()))
    explicit_override = pickle.loads(pickle.dumps(provider.get_text_embedder(dimensions=3)))

    metadata_embedder = metadata_only.instantiate()
    explicit_embedder = explicit_override.instantiate()
    assert metadata_only.get_dimensions().size == 4
    assert metadata_embedder._dimensions is None
    assert explicit_override.get_dimensions().size == 3
    assert explicit_embedder._dimensions == 3
    assert [call["http_options"].retry_options.attempts for call in calls] == [1, 1]


def test_anthropic_zero_token_structured_contract_without_optional_sdk():
    from vane.ai._schema import compile_return_format
    from vane.ai.providers.anthropic import AnthropicProvider

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    provider = AnthropicProvider()
    return_format = compile_return_format(schema)

    with pytest.raises(ValueError, match="max_tokens=0.*structured"):
        provider.get_prompter(model="claude-test", return_format=return_format, max_tokens=0)
    assert provider.get_prompter(model="claude-test", max_tokens=0).prompt_options["max_tokens"] == 0


def test_non_pydantic_schema_class_is_rejected_without_optional_dependency(monkeypatch):
    from vane.ai._schema import compile_return_format

    class BaseModel:
        pass

    fake_pydantic = ModuleType("pydantic")
    fake_pydantic.BaseModel = BaseModel
    monkeypatch.setitem(sys.modules, "pydantic", fake_pydantic)

    class FakeModel:
        @classmethod
        def model_json_schema(cls):
            return {"type": "object", "properties": {}, "additionalProperties": False}

    with pytest.raises(TypeError, match="Pydantic BaseModel subclass"):
        compile_return_format(FakeModel)


def test_non_capability_provider_failures_are_safe_before_execution_wires(monkeypatch):
    from duckdb.execution.udf_ray_stream_protocol import make_stream_error_pair
    from duckdb.execution.udf_subprocess_worker import _format_exception
    from vane.ai.functions import _EmbedTextBatch, _PromptBatch
    from vane.ai.providers.openai import OpenAITextEmbedder

    secret = "AIzaSyD4n0m5M_NTpvI_GlTgQeX82aBcDeFgHi"

    class AuthenticationError(Exception):
        status_code = 401

    original = AuthenticationError(f"Authorization Bearer {secret}")
    fake_openai = ModuleType("openai")
    fake_openai.OpenAIError = AuthenticationError
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    embedder = OpenAITextEmbedder.__new__(OpenAITextEmbedder)
    embedder._client = MagicMock()
    embedder._client.embeddings.create = AsyncMock(side_effect=original)
    embedder._provider_name = "openai-compatible"
    embedder._model = "embedding-model"
    embedder._dimensions = 4
    embedder._encoding_format = "float"
    embedder._batch_token_limit = 100
    embedder._input_text_token_limit = 100

    class EmbedDescriptor:
        def get_provider(self):
            return "openai-compatible"

        def get_model(self):
            return "embedding-model"

        def instantiate(self):
            return embedder

    class Prompter:
        async def prompt(self, _messages):
            raise original

    class PromptDescriptor:
        def get_provider(self):
            return "openai-compatible"

        def get_model(self):
            return "prompt-model"

        def instantiate(self):
            return Prompter()

    embed_wrapper = _EmbedTextBatch(
        EmbedDescriptor(),
        "text",
        "embedding",
        4,
        max_retries=0,
        on_error="raise",
    )
    prompt_wrapper = _PromptBatch(
        PromptDescriptor(),
        ["message"],
        "answer",
        max_retries=0,
        on_error="raise",
    )

    errors: list[RuntimeError] = []
    for wrapper, table in (
        (embed_wrapper, pa.table({"text": ["hello"]})),
        (prompt_wrapper, pa.table({"message": ["hello"]})),
    ):
        with pytest.raises(RuntimeError) as exc_info:
            _drive(wrapper, table)
        errors.append(exc_info.value)

    for error in errors:
        assert "AuthenticationError (status_code=401)" in str(error)
        assert error.__cause__ is None
        assert error.__context__ is None
        traceback_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        _, ray_metadata = make_stream_error_pair(
            {
                "query_id": "query",
                "stage_id": "stage",
                "task_lease_id": "lease",
                "attempt_id": "attempt",
            },
            error,
        )
        surfaces = (str(error), traceback_text, _format_exception(error), repr(ray_metadata))
        assert all(secret not in surface for surface in surfaces)


def test_custom_capability_error_chain_is_rebuilt_before_execution_wires():
    from duckdb.execution.udf_subprocess_worker import _format_exception
    from vane.ai.functions import _PromptBatch
    from vane.ai.provider import ProviderCapabilityError

    secret = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    class Prompter:
        async def prompt(self, _messages):
            try:
                raise RuntimeError(secret)
            except RuntimeError as exc:
                raise ProviderCapabilityError(
                    "custom",
                    "model",
                    "structured output",
                    original_error=exc,
                ) from exc

    class Descriptor:
        def get_provider(self):
            return "custom"

        def get_model(self):
            return "model"

        def instantiate(self):
            return Prompter()

    wrapper = _PromptBatch(
        Descriptor(),
        ["message"],
        "answer",
        max_retries=0,
        on_error="raise",
    )
    with pytest.raises(ProviderCapabilityError) as exc_info:
        _drive(wrapper, pa.table({"message": ["hello"]}))

    error = exc_info.value
    assert error.original_error_summary == "RuntimeError"
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert secret not in traceback_text
    assert secret not in _format_exception(error)


def test_retry_after_cancellation_retains_no_original_exception_data():
    from vane.ai.functions import RetryAfterError, _retry_call_async

    secret = "AIzaSyD4n0m5M_NTpvI_GlTgQeX82aBcDeFgHi"

    async def fail_with_retry_after():
        raise RetryAfterError(60, RuntimeError(secret))

    async def cancel_during_backoff():
        task = asyncio.create_task(_retry_call_async(fail_with_retry_after, max_retries=1))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        return exc_info.value

    error = asyncio.run(cancel_during_backoff())
    assert error.__context__ is None
    surfaces = ("".join(traceback.format_exception(type(error), error, error.__traceback__)),)
    assert all(secret not in surface for surface in surfaces)


def test_ordinary_provider_error_is_not_attached_to_backoff_cancellation(monkeypatch):
    from vane.ai.functions import _retry_call_async

    secret = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    async def fail():
        raise RuntimeError(secret)

    async def cancel_sleep(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", cancel_sleep)

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(_retry_call_async(fail, max_retries=1))

    error = exc_info.value
    assert error.__context__ is None
    traceback_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert secret not in traceback_text


def test_vllm_engine_initialization_error_is_credential_safe(monkeypatch):
    from duckdb.execution.vllm import LocalVLLMExecutor

    secret = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    class AuthenticationError(Exception):
        status_code = 401

    class AsyncEngineArgs:
        def __init__(self, **_kwargs):
            raise AuthenticationError(f"token {secret}")

    fake_vllm = ModuleType("vllm")
    fake_vllm.AsyncEngineArgs = AsyncEngineArgs
    fake_vllm.AsyncLLMEngine = object
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    executor = LocalVLLMExecutor.__new__(LocalVLLMExecutor)
    executor.model = "local-model"
    executor.engine_args = {}
    executor.on_error = "raise"
    executor.error_lock = threading.Lock()
    executor.error_message = None
    executor.engine_error_message = None
    executor.engine_ready = threading.Event()
    executor._init_engine_sync()

    assert executor.engine_ready.is_set()
    assert "AuthenticationError (status_code=401)" in executor.engine_error_message
    assert secret not in executor.engine_error_message
    assert secret not in executor.error_message
