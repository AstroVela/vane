# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import builtins
import json
import pickle
import sys
import threading
import traceback
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pytest


@pytest.mark.parametrize("kind", ["text", "image", "audio", "video", "multimodal"])
def test_mlx_optional_registration_and_safe_dependency_hint(monkeypatch, tmp_path, kind):
    from vane.ai.provider import ProviderImportError, _safe_provider_execution_error, load_provider

    imported = []
    original_import = builtins.__import__

    def missing_sdk(name, *args, **kwargs):
        if name.split(".")[0] in {"mlx", "mlx_vlm", "transformers", "huggingface_hub", "PIL"}:
            imported.append(name)
            raise ModuleNotFoundError("optional SDK is absent", name=name.split(".")[0])
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_sdk)
    descriptor = load_provider("mlx").get_text_embedder()
    selected = pickle.loads(pickle.dumps(getattr(load_provider("mlx"), "get_" + kind + "_embedder")()))
    assert selected.get_dimensions() == 768 and selected.get_udf_options().num_gpus == 0
    assert ("input_type" in selected.get_options()) == (kind == "text")
    (output,) = selected.get_output_spec().outputs
    expected_sources = ("text", "image", "video", "audio") if kind in {"text", "multimodal"} else (kind,)
    assert output.source_modalities == expected_sources
    assert output.space == descriptor.get_output_spec().outputs[0].space
    service = pickle.loads(
        pickle.dumps(
            getattr(load_provider("mlx"), "get_" + kind + "_embedder")(
                options={
                    "runtime": "service",
                    "service_path": str(tmp_path / "mlx.json"),
                    "service_id": "metadata-only",
                }
            )
        )
    )
    assert service.get_output_spec() == selected.get_output_spec()
    assert service.get_udf_options().num_gpus == 0 and service.is_async()
    assert service.instantiate()._zmq_client is None
    assert imported == []
    monkeypatch.setattr("vane.ai.providers.mlx.platform.system", lambda: "Darwin")
    monkeypatch.setattr("vane.ai.providers.mlx.platform.machine", lambda: "arm64")
    with pytest.raises(ProviderImportError, match=r"vane-ai\[mlx\]"):
        selected.instantiate()
    assert imported == ["mlx.core"]
    error = pickle.loads(pickle.dumps(ProviderImportError("mlx")))
    safe = _safe_provider_execution_error("mlx", descriptor.get_model(), "Embed", error)
    assert isinstance(safe, ProviderImportError)
    assert "vane-ai[mlx]" in str(safe)


def test_jev_optional_sdk_is_loaded_only_when_preparing_questions(monkeypatch):
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)

    import vane
    from vane.ai import JevOptions, jev
    from vane.ai.provider import ProviderImportError

    assert callable(jev)
    assert hasattr(vane.Relation, "jev")
    assert "max_concurrency_per_actor" in JevOptions.__annotations__
    with pytest.raises(ProviderImportError, match=r"vane-ai\[typesafe\]"):
        jev(vane.col("text"), questions={"billing": {"type": "noul"}})
    with vane.connect() as connection:
        with pytest.raises(ProviderImportError, match=r"vane-ai\[typesafe\]"):
            connection.sql('SELECT ai_jev(\'hello\', questions := \'{"billing": {"type": "noul"}}\')')


def _drive(wrapper, table: pa.Table) -> pa.Table:
    loop = asyncio.new_event_loop()
    wrapper.bind_async_runtime(loop.run_until_complete)
    try:
        return wrapper(table)
    finally:
        loop.close()


@pytest.mark.parametrize("entry_point", ["relation", "expression"])
@pytest.mark.parametrize("chat_completions", [True, False])
def test_prompt_image_url_preserves_part_order_and_provider_format(monkeypatch, entry_point, chat_completions):
    import vane
    from vane.ai import image_url, prompt
    from vane.ai.providers.openai import OpenAIPrompter, OpenAIPrompterDescriptor, OpenAIProvider

    class RecordingPrompter(OpenAIPrompter):
        def __init__(self, use_chat_completions):
            self._use_chat_completions = use_chat_completions
            self._system_message = None

        async def _prompt_chat_completions(self, messages):
            return json.dumps(messages[0]["content"])

        _prompt_responses = _prompt_chat_completions

        async def aclose(self):
            pass

    class RecordingDescriptor(OpenAIPrompterDescriptor):
        def instantiate(self):
            return RecordingPrompter(self.options["use_chat_completions"])

    origin = "https://frames.example.test"
    monkeypatch.setenv("VANE_AI_IMAGE_URL_ORIGIN", origin)
    text = f"Look at the image and see {origin}/help for context."
    url = f"{origin}/frame.jpg?signature=fixture"
    provider = OpenAIProvider()
    monkeypatch.setattr(provider, "get_prompter", lambda **kwargs: RecordingDescriptor(options=kwargs["options"]))

    with vane.connect() as con:
        relation = con.sql("SELECT ?::VARCHAR AS text, ?::VARCHAR AS url", params=[text, url])
        parts = [vane.col("text"), image_url(vane.col("url"))]
        options = {"provider": provider, "use_chat_completions": chat_completions}
        if entry_point == "relation":
            query = prompt(relation, parts, **options).project("response")
        else:
            query = relation.select(prompt(parts, **options))
        (response,) = query.fetchone()

    expected = (
        [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}]
        if chat_completions
        else [{"type": "input_text", "text": text}, {"type": "input_image", "image_url": url}]
    )
    assert json.loads(response) == expected


@pytest.mark.parametrize("origin", [None, "https://frames.example.test/api", "file://frames.example.test"])
def test_prompt_image_url_invalid_worker_configuration_fails_with_ignore(monkeypatch, origin):
    from vane.ai.functions import _PromptBatch

    if origin is None:
        monkeypatch.delenv("VANE_AI_IMAGE_URL_ORIGIN", raising=False)
    else:
        monkeypatch.setenv("VANE_AI_IMAGE_URL_ORIGIN", origin)
    wrapper = _PromptBatch(object(), ["url"], "response", on_error="ignore", image_url_indices=frozenset({0}))

    with pytest.raises(ValueError, match="VANE_AI_IMAGE_URL_ORIGIN"):
        _drive(wrapper, pa.table({"url": ["https://frames.example.test/frame.jpg"]}))


@pytest.mark.parametrize(
    "url",
    [
        "https://other.example.test/frame.jpg",
        "http://frames.example.test/frame.jpg",
        "https://user@frames.example.test/frame.jpg",
        "https://frames.example.test/frame.jpg#fragment",
        "https://frames.example.test:invalid/frame.jpg",
        "/frame.jpg",
        "file:///frame.jpg",
        "ftp://frames.example.test/frame.jpg",
    ],
)
def test_prompt_image_url_rejects_invalid_row_url(monkeypatch, url):
    from vane.ai.functions import _PromptBatch

    monkeypatch.setenv("VANE_AI_IMAGE_URL_ORIGIN", "https://frames.example.test")
    wrapper = _PromptBatch(object(), ["url"], "response", image_url_indices=frozenset({0}))

    with pytest.raises(ValueError, match="Prompt image URL"):
        _drive(wrapper, pa.table({"url": [url]}))


@pytest.mark.parametrize("entry_point", ["relation", "expression"])
def test_prompt_image_url_requires_varchar(entry_point):
    import vane
    from vane.ai import image_url, prompt

    with vane.connect() as con:
        relation = con.sql("SELECT 'not a URL'::BLOB AS image")
        message = image_url(vane.col("image"))
        with pytest.raises((TypeError, vane.BinderException), match="VARCHAR"):
            if entry_point == "relation":
                prompt(relation, message)
            else:
                relation.select(prompt(message))


def test_prompt_image_url_rejects_unsupported_provider():
    import vane
    from vane.ai import image_url, prompt

    with pytest.raises(ValueError, match="OpenAI-compatible provider with image support"):
        prompt(image_url(vane.col("image")), provider="anthropic", model="claude-test", max_tokens=64)


def test_prompt_batch_unpickles_state_created_before_image_url_support():
    from vane.ai.functions import _PromptBatch

    wrapper = _PromptBatch(object(), ["text"], "response")
    del wrapper._image_url_indices
    restored = pickle.loads(pickle.dumps(wrapper))
    table = pa.table({"text": pa.array([None], type=pa.string())})

    assert _drive(restored, table).to_pylist() == [{"response": None}]


def _assert_only_cancellation_context(error: asyncio.CancelledError) -> None:
    context = error.__context__
    while context is not None:
        assert isinstance(context, asyncio.CancelledError)
        context = context.__context__


def _install_fake_google(monkeypatch, calls: list[dict[str, object]]) -> None:
    class HttpRetryOptions:
        def __init__(self, *, attempts):
            self.attempts = attempts

    class HttpOptions:
        def __init__(self, *, retry_options, base_url):
            self.retry_options = retry_options
            self.base_url = base_url

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
    assert metadata_only.get_dimensions() == 4
    assert metadata_embedder._dimensions is None
    assert explicit_override.get_dimensions() == 3
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
        provider.get_prompter(
            model="claude-test",
            return_format=return_format,
            options={"max_tokens": 0},
        )
    assert provider.get_prompter(model="claude-test", options={"max_tokens": 0}).options["max_tokens"] == 0


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
    from vane.ai.functions import _EmbedTextBatch, _PromptBatch
    from vane.ai.providers.openai import OpenAITextEmbedder
    from vane.execution.udf_ray_stream_protocol import make_stream_error_pair
    from vane.execution.udf_subprocess_worker import _format_exception

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
    embedder._estimate_tokens = lambda value: len(value.encode("utf-8"))

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
                "resource_unit_id": "resource:query:node:1:udf",
                "task_lease_id": "lease",
                "attempt_id": "attempt",
            },
            error,
        )
        surfaces = (str(error), traceback_text, _format_exception(error), repr(ray_metadata))
        assert all(secret not in surface for surface in surfaces)


def test_custom_capability_error_chain_is_rebuilt_before_execution_wires():
    from vane.ai.functions import _PromptBatch
    from vane.ai.provider import ProviderCapabilityError
    from vane.execution.udf_subprocess_worker import _format_exception

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
    _assert_only_cancellation_context(error)
    surfaces = ("".join(traceback.format_exception(type(error), error, error.__traceback__)),)
    assert all(secret not in surface for surface in surfaces)


def test_transient_provider_error_is_not_attached_to_backoff_cancellation(monkeypatch):
    from vane.ai.functions import _retry_call_async

    secret = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

    class ServiceUnavailableError(RuntimeError):
        status_code = 503

    async def fail():
        raise ServiceUnavailableError(secret)

    async def cancel_sleep(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", cancel_sleep)

    with pytest.raises(asyncio.CancelledError) as exc_info:
        asyncio.run(_retry_call_async(fail, max_retries=1))

    error = exc_info.value
    _assert_only_cancellation_context(error)
    traceback_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert secret not in traceback_text


def test_vllm_engine_initialization_error_is_credential_safe(monkeypatch):
    from vane.execution.vllm import LocalVLLMExecutor

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


pytestmark = pytest.mark.usefixtures("application_provider_credentials")
