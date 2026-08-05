# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

SCHEMA = {
    "title": "Answer",
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class _RawBody:
    def __init__(self, body):
        self.body = body
        self.usage = None
        self.usage_metadata = None

    def model_dump(self, **kwargs):
        excluded = kwargs.get("exclude", set())
        return {key: value for key, value in self.body.items() if key not in excluded}


def _openai_prompter(*, chat: bool, raw: bool, strict: bool = True):
    from vane.ai.providers.openai import OpenAIPrompter

    prompter = OpenAIPrompter.__new__(OpenAIPrompter)
    prompter._provider_name = "openai"
    prompter._model = "gpt-5-mini"
    prompter._system_message = None
    prompter._use_chat_completions = chat
    prompter._return_format = SCHEMA
    prompter._return_raw_response = raw
    prompter._strict_structured_outputs = strict
    prompter._options = {}
    prompter._client = MagicMock()
    return prompter


def test_openai_responses_structured_and_raw_request_contracts():
    structured = _openai_prompter(chat=False, raw=False)
    structured._client.responses.create = AsyncMock(
        return_value=SimpleNamespace(output_text='{"answer":"ok"}', usage=None)
    )

    assert asyncio.run(structured.prompt(("question",))) == '{"answer":"ok"}'
    kwargs = structured._client.responses.create.await_args.kwargs
    assert kwargs["text"] == {
        "format": {
            "type": "json_schema",
            "name": "Answer",
            "schema": SCHEMA,
            "strict": True,
        }
    }

    raw = _openai_prompter(chat=False, raw=True)
    raw._client.responses.create = AsyncMock(return_value=_RawBody({"id": "response-1", "output": []}))
    assert json.loads(asyncio.run(raw.prompt(("question",)))) == {"id": "response-1", "output": []}
    assert raw._client.responses.create.await_args.kwargs["text"]["format"]["schema"] == SCHEMA


def test_openai_chat_completions_uses_json_schema_response_format():
    prompter = _openai_prompter(chat=True, raw=False)
    prompter._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok"}'))],
            usage=None,
        )
    )

    asyncio.run(prompter.prompt(("question",)))

    assert prompter._client.chat.completions.create.await_args.kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Answer", "schema": SCHEMA, "strict": True},
    }


@pytest.mark.parametrize("chat", [False, True])
def test_openai_compatible_endpoint_omits_strict_schema_flag(chat):
    prompter = _openai_prompter(chat=chat, raw=False, strict=False)
    if chat:
        prompter._client.chat.completions.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"answer":"ok"}'))],
                usage=None,
            )
        )
    else:
        prompter._client.responses.create = AsyncMock(
            return_value=SimpleNamespace(output_text='{"answer":"ok"}', usage=None)
        )

    asyncio.run(prompter.prompt(("question",)))

    if chat:
        schema = prompter._client.chat.completions.create.await_args.kwargs["response_format"]["json_schema"]
    else:
        schema = prompter._client.responses.create.await_args.kwargs["text"]["format"]
    assert "strict" not in schema


def test_anthropic_structured_tool_and_raw_body_contracts():
    from vane.ai.provider import _ProviderResultError
    from vane.ai.providers.anthropic import AnthropicPrompter

    prompter = AnthropicPrompter.__new__(AnthropicPrompter)
    prompter._provider_name = "anthropic"
    prompter._model = "claude-test"
    prompter._system_message = None
    prompter._return_format = SCHEMA
    prompter._return_raw_response = False
    prompter._options = {"max_tokens": 64}
    prompter._client = MagicMock()
    prompter._client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name="vane_structured_output", input={"answer": "ok"})],
            usage=None,
            stop_reason="tool_use",
        )
    )

    assert asyncio.run(prompter.prompt(("question",))) == {"answer": "ok"}
    kwargs = prompter._client.messages.create.await_args.kwargs
    assert kwargs["tools"] == [
        {
            "name": "vane_structured_output",
            "description": "Return the response in the requested structured format.",
            "input_schema": SCHEMA,
            "strict": True,
        }
    ]
    assert kwargs["tool_choice"] == {
        "type": "tool",
        "name": "vane_structured_output",
        "disable_parallel_tool_use": True,
    }

    prompter._client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name="vane_structured_output", input={"answer": "ok"})],
            usage=None,
            stop_reason="end_turn",
        )
    )
    with pytest.raises(
        _ProviderResultError,
        match=r"stop_reason 'end_turn'.*structured Prompt output.*'tool_use'",
    ):
        asyncio.run(prompter.prompt(("question",)))

    prompter._return_raw_response = True
    prompter._client.messages.create = AsyncMock(return_value=_RawBody({"id": "message-1", "content": []}))
    assert json.loads(asyncio.run(prompter.prompt(("question",)))) == {"id": "message-1", "content": []}


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("google.genai") is None,
    reason="google-genai not installed",
)
def test_google_structured_config_and_raw_body_contracts():
    from vane.ai.providers.google import GooglePrompter

    prompter = GooglePrompter.__new__(GooglePrompter)
    prompter._provider_name = "google"
    prompter._model = "gemini-test"
    prompter._system_message = None
    prompter._return_format = SCHEMA
    prompter._return_raw_response = False
    prompter._options = {}
    prompter._client = MagicMock()
    prompter._client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(text='{"answer":"ok"}', usage_metadata=None)
    )

    assert asyncio.run(prompter.prompt(("question",))) == '{"answer":"ok"}'
    config = prompter._client.aio.models.generate_content.await_args.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == SCHEMA

    prompter._return_raw_response = True
    prompter._client.aio.models.generate_content = AsyncMock(
        return_value=_RawBody(
            {
                "candidates": [],
                "sdk_http_response": {"headers": {"authorization": "secret"}},
                "parsed": {"answer": "SDK-only"},
            }
        )
    )
    assert json.loads(asyncio.run(prompter.prompt(("question",)))) == {"candidates": []}
