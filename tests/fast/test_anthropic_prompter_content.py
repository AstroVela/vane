# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""AnthropicPrompter terminal-state validation and plain-text extraction.

Extended-thinking models return a leading ``thinking`` block (which has
``.thinking`` and no ``.text``), so ``prompt`` must concatenate the
``text`` blocks rather than assuming ``content[0]`` is text.
"""

import asyncio
import sys
from types import SimpleNamespace

import pytest


def _make_prompter(monkeypatch, content, *, stop_reason="end_turn", max_tokens=64):
    """Build an AnthropicPrompter whose SDK client returns ``content``."""

    class FakeMessages:
        async def create(self, **kwargs):
            return SimpleNamespace(content=content, usage=None, stop_reason=stop_reason)

    class FakeAsyncAnthropic:
        def __init__(self, **options):
            self.messages = FakeMessages()

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(AsyncAnthropic=FakeAsyncAnthropic),
    )

    from vane.ai.providers.anthropic import AnthropicPrompter

    return AnthropicPrompter(options={"max_tokens": max_tokens}, model="claude-test")


def _thinking_block(text="Let me think about this."):
    # Real thinking blocks carry ``.thinking`` and have no ``.text``.
    return SimpleNamespace(type="thinking", thinking=text)


def _text_block(text):
    return SimpleNamespace(type="text", text=text)


def test_prompt_skips_leading_thinking_block(monkeypatch):
    prompter = _make_prompter(
        monkeypatch,
        [_thinking_block(), _text_block("the answer")],
    )

    assert asyncio.run(prompter.prompt(("hello",))) == "the answer"


def test_prompt_returns_text_when_first_block_is_text(monkeypatch):
    prompter = _make_prompter(monkeypatch, [_text_block("plain reply")])

    assert asyncio.run(prompter.prompt(("hello",))) == "plain reply"


def test_prompt_concatenates_multiple_text_blocks(monkeypatch):
    prompter = _make_prompter(
        monkeypatch,
        [_thinking_block(), _text_block("a"), _text_block("b")],
    )

    assert asyncio.run(prompter.prompt(("hello",))) == "ab"


def test_prompt_returns_none_for_thinking_only_content(monkeypatch):
    prompter = _make_prompter(monkeypatch, [_thinking_block()])

    assert asyncio.run(prompter.prompt(("hello",))) is None


def test_prompt_returns_none_for_empty_content(monkeypatch):
    prompter = _make_prompter(monkeypatch, [])

    assert asyncio.run(prompter.prompt(("hello",))) is None


def test_prompt_accepts_configured_stop_sequence(monkeypatch):
    prompter = _make_prompter(
        monkeypatch,
        [_text_block("stopped as configured")],
        stop_reason="stop_sequence",
    )

    assert asyncio.run(prompter.prompt(("hello",))) == "stopped as configured"


@pytest.mark.parametrize(
    ("stop_reason", "content", "message"),
    [
        ("max_tokens", [_text_block("partial answer")], "max_tokens=64"),
        (
            "model_context_window_exceeded",
            [_text_block("partial answer")],
            "model context window",
        ),
        ("refusal", [], "refused"),
        ("pause_turn", [_text_block("partial answer")], "paused before completion"),
    ],
)
def test_prompt_rejects_unsuccessful_terminal_reasons(monkeypatch, stop_reason, content, message):
    from vane.ai.provider import _ProviderResultError

    prompter = _make_prompter(monkeypatch, content, stop_reason=stop_reason)

    with pytest.raises(_ProviderResultError, match=message):
        asyncio.run(prompter.prompt(("hello",)))


@pytest.mark.parametrize("stop_reason", [None, "future_stop_reason", "tool_use"])
def test_plain_prompt_rejects_missing_or_unsupported_stop_reason(monkeypatch, stop_reason):
    from vane.ai.provider import _ProviderResultError

    prompter = _make_prompter(
        monkeypatch,
        [_text_block("must not be accepted")],
        stop_reason=stop_reason,
    )

    with pytest.raises(_ProviderResultError, match="missing or unsupported stop_reason"):
        asyncio.run(prompter.prompt(("hello",)))


def test_zero_max_tokens_prewarm_remains_successful(monkeypatch):
    prompter = _make_prompter(monkeypatch, [], stop_reason="max_tokens", max_tokens=0)

    assert asyncio.run(prompter.prompt(("hello",))) is None
