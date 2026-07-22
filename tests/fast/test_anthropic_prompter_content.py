# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""AnthropicPrompter plain-text extraction across response content blocks.

Extended-thinking models return a leading ``thinking`` block (which has
``.thinking`` and no ``.text``), so ``prompt`` must return the first
``text`` block rather than assuming ``content[0]`` is text.
"""

import asyncio
import sys
from types import SimpleNamespace


def _make_prompter(monkeypatch, content):
    """Build an AnthropicPrompter whose SDK client returns ``content``."""

    class FakeMessages:
        async def create(self, **kwargs):
            return SimpleNamespace(content=content, usage=None)

    class FakeAsyncAnthropic:
        def __init__(self, **options):
            self.messages = FakeMessages()

    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(AsyncAnthropic=FakeAsyncAnthropic),
    )

    from vane.ai.providers.anthropic import AnthropicPrompter

    return AnthropicPrompter(provider_options={}, model="claude-test")


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


def test_prompt_returns_none_for_thinking_only_content(monkeypatch):
    prompter = _make_prompter(monkeypatch, [_thinking_block()])

    assert asyncio.run(prompter.prompt(("hello",))) is None


def test_prompt_returns_none_for_empty_content(monkeypatch):
    prompter = _make_prompter(monkeypatch, [])

    assert asyncio.run(prompter.prompt(("hello",))) is None
