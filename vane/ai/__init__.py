# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Vane AI — high-level AI function APIs.

Provides one-line functions for common AI tasks (embedding, classification,
prompting) that integrate with Vane's distributed execution engine.

Quick start::

    import vane
    from vane.ai import classify_text, embed, prompt

    conn = vane.connect()
    rel = conn.sql("SELECT text FROM documents")
    embedded = embed(rel, vane.col("text"), provider="transformers")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vane.ai.functions import embed
    from vane.ai.options import EmbedOptions
    from vane.ai.provider import ProviderCapabilityError

__all__ = [
    "AnthropicPromptOptions",
    "AnthropicProviderOptions",
    "Descriptor",
    "EmbedOptions",
    "GooglePromptOptions",
    "GoogleProviderOptions",
    "OpenAIPromptOptions",
    "OpenAIProviderOptions",
    "Provider",
    "ProviderCapabilityError",
    "RetryAfterError",
    "TokenMetricsEntry",
    "UDFOptions",
    "VLLMPromptOptions",
    "VLLMProviderOptions",
    "classify_text",
    "embed",
    "get_token_metrics",
    "get_token_metrics_summary",
    "load_provider",
    "prompt",
    "record_token_metrics",
    "reset_token_metrics",
    "set_token_metrics_callback",
]

_LAZY_EXPORTS = {
    "Descriptor": ("vane.ai.typing", "Descriptor"),
    "AnthropicPromptOptions": ("vane.ai.options", "AnthropicPromptOptions"),
    "AnthropicProviderOptions": ("vane.ai.options", "AnthropicProviderOptions"),
    "EmbedOptions": ("vane.ai.options", "EmbedOptions"),
    "GooglePromptOptions": ("vane.ai.options", "GooglePromptOptions"),
    "GoogleProviderOptions": ("vane.ai.options", "GoogleProviderOptions"),
    "OpenAIProviderOptions": ("vane.ai.options", "OpenAIProviderOptions"),
    "OpenAIPromptOptions": ("vane.ai.options", "OpenAIPromptOptions"),
    "Provider": ("vane.ai.provider", "Provider"),
    "ProviderCapabilityError": ("vane.ai.provider", "ProviderCapabilityError"),
    "RetryAfterError": ("vane.ai.functions", "RetryAfterError"),
    "TokenMetricsEntry": ("vane.ai.metrics", "TokenMetricsEntry"),
    "UDFOptions": ("vane.ai.typing", "UDFOptions"),
    "VLLMProviderOptions": ("vane.ai.options", "VLLMProviderOptions"),
    "VLLMPromptOptions": ("vane.ai.options", "VLLMPromptOptions"),
    "classify_text": ("vane.ai.functions", "classify_text"),
    "embed": ("vane.ai.functions", "embed"),
    "get_token_metrics": ("vane.ai.metrics", "get_token_metrics"),
    "get_token_metrics_summary": ("vane.ai.metrics", "get_token_metrics_summary"),
    "load_provider": ("vane.ai.provider", "load_provider"),
    "prompt": ("vane.ai.functions", "prompt"),
    "record_token_metrics": ("vane.ai.metrics", "record_token_metrics"),
    "reset_token_metrics": ("vane.ai.metrics", "reset_token_metrics"),
    "set_token_metrics_callback": ("vane.ai.metrics", "set_token_metrics_callback"),
}


def __getattr__(name: str):
    """Lazily import AI helpers so base ``import vane`` has minimal deps."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)

    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
