# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Vane AI — high-level AI function APIs.

Provides one-line functions for embedding and prompting that integrate with
Vane's distributed execution engine.

Quick start::

    import vane
    from vane.ai import embed, prompt

    conn = vane.connect()
    rel = conn.sql("SELECT text FROM documents")
    embedded = embed(rel, vane.col("text"), provider="transformers")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vane.ai._schema import OutputValidationError, SchemaValidationError
    from vane.ai.functions import embed, prompt
    from vane.ai.options import EmbedOptions, PromptOptions
    from vane.ai.provider import ProviderCapabilityError
    from vane.ai.typing import JSONSchema

__all__ = [
    "Descriptor",
    "EmbedOptions",
    "JSONSchema",
    "OutputValidationError",
    "PromptOptions",
    "Provider",
    "ProviderCapabilityError",
    "RetryAfterError",
    "SchemaValidationError",
    "UDFOptions",
    "embed",
    "load_provider",
    "prompt",
]

_LAZY_EXPORTS = {
    "Descriptor": ("vane.ai.typing", "Descriptor"),
    "EmbedOptions": ("vane.ai.options", "EmbedOptions"),
    "JSONSchema": ("vane.ai.typing", "JSONSchema"),
    "OutputValidationError": ("vane.ai._schema", "OutputValidationError"),
    "PromptOptions": ("vane.ai.options", "PromptOptions"),
    "Provider": ("vane.ai.provider", "Provider"),
    "ProviderCapabilityError": ("vane.ai.provider", "ProviderCapabilityError"),
    "RetryAfterError": ("vane.ai.functions", "RetryAfterError"),
    "SchemaValidationError": ("vane.ai._schema", "SchemaValidationError"),
    "UDFOptions": ("vane.ai.typing", "UDFOptions"),
    "embed": ("vane.ai.functions", "embed"),
    "load_provider": ("vane.ai.provider", "load_provider"),
    "prompt": ("vane.ai.functions", "prompt"),
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
