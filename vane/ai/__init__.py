# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Vane AI — high-level AI function APIs.

Provides functions for embedding, prompting, and Jev judgments that integrate with
Vane's distributed execution engine.

Quick start::

    import vane
    from vane.ai import embed, prompt

    conn = vane.connect()
    rel = conn.sql("SELECT text FROM documents")
    embedded = embed(rel, vane.col("text"), provider="transformers")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vane.ai._jev import jev
    from vane.ai._schema import OutputValidationError, SchemaValidationError
    from vane.ai._video_embedding import VideoClip, VideoInputSpec
    from vane.ai.functions import embed, embed_image, embed_video, prompt
    from vane.ai.options import EmbedImageOptions, EmbedOptions, EmbedVideoOptions, JevOptions, PromptOptions
    from vane.ai.provider import ProviderCapabilityError
    from vane.ai.typing import JSONSchema

__all__ = [
    "Descriptor",
    "EmbedImageOptions",
    "EmbedOptions",
    "EmbedVideoOptions",
    "JSONSchema",
    "JevOptions",
    "OutputValidationError",
    "PromptOptions",
    "Provider",
    "ProviderCapabilityError",
    "RetryAfterError",
    "SchemaValidationError",
    "UDFOptions",
    "VideoClip",
    "VideoInputSpec",
    "embed",
    "embed_image",
    "embed_video",
    "jev",
    "load_provider",
    "prompt",
]

_LAZY_EXPORTS = {
    "Descriptor": ("vane.ai.typing", "Descriptor"),
    "EmbedImageOptions": ("vane.ai.options", "EmbedImageOptions"),
    "EmbedVideoOptions": ("vane.ai.options", "EmbedVideoOptions"),
    "EmbedOptions": ("vane.ai.options", "EmbedOptions"),
    "JSONSchema": ("vane.ai.typing", "JSONSchema"),
    "JevOptions": ("vane.ai.options", "JevOptions"),
    "OutputValidationError": ("vane.ai._schema", "OutputValidationError"),
    "PromptOptions": ("vane.ai.options", "PromptOptions"),
    "Provider": ("vane.ai.provider", "Provider"),
    "ProviderCapabilityError": ("vane.ai.provider", "ProviderCapabilityError"),
    "RetryAfterError": ("vane.ai.functions", "RetryAfterError"),
    "SchemaValidationError": ("vane.ai._schema", "SchemaValidationError"),
    "VideoClip": ("vane.ai._video_embedding", "VideoClip"),
    "VideoInputSpec": ("vane.ai._video_embedding", "VideoInputSpec"),
    "UDFOptions": ("vane.ai.typing", "UDFOptions"),
    "embed_video": ("vane.ai.functions", "embed_video"),
    "embed_image": ("vane.ai.functions", "embed_image"),
    "embed": ("vane.ai.functions", "embed"),
    "load_provider": ("vane.ai.provider", "load_provider"),
    "jev": ("vane.ai._jev", "jev"),
    "prompt": ("vane.ai.functions", "prompt"),
}


def __getattr__(name: str) -> Any:
    """Lazily import AI helpers so base ``import vane`` has minimal deps."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)

    from importlib import import_module

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
