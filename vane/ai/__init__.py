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
    from vane.ai._audio_embedding import AudioClip, AudioInputSpec
    from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec, MultimodalInputSpec
    from vane.ai._embedding_outputs import EmbeddingOutput, EmbeddingOutputSpec
    from vane.ai._errors import DiagnosticError, summarize_error
    from vane.ai._jev import jev
    from vane.ai._schema import OutputValidationError, SchemaValidationError
    from vane.ai._video_embedding import VideoClip, VideoInputSpec
    from vane.ai.embedding import AudioInput, ImageInput, MultimodalInput, TextInput, VideoInput
    from vane.ai.functions import embed, prompt
    from vane.ai.options import (
        EmbedOptions,
        JevOptions,
        PromptOptions,
    )
    from vane.ai.provider import ProviderCapabilityError
    from vane.ai.typing import JSONSchema

__all__ = [
    "EmbeddingOutput",
    "EmbeddingOutputSpec",
    "EmbeddingContent",
    "EmbeddingPart",
    "EmbeddingPartSpec",
    "MultimodalInputSpec",
    "AudioInput",
    "ImageInput",
    "VideoInput",
    "TextInput",
    "MultimodalInput",
    "AudioClip",
    "AudioInputSpec",
    "Descriptor",
    "DiagnosticError",
    "EmbedOptions",
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
    "jev",
    "load_provider",
    "prompt",
    "summarize_error",
]

_LAZY_EXPORTS = {
    "EmbeddingOutput": ("vane.ai._embedding_outputs", "EmbeddingOutput"),
    "EmbeddingOutputSpec": ("vane.ai._embedding_outputs", "EmbeddingOutputSpec"),
    "EmbeddingContent": ("vane.ai._embedding_content", "EmbeddingContent"),
    "EmbeddingPart": ("vane.ai._embedding_content", "EmbeddingPart"),
    "EmbeddingPartSpec": ("vane.ai._embedding_content", "EmbeddingPartSpec"),
    "MultimodalInputSpec": ("vane.ai._embedding_content", "MultimodalInputSpec"),
    "AudioInput": ("vane.ai.embedding", "AudioInput"),
    "ImageInput": ("vane.ai.embedding", "ImageInput"),
    "VideoInput": ("vane.ai.embedding", "VideoInput"),
    "TextInput": ("vane.ai.embedding", "TextInput"),
    "MultimodalInput": ("vane.ai.embedding", "MultimodalInput"),
    "DiagnosticError": ("vane.ai._errors", "DiagnosticError"),
    "summarize_error": ("vane.ai._errors", "summarize_error"),
    "AudioClip": ("vane.ai._audio_embedding", "AudioClip"),
    "AudioInputSpec": ("vane.ai._audio_embedding", "AudioInputSpec"),
    "Descriptor": ("vane.ai.typing", "Descriptor"),
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
