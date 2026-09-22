# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Monkey-patch AI convenience methods onto DuckDBPyRelation.

This module adds AI helpers including ``.embed()``, ``.prompt()``, and ``.jev()`` to
:class:`vane.DuckDBPyRelation` so users can write::

    rel.embed(vane.col("text_col"), provider="transformers")

instead of the functional form::

    from vane.ai import embed

    embed(rel, vane.col("text_col"), provider="transformers")

The patch is applied once when this module is imported.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Unpack

from vane import DuckDBPyRelation, Expression
from vane.ai.options import (
    EmbedImageOptions,
    EmbedOptions,
    EmbedVideoOptions,
    JevOptions,
    PromptOptions,
    TranscribeOptions,
)
from vane.ai.provider import Provider
from vane.ai.typing import JSONSchema

if TYPE_CHECKING:
    from pydantic import BaseModel  # type: ignore[import-not-found, import-untyped, unused-ignore]
else:
    BaseModel = Any


def _embed(
    self: DuckDBPyRelation,
    text: Expression,
    *,
    provider: str | Provider = "openai",
    model: str | None = None,
    dimensions: int | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "embedding",
    **options: Unpack[EmbedOptions],
) -> DuckDBPyRelation:
    """Append a fixed-size embedding column. See :func:`vane.ai.embed`."""
    from vane.ai.functions import embed

    return embed(
        self,
        text,
        provider=provider,
        model=model,
        dimensions=dimensions,
        on_error=on_error,
        output_column=output_column,
        **options,
    )


def _embed_image(
    self: DuckDBPyRelation,
    image: Expression,
    *,
    provider: str | Provider = "transformers",
    model: str | None = None,
    dimensions: int | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "embedding",
    **options: Unpack[EmbedImageOptions],
) -> DuckDBPyRelation:
    """Append a fixed-size embedding column. See :func:`vane.ai.embed_image`."""
    from vane.ai.functions import embed_image

    return embed_image(
        self,
        image,
        provider=provider,
        model=model,
        dimensions=dimensions,
        on_error=on_error,
        output_column=output_column,
        **options,
    )


def _embed_video(
    self: DuckDBPyRelation,
    frames: Expression,
    *,
    provider: str | Provider = "transformers",
    model: str | None = None,
    dimensions: int | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "embedding",
    **options: Unpack[EmbedVideoOptions],
) -> DuckDBPyRelation:
    """Append a fixed-size embedding column. See :func:`vane.ai.embed_video`."""
    from vane.ai.functions import embed_video

    return embed_video(
        self,
        frames,
        provider=provider,
        model=model,
        dimensions=dimensions,
        on_error=on_error,
        output_column=output_column,
        **options,
    )


def _prompt(
    self: DuckDBPyRelation,
    messages: Expression | list[Expression],
    *,
    return_format: type[BaseModel] | JSONSchema | None = None,
    system_message: str | None = None,
    provider: str | Provider = "openai",
    model: str | None = None,
    return_raw_response: bool = False,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "response",
    **options: Unpack[PromptOptions],
) -> DuckDBPyRelation:
    """Append text, structured, or raw Prompt responses. See :func:`vane.ai.prompt`."""
    from vane.ai.functions import prompt

    return prompt(
        self,
        messages,
        return_format=return_format,
        system_message=system_message,
        provider=provider,
        model=model,
        return_raw_response=return_raw_response,
        on_error=on_error,
        output_column=output_column,
        **options,
    )


def _jev(
    self: DuckDBPyRelation,
    state: Expression,
    *,
    questions: Mapping[str, Any],
    model: str = "jev-latest",
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "response",
    **options: Unpack[JevOptions],
) -> DuckDBPyRelation:
    """Append Jev judgments as JSON text. See :func:`vane.ai.jev`."""
    from vane.ai._jev import jev

    return jev(self, state, questions=questions, model=model, on_error=on_error, output_column=output_column, **options)


def _transcribe(
    self: DuckDBPyRelation,
    audio: Expression,
    *,
    provider: str | Provider = "openai",
    model: str | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "transcription",
    **options: Unpack[TranscribeOptions],
) -> DuckDBPyRelation:
    """Append timed speech segments. See :func:`vane.ai.transcribe`."""
    from vane.ai._transcription import transcribe

    return transcribe(
        self, audio, provider=provider, model=model, on_error=on_error, output_column=output_column, **options
    )


def _patch() -> None:
    """Apply AI methods to DuckDBPyRelation (idempotent)."""
    if not hasattr(DuckDBPyRelation, "embed"):
        setattr(DuckDBPyRelation, "embed", _embed)
    if not hasattr(DuckDBPyRelation, "embed_image"):
        setattr(DuckDBPyRelation, "embed_image", _embed_image)
    if not hasattr(DuckDBPyRelation, "embed_video"):
        setattr(DuckDBPyRelation, "embed_video", _embed_video)
    if not hasattr(DuckDBPyRelation, "prompt"):
        setattr(DuckDBPyRelation, "prompt", _prompt)
    if not hasattr(DuckDBPyRelation, "jev"):
        setattr(DuckDBPyRelation, "jev", _jev)
    if not hasattr(DuckDBPyRelation, "transcribe"):
        setattr(DuckDBPyRelation, "transcribe", _transcribe)


_patch()
