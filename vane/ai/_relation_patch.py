# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Monkey-patch AI convenience methods onto DuckDBPyRelation.

This module adds ``.embed()`` and ``.prompt()`` directly to
:class:`vane.DuckDBPyRelation` so users can write::

    rel.embed(vane.col("text_col"), provider="transformers")

instead of the functional form::

    from vane.ai import embed

    embed(rel, vane.col("text_col"), provider="transformers")

The patch is applied once when this module is imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Unpack

from vane import DuckDBPyRelation, Expression
from vane.ai.options import EmbedOptions, PromptOptions
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


def _patch() -> None:
    """Apply AI methods to DuckDBPyRelation (idempotent)."""
    if not hasattr(DuckDBPyRelation, "embed"):
        setattr(DuckDBPyRelation, "embed", _embed)
    if not hasattr(DuckDBPyRelation, "prompt"):
        setattr(DuckDBPyRelation, "prompt", _prompt)


_patch()
