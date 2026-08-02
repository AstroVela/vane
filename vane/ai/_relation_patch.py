# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Monkey-patch AI convenience methods onto DuckDBPyRelation.

This module adds ``.embed()``, ``.classify_text()``, and ``.prompt()``
directly to :class:`duckdb.DuckDBPyRelation` so users can write::

    rel.embed(vane.col("text_col"), provider="transformers")

instead of the functional form::

    from vane.ai import embed

    embed(rel, vane.col("text_col"), provider="transformers")

The patch is applied once when this module is imported.
"""

from __future__ import annotations

from typing import Any, Literal

from typing_extensions import Unpack

from duckdb import DuckDBPyRelation, Expression
from vane.ai.options import EmbedOptions, PromptOptions
from vane.ai.provider import Provider


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


def _classify_text(
    self: DuckDBPyRelation,
    column: str,
    *,
    labels: list[str],
    provider: Any = None,
    model: str | None = None,
    output_column: str = "label",
    execution_backend: str | None = None,
    **options: Any,
) -> DuckDBPyRelation:
    """Classify a text column. See :func:`vane.ai.classify_text` for details."""
    from vane.ai.functions import classify_text

    return classify_text(
        self,
        column,
        labels=labels,
        provider=provider,
        model=model,
        output_column=output_column,
        execution_backend=execution_backend,
        **options,
    )


def _prompt(
    self: DuckDBPyRelation,
    messages: Expression | list[Expression],
    *,
    system_message: str | None = None,
    provider: str | Provider = "openai",
    model: str | None = None,
    on_error: Literal["raise", "ignore"] = "raise",
    output_column: str = "response",
    **options: Unpack[PromptOptions],
) -> DuckDBPyRelation:
    """Append basic Prompt responses. See :func:`vane.ai.prompt`."""
    from vane.ai.functions import prompt

    return prompt(
        self,
        messages,
        system_message=system_message,
        provider=provider,
        model=model,
        on_error=on_error,
        output_column=output_column,
        **options,
    )


def _patch() -> None:
    """Apply AI methods to DuckDBPyRelation (idempotent)."""
    if not hasattr(DuckDBPyRelation, "embed"):
        DuckDBPyRelation.embed = _embed  # type: ignore[attr-defined]
    if not hasattr(DuckDBPyRelation, "classify_text"):
        DuckDBPyRelation.classify_text = _classify_text  # type: ignore[attr-defined]
    if not hasattr(DuckDBPyRelation, "prompt"):
        DuckDBPyRelation.prompt = _prompt  # type: ignore[attr-defined]


_patch()
