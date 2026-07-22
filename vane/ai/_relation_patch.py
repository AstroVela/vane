# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Monkey-patch AI convenience methods onto DuckDBPyRelation.

This module adds ``.embed_text()``, ``.classify_text()``, and ``.prompt()``
directly to :class:`vane.DuckDBPyRelation` so users can write::

    rel.embed_text("text_col", provider="transformers")

instead of the functional form::

    from vane.ai import embed_text

    embed_text(rel, "text_col", provider="transformers")

The patch is applied once when this module is imported.
"""

from __future__ import annotations

from typing import Any

from vane import DuckDBPyRelation


def _embed_text(
    self: DuckDBPyRelation,
    column: str,
    *,
    provider: Any = None,
    model: str | None = None,
    dimensions: int | None = None,
    output_column: str = "embedding",
    execution_backend: str | None = None,
    **options: Any,
) -> DuckDBPyRelation:
    """Embed a text column. See :func:`vane.ai.embed_text` for details."""
    from vane.ai.functions import embed_text

    return embed_text(
        self,
        column,
        provider=provider,
        model=model,
        dimensions=dimensions,
        output_column=output_column,
        execution_backend=execution_backend,
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
    column: str,
    *,
    image_columns: list[str] | None = None,
    provider: Any = None,
    model: str | None = None,
    system_message: str | None = None,
    return_format: Any | None = None,
    use_chat_completions: bool = True,
    output_column: str = "response",
    execution_backend: str | None = None,
    **options: Any,
) -> DuckDBPyRelation:
    """Generate LLM responses. See :func:`vane.ai.prompt` for details."""
    from vane.ai.functions import prompt

    return prompt(
        self,
        column,
        image_columns=image_columns,
        provider=provider,
        model=model,
        system_message=system_message,
        return_format=return_format,
        use_chat_completions=use_chat_completions,
        output_column=output_column,
        execution_backend=execution_backend,
        **options,
    )


def _patch() -> None:
    """Apply AI methods to DuckDBPyRelation (idempotent)."""
    if hasattr(DuckDBPyRelation, "embed_text"):
        return
    DuckDBPyRelation.embed_text = _embed_text  # type: ignore[attr-defined]
    DuckDBPyRelation.classify_text = _classify_text  # type: ignore[attr-defined]
    DuckDBPyRelation.prompt = _prompt  # type: ignore[attr-defined]


_patch()
