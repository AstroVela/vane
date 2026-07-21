# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Typed option objects for high-level AI helper functions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Literal

from vane.ai._redaction import REDACTED_PLACEHOLDER, wrap_sensitive_options


def _set_if_not_none(target: dict[str, Any], key: str, value: object) -> None:
    if value is not None:
        target[key] = value


class _RedactedOptionsRepr:
    """Mixin rendering credential-bearing fields redacted in the dataclass-style repr.

    Scalar fields named in ``_REDACTED_FIELDS`` render as a fixed placeholder
    when set (``None`` still renders as ``None`` — masking an absent key would
    be misleading). Mapping-valued fields render with sensitive keys sealed at
    any nesting depth, so nested credentials (e.g. an HF hub token inside
    ``engine_args``) never reach the repr. Dataclasses opting in must be
    declared with ``repr=False`` so the generated repr does not shadow this one.
    """

    _REDACTED_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key"})

    def __repr__(self) -> str:
        parts = []
        for field in fields(self):  # type: ignore[arg-type]
            value = getattr(self, field.name)
            if field.name in self._REDACTED_FIELDS and value is not None:
                parts.append(f"{field.name}={REDACTED_PLACEHOLDER}")
            elif isinstance(value, Mapping):
                parts.append(f"{field.name}={wrap_sensitive_options(value)!r}")
            else:
                parts.append(f"{field.name}={value!r}")
        return f"{type(self).__qualname__}({', '.join(parts)})"


@dataclass(frozen=True, repr=False)
class OpenAIProviderOptions(_RedactedOptionsRepr):
    """OpenAI-compatible provider options shared by prompt and embedding calls."""

    # ``organization`` identifies the paying account; the OpenAI provider seals
    # it at the descriptor layer, so the public repr must not leak it either.
    _REDACTED_FIELDS: ClassVar[frozenset[str]] = frozenset({"api_key", "organization"})

    base_url: str | None = None
    api_key: str | None = None
    organization: str | None = None
    timeout: float | None = None
    concurrency: int | None = None
    max_api_concurrency: int | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        _set_if_not_none(options, "base_url", self.base_url)
        _set_if_not_none(options, "api_key", self.api_key)
        _set_if_not_none(options, "organization", self.organization)
        _set_if_not_none(options, "timeout", self.timeout)
        _set_if_not_none(options, "actor_number", self.concurrency)
        _set_if_not_none(options, "max_api_concurrency", self.max_api_concurrency)
        return options


@dataclass(frozen=True, repr=False)
class VLLMProviderOptions(_RedactedOptionsRepr):
    """vLLM provider options for actor count, GPU allocation, and engine args."""

    engine_args: Mapping[str, Any] | None = None
    concurrency: int | None = None
    gpus_per_actor: float | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        if self.engine_args is not None:
            options["engine_args"] = dict(self.engine_args)
        _set_if_not_none(options, "actor_number", self.concurrency)
        _set_if_not_none(options, "gpus_per_actor", self.gpus_per_actor)
        return options


@dataclass(frozen=True)
class OpenAIPromptOptions:
    """OpenAI-compatible prompt request options."""

    use_chat_completions: bool | None = None
    max_output_tokens: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    on_error: Literal["raise", "log", "ignore"] | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        _set_if_not_none(options, "use_chat_completions", self.use_chat_completions)
        _set_if_not_none(options, "max_output_tokens", self.max_output_tokens)
        _set_if_not_none(options, "max_tokens", self.max_tokens)
        _set_if_not_none(options, "temperature", self.temperature)
        _set_if_not_none(options, "on_error", self.on_error)
        return options


@dataclass(frozen=True)
class OpenAIEmbeddingOptions:
    """OpenAI-compatible embedding request options."""

    encoding_format: Literal["float", "base64"] = "float"
    on_error: Literal["raise", "log", "ignore"] | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {"encoding_format": self.encoding_format}
        _set_if_not_none(options, "on_error", self.on_error)
        return options


@dataclass(frozen=True, repr=False)
class AnthropicProviderOptions(_RedactedOptionsRepr):
    """Anthropic provider options for client configuration and execution limits."""

    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    max_retries: int | None = None
    concurrency: int | None = None
    max_api_concurrency: int | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        _set_if_not_none(options, "api_key", self.api_key)
        _set_if_not_none(options, "base_url", self.base_url)
        _set_if_not_none(options, "timeout", self.timeout)
        _set_if_not_none(options, "max_retries", self.max_retries)
        _set_if_not_none(options, "actor_number", self.concurrency)
        _set_if_not_none(options, "max_api_concurrency", self.max_api_concurrency)
        return options


@dataclass(frozen=True)
class AnthropicPromptOptions:
    """Anthropic prompt request options."""

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    on_error: Literal["raise", "log", "ignore"] | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        _set_if_not_none(options, "max_tokens", self.max_tokens)
        _set_if_not_none(options, "temperature", self.temperature)
        _set_if_not_none(options, "top_p", self.top_p)
        _set_if_not_none(options, "top_k", self.top_k)
        if self.stop_sequences is not None:
            options["stop_sequences"] = list(self.stop_sequences)
        _set_if_not_none(options, "on_error", self.on_error)
        return options


@dataclass(frozen=True, repr=False)
class GoogleProviderOptions(_RedactedOptionsRepr):
    """Google provider options for client configuration and execution limits."""

    api_key: str | None = None
    concurrency: int | None = None
    max_api_concurrency: int | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        _set_if_not_none(options, "api_key", self.api_key)
        _set_if_not_none(options, "actor_number", self.concurrency)
        _set_if_not_none(options, "max_api_concurrency", self.max_api_concurrency)
        return options


@dataclass(frozen=True)
class GooglePromptOptions:
    """Google Gemini prompt request options."""

    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    on_error: Literal["raise", "log", "ignore"] | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        _set_if_not_none(options, "max_output_tokens", self.max_output_tokens)
        _set_if_not_none(options, "temperature", self.temperature)
        _set_if_not_none(options, "top_p", self.top_p)
        _set_if_not_none(options, "top_k", self.top_k)
        _set_if_not_none(options, "on_error", self.on_error)
        return options


@dataclass(frozen=True)
class GoogleEmbeddingOptions:
    """Google embedding request options."""

    task_type: str | None = None
    title: str | None = None
    on_error: Literal["raise", "log", "ignore"] | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        _set_if_not_none(options, "task_type", self.task_type)
        _set_if_not_none(options, "title", self.title)
        _set_if_not_none(options, "on_error", self.on_error)
        return options


@dataclass(frozen=True, repr=False)
class VLLMPromptOptions(_RedactedOptionsRepr):
    """vLLM prompt generation options."""

    generate_args: Mapping[str, Any] | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    on_error: Literal["raise", "log", "ignore"] | None = None

    def to_descriptor_options(self) -> dict[str, Any]:
        """Convert public options to provider descriptor keyword arguments."""
        options: dict[str, Any] = {}
        if self.generate_args is not None:
            options["generate_args"] = dict(self.generate_args)
        _set_if_not_none(options, "max_tokens", self.max_tokens)
        _set_if_not_none(options, "temperature", self.temperature)
        _set_if_not_none(options, "on_error", self.on_error)
        return options
