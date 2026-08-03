# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Provider base class and registry for AI model backends.

A :class:`Provider` maps high-level intents (embed text, classify text, prompt)
to either concrete :class:`~vane.ai.typing.Descriptor` factories or native
planning metadata. Both forms are lightweight and serializable.

Supported providers are loaded lazily so optional dependencies (e.g.
``transformers``, ``openai``) are only imported when actually used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from vane.ai.protocols import (
        NativePrompterPlan,
        PrompterDescriptor,
        TextClassifierDescriptor,
        TextEmbedderDescriptor,
    )


class ProviderImportError(ImportError):
    """Raised when an optional provider dependency is not installed."""

    def __init__(self, extra: str, *, function: str | None = None):
        fn_msg = f" to use the {function} function" if function else ""
        super().__init__(f"Please `pip install 'vane-ai[{extra}]'`{fn_msg} with this provider.")


_MAX_ERROR_TYPE_CHARS = 128
_SAFE_ERROR_DETAIL_NAMES = ("status_code", "status", "code")


class _SafeProviderError(RuntimeError):
    """A bounded, credential-safe summary of an upstream provider error."""


def _safe_error_type(original_error: Exception) -> str:
    try:
        name = type(original_error).__name__
    except Exception:
        return "Exception"
    name = name[:_MAX_ERROR_TYPE_CHARS]
    if not name or not name.isascii() or not name.isidentifier():
        return "Exception"
    return name


def _safe_original_error_summary(original_error: Exception) -> str:
    if isinstance(original_error, _SafeProviderError):
        return str(original_error)
    error_type = _safe_error_type(original_error)
    details: list[str] = []
    for name in _SAFE_ERROR_DETAIL_NAMES:
        try:
            value = getattr(original_error, name, None)
        except Exception:
            continue
        # Provider messages and string-valued fields can echo opaque API keys,
        # tokens, request bodies, or other user data without a recognizable
        # label.  Preserve only builtin numeric status metadata; never inspect
        # or stringify arbitrary upstream text on an exception path.
        if type(value) is not int or not -999_999 <= value <= 999_999:
            continue
        details.append(f"{name}={value}")
    summary = error_type
    if details:
        summary += " (" + ", ".join(details) + ")"
    return summary


def _safe_provider_execution_error(
    provider: str,
    model: str,
    operation: str,
    original_error: Exception,
) -> RuntimeError:
    """Return a public-safe final error after provider retry handling."""
    summary = _safe_original_error_summary(original_error)
    return RuntimeError(f"Provider {provider!r} model {model!r} failed during {operation}; upstream error: {summary}")


class ProviderCapabilityError(RuntimeError):
    """A runtime endpoint or model cannot satisfy a requested AI capability.

    Static capability mismatches are rejected while preparing the call. This
    error is reserved for facts that can only be learned from the selected
    endpoint or loaded model at execution time.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        capability: str,
        *,
        original_error: Exception | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.capability = capability
        self.original_error_summary = (
            _safe_original_error_summary(original_error) if original_error is not None else None
        )
        self.original_error = (
            _SafeProviderError(self.original_error_summary) if self.original_error_summary is not None else None
        )
        message = f"Provider {provider!r} model {model!r} does not support the requested {capability} capability"
        if self.original_error_summary is not None:
            message += f"; upstream error: {self.original_error_summary}"
        super().__init__(message)

    @classmethod
    def _from_safe_summary(
        cls,
        provider: str,
        model: str,
        capability: str,
        original_error_summary: str | None,
    ) -> ProviderCapabilityError:
        original_error = _SafeProviderError(original_error_summary) if original_error_summary is not None else None
        return cls(provider, model, capability, original_error=original_error)

    def __reduce__(self) -> tuple[Any, tuple[str, str, str, str | None]]:
        return (
            _restore_provider_capability_error,
            (self.provider, self.model, self.capability, self.original_error_summary),
        )


def _restore_provider_capability_error(
    provider: str,
    model: str,
    capability: str,
    original_error_summary: str | None,
) -> ProviderCapabilityError:
    return ProviderCapabilityError._from_safe_summary(provider, model, capability, original_error_summary)


def _safe_provider_capability_error(error: ProviderCapabilityError) -> ProviderCapabilityError:
    """Rebuild a capability error without trusting its exception chain."""
    return ProviderCapabilityError._from_safe_summary(
        error.provider,
        error.model,
        error.capability,
        error.original_error_summary,
    )


class _ProviderResultError(TypeError):
    """A Provider response violates the row-preserving typed result contract."""


# ---------------------------------------------------------------------------
# Lazy loader functions
# ---------------------------------------------------------------------------


def _load_transformers(name: str | None = None, **options: Any) -> Provider:
    try:
        from vane.ai.providers.transformers import TransformersProvider

        return TransformersProvider(name, **options)
    except ImportError as e:
        raise ProviderImportError("transformers") from e


def _load_openai(name: str | None = None, **options: Any) -> Provider:
    try:
        from vane.ai.providers.openai import OpenAIProvider

        return OpenAIProvider(name, **options)
    except ImportError as e:
        raise ProviderImportError("openai") from e


def _load_vllm(name: str | None = None, **options: Any) -> Provider:
    try:
        from vane.ai.providers.vllm import VLLMProvider

        return VLLMProvider(name, **options)
    except ImportError as e:
        raise ProviderImportError("vllm") from e


def _load_anthropic(name: str | None = None, **options: Any) -> Provider:
    try:
        from vane.ai.providers.anthropic import AnthropicProvider

        return AnthropicProvider(name, **options)
    except ImportError as e:
        raise ProviderImportError("anthropic") from e


def _load_google(name: str | None = None, **options: Any) -> Provider:
    try:
        from vane.ai.providers.google import GoogleProvider

        return GoogleProvider(name, **options)
    except ImportError as e:
        raise ProviderImportError("google") from e


PROVIDERS: dict[str, Callable[..., Provider]] = {
    "transformers": _load_transformers,
    "openai": _load_openai,
    "vllm": _load_vllm,
    "anthropic": _load_anthropic,
    "google": _load_google,
}


def load_provider(provider: str, name: str | None = None, **options: Any) -> Provider:
    """Load a provider by name.

    Args:
        provider: One of the registered provider names (e.g. ``"transformers"``).
        name: Optional display name override.
        **options: Forwarded to the provider constructor.

    Raises:
        ValueError: If the provider name is not registered.
        ProviderImportError: If the provider's dependencies are missing.
    """
    factory = PROVIDERS.get(provider)
    if factory is None:
        raise ValueError(f"Provider {provider!r} is not supported. Available: {sorted(PROVIDERS)}")
    return factory(name, **options)


def _not_implemented(provider: Provider, method: str) -> NotImplementedError:
    return NotImplementedError(f"{method} is not implemented for the {provider.name!r} provider")


# ---------------------------------------------------------------------------
# Provider ABC
# ---------------------------------------------------------------------------


class Provider(ABC):
    """Base class for AI model providers.

    Subclasses implement ``get_text_embedder``, ``get_text_classifier``, etc.
    to return lightweight descriptors or native planning metadata.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short provider identifier (e.g. ``"transformers"``)."""
        ...

    # -- Text embedding -----------------------------------------------------

    def get_text_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        **options: Any,
    ) -> TextEmbedderDescriptor:
        raise _not_implemented(self, "embed_text")

    # -- Text classification ------------------------------------------------

    def get_text_classifier(self, model: str | None = None, **options: Any) -> TextClassifierDescriptor:
        raise _not_implemented(self, "classify_text")

    # -- Prompting / chat completion ----------------------------------------

    def get_prompter(
        self,
        model: str | None = None,
        system_message: str | None = None,
        return_format: dict[str, Any] | None = None,
        return_raw_response: bool = False,
        **options: Any,
    ) -> PrompterDescriptor | NativePrompterPlan:
        raise _not_implemented(self, "prompt")
