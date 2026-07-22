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
        **options: Any,
    ) -> PrompterDescriptor | NativePrompterPlan:
        raise _not_implemented(self, "prompt")
