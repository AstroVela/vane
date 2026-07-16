"""Protocols defining the contracts for AI model implementations.

Each protocol is a structural type (``Protocol``) that any backend can
satisfy without inheriting from a base class. The corresponding
``*Descriptor`` classes are serializable factories that produce instances
conforming to the protocol.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from vane.ai.typing import Descriptor

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from vane.ai.typing import Embedding, EmbeddingDimensions, Label


# ---------------------------------------------------------------------------
# Text embedding
# ---------------------------------------------------------------------------


@runtime_checkable
class TextEmbedder(Protocol):
    """Embeds a batch of text strings into dense vectors."""

    def embed_text(self, text: list[str]) -> list[Embedding] | Awaitable[list[Embedding]]: ...


class TextEmbedderDescriptor(Descriptor["TextEmbedder"]):
    """Serializable factory for a :class:`TextEmbedder`."""

    @abstractmethod
    def get_dimensions(self) -> EmbeddingDimensions:
        """Return the embedding dimensions produced by this embedder."""
        ...

    def is_async(self) -> bool:
        """Whether ``embed_text`` returns an awaitable."""
        return False


# ---------------------------------------------------------------------------
# Text classification
# ---------------------------------------------------------------------------


@runtime_checkable
class TextClassifier(Protocol):
    """Classifies a batch of text strings."""

    def classify_text(self, text: list[str], labels: Label | list[Label]) -> list[Label]: ...


class TextClassifierDescriptor(Descriptor["TextClassifier"]):
    """Serializable factory for a :class:`TextClassifier`."""


# ---------------------------------------------------------------------------
# Prompting / chat completion
# ---------------------------------------------------------------------------


@runtime_checkable
class Prompter(Protocol):
    """Generates LLM responses for prompt messages."""

    async def prompt(self, messages: tuple[Any, ...]) -> Any: ...


class PrompterDescriptor(Descriptor["Prompter"]):
    """Serializable factory for a :class:`Prompter`."""
