# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Protocols defining the contracts for AI model implementations.

Each protocol is a structural type (``Protocol``) that any backend can
satisfy without inheriting from a base class. The corresponding
``*Descriptor`` classes are serializable factories that produce instances
conforming to the protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from vane.ai.typing import Descriptor

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from vane._image import Image
    from vane.ai._transcription_types import Transcription, TranscriptionInput
    from vane.ai._video_embedding import VideoClip, VideoInputSpec
    from vane.ai.typing import Embedding


# ---------------------------------------------------------------------------
# Speech transcription
# ---------------------------------------------------------------------------


class Transcriber(Protocol):
    """Transcribe one encoded audio input into text and model-aligned segments."""

    async def transcribe(self, audio: TranscriptionInput) -> Transcription: ...

    async def aclose(self) -> None: ...


class TranscriberDescriptor(Descriptor["Transcriber"]):
    """Serializable transcription factory with a closed MIME capability list."""

    @abstractmethod
    def supported_media_mime_types(self) -> frozenset[str]: ...


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
    def get_dimensions(self) -> int:
        """Return the positive number of float32 values in each embedding."""
        ...

    def is_async(self) -> bool:
        """Whether ``embed_text`` returns an awaitable."""
        return False

    def supports_chunking(self) -> bool:
        """Whether explicit character chunking may average this model's vectors."""
        return True


# ---------------------------------------------------------------------------
# Image embedding
# ---------------------------------------------------------------------------


@runtime_checkable
class ImageEmbedder(Protocol):
    """Embed decoded HWC image arrays, preserving input order."""

    def embed_image(self, images: list[Image]) -> list[Embedding] | Awaitable[list[Embedding]]: ...


class ImageEmbedderDescriptor(Descriptor["ImageEmbedder"]):
    """Serializable image model configuration; metadata must require no I/O."""

    @abstractmethod
    def get_dimensions(self) -> int: ...

    def is_async(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Video embedding
# ---------------------------------------------------------------------------


@runtime_checkable
class VideoEmbedder(Protocol):
    """Embed a batch of ordered clips, returning one vector per clip."""

    def embed_video(self, clips: list[VideoClip]) -> list[Embedding] | Awaitable[list[Embedding]]: ...


class VideoEmbedderDescriptor(Descriptor["VideoEmbedder"]):
    """Serializable video model metadata; planning must perform no model I/O."""

    @abstractmethod
    def get_dimensions(self) -> int: ...

    @abstractmethod
    def get_input_spec(self) -> VideoInputSpec: ...

    def is_async(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Prompting / chat completion
# ---------------------------------------------------------------------------


@runtime_checkable
class Prompter(Protocol):
    """Generates LLM responses for prompt messages."""

    async def prompt(self, messages: tuple[Any, ...]) -> Any: ...


class PrompterDescriptor(Descriptor["Prompter"]):
    """Serializable factory for a :class:`Prompter`."""

    def supports_image_inputs(self) -> bool:
        """Whether this descriptor's statically selected model accepts images."""
        return True

    def supported_media_mime_types(self) -> frozenset[str] | None:
        """Return a closed Prompt MIME allowlist, or ``None`` when support is provider/model-dynamic."""
        return None


class NativePrompterPlan(ABC):
    """Serializable prompt metadata consumed directly by a native planner.

    Unlike :class:`PrompterDescriptor`, a native plan does not instantiate a
    worker-side Python object. Native-aware callers must recognize the plan
    and lower it before entering the generic Python UDF execution path.
    """

    @abstractmethod
    def get_provider(self) -> str:
        """Return the provider that produced this plan."""
        ...

    @abstractmethod
    def get_model(self) -> str:
        """Return the native model identifier."""
        ...

    @abstractmethod
    def get_options(self) -> dict[str, Any]:
        """Return the provider-specific native planning options."""
        ...


class NativeInferencePlan(NativePrompterPlan):
    """Common base for native inference-backend plans (vLLM, SGLang, ...).

    Native inference plans lower into the shared native operator; the engine
    field selects which executor factory runs at execution time.
    """

    model_name: str
    system_message: str | None
    on_error: str

    @abstractmethod
    def get_engine(self) -> str:
        """Return the inference engine name ("vllm" | "sglang")."""
        ...

    @abstractmethod
    def build_physical_vllm_options(self) -> dict[str, Any]:
        """Build options for the native PhysicalVLLM operator."""
        ...
