# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Typed inputs for :func:`vane.ai.embed`.

Values are expressions, not eagerly loaded media. A media input without a MIME
type contains decoded IMAGE, video frame records, or PCM. An explicit MIME type
selects encoded BLOB bytes; the provider must support that representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias

from vane._expressions import is_expression
from vane._typing import Expression

Modality: TypeAlias = Literal["text", "image", "video", "audio"]


@dataclass(frozen=True)
class TextInput:
    value: Expression
    kind: ClassVar[Modality] = "text"

    def __post_init__(self) -> None:
        if not is_expression(self.value):
            raise TypeError("TextInput requires an Expression")


@dataclass(frozen=True)
class _MediaInput:
    value: Expression
    mime_type: str | None = None
    kind: ClassVar[Modality]

    def __post_init__(self) -> None:
        if not is_expression(self.value):
            raise TypeError(f"{type(self).__name__} requires an Expression")
        if self.mime_type is not None and (
            not isinstance(self.mime_type, str)
            or not self.mime_type.startswith(f"{self.kind}/")
            or not self.mime_type.removeprefix(f"{self.kind}/")
            or any(character.isspace() for character in self.mime_type)
            or ";" in self.mime_type
        ):
            raise ValueError(f"{type(self).__name__} requires an explicit {self.kind} MIME type")


@dataclass(frozen=True)
class ImageInput(_MediaInput):
    """Decoded IMAGE, or encoded BLOB when ``mime_type`` is supplied."""

    kind: ClassVar[Modality] = "image"


@dataclass(frozen=True)
class VideoInput(_MediaInput):
    """Ordered frame records with source timestamps, or encoded video BLOB."""

    kind: ClassVar[Modality] = "video"


@dataclass(frozen=True)
class AudioInput(_MediaInput):
    """Decoded STRUCT(sample_rate, data TENSOR), or encoded audio BLOB."""

    kind: ClassVar[Modality] = "audio"


InputPart: TypeAlias = TextInput | ImageInput | VideoInput | AudioInput


@dataclass(frozen=True)
class MultimodalInput:
    """Ordered optional parts of ONE input, producing ONE vector per row.

    NULL parts and empty/whitespace-only text are absent. At least one part
    must remain in every row. Extraction failures must be recorded by the
    caller before assembling these inputs; they are not inferred from NULLs.
    """

    parts: tuple[InputPart, ...]

    def __init__(self, parts: list[InputPart] | tuple[InputPart, ...]) -> None:
        if not isinstance(parts, (list, tuple)) or not parts:
            raise ValueError("MultimodalInput requires a nonempty list or tuple of typed parts")
        if any(not isinstance(part, (TextInput, ImageInput, VideoInput, AudioInput)) for part in parts):
            raise TypeError("MultimodalInput parts must be TextInput, ImageInput, VideoInput or AudioInput")
        object.__setattr__(self, "parts", tuple(parts))


EmbeddingInput: TypeAlias = Expression | InputPart | MultimodalInput
