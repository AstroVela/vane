# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Static, serializable contracts for the native outputs of one input row."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.embedding import Modality
from vane.ai.typing import Descriptor

if TYPE_CHECKING:
    import pyarrow as pa

_Embedder = TypeVar("_Embedder")


@dataclass(frozen=True)
class EmbeddingOutput:
    """One native output, its vector space and input-modality provenance.

    ``space`` is a provider-owned semantic identity, shared only by paired
    encoders. Applications must also retain the model/revision, preprocessing
    and encoding roles; matching dimensions alone never establish compatibility.
    ``source_modalities`` maps this output to contributing input parts, not to
    request rows. An absent optional output is represented by NULL.
    """

    name: str
    dimensions: int
    space: str
    source_modalities: tuple[Modality, ...] = ("text", "image", "video", "audio")
    optional: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", self.name) is None:
            raise EmbeddingConfigurationError("Embedding output names must be 1-64 ASCII identifier characters")
        if isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int) or self.dimensions < 1:
            raise EmbeddingConfigurationError("Embedding output dimensions must be a positive integer")
        if not isinstance(self.space, str) or not self.space.strip() or len(self.space) > 1024:
            raise EmbeddingConfigurationError(
                "Embedding output space must be a nonempty identity of at most 1024 characters"
            )
        modalities = self.source_modalities
        if (
            not isinstance(modalities, tuple)
            or not modalities
            or any(not isinstance(kind, str) or kind not in {"text", "image", "video", "audio"} for kind in modalities)
            or len(set(modalities)) != len(modalities)
        ):
            raise EmbeddingConfigurationError(
                "Embedding output source_modalities must be a nonempty tuple of distinct modalities"
            )
        if not isinstance(self.optional, bool):
            raise EmbeddingConfigurationError("Embedding output optional must be BOOLEAN")


@dataclass(frozen=True)
class EmbeddingOutputSpec:
    """Native result shape: a single vector, or a closed set of named vectors.

    A named result is one mapping per input row. Every declared key must be
    present, even for an optional NULL output. Outputs may have different
    dimensions and spaces. No independent requests or averaged vectors are
    synthesized from this declaration.
    """

    outputs: tuple[EmbeddingOutput, ...]
    layout: Literal["vector", "named"] = "vector"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.outputs, tuple)
            or not 1 <= len(self.outputs) <= 64
            or any(not isinstance(output, EmbeddingOutput) for output in self.outputs)
        ):
            raise EmbeddingConfigurationError("Embedding outputs must be a tuple of 1-64 EmbeddingOutput declarations")
        if len({output.name.casefold() for output in self.outputs}) != len(self.outputs):
            raise EmbeddingConfigurationError("Embedding output names must be unique ignoring case")
        if not isinstance(self.layout, str) or self.layout not in {"vector", "named"}:
            raise EmbeddingConfigurationError("Embedding output layout must be vector or named")
        if self.layout == "vector" and (len(self.outputs) != 1 or self.outputs[0].optional):
            raise EmbeddingConfigurationError("Vector layout requires exactly one required output")

    def sql_type(self) -> str:
        """Closed native type, without consulting a model or a remote service."""
        if self.layout == "vector":
            return f"FLOAT[{self.outputs[0].dimensions}]"
        return "STRUCT(" + ", ".join(f'"{output.name}" FLOAT[{output.dimensions}]' for output in self.outputs) + ")"

    def arrow_type(self) -> pa.DataType:
        import pyarrow as pa

        if self.layout == "vector":
            return pa.list_(pa.float32(), self.outputs[0].dimensions)
        return pa.struct([(output.name, pa.list_(pa.float32(), output.dimensions)) for output in self.outputs])


class EmbeddingDescriptor(Descriptor[_Embedder]):
    """Default contract for encoders whose native result is one vector.

    Named-output descriptors override ``get_output_spec``. ``get_dimensions``
    remains a convenience for single-vector encoders; multiple outputs declare
    their dimensions in the output spec instead.
    """

    def get_dimensions(self) -> int:
        raise EmbeddingConfigurationError("Named embedding outputs declare dimensions in get_output_spec()")

    def get_output_spec(self, *, dimensions: int | None = None) -> EmbeddingOutputSpec:
        """Describe results for the requested size without model or network I/O.

        A provided size overrides the default dimension lookup, as in Embed.
        Named-output descriptors reject it: each output owns its own dimension.
        """
        if dimensions is None:
            try:
                dimensions = self.get_dimensions()
            except Exception as exc:
                from vane.ai._errors import _raise_if_resource_error

                _raise_if_resource_error(exc)
                raise ValueError(
                    f"Cannot determine embedding dimensions for model {self.get_model()!r} without network or model loading; "
                    "pass dimensions=... explicitly"
                ) from exc
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("Provider get_dimensions() must return a positive integer")
        return EmbeddingOutputSpec((EmbeddingOutput("embedding", dimensions, self.get_model()),))
