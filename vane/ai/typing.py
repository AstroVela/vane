# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Core type definitions for the Vane AI module."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar

import pyarrow as pa

if TYPE_CHECKING:
    import numpy as np

    Embedding: TypeAlias = np.typing.NDArray[Any]
else:
    Embedding: TypeAlias = Any

Options = dict[str, Any]
Label = str

T = TypeVar("T")


def actor_number_from_options(options: Mapping[str, Any]) -> int | None:
    """Resolve UDF ``actor_number`` from execution options.

    Accepts ``concurrency`` as the public alias for ``actor_number`` —
    mirroring the SQL layer and the typed options objects — so the bare
    Python kwarg is not silently dropped. An explicit ``actor_number`` wins.
    """
    name = "actor_number"
    value = options.get("actor_number")
    if value is None:
        name = "concurrency"
        value = options.get("concurrency")
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        # Mirror the SQL layer's validation (_int_or_none) so both surfaces
        # reject the same values instead of silently misconfiguring workers.
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def api_worker_options(options: Mapping[str, Any], *, default_batch_size: int | None = None) -> dict[str, Any]:
    """Shared execution-option reads for pure-HTTP provider descriptors.

    Pure HTTP providers need no GPU unless one is explicitly declared,
    honour an explicit ``batch_size``, and accept ``concurrency`` as the
    public alias for ``actor_number``. Returns keyword arguments for
    :class:`UDFOptions`.
    """
    batch_size = options.get("batch_size", default_batch_size)
    if batch_size is not None:
        batch_size = int(batch_size)
        if batch_size <= 0:
            # Mirror the SQL layer's validation (_int_or_none) so both surfaces
            # reject the same values instead of silently falling back to the
            # downstream default batch size.
            raise ValueError("batch_size must be a positive integer")
    return {
        "batch_size": batch_size,
        "actor_number": actor_number_from_options(options),
        "num_gpus": options.get("num_gpus", 0),
    }


class Descriptor(ABC, Generic[T]):
    """A serializable factory that can instantiate a model on a remote worker.

    Descriptors are lightweight and picklable. They carry only the
    configuration needed to reconstruct a model instance. The heavy
    ``instantiate()`` call happens lazily on the worker that actually
    runs inference, ensuring models are loaded exactly once per actor.
    """

    @abstractmethod
    def get_provider(self) -> str:
        """Return the name of the provider that created this descriptor."""
        ...

    @abstractmethod
    def get_model(self) -> str:
        """Return the model identifier (e.g. HuggingFace repo id)."""
        ...

    @abstractmethod
    def get_options(self) -> Options:
        """Return provider-specific instantiation options."""
        ...

    @abstractmethod
    def instantiate(self) -> T:
        """Create and return the concrete model instance.

        This is called on the worker side after deserialization.
        """
        ...

    def get_udf_options(self) -> UDFOptions:
        """Extract UDF execution options from the provider options."""
        opts = self.get_options()
        return UDFOptions(
            actor_number=opts.get("actor_number"),
            num_gpus=opts.get("num_gpus"),
            max_retries=opts.get("max_retries", 3),
            on_error=opts.get("on_error", "raise"),
            batch_size=opts.get("batch_size"),
        )


@dataclass(frozen=True)
class EmbeddingDimensions:
    """Describes the shape and dtype of an embedding vector."""

    size: int
    dtype: pa.DataType = pa.float32()

    def as_arrow_type(self) -> pa.DataType:
        return pa.list_(self.dtype, self.size)


@dataclass
class UDFOptions:
    """Execution options for AI UDFs."""

    actor_number: int | None = None
    num_gpus: int | None = None
    max_retries: int = 3
    on_error: Literal["raise", "log", "ignore"] = "raise"
    batch_size: int | None = None
    max_api_concurrency: int | None = None
