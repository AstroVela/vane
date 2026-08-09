# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers provider for Vane AI text embedding.

Requires::

    pip install 'vane-ai[transformers]'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai.options import validate_embed_options
from vane.ai.protocols import TextEmbedderDescriptor
from vane.ai.provider import Provider, ProviderCapabilityError
from vane.ai.typing import UDFOptions

if TYPE_CHECKING:
    from collections.abc import Mapping

    from vane.ai.protocols import TextEmbedder
    from vane.ai.typing import Embedding, Options


_EMBEDDING_DIMS = {"sentence-transformers/all-MiniLM-L6-v2": 384}
_EMBED_OPTIONS = frozenset({"cache_folder", "device", "local_files_only", "revision", "trust_remote_code"})


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class TransformersProvider(Provider):
    """Provider backed by HuggingFace Transformers / SentenceTransformers."""

    DEFAULT_TEXT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, name: str | None = None):
        self._name = name or "transformers"

    @property
    def name(self) -> str:
        return self._name

    def get_text_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> TextEmbedderDescriptor:
        resolved_options = dict(options or {})
        return TransformersTextEmbedderDescriptor(
            model=model or self.DEFAULT_TEXT_EMBEDDER,
            provider_name=self._name,
            dimensions=dimensions,
            options=resolved_options,
        )


# ---------------------------------------------------------------------------
# Text Embedding
# ---------------------------------------------------------------------------


@dataclass
class TransformersTextEmbedderDescriptor(TextEmbedderDescriptor):
    """Serializable factory for a SentenceTransformer-based text embedder."""

    model: str
    dimensions: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    provider_name: str = "transformers"

    def __post_init__(self) -> None:
        unknown = sorted(set(self.options) - _EMBED_OPTIONS)
        if unknown:
            raise TypeError(f"Unsupported Transformers Embed option(s): {', '.join(unknown)}")
        validated_options = validate_embed_options("transformers", self.options, relation=False)
        if self.dimensions is not None and (
            isinstance(self.dimensions, bool) or not isinstance(self.dimensions, int) or self.dimensions <= 0
        ):
            raise ValueError("Embedding dimensions must be a positive integer")
        native_dimensions = _EMBEDDING_DIMS.get(self.model)
        if self.dimensions is not None and native_dimensions is not None and self.dimensions > native_dimensions:
            raise ValueError(
                f"Transformers model {self.model!r} has {native_dimensions} dimensions and cannot produce "
                f"{self.dimensions} dimensions"
            )
        resolved_options = validated_options
        if resolved_options.get("device") is None:
            resolved_options["device"] = "cpu"
        self.options = wrap_sensitive_options(resolved_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model

    def get_options(self) -> Options:
        return dict(self.options)

    def get_dimensions(self) -> int:
        if self.dimensions is not None:
            return self.dimensions
        if self.model in _EMBEDDING_DIMS:
            return _EMBEDDING_DIMS[self.model]
        raise ValueError(
            f"Cannot determine embedding dimensions for Transformers model {self.model!r} "
            "from trusted local metadata; pass dimensions=... explicitly"
        )

    def get_udf_options(self) -> UDFOptions:
        has_gpu = str(self.options["device"]).startswith("cuda")
        return UDFOptions(num_gpus=1 if has_gpu else 0)

    def instantiate(self) -> TextEmbedder:
        model_options = {
            name: value
            for name, value in self.options.items()
            if name in {"cache_folder", "device", "local_files_only", "revision", "trust_remote_code"}
        }
        return TransformersTextEmbedder(
            self.model,
            dimensions=self.dimensions,
            provider_name=self.provider_name,
            **model_options,
        )


class TransformersTextEmbedder:
    """Concrete text embedder using ``sentence-transformers``."""

    def __init__(
        self,
        model_name_or_path: str,
        dimensions: int | None = None,
        provider_name: str = "transformers",
        **model_options: Any,
    ):
        from sentence_transformers import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
            SentenceTransformer,
        )

        # Restore plaintext credentials sealed by the descriptor; plain dicts
        # from direct callers pass through unchanged.
        model_options = unwrap_sensitive_options(model_options)
        if model_options.get("device") is None:
            model_options["device"] = "cpu"
        trust_remote_code = model_options.pop("trust_remote_code", False) is True
        self._provider_name = provider_name
        self._model_name = model_name_or_path
        capability_error: ProviderCapabilityError | None = None
        try:
            self.model = SentenceTransformer(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                backend="torch",
                **model_options,
            )
        except NotImplementedError as exc:
            capability_error = ProviderCapabilityError(
                getattr(self, "_provider_name", "transformers"),
                model_name_or_path,
                "embedding model",
                original_error=exc,
            )
        if capability_error is not None:
            raise capability_error from None
        self.model.eval()
        self.dimensions = dimensions

    def embed_text(self, text: list[str]) -> list[Embedding]:
        import torch  # type: ignore[import-not-found, import-untyped, unused-ignore]

        capability_error: ProviderCapabilityError | None = None
        with torch.inference_mode():
            try:
                batch = self.model.encode(text, convert_to_numpy=True, truncate_dim=self.dimensions)
            except NotImplementedError as exc:
                capability_error = ProviderCapabilityError(
                    getattr(self, "_provider_name", "transformers"),
                    self._model_name,
                    "embedding model",
                    original_error=exc,
                )
        if capability_error is not None:
            raise capability_error from None
        return list(batch)
