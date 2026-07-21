# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers provider for Vane AI.

Supports text embedding via ``sentence-transformers`` and text
classification via ``transformers`` zero-shot-classification pipelines.

Requires::

    pip install 'vane-ai[transformers]'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai.protocols import TextClassifierDescriptor, TextEmbedderDescriptor
from vane.ai.provider import Provider
from vane.ai.typing import EmbeddingDimensions, UDFOptions

if TYPE_CHECKING:
    from vane.ai.protocols import TextClassifier, TextEmbedder
    from vane.ai.typing import Embedding, Label, Options


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class TransformersProvider(Provider):
    """Provider backed by HuggingFace Transformers / SentenceTransformers."""

    DEFAULT_TEXT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_TEXT_CLASSIFIER = "facebook/bart-large-mnli"

    def __init__(self, name: str | None = None, **options: Any):
        self._name = name or "transformers"
        self._options: dict[str, Any] = options

    @property
    def name(self) -> str:
        return self._name

    def get_text_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        **options: Any,
    ) -> TextEmbedderDescriptor:
        return TransformersTextEmbedderDescriptor(
            model=model or self.DEFAULT_TEXT_EMBEDDER,
            dimensions=dimensions,
            embed_options={**self._options, **options},
        )

    def get_text_classifier(self, model: str | None = None, **options: Any) -> TextClassifierDescriptor:
        return TransformersTextClassifierDescriptor(
            model=model or self.DEFAULT_TEXT_CLASSIFIER,
            classify_options={**self._options, **options},
        )


# ---------------------------------------------------------------------------
# Text Embedding
# ---------------------------------------------------------------------------


@dataclass
class TransformersTextEmbedderDescriptor(TextEmbedderDescriptor):
    """Serializable factory for a SentenceTransformer-based text embedder."""

    model: str
    dimensions: int | None = None
    embed_options: dict[str, Any] = field(default_factory=lambda: {"batch_size": 64})

    def __post_init__(self) -> None:
        self.embed_options = wrap_sensitive_options(self.embed_options)

    def get_provider(self) -> str:
        return "transformers"

    def get_model(self) -> str:
        return self.model

    def get_options(self) -> Options:
        return dict(self.embed_options)

    def get_dimensions(self) -> EmbeddingDimensions:
        from transformers import AutoConfig

        if self.dimensions is not None:
            return EmbeddingDimensions(size=self.dimensions, dtype=pa.float32())
        config_options: dict[str, Any] = {
            "trust_remote_code": self.embed_options.get("trust_remote_code") is True,
        }
        for name in ("local_files_only", "revision", "token"):
            if name in self.embed_options:
                config_options[name] = self.embed_options[name]
        if "cache_folder" in self.embed_options:
            config_options["cache_dir"] = self.embed_options["cache_folder"]
        # Hub access is an execution boundary: restore the plaintext token.
        config_options = unwrap_sensitive_options(config_options)
        hidden = AutoConfig.from_pretrained(self.model, **config_options).hidden_size
        return EmbeddingDimensions(size=hidden, dtype=pa.float32())

    def get_udf_options(self) -> UDFOptions:
        import torch

        has_gpu = torch.cuda.is_available()
        opts = UDFOptions(
            batch_size=self.embed_options.get("batch_size", 64),
            max_retries=self.embed_options.get("max_retries", 3),
            on_error=self.embed_options.get("on_error", "raise"),
        )
        if has_gpu:
            opts.num_gpus = 1
        return opts

    def instantiate(self) -> TextEmbedder:
        model_options = {
            name: value
            for name, value in self.embed_options.items()
            if name in {"cache_folder", "device", "local_files_only", "revision", "token", "trust_remote_code"}
        }
        return TransformersTextEmbedder(
            self.model,
            dimensions=self.dimensions,
            **model_options,
        )


class TransformersTextEmbedder:
    """Concrete text embedder using ``sentence-transformers``."""

    def __init__(
        self,
        model_name_or_path: str,
        dimensions: int | None = None,
        **model_options: Any,
    ):
        from sentence_transformers import SentenceTransformer

        # Restore plaintext credentials sealed by the descriptor; plain dicts
        # from direct callers pass through unchanged.
        model_options = unwrap_sensitive_options(model_options)
        trust_remote_code = model_options.pop("trust_remote_code", False) is True
        self.model = SentenceTransformer(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            backend="torch",
            **model_options,
        )
        self.model.eval()
        self.dimensions = dimensions

    def embed_text(self, text: list[str]) -> list[Embedding]:
        import torch

        with torch.inference_mode():
            batch = self.model.encode(text, convert_to_numpy=True, truncate_dim=self.dimensions)
            return list(batch)


# ---------------------------------------------------------------------------
# Text Classification
# ---------------------------------------------------------------------------


@dataclass
class TransformersTextClassifierDescriptor(TextClassifierDescriptor):
    """Serializable factory for a Transformers zero-shot classifier."""

    model: str
    classify_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.classify_options = wrap_sensitive_options(self.classify_options)

    def get_provider(self) -> str:
        return "transformers"

    def get_model(self) -> str:
        return self.model

    def get_options(self) -> Options:
        return dict(self.classify_options)

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(
            batch_size=self.classify_options.get("batch_size"),
            max_retries=self.classify_options.get("max_retries", 3),
            on_error=self.classify_options.get("on_error", "raise"),
        )

    def instantiate(self) -> TextClassifier:
        pipeline_options = {
            k: v
            for k, v in self.classify_options.items()
            if k
            not in {
                "batch_size",
                "max_retries",
                "on_error",
                "actor_number",
                "num_gpus",
            }
        }
        return TransformersTextClassifier(self.model, **pipeline_options)


class TransformersTextClassifier:
    """Concrete text classifier using ``transformers`` zero-shot pipeline."""

    def __init__(self, model_name: str, **options: Any):
        from transformers import pipeline

        # Restore plaintext credentials sealed by the descriptor; plain dicts
        # from direct callers pass through unchanged.
        options = unwrap_sensitive_options(options)
        options["trust_remote_code"] = options.get("trust_remote_code") is True
        self.pipeline = pipeline(
            "zero-shot-classification",
            model=model_name,
            **options,
        )

    def classify_text(self, text: list[str], labels: Label | list[Label]) -> list[Label]:
        if isinstance(labels, str):
            labels = [labels]
        results = self.pipeline(text, candidate_labels=labels)
        if not isinstance(results, list):
            results = [results]
        return [r["labels"][0] for r in results]
