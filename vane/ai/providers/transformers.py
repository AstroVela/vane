# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers provider for Vane AI text and image embedding.

Requires::

    pip install 'vane-ai[transformers]'
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from vane.ai._embedding_inputs import EmbeddingConfigurationError, split_text
from vane.ai._redaction import unwrap_sensitive_options, wrap_sensitive_options
from vane.ai.options import validate_embed_image_options, validate_embed_options
from vane.ai.protocols import ImageEmbedderDescriptor, TextEmbedderDescriptor, VideoEmbedderDescriptor
from vane.ai.provider import (
    Provider,
    ProviderCapabilityError,
    _translate_missing_provider_dependency,
)
from vane.ai.typing import UDFOptions

if TYPE_CHECKING:
    from vane._image import Image
    from vane.ai.protocols import ImageEmbedder, TextEmbedder
    from vane.ai.typing import Embedding, Options


# A shared identity/dimension table for the paired CLIP text/image encoders.
_IMAGE_EMBEDDING_DIMS = {"sentence-transformers/clip-ViT-B-32": 512, "clip-ViT-B-32": 512}
_EMBEDDING_DIMS = {"sentence-transformers/all-MiniLM-L6-v2": 384, **_IMAGE_EMBEDDING_DIMS}
_MODEL_OPTIONS = frozenset({"cache_folder", "device", "local_files_only", "revision", "trust_remote_code"})
_ENCODING_OPTIONS = frozenset({"input_type", "prompt_name", "prompt", "overlength", "max_concurrency_per_actor"})
_EMBED_OPTIONS = _MODEL_OPTIONS | _ENCODING_OPTIONS


def _effective_text_limit(module: Any, tokenizer: Any, task: str | None) -> int:
    """Resolve the plain-text preprocessing budget, or reject unknown rendering."""
    # A processor or chat template can add tokens that tokenizer.encode does
    # not see. Query expansion also overrides length/padding after kwargs merge.
    if (
        getattr(module, "processor", tokenizer) is not tokenizer
        or "message" in getattr(module, "modality_config", {})
        or (task == "query" and getattr(module, "query_expansion", None) is not None)
        or getattr(module, "do_lower_case", False)
    ):
        raise ValueError("Cannot establish the preprocessing token budget")
    default_limit = getattr(module, "max_seq_length", None)
    limit = default_limit
    if task in {"query", "document"}:
        task_limit = getattr(module, f"{task}_length", None)
        if task_limit is not None:
            limit = task_limit

    processing = getattr(module, "processing_kwargs", {})
    if not isinstance(processing, Mapping) or processing.get("chat_template"):
        raise ValueError("Cannot establish the preprocessing token budget")
    allowed = {
        "max_length",
        "padding",
        "truncation",
        "return_tensors",
        "return_attention_mask",
        "return_token_type_ids",
    }
    # For a bare text tokenizer, common kwargs are applied after text kwargs.
    # An explicit None resets max_length to the tokenizer's default budget.
    for key in ("text", "common"):
        overrides = processing.get(key, {})
        if not isinstance(overrides, Mapping) or set(overrides) - allowed:
            raise ValueError("Cannot establish the preprocessing token budget")
        if "max_length" in overrides:
            limit = default_limit if overrides["max_length"] is None else overrides["max_length"]
    if type(limit) is not int or limit <= 0:
        raise ValueError("Cannot establish the preprocessing token budget")
    return limit


def _strips_input_whitespace(module: Any) -> bool:
    """Match the selected module's known text preprocessing implementation."""
    preprocess = getattr(module, "preprocess", None)
    if callable(preprocess):
        method = preprocess
        expected = ("sentence_transformers.base.modules.transformer", "Transformer.preprocess")
        strips = False
    else:
        method = getattr(module, "tokenize", None)
        if not callable(method):
            raise ValueError("Cannot establish the text preprocessing behavior")
        expected = ("sentence_transformers.models.Transformer", "Transformer.tokenize")
        strips = True
    # Custom overrides may change text before tokenizing. Metadata alone cannot
    # establish their token counts, so explicit policies must not guess.
    if (getattr(method, "__module__", None), getattr(method, "__qualname__", None)) != expected:
        raise ValueError("Cannot establish the text preprocessing behavior")
    return strips


def _resolve_token_metadata(model: Any, task: str | None) -> tuple[Any, int, bool]:
    """Follow the text input path instead of Router's aggregate metadata."""
    seen: set[int] = set()
    task_is_forwarded = True
    try:
        module = model._first_module()
        while id(module) not in seen:
            seen.add(id(module))
            first_module = getattr(module, "_first_module", None)
            if callable(first_module):
                module = first_module()
                continue
            routes = getattr(module, "sub_modules", None)
            if routes is not None:
                if not task_is_forwarded:
                    break
                resolve_route = getattr(module, "_resolve_route", None)
                if callable(resolve_route):
                    # Let newer Routers apply task/modality mapping priority.
                    route = resolve_route(task=task, modality="text")
                elif not hasattr(module, "route_mappings"):
                    # Legacy Routers dispatch directly by task or default_route.
                    route = task if task is not None else module.default_route
                    # Their tokenize() consumes task without forwarding it.
                    # Further routing cannot safely mirror the forward path.
                    task_is_forwarded = False
                    task = None
                else:
                    break
                module = routes[route][0]
                continue
            tokenizer = getattr(module, "tokenizer", None)
            if callable(getattr(tokenizer, "encode", None)):
                return tokenizer, _effective_text_limit(module, tokenizer, task), _strips_input_whitespace(module)
            break
    except (AttributeError, LookupError, TypeError, ValueError, NotImplementedError):
        pass
    # Configuration errors must survive on_error="ignore", without retaining
    # an exception from model-defined metadata or route resolution.
    raise EmbeddingConfigurationError(
        "Explicit overlength requires the selected input route's tokenizer and effective preprocessing token budget"
    ) from None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class TransformersProvider(Provider):
    """Provider backed by HuggingFace Transformers / SentenceTransformers."""

    DEFAULT_IMAGE_EMBEDDER = "sentence-transformers/clip-ViT-B-32"
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
        from vane.ai.providers._cosmos_embed1 import CosmosTextEmbedderDescriptor

        resolved_options = dict(options or {})
        if model is not None and model.startswith("nvidia/Cosmos-Embed1"):
            return CosmosTextEmbedderDescriptor(model, dimensions, resolved_options, self._name)
        return TransformersTextEmbedderDescriptor(
            model=model or self.DEFAULT_TEXT_EMBEDDER,
            provider_name=self._name,
            dimensions=dimensions,
            options=resolved_options,
        )

    def get_image_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> ImageEmbedderDescriptor:
        return TransformersImageEmbedderDescriptor(
            model=model or self.DEFAULT_IMAGE_EMBEDDER,
            dimensions=dimensions,
            options=dict(options or {}),
            provider_name=self._name,
        )

    def get_video_embedder(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> VideoEmbedderDescriptor:
        from vane.ai.providers._cosmos_embed1 import COSMOS_MODEL, CosmosVideoEmbedderDescriptor

        if model != COSMOS_MODEL:
            raise EmbeddingConfigurationError(f"Transformers video embedding requires model={COSMOS_MODEL!r}")
        return CosmosVideoEmbedderDescriptor(model, dimensions, dict(options or {}), self._name)


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
        model_options = {name: value for name, value in self.options.items() if name in _EMBED_OPTIONS}
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
        with _translate_missing_provider_dependency("transformers", "sentence_transformers"):
            from sentence_transformers import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                SentenceTransformer,
            )

        # Restore plaintext credentials sealed by the descriptor; plain dicts
        # from direct callers pass through unchanged.
        model_options = unwrap_sensitive_options(model_options)
        encoding_options = {name: model_options.pop(name) for name in _ENCODING_OPTIONS if name in model_options}
        validate_embed_options("transformers", encoding_options, relation=False)
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
        self._overlength = encoding_options.get("overlength")
        self._encode_options: dict[str, Any] = {}
        self._encode: Any = getattr(self.model, "encode", None)
        prompts = getattr(self.model, "prompts", {})
        prompt_name = encoding_options.get("prompt_name")
        input_type = encoding_options.get("input_type")
        encoding_task = None
        if input_type is not None:
            candidates = ("query",) if input_type == "query" else ("document", "passage", "corpus")
            prompt_name = next((name for name in candidates if name in prompts), None)
            if prompt_name is None:
                raise EmbeddingConfigurationError("Selected model has no declared template for input_type")
            task_encoder = getattr(self.model, f"encode_{input_type}", None)
            if callable(task_encoder):
                self._encode = task_encoder
                encoding_task = input_type
        if prompt_name is not None:
            if prompt_name not in prompts:
                raise EmbeddingConfigurationError("Selected model does not define the requested prompt_name")
            self._encode_options["prompt_name"] = prompt_name
        if "prompt" in encoding_options:
            self._encode_options["prompt"] = encoding_options["prompt"]
        self._prefix = encoding_options.get(
            "prompt", prompts.get(prompt_name or getattr(self.model, "default_prompt_name", None), "")
        )
        if self._overlength is not None:
            self._tokenizer, self._token_limit, self._strip_whitespace = _resolve_token_metadata(
                self.model, encoding_task
            )
            if not isinstance(self._prefix, str):
                raise EmbeddingConfigurationError("Explicit overlength requires a text prompt")
            if self._count_tokens("") >= self._token_limit:
                raise EmbeddingConfigurationError("Embedding prompt leaves no input token budget")

    def _count_tokens(self, text: str) -> int:
        return self._text_token_count(self._prefix + text, add_special_tokens=True)

    def _text_token_count(self, text: str, *, add_special_tokens: bool) -> int:
        # Legacy Transformer.tokenize strips after SentenceTransformer adds the
        # prompt. BPE token counts can increase when leading whitespace is removed.
        if self._strip_whitespace:
            text = text.strip()
        return len(self._tokenizer.encode(text, add_special_tokens=add_special_tokens, truncation=False))

    def embed_text(self, text: list[str]) -> list[Embedding]:
        with _translate_missing_provider_dependency("transformers", "torch"):
            import torch  # type: ignore[import-not-found, import-untyped, unused-ignore]

        capability_error: ProviderCapabilityError | None = None
        with torch.inference_mode():
            try:
                if self._overlength is None:
                    batch = self._encode(
                        text, convert_to_numpy=True, truncate_dim=self.dimensions, **self._encode_options
                    )
                else:
                    chunks: list[str] = []
                    rows: list[list[int]] = []
                    for item in text:
                        if self._count_tokens(item) <= self._token_limit:
                            pieces = [item]
                        elif self._overlength == "error":
                            raise ValueError("Embedding input exceeds model max_seq_length")
                        else:
                            pieces = split_text(
                                item, self._token_limit, self._count_tokens, first_only=self._overlength == "truncate"
                            )
                        rows.append(list(range(len(chunks), len(chunks) + len(pieces))))
                        chunks.extend(pieces)
                    vectors = self._encode(
                        chunks, convert_to_numpy=True, truncate_dim=self.dimensions, **self._encode_options
                    )
                    if len(vectors) != len(chunks):
                        from vane.ai.provider import _ProviderResultError

                        raise _ProviderResultError("Embedding encoding must preserve input row count")
                    batch = [
                        vectors[indices[0]]
                        if len(indices) == 1
                        else np.average(
                            np.asarray([vectors[i] for i in indices], dtype=np.float64),
                            axis=0,
                            weights=[
                                max(1, self._text_token_count(chunks[i], add_special_tokens=False)) for i in indices
                            ],
                        )
                        for indices in rows
                    ]
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


@dataclass
class TransformersImageEmbedderDescriptor(ImageEmbedderDescriptor):
    """Declare a paired CLIP image encoder without importing or loading models."""

    model: str
    dimensions: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    provider_name: str = "transformers"

    def __post_init__(self) -> None:
        if self.model not in _IMAGE_EMBEDDING_DIMS:
            raise ValueError(
                "Transformers image embedding currently supports sentence-transformers/clip-ViT-B-32; "
                "text-only or undeclared image models are not supported"
            )
        unknown = set(self.options) - _MODEL_OPTIONS
        if unknown:
            raise TypeError("Unsupported Transformers EmbedImage option(s): " + ", ".join(sorted(unknown)))
        validated = validate_embed_image_options("transformers", self.options, relation=False)
        # Reuse the text encoder's dimension, loading, and GPU resource rules.
        self._text_descriptor = TransformersTextEmbedderDescriptor(
            self.model, self.dimensions, validated, self.provider_name
        )
        self.options = self._text_descriptor.get_options()

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model

    def get_options(self) -> Options:
        return dict(self.options)

    def get_dimensions(self) -> int:
        return self._text_descriptor.get_dimensions()

    def get_udf_options(self) -> UDFOptions:
        return self._text_descriptor.get_udf_options()

    def instantiate(self) -> ImageEmbedder:
        return TransformersImageEmbedder(
            self.model, dimensions=self.dimensions, provider_name=self.provider_name, **self.options
        )


class TransformersImageEmbedder(TransformersTextEmbedder):
    """Use the same CLIP weights and processor as the paired text embedder."""

    def __init__(self, model_name_or_path: str, **options: Any) -> None:
        super().__init__(model_name_or_path, **options)
        from sentence_transformers.models import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
            CLIPModel,
        )

        if not isinstance(self.model._first_module(), CLIPModel):
            raise EmbeddingConfigurationError("Selected image model must use the SentenceTransformers CLIP module")

    def embed_image(self, images: list[Image]) -> list[Embedding]:
        import torch
        from PIL import Image as PILImage  # type: ignore[import-not-found, import-untyped, unused-ignore]

        prepared: list[Any] = []
        try:
            for pixels in images:
                if pixels.dtype != np.uint8:
                    raise ValueError("CLIP requires UInt8 images; use convert_image(image, 'RGB') explicitly")
                # IMAGE has HWC layout; Pillow's grayscale constructor expects HW.
                data = pixels[:, :, 0] if pixels.shape[2] == 1 else pixels
                with PILImage.fromarray(data) as original:
                    prepared.append(original.convert("RGB"))
            with torch.inference_mode():
                batch = self.model.encode(
                    prepared, convert_to_numpy=True, truncate_dim=self.dimensions, show_progress_bar=False
                )
            return [np.asarray(row, dtype=np.float32) for row in batch]
        finally:
            for image in prepared:
                image.close()
