# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers provider for Vane AI text embedding.

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
from vane.ai.options import validate_embed_options
from vane.ai.protocols import TextEmbedderDescriptor
from vane.ai.provider import (
    Provider,
    ProviderCapabilityError,
    _translate_missing_provider_dependency,
)
from vane.ai.typing import UDFOptions

if TYPE_CHECKING:
    from vane.ai.protocols import TextEmbedder
    from vane.ai.typing import Embedding, Options


_EMBEDDING_DIMS = {"sentence-transformers/all-MiniLM-L6-v2": 384}
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


def _resolve_token_metadata(model: Any, task: str | None) -> tuple[Any, int]:
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
                return tokenizer, _effective_text_limit(module, tokenizer, task)
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
            self._tokenizer, self._token_limit = _resolve_token_metadata(self.model, encoding_task)
            if not isinstance(self._prefix, str):
                raise EmbeddingConfigurationError("Explicit overlength requires a text prompt")
            if self._count_tokens("") >= self._token_limit:
                raise EmbeddingConfigurationError("Embedding prompt leaves no input token budget")

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(self._prefix + text, add_special_tokens=True, truncation=False))

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
                                max(1, len(self._tokenizer.encode(chunks[i], add_special_tokens=False)))
                                for i in indices
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
