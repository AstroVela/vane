# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Paired CLAP audio/text encoders, loaded only in the executing worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vane.ai._audio_embedding import AudioClip, AudioInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.options import validate_embed_options
from vane.ai.protocols import AudioEmbedderDescriptor, TextEmbedderDescriptor
from vane.ai.provider import _translate_missing_provider_dependency
from vane.ai.typing import UDFOptions

CLAP_MODEL = "laion/clap-htsat-unfused"
_MODEL_OPTIONS = frozenset({"cache_folder", "device", "local_files_only", "revision", "trust_remote_code"})


@dataclass
class _ClapDescriptor:
    model: str
    dimensions: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    provider_name: str = "transformers"

    def __post_init__(self) -> None:
        if self.model != CLAP_MODEL:
            raise EmbeddingConfigurationError(f"CLAP embedding supports only {CLAP_MODEL}")
        if self.dimensions is not None and (type(self.dimensions) is not int or self.dimensions != 512):
            raise EmbeddingConfigurationError("CLAP produces exactly 512 dimensions")
        unknown = sorted(set(self.options) - _MODEL_OPTIONS)
        if unknown:
            raise TypeError(f"Unsupported CLAP embedding option(s): {', '.join(unknown)}")
        self.options = validate_embed_options("transformers", self.options, relation=False)
        if self.options.get("trust_remote_code"):
            raise EmbeddingConfigurationError("CLAP uses built-in Transformers code; trust_remote_code is unsupported")
        self.options["device"] = self.options.get("device") or "cpu"
        if self.options["device"] not in {"cpu", "cuda", "cuda:0"}:
            raise EmbeddingConfigurationError("CLAP device must be cpu or cuda (the worker's allocated GPU)")

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model

    def get_options(self) -> dict[str, Any]:
        return dict(self.options)

    def get_dimensions(self) -> int:
        return 512

    def get_udf_options(self) -> UDFOptions:
        return UDFOptions(num_gpus=1 if self.options["device"].startswith("cuda") else 0)

    def instantiate(self) -> ClapEmbedder:
        return ClapEmbedder(self.model, self.options)


class ClapTextEmbedderDescriptor(_ClapDescriptor, TextEmbedderDescriptor):
    def supports_chunking(self) -> bool:
        return False


class ClapAudioEmbedderDescriptor(_ClapDescriptor, AudioEmbedderDescriptor):
    def get_input_spec(self) -> AudioInputSpec:
        return AudioInputSpec(sample_rate=48000, max_samples=480000)


class ClapEmbedder:
    def __init__(self, model: str, options: dict[str, Any]) -> None:
        with (
            _translate_missing_provider_dependency("clap", "torch"),
            _translate_missing_provider_dependency("clap", "transformers"),
        ):
            import torch
            from transformers import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
                ClapModel,
                ClapProcessor,
            )

            self._torch, self._device = torch, options["device"]
            if self._device.startswith("cuda") and not torch.cuda.is_available():
                raise EmbeddingConfigurationError("CLAP requires CUDA in the executing worker for device=cuda")
            loading: dict[str, Any] = {key: options[key] for key in ("revision", "local_files_only") if key in options}
            if options.get("cache_folder") is not None:
                loading["cache_dir"] = options["cache_folder"]
            self._processor = ClapProcessor.from_pretrained(model, **loading)
            feature = self._processor.feature_extractor
            if feature.sampling_rate != 48000 or feature.nb_max_samples != 480000:
                raise EmbeddingConfigurationError("CLAP processor must use 48 kHz and ten-second windows")
            self._model = ClapModel.from_pretrained(model, **loading)
            if self._model.config.projection_dim != 512 or self._model.config.audio_config.enable_fusion:
                raise EmbeddingConfigurationError("CLAP model must use the unfused 512-dimensional contract")
            self._model.to(self._device).eval()

    def embed_text(self, texts: list[str]) -> list[np.ndarray]:
        # The released checkpoint is trained with 77 tokens, including special
        # tokens. Never silently truncate or average unrelated sound queries.
        for text in texts:
            if len(self._processor.tokenizer.encode(text, add_special_tokens=True, truncation=False)) > 77:
                raise EmbeddingConfigurationError("CLAP text exceeds 77 tokens including special tokens")
        with self._torch.inference_mode():
            inputs = self._processor(text=texts, padding=True, truncation=False, return_tensors="pt").to(self._device)
            return list(self._model.get_text_features(**inputs).float().cpu().numpy())

    def embed_audio(self, clips: list[AudioClip]) -> list[np.ndarray]:
        waveforms = []
        for clip in clips:
            samples = clip.samples
            if (
                clip.sample_rate != 48000
                or samples.ndim != 2
                or not 1 <= samples.shape[0] <= 480000
                or not 1 <= samples.shape[1] <= 2
                or not np.isfinite(samples).all()
                or np.any(np.abs(samples) > 1)
            ):
                raise EmbeddingConfigurationError(
                    "CLAP requires finite 48 kHz PCM, 1–2 channels, and at most ten seconds"
                )
            # Explicit equal-weight downmix. Short final windows use the
            # checkpoint's repeat-padding policy; long windows are rejected.
            waveforms.append(samples.mean(axis=1, dtype=np.float32))
        with self._torch.inference_mode():
            inputs = self._processor(
                audios=waveforms, sampling_rate=48000, padding="repeatpad", truncation="rand_trunc", return_tensors="pt"
            ).to(self._device)
            return list(self._model.get_audio_features(**inputs).float().cpu().numpy())
