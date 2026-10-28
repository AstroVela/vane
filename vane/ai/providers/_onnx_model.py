# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""One owned set of ONNX sessions and the official local-only processor."""

from __future__ import annotations

import json
import math
import threading
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from vane.ai._audio_embedding import AudioClip
from vane.ai._embedding_content import EmbeddingContent
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip
from vane.ai.provider import _ProviderResultError, _translate_missing_provider_dependency
from vane.ai.providers._onnx_artifacts import bundle_sha256, validate_external_data

DOCUMENT_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: search result | query: "
MAX_IMAGE_PIXELS = 16 * 1024**2
MAX_BATCH_BYTES = 64 * 1024**2
MEDIA_TOKENS = {"image": "<|image|>", "audio": "<|audio|>", "video": "<|video|>"}
EXECUTION_PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}

PROCESSOR_VERSIONS = {"transformers": "5.21.0", "torch": "2.14.1", "torchvision": "0.29.1", "Pillow": "12.3.0"}


def processor_versions() -> dict[str, str]:
    return {name: version(name) for name in PROCESSOR_VERSIONS}


def validate_software_contract(ort: Any) -> None:
    if ort.__version__ != "1.26.0" or any(
        value.split("+", 1)[0] != PROCESSOR_VERSIONS[name] for name, value in processor_versions().items()
    ):
        raise EmbeddingConfigurationError("ONNX requires runtime 1.26.0 and the declared onnx-py1 processor versions")


def validate_parts(rows: list[list[tuple[str, Any]]], max_input_bytes: int) -> None:
    if not isinstance(rows, list) or len(rows) > 256:
        raise EmbeddingConfigurationError("ONNX requires at most 256 input rows per batch")
    batch_size = 0
    for row in rows:
        if not isinstance(row, list) or not 1 <= len(row) <= 16:
            raise EmbeddingConfigurationError("ONNX requires 1 to 16 decoded input parts per row")
        size = 0
        for kind, value in row:
            if kind == "text":
                if not isinstance(value, str):
                    raise EmbeddingConfigurationError("ONNX text requires a string")
                try:
                    size += len(value.encode("utf-8"))
                except UnicodeError:
                    raise EmbeddingConfigurationError("ONNX text requires valid UTF-8") from None
                if any(token in value for token in (*MEDIA_TOKENS.values(), "<|image_start|>", "<|audio_start|>")):
                    raise EmbeddingConfigurationError("ONNX media placeholders must come from typed input parts")
            elif kind in {"image", "video"}:
                if kind == "video":
                    if not isinstance(value, VideoClip) or not 1 <= len(value.frames) <= 32:
                        raise EmbeddingConfigurationError("ONNX video requires 1 to 32 decoded frames")
                    if (
                        len(value.frame_times) != len(value.frames)
                        or len(value.frame_indices) != len(value.frames)
                        or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in value.frame_times)
                        or any(a > b for a, b in zip(value.frame_times, value.frame_times[1:]))
                        or any(type(index) is not int or index < 0 for index in value.frame_indices)
                        or any(a >= b for a, b in zip(value.frame_indices, value.frame_indices[1:]))
                    ):
                        raise EmbeddingConfigurationError("ONNX video frame metadata must be aligned and ordered")
                    frames = value.frames
                else:
                    frames = (value,)
                for frame in frames:
                    if (
                        not isinstance(frame, np.ndarray)
                        or frame.dtype != np.uint8
                        or frame.ndim != 3
                        or frame.shape[-1] != 3
                        or not 0 < frame.shape[0] * frame.shape[1] <= MAX_IMAGE_PIXELS
                    ):
                        raise EmbeddingConfigurationError("ONNX images require bounded uint8 HWC RGB arrays")
                    if kind == "video" and frame.shape != frames[0].shape:
                        raise EmbeddingConfigurationError("ONNX video frames require equal dimensions")
                    size += frame.nbytes
            elif kind == "audio":
                if not isinstance(value, AudioClip) or value.sample_rate != 16000:
                    raise EmbeddingConfigurationError("ONNX audio requires decoded 16 kHz mono PCM")
                samples = value.samples
                if (
                    not isinstance(samples, np.ndarray)
                    or samples.dtype not in (np.float32, np.float64)
                    or samples.ndim != 2
                    or samples.shape[1] != 1
                    or not 321 <= len(samples) <= 16000 * 180
                    or not np.isfinite(samples).all()
                ):
                    raise EmbeddingConfigurationError("ONNX audio requires 321 to 2880000 finite mono samples")
                size += samples.nbytes
            else:
                raise EmbeddingConfigurationError("ONNX requires decoded text, image, audio or video parts")
        if size > max_input_bytes:
            raise EmbeddingConfigurationError("ONNX row exceeds max_input_bytes")
        batch_size += size
        if batch_size > MAX_BATCH_BYTES:
            raise EmbeddingConfigurationError("ONNX batch exceeds the 64 MiB input budget")


class ONNXEmbedder:
    def __init__(self, dimensions: int, options: dict[str, Any]):
        self.options = dict(options)
        self.dimensions = dimensions
        self.sessions: dict[str, Any] = {}
        self.processor: Any = None
        self.closed = False
        self.lock = threading.RLock()
        try:
            self._initialize()
        except BaseException:
            self.close()
            raise

    def _initialize(self) -> None:
        from vane.ai._onnx_options import ONNX_DTYPES

        root = Path(self.options["model_path"])
        if bundle_sha256(root, self.options["dtype"]) != self.options["model_sha256"]:
            raise EmbeddingConfigurationError("ONNX model bundle SHA-256 mismatch")
        config = json.loads((root / "config.json").read_text())
        if (
            config.get("model_type") != "embedding_gemma2"
            or config.get("text_config", {}).get("embedding_dim") != 768
            or config["text_config"].get("hidden_size") != 512
        ):
            raise EmbeddingConfigurationError("ONNX requires the declared EmbeddingGemma 2 geometry")
        self.config = config
        modules = {}
        for dependency in ("onnxruntime", "onnx", "transformers"):
            with _translate_missing_provider_dependency(
                "onnx_cuda" if self.options["backend"] == "cuda" else "onnx", dependency
            ):
                modules[dependency] = import_module(dependency)
        onnx, ort = modules["onnx"], modules["onnxruntime"]
        AutoProcessor = modules["transformers"].AutoProcessor
        validate_software_contract(ort)

        selected = EXECUTION_PROVIDERS[self.options["backend"]]
        if selected not in ort.get_available_providers():
            raise EmbeddingConfigurationError(f"ONNX execution provider {selected} is unavailable")
        if self.options["backend"] == "cuda":
            ort.preload_dlls(directory="")
        provider_options = (
            {"device_id": 0, "use_tf32": 0}
            if self.options["backend"] == "cuda"
            else {"ModelFormat": "MLProgram"}
            if self.options["backend"] == "coreml"
            else {}
        )
        suffix = ONNX_DTYPES[self.options["dtype"]]
        for tower in ("model", "vision_encoder", "audio_encoder"):
            path = root / "onnx" / f"{tower}{suffix}.onnx"
            graph = onnx.load(str(path), load_external_data=False)
            validate_external_data(graph, path.name + "_data")
            settings = ort.SessionOptions()
            settings.intra_op_num_threads = self.options["num_threads"]
            settings.inter_op_num_threads = 1
            # ORT assigns shape/control nodes to CPU even with CUDA. Keep that
            # graph partitioning; enable_fallback=False forbids backend retries.
            self.sessions[tower] = ort.InferenceSession(
                str(path), sess_options=settings, providers=[(selected, provider_options)], enable_fallback=False
            )
            self.sessions[tower].disable_fallback()
            if self.sessions[tower].get_providers()[0] != selected:
                raise EmbeddingConfigurationError("ONNX did not initialize the explicitly selected execution provider")
            expected_inputs = (
                {"input_ids", "attention_mask", "image_features", "video_features", "audio_features"}
                if tower == "model"
                else {"pixel_values", "pixel_position_ids"}
                if tower == "vision_encoder"
                else {"input_features", "input_features_mask"}
            )
            expected_output = {
                "model": "sentence_embedding",
                "vision_encoder": "image_features",
                "audio_encoder": "audio_features",
            }[tower]
            if {item.name for item in self.sessions[tower].get_inputs()} != expected_inputs or expected_output not in {
                item.name for item in self.sessions[tower].get_outputs()
            }:
                raise EmbeddingConfigurationError("ONNX graph signature does not match the declared export")
        self.processor = AutoProcessor.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
        tokenizer = self.processor.tokenizer
        for kind, token in MEDIA_TOKENS.items():
            if tokenizer.convert_tokens_to_ids(token) != config[f"{kind}_token_id"]:
                raise EmbeddingConfigurationError("ONNX tokenizer/media token identity mismatch")

    @staticmethod
    def _feed(session: Any, values: dict[str, Any]) -> dict[str, np.ndarray]:
        dtypes = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(bool)": np.bool_,
        }
        result = {}
        for field in session.get_inputs():
            if field.name not in values or field.type not in dtypes:
                raise EmbeddingConfigurationError("ONNX graph input signature does not match the declared export")
            result[field.name] = np.ascontiguousarray(values[field.name], dtype=dtypes[field.type])
        return result

    def _one(self, parts: list[tuple[str, Any]], role: str) -> np.ndarray:
        text: list[str] = []
        media: dict[str, list[np.ndarray]] = {"image": [], "audio": [], "video": []}
        first_text = True
        for kind, value in parts:
            if kind == "text":
                value = ((QUERY_PREFIX if role == "query" else DOCUMENT_PREFIX) + value) if first_text else value
                first_text = False
                text.append(value)
            else:
                text.append(MEDIA_TOKENS[kind])
                media[kind].append(
                    value.samples[:, 0].astype(np.float32)
                    if kind == "audio"
                    else np.stack(value.frames)
                    if kind == "video"
                    else value
                )
        prompt = "".join(text)
        if (
            len(prompt.encode("utf-8")) + sum(value.nbytes for values in media.values() for value in values)
            > self.options["max_input_bytes"]
        ):
            raise EmbeddingConfigurationError("ONNX row including retrieval prefixes exceeds max_input_bytes")
        kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "np",
            "text_kwargs": {"padding": False},
            "truncation": False,
        }
        if media["image"]:
            kwargs.update(
                images=[media["image"]], images_kwargs={"device": "cpu", "input_data_format": "channels_last"}
            )
        if media["audio"]:
            kwargs.update(audio=media["audio"], sampling_rate=16000, audio_kwargs={"padding": True})
        if media["video"]:
            kwargs.update(
                videos=media["video"],
                videos_kwargs={
                    "device": "cpu",
                    "input_data_format": "channels_last",
                    "do_sample_frames": False,
                    "add_timestamps": False,
                },
            )
        processed = dict(self.processor(**kwargs))
        if np.asarray(processed["input_ids"]).shape[1] > self.options["max_length"]:
            raise EmbeddingConfigurationError("ONNX input exceeds max_length; truncation is disabled")
        features: dict[str, Any] = {"input_ids": processed["input_ids"], "attention_mask": processed["attention_mask"]}
        values: np.ndarray
        for kind in ("image", "video", "audio"):
            if not media[kind]:
                values = np.empty((0, 512), dtype=np.float32)
            elif kind == "audio":
                session = self.sessions["audio_encoder"]
                values = session.run(["audio_features"], self._feed(session, processed))[0]
            else:
                session = self.sessions["vision_encoder"]
                inputs = {
                    "pixel_values": processed["pixel_values" if kind == "image" else "pixel_values_videos"],
                    "pixel_position_ids": processed["image_position_ids" if kind == "image" else "video_position_ids"],
                }
                values = session.run(["image_features"], self._feed(session, inputs))[0]
            values = np.asarray(values)
            if (
                values.ndim != 2
                or values.shape[1] != 512
                or not np.isfinite(values).all()
                or np.count_nonzero(features["input_ids"] == self.config[f"{kind}_token_id"]) != len(values)
            ):
                raise _ProviderResultError("ONNX media feature geometry or placeholder alignment is invalid")
            features[kind + "_features"] = values
        session = self.sessions["model"]
        value = np.asarray(session.run(["sentence_embedding"], self._feed(session, features))[0], dtype=np.float32)
        if value.shape != (1, 768) or not np.isfinite(value).all():
            raise _ProviderResultError("ONNX must return one finite 768-dimensional sentence embedding")
        norm = np.linalg.norm(value.astype(np.float64))
        if not np.isfinite(norm) or norm <= 0 or abs(norm - 1) > 1e-3:
            raise _ProviderResultError("ONNX sentence embeddings must have nonzero unit norm within 1e-3")
        vector = value[0, : self.dimensions].astype(np.float64)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0:
            raise _ProviderResultError("ONNX truncated embedding has zero or invalid norm")
        return (vector / norm).astype(np.float32)

    def _run(self, rows: list[list[tuple[str, Any]]], *, role: str = "document") -> list[np.ndarray]:
        validate_parts(rows, self.options["max_input_bytes"])
        if (
            role not in ("query", "document")
            or role == "query"
            and any([kind for kind, _ in row] != ["text"] for row in rows)
        ):
            raise EmbeddingConfigurationError("ONNX queries require exactly one text part")
        with self.lock:
            if self.closed:
                raise RuntimeError("ONNX embedder is closed")
            return [self._one(row, role) for row in rows]

    def embed_text(self, texts: list[str]) -> list[np.ndarray]:
        if not isinstance(texts, list):
            raise EmbeddingConfigurationError("ONNX embed_text requires a list of strings")
        return self._run([[("text", text)] for text in texts], role=self.options.get("input_type", "document"))

    def embed_image(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if not isinstance(images, list):
            raise EmbeddingConfigurationError("ONNX embed_image requires a list of RGB arrays")
        return self._run([[("image", image)] for image in images])

    def embed_audio(self, clips: list[AudioClip]) -> list[np.ndarray]:
        if not isinstance(clips, list):
            raise EmbeddingConfigurationError("ONNX embed_audio requires a list of AudioClip values")
        return self._run([[("audio", clip)] for clip in clips])

    def embed_video(self, clips: list[VideoClip]) -> list[np.ndarray]:
        if not isinstance(clips, list):
            raise EmbeddingConfigurationError("ONNX embed_video requires a list of VideoClip values")
        return self._run([[("video", clip)] for clip in clips])

    def embed_multimodal(self, inputs: list[EmbeddingContent]) -> list[np.ndarray]:
        if not isinstance(inputs, list) or any(
            not isinstance(value, EmbeddingContent) or any(part.spec.mime_type is not None for part in value.parts)
            for value in inputs
        ):
            raise EmbeddingConfigurationError("ONNX multimodal inputs require decoded typed parts")
        return self._run([[(part.spec.kind, part.value) for part in value.parts] for value in inputs])

    def close(self) -> None:
        with self.lock:
            self.sessions.clear()
            self.processor = None
            self.closed = True

    async def aclose(self) -> None:
        self.close()
