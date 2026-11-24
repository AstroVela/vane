# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Lazy local and cloud adapters for timed video extraction and joint encoding.

Cloud credentials are environment-variable references resolved by each worker.
Local model paths must be provisioned on workers; these adapters do not download
weights or run remote Python code. Models and clients are never loaded at bind.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

import numpy as np

from vane.ai._audio_embedding import AudioClip
from vane.ai._video_embedding import VideoClip
from vane.ai.media import MediaWindow, TimedContent
from vane.ai.providers._google_multimodal import GoogleJointEmbedder as GoogleJointEmbedder

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

    from vane.ai.providers.qwen3_vl import Qwen3VLEmbedder


def _local_path(value: str) -> str:
    if not isinstance(value, str) or not Path(value).is_absolute() or "://" in value:
        raise ValueError("Local models require an absolute worker-local path")
    return value


def _name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Model and profile identities must be nonempty strings")
    return value


def _segment(item: TimedContent, start: float, end: float, text: str) -> TimedContent | None:
    if not isinstance(text, str):
        raise ValueError("ASR must return string text")
    if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) for t in (start, end)):
        raise ValueError("ASR timestamps must be finite numbers")
    if start < 0 or end < start or end > item.end - item.start + 0.001:
        raise ValueError("ASR timestamp lies outside its decoded audio window")
    return TimedContent(item.start + start, min(item.start + end, item.end), text) if text.strip() else None


def _validate_window(clip: MediaWindow, content: Sequence[TimedContent]) -> None:
    if any(not clip.start <= item.start <= item.end <= clip.end for item in content):
        raise ValueError("Model content must lie within its media window")


def _mono(item: TimedContent, sample_rate: int) -> np.ndarray:
    audio = item.value
    if not isinstance(audio, AudioClip):
        raise TypeError("ASR requires decoded AudioClip content")
    samples = np.asarray(audio.samples, dtype=np.float32)
    if (
        samples.ndim != 2
        or samples.shape[1] == 0
        or not np.isfinite(samples).all()
        or np.abs(samples).max(initial=0) > 1
    ):
        raise ValueError("ASR requires finite PCM in [-1, 1]")
    mono = samples.mean(axis=1)
    if audio.sample_rate == sample_rate:
        return mono
    import av

    frame = av.AudioFrame.from_ndarray(mono.reshape(1, -1), format="fltp", layout="mono")
    frame.sample_rate = audio.sample_rate
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
    frames = [*resampler.resample(frame), *resampler.resample(None)]
    return np.concatenate([f.to_ndarray().reshape(-1) for f in frames]) if frames else np.empty(0, np.float32)


def _frames(content: Sequence[TimedContent]) -> Iterator[tuple[float, np.ndarray[Any, Any]]]:
    for item in content:
        if not isinstance(item.value, VideoClip):
            raise TypeError("OCR requires decoded VideoClip content")
        for timestamp, frame in zip(item.value.frame_times, item.value.frames, strict=True):
            if not item.start <= timestamp < item.end:
                raise ValueError("OCR frame timestamp lies outside its content window")
            yield timestamp, frame


def _png(frame: np.ndarray) -> bytes:
    from PIL import Image

    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Video model images require UInt8 RGB pixels")
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


class FasterWhisperASR:
    """Local faster-whisper with VAD enabled and times on the asset timeline.

    ``vad_filter=False`` explicitly transcribes without speech detection. VAD
    only filters the transcription input; it does not replace original sound.
    ``vad_speech_pad_ms`` keeps context around detected speech, including quiet
    word onsets that precede the detector's boundary.
    """

    def __init__(
        self,
        model: str,
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        vad_filter: bool = True,
        vad_speech_pad_ms: int = 800,
    ):
        self.model = _local_path(model)
        if device not in {"cpu", "cuda"} or compute_type not in {"int8", "float16", "float32", "int8_float16"}:
            raise ValueError("Select an explicit CPU/CUDA device and compute_type")
        if not isinstance(vad_filter, bool):
            raise ValueError("vad_filter must be a bool")
        if type(vad_speech_pad_ms) is not int or vad_speech_pad_ms < 0:
            raise ValueError("vad_speech_pad_ms must be a nonnegative int")
        self.device, self.compute_type, self.language = device, compute_type, language
        self.vad_filter = vad_filter
        self.vad_speech_pad_ms = vad_speech_pad_ms
        self._runtime: WhisperModel | None = None

    def identity(self) -> dict[str, str]:
        return {"provider": "faster_whisper", "model": self.model}

    def __call__(self, clip: MediaWindow, content: Sequence[TimedContent]) -> list[TimedContent]:
        _validate_window(clip, content)
        if not content:
            return []
        if self._runtime is None:
            from faster_whisper import WhisperModel

            if not all((Path(self.model) / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json")):
                raise ValueError("Whisper model.bin, config.json and tokenizer.json must exist on the worker")
            self._runtime = WhisperModel(
                self.model, device=self.device, compute_type=self.compute_type, local_files_only=True
            )
        result = []
        for item in content:
            segments, _ = self._runtime.transcribe(
                _mono(item, 16000),
                language=self.language,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=self.vad_filter,
                vad_parameters={"speech_pad_ms": self.vad_speech_pad_ms},
            )
            for segment in segments:
                text = _segment(item, segment.start, segment.end, segment.text)
                if text is not None:
                    result.append(text)
        return result

    def close(self) -> None:
        self._runtime = None


class _RapidRuntime:
    """Use detection/recognition directly; do not load an unused classifier or
    let the high-level OCR convenience wrapper turn recognition errors into empty results.
    """

    def __init__(self, detection_model: str, recognition_model: str, keys: str):
        import rapidocr
        from rapidocr.ch_ppocr_det import TextDetector
        from rapidocr.ch_ppocr_rec import TextRecognizer
        from rapidocr.utils.parse_parameters import ParseParams

        config = ParseParams.load(Path(rapidocr.__file__).parent / "config.yaml")
        config.Det.model_path = detection_model
        config.Rec.model_path = recognition_model
        config.Rec.rec_keys_path = keys
        config.Rec.font_path = None
        for section in (config.Det, config.Rec):
            section.engine_cfg = config.EngineConfig[section.engine_type.value]
            section.engine_cfg.use_cuda = False
            section.model_root_dir = str(Path(section.model_path).parent)
        self.detector = TextDetector(config.Det)
        self.recognizer = TextRecognizer(config.Rec)
        self.threshold = float(config.Global.text_score)

    def __call__(self, pixels: np.ndarray) -> tuple[str, ...]:
        from rapidocr.ch_ppocr_rec import TextRecInput
        from rapidocr.utils.process_img import get_rotate_crop_image

        detected = self.detector(pixels)
        if detected.boxes is None or len(detected.boxes) == 0:
            return ()
        crops = [get_rotate_crop_image(pixels, box.copy()) for box in detected.boxes]
        result = self.recognizer(TextRecInput(img=crops))
        if (
            result.txts is None
            or result.scores is None
            or len(result.txts) != len(crops)
            or len(result.scores) != len(crops)
        ):
            raise ValueError("OCR recognition returned incomplete output")
        if any(
            not isinstance(text, str) or not math.isfinite(score)
            for text, score in zip(result.txts, result.scores, strict=True)
        ):
            raise ValueError("OCR recognition returned invalid text or confidence")
        return tuple(text for text, score in zip(result.txts, result.scores, strict=True) if score >= self.threshold)


class RapidOCRExtractor:
    """CPU ONNX OCR with explicit detection/recognition models and dictionary."""

    def __init__(self, detection_model: str, recognition_model: str, keys: str):
        self.detection_model, self.recognition_model, self.keys = map(
            _local_path, (detection_model, recognition_model, keys)
        )
        self._runtime: _RapidRuntime | None = None

    def identity(self) -> dict[str, str]:
        return {"provider": "rapidocr", "model": self.recognition_model}

    def __call__(self, clip: MediaWindow, content: Sequence[TimedContent]) -> list[TimedContent]:
        _validate_window(clip, content)
        if not content:
            return []
        if self._runtime is None:
            if not all(Path(path).is_file() for path in (self.detection_model, self.recognition_model, self.keys)):
                raise ValueError("RapidOCR models and character dictionary must exist on the worker")
            self._runtime = _RapidRuntime(self.detection_model, self.recognition_model, self.keys)
        result = []
        for timestamp, frame in _frames(content):
            # RapidOCR treats ndarray input as BGR; Vane's decoded frames are RGB.
            recognized = self._runtime(np.ascontiguousarray(frame[:, :, ::-1]))
            if not all(isinstance(text, str) for text in recognized):
                raise ValueError("OCR returned invalid text")
            text = "\n".join(text for text in recognized if text.strip())
            if text:
                result.append(TimedContent(timestamp, timestamp, text))
        return result

    def close(self) -> None:
        self._runtime = None


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def _post(url: str, data: bytes, headers: dict[str, str], timeout: float, limit: int = 16 * 1024**2) -> dict[str, Any]:
    # Fixed provider endpoints, no redirects, no retries, bounded response bodies.
    if len(data) > 20 * 1024**2:
        raise ValueError("Video provider request exceeds 20 MiB")
    try:
        with build_opener(_NoRedirect()).open(
            Request(url, data=data, headers=headers, method="POST"), timeout=timeout
        ) as response:
            body = response.read(limit + 1)
    except HTTPError as error:
        status = error.code
        error.close()
        raise RuntimeError(f"Video provider HTTP status {status}") from None
    except OSError:
        raise RuntimeError("Video provider transport failed") from None
    if len(body) > limit:
        raise ValueError("Video provider response exceeds the byte limit")
    try:
        result = json.loads(body)
    except (ValueError, UnicodeError):
        raise ValueError("Video provider returned invalid JSON") from None
    if not isinstance(result, dict):
        raise ValueError("Video provider returned a non-object response")
    return result


def _credential(variable: str) -> str:
    value = os.environ.get(variable)
    if not value or "\r" in value or "\n" in value:
        raise ValueError("Configure the video provider credential on each worker")
    return value


class DeepgramASR:
    def __init__(
        self,
        *,
        model: str = "nova-3",
        api_key_env: str = "DEEPGRAM_API_KEY",
        language: str = "en",
        timeout_seconds: float = 30,
    ):
        self.model = _name(model)
        self.api_key_env, self.language = _name(api_key_env), _name(language)
        self.timeout = _timeout(timeout_seconds)

    def identity(self) -> dict[str, str]:
        return {"provider": "deepgram", "model": self.model}

    def __call__(self, clip: MediaWindow, content: Sequence[TimedContent]) -> list[TimedContent]:
        _validate_window(clip, content)
        result = []
        for item in content:
            samples = _mono(item, 16000)
            pcm = np.rint(samples * 32767).astype("<i2").tobytes()
            query = urlencode(
                {
                    "model": self.model,
                    "language": self.language,
                    "encoding": "linear16",
                    "sample_rate": 16000,
                    "channels": 1,
                    "punctuate": "true",
                }
            )
            response = _post(
                "https://api.deepgram.com/v1/listen?" + query,
                pcm,
                {"Authorization": "Token " + _credential(self.api_key_env), "Content-Type": "application/octet-stream"},
                self.timeout,
            )
            try:
                alternatives = response["results"]["channels"][0]["alternatives"]
                words = alternatives[0]["words"]
                if not isinstance(words, list) or not isinstance(alternatives[0]["transcript"], str):
                    raise TypeError()
                if not words and alternatives[0]["transcript"].strip():
                    raise ValueError("ASR returned text without timestamps")
                for word in words:
                    text = _segment(item, word["start"], word["end"], word.get("punctuated_word", word["word"]))
                    if text is not None:
                        result.append(text)
            except (KeyError, IndexError, TypeError):
                raise ValueError("Deepgram returned invalid timed transcription") from None
        return result


class GoogleVisionOCR:
    def __init__(self, *, api_key_env: str = "GOOGLE_VISION_API_KEY", timeout_seconds: float = 30):
        self.api_key_env = _name(api_key_env)
        self.timeout = _timeout(timeout_seconds)

    def identity(self) -> dict[str, str]:
        return {"provider": "google_vision", "model": "TEXT_DETECTION"}

    def __call__(self, clip: MediaWindow, content: Sequence[TimedContent]) -> list[TimedContent]:
        _validate_window(clip, content)
        result = []
        for timestamp, frame in _frames(content):
            payload = {
                "requests": [
                    {
                        "image": {"content": base64.b64encode(_png(frame)).decode()},
                        "features": [{"type": "TEXT_DETECTION"}],
                    }
                ]
            }
            response = _post(
                "https://vision.googleapis.com/v1/images:annotate",
                json.dumps(payload).encode(),
                {"Content-Type": "application/json", "X-Goog-Api-Key": _credential(self.api_key_env)},
                self.timeout,
            )
            try:
                entries = response["responses"]
                if len(entries) != 1 or not isinstance(entries[0], dict) or entries[0].get("error"):
                    raise ValueError("Google Vision OCR failed")
                annotations = entries[0].get("textAnnotations", [])
                text = annotations[0]["description"] if annotations else ""
                if not isinstance(text, str):
                    raise TypeError()
                if text.strip():
                    result.append(TimedContent(timestamp, timestamp, text))
            except (KeyError, IndexError, TypeError):
                raise ValueError("Google Vision returned invalid OCR output") from None
        return result


class Qwen3VLJointEmbedder:
    """Joint video/text encoding using the existing explicit Qwen BF16 contract."""

    def __init__(self, model: str, *, frame_size: int = 224):
        if type(frame_size) is not int or frame_size <= 0 or frame_size % 32 or frame_size > 256:
            raise ValueError("Qwen fusion frame_size must be a positive multiple of 32 up to 256")
        self.model = _local_path(model)
        self._runtime: Qwen3VLEmbedder | None = None
        self.frame_size = frame_size

    dimensions = 2048
    input_modalities = frozenset({"video", "text"})

    def embed_joint(
        self, video: VideoClip | None, text: str, *, audio: Sequence[TimedContent] = ()
    ) -> np.ndarray[Any, Any]:
        from vane.ai.providers.qwen3_vl import Qwen3VLEmbedder

        if audio:
            raise ValueError("Qwen3-VL joint embedding does not support raw audio")
        if video is None and not text.strip():
            raise ValueError("Joint embedding requires video or text")
        if self._runtime is None:
            self._runtime = Qwen3VLEmbedder(self.model)
        if video is not None:
            from PIL import Image

            video = VideoClip(
                tuple(
                    np.asarray(
                        Image.fromarray(frame).resize((self.frame_size, self.frame_size), Image.Resampling.BICUBIC)
                    )
                    for frame in video.frames
                ),
                video.frame_times,
                video.frame_indices,
            )
        return self._runtime.embed_joint(video, text)

    def embed_query(self, text: str) -> np.ndarray[Any, Any]:
        return self.embed_joint(None, text)

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None


class CohereJointEmbedder:
    """One joint Embed v4 input containing timestamped frames and text."""

    def __init__(
        self,
        *,
        dimensions: int = 1024,
        api_key_env: str = "COHERE_API_KEY",
        timeout_seconds: float = 30,
        max_request_bytes: int = 20 * 1024**2,
    ):
        if type(dimensions) is not int or dimensions not in {256, 512, 1024, 1536}:
            raise ValueError("Cohere Embed v4 dimensions must be 256, 512, 1024 or 1536")
        if type(max_request_bytes) is not int or not 0 < max_request_bytes <= 20 * 1024**2:
            raise ValueError("Cohere max_request_bytes must be within 20 MiB")
        self.dimensions, self.api_key_env = dimensions, _name(api_key_env)
        self.timeout = _timeout(timeout_seconds)
        self.max_request_bytes = max_request_bytes

    input_modalities = frozenset({"video", "text"})

    def embed_query(self, text: str) -> list[float]:
        return self._embed(None, text, "search_query")

    def embed_joint(self, video: VideoClip | None, text: str, *, audio: Sequence[TimedContent] = ()) -> list[float]:
        if audio:
            raise ValueError("Cohere joint embedding does not support raw audio")
        return self._embed(video, text, "search_document")

    def _embed(self, video: VideoClip | None, text: str, input_type: str) -> list[float]:
        if video is None and not text.strip():
            raise ValueError("Joint embedding requires video or text")
        content: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        frames = zip(video.frame_times, video.frames, strict=True) if video is not None else ()
        for timestamp, frame in frames:
            content.extend(
                [
                    {"type": "text", "text": f"frame_time_sec={timestamp}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + base64.b64encode(_png(frame)).decode()},
                    },
                ]
            )
        payload = json.dumps(
            {
                "model": "embed-v4.0",
                "input_type": input_type,
                "inputs": [{"content": content}],
                "embedding_types": ["float"],
                "output_dimension": self.dimensions,
                "truncate": "NONE",
            }
        ).encode()
        if len(payload) > self.max_request_bytes:
            raise ValueError("Cohere input exceeds max_request_bytes")
        response = _post(
            "https://api.cohere.com/v2/embed",
            payload,
            {"Authorization": "Bearer " + _credential(self.api_key_env), "Content-Type": "application/json"},
            self.timeout,
        )
        try:
            vectors = response["embeddings"]["float"]
            if not isinstance(vectors, list) or len(vectors) != 1:
                raise ValueError()
            return vectors[0]
        except (KeyError, TypeError, ValueError):
            raise ValueError("Cohere returned invalid joint embedding output") from None


def _timeout(value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    return float(value)
