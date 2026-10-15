# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Gemini Embedding 2 over bounded PCM and timestamped sampled RGB frames.

Each Content is one embedding. VideoClip uses up to six PNG frame parts, so
sampling and absolute timestamps remain owned by Vane, not Google's video
decoder. Audio is always an explicit WAV part, including joint inputs.
"""

from __future__ import annotations

import io
import math
import wave
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from vane.ai._audio_embedding import AudioClip, AudioInputSpec
from vane.ai._video_embedding import VideoClip, VideoInputSpec
from vane.ai.media import TimedContent
from vane.ai.protocols import AudioEmbedderDescriptor, ImageEmbedderDescriptor, VideoEmbedderDescriptor
from vane.ai.providers.google import GoogleTextEmbedder, GoogleTextEmbedderDescriptor, _canonical_model_id

VIDEO_SPEC = VideoInputSpec(max_frames=6, max_input_bytes=16 * 1024**2)
AUDIO_SPEC = AudioInputSpec(sample_rate=16000, max_samples=16000 * 180)
MAX_REQUEST_BYTES = 16 * 1024**2

if TYPE_CHECKING:
    from google.genai import types

    from vane._image import Image
    from vane.ai.typing import Embedding


class GoogleMediaEmbedderDescriptor(GoogleTextEmbedderDescriptor):
    def __post_init__(self) -> None:
        super().__post_init__()
        if _canonical_model_id(self.model_name) != "gemini-embedding-2":
            raise ValueError("Google media embeddings require gemini-embedding-2")
        if set(self.options) - {"max_concurrency_per_actor"}:
            raise ValueError("Google media embeddings accept only request concurrency options")

    def instantiate(self) -> GoogleMultimodalEmbedder:
        return GoogleMultimodalEmbedder(
            options=self.options,
            client_options=self.client_options,
            model=self.model_name,
            dimensions=self.get_dimensions(),
            provider_name=self.provider_name,
        )


class GoogleAudioEmbedderDescriptor(GoogleMediaEmbedderDescriptor, AudioEmbedderDescriptor):
    def get_input_spec(self) -> AudioInputSpec:
        return AUDIO_SPEC


class GoogleVideoEmbedderDescriptor(GoogleMediaEmbedderDescriptor, VideoEmbedderDescriptor):
    def get_input_spec(self) -> VideoInputSpec:
        return VIDEO_SPEC

    def supports_image_queries(self) -> bool:
        return True


class GoogleImageEmbedderDescriptor(GoogleMediaEmbedderDescriptor, ImageEmbedderDescriptor):
    pass


def _image_part(frame: Image) -> types.Part:
    from google.genai import types
    from PIL import Image

    pixels = np.asarray(frame)
    if (
        pixels.dtype != np.uint8
        or pixels.ndim != 3
        or pixels.shape[2] != 3
        or min(pixels.shape[:2]) < 1
        or pixels.nbytes > VIDEO_SPEC.max_input_bytes
    ):
        raise ValueError("Google image embedding requires bounded UInt8 RGB pixels")
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return types.Part.from_bytes(data=output.getvalue(), mime_type="image/png")


def _audio_part(clip: AudioClip) -> types.Part:
    from google.genai import types

    if not isinstance(clip, AudioClip) or type(clip.sample_rate) is not int or not 1 <= clip.sample_rate <= 192000:
        raise ValueError("Google audio embedding requires PCM with a valid sample rate")
    samples = np.asarray(clip.samples)
    if (
        samples.dtype.kind != "f"
        or samples.ndim != 2
        or not 1 <= samples.shape[1] <= 2
        or not 0 < len(samples) <= 180 * clip.sample_rate
        or samples.nbytes > 64 * 1024**2
        or not np.isfinite(samples).all()
        or np.any(np.abs(samples) > 1)
    ):
        raise ValueError("Google audio embedding requires bounded finite PCM in [-1, 1], at most 180 seconds")
    pcm = np.clip(np.rint(samples.mean(axis=1) * 32768), -32768, 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(clip.sample_rate)
        writer.writeframes(pcm.tobytes())
    return types.Part.from_bytes(data=output.getvalue(), mime_type="audio/wav")


def media_content(
    video: VideoClip | None = None, text: str = "", *, audio: Sequence[TimedContent] = ()
) -> types.Content:
    """Build one aligned multimodal input, never average independent vectors."""
    from google.genai import types

    if not isinstance(text, str):
        raise TypeError("Google joint text must be a string")
    parts = [types.Part.from_text(text=text)] if text.strip() else []
    seconds = 0.0
    if video is not None:
        VIDEO_SPEC.validate_count(len(video.frames))
        if len(video.frames) != len(video.frame_indices) or len(video.frames) != len(video.frame_times):
            raise ValueError("Video frames require matching indices and timestamps")
        if any(type(t) not in (float, int) or not math.isfinite(t) or t < 0 for t in video.frame_times):
            raise ValueError("Video timestamps must be finite and nonnegative")
        if (
            any(a > b for a, b in zip(video.frame_times, video.frame_times[1:]))
            or any(type(index) is not int or index < 0 for index in video.frame_indices)
            or any(a >= b for a, b in zip(video.frame_indices, video.frame_indices[1:]))
        ):
            raise ValueError("Video frames require ordered timestamps and increasing indices")
        if sum(np.asarray(frame).nbytes for frame in video.frames) > VIDEO_SPEC.max_input_bytes:
            raise ValueError("Google video exceeds the decoded input budget")
        for timestamp, frame in zip(video.frame_times, video.frames, strict=True):
            parts.extend([types.Part.from_text(text=f"frame_time_sec={timestamp:.17g}"), _image_part(frame)])
    previous_end = None
    for item in audio:
        if not isinstance(item, TimedContent) or not isinstance(item.value, AudioClip):
            raise TypeError("Joint audio requires timestamped AudioClip inputs")
        clip = item.value
        part = _audio_part(clip)
        duration = len(clip.samples) / clip.sample_rate
        if abs(item.end - item.start - duration) > 1 / clip.sample_rate or (
            previous_end is not None and item.start < previous_end - 1 / clip.sample_rate
        ):
            raise ValueError("Joint audio must have ordered, nonoverlapping sample-aligned timestamps")
        previous_end = item.end
        seconds += duration
        parts.extend(
            [types.Part.from_text(text=f"audio_start_sec={item.start:.17g}; audio_end_sec={item.end:.17g}"), part]
        )
    if not parts or seconds > 180:
        raise ValueError("Google joint input requires content and at most 180 seconds of audio")
    # Bound text locally by bytes; the endpoint owns the combined multimodal
    # token count. Do not approximate image/audio tokens or truncate to fit.
    text_bytes = sum(len(part.text.encode("utf-8")) for part in parts if part.text)
    if text_bytes > 8192:
        raise ValueError("Google multimodal input exceeds the local 8192-byte text budget")
    content = types.Content(parts=parts)
    if len(content.model_dump_json(exclude_none=True).encode()) > MAX_REQUEST_BYTES:
        raise ValueError("Google multimodal input exceeds the encoded request byte budget")
    return content


class GoogleMultimodalEmbedder(GoogleTextEmbedder):
    async def _media(self, inputs: Sequence[Any], encode: Callable[[Any], types.Content]) -> list[Embedding]:
        async def one(batch: list[Any]) -> list[Embedding]:
            return await self._embed_contents([encode(batch[0])])

        # One clip per RPC bounds encoded payloads and also works with Vertex's
        # single-Content contract. The shared request runner bounds concurrency.
        return await self._run_requests((([i], [item]) for i, item in enumerate(inputs)), one, [None] * len(inputs))

    async def embed_video(self, clips: list[VideoClip]) -> list[Embedding]:
        return await self._media(clips, lambda clip: media_content(video=clip))

    async def embed_audio(self, clips: list[AudioClip]) -> list[Embedding]:
        def encode(clip: AudioClip) -> types.Content:
            if clip.sample_rate != AUDIO_SPEC.sample_rate:
                raise ValueError("Google audio embedding requires 16000 Hz; resample before embedding")
            return media_content(audio=(TimedContent(0, len(clip.samples) / clip.sample_rate, clip),))

        return await self._media(clips, encode)

    async def embed_image(self, images: list[Image]) -> list[Embedding]:
        return await self._media(images, lambda frame: media_content(video=VideoClip((frame,), (0.0,), (0,))))

    async def embed_joint(self, video: VideoClip | None, text: str, *, audio: Sequence[TimedContent] = ()) -> Embedding:
        return (await self._embed_contents([media_content(video, text, audio=audio)]))[0]

    async def _embed_contents(self, contents: list[Any], *, reject_truncation: bool = True) -> list[Embedding]:
        values = await super()._embed_contents(contents, reject_truncation=reject_truncation)
        if any(
            value.shape != (self._dimensions,) or not np.isfinite(value).all() or not np.any(value) for value in values
        ):
            raise ValueError("Google multimodal model returned an invalid embedding")
        return values


class GoogleJointEmbedder:
    """Serializable configuration; the async SDK client is created on a worker."""

    input_modalities = frozenset({"video", "audio", "text"})
    video_input_spec = VIDEO_SPEC
    audio_input_spec = AUDIO_SPEC

    def __init__(
        self, model: str = "gemini-embedding-2", *, dimensions: int = 3072, max_concurrency_per_actor: int = 1
    ) -> None:
        self.descriptor = GoogleMediaEmbedderDescriptor(
            model_name=model,
            dimensions=dimensions,
            request_dimensions=dimensions,
            options={"max_concurrency_per_actor": max_concurrency_per_actor},
        )
        self.dimensions = self.descriptor.get_dimensions()
        self._runtime: GoogleMultimodalEmbedder | None = None

    def client(self) -> GoogleMultimodalEmbedder:
        if self._runtime is None:
            self._runtime = self.descriptor.instantiate()
        return self._runtime

    async def embed_joint(self, video: VideoClip | None, text: str, *, audio: Sequence[TimedContent] = ()) -> Embedding:
        return await self.client().embed_joint(video, text, audio=audio)

    async def embed_query(self, text: str) -> Embedding:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Google query text must be nonempty")
        return await self.client().embed_joint(None, f"task: search result | query: {text}")

    async def aclose(self) -> None:
        if self._runtime is not None:
            try:
                await self._runtime.aclose()
            finally:
                self._runtime = None
