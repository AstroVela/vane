# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI segment-timestamp transcription using the shared client snapshot."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from vane.ai._client_config import copy_client_options
from vane.ai._transcription_types import Transcription, TranscriptionInput, TranscriptionSegment, validate_transcription
from vane.ai.options import _validate_base_url_option
from vane.ai.protocols import TranscriberDescriptor
from vane.ai.provider import _translate_missing_provider_dependency
from vane.ai.providers._openai_client_config import create_openai_client

_AUDIO_EXTENSIONS = {
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
    "audio/flac": "flac",
    "audio/ogg": "ogg",
    "audio/mp4": "m4a",
    "audio/webm": "webm",
}


@dataclass
class OpenAITranscriberDescriptor(TranscriberDescriptor):
    provider_name: str
    model_name: str
    options: dict[str, Any]
    client_options: dict[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        # Only Whisper exposes the segment timestamps required by this API.
        # https://developers.openai.com/api/docs/guides/speech-to-text#timestamp-options
        if self.model_name != "whisper-1":
            raise ValueError("OpenAI Transcribe requires model='whisper-1' for segment timestamps")
        self.options = dict(self.options)
        unknown = set(self.options) - {"language", "prompt", "base_url", "timeout"}
        if unknown:
            raise TypeError(f"Unsupported OpenAI transcription option(s): {', '.join(sorted(unknown))}")
        language = self.options.get("language")
        if language is not None and (not isinstance(language, str) or re.fullmatch(r"[a-z]{2}", language) is None):
            raise ValueError("Transcribe language must be a lowercase ISO-639-1 code")
        prompt = self.options.get("prompt")
        if prompt is not None and (not isinstance(prompt, str) or len(prompt) > 10_000):
            raise ValueError("Transcribe prompt must be a string of at most 10,000 characters")
        timeout = self.options.get("timeout")
        if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Transcribe timeout must be finite and positive")
        _validate_base_url_option(self.options, api="Transcribe")
        self.client_options = copy_client_options(self.client_options)

    def get_provider(self) -> str:
        return self.provider_name

    def get_model(self) -> str:
        return self.model_name

    def get_options(self) -> dict[str, Any]:
        return dict(self.options)

    def supported_media_mime_types(self) -> frozenset[str]:
        return frozenset(_AUDIO_EXTENSIONS)

    def instantiate(self) -> OpenAITranscriber:
        with _translate_missing_provider_dependency("openai", "openai"):
            from openai import AsyncOpenAI

        return OpenAITranscriber(create_openai_client(AsyncOpenAI, self.client_options, self.options), self.options)


class OpenAITranscriber:
    def __init__(self, client: Any, options: dict[str, Any]):
        self._client = client
        self._options = {name: options[name] for name in ("language", "prompt") if options.get(name) is not None}

    async def aclose(self) -> None:
        await self._client.close()

    async def transcribe(self, audio: TranscriptionInput) -> Transcription:
        extension = _AUDIO_EXTENSIONS.get(audio.content_type)
        if extension is None:
            raise ValueError("Unsupported transcription audio MIME type")
        response = await self._client.audio.transcriptions.create(
            model="whisper-1",
            file=(f"audio.{extension}", audio.data, audio.content_type),
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            **self._options,
        )
        if response.segments is None:
            raise ValueError("Transcribe endpoint did not return segment timestamps")
        return validate_transcription(
            Transcription(
                text=response.text,
                language=response.language,
                duration=response.duration,
                segments=tuple(TranscriptionSegment(s.start, s.end, s.text) for s in response.segments),
            )
        )
