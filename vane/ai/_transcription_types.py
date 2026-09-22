# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable, bounded speech transcription values shared by providers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

MAX_TRANSCRIPTION_CHARS = 100_000
MAX_TRANSCRIPTION_SEGMENTS = 4096
MAX_TRANSCRIPTION_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class TranscriptionInput:
    """Encoded audio transported as bytes, with no worker-local path."""

    data: bytes = field(repr=False)
    content_type: str


@dataclass(frozen=True)
class TranscriptionSegment:
    """Model-aligned seconds relative to the start of the supplied recording."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcription:
    """A provider transcript; timestamps are model alignments, not exact speech boundaries."""

    text: str
    language: str | None
    duration: float
    segments: tuple[TranscriptionSegment, ...]


def validate_transcription(value: Any) -> Transcription:
    """Reject missing timing and malformed output without rewriting model values."""
    if not isinstance(value, Transcription):
        raise ValueError("Transcribe provider must return a Transcription")
    if not isinstance(value.text, str) or len(value.text) > MAX_TRANSCRIPTION_CHARS:
        raise ValueError("Transcribe text exceeds the output budget or has an invalid type")
    if value.language is not None and (
        not isinstance(value.language, str) or not value.language.strip() or len(value.language) > 128
    ):
        raise ValueError("Transcribe language must be a nonempty string or None")
    if type(value.duration) not in (int, float) or not math.isfinite(value.duration) or value.duration <= 0:
        raise ValueError("Transcribe duration must be finite and positive")
    if not isinstance(value.segments, tuple) or len(value.segments) > MAX_TRANSCRIPTION_SEGMENTS:
        raise ValueError("Transcribe segments exceed the output budget or have an invalid type")
    previous_start, previous_end, chars = -1.0, -1.0, 0
    for segment in value.segments:
        if not isinstance(segment, TranscriptionSegment):
            raise ValueError("Transcribe segments must be TranscriptionSegment values")
        if any(type(t) not in (int, float) or not math.isfinite(t) for t in (segment.start, segment.end)):
            raise ValueError("Transcribe timestamps must be finite numbers")
        if not 0 <= segment.start < segment.end <= value.duration:
            raise ValueError("Transcribe timestamps must lie within the reported recording duration")
        # Overlap is valid for concurrent speech, but the provider must return
        # segments in temporal order. Never sort or clamp an invalid response.
        if segment.start < previous_start or segment.end < previous_end:
            raise ValueError("Transcribe segments must be ordered by start and end time")
        if not isinstance(segment.text, str) or not segment.text.strip():
            raise ValueError("Transcribe segment text must be nonempty")
        chars += len(segment.text)
        if chars > MAX_TRANSCRIPTION_CHARS:
            raise ValueError("Transcribe segment text exceeds the output budget")
        previous_start, previous_end = segment.start, segment.end
    # Segment boundaries may omit whitespace present in the full transcript.
    if "".join(value.text.split()) != "".join("".join(s.text for s in value.segments).split()):
        raise ValueError("Transcribe text must be covered by its timed segments")
    return value
