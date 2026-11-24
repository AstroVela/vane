# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Internal, worker-local media values used by AI provider adapters."""

from __future__ import annotations

import os
import re
from dataclasses import InitVar, dataclass
from urllib.parse import SplitResult, urlsplit

_MIME_TOKEN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def normalize_media_content_type(value: object) -> str:
    """Return a provider-safe base MIME type without optional parameters."""
    if not isinstance(value, str):
        raise TypeError(f"FILE content_type must be str, not {type(value).__name__!r}")
    content_type = value.split(";", 1)[0].strip().lower()
    if content_type.count("/") != 1:
        raise ValueError("FILE content_type must be a valid MIME type")
    media_type, media_subtype = content_type.split("/", 1)
    if (
        media_type == "*"
        or media_subtype == "*"
        or not _MIME_TOKEN.fullmatch(media_type)
        or not _MIME_TOKEN.fullmatch(media_subtype)
    ):
        raise ValueError("FILE content_type must be a valid MIME type")
    return content_type


@dataclass(frozen=True, repr=False, slots=True)
class PromptMedia:
    """Resolved FILE bytes plus their routing MIME type.

    Instances are created only inside the AI execution worker. They contain no
    locator, checksum, provider configuration, or credentials and are never
    persisted as FILE values or plan metadata.
    """

    data: bytes
    content_type: str

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes):
            raise TypeError(f"Prompt media data must be bytes, not {type(self.data).__name__!r}")
        if not self.data:
            raise ValueError("Prompt FILE content cannot be zero length")
        object.__setattr__(self, "content_type", normalize_media_content_type(self.content_type))

    def __bytes__(self) -> bytes:
        return self.data

    def __repr__(self) -> str:
        return f"PromptMedia(content_type={self.content_type!r}, size={len(self.data)})"


def _split_image_url(value: str) -> SplitResult:
    if not isinstance(value, str) or any(char.isspace() or ord(char) < 32 or char == "\\" for char in value):
        raise ValueError("Prompt image URL must be an HTTP(S) URL")
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError:
        raise ValueError("Prompt image URL must be an HTTP(S) URL") from None
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.fragment
        or "%" in url.netloc
        or port == 0
    ):
        raise ValueError("Prompt image URL must be an HTTP(S) URL without credentials or a fragment")
    return url


def configured_prompt_image_origin() -> tuple[str, str]:
    configured_origin = os.environ.get("VANE_AI_IMAGE_URL_ORIGIN")
    if not configured_origin:
        raise ValueError("Prompt image URLs require VANE_AI_IMAGE_URL_ORIGIN on the execution worker")
    message = "VANE_AI_IMAGE_URL_ORIGIN must contain only an HTTP(S) scheme, host and optional port"
    try:
        origin = _split_image_url(configured_origin)
    except ValueError:
        raise ValueError(message) from None
    if origin.path not in {"", "/"} or origin.query:
        raise ValueError(message)
    return origin.scheme, origin.netloc.lower()


@dataclass(frozen=True, repr=False, slots=True)
class PromptImageURL:
    """Worker-local URL forwarded to an OpenAI-compatible image fetcher."""

    url: str
    allowed_origin: InitVar[tuple[str, str]]

    def __post_init__(self, allowed_origin: tuple[str, str]) -> None:
        url = _split_image_url(self.url)
        if (url.scheme, url.netloc.lower()) != allowed_origin:
            raise ValueError("Prompt image URL does not match VANE_AI_IMAGE_URL_ORIGIN")
