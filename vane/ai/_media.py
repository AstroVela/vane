# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Internal, worker-local media values used by AI provider adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass

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
