# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit SGLang 0.5.17 and vLLM 0.31 embedding wire contracts.

No runtime SDK, file access, video transcoding or model discovery. Encoded media
is transported unchanged. Decoded images have one explicit lossless PNG encoding.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any, cast

import numpy as np

from vane.ai._embedding_content import EmbeddingContent, EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.embedding import Modality
from vane.ai.provider import _translate_missing_provider_dependency

IMAGE_TYPES = frozenset({"image/png", "image/jpeg"})
VIDEO_TYPES = frozenset({"video/mp4", "video/quicktime", "video/webm", "video/x-nut"})
AUDIO_TYPES = frozenset({"audio/wav", "audio/mpeg"})


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


class EmbeddingHTTPCodec:
    def __init__(self, family: str, model: str, dimensions: int, options: dict[str, Any]):
        if family not in {"sglang", "vllm"}:
            raise EmbeddingConfigurationError("Unknown embedding HTTP protocol")
        self.family = family
        self.input_format = options.get("http_input_format", "text")
        allowed = (
            IMAGE_TYPES
            | VIDEO_TYPES
            | (AUDIO_TYPES if family == "vllm" or self.input_format == "encode" else frozenset())
        )
        self.media_tokens = options.get("http_media_tokens", {})
        self.dimensions = dimensions if options.get("supports_overriding_dimensions", False) else None
        declared = frozenset(options.get("media_mime_types", []))
        if declared - allowed:
            raise EmbeddingConfigurationError(
                "media_mime_types contains a type unsupported by the selected HTTP protocol"
            )
        parts = {EmbeddingPartSpec("text")}
        parts.update(EmbeddingPartSpec(cast(Modality, mime.split("/")[0]), mime) for mime in declared)
        if "image/png" in declared:
            parts.add(EmbeddingPartSpec("image"))
        self.spec = MultimodalInputSpec(
            supported_parts=frozenset(parts), max_input_bytes=options.get("max_input_bytes", 16 * 1024**2)
        )
        metadata: dict[str, Any] = {"model": model, "encoding_format": "float"}
        if options.get("supports_overriding_dimensions", False):
            metadata["dimensions"] = dimensions
        field = "messages" if family == "vllm" and self.input_format == "multimodal" else "input"
        self.prefix = json_bytes(metadata)[:-1] + b',"' + field.encode() + b'":['
        self.suffix = b"]}"
        if self.input_format == "encode":
            self.prefix = self.suffix = b""

    def encode(self, content: EmbeddingContent) -> bytes:
        self.spec.validate_layout(tuple(part.spec for part in content.parts))
        active: list[tuple[EmbeddingPartSpec, Any]] = []
        total = 0
        for part in content.parts:
            value, spec = part.value, part.spec
            if value is None:
                continue
            if spec.kind == "text":
                if not isinstance(value, str):
                    raise EmbeddingConfigurationError("HTTP text embedding requires strings")
                if not value.strip():
                    continue
                total += len(value.encode("utf-8"))
            elif spec.mime_type is not None:
                if not isinstance(value, bytes) or not value:
                    raise EmbeddingConfigurationError("HTTP encoded media requires nonempty BLOB bytes")
                total += len(value)
            else:
                pixels = np.asarray(value)
                if (
                    pixels.dtype != np.uint8
                    or pixels.ndim != 3
                    or pixels.shape[2] not in {1, 3, 4}
                    or min(pixels.shape[:2]) < 1
                ):
                    raise EmbeddingConfigurationError("HTTP decoded images require UInt8 HWC grayscale, RGB or RGBA")
                total += pixels.nbytes
            if total > self.spec.max_input_bytes:
                raise EmbeddingConfigurationError("HTTP embedding row exceeds the combined input byte budget")
            active.append((spec, value))
        if not active:
            raise EmbeddingConfigurationError("Multimodal embedding row has no nonempty parts")

        if self.input_format == "text":
            if len(active) != 1 or active[0][0].kind != "text":
                raise EmbeddingConfigurationError("HTTP text input format requires exactly one text part")
            return json_bytes(active[0][1])

        fields: dict[str, str] = {}
        messages = []
        for spec, value in active:
            kind, mime = spec.kind, spec.mime_type
            if self.family == "sglang" and kind in fields:
                raise EmbeddingConfigurationError("SGLang HTTP accepts at most one nonempty part of each modality")
            if kind == "text":
                fields[kind] = value
                messages.append({"type": "text", "text": value})
                continue
            if mime is None:
                with _translate_missing_provider_dependency("embedding_http", "PIL"):
                    from PIL import Image

                pixels = np.asarray(value)
                output = io.BytesIO()
                Image.fromarray(pixels[:, :, 0] if pixels.shape[2] == 1 else pixels).save(output, format="PNG")
                value, mime = output.getvalue(), "image/png"
            encoded = base64.b64encode(value).decode("ascii")
            url = f"data:{mime};base64,{encoded}"
            fields[kind] = url
            if kind == "audio":
                messages.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded, "format": "wav" if mime == "audio/wav" else "mp3"},
                    }
                )
            else:
                messages.append({"type": f"{kind}_url", f"{kind}_url": {"url": url}})
        if self.input_format == "encode":
            # /encode is a raw-input endpoint: unlike /v1/embeddings, it does
            # not render a chat template. Markers belong to explicit deployment
            # configuration, never a guessed model-family convention.
            text = fields.get("text", "")
            if any(token in text for token in self.media_tokens.values()):
                raise EmbeddingConfigurationError("Text cannot contain configured HTTP media markers")
            prompt = "".join(self.media_tokens[kind] for kind in fields if kind != "text") + text
            row: dict[str, Any] = {
                "text": prompt,
                **{kind + "_data": fields.get(kind) for kind in ("image", "video", "audio")},
            }
            if self.dimensions is not None:
                row["dimensions"] = self.dimensions
            return json_bytes(row)
        if self.family == "sglang":
            # The protocol has named fields; its server template owns their order.
            return json_bytes(fields)
        # One conversation per ROW, including when a row contains only text.
        return json_bytes([{"role": "user", "content": messages}])

    def body(self, rows: list[bytes]) -> bytes:
        if self.input_format == "encode":
            if len(rows) == 1:
                return rows[0]
            values = [json.loads(row) for row in rows]
            # Every encoded row includes every media column, including nulls.
            # The sum of row sizes therefore bounds the column-oriented body.
            body: dict[str, Any] = {
                key: [value[key] for value in values] for key in ("text", "image_data", "video_data", "audio_data")
            }
            if self.dimensions is not None:
                body["dimensions"] = self.dimensions
            return json_bytes(body)
        return self.prefix + b",".join(rows) + self.suffix
