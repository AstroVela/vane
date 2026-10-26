# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validate generated requests with the actual optional serving SDK schemas.

Run in CPU processes with the pinned SGLang 0.5.17 / vLLM 0.31 dependencies.
These checks load no weights; model/processor acceptance remains a separate smoke.
"""

from __future__ import annotations

import json

import pytest

from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai.providers._embedding_http_codecs import EmbeddingHTTPCodec


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("input_format", ["text", "multimodal"])
def test_actual_runtime_embedding_request_schema(family, input_format):
    if family == "sglang":
        protocol = pytest.importorskip("sglang.srt.entrypoints.openai.protocol")
        validate = protocol.EmbeddingRequest.model_validate
    else:
        protocol = pytest.importorskip("vllm.entrypoints.pooling.embed.protocol")
        from pydantic import TypeAdapter

        validate = TypeAdapter(protocol.EmbeddingRequest).validate_python

    codec = EmbeddingHTTPCodec(
        family,
        "model",
        768,
        {
            "http_input_format": input_format,
            "media_mime_types": ["video/mp4", "image/png"] if input_format == "multimodal" else [],
        },
    )
    parts = [EmbeddingPart(EmbeddingPartSpec("text"), "query")]
    rows = [codec.encode(EmbeddingContent(tuple(parts)))]
    if input_format == "multimodal":
        parts.append(EmbeddingPart(EmbeddingPartSpec("video", "video/mp4"), b"encoded bytes"))
        rows.append(codec.encode(EmbeddingContent(tuple(parts))))
    payload = json.loads(codec.body(rows))
    parsed = validate(payload)
    field = "messages" if family == "vllm" and input_format == "multimodal" else "input"
    assert len(getattr(parsed, field)) == len(rows)
    if family == "vllm" and input_format == "multimodal":
        assert isinstance(parsed, protocol.EmbeddingBatchChatRequest)
        assert parsed.messages[1][0]["content"][1]["video_url"]["url"].startswith("data:video/mp4;base64,")
    if family == "sglang" and input_format == "multimodal":
        assert parsed.input[1].video.startswith("data:video/mp4;base64,")


@pytest.mark.parametrize("mime,format", [("audio/wav", "wav"), ("audio/mpeg", "mp3")])
def test_vllm_actual_audio_schema(mime, format):
    protocol = pytest.importorskip("vllm.entrypoints.pooling.embed.protocol")
    from pydantic import TypeAdapter

    codec = EmbeddingHTTPCodec("vllm", "model", 768, {"http_input_format": "multimodal", "media_mime_types": [mime]})
    row = codec.encode(EmbeddingContent((EmbeddingPart(EmbeddingPartSpec("audio", mime), b"encoded audio"),)))
    parsed = TypeAdapter(protocol.EmbeddingRequest).validate_python(json.loads(codec.body([row])))
    assert parsed.messages[0][0]["content"][0]["input_audio"]["format"] == format
