# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared model ownership and the service's bounded execution lifecycle."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from tests.fast.test_mlx_provider import Tokenizer
from vane.ai._audio_embedding import AudioClip
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai._video_embedding import VideoClip

pytest_plugins = ("mlx_sdk_fixture",)


def content(*parts):
    return EmbeddingContent(tuple(EmbeddingPart(EmbeddingPartSpec(kind), value) for kind, value in parts))


def test_one_full_media_core_reuses_weights_processor_and_original_text_tokenizer(media_sdk, monkeypatch):
    from vane.ai.providers.mlx import _MLXModelCore

    tokenizer = Tokenizer()
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    image = np.zeros((3, 5, 3), np.uint8)
    audio = AudioClip(np.zeros((161, 1), np.float32), 16000)
    video = VideoClip((image,), (0.125,), (7,))
    cases = [
        ("text_document", content(("text", "same text"))),
        ("text_query", content(("text", "same text"))),
        ("image", content(("image", image))),
        ("audio", content(("audio", audio))),
        ("video", content(("video", video))),
        ("multimodal", content(("text", "scene"), ("image", image), ("audio", audio), ("video", video))),
    ]
    core = _MLXModelCore({})
    model, processor, options = core._model, core._processor, dict(core._options)
    try:
        for encoding, row in cases:
            result = core.encode(row, encoding)
            assert result.dtype == np.float32 and result.shape == (768,)
            assert core._model is model and core._processor is processor
            assert core._options == options
        assert tokenizer.texts == ["title: none | text: same text", "task: search result | query: same text"]
        assert media_sdk.loads == media_sdk.processor_loads == 1
        assert len(media_sdk.model_calls) == 6
    finally:
        core.close()
        core.close()
    assert core._model is core._processor is core._tokenizer is None
    assert media_sdk.clears == 1
