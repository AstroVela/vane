# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest

from vane.ai.providers.transformers import (
    TransformersProvider,
    TransformersTextEmbedder,
    TransformersTextEmbedderDescriptor,
)

_PINNED_REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: TransformersProvider().get_text_embedder(options={"trust_remote_code": True}),
            id="provider",
        ),
        pytest.param(
            lambda: TransformersTextEmbedderDescriptor(
                model="reviewed-model",
                options={"trust_remote_code": True},
            ),
            id="descriptor",
        ),
    ],
)
def test_remote_code_requires_a_pinned_revision_at_descriptor_boundary(factory):
    with pytest.raises(ValueError, match="requires a pinned revision"):
        factory()


@pytest.mark.parametrize("revision", ["main", "latest", "deadbee", "g" * 40])
def test_remote_code_rejects_mutable_or_invalid_revisions(revision):
    with pytest.raises(ValueError, match="full 40-character commit SHA"):
        TransformersTextEmbedderDescriptor(
            model="reviewed-model",
            options={"revision": revision, "trust_remote_code": True},
        )


def test_sentence_transformer_remote_code_is_disabled_by_default(monkeypatch):
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model, **options):
            calls.append((model, options))

        def eval(self):
            return self

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    TransformersTextEmbedder("trusted-model")

    assert calls == [("trusted-model", {"trust_remote_code": False, "backend": "torch", "device": "cpu"})]


def test_remote_code_requires_an_explicit_option(monkeypatch):
    sentence_transformer_calls = []
    auto_config_calls = []

    class FakeSentenceTransformer:
        def __init__(self, model, **options):
            sentence_transformer_calls.append((model, options))

        def eval(self):
            return self

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model, **options):
            auto_config_calls.append((model, options))
            return SimpleNamespace(hidden_size=384)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoConfig=FakeAutoConfig))

    descriptor = TransformersTextEmbedderDescriptor(
        model="reviewed-model",
        dimensions=384,
        options={"revision": _PINNED_REVISION, "trust_remote_code": True},
    )

    assert descriptor.get_dimensions().size == 384
    descriptor.instantiate()

    assert auto_config_calls == []
    assert sentence_transformer_calls == [
        (
            "reviewed-model",
            {
                "trust_remote_code": True,
                "backend": "torch",
                "revision": _PINNED_REVISION,
                "device": "cpu",
            },
        )
    ]


def test_remote_code_string_does_not_enable_remote_code(monkeypatch):
    sentence_transformer_calls = []

    class FakeSentenceTransformer:
        def __init__(self, model, **options):
            sentence_transformer_calls.append((model, options))

        def eval(self):
            return self

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    TransformersTextEmbedder("reviewed-model", trust_remote_code="true")

    assert sentence_transformer_calls[0][1]["trust_remote_code"] is False
