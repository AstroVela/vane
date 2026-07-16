import sys
from types import SimpleNamespace

from vane.ai.providers.transformers import (
    TransformersTextClassifier,
    TransformersTextEmbedder,
    TransformersTextEmbedderDescriptor,
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

    assert calls == [("trusted-model", {"trust_remote_code": False, "backend": "torch"})]


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
        embed_options={"batch_size": 8, "revision": "pinned-revision", "trust_remote_code": True},
    )

    assert descriptor.get_dimensions().size == 384
    descriptor.instantiate()

    assert auto_config_calls == [
        (
            "reviewed-model",
            {"trust_remote_code": True, "revision": "pinned-revision"},
        )
    ]
    assert sentence_transformer_calls == [
        (
            "reviewed-model",
            {"trust_remote_code": True, "backend": "torch", "revision": "pinned-revision"},
        )
    ]


def test_remote_code_is_enabled_only_by_the_boolean_true(monkeypatch):
    sentence_transformer_calls = []
    pipeline_calls = []

    class FakeSentenceTransformer:
        def __init__(self, model, **options):
            sentence_transformer_calls.append((model, options))

        def eval(self):
            return self

    def fake_pipeline(task, **options):
        pipeline_calls.append((task, options))
        return object()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=fake_pipeline))

    TransformersTextEmbedder("reviewed-model", trust_remote_code="true")
    TransformersTextClassifier("reviewed-classifier", trust_remote_code="true")

    assert sentence_transformer_calls[0][1]["trust_remote_code"] is False
    assert pipeline_calls[0][1]["trust_remote_code"] is False
