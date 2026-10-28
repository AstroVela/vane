# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""CLAP SDK contract without model downloads."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from vane.ai import AudioClip
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.providers._clap import CLAP_MODEL, ClapAudioEmbedderDescriptor


@pytest.fixture
def sdk(monkeypatch):
    state = SimpleNamespace(loads=[], calls=[], cuda=True, rate=48000, dim=512)

    class Inputs(dict):
        def to(self, device):
            return self

    class Tensor:
        def __init__(self, count):
            self.count = count

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.ones((self.count, 512), dtype=np.float32)

    class Processor:
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split() + ["CLS", "SEP"])

        @property
        def feature_extractor(self):
            return SimpleNamespace(sampling_rate=state.rate, nb_max_samples=480000)

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            state.loads.append(("processor", model, kwargs))
            return cls()

        def __call__(self, **kwargs):
            state.calls.append(kwargs)
            return Inputs(count=len(kwargs.get("audios", kwargs.get("text", []))))

    class Model:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            state.loads.append(("model", model, kwargs))
            return cls()

        @property
        def config(self):
            return SimpleNamespace(projection_dim=state.dim, audio_config=SimpleNamespace(enable_fusion=False))

        def to(self, device):
            state.device = device
            return self

        def eval(self):
            return self

        def get_audio_features(self, count):
            return Tensor(count)

        def get_text_features(self, count):
            return Tensor(count)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(is_available=lambda: state.cuda)),
    )
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(ClapProcessor=Processor, ClapModel=Model))
    return state


def descriptor(**options):
    return ClapAudioEmbedderDescriptor(
        CLAP_MODEL, options={"device": "cpu", "revision": "a" * 40, "local_files_only": True, **options}
    )


def test_worker_loading_and_pairing(sdk):
    model = descriptor(cache_folder="/model-cache").instantiate()
    expected = {"revision": "a" * 40, "local_files_only": True, "cache_dir": "/model-cache"}
    assert sdk.loads == [("processor", CLAP_MODEL, expected), ("model", CLAP_MODEL, expected)]
    clip = AudioClip(np.tile([0.25, 0.75], (400, 1)), 48000)
    result = model.embed_audio([clip, clip])
    assert len(result) == 2 and result[0].shape == (512,)
    np.testing.assert_array_equal(sdk.calls[0]["audios"][0], np.full(400, 0.5, dtype=np.float32))
    assert sdk.calls[0]["sampling_rate"] == 48000
    assert sdk.calls[0]["padding"] == "repeatpad"
    assert len(model.embed_text(["a dog barking"])) == 1
    assert sdk.calls[-1]["truncation"] is False
    assert len(sdk.loads) == 2


def test_no_implicit_truncation(sdk):
    model = descriptor().instantiate()
    with pytest.raises(EmbeddingConfigurationError, match="77 tokens"):
        model.embed_text(["word " * 76])
    with pytest.raises(EmbeddingConfigurationError, match="ten seconds"):
        model.embed_audio([AudioClip(np.zeros((480001, 1)), 48000)])
    assert sdk.calls == []


@pytest.mark.parametrize(
    "field,value,options,match",
    [("cuda", False, {"device": "cuda"}, "CUDA"), ("rate", 16000, {}, "processor"), ("dim", 128, {}, "model")],
)
def test_worker_rejects_incompatible_runtime(sdk, field, value, options, match):
    setattr(sdk, field, value)
    with pytest.raises(EmbeddingConfigurationError, match=match):
        descriptor(**options).instantiate()
    assert sdk.calls == []
