# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native device and GPU admission contracts without importing model SDKs."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vane.ai.provider import load_provider
from vane.ai.providers.vllm import _build_native_vllm_options_argument


def make_descriptor(family, entry, options):
    provider = load_provider(family)
    if entry == "embedding":
        return provider.get_text_embedder("configured-model", 2, options=options)
    if entry == "media_prompt":
        return provider.get_prompter("configured-model", options={**options, "media_mime_types": ["image/png"]})
    return provider.get_prompter("configured-model", options=options)


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("entry", ["embedding", "media_prompt", "text_prompt"])
def test_zero_gpu_native_execution_is_rejected_without_loading_sdk(family, entry, monkeypatch):
    monkeypatch.setitem(sys.modules, family, None)
    with pytest.raises(ValueError, match="gpus_per_actor"):
        make_descriptor(family, entry, {"gpus_per_actor": 0})


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("entry", ["embedding", "media_prompt", "text_prompt"])
@pytest.mark.parametrize("device", ["cpu", "cuda:7", "auto", "xpu"])
def test_unsupported_device_configuration_fails_during_planning(family, entry, device, monkeypatch):
    monkeypatch.setitem(sys.modules, family, None)
    with pytest.raises(ValueError, match="device"):
        make_descriptor(family, entry, {"engine_args": {"device": device}})


@pytest.mark.parametrize("entry", ["embedding", "media_prompt", "text_prompt"])
def test_vllm_device_keyword_is_not_an_async_engine_argument(entry):
    with pytest.raises(ValueError, match="engine_args.device"):
        make_descriptor("vllm", entry, {"engine_args": {"device": "cuda"}})


@pytest.mark.parametrize("entry", ["embedding", "media_prompt", "text_prompt"])
@pytest.mark.parametrize("device", [None, "cuda", "cuda:0"])
def test_sglang_retains_worker_local_cuda_selection(entry, device):
    descriptor = make_descriptor("sglang", entry, {"engine_args": {"device": device}})
    assert descriptor.get_options()["engine_args"]["device"] == device
    assert descriptor.get_options()["gpus_per_actor"] == 1


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("entry", ["embedding", "media_prompt", "text_prompt"])
def test_parallel_engines_reserve_their_full_gpu_count(family, entry):
    keys = ("tensor_parallel_size", "pipeline_parallel_size") if family == "vllm" else ("tp_size", "pp_size")
    options = {"engine_args": dict.fromkeys(keys, 2)}
    descriptor = make_descriptor(family, entry, options)
    if entry == "text_prompt":
        assert descriptor.get_options()["gpus_per_actor"] == 4
    else:
        assert descriptor.get_udf_options().num_gpus == 4
    with pytest.raises(ValueError, match="parallelism"):
        make_descriptor(family, entry, {**options, "gpus_per_actor": 1})


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_text_prompt_retains_single_gpu_fractional_reservations(family):
    descriptor = make_descriptor(family, "text_prompt", {"gpus_per_actor": 0.25})
    assert descriptor.get_options()["gpus_per_actor"] == 0.25


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("options", [{"gpus_per_actor": 0}, {"engine_args": {"device": "cpu"}}])
def test_native_wire_options_cannot_bypass_device_admission(family, options):
    envelope = _build_native_vllm_options_argument(options)
    envelope["engine"] = family
    normalize = importlib.import_module(f"vane.execution.{family}").normalize_options
    with pytest.raises(ValueError, match="device|gpus_per_actor"):
        normalize(envelope)


def install_device_sdk(monkeypatch, family, device):
    sdk = ModuleType(family)
    sdk.AsyncEngineArgs = SimpleNamespace
    sdk.AsyncLLMEngine = SimpleNamespace(from_engine_args=lambda *_args, **_kwargs: pytest.fail("loaded weights"))
    sdk.Engine = lambda **_kwargs: pytest.fail("loaded weights")
    monkeypatch.setitem(sys.modules, family, sdk)
    if family == "vllm":
        platform = ModuleType("vllm.platforms")
        platform.current_platform = SimpleNamespace(device_type=device)
        monkeypatch.setitem(sys.modules, "vllm.platforms", platform)
    else:
        utils = ModuleType("sglang.srt.utils")
        utils.get_device = lambda: device
        monkeypatch.setitem(sys.modules, "sglang.srt.utils", utils)


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("device", ["cpu", "xpu"])
@pytest.mark.parametrize("entry", ["embedding", "text_prompt"])
def test_worker_rejects_non_cuda_sdk_platform_before_loading_weights(family, device, entry, monkeypatch):
    install_device_sdk(monkeypatch, family, device)
    if entry == "embedding":
        # Platform validation must precede processor/model I/O as well.
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(ValueError, match="CUDA"):
            make_descriptor(family, entry, {}).instantiate()
    else:
        module = importlib.import_module(f"vane.execution.{family}")
        cls = module.LocalVLLMExecutor if family == "vllm" else module.SGLangLocalExecutor
        executor = cls.__new__(cls)
        executor.model = "configured-model"
        executor.engine_args = {}
        executor.llm = None
        with pytest.raises(ValueError, match="CUDA"):
            executor._create_engine()
        assert executor.llm is None


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_sdk_initialization_failure_is_not_retried_on_cpu(family, monkeypatch):
    install_device_sdk(monkeypatch, family, "cuda")
    sdk = sys.modules[family]
    calls = []

    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("GPU initialization failed")

    if family == "vllm":
        sdk.AsyncLLMEngine.from_engine_args = fail
    else:
        sdk.Engine = fail
    module = importlib.import_module(f"vane.execution.{family}")
    cls = module.LocalVLLMExecutor if family == "vllm" else module.SGLangLocalExecutor
    executor = cls.__new__(cls)
    executor.model = "configured-model"
    executor.engine_args = {}
    executor.llm = None
    with pytest.raises(RuntimeError, match="GPU initialization failed"):
        executor._create_engine()
    assert len(calls) == 1 and executor.llm is None
