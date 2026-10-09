# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native SDK contracts without model downloads or a serving endpoint."""

from __future__ import annotations

import asyncio
import pickle
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vane.ai._media import PromptMedia
from vane.ai._video_embedding import VideoClip
from vane.ai.functions import _prepare_embed_call, _prepare_prompt_call
from vane.ai.protocols import NativeInferencePlan, PrompterDescriptor
from vane.ai.provider import load_provider


def install_sdk(setitem, *, mock_transformers=True):
    """Doubles for the documented Python APIs, installed inside each worker."""
    state = SimpleNamespace(loads=[], engines=[], calls=[], processing=[], finish="stop", fail=False, requests_closed=0)
    state.config = SimpleNamespace()

    class Processor:
        model_input_names = ["input_ids", "attention_mask"]

        def __init__(self):
            self.tokenizer = self

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            state.loads.append((model, kwargs))
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            state.messages = messages
            state.template = kwargs
            return "rendered:" + "\n".join(part.get("text", "") for item in messages for part in item["content"])

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return [10, 20, 30]

        def __call__(self, **kwargs):
            state.processing.append(kwargs)
            result = {"input_ids": np.array([[10, 20, 30]]), "attention_mask": np.array([[1, 1, 1]])}
            if "videos" in kwargs:
                result.update(pixel_values_videos=np.array([[1.0]]), video_grid_thw=np.array([[1, 2, 2]]))
            if "images" in kwargs:
                result.update(pixel_values=np.array([[1.0]]), image_grid_thw=np.array([[1, 2, 2]]))
            return result

    class EmbeddingRequest(SimpleNamespace):
        def __init__(self, *, input_ids, dimensions=None, image_data=None, video_data=None):
            super().__init__(input_ids=input_ids, dimensions=dimensions, image_data=image_data, video_data=video_data)

    class Engine:
        def __init__(self, **kwargs):
            self.args = kwargs
            self.loop = asyncio.get_running_loop()
            self.closed = 0
            self.tokenizer_manager = self
            state.engines.append(self)

        @classmethod
        def from_engine_args(cls, args):
            return cls(**vars(args))

        async def generate(self, prompt, sampling_params, request_id, **kwargs):
            assert self.loop is asyncio.get_running_loop()
            state.calls.append(("generate", prompt, vars(sampling_params), kwargs))
            if state.fail:
                raise ValueError("engine failure")
            yield SimpleNamespace(
                finished=True, outputs=[SimpleNamespace(text='{"answer":"ok"}', finish_reason=state.finish)]
            )

        async def encode(self, prompt, pooling_params, request_id, **kwargs):
            assert self.loop is asyncio.get_running_loop()
            state.calls.append(("encode", prompt, vars(pooling_params), kwargs))
            if state.fail:
                raise ValueError("engine failure")
            yield SimpleNamespace(finished=True, outputs=SimpleNamespace(data=np.array([3.0, 4.0])))

        async def async_generate(self, **kwargs):
            assert self.loop is asyncio.get_running_loop()
            state.calls.append(("generate", kwargs))
            if not kwargs.get("image_data") and not kwargs.get("video_data"):
                assert "input_ids" in kwargs and "prompt" not in kwargs
            if state.fail:
                raise ValueError("engine failure")
            return {"text": '{"answer":"ok"}', "meta_info": {"finish_reason": {"type": state.finish}}}

        async def generate_request(self, request, http_request):
            assert self.loop is asyncio.get_running_loop() and http_request is None
            assert isinstance(request, EmbeddingRequest)
            state.calls.append(("encode", vars(request)))
            try:
                if state.fail:
                    raise ValueError("engine failure")
                yield {"embedding": [3.0, 4.0]}
            finally:
                state.requests_closed += 1

        def shutdown(self):
            assert self.loop is asyncio.get_running_loop()
            self.closed += 1

    class VideoIO:
        def __init__(self, image_io, num_frames):
            self.num_frames = num_frames

        def load_bytes(self, data):
            state.video_bytes = data
            return SimpleNamespace(media=(np.zeros((2, 4, 4, 3), dtype=np.uint8), {"fps": 2, "frames_indices": [0, 1]}))

    for name, entries in {
        "transformers": {
            "AutoProcessor": Processor,
            "AutoConfig": SimpleNamespace(from_pretrained=lambda *args, **kwargs: state.config),
            "PreTrainedTokenizerBase": type("TokenizerBase", (), {}),
        },
        "transformers.video_utils": {"VideoMetadata": SimpleNamespace},
        "vllm": {
            "AsyncEngineArgs": SimpleNamespace,
            "AsyncLLMEngine": Engine,
            "PoolingParams": SimpleNamespace,
            "SamplingParams": SimpleNamespace,
        },
        "vllm.sampling_params": {"StructuredOutputsParams": SimpleNamespace},
        "vllm.multimodal.media": {"ImageMediaIO": SimpleNamespace, "VideoMediaIO": VideoIO},
        "sglang": {"Engine": Engine},
        "sglang.srt.managers.io_struct": {"EmbeddingReqInput": EmbeddingRequest},
    }.items():
        if not mock_transformers and name.startswith("transformers"):
            continue
        module = ModuleType(name)
        module.__dict__.update(entries)
        setitem(sys.modules, name, module)
    return state


@pytest.fixture
def sdk(monkeypatch):
    return install_sdk(monkeypatch.setitem)


def clip():
    return VideoClip((np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8)), (0.125, 1.375), (4, 18))


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_models_and_dimensions_are_explicit_and_planning_is_lazy(family, sdk):
    provider = load_provider(family)
    with pytest.raises(ValueError, match="model"):
        provider.get_prompter()
    for model in (None, "", " "):
        with pytest.raises(ValueError, match="model"):
            provider.get_prompter(model=model)
        with pytest.raises(ValueError, match="model"):
            provider.get_prompter(model=model, options={"media_mime_types": ["video/mp4"]})
    assert isinstance(provider.get_prompter(model="configured-text-model"), NativeInferencePlan)
    for kind in ("text", "image", "video"):
        for model in (None, "", " "):
            with pytest.raises(ValueError, match="model"):
                getattr(provider, f"get_{kind}_embedder")(model, dimensions=17)
        with pytest.raises(ValueError, match="dimensions"):
            getattr(provider, f"get_{kind}_embedder")("configured-encoder")
        desc, dims, udf, *_ = _prepare_embed_call(
            family, "/models/custom-encoder", 17, "raise", {"gpus_per_actor": 2}, relation=True, input_kind=kind
        )
        assert pickle.loads(pickle.dumps(desc)).get_model() == "/models/custom-encoder"
        assert dims == 17 and udf.num_gpus == 2
    assert sdk.loads == [] and sdk.engines == []


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("options", [{"transport": "http"}, {"base_url": "http://localhost/v1"}, {"timeout": 12}])
def test_http_options_are_rejected(family, options):
    provider = load_provider(family)
    with pytest.raises((TypeError, ValueError)):
        provider.get_text_embedder("configured", 2, options=options)
    with pytest.raises((TypeError, ValueError)):
        provider.get_prompter("configured", options={"media_mime_types": ["video/mp4"], **options})


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("kind", ["text", "image", "video", "prompt"])
def test_explicit_processing_source_is_used_on_the_first_load(family, kind, sdk):
    key = "tokenizer" if family == "vllm" else "tokenizer_path"
    args = {key: "separate-processing-source", "revision": "model-revision"}
    if family == "vllm":
        args["tokenizer_revision"] = "processing-revision"
    options = {"engine_args": args, "gpus_per_actor": 0}
    provider = load_provider(family)
    if kind == "prompt":
        descriptor = provider.get_prompter(
            "weights-only-checkpoint", options={**options, "media_mime_types": ["image/png"]}
        )
    else:
        descriptor = getattr(provider, f"get_{kind}_embedder")("weights-only-checkpoint", 2, options=options)
    runtime = descriptor.instantiate()
    try:
        assert sdk.loads == [
            (
                "separate-processing-source",
                {
                    "revision": "processing-revision" if family == "vllm" else "model-revision",
                    "trust_remote_code": False,
                    "config": sdk.config,
                },
            )
        ]
        assert descriptor.get_model() == "weights-only-checkpoint" and sdk.engines == []
    finally:
        asyncio.run(runtime.aclose())


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("processing", [False, True])
def test_native_text_embedding_formats_tokenizer_messages(family, processing, sdk, monkeypatch):
    transformers = sys.modules["transformers"]

    class Tokenizer(transformers.AutoProcessor, transformers.PreTrainedTokenizerBase):
        def __init__(self):
            # A text tokenizer does not have a nested .tokenizer attribute.
            pass

        def apply_chat_template(self, messages, **kwargs):
            sdk.messages = messages
            return "\n".join(message["role"] + ": " + message["content"] for message in messages)

    monkeypatch.setattr(transformers, "AutoProcessor", Tokenizer)
    options = {"gpus_per_actor": 0, "instruction": "Find related items"}
    if processing:
        options["processor_kwargs"] = {"add_special_tokens": False}
    descriptor = load_provider(family).get_text_embedder("configured-text-encoder", 2, options=options)

    async def run():
        runtime = descriptor.instantiate()
        try:
            assert await runtime.embed_text(["query"]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert sdk.messages == [
        {"role": "system", "content": "Find related items"},
        {"role": "user", "content": "query"},
    ]
    if family == "sglang":
        assert sdk.calls[0][1]["input_ids"] == [10, 20, 30]
        assert sdk.processing[0]["text"] == ["system: Find related items\nuser: query"]
        assert sdk.processing[0]["add_special_tokens"] is False
    else:
        assert sdk.calls[0][1]["prompt"] == "system: Find related items\nuser: query"


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("kind", ["image", "video"])
def test_tokenizer_rejects_media_before_engine_initialization(family, kind, sdk, monkeypatch):
    transformers = sys.modules["transformers"]
    monkeypatch.setattr(
        transformers,
        "AutoProcessor",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: transformers.PreTrainedTokenizerBase()),
    )
    descriptor = getattr(load_provider(family), f"get_{kind}_embedder")(
        "configured-text-encoder", 2, options={"gpus_per_actor": 0}
    )

    async def run():
        runtime = descriptor.instantiate()
        try:
            with pytest.raises(ValueError, match="tokenizer cannot format"):
                await getattr(runtime, f"embed_{kind}")([clip().frames[0] if kind == "image" else clip()])
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert sdk.engines == []


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("kind", ["text", "image", "video"])
def test_native_embedding_enforces_input_bytes_before_preprocessing(family, kind, sdk):
    value = "中文" if kind == "text" else clip().frames[0] if kind == "image" else clip()
    size = 6 if kind == "text" else 192 if kind == "image" else 384

    async def run(limit):
        descriptor = getattr(load_provider(family), f"get_{kind}_embedder")(
            "configured-encoder", 2, options={"gpus_per_actor": 0, "max_input_bytes": limit}
        )
        runtime = descriptor.instantiate()
        try:
            return await getattr(runtime, f"embed_{kind}")([value])
        finally:
            await runtime.aclose()

    with pytest.raises(ValueError, match="max_input_bytes"):
        asyncio.run(run(1))
    with pytest.raises(ValueError, match="max_input_bytes"):
        asyncio.run(run(size - 1))
    assert sdk.engines == [] and sdk.processing == [] and not hasattr(sdk, "messages")
    assert asyncio.run(run(size)) == [[3, 4]]


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("kind", ["text", "image", "video"])
def test_native_embedding_preserves_inputs_and_reuses_engine(family, kind, sdk):
    options = {"gpus_per_actor": 0, "instruction": "Find related items", "paired_image_queries": True}
    desc = getattr(load_provider(family), f"get_{kind}_embedder")("configured-encoder", 2, options=options)
    value = "query" if kind == "text" else clip().frames[0] if kind == "image" else clip()

    async def run():
        runtime = desc.instantiate()
        try:
            assert await getattr(runtime, f"embed_{kind}")([value, value]) == [[3, 4], [3, 4]]
        finally:
            await runtime.aclose()
            await runtime.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            runtime._ensure_engine()

    asyncio.run(run())
    assert len(sdk.engines) == 1 and sdk.engines[0].closed == 1
    assert len(sdk.calls) == 2 and sdk.loads[0][0] == "configured-encoder"
    # Fixed-width models reject a dimensions override even at their natural width.
    assert (sdk.calls[0][2] if family == "vllm" else sdk.calls[0][1])["dimensions"] is None
    if kind == "video":
        if family == "vllm":
            frames, metadata = sdk.calls[0][1]["multi_modal_data"]["video"][0]
            assert frames.shape == (2, 8, 8, 3)
        else:
            metadata = vars(sdk.processing[0]["videos_kwargs"]["video_metadata"][0])
            assert sdk.processing[0]["do_sample_frames"] is False
            assert sdk.calls[0][1]["video_data"][0]["format"] == "processor_output"
            assert sdk.calls[0][1]["input_ids"] == sdk.calls[0][1]["video_data"][0]["input_ids"][0].tolist()
        assert metadata["frames_indices"] == [125000, 1375000] and metadata["fps"] == 1000000
    if family == "sglang":
        assert sdk.requests_closed == 2


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_embedding_dimension_override_is_explicit(family, sdk):
    descriptor = load_provider(family).get_text_embedder(
        "matryoshka-model", 2, options={"gpus_per_actor": 0, "supports_overriding_dimensions": True}
    )

    async def run():
        runtime = descriptor.instantiate()
        try:
            assert await runtime.embed_text(["query"]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert (sdk.calls[0][2] if family == "vllm" else sdk.calls[0][1])["dimensions"] == 2


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_native_prompt_forwards_model_settings_and_structured_output(family, sdk):
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    options = {
        "media_mime_types": ["video/mp4"],
        "gpus_per_actor": 0,
        "max_tokens": 128,
        "engine_args": {"dtype": "float16"},
        "generate_args": {"sampling_params": {"top_k": 7}},
        "chat_template": "custom template",
        "chat_template_kwargs": {"enable_thinking": False},
    }
    desc, udf, *_ = _prepare_prompt_call(
        family, "/models/custom-vlm", schema, "system", False, "raise", options, relation=True
    )
    assert isinstance(desc, PrompterDescriptor) and udf.num_gpus == 0

    async def run():
        runtime = desc.instantiate()
        try:
            assert await runtime.prompt(("question", PromptMedia(b"video-bytes", "video/mp4"))) == '{"answer":"ok"}'
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert sdk.engines[0].args["dtype"] == "float16" and sdk.engines[0].closed == 1
    assert sdk.template["chat_template"] == "custom template" and sdk.template["enable_thinking"] is False
    if family == "vllm":
        assert sdk.video_bytes == b"video-bytes"
        assert sdk.calls[0][2]["top_k"] == 7 and sdk.calls[0][2]["max_tokens"] == 128
        assert sdk.calls[0][2]["structured_outputs"].json == schema
        assert sdk.calls[0][3]["tokenization_kwargs"] == {"add_special_tokens": False}
    else:
        assert sdk.calls[0][1]["video_data"] == [b"video-bytes"]
        assert sdk.calls[0][1]["sampling_params"]["top_k"] == 7
        assert sdk.calls[0][1]["sampling_params"]["max_new_tokens"] == 128


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("mode", ["engine_error", "length", "budget", "mime"])
def test_failures_are_explicit_and_engine_is_released(family, mode, sdk):
    desc = load_provider(family).get_prompter(
        "model", options={"media_mime_types": ["video/mp4"], "gpus_per_actor": 0, "max_input_bytes": 32}
    )
    sdk.fail = mode == "engine_error"
    sdk.finish = "length" if mode == "length" else "stop"

    async def run():
        runtime = desc.instantiate()
        try:
            with pytest.raises(ValueError):
                await runtime.prompt(
                    (
                        "q" * (100 if mode == "budget" else 1),
                        PromptMedia(b"video", "image/png" if mode == "mime" else "video/mp4"),
                    )
                )
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert all(engine.closed == 1 for engine in sdk.engines)
    if mode in {"budget", "mime"}:
        assert not sdk.engines


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_engine_resource_admission_and_input_ownership(family):
    options = {"engine_args": {"tensor_parallel_size" if family == "vllm" else "tp_size": 2}}
    desc = load_provider(family).get_text_embedder("configured", 2, options=options)
    assert desc.get_udf_options().num_gpus == 2
    with pytest.raises(ValueError, match="parallelism"):
        load_provider(family).get_text_embedder("configured", 2, options={**options, "gpus_per_actor": 1})
    for invalid in (
        {"processor_kwargs": {"videos_kwargs": {"do_sample_frames": True}}},
        {"processor_kwargs": {"text_kwargs": {"padding": True}}},
        {"processor_kwargs": {"text_kwargs": {"truncation": True}}},
        {"processor_kwargs": {"text_kwargs": {"return_tensors": "np"}}},
        {"engine_args": {"model": "other"}},
        {"pooling_args": {"dimensions": 9}},
    ):
        with pytest.raises(ValueError):
            load_provider(family).get_video_embedder("configured", 2, options=invalid)


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("processing", [{"add_special_tokens": True}, {"text_kwargs": {"add_special_tokens": True}}])
def test_native_chat_template_owns_special_tokens(family, processing):
    with pytest.raises(ValueError, match="chat templates own special tokens"):
        load_provider(family).get_text_embedder("configured", 2, options={"processor_kwargs": processing})


@pytest.mark.parametrize("tokenization", [{"add_special_tokens": True}, None])
def test_vllm_prompt_chat_template_owns_special_tokens(tokenization):
    with pytest.raises(ValueError, match="chat templates own special tokens"):
        load_provider("vllm").get_prompter(
            "configured",
            options={"media_mime_types": ["image/png"], "generate_args": {"tokenization_kwargs": tokenization}},
        )


@pytest.mark.parametrize("failure", [False, True])
def test_sglang_embedding_request_is_closed_on_success_or_failure(sdk, failure):
    sdk.fail = failure

    async def run():
        runtime = (
            load_provider("sglang").get_text_embedder("configured", 2, options={"gpus_per_actor": 0}).instantiate()
        )
        try:
            if failure:
                with pytest.raises(ValueError, match="engine failure"):
                    await runtime.embed_text(["query"])
            else:
                assert await runtime.embed_text(["query"]) == [[3, 4]]
            assert sdk.requests_closed == 1
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert sdk.engines[0].closed == 1


def test_sglang_text_token_budget_is_checked_before_engine_startup(sdk):
    async def run():
        runtime = (
            load_provider("sglang")
            .get_text_embedder("configured", 2, options={"gpus_per_actor": 0, "engine_args": {"context_length": 2}})
            .instantiate()
        )
        try:
            with pytest.raises(ValueError, match="context_length"):
                await runtime.embed_text(["query"])
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert sdk.engines == []


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("times", [(0.0,), (0.0, 0.5, 1.5, 2.0), (0.0, 0.5, 0.5, 1.0), (1.0, 0.5)])
def test_scalar_video_timing_rejects_unrepresentable_clips_before_engine_startup(family, times, sdk):
    sys.modules["transformers"].AutoProcessor.model_input_names = ["input_ids", "second_per_grid_ts"]
    frames = tuple(np.zeros((8, 8, 3), dtype=np.uint8) for _ in times)

    async def run():
        runtime = load_provider(family).get_video_embedder("configured", 2, options={"gpus_per_actor": 0}).instantiate()
        try:
            with pytest.raises(ValueError, match="at least two uniformly spaced video frames"):
                await runtime.embed_video([VideoClip(frames, times, tuple(range(len(times))))])
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert sdk.engines == [] and sdk.calls == [] and sdk.processing == []


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("processing", [False, True])
def test_native_embedding_with_real_text_tokenizer(family, processing, monkeypatch):
    transformers = pytest.importorskip("transformers", minversion="4.57.1")
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE

    vocabulary = ["<unk>", "<pad>"] + sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(BPE({word: index for index, word in enumerate(vocabulary)}, [], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}",
    )
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: tokenizer)
    descriptor = load_provider(family).get_text_embedder(
        "configured-text-encoder",
        2,
        options={
            "gpus_per_actor": 0,
            "instruction": "Find related items",
            "processor_kwargs": {"add_special_tokens": False} if processing else {},
        },
    )

    async def run():
        runtime = descriptor.instantiate()
        try:
            assert await runtime.embed_text(["query"]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    expected_text = "system: Find related items\nuser: query\n"
    if family == "sglang":
        assert state.calls[0][1]["input_ids"] == tokenizer.encode(expected_text, add_special_tokens=False)
        assert "prompt" not in state.calls[0][1]
    else:
        assert state.calls[0][1]["prompt"] == expected_text
        assert state.calls[0][3]["tokenization_kwargs"] == {"add_special_tokens": False}


@pytest.fixture
def bos_tokenizer():
    transformers = pytest.importorskip("transformers", minversion="4.57.1")
    from tokenizers import Tokenizer, pre_tokenizers, processors
    from tokenizers.models import WordLevel

    backend = Tokenizer(WordLevel({"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        chat_template="{{ bos_token }}{{ messages[-1]['content'] }}{{ eos_token }}",
    )


@pytest.mark.parametrize("processor_kind", ["tokenizer", "gemma3"])
def test_sglang_text_only_prompt_preserves_real_template_tokens(tmp_path, processor_kind, bos_tokenizer, monkeypatch):
    import transformers

    tokenizer = bos_tokenizer
    if processor_kind == "gemma3":
        pytest.importorskip("torchvision")
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=bos_tokenizer.backend_tokenizer,
            unk_token=bos_tokenizer.unk_token,
            bos_token=bos_tokenizer.bos_token,
            eos_token=bos_tokenizer.eos_token,
            chat_template=bos_tokenizer.chat_template,
            extra_special_tokens={
                "image_token": "<image_soft_token>",
                "boi_token": "<start_of_image>",
                "eoi_token": "<end_of_image>",
            },
        )
        processor = transformers.Gemma3Processor(
            image_processor=transformers.Gemma3ImageProcessor(),
            tokenizer=tokenizer,
            chat_template="{{ bos_token }}{{ messages[-1]['content'][0]['text'] }}{{ eos_token }}",
        )
        transformers.Gemma3Config().save_pretrained(tmp_path)
        processor.save_pretrained(tmp_path)
    else:
        transformers.LlamaConfig().save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    descriptor = load_provider("sglang").get_prompter(
        str(tmp_path), options={"media_mime_types": ["image/png"], "gpus_per_actor": 0, "max_tokens": 64}
    )

    async def run():
        runtime = descriptor.instantiate()
        try:
            assert await runtime.prompt(("hello",)) == '{"answer":"ok"}'
        finally:
            await runtime.aclose()

    asyncio.run(run())
    expected = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}], tokenize=True, return_dict=False, add_generation_prompt=True
    )
    assert expected == [1, 3, 2]
    assert tokenizer.encode("<s>hello</s>") == [1, 1, 3, 2, 2]
    request = state.calls[0][1]
    assert request["input_ids"] == expected and "prompt" not in request
    assert request["image_data"] is None and request["video_data"] is None
    assert request["sampling_params"]["max_new_tokens"] == 64
    assert state.engines[0].closed == 1


@pytest.mark.parametrize("family", ["vllm", "sglang"])
def test_real_standalone_tokenizer_loads_with_weights_only_checkpoint(tmp_path, family, bos_tokenizer, monkeypatch):
    import transformers

    model, processing = tmp_path / "checkpoint", tmp_path / "processing"
    transformers.LlamaConfig().save_pretrained(model)
    bos_tokenizer.save_pretrained(processing)
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    key = "tokenizer" if family == "vllm" else "tokenizer_path"

    async def run():
        runtime = (
            load_provider(family)
            .get_text_embedder(str(model), 2, options={"gpus_per_actor": 0, "engine_args": {key: str(processing)}})
            .instantiate()
        )
        try:
            assert runtime._render([{"type": "text", "text": "hello"}], None) == "<s>hello</s>"
            assert await runtime.embed_text(["hello"]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert state.engines[0].args["model" if family == "vllm" else "model_path"] == str(model)
    assert state.engines[0].args[key] == str(processing)


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("kind", ["image", "video", "prompt"])
def test_vllm_engine_loads_separate_real_processor_with_original_weights(
    tmp_path, kind, remote, bos_tokenizer, monkeypatch
):
    from pathlib import Path

    import transformers

    pytest.importorskip("torchvision")
    processor = transformers.Qwen3VLProcessor(
        image_processor=transformers.Qwen2VLImageProcessor(),
        tokenizer=bos_tokenizer,
        video_processor=transformers.Qwen3VLVideoProcessor(),
        chat_template="hello",
    )
    checkpoint, processing = tmp_path / "checkpoint", tmp_path / "processing"
    transformers.Qwen3VLConfig().save_pretrained(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"fixture weights")
    (checkpoint / "preprocessor_config.json").write_text('{"image_processor_type": "UnrelatedProcessor"}')
    (checkpoint / "video_preprocessor_config.json").write_text('{"video_processor_type": "UnrelatedProcessor"}')
    processor.save_pretrained(processing)
    before = {path.name: path.read_bytes() for path in checkpoint.iterdir()}
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    model = "fixture/model" if remote else str(checkpoint)
    downloads = []

    def snapshot(name, **kwargs):
        downloads.append((name, kwargs))
        return str(checkpoint)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    original_config = transformers.AutoConfig.from_pretrained
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda name, **kwargs: original_config(checkpoint if name == model else name, **kwargs),
    )
    engine = sys.modules["vllm"].AsyncLLMEngine
    create = engine.from_engine_args
    model_views = []

    def create_with_processor(args):
        view = Path(args.model)
        model_views.append(view)
        # The SDK loads its processor from model, not tokenizer. Reload from
        # precisely that source while checking the checkpoint is unchanged.
        loaded = transformers.AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
        assert isinstance(loaded, transformers.Qwen3VLProcessor)
        assert loaded.chat_template == "hello"
        assert type(transformers.AutoImageProcessor.from_pretrained(view)) is type(loaded.image_processor)
        assert type(transformers.AutoVideoProcessor.from_pretrained(view)) is type(loaded.video_processor)
        assert (view / "model.safetensors").samefile(checkpoint / "model.safetensors")
        assert (view / "config.json").samefile(checkpoint / "config.json")
        assert args.served_model_name == model
        assert args.tokenizer == str(processing)
        return create(args)

    monkeypatch.setattr(engine, "from_engine_args", create_with_processor)
    options = {
        "gpus_per_actor": 0,
        "engine_args": {
            "tokenizer": str(processing),
            "revision": "weights-revision",
            "tokenizer_revision": "processor-revision",
            "download_dir": str(tmp_path / "cache"),
        },
    }
    provider = load_provider("vllm")
    descriptor = (
        provider.get_prompter(model, options={**options, "media_mime_types": ["image/png"]})
        if kind == "prompt"
        else getattr(provider, f"get_{kind}_embedder")(model, 2, options=options)
    )
    original_options = descriptor.get_options()

    async def run():
        runtime = descriptor.instantiate()
        try:
            runtime._ensure_engine()
            assert runtime._ensure_engine() is state.engines[0]
            assert model_views[0].is_dir()
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert len(model_views) == 1 and not model_views[0].exists()
    assert {path.name: path.read_bytes() for path in checkpoint.iterdir()} == before
    assert descriptor.get_model() == model and descriptor.get_options() == original_options
    assert downloads == (
        [(model, {"revision": "weights-revision", "cache_dir": str(tmp_path / "cache"), "token": None})]
        if remote
        else []
    )


@pytest.mark.parametrize("failure", ["save", "engine", "shutdown"])
def test_vllm_processor_model_view_is_cleaned_on_failures(tmp_path, failure, sdk, monkeypatch):
    from pathlib import Path

    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    directories = []

    def save(self, path):
        directories.append(Path(path))
        (Path(path) / "preprocessor_config.json").write_text("{}")
        if failure == "save":
            raise RuntimeError("save failed")

    monkeypatch.setattr(sys.modules["transformers"].AutoProcessor, "save_pretrained", save, raising=False)
    engine = sys.modules["vllm"].AsyncLLMEngine
    create = engine.from_engine_args

    def create_or_fail(args):
        if failure == "engine":
            raise RuntimeError("engine failed")
        result = create(args)
        result.shutdown = lambda: (_ for _ in ()).throw(RuntimeError("shutdown failed"))
        return result

    monkeypatch.setattr(engine, "from_engine_args", create_or_fail)

    async def run():
        runtime = (
            load_provider("vllm")
            .get_image_embedder(
                str(checkpoint), 2, options={"gpus_per_actor": 0, "engine_args": {"tokenizer": "separate-processor"}}
            )
            .instantiate()
        )
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            try:
                runtime._ensure_engine()
            finally:
                await runtime.aclose()
        assert runtime._model_directory is None

    asyncio.run(run())
    assert len(directories) == 1 and not directories[0].exists()
    assert list(checkpoint.iterdir()) == [checkpoint / "config.json"]


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize(
    "processing",
    [
        {},
        {"add_special_tokens": False},
        {"text_kwargs": {"padding": False, "truncation": False, "add_special_tokens": False, "return_tensors": "pt"}},
    ],
)
def test_native_template_token_ids_are_submitted_without_retokenization(
    tmp_path, family, processing, bos_tokenizer, monkeypatch
):
    import transformers

    transformers.LlamaConfig().save_pretrained(tmp_path)
    bos_tokenizer.save_pretrained(tmp_path)
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)

    async def run():
        runtime = (
            load_provider(family)
            .get_text_embedder(str(tmp_path), 2, options={"gpus_per_actor": 0, "processor_kwargs": processing})
            .instantiate()
        )
        try:
            assert await runtime.embed_text(["hello"]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    expected = bos_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}], tokenize=True, return_dict=False
    )
    assert expected == [1, 3, 2]
    assert bos_tokenizer.encode("<s>hello</s>") == [1, 1, 3, 2, 2]
    if family == "sglang":
        assert state.calls[0][1]["input_ids"] == expected
        assert "prompt" not in state.calls[0][1]
    else:
        # vLLM completion tokenization defaults to adding special tokens. Check
        # the actual engine argument, not the unrelated multimodal kwargs.
        call = state.calls[0]
        assert bos_tokenizer.encode(call[1]["prompt"], **call[3].get("tokenization_kwargs", {})) == expected


@pytest.fixture
def make_real_video_processor():
    transformers = pytest.importorskip("transformers", minversion="4.57.1")
    pytest.importorskip("torchvision")
    from tokenizers import Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import BPE

    specials = ["<unk>", "<pad>", "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
    vocabulary = specials + sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(BPE({word: index for index, word in enumerate(vocabulary)}, [], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = transformers.Qwen2TokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", additional_special_tokens=specials[2:]
    )

    def make(*, scalar_timing=False):
        processor = transformers.Qwen2_5_VLProcessor if scalar_timing else transformers.Qwen3VLProcessor
        video_processor = transformers.Qwen2VLVideoProcessor if scalar_timing else transformers.Qwen3VLVideoProcessor
        return processor(
            image_processor=transformers.Qwen2VLImageProcessor(),
            tokenizer=tokenizer,
            video_processor=video_processor(),
            chat_template="<|video_pad|>",
        )

    return make


# Transformers 5.12 builds modality IDs with np.array(torch.Tensor), whose
# NumPy 2 copy-keyword warning is unrelated to the processor-output contract.
@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning:transformers.processing_utils"
)
@pytest.mark.parametrize(
    "text_options",
    [
        {},
        {"text_kwargs": {}},
        {"text_kwargs": {"add_special_tokens": False}},
        {"text_kwargs": {"truncation": False}},
        {"text_kwargs": {"padding": False}},
        {"text_kwargs": {"padding": False, "truncation": False, "add_special_tokens": False, "return_tensors": "pt"}},
    ],
)
def test_sglang_with_real_video_processor_retains_timestamps(monkeypatch, text_options, make_real_video_processor):
    import transformers

    processor = make_real_video_processor()
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    descriptor = load_provider("sglang").get_video_embedder(
        "custom-local-checkpoint",
        2,
        options={
            "gpus_per_actor": 0,
            "processor_kwargs": {
                "videos_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": 8192}, "do_sample_frames": False},
                **text_options,
            },
        },
    )
    before = descriptor.get_options()
    frames = tuple(np.full((64, 64, 3), value, dtype=np.uint8) for value in (10, 20, 30))

    async def run():
        runtime = descriptor.instantiate()
        try:
            assert await runtime.embed_video([VideoClip(frames, (0.2, 0.6, 1.4), (2, 6, 14))]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    request = state.calls[0][1]
    text = processor.tokenizer.decode(request["input_ids"], skip_special_tokens=False)
    assert "<0.4 seconds>" in text and "<1.4 seconds>" in text
    assert request["video_data"][0]["format"] == "processor_output"
    assert request["input_ids"] == request["video_data"][0]["input_ids"][0].tolist()
    assert descriptor.get_options() == before


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning:transformers.processing_utils"
)
@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("start", [0.0, 10.0])
@pytest.mark.parametrize("interval", [0.5, 1 / 30])
def test_real_video_processor_preserves_sampling_interval(
    monkeypatch, family, start, interval, make_real_video_processor
):
    import transformers
    from transformers.video_utils import VideoMetadata

    processor = make_real_video_processor(scalar_timing=True)
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    descriptor = load_provider(family).get_video_embedder(
        "custom-local-checkpoint",
        2,
        options={
            "gpus_per_actor": 0,
            "processor_kwargs": {
                "videos_kwargs": {"size": {"shortest_edge": 4096, "longest_edge": 8192}},
            },
        },
    )
    before = descriptor.get_options()
    frames = tuple(np.full((64, 64, 3), value, dtype=np.uint8) for value in (10, 20, 30, 40))
    times = tuple(start + i * interval for i in range(4))

    async def run():
        runtime = descriptor.instantiate()
        try:
            assert await runtime.embed_video([VideoClip(frames, times, (1, 5, 9, 13))]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    request = state.calls[0][1]
    if family == "vllm":
        assert request["mm_processor_kwargs"]["do_sample_frames"] is False
        assert "do_sample_frames" not in request["mm_processor_kwargs"]["videos_kwargs"]
        array, metadata = request["multi_modal_data"]["video"][0]
        # vLLM's native adapter reconstructs HF metadata from this dictionary.
        # Exercise the submitted metadata/options with the real processor.
        output = processor(
            text=[request["prompt"]],
            videos=[array],
            video_metadata=[
                VideoMetadata(**{key: value for key, value in metadata.items() if key != "do_sample_frames"})
            ],
            **request["mm_processor_kwargs"],
        )
    else:
        output = request["video_data"][0]
    np.testing.assert_allclose(output["second_per_grid_ts"], [2 * interval], atol=1e-6, rtol=0)
    assert output["video_grid_thw"][0][0] == 2  # Four frames, without resampling.
    assert descriptor.get_options() == before


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning:transformers.processing_utils"
)
@pytest.mark.parametrize("kind", ["text", "image"])
def test_sglang_real_processor_accepts_nested_text_options(monkeypatch, kind, make_real_video_processor):
    import transformers

    processor = make_real_video_processor()
    processor.chat_template = "hello" if kind == "text" else "<|image_pad|>"
    state = install_sdk(monkeypatch.setitem, mock_transformers=False)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    descriptor = getattr(load_provider("sglang"), f"get_{kind}_embedder")(
        "custom-local-checkpoint",
        2,
        options={
            "gpus_per_actor": 0,
            "processor_kwargs": {
                "text_kwargs": {"padding": False, "truncation": False, "add_special_tokens": False},
            },
        },
    )

    async def run():
        runtime = descriptor.instantiate()
        try:
            value = "hello" if kind == "text" else np.zeros((64, 64, 3), dtype=np.uint8)
            assert await getattr(runtime, f"embed_{kind}")([value]) == [[3, 4]]
        finally:
            await runtime.aclose()

    asyncio.run(run())
    assert state.requests_closed == 1 and state.engines[0].closed == 1


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.real_ray
def test_native_descriptors_run_on_existing_ray_worker(family, ray_local):
    import cloudpickle
    import ray

    cloudpickle.register_pickle_by_value(sys.modules[__name__])

    descriptor = load_provider(family).get_prompter(
        "configured-model", options={"media_mime_types": ["video/mp4"], "gpus_per_actor": 0}
    )
    embedding = load_provider(family).get_video_embedder("configured-encoder", 2, options={"gpus_per_actor": 0})

    @ray.remote
    def execute(prompt_desc, embed_desc, install):
        previous = {}

        def replace(mapping, key, value):
            previous[key] = mapping.get(key)
            mapping[key] = value

        state = install(replace)

        async def run():
            prompt_runtime, embed_runtime = prompt_desc.instantiate(), embed_desc.instantiate()
            try:
                answer = await prompt_runtime.prompt(("question", PromptMedia(b"video", "video/mp4")))
                text_answer = await prompt_runtime.prompt(("question",))
                result = await embed_runtime.embed_video([clip()])
                return answer, text_answer, result
            finally:
                await prompt_runtime.aclose()
                await embed_runtime.aclose()

        try:
            result = asyncio.run(run())
        finally:
            for key, value in previous.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value
        assert len(state.engines) == 2 and all(engine.closed == 1 for engine in state.engines)
        return result

    assert ray.get(execute.remote(descriptor, embedding, install_sdk), timeout=30) == (
        '{"answer":"ok"}',
        '{"answer":"ok"}',
        [[3.0, 4.0]],
    )


@pytest.mark.parametrize("family", ["vllm", "sglang"])
@pytest.mark.parametrize("entry", ["python", "sql"])
def test_public_native_prompt_and_embedding_with_null_rows(family, entry, monkeypatch, tmp_path):
    import cloudpickle

    import vane
    from vane.ai import embed, prompt
    from vane.ai.provider import PROVIDERS
    from vane.ai.providers._native_inference import NativeMediaPrompterDescriptor, NativeTextEmbedderDescriptor

    cloudpickle.register_pickle_by_value(sys.modules[__name__])

    class WorkerPrompt(NativeMediaPrompterDescriptor):
        def instantiate(self):
            install_sdk(lambda mapping, key, value: mapping.__setitem__(key, value))
            return super().instantiate()

    class WorkerEmbedding(NativeTextEmbedderDescriptor):
        def instantiate(self):
            install_sdk(lambda mapping, key, value: mapping.__setitem__(key, value))
            return super().instantiate()

    base = type(load_provider(family))

    class WorkerProvider(base):
        def get_prompter(self, *args, **kwargs):
            descriptor = super().get_prompter(*args, **kwargs)
            descriptor.__class__ = WorkerPrompt
            return descriptor

        def get_text_embedder(self, *args, **kwargs):
            descriptor = super().get_text_embedder(*args, **kwargs)
            descriptor.__class__ = WorkerEmbedding
            return descriptor

    monkeypatch.setitem(PROVIDERS, family, WorkerProvider)
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    media = tmp_path / "video.mp4"
    media.write_bytes(b"fixture-video")
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    with vane.connect() as conn:
        conn.sql(
            """create table prompts as select id, 'question' as q,
            case when id = 0 then NULL::FILE else file(?, 'video/mp4', NULL, NULL, NULL) end as media
            from range(2) t(id)""",
            params=[str(media)],
        )
        if entry == "python":
            relation = conn.sql("select * from (values (NULL::VARCHAR), ('question')) t(q)")
            vectors = embed(
                relation,
                vane.col("q"),
                provider=family,
                model="configured-encoder",
                dimensions=2,
                gpus_per_actor=0,
                normalize=True,
                max_retries=0,
            ).fetchall()
            assert vectors[0][1] is None
            np.testing.assert_allclose(vectors[1][1], [0.6, 0.8])
            result = prompt(
                conn.sql("select * from prompts order by id"),
                [vane.col("q"), vane.col("media")],
                provider=family,
                model="configured-vlm",
                gpus_per_actor=0,
                media_mime_types=["video/mp4"],
                return_format=schema,
                max_retries=0,
            ).fetchall()
        else:
            import json

            result = conn.sql(
                """select id, ai_prompt(q, media,
                provider => ?, model => 'configured-vlm', return_format => ?,
                options => {gpus_per_actor: 0, media_mime_types: ['video/mp4']}) from prompts order by id""",
                params=[family, json.dumps(schema)],
            ).fetchall()
        assert [(row[0], row[-1]) for row in result] == [(0, {"answer": "ok"}), (1, {"answer": "ok"})]
