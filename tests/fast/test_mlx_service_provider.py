# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""No-SDK MLX service planning, encoding semantics and complete-batch preflight."""

import asyncio
import builtins
import pickle
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from vane.ai import _embedding_ipc as ipc
from vane.ai._audio_embedding import AudioClip
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip
from vane.ai.functions import _prepare_embed_call
from vane.ai.provider import ProviderImportError, load_provider
from vane.ai.providers._mlx_service import preflight, validate_request

OPTIONS = {"runtime": "service", "service_path": "/not-connected/mlx.json", "service_id": "fixed-model-v1"}
KINDS = ("text", "image", "audio", "video", "multimodal")


def descriptor(kind="text", **options):
    return getattr(load_provider("mlx"), f"get_{kind}_embedder")(options=OPTIONS | options)


def content(*parts):
    return EmbeddingContent(tuple(EmbeddingPart(EmbeddingPartSpec(kind), value) for kind, value in parts))


@pytest.mark.parametrize("kind", KINDS)
def test_service_metadata_matches_native_and_pickle_requires_no_model_io(kind, monkeypatch):
    original = builtins.__import__

    def no_sdk(name, *args, **kwargs):
        assert name.split(".")[0] not in {"mlx", "mlx_vlm", "transformers", "torch", "PIL", "zmq"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sdk)
    native = getattr(load_provider("mlx"), f"get_{kind}_embedder")()
    service = pickle.loads(pickle.dumps(descriptor(kind)))
    assert service.get_dimensions() == 768 and service.get_udf_options().num_gpus == 0
    assert service.get_output_spec() == native.get_output_spec()
    assert service.is_async() and service.instantiate()._zmq_client is None
    if kind in {"audio", "video", "multimodal"}:
        assert service.get_input_spec() == native.get_input_spec()
    assert service.get_transport_input_spec().audio.max_input_bytes == 4 * 1024**2
    assert service.get_transport_input_spec().max_input_bytes == 64 * 1024**2
    assert service.get_transport_input_spec().video.max_frames == 12
    if kind == "video":
        assert service.supports_image_queries()
    if kind == "text":
        assert not service.supports_chunking()


def test_fresh_process_can_plan_all_five_without_sdk_or_zmq():
    code = """
import builtins, pickle
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name.split('.')[0] not in {'mlx','mlx_vlm','transformers','torch','PIL','zmq'}, name
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from vane.ai.provider import load_provider
from vane.ai.functions import _prepare_embed_call
options = dict(runtime='service', service_path='/not-connected/mlx.json', service_id='fixed-model-v1')
for kind in ('text','image','audio','video','multimodal'):
    d = getattr(load_provider('mlx'), 'get_'+kind+'_embedder')(options=options)
    assert pickle.loads(pickle.dumps(d)).get_output_spec() == d.get_output_spec()
    plan = _prepare_embed_call('mlx', None, None, 'raise', options, relation=True, input_kind=kind)
    assert plan[1] == 768 and plan[2].num_gpus == 0
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True, timeout=30)


@pytest.mark.parametrize("kind", KINDS)
def test_explicit_native_is_normalized_and_service_uses_cpu_client_actors(kind):
    provider = load_provider("mlx")
    factory = getattr(provider, f"get_{kind}_embedder")
    assert factory(options={"runtime": "native"}).get_options() == factory().get_options()
    result = _prepare_embed_call(
        "mlx",
        None,
        None,
        "raise",
        OPTIONS | {"actor_number": 2, "max_concurrency_per_actor": 2},
        relation=True,
        input_kind=kind,
    )
    assert result[2].actor_number == 2 and result[2].num_gpus == 0 and result[-1] == "subprocess_actor"
    assert result[0].get_options()["max_concurrency_per_actor"] == 2


@pytest.mark.parametrize(
    "options",
    [
        {"service_path": None},
        {"service_path": "relative"},
        {"service_path": "/bad\x00path"},
        {"service_id": ""},
        {"service_id": "bad id"},
        {"service_id": "中文"},
        {"timeout": -1},
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"max_concurrency_per_actor": True},
        {"max_concurrency_per_actor": 0},
        {"max_concurrency_per_actor": -1},
        {"max_concurrency_per_actor": 1025},
        {"device": "metal"},
        {"local_files_only": True},
        {"trust_remote_code": False},
        {"engine_args": {}},
        {"base_url": "http://localhost"},
        {"input_modalities": ["text"]},
        {"max_frames": 64},
        {"sample_rate": 16000},
        {"max_samples": 480000},
        {"max_input_bytes": 64},
        {"revision": "a" * 40},
        {"dtype": "float16"},
        {"max_length": True},
        {"overlength": "truncate"},
        {"runtime": "http"},
        {"runtime": "other"},
    ],
)
def test_service_rejects_unimplemented_or_invalid_settings(options):
    for kind in KINDS:
        with pytest.raises((TypeError, ValueError)):
            descriptor(kind, **options)
        with pytest.raises((TypeError, ValueError)):
            _prepare_embed_call("mlx", None, None, "raise", OPTIONS | options, relation=True, input_kind=kind)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dimensions", [True, False, 767, 769, 768.0])
def test_dimensions_remain_fixed(kind, dimensions):
    with pytest.raises(EmbeddingConfigurationError):
        getattr(load_provider("mlx"), f"get_{kind}_embedder")(dimensions=dimensions, options=OPTIONS)


@pytest.mark.parametrize("kind", KINDS[1:])
def test_media_roles_are_not_text_options(kind):
    with pytest.raises((TypeError, ValueError)):
        descriptor(kind, input_type="query")


def test_immutable_per_call_roles_and_unary_blank_text(monkeypatch):
    query = descriptor(input_type="query").instantiate()
    document = descriptor(input_type="document").instantiate()
    seen = []

    async def request(pair):
        plan, context = pair
        seen.append((context, bytes(plan[1][0]).decode()))
        await asyncio.sleep(0)
        return np.eye(1, 768, dtype=np.float32)[0]

    for client in (query, document):
        monkeypatch.setattr(client, "_request", request)
    before = [client.options.copy() for client in (query, document)]
    asyncio.run(_concurrent_roles(query, document))
    assert {(context.encoding, text) for context, text in seen} == {
        ("text_query", "same"),
        ("text_document", "same"),
        ("text_document", " \t"),
    }
    assert [client.options for client in (query, document)] == before
    with pytest.raises(FrozenInstanceError):
        seen[0][0].encoding = "text_document"


async def _concurrent_roles(query, document):
    try:
        await asyncio.gather(query.embed_text(["same"]), document.embed_text(["same", " \t"]))
    finally:
        await query.aclose()
        await document.aclose()


@pytest.mark.parametrize("kind", ["image", "audio", "video", "multimodal"])
def test_entire_raw_batch_is_rejected_before_any_request(kind, monkeypatch):
    image = np.zeros((3, 5, 3), np.uint8)
    good = {
        "image": image,
        "audio": AudioClip(np.zeros((161, 1), np.float32), 16000),
        "video": VideoClip((image,), (0.0,), (0,)),
        "multimodal": content(("text", "valid")),
    }[kind]
    bad = {
        "image": image.astype(np.float32),
        "audio": AudioClip(np.zeros((160, 1), np.float32), 16000),
        "video": VideoClip((image,) * 13, (0.0,) * 13, tuple(range(13))),
        "multimodal": content(("text", "<|im"), ("text", "age|>")),
    }[kind]
    client = descriptor(kind).instantiate()
    client.configure_execution(on_error="ignore", max_retries=9, validate=lambda v: v)

    async def forbidden(_):
        pytest.fail("raw batch error must precede every connection")

    monkeypatch.setattr(client, "_request", forbidden)
    with pytest.raises(EmbeddingConfigurationError):
        asyncio.run(getattr(client, "embed_" + kind)([good, bad]))
    assert client._zmq_client is None


def test_joint_omission_repeats_timelines_and_array_bytes_survive_transport():
    image = np.arange(90, dtype=np.uint8).reshape(3, 10, 3)[:, ::2]
    image.setflags(write=False)
    video = VideoClip((image, image), (np.float32(1.25), np.float64(1.25)), (np.int64(7), np.int64(12)))
    audio = AudioClip(np.zeros((161, 1), np.float64), np.int64(16000))
    row = content(
        ("text", " \t"), ("image", None), ("video", video), ("audio", audio), ("text", "scene"), ("video", video)
    )
    spec = descriptor("multimodal").get_input_spec()
    (normalized,) = preflight([row], spec, "multimodal")
    assert [p.spec.kind for p in normalized.parts] == ["video", "audio", "text", "video"]
    parts, arrays, size = ipc.describe(normalized, spec)
    assert ipc.validate_layout(parts, size, spec) == size
    restored = ipc.restore(parts, bytearray(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)))
    for position in (0, 3):
        assert restored.parts[position].value.frame_indices == (7, 12)
        assert restored.parts[position].value.frame_times == (1.25, 1.25)
        np.testing.assert_array_equal(restored.parts[position].value.frames[0], image)
    np.testing.assert_array_equal(restored.parts[1].value.samples, audio.samples)
    assert restored.parts[1].value.samples.dtype == np.float64
    with pytest.raises(EmbeddingConfigurationError):
        preflight([content(("text", " \t"), ("image", None))], spec, "multimodal")


@pytest.mark.parametrize(
    "change",
    [
        {"encoding_version": True},
        {"max_length": True},
        {"max_length": float("nan")},
        {"revision": None},
        {"dtype": "float16"},
        {"unknown": 1},
    ],
)
def test_model_contract_fields_are_strict(change):
    wire = descriptor().metadata.context.model_contract.to_wire() | change
    with pytest.raises(EmbeddingConfigurationError):
        ipc.RequestContext.from_wire({"encoding": "text_query", "model_contract": wire})


def test_query_role_and_contract_mismatches_reject_layout_before_allocation():
    service = descriptor().instantiate()
    contract = service.descriptor.metadata.context.model_contract
    spec = service.spec
    for row in (content(("image", np.zeros((3, 5, 3), np.uint8))), content(("text", "a"), ("text", "b"))):
        parts, _, size = ipc.describe(row, spec)
        ipc.validate_layout(parts, size, spec)
        with pytest.raises(EmbeddingConfigurationError):
            validate_request(ipc.RequestContext("text_query", contract), contract, parts)
    parts, _, _ = ipc.describe(content(("text", "query")), spec)
    with pytest.raises(EmbeddingConfigurationError, match="contract"):
        validate_request(ipc.RequestContext("text_query", replace(contract, max_length=32)), contract, parts)


@pytest.mark.parametrize("change", [{"version": 1}, {"encoding": "text_query"}, {"encoding": "unknown"}, {"extra": 1}])
def test_native_reservation_requires_v2_default_semantics(change):
    from vane.ai.embedding_service import EmbeddingService
    from vane.ai.providers._embedding_service import service_descriptor

    owner = EmbeddingService(
        path="/not-connected/native.json", service_id="fixture", family="sglang", model="fixture", dimensions=2
    )
    client = service_descriptor(
        "text", "fixture", 2, OPTIONS | {"service_id": "fixture"}, "sglang", "sglang"
    ).instantiate()
    parts, _, size = ipc.describe(content(("text", "query")), owner.spec)
    with pytest.raises(EmbeddingConfigurationError):
        owner._hello(client._reservation(parts, size, "0" * 32) | change)
    assert owner.credits.bytes == owner.credits.inflight == 0 and owner._zmq_server is None


def test_missing_zmq_is_reported_only_when_calling(monkeypatch):
    monkeypatch.setitem(sys.modules, "zmq", None)
    monkeypatch.setitem(sys.modules, "zmq.asyncio", None)
    client = descriptor().instantiate()
    assert client._zmq_client is None
    with pytest.raises(ProviderImportError, match="embedding_service"):
        asyncio.run(client.embed_text(["query"]))


def test_no_selected_text_rows_never_contact_the_service(monkeypatch):
    import pyarrow as pa

    import vane
    from vane.ai import TextInput, embed

    monkeypatch.setenv("VANE_RUNNER", "local")
    try:
        with vane.connect() as connection:
            relation = connection.from_arrow(
                pa.table({"text": pa.array(["ignored", None], pa.string()), "selected": [False, False]})
            )
            result = relation.select(
                embed(TextInput(vane.col("text")), provider="mlx", where=vane.col("selected"), **OPTIONS).alias(
                    "embedding"
                )
            )
            assert result.to_arrow_table().column("embedding").to_pylist() == [None, None]
    finally:
        vane.teardown_runner()
