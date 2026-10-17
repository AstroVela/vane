# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Fatal allocation errors and native IMAGE semantics at embedding boundaries."""

import asyncio
import errno
import pickle
import socket
import sys

import numpy as np
import pyarrow as pa
import pytest

from tests.fast.test_ai_error_diagnostics import native_error, onnx_allocation_error
from tests.fast.test_embedding_service import client, content
from tests.fast.test_image_modes import MODES, pixels_for
from tests.fast.test_native_multimodal import install_sdk
from tests.fast.test_native_multimodal import make_real_video_processor as make_real_video_processor
from vane.ai import DiagnosticError
from vane.ai import _embedding_ipc as ipc
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._errors import _is_resource_error
from vane.ai.embedding_service import EmbeddingService
from vane.ai.functions import _EmbedImageBatch, _EmbedMultimodalBatch, _EmbedTextBatch
from vane.ai.provider import load_provider
from vane.execution._async_runtime import AsyncRuntime


def allocation_error(kind):
    if kind in ("enomem", "enospc"):
        return OSError(getattr(errno, kind.upper()), "private allocation path")
    if kind == "memory":
        return MemoryError("private allocation data")
    if kind == "torch":
        return pytest.importorskip("torch").OutOfMemoryError("private allocation data")
    if kind == "cpu":
        return RuntimeError(
            "DefaultCPUAllocator: can't allocate memory: you tried to allocate 1073741824 bytes. "
            "Error code 12 (Cannot allocate memory)"
        )
    if kind == "cuda":
        return RuntimeError("CUDA failed with error out of memory")
    if kind == "onnx":
        return onnx_allocation_error(wrapped=True)
    assert kind == "kv"
    return ValueError(
        "Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.8. "
        "Raise --mem-fraction-static above 0.9."
    )


def wrap(error, kind):
    if kind == "direct":
        return error
    if kind in ("cause", "context"):
        wrapper = RuntimeError("private wrapper data")
        setattr(wrapper, f"__{kind}__", error)
        return wrapper
    diagnostic = pickle.loads(pickle.dumps(DiagnosticError("embedding", error)))
    return native_error(diagnostic) if kind == "native" else diagnostic


@pytest.mark.parametrize("kind", ["memory", "torch", "enomem", "enospc", "cpu", "cuda", "onnx", "kv"])
@pytest.mark.parametrize("transport", ["direct", "cause", "context", "diagnostic", "native"])
def test_resource_classification_survives_transport(kind, transport):
    assert _is_resource_error(wrap(allocation_error(kind), transport))


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("OSError (errno=12)"),
        RuntimeError("MemoryError: allocation failed"),
        RuntimeError(str(DiagnosticError("embedding", OSError(errno.ENOSPC, "private")))),
        OSError(errno.EACCES, "ENOMEM or ENOSPC in private data"),
        OSError("no numeric errno"),
        OSError([errno.ENOMEM], "invalid errno type"),
        type("StatusError", (Exception,), {"status_code": errno.ENOMEM})("private data"),
        type("CodeError", (Exception,), {"code": errno.ENOSPC})("private data"),
    ],
)
def test_resource_classification_rejects_opaque_text_and_other_statuses(error):
    assert not _is_resource_error(error)
    assert not _is_resource_error(wrap(error, "diagnostic"))


def test_resource_classification_handles_cleanup_and_cycles():
    error = RuntimeError("private data")
    error.__cause__ = error
    assert not _is_resource_error(error)
    error.cleanup_errors = [wrap(OSError(errno.ENOSPC, "private data"), "cause")]
    assert _is_resource_error(error)
    assert _is_resource_error(wrap(error, "diagnostic"))


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("kind", ["memory", "enospc", "cuda", "onnx"])
def test_native_allocation_failure_precedes_ignore_and_retry(monkeypatch, family, joint, kind):
    sdk = install_sdk(monkeypatch.setitem)
    error = wrap(allocation_error(kind), "diagnostic" if kind == "enospc" else "direct")
    # Even a provider retry hint must not replay an allocation failure.
    error.status_code = 503

    async def fail(_):
        raise error

    sdk.on_embedding = fail
    descriptor = getattr(load_provider(family), f"get_{'multimodal' if joint else 'text'}_embedder")(
        "fixture", 2, options={"gpus_per_actor": 0}
    )
    kwargs = {"parts": (EmbeddingPartSpec("text"),)} if joint else {}
    wrapper = (_EmbedMultimodalBatch if joint else _EmbedTextBatch)(
        descriptor, "input", "vector", 2, on_error="ignore", max_retries=3, **kwargs
    )
    values = pa.table({"part_0": ["query"]}).to_struct_array() if joint else ["query"]
    runtime = AsyncRuntime()
    wrapper.bind_async_runtime(runtime.run)
    try:
        with pytest.raises(MemoryError, match="resources") as raised:
            wrapper(pa.table({"input": values}))
        assert "private" not in str(raised.value)
        assert len(sdk.calls) == sdk.requests_closed == 1
    finally:
        wrapper.close()
        runtime.close()
    assert len(sdk.engines) == 1 and sdk.engines[0].closed == 1


@pytest.mark.skipif(sys.platform != "linux", reason="Linux embedding service")
@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("kind,transport", [(k, t) for k in ("enomem", "enospc") for t in ("cause", "diagnostic")])
def test_service_wrapped_allocation_returns_fatal_category(tmp_path, monkeypatch, family, kind, transport):
    sdk = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)

    async def fail(_):
        raise wrap(allocation_error(kind), transport)

    sdk.on_embedding = fail
    service = EmbeddingService(
        path=str(root / "engine.sock"),
        service_id="fixture-v1",
        family=family,
        model="fixture",
        dimensions=2,
        options={"gpus_per_actor": 0},
        max_inflight=1,
    )

    async def run():
        await service.start()
        instance = client(root, family)
        instance.configure_execution(on_error="ignore", max_retries=3, validate=lambda v: v)
        try:
            with pytest.raises(MemoryError, match="resources"):
                await instance.embed_text(["query"])
            assert service.credits.bytes == service.credits.inflight == 0
            sdk.on_embedding = None
            np.testing.assert_array_equal(await instance.embed_text(["healthy"]), [[3, 4]])
            assert service.credits.bytes == service.credits.inflight == 0
            assert len(sdk.calls) == 2
        finally:
            await instance.aclose()
            await service.aclose()

    asyncio.run(run())


@pytest.mark.skipif(sys.platform != "linux", reason="Linux embedding service")
@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("phase", ["socket", "setblocking", "connect", "buffer", "result"])
def test_service_client_allocation_failure_is_fatal_and_closes_socket(tmp_path, monkeypatch, family, phase):
    sdk = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = EmbeddingService(
        path=str(root / "engine.sock"),
        service_id="fixture-v1",
        family=family,
        model="fixture",
        dimensions=2,
        options={"gpus_per_actor": 0},
        max_inflight=1,
    )

    async def run():
        await service.start()
        instance = client(root, family)
        instance.configure_execution(on_error="ignore", max_retries=3, validate=lambda v: v)
        sockets = []

        def fail(*args, **kwargs):
            raise wrap(OSError(errno.ENOMEM, "private socket allocation"), "diagnostic")

        class AllocatingSocket(socket.socket):
            def setblocking(self, flag):
                sockets.append(self)
                fail()

        try:
            with monkeypatch.context() as patch:
                if phase == "socket":
                    patch.setattr(socket, "socket", fail)
                elif phase == "setblocking":
                    patch.setattr(socket, "socket", AllocatingSocket)
                elif phase == "connect":

                    async def connect(*args):
                        fail()

                    patch.setattr(asyncio.get_running_loop(), "sock_connect", connect)
                elif phase == "buffer":
                    patch.setattr(ipc, "sealed_buffer", fail)
                else:
                    instance._validate_vector = fail
                with pytest.raises(MemoryError, match="resources"):
                    await instance.embed_text(["query"])
            assert not instance._sockets
            assert all(sock.fileno() == -1 for sock in sockets)
            instance._validate_vector = lambda v: v
            np.testing.assert_array_equal(await instance.embed_text(["healthy"]), [[3, 4]])
            assert service.credits.bytes == service.credits.inflight == 0
            assert len(sdk.calls) == (2 if phase == "result" else 1)
        finally:
            await instance.aclose()
            await service.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("mode,channels,dtype", MODES)
def test_native_image_modes_reach_processor_without_conversion(monkeypatch, family, joint, mode, channels, dtype):
    sdk = install_sdk(monkeypatch.setitem)
    pixels = pixels_for(mode, channels, dtype)
    descriptor = getattr(load_provider(family), f"get_{'multimodal' if joint else 'image'}_embedder")(
        "fixture", 2, options={"gpus_per_actor": 0}
    )

    async def run():
        instance = descriptor.instantiate()
        try:
            result = (
                await instance.embed_multimodal(
                    [
                        EmbeddingContent(
                            (
                                EmbeddingPart(EmbeddingPartSpec("text"), "query"),
                                EmbeddingPart(EmbeddingPartSpec("image"), pixels),
                            )
                        )
                    ]
                )
                if joint
                else await instance.embed_image([pixels])
            )
            np.testing.assert_array_equal(result, [[3, 4]])
            image = (
                sdk.processing[0]["images"][0]
                if family == "sglang"
                else sdk.calls[0][1]["multi_modal_data"]["image"][0]
            )
            assert image is pixels
        finally:
            await instance.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("invalid", ["budget", "nan", "shape", "dtype", "layout"])
def test_native_image_validation_keeps_joint_limits(monkeypatch, family, invalid):
    sdk = install_sdk(monkeypatch.setitem)
    pixels = np.zeros((2, 2, 3), dtype=np.float32)
    if invalid == "nan":
        pixels[0, 0, 0] = np.nan
    elif invalid == "shape":
        pixels = pixels[:, :, 0]
    elif invalid == "dtype":
        pixels = pixels.astype(np.float64)
    options = {"gpus_per_actor": 0, "max_input_bytes": pixels.nbytes + 2}
    if invalid == "layout":
        options["input_modalities"] = ["text"]
    descriptor = load_provider(family).get_multimodal_embedder("fixture", 2, options=options)

    async def run():
        instance = descriptor.instantiate()
        instance.configure_execution(on_error="ignore", max_retries=0, validate=lambda v: v)
        try:
            with pytest.raises(EmbeddingConfigurationError):
                await instance.embed_multimodal(
                    [
                        EmbeddingContent(
                            (
                                EmbeddingPart(EmbeddingPartSpec("text"), "123" if invalid == "budget" else "q"),
                                EmbeddingPart(EmbeddingPartSpec("image"), pixels),
                            )
                        )
                    ]
                )
            assert not sdk.calls and not sdk.engines
        finally:
            await instance.aclose()

    asyncio.run(run())


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning:transformers.processing_utils"
)
@pytest.mark.parametrize("family", ["sglang", "vllm"])
def test_native_float_image_real_processor_parity(monkeypatch, make_real_video_processor, family):
    import transformers

    import vane
    from vane.execution.udf_file_contract import FileUDFContract

    processor = make_real_video_processor()
    processor.chat_template = "<|image_pad|>"
    sdk = install_sdk(monkeypatch.setitem, mock_transformers=False)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: processor)
    pixels = np.full((64, 64, 3), 0.5, dtype=np.float32)
    expected = processor(
        text=["<|image_pad|>"], images=[pixels], do_rescale=False, return_tensors="pt", add_special_tokens=False
    )
    descriptor = load_provider(family).get_image_embedder(
        "fixture", 2, options={"gpus_per_actor": 0, "processor_kwargs": {"images_kwargs": {"do_rescale": False}}}
    )
    wrapper = _EmbedImageBatch(descriptor, "image", "vector", 2, on_error="ignore")
    runtime = AsyncRuntime()
    wrapper.bind_async_runtime(runtime.run)
    array = FileUDFContract("fixture", (), (vane.image_type("RGB32F"),)).scalar_outputs_to_array([pixels])
    try:
        assert wrapper(pa.table({"image": array}))["vector"].to_pylist() == [[3, 4]]
        if family == "sglang":
            actual = sdk.calls[0][1]["image_data"][0]
        else:
            request = sdk.calls[0][1]
            actual = processor(
                text=[request["prompt"]], images=request["multi_modal_data"]["image"], **request["mm_processor_kwargs"]
            )
        for key in ("input_ids", "pixel_values", "image_grid_thw"):
            np.testing.assert_array_equal(actual[key], expected[key])
    finally:
        wrapper.close()
        runtime.close()
    assert len(sdk.calls) == 1


def test_service_image_wire_contract_stays_uint8_rgb():
    spec = ipc.input_spec({"input_modalities": ["image"], "max_input_bytes": 1024, "max_frames": 4})
    with pytest.raises(EmbeddingConfigurationError, match="uint8 HWC RGB"):
        ipc.describe(content("image", np.zeros((2, 2, 3), dtype=np.float32)), spec)


def check_public_native_resource_failure(monkeypatch, family, backend, *, phase="execution", on_error="ignore"):
    import cloudpickle

    import vane
    from vane.ai import MultimodalInput, TextInput
    from vane.ai.provider import PROVIDERS
    from vane.ai.providers._native_inference import NativeMultimodalEmbedderDescriptor

    cloudpickle.register_pickle_by_value(sys.modules[__name__])

    class WorkerDescriptor(NativeMultimodalEmbedderDescriptor):
        def instantiate(self):
            if phase == "initialization":
                raise MemoryError("private native initialization credential")
            sdk = install_sdk(lambda mapping, key, value: mapping.__setitem__(key, value))

            async def fail(_):
                raise MemoryError("private native model allocation data")

            sdk.on_embedding = fail
            return super().instantiate()

    class WorkerProvider(type(load_provider(family))):
        def get_multimodal_embedder(self, *args, **kwargs):
            descriptor = super().get_multimodal_embedder(*args, **kwargs)
            descriptor.__class__ = WorkerDescriptor
            return descriptor

    monkeypatch.setitem(PROVIDERS, family, WorkerProvider)
    try:
        with vane.connect() as connection:
            result = connection.from_arrow(pa.table({"text": ["query"]})).embed(
                MultimodalInput([TextInput(vane.col("text"))]),
                provider=family,
                model="fixture",
                dimensions=2,
                gpus_per_actor=0,
                execution_backend=backend,
                on_error=on_error,
                max_retries=3,
                output_column="vector",
                normalize=False,
            )
            with pytest.raises(Exception, match="MemoryError|resources") as raised:
                result.select("vector").fetchall()
            assert "private native" not in str(raised.value)
    finally:
        cloudpickle.unregister_pickle_by_value(sys.modules[__name__])


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("backend", ["subprocess_task", "subprocess_actor"])
@pytest.mark.parametrize("phase", ["initialization", "execution"])
@pytest.mark.parametrize("on_error", ["raise", "ignore"])
def test_public_native_resource_failure_aborts(monkeypatch, family, backend, phase, on_error):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    check_public_native_resource_failure(monkeypatch, family, backend, phase=phase, on_error=on_error)


@pytest.mark.real_ray
@pytest.mark.ray_cluster_owner
@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("backend", ["ray_task", "ray_actor"])
def test_public_native_resource_failure_aborts_on_ray(monkeypatch, family, backend):
    import ray

    import vane

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    try:
        ray.init(address="local", num_cpus=3, include_dashboard=False)
        for phase in ("initialization", "execution"):
            check_public_native_resource_failure(monkeypatch, family, backend, phase=phase)
    finally:
        vane.teardown_runner()
        ray.shutdown()
