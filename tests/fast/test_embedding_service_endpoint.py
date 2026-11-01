# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit service addresses: real sockets and RPC, without model downloads."""

import asyncio
import json
import socket

import numpy as np
import pytest
from tests.fast.test_embedding_service import client
from tests.fast.test_native_multimodal import install_sdk
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai.embedding_service import EmbeddingService

pytestmark = pytest.mark.filterwarnings(
    r"ignore:Proactor event loop does not implement add_reader family of methods required for zmq\.:RuntimeWarning"
)


def service(root, endpoint):
    return EmbeddingService(
        path=str(root / "engine.json"),
        service_id="fixture-v1",
        family="sglang",
        model="fixture",
        dimensions=2,
        endpoint=endpoint,
        options={"input_modalities": ["text"]},
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:8000",
        "tcp://0.0.0.0:8000",
        "tcp://192.0.2.1:8000",
        "tcp://127.0.0.1:0",
        "tcp://127.0.0.1:65536",
        "tcp://localhost:8000",
        "",
    ],
)
def test_invalid_endpoint_rejected_before_backend_or_files(tmp_path, endpoint, monkeypatch):
    monkeypatch.setattr(
        "vane.ai.providers._embedding_service_backend.create_service_backend",
        lambda *a: pytest.fail("invalid address must fail before backend construction"),
    )
    with pytest.raises(EmbeddingConfigurationError):
        service(tmp_path, endpoint)
    assert not list(tmp_path.iterdir())


def test_fixed_endpoint_serves_rpc_and_releases_port_for_restart(tmp_path, monkeypatch):
    install_sdk(monkeypatch.setitem)
    tmp_path.chmod(0o700)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{probe.getsockname()[1]}"

    async def run():
        for _ in range(2):
            engine = service(tmp_path, endpoint)
            await engine.start()
            instance = client(tmp_path, timeout=5)
            try:
                assert json.loads((tmp_path / "engine.json").read_text())["endpoint"] == endpoint
                result = await instance.embed_text(["query"])
                assert np.asarray(result).shape == (1, 2)
            finally:
                await instance.aclose()
                await engine.aclose()
            assert not (tmp_path / "engine.json").exists()

    asyncio.run(run())


def test_occupied_endpoint_fails_without_fallback_and_preserves_owner(tmp_path, monkeypatch):
    import zmq

    install_sdk(monkeypatch.setitem)
    roots = [tmp_path / name for name in ("owner", "contender")]
    for root in roots:
        root.mkdir(mode=0o700)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        endpoint = f"tcp://127.0.0.1:{probe.getsockname()[1]}"

    async def run():
        owner = service(roots[0], endpoint)
        await owner.start()
        original = (roots[0] / "engine.json").read_bytes()
        contender = service(roots[1], endpoint)
        try:
            with pytest.raises(zmq.ZMQError):
                await contender.start()
            assert not (roots[1] / "engine.json").exists()
            assert (roots[0] / "engine.json").read_bytes() == original
            instance = client(roots[0], timeout=5)
            try:
                assert np.asarray(await instance.embed_text(["still alive"])).shape == (
                    1,
                    2,
                )
            finally:
                await instance.aclose()
        finally:
            await contender.aclose()
            await owner.aclose()

    asyncio.run(run())
