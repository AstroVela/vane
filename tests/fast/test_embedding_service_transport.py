# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Transport bounds before authentication, using independent TCP/ZMTP peers."""

import asyncio
import contextlib
import hashlib
import hmac
import uuid

import pytest

from tests.fast.test_embedding_service import client, content, make_service, until
from tests.fast.test_native_multimodal import install_sdk
from vane.ai import _embedding_ipc as ipc
from vane.ai import _embedding_stream as stream
from vane.ai import _embedding_zmq as transport
from vane.ai._embedding_inputs import EmbeddingConfigurationError

pytestmark = pytest.mark.filterwarnings(
    r"ignore:Proactor event loop does not implement add_reader family of methods required for zmq\.:RuntimeWarning"
)

pytest.importorskip("zmq")
import zmq.asyncio


@contextlib.asynccontextmanager
async def service_fixture(tmp_path, monkeypatch, **options):
    sdk = install_sdk(monkeypatch.setitem)
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    service = make_service(root, **options)
    await service.start()
    try:
        yield service, root, sdk
    finally:
        await service.aclose()


async def healthy_request(root):
    instance = client(root, timeout=3)
    try:
        assert (await instance.embed_text(["healthy successor"]))[0].tolist() == [3, 4]
    finally:
        await instance.aclose()


async def read_record(reader):
    size = int.from_bytes(await reader.readexactly(4), "big")
    assert 0 < size <= ipc.MAX_CONTROL_BYTES + 32
    return await reader.readexactly(size)


def frame(body):
    return len(body).to_bytes(4, "big") + body


def proof(key, purpose, challenge, nonce):
    return hmac.digest(key, b"vane-embedding-stream-3/" + purpose + challenge + nonce, hashlib.sha256)


@contextlib.asynccontextmanager
async def tcp_peer(root, *, authenticate=False, fragmented=False, request_id=None):
    endpoint, key = transport._read_endpoint(str(root / "engine.json"))
    reader, writer = await asyncio.open_connection("127.0.0.1", int(endpoint.rsplit(":", 1)[1]))
    try:
        challenge = await asyncio.wait_for(read_record(reader), 2)
        request_id = request_id or uuid.uuid4().hex
        nonce = request_id.encode("ascii")
        if authenticate:
            message = frame(nonce + proof(key, b"client", challenge, nonce))
            if fragmented:
                for byte in message:
                    writer.write(bytes([byte]))
                    await writer.drain()
                    await asyncio.sleep(0.001)
            else:
                writer.write(message)
                await writer.drain()
            assert await asyncio.wait_for(read_record(reader), 2) == proof(key, b"server", challenge, nonce)
        yield reader, writer, challenge, request_id, proof(key, b"session", challenge, nonce)
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()


def test_unauthenticated_multipart_is_bounded_before_materialization(tmp_path, monkeypatch):
    async def run():
        async with service_fixture(tmp_path, monkeypatch, max_bytes=1, max_inflight=1, max_connections=1) as (
            service,
            root,
            sdk,
        ):
            observed = []
            original_receive = zmq.asyncio.Socket.recv_multipart

            async def receive(socket, *args, **kwargs):
                frames = await original_receive(socket, *args, **kwargs)
                if socket is service._zmq_server.socket and any(frames[1:]):
                    observed.append(sum(map(len, frames)))
                return frames

            monkeypatch.setattr(zmq.asyncio.Socket, "recv_multipart", receive)
            context = zmq.asyncio.Context()
            peer = context.socket(zmq.DEALER)
            peer.setsockopt(zmq.LINGER, 0)
            peer.setsockopt(zmq.RECONNECT_IVL, -1)
            peer.setsockopt(zmq.IDENTITY, b"0" * 32)
            peer.connect(service._zmq_server.endpoint)
            try:
                await peer.send_multipart([b"x" * ipc.MAX_CONTROL_BYTES] * 16)
                await until(lambda: observed)
                assert max(observed) <= 64 * 1024
                assert not sdk.calls and not service._tasks
                assert service.credits.bytes == service.credits.inflight == 0
            finally:
                peer.close(linger=0)
                context.term()
            await until(lambda: not service._zmq_server.peers)
            # Empty input fits the deliberately one-byte admission budget.
            instance = client(root)
            try:
                assert (await instance.embed_text([""]))[0].tolist() == [3, 4]
            finally:
                await instance.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("authenticated", [False, True])
@pytest.mark.parametrize("length", [0, 2**32 - 1])
def test_record_length_rejected_before_body_arrives(tmp_path, monkeypatch, authenticated, length):
    async def run():
        async with service_fixture(tmp_path, monkeypatch) as (service, root, sdk):
            async with tcp_peer(root, authenticate=authenticated) as (reader, writer, _, _, _):
                # Fragment the header; send no body. Waiting for the body is a bug.
                prefix = length.to_bytes(4, "big")
                writer.write(prefix[:3])
                await writer.drain()
                await asyncio.sleep(0.01)
                writer.write(prefix[3:])
                await writer.drain()
                assert await asyncio.wait_for(reader.read(1), 2) == b""
                assert not sdk.calls and not service._tasks
                assert service.credits.bytes == service.credits.inflight == 0
            await healthy_request(root)

    asyncio.run(run())


def test_raw_tcp_flood_stays_bounded_before_authentication(tmp_path, monkeypatch):
    async def run():
        async with service_fixture(tmp_path, monkeypatch) as (service, root, sdk):
            observed = []
            original_feed = stream.Decoder.feed

            def feed(decoder, data):
                try:
                    yield from original_feed(decoder, data)
                finally:
                    observed.append((len(data), len(decoder.buffer)))

            monkeypatch.setattr(stream.Decoder, "feed", feed)
            async with tcp_peer(root) as (reader, writer, _, _, _):
                writer.write((16 * ipc.MAX_CONTROL_BYTES).to_bytes(4, "big"))
                writer.write(b"x" * (16 * ipc.MAX_CONTROL_BYTES))
                with contextlib.suppress(ConnectionError):
                    await writer.drain()
                try:
                    assert await asyncio.wait_for(reader.read(1), 2) == b""
                except ConnectionError:
                    pass
                assert observed and max(size for size, _ in observed) <= 64 * 1024
                assert max(buffered for _, buffered in observed) <= stream.AUTH_BYTES
                assert not sdk.calls and service.credits.bytes == 0
            await healthy_request(root)

    asyncio.run(run())


def test_authentication_proof_cannot_be_replayed_on_another_connection(tmp_path, monkeypatch):
    async def run():
        async with service_fixture(tmp_path, monkeypatch) as (service, root, sdk):
            _, key = transport._read_endpoint(str(root / "engine.json"))
            async with tcp_peer(root) as (_, _, original_challenge, request_id, _):
                nonce = request_id.encode("ascii")
                response = nonce + proof(key, b"client", original_challenge, nonce)
                async with tcp_peer(root) as (reader, writer, challenge, _, _):
                    assert challenge != original_challenge
                    writer.write(frame(response))
                    await writer.drain()
                    assert await asyncio.wait_for(reader.read(1), 2) == b""
                    assert not sdk.calls and service.credits.bytes == 0
            await healthy_request(root)

    asyncio.run(run())


def test_signed_request_cannot_be_replayed_on_another_authenticated_connection(tmp_path, monkeypatch):
    async def run():
        async with service_fixture(tmp_path, monkeypatch) as (service, root, sdk):
            owner = client(root)
            try:
                async with tcp_peer(root, authenticate=True) as (_, _, _, request_id, key):
                    parts, _, size = ipc.describe(content("text", ""), owner.spec)
                    recorded = transport._pack(owner._reservation(parts, size, request_id), key, b"request")
                async with tcp_peer(root, authenticate=True, request_id=request_id) as (reader, writer, _, _, _):
                    writer.write(frame(recorded))
                    await writer.drain()
                    assert await asyncio.wait_for(reader.read(1), 2) == b""
                    assert not sdk.calls and not service._tasks and service.credits.bytes == 0
                await healthy_request(root)
            finally:
                await owner.aclose()

    asyncio.run(run())


def test_extra_authenticated_records_cannot_grow_pending_work(tmp_path, monkeypatch):
    async def run():
        async with service_fixture(tmp_path, monkeypatch, publication_timeout=0.2) as (service, root, sdk):
            owner = client(root)
            try:
                async with tcp_peer(root, authenticate=True) as (reader, writer, _, request_id, key):
                    parts, _, size = ipc.describe(content("text", ""), owner.spec)
                    reservation = frame(transport._pack(owner._reservation(parts, size, request_id), key, b"request"))
                    publication = frame(
                        transport._pack({"request_id": request_id, "state": "published"}, key, b"request")
                    )
                    writer.write(reservation + publication * 1000)
                    await writer.drain()
                    # A legal response can precede the disconnect if it was already sent.
                    with contextlib.suppress(ConnectionError):
                        await asyncio.wait_for(reader.read(), 2)
                await until(lambda: not service._tasks and not service._zmq_server.peers)
                assert len(sdk.calls) <= 1
                assert not service._zmq_server.pending
                assert service.credits.bytes == service.credits.inflight == 0
                await healthy_request(root)
            finally:
                await owner.aclose()

    asyncio.run(run())


def test_control_decode_allocation_failure_remains_fatal(tmp_path, monkeypatch):
    async def run():
        async with service_fixture(tmp_path, monkeypatch) as (service, root, sdk):
            instance = client(root, timeout=2)
            unpack = transport._unpack

            def fail_request(body, key, direction):
                if direction == b"request":
                    raise MemoryError("control decode allocation")
                return unpack(body, key, direction)

            try:
                with monkeypatch.context() as patch:
                    patch.setattr(transport, "_unpack", fail_request)
                    with pytest.raises(MemoryError, match="resources"):
                        await instance.embed_text(["query"])
                assert not sdk.calls and service.credits.bytes == 0
                await healthy_request(root)
            finally:
                await instance.aclose()

    asyncio.run(run())


def test_client_rejects_forged_server_proof_before_sending_control():
    async def run():
        finished = asyncio.get_running_loop().create_future()

        async def forge(reader, writer):
            try:
                writer.write(frame(b"c" * 32))
                await writer.drain()
                assert len(await read_record(reader)) == stream.AUTH_BYTES
                writer.write(frame(b"forged"))
                await writer.drain()
                finished.set_result(await reader.read())
            except Exception as error:
                finished.set_exception(error)
            finally:
                writer.close()
                await writer.wait_closed()

        server = await asyncio.start_server(forge, "127.0.0.1", 0)
        context = zmq.asyncio.Context()
        socket = transport._socket(context, zmq.STREAM)
        try:
            socket.connect(f"tcp://127.0.0.1:{server.sockets[0].getsockname()[1]}")
            connection = stream.Connection(socket, b"k" * 32, uuid.uuid4().hex)
            with pytest.raises(EmbeddingConfigurationError, match="authentication"):
                await asyncio.wait_for(connection.authenticate(), 2)
            assert connection.key is None
            socket.close(linger=0)
            assert await asyncio.wait_for(finished, 2) == b""
        finally:
            socket.close(linger=0)
            context.term()
            server.close()
            await server.wait_closed()

    asyncio.run(run())


@pytest.mark.parametrize("authenticated", [False, True])
def test_incomplete_control_records_expire_and_return_connection_slot(tmp_path, monkeypatch, authenticated):
    async def run():
        async with service_fixture(
            tmp_path, monkeypatch, max_connections=1, max_inflight=1, publication_timeout=0.2
        ) as (
            service,
            root,
            sdk,
        ):
            async with tcp_peer(root, authenticate=authenticated) as (reader, writer, _, _, _):
                writer.write(b"\x00\x00")
                await writer.drain()
                assert await asyncio.wait_for(reader.read(1), 2) == b""
                assert not sdk.calls and service.credits.bytes == 0
            await until(lambda: not service._zmq_server.peers)
            await healthy_request(root)

    asyncio.run(run())


def test_fragmented_authenticated_request_round_trip(tmp_path, monkeypatch):
    async def run():
        async with service_fixture(tmp_path, monkeypatch) as (service, root, sdk):
            async with tcp_peer(root, authenticate=True, fragmented=True) as (reader, writer, _, request_id, key):
                owner = client(root)
                try:
                    parts, _, size = ipc.describe(content("text", ""), owner.spec)
                    body = frame(transport._pack(owner._reservation(parts, size, request_id), key, b"request"))
                    for start in range(0, len(body), 3):
                        writer.write(body[start : start + 3])
                        await writer.drain()
                        await asyncio.sleep(0)
                    reserved = transport._unpack(await asyncio.wait_for(read_record(reader), 2), key, b"response")
                    assert reserved["state"] == "reserved" and reserved["shm_name"] is None
                    writer.write(
                        frame(transport._pack({"request_id": request_id, "state": "published"}, key, b"request"))
                    )
                    await writer.drain()
                    released = transport._unpack(await asyncio.wait_for(read_record(reader), 2), key, b"response")
                    assert released["vector"] == [3, 4]
                    assert service.credits.bytes == 0 and len(sdk.calls) == 1
                finally:
                    await owner.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["success", "configuration", "resource", "execution"])
def test_terminal_receipt_reclaims_slot_without_losing_response(tmp_path, monkeypatch, outcome):
    async def run():
        async with service_fixture(tmp_path, monkeypatch, max_connections=1, max_inflight=1) as (
            service,
            root,
            sdk,
        ):
            expected = [0.123456789012345] * 16384
            original_infer = service._infer

            async def infer(_):
                if outcome == "resource":
                    raise MemoryError("fixture allocation")
                if outcome == "execution":
                    raise RuntimeError("fixture inference")
                return {"vector": expected}

            with monkeypatch.context() as patch:
                patch.setattr(service, "_infer", infer)
                async with tcp_peer(root, authenticate=True) as (reader, writer, _, request_id, key):
                    owner = client(root)
                    try:
                        parts, _, size = ipc.describe(content("text", ""), owner.spec)
                        message = owner._reservation(parts, size, request_id)
                        if outcome == "configuration":
                            message["service_id"] = "wrong-service"
                        writer.write(frame(transport._pack(message, key, b"request")))
                        await writer.drain()
                        response = transport._unpack(await asyncio.wait_for(read_record(reader), 2), key, b"response")
                        if outcome != "configuration":
                            assert response["state"] == "reserved"
                            writer.write(
                                frame(
                                    transport._pack({"request_id": request_id, "state": "published"}, key, b"request")
                                )
                            )
                            await writer.drain()
                            response = transport._unpack(
                                await asyncio.wait_for(read_record(reader), 2), key, b"response"
                            )
                        if outcome == "success":
                            # This exceeds a TCP/STREAM chunk; closing right
                            # after enqueueing it could discard part or all.
                            assert response["vector"] == expected
                        else:
                            assert response["error"] == outcome
                        assert len(service._zmq_server.peers) == 1
                        assert service.credits.bytes == service.credits.inflight == 0
                        writer.write(
                            frame(transport._pack({"request_id": request_id, "state": "received"}, key, b"request"))
                        )
                        await writer.drain()
                        assert await asyncio.wait_for(reader.read(1), 2) == b""
                        # Keep the TCP writer open across the next request.
                        assert not service._zmq_server.peers
                        patch.setattr(service, "_infer", original_infer)
                        await healthy_request(root)
                    finally:
                        await owner.aclose()

    asyncio.run(run())


def test_missing_terminal_receipt_expires_and_returns_slot(tmp_path, monkeypatch):
    async def run():
        async with service_fixture(
            tmp_path, monkeypatch, max_connections=1, max_inflight=1, publication_timeout=0.2
        ) as (service, root, sdk):
            async with tcp_peer(root, authenticate=True) as (reader, writer, _, request_id, key):
                owner = client(root)
                try:
                    parts, _, size = ipc.describe(content("text", ""), owner.spec)
                    message = owner._reservation(parts, size, request_id)
                    message["service_id"] = "wrong-service"
                    writer.write(frame(transport._pack(message, key, b"request")))
                    await writer.drain()
                    response = transport._unpack(await asyncio.wait_for(read_record(reader), 2), key, b"response")
                    assert response["error"] == "configuration"
                    assert await asyncio.wait_for(reader.read(1), 2) == b""
                    assert not service._zmq_server.peers and not sdk.calls
                    await healthy_request(root)
                finally:
                    await owner.aclose()

    asyncio.run(run())


def test_terminal_receipt_waits_for_unpublished_cleanup(tmp_path, monkeypatch):
    async def run():
        async with service_fixture(tmp_path, monkeypatch, max_connections=1, max_inflight=1) as (
            service,
            root,
            sdk,
        ):
            async with tcp_peer(root, authenticate=True) as (reader, writer, _, request_id, key):
                owner = client(root)
                locked = False
                try:
                    parts, _, size = ipc.describe(content("text", "input"), owner.spec)
                    writer.write(frame(transport._pack(owner._reservation(parts, size, request_id), key, b"request")))
                    await writer.drain()
                    reserved = transport._unpack(await asyncio.wait_for(read_record(reader), 2), key, b"response")
                    assert reserved["state"] == "reserved"
                    await service.credits.condition.acquire()
                    locked = True
                    # A mismatched publication is rejected by the receiver
                    # while the handler still owns the unpublished input.
                    writer.write(frame(transport._pack({"request_id": "wrong", "state": "published"}, key, b"request")))
                    await writer.drain()
                    response = transport._unpack(await asyncio.wait_for(read_record(reader), 2), key, b"response")
                    assert response["error"] == "configuration"
                    peer = next(iter(service._zmq_server.peers.values()))
                    writer.write(
                        frame(transport._pack({"request_id": request_id, "state": "received"}, key, b"request"))
                    )
                    await writer.drain()
                    # Receipt handling cancels the unpublished task before yielding.
                    await until(lambda: peer.receipt_received)
                    assert service._zmq_server.peers
                    assert service._zmq_server.pending
                    assert service.credits.inflight == 1 and not sdk.calls
                    service.credits.condition.release()
                    locked = False
                    assert await asyncio.wait_for(reader.read(1), 2) == b""
                    assert not service._zmq_server.peers and not service._zmq_server.pending
                    assert not service._zmq_server.buffers
                    assert service.credits.bytes == service.credits.inflight == 0
                    await healthy_request(root)
                finally:
                    if locked:
                        service.credits.condition.release()
                    await owner.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("invalid", ["identity", "state", "signature"])
def test_invalid_terminal_receipt_cannot_start_another_request(tmp_path, monkeypatch, invalid):
    async def run():
        async with service_fixture(tmp_path, monkeypatch, max_connections=1, max_inflight=1) as (
            service,
            root,
            sdk,
        ):
            async with tcp_peer(root, authenticate=True) as (reader, writer, _, request_id, key):
                owner = client(root)
                try:
                    parts, _, size = ipc.describe(content("text", ""), owner.spec)
                    message = owner._reservation(parts, size, request_id)
                    message["service_id"] = "wrong-service"
                    writer.write(frame(transport._pack(message, key, b"request")))
                    await writer.drain()
                    response = transport._unpack(await asyncio.wait_for(read_record(reader), 2), key, b"response")
                    assert response["error"] == "configuration"
                    receipt = {"request_id": request_id, "state": "received"}
                    if invalid == "identity":
                        receipt["request_id"] = uuid.uuid4().hex
                    elif invalid == "state":
                        receipt["state"] = "published"
                    body = transport._pack(receipt, b"wrong key" if invalid == "signature" else key, b"request")
                    writer.write(frame(body))
                    await writer.drain()
                    assert await asyncio.wait_for(reader.read(1), 2) == b""
                    assert not service._zmq_server.peers and not sdk.calls
                    await healthy_request(root)
                finally:
                    await owner.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("error", ["resource", "configuration"])
def test_receipt_send_failure_preserves_fatal_response(tmp_path, monkeypatch, error):
    async def run():
        async with service_fixture(tmp_path, monkeypatch) as (service, root, sdk):
            instance = client(root, **({"service_id": "wrong-service"} if error == "configuration" else {}))
            instance.configure_execution(on_error="ignore", max_retries=3, validate=lambda value: value)
            send = stream.Connection.send

            async def fail_receipt(connection, body):
                if connection.key is not None:
                    message = transport._unpack(body, connection.key, b"request")
                    if message.get("state") == "received":
                        raise ConnectionError("receipt transport failed")
                await send(connection, body)

            async def fail_inference(_):
                raise MemoryError("fixture allocation")

            monkeypatch.setattr(stream.Connection, "send", fail_receipt)
            sdk.on_embedding = fail_inference
            try:
                expected = MemoryError if error == "resource" else EmbeddingConfigurationError
                with pytest.raises(expected):
                    await instance.embed_text(["query"])
                await until(lambda: not service._zmq_server.peers)
                assert service.credits.bytes == service.credits.inflight == 0
                assert len(sdk.calls) == (1 if error == "resource" else 0)
            finally:
                await instance.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("size", [stream.AUTH_BYTES, stream.MAX_RECORD_BYTES])
def test_decoder_accepts_exact_limit_and_rejects_larger_header_without_copy(size):
    decoder = stream.Decoder(size)
    payload = b"x" * size
    wire = frame(payload)
    assert list(decoder.feed(wire[:2])) == []
    assert list(decoder.feed(wire[2:-1])) == []
    assert len(decoder.buffer) == size - 1
    assert list(decoder.feed(wire[-1:])) == [payload]
    assert not decoder.buffer and decoder.length is None
    with pytest.raises(EmbeddingConfigurationError, match="byte limit"):
        list(decoder.feed((size + 1).to_bytes(4, "big") + payload))
    assert not decoder.buffer
