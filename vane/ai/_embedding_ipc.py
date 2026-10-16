# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Linux local embedding IPC: bounded JSON and sealed, anonymous CPU buffers.

No pickle, paths to media, resource tracker, named segments or CUDA handles cross
this boundary. The receiver owns a duplicated FD once SCM_RIGHTS is received.
"""

from __future__ import annotations

import array
import asyncio
import json
import math
import os
import socket
import struct
import sys
import weakref
from typing import Any

import numpy as np

from vane.ai._audio_embedding import AudioClip, AudioInputSpec
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip, VideoInputSpec

VERSION = 1
MAX_CONTROL_BYTES = 1024**2


class ReceivedFD:
    """Close also when cancellation discards a completed receive result."""

    def __init__(self, value: int):
        self.value = value
        self._close = weakref.finalize(self, os.close, value)

    def close(self) -> None:
        self._close()


def require_linux() -> None:
    if sys.platform != "linux":
        raise EmbeddingConfigurationError("Embedding service requires Linux Unix sockets and memfd")


def input_spec(options: dict[str, Any]) -> MultimodalInputSpec:
    modalities = options.get("input_modalities", ["text", "image", "video"])
    if (
        not isinstance(modalities, list)
        or not modalities
        or any(not isinstance(item, str) or item not in {"text", "image", "video", "audio"} for item in modalities)
        or len(set(modalities)) != len(modalities)
    ):
        raise EmbeddingConfigurationError("input_modalities must declare distinct decoded modalities")
    limit = options.get("max_input_bytes", 64 * 1024**2)
    frames = options.get("max_frames", 64)
    if type(limit) is not int or not 1 <= limit <= 64 * 1024**2:
        raise EmbeddingConfigurationError("max_input_bytes must be between 1 and 64 MiB")
    if type(frames) is not int or not 1 <= frames <= 256:
        raise EmbeddingConfigurationError("max_frames must be between 1 and 256")
    audio = None
    if "audio" in modalities:
        rate, samples = options.get("sample_rate"), options.get("max_samples")
        if type(rate) is not int or not 1 <= rate <= 384000 or type(samples) is not int or not 1 <= samples <= 2**24:
            raise EmbeddingConfigurationError("Audio requires explicit bounded sample_rate and max_samples")
        audio = AudioInputSpec(rate, samples, max_channels=1, max_input_bytes=limit)
    elif "sample_rate" in options or "max_samples" in options:
        raise EmbeddingConfigurationError("Audio limits require audio in input_modalities")
    return MultimodalInputSpec(
        frozenset(EmbeddingPartSpec(kind) for kind in modalities),
        video=VideoInputSpec(max_frames=frames, max_input_bytes=limit) if "video" in modalities else None,
        audio=audio,
        max_input_bytes=limit,
    )


def describe(
    content: EmbeddingContent, spec: MultimodalInputSpec
) -> tuple[list[dict[str, Any]], list[np.ndarray[Any, Any]], int]:
    """Validate before admission/allocation; describe contiguous array copies."""
    spec.validate_layout(tuple(part.spec for part in content.parts))
    parts: list[dict[str, Any]] = []
    arrays: list[np.ndarray[Any, Any]] = []
    offset, text_bytes = 0, 0

    def buffer(value: Any, kind: str) -> dict[str, Any]:
        nonlocal offset
        if not isinstance(value, np.ndarray) or not value.size:
            raise EmbeddingConfigurationError("Decoded media requires nonempty CPU arrays")
        if kind in {"image", "video"}:
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise EmbeddingConfigurationError("Decoded images and frames require uint8 HWC RGB")
        elif value.dtype not in (np.dtype("float32"), np.dtype("float64")) or value.ndim != 2 or value.shape[1] != 1:
            raise EmbeddingConfigurationError("Native audio requires mono floating PCM shaped (samples, 1)")
        if offset + value.nbytes > spec.max_input_bytes:
            raise EmbeddingConfigurationError("Embedding media exceeds input byte budget")
        result = {"offset": offset, "shape": list(value.shape), "dtype": value.dtype.str, "nbytes": value.nbytes}
        arrays.append(value)
        offset += value.nbytes
        return result

    for part in content.parts:
        kind, value = part.spec.kind, part.value
        entry: dict[str, Any] = {"kind": kind}
        if kind == "text":
            if not isinstance(value, str) or not value.strip():
                raise EmbeddingConfigurationError("Embedding text must be nonempty")
            entry["text"] = value
            text_bytes += len(value.encode("utf-8"))
        elif kind == "image":
            entry["buffer"] = buffer(value, kind)
        elif kind == "video":
            if not isinstance(value, VideoClip) or spec.video is None:
                raise EmbeddingConfigurationError("Expected a decoded VideoClip")
            spec.video.validate_count(len(value.frames))
            times, indices = value.frame_times, value.frame_indices
            if (
                len(times) != len(value.frames)
                or len(indices) != len(times)
                or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
                or any(a >= b for a, b in zip(times, times[1:]))
                or any(type(i) is not int or i < 0 for i in indices)
                or any(a >= b for a, b in zip(indices, indices[1:]))
            ):
                raise EmbeddingConfigurationError("Video timestamps and indices must be ordered and aligned")
            entry.update(
                buffers=[buffer(frame, kind) for frame in value.frames], times=list(times), indices=list(indices)
            )
        else:
            if not isinstance(value, AudioClip) or spec.audio is None or value.sample_rate != spec.audio.sample_rate:
                raise EmbeddingConfigurationError("Audio must match the declared sample_rate; resample first")
            entry.update(buffer=buffer(value.samples, kind), sample_rate=value.sample_rate)
            if (
                len(value.samples) > spec.audio.max_samples
                or not np.isfinite(value.samples).all()
                or np.any(np.abs(value.samples) > 1)
            ):
                raise EmbeddingConfigurationError("Audio exceeds sample budget or contains invalid PCM")
        parts.append(entry)
    if offset + text_bytes > spec.max_input_bytes:
        raise EmbeddingConfigurationError("Embedding row exceeds combined input byte budget")
    return parts, arrays, offset


def validate_layout(parts: Any, size: Any, spec: MultimodalInputSpec) -> int:
    """Validate ALL offsets/shapes/dtypes and metadata before mapping any FD."""
    if type(size) is not int or not 0 <= size <= spec.max_input_bytes or not isinstance(parts, list):
        raise EmbeddingConfigurationError("Invalid embedding buffer size or parts")
    if not 1 <= len(parts) <= spec.max_parts:
        raise EmbeddingConfigurationError("Invalid embedding part count")
    offset, text_bytes = 0, 0
    for part in parts:
        if not isinstance(part, dict) or part.get("kind") not in {p.kind for p in spec.supported_parts}:
            raise EmbeddingConfigurationError("Unsupported embedding part")
        kind = part["kind"]
        keys = {
            "text": {"kind", "text"},
            "image": {"kind", "buffer"},
            "video": {"kind", "buffers", "times", "indices"},
            "audio": {"kind", "buffer", "sample_rate"},
        }[kind]
        if set(part) != keys:
            raise EmbeddingConfigurationError("Invalid embedding part fields")
        if kind == "text":
            if not isinstance(part["text"], str) or not part["text"].strip():
                raise EmbeddingConfigurationError("Embedding text must be nonempty")
            text_bytes += len(part["text"].encode("utf-8"))
            continue
        buffers = part["buffers"] if kind == "video" else [part["buffer"]]
        if not isinstance(buffers, list) or not buffers or len(buffers) > 256:
            raise EmbeddingConfigurationError("Invalid media buffer count")
        for item in buffers:
            if not isinstance(item, dict) or set(item) != {"offset", "shape", "dtype", "nbytes"}:
                raise EmbeddingConfigurationError("Invalid array descriptor")
            shape = item["shape"]
            if (
                not isinstance(shape, list)
                or not 1 <= len(shape) <= 3
                or any(type(n) is not int or n <= 0 or n > spec.max_input_bytes for n in shape)
            ):
                raise EmbeddingConfigurationError("Invalid array shape")
            dtype = item["dtype"]
            allowed = (np.dtype("float32").str, np.dtype("float64").str) if kind == "audio" else ("|u1",)
            if not isinstance(dtype, str) or dtype not in allowed:
                raise EmbeddingConfigurationError("Invalid array dtype")
            nbytes = math.prod(shape) * np.dtype(dtype).itemsize
            if (
                type(item["offset"]) is not int
                or item["offset"] != offset
                or type(item["nbytes"]) is not int
                or item["nbytes"] != nbytes
                or offset + nbytes > size
            ):
                raise EmbeddingConfigurationError("Array exceeds or overlaps the reserved buffer")
            if (kind == "audio" and (len(shape) != 2 or shape[1] != 1)) or (
                kind != "audio" and (len(shape) != 3 or shape[2] != 3)
            ):
                raise EmbeddingConfigurationError("Invalid decoded media layout")
            offset += nbytes
        if kind == "video":
            assert spec.video is not None
            spec.video.validate_count(len(buffers))
            times, indices = part["times"], part["indices"]
            if (
                not isinstance(times, list)
                or not isinstance(indices, list)
                or len(times) != len(buffers)
                or len(indices) != len(buffers)
                or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times)
                or any(type(i) is not int or i < 0 for i in indices)
                or any(a >= b for a, b in zip(times, times[1:]))
                or any(a >= b for a, b in zip(indices, indices[1:]))
            ):
                raise EmbeddingConfigurationError("Invalid frame timeline")
        if kind == "audio":
            assert spec.audio is not None
            if (
                type(part["sample_rate"]) is not int
                or part["sample_rate"] != spec.audio.sample_rate
                or buffers[0]["shape"][0] > spec.audio.max_samples
            ):
                raise EmbeddingConfigurationError("Invalid audio sample rate or length")
    if offset != size or size + text_bytes > spec.max_input_bytes:
        raise EmbeddingConfigurationError("Embedding exceeds or mismatches its byte budget")
    return max(1, size + text_bytes)


def restore(parts: list[dict[str, Any]], memory: Any) -> EmbeddingContent:
    def array_value(item: dict[str, Any]) -> np.ndarray[Any, Any]:
        return np.ndarray(item["shape"], dtype=item["dtype"], buffer=memory, offset=item["offset"])

    values = []
    for part in parts:
        kind = part["kind"]
        if kind == "text":
            value = part["text"]
        elif kind == "video":
            value = VideoClip(
                tuple(array_value(item) for item in part["buffers"]), tuple(part["times"]), tuple(part["indices"])
            )
        elif kind == "audio":
            value = AudioClip(array_value(part["buffer"]), part["sample_rate"])
        else:
            value = array_value(part["buffer"])
        values.append(EmbeddingPart(EmbeddingPartSpec(kind), value))
    return EmbeddingContent(tuple(values))


def sealed_buffer(arrays: list[np.ndarray[Any, Any]], size: int) -> int | None:
    if not size:
        return None
    import fcntl
    import mmap

    fd = os.memfd_create("vane_embedding", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        # Avoid SIGBUS on first write if the backing store cannot reserve pages.
        os.posix_fallocate(fd, 0, size)
        with mmap.mmap(fd, size, access=mmap.ACCESS_WRITE) as memory:
            offset = 0
            for value in arrays:
                target = np.ndarray(value.shape, dtype=value.dtype, buffer=memory, offset=offset)
                np.copyto(target, value)
                del target
                offset += value.nbytes
        fcntl.fcntl(fd, fcntl.F_ADD_SEALS, required_seals())
        return fd
    except BaseException:
        os.close(fd)
        raise


def required_seals() -> int:
    import fcntl

    return fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL


def encode(value: Any) -> bytes:
    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_CONTROL_BYTES:
        raise EmbeddingConfigurationError("Embedding control message exceeds 1 MiB")
    return data


def check_peer(sock: socket.socket) -> None:
    _, uid, _ = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
    if uid != os.getuid():
        raise EmbeddingConfigurationError("Embedding service requires the same Linux UID")


async def _ready(sock: socket.socket, *, write: bool = False) -> None:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[None] = loop.create_future()
    add, remove = (loop.add_writer, loop.remove_writer) if write else (loop.add_reader, loop.remove_reader)
    add(sock.fileno(), lambda: None if future.done() else future.set_result(None))
    try:
        await future
    finally:
        remove(sock.fileno())


async def send(sock: socket.socket, value: Any, fd: int | None = None) -> None:
    data = encode(value)
    ancillary = [] if fd is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
    while True:
        try:
            sock.sendmsg([b"E"], ancillary)
            break
        except BlockingIOError:
            await _ready(sock, write=True)
    await asyncio.get_running_loop().sock_sendall(sock, struct.pack("!I", len(data)) + data)


async def receive(sock: socket.socket) -> tuple[Any, ReceivedFD | None]:
    fds: list[int] = []
    try:
        while True:
            try:
                marker, ancillary, flags, _ = sock.recvmsg(1, socket.CMSG_SPACE(4 * 8), socket.MSG_CMSG_CLOEXEC)
                break
            except BlockingIOError:
                await _ready(sock)
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                values = array.array("i")
                values.frombytes(data[: len(data) - len(data) % values.itemsize])
                fds.extend(values)
        if not marker:
            raise ConnectionError("Embedding peer disconnected")
        if marker != b"E" or flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC) or len(fds) > 1:
            raise EmbeddingConfigurationError("Invalid embedding IPC header")

        async def exact(size: int) -> bytes:
            data = bytearray()
            while len(data) < size:
                chunk = await asyncio.get_running_loop().sock_recv(sock, size - len(data))
                if not chunk:
                    raise ConnectionError("Embedding peer disconnected")
                data.extend(chunk)
            return bytes(data)

        size = struct.unpack("!I", await exact(4))[0]
        if not 0 < size <= MAX_CONTROL_BYTES:
            raise EmbeddingConfigurationError("Embedding control message exceeds budget")
        value = json.loads(await exact(size))
        return value, ReceivedFD(fds.pop()) if fds else None
    finally:
        for fd in fds:
            os.close(fd)


async def control(sock: socket.socket) -> Any:
    value, fd = await receive(sock)
    if fd is not None:
        fd.close()
        raise EmbeddingConfigurationError("Unexpected embedding buffer")
    return value
