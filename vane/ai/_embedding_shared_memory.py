# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Service-owned SharedMemory with independently launched, non-owning clients."""

from __future__ import annotations

import contextlib
import mmap
import os
import sys
from multiprocessing import shared_memory
from typing import Any, Iterator

from vane.ai._embedding_inputs import EmbeddingConfigurationError


@contextlib.contextmanager
def attach(name: str, size: int) -> Iterator[Any]:
    """Borrow a mapping without letting the client's tracker unlink the owner.

    Python 3.10-3.12 have no track=False. Opening the existing POSIX mapping
    directly avoids registering/unregistering a shared resource tracker, which
    would also unregister an owner in the same process or multiprocessing tree.
    Windows never uses that tracker and deletes only after the last handle closes.
    """
    if os.name == "posix" and sys.version_info < (3, 13):
        posix = shared_memory._posixshmem  # type: ignore[attr-defined]
        fd = posix.shm_open("/" + name, os.O_RDWR, mode=0o600)
        try:
            if os.fstat(fd).st_size != size:
                raise EmbeddingConfigurationError("Embedding shared memory size mismatch")
            mapping = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        try:
            yield mapping
        finally:
            mapping.close()
        return
    options = {"track": False} if sys.version_info >= (3, 13) else {}
    memory = shared_memory.SharedMemory(name=name, **options)
    try:
        # Windows rounds the mapping size up to allocation granularity.
        if memory.size < size or os.name != "nt" and memory.size != size:
            raise EmbeddingConfigurationError("Embedding shared memory size mismatch")
        yield memory.buf
    finally:
        memory.close()


def create(size: int) -> shared_memory.SharedMemory:
    memory = shared_memory.SharedMemory(create=True, size=size)
    try:
        # Reserve backing pages before returning a buffer to a producer on Linux.
        # Other platforms use their native SharedMemory allocation semantics.
        if sys.platform == "linux":
            os.posix_fallocate(memory._fd, 0, size)  # type: ignore[attr-defined]
        return memory
    except BaseException:
        release(memory)
        raise


def release(memory: shared_memory.SharedMemory) -> None:
    try:
        memory.unlink()
    finally:
        memory.close()
