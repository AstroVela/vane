# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import gc
import os

import pytest


@pytest.fixture
def pooled_shm_worker(monkeypatch):
    """Exercise the real allocation and release channel without launching a UDF."""
    from vane.execution import udf_shm_store as storage

    store = storage.LocalShmStore(4 * 1024**2)

    def acquire(client_id):
        store.add_client(client_id)
        return store

    monkeypatch.setattr(storage, "current_store", acquire)
    peer = storage.ParentShmPeer()
    client = storage.WorkerShmClient(os.dup(peer.child_sock.fileno()))
    peer.child_sock.close()
    monkeypatch.setattr(storage, "_worker_client", client)
    try:
        yield peer
    finally:
        gc.collect()
        client.close()
        peer.close_after_exit()
        assert store.snapshot()["live_allocations"] == 0
        assert store.snapshot()["mapped_capacity_bytes"] == 0
