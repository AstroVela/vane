# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Direct output serialization preserves Arrow layouts and allocation bounds."""

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from vane.execution import ref_bundle as refs


def _tables():
    tensor = pa.table(
        {"tensor": pa.FixedShapeTensorArray.from_numpy_ndarray(np.arange(96, dtype=np.float32).reshape(8, 3, 4))}
    )
    return {
        "tensor": tensor,
        "sliced_tensor": tensor.slice(2, 3),
        "nullable_slice": pa.table({"i": [0, None, 2, 3], "s": ["a", "", None, "last"]}).slice(1, 3),
        "chunked": pa.table({"x": pa.chunked_array([["a", None], ["b" * 100], [""]])}),
        "dictionary": pa.table({"d": pa.array(["a", "b", None, "a"]).dictionary_encode()}),
        "dictionary_replacement": pa.table(
            {
                "d": pa.chunked_array(
                    [pa.array(["a", "b"]).dictionary_encode(), pa.array(["c", "d"]).dictionary_encode()]
                )
            }
        ),
        "nested": pa.table({"list": [[1, 2], None, []], "struct": [{"x": 1}, None, {"x": None}]}),
        "large_binary": pa.table({"blob": pa.array([b"x" * 8192, None, b""], type=pa.large_binary())}),
        "metadata": pa.table(
            {"time": pa.array([0, None, 100], type=pa.timestamp("us", tz="UTC"))}
        ).replace_schema_metadata({b"purpose": b"direct-ipc"}),
        "empty": pa.table({"x": pa.array([], type=pa.int64())}),
    }


@pytest.mark.parametrize("case", list(_tables()))
def test_direct_ipc_roundtrip_matches_staged_size_without_staging(pooled_shm_worker, monkeypatch, case):
    table = _tables()[case]
    expected_size = refs.prepare_local_shm_block(table).ipc_size_bytes

    def forbid_staging(*args, **kwargs):
        raise AssertionError("direct IPC allocated a staging buffer")

    monkeypatch.setattr(pa, "BufferOutputStream", forbid_staging)
    block = refs.prepare_pooled_shm_block(table)
    assert block.ipc_size_bytes == expected_size
    grant = refs.request_local_shm_output_grant(expected_size)
    allocation = pooled_shm_worker.reserve_write(grant, expected_size)
    result = None
    try:
        descriptor = refs.make_pooled_shm_descriptor([block], allocation=allocation, grant_id=grant)
        result = refs.make_local_shm_ref_bundle_result_from_descriptor(descriptor)
        pooled_shm_worker.finish_write(grant)
        decoded = result[1][0].to_table()
        assert decoded.equals(table, check_metadata=True)
        del decoded
    finally:
        if result is not None:
            for ref in result[1]:
                ref.release()
        pooled_shm_worker.finish_write(grant)
        refs.release_local_shm_output_grant(grant)
    assert pooled_shm_worker.store.snapshot()["live_allocations"] == 0


@pytest.mark.parametrize("size_delta", [-16, 16])
def test_size_mismatch_cannot_overwrite_adjacent_allocation(pooled_shm_worker, size_delta):
    peer = pooled_shm_worker
    block = refs.prepare_pooled_shm_block(pa.table({"blob": [b"x" * 4096]}))
    block = replace(block, ipc_size_bytes=block.ipc_size_bytes + size_delta)
    allocation = peer.reserve_write(1, block.ipc_size_bytes)
    guard = peer.store.allocate(512)
    region = peer.store.buffer(guard.allocation)
    region[:] = b"g" * 512
    try:
        with pytest.raises((pa.ArrowException, BufferError, OSError)):
            refs.make_pooled_shm_descriptor([block], allocation=allocation, grant_id=1)
        assert bytes(region) == b"g" * 512
    finally:
        region.release()
        guard.release()
        peer.finish_write(1)
    assert peer.store.snapshot()["live_allocations"] == 0
