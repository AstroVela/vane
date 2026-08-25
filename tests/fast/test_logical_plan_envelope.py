# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest

import vane

_MAGIC = b"VANEPLAN"
_SOURCE_ID_SIZE_OFFSET = 12
_PAYLOAD_SIZE_OFFSET = 16
_HEADER_SIZE = 24


def _require_ray_cxx():
    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")
    return ray_cxx


def _logical_plan_state():
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    plan = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        connection.sql("SELECT 42 AS answer"),
        "logical-plan-envelope",
    )
    return ray_cxx, connection, list(plan.__getstate__())


def _restore(ray_cxx, state):
    restored = ray_cxx.PyLogicalPlan.__new__(ray_cxx.PyLogicalPlan)
    restored.__setstate__(tuple(state))
    return restored


def test_logical_plan_pickle_uses_a_versioned_envelope():
    ray_cxx, connection, state = _logical_plan_state()
    payload = state[1]

    assert payload.startswith(_MAGIC)
    assert int.from_bytes(payload[8:12], "little") == 1
    source_id_size = int.from_bytes(payload[_SOURCE_ID_SIZE_OFFSET:_PAYLOAD_SIZE_OFFSET], "little")
    logical_payload_size = int.from_bytes(payload[_PAYLOAD_SIZE_OFFSET:_HEADER_SIZE], "little")
    assert source_id_size > 0
    assert logical_payload_size > 0
    assert len(payload) == _HEADER_SIZE + source_id_size + logical_payload_size

    restored = pickle.loads(pickle.dumps(_restore(ray_cxx, state)))
    assert restored.to_physical_plan(connection) is not None


def test_distributed_write_plan_rejects_source_explicit_transaction():
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("BEGIN")
    try:
        with pytest.raises(Exception, match="cannot participate in an explicit transaction"):
            ray_cxx.PyLogicalPlan.from_duckdb_write_relation(
                connection.sql("SELECT 42 AS value"),
                "explicit-transaction-write",
            )
    finally:
        connection.execute("ROLLBACK")


def test_distributed_write_plan_rejects_read_relation():
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()

    with pytest.raises(Exception, match="requires a write relation"):
        ray_cxx.PyLogicalPlan.from_duckdb_write_relation(
            connection.sql("SELECT 42 AS value"),
            "read-passed-to-write-path",
        )


def test_datasink_plan_factory_requires_exact_terminal_relation():
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    relation = connection.sql("SELECT 42 AS value")
    terminal = relation._mark_datasink("datasink-terminal")

    with pytest.raises(ValueError, match="does not accept terminal write relations"):
        ray_cxx.PyLogicalPlan.from_duckdb_relation(terminal, "datasink-passed-to-read-path")
    with pytest.raises(ValueError, match="requires a DataSink relation"):
        ray_cxx.PyLogicalPlan.from_duckdb_datasink_relation(relation, "read-passed-to-datasink-path")


def test_datasink_plan_factory_rechecks_transaction_at_serialization_boundary():
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    terminal = connection.sql("SELECT 42 AS value")._mark_datasink("datasink-before-transaction")
    connection.execute("BEGIN")
    try:
        with pytest.raises(ValueError, match="cannot participate in an explicit transaction"):
            ray_cxx.PyLogicalPlan.from_duckdb_datasink_relation(
                terminal,
                "explicit-transaction-datasink-plan",
            )
    finally:
        connection.execute("ROLLBACK")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: b"raw-logical-plan", "not a Vane logical plan envelope"),
        (
            lambda payload: payload[:8] + (2).to_bytes(4, "little") + payload[12:],
            "Unsupported logical plan protocol version",
        ),
        (lambda payload: payload[:-1], "payload length mismatch"),
        (lambda payload: payload + b"trailing-byte", "payload length mismatch"),
    ],
)
def test_logical_plan_pickle_rejects_invalid_envelopes(mutate, message):
    ray_cxx, _, state = _logical_plan_state()
    state[1] = mutate(state[1])

    with pytest.raises(Exception, match=message):
        _restore(ray_cxx, state)


def test_logical_plan_pickle_rejects_a_different_engine_source_id():
    ray_cxx, _, state = _logical_plan_state()
    payload = bytearray(state[1])
    source_id_size = int.from_bytes(payload[_SOURCE_ID_SIZE_OFFSET:_PAYLOAD_SIZE_OFFSET], "little")
    assert source_id_size > 0
    payload[_HEADER_SIZE] ^= 1
    state[1] = bytes(payload)

    with pytest.raises(Exception, match="Logical plan SourceID mismatch"):
        _restore(ray_cxx, state)
