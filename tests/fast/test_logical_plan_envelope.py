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


def test_logical_plan_operation_fingerprint_is_stable_and_plan_specific():
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    connection.execute("SET GLOBAL http_proxy_username='credential-a'")
    relation = connection.sql("SELECT i FROM range(4) values_table(i)")
    first = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "operation-a")
    repeated = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "operation-b")
    different = ray_cxx.PyLogicalPlan.from_duckdb_relation(
        connection.sql("SELECT i FROM range(5) values_table(i)"),
        "operation-a",
    )

    connection.execute("SET GLOBAL http_proxy_username='credential-b'")
    refreshed_snapshot = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "operation-a")

    def proxy_username(plan):
        snapshot = plan.__getstate__()[3]
        return next(setting["value"] for setting in snapshot["settings"] if setting["name"] == "http_proxy_username")

    assert proxy_username(first) == "credential-a"
    assert proxy_username(refreshed_snapshot) == "credential-b"
    assert first.operation_fingerprint() == repeated.operation_fingerprint()
    assert first.operation_fingerprint() != different.operation_fingerprint()
    assert first.operation_fingerprint() == refreshed_snapshot.operation_fingerprint()
    assert pickle.loads(pickle.dumps(first)).operation_fingerprint() == first.operation_fingerprint()


def test_logical_plan_operation_fingerprint_includes_semantic_connection_settings():
    ray_cxx = _require_ray_cxx()
    connection = vane.connect()
    relation = connection.sql("SELECT TIMESTAMP '2026-01-01 00:00:00' AS value")

    connection.execute("SET TimeZone='UTC'")
    utc = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "timezone-a")
    connection.execute("SET TimeZone='America/Los_Angeles'")
    los_angeles = ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "timezone-b")

    assert utc.operation_fingerprint() != los_angeles.operation_fingerprint()


def test_logical_plan_operation_fingerprint_rejects_duplicate_connection_settings():
    ray_cxx, _, state = _logical_plan_state()
    changed_state = list(state)
    changed_snapshot = dict(changed_state[3])
    changed_snapshot["settings"] = [
        *changed_snapshot["settings"],
        {"name": "TimeZone", "value": "UTC", "input_type": "VARCHAR"},
        {"name": "timezone", "value": "America/Los_Angeles", "input_type": "VARCHAR"},
    ]
    changed_state[3] = changed_snapshot

    with pytest.raises(Exception, match="settings contain duplicate name timezone"):
        _restore(ray_cxx, changed_state).operation_fingerprint()


def test_logical_plan_operation_fingerprint_ignores_session_transport_identity():
    ray_cxx, _, state = _logical_plan_state()
    original = _restore(ray_cxx, state)
    changed_state = list(state)
    changed_snapshot = dict(changed_state[3])
    changed_session = dict(changed_snapshot["vane_session"])
    changed_session["id"] = "different-transport-session"
    changed_snapshot["vane_session"] = changed_session
    changed_state[3] = changed_snapshot

    assert _restore(ray_cxx, changed_state).operation_fingerprint() == original.operation_fingerprint()


def test_logical_plan_operation_fingerprint_includes_python_udf_implementation():
    ray_cxx, _, state = _logical_plan_state()
    state[2] = [
        {
            "kind": "scalar",
            "name": "fingerprint_add_one",
            "digest": "implementation-a",
            "function_pickle": b"implementation-a",
            "parameters": ["INTEGER"],
            "return_type": "INTEGER",
            "udf_type": "native",
            "null_handling": 0,
            "exception_handling": 0,
            "side_effects": False,
        }
    ]
    plan = _restore(ray_cxx, state)
    changed_state = list(plan.__getstate__())
    changed_registrations = [dict(registration) for registration in changed_state[2]]
    assert changed_registrations
    changed_registrations[0]["function_pickle"] += b"different-implementation"
    changed_registrations[0]["digest"] = "different-implementation-digest"
    changed_state[2] = changed_registrations

    changed = _restore(ray_cxx, changed_state)
    assert changed.operation_fingerprint() != plan.operation_fingerprint()

    duplicate_state = list(plan.__getstate__())
    duplicate_registration = dict(duplicate_state[2][0])
    duplicate_registration["name"] = duplicate_registration["name"].upper()
    duplicate_state[2] = [*duplicate_state[2], duplicate_registration]
    with pytest.raises(Exception, match="UDF registrations contain duplicate name fingerprint_add_one"):
        _restore(ray_cxx, duplicate_state).operation_fingerprint()

    invalid_name_state = list(plan.__getstate__())
    invalid_name_registration = dict(invalid_name_state[2][0])
    invalid_name_registration["name"] = b"fingerprint_add_one"
    invalid_name_state[2] = [invalid_name_registration]
    with pytest.raises(Exception, match="UDF registration name must be a string"):
        _restore(ray_cxx, invalid_name_state).operation_fingerprint()

    ordered_schema_state = list(plan.__getstate__())
    ordered_schema_registration = dict(ordered_schema_state[2][0])
    ordered_schema_registration.update(
        kind="table",
        schema={"first": "INTEGER", "second": "VARCHAR"},
        batch_size=None,
    )
    ordered_schema_state[2] = [ordered_schema_registration]
    reversed_schema_state = list(plan.__getstate__())
    reversed_schema_registration = dict(ordered_schema_registration)
    reversed_schema_registration["schema"] = {"second": "VARCHAR", "first": "INTEGER"}
    reversed_schema_state[2] = [reversed_schema_registration]

    assert (
        _restore(ray_cxx, ordered_schema_state).operation_fingerprint()
        != _restore(ray_cxx, reversed_schema_state).operation_fingerprint()
    )


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
