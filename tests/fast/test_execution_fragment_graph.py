# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Plan transport tests, not evidence that the opaque fixtures are executable."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from vane.execution.plan import Distribution, ExchangeSpec, FragmentGraph, FragmentSpec, PortSpec, ResultSpec

SCHEMA = b"opaque-native-schema"


def _fragment(name, *, partitions=1, inputs=(), outputs=("out",), schema=SCHEMA):
    return FragmentSpec(
        fragment_id=name,
        native_plan=f"opaque-plan:{name}".encode(),
        partition_count=partitions,
        inputs=tuple(PortSpec(port, schema) for port in inputs),
        outputs=tuple(PortSpec(port, schema) for port in outputs),
    )


def _edge(name, producer, consumer, *, input_port="in", distribution=Distribution.GATHER, partitioning=None):
    return ExchangeSpec(name, producer, "out", consumer, input_port, distribution, partitioning)


def _graph(fragments=None, exchanges=None, *, root="root"):
    if fragments is None:
        fragments = (_fragment("scan", partitions=3), _fragment("root", inputs=("in",)))
    if exchanges is None:
        exchanges = (_edge("gather", "scan", "root"),)
    return FragmentGraph("query-1", "engine-test", tuple(fragments), tuple(exchanges), ResultSpec(root, "out"))


def test_graph_roundtrip_does_not_encode_a_distributed_mode_or_storage_handles():
    graph = _graph()
    payload = graph.to_dict()

    assert FragmentGraph.from_dict(json.loads(json.dumps(payload)), expected_engine_identity="engine-test") == graph
    assert graph.topological_fragment_ids() == ("scan", "root")
    assert "execution" not in payload
    assert set(payload["exchanges"][0]) == {
        "exchange_id",
        "producer_fragment_id",
        "producer_port",
        "consumer_fragment_id",
        "consumer_port",
        "distribution",
        "partitioning",
    }


def test_enumeration_order_does_not_change_graph_order_or_fingerprint():
    fragments = (
        _fragment("right", partitions=2),
        _fragment("root", inputs=("left", "right")),
        _fragment("left", partitions=2),
    )
    edges = (
        _edge("right-edge", "right", "root", input_port="right"),
        _edge("left-edge", "left", "root", input_port="left"),
    )
    first = _graph(fragments, edges)
    second = _graph(tuple(reversed(fragments)), tuple(reversed(edges)))

    assert first.topological_fragment_ids() == ("left", "right", "root")
    assert first == second
    assert first.fingerprint() == second.fingerprint()
    assert replace(first, engine_identity="different-engine").fingerprint() != first.fingerprint()
    assert replace(first, query_id="another-submission").fingerprint() != first.fingerprint()


def test_two_edges_from_one_producer_do_not_duplicate_dependency_counts():
    graph = _graph(
        (_fragment("scan"), _fragment("root", inputs=("left", "right"))),
        (_edge("left", "scan", "root", input_port="left"), _edge("right", "scan", "root", input_port="right")),
    )
    assert graph.topological_fragment_ids() == ("scan", "root")


def test_scanless_constant_fragment_has_schema_without_exchanges():
    graph = _graph((_fragment("root"),), ())
    assert graph.topological_fragment_ids() == ("root",)
    assert FragmentGraph.from_dict(graph.to_dict(), expected_engine_identity="engine-test") == graph


def test_payload_buffers_and_lists_are_snapshotted():
    native_plan = bytearray(b"native-plan")
    schema = bytearray(SCHEMA)
    outputs = [PortSpec("out", schema)]
    fragment = FragmentSpec("root", native_plan, 1, [], outputs)
    fragments = [fragment]
    graph = FragmentGraph("q", "engine", fragments, [], ResultSpec("root", "out"))
    original = graph.fingerprint()

    native_plan[:] = b"changed"
    schema[:] = b"different"
    outputs.clear()
    fragments.clear()
    wire = graph.to_dict()
    wire["fragments"][0]["outputs"].clear()

    assert graph.fingerprint() == original
    with pytest.raises(FrozenInstanceError):
        graph.query_id = "other"


@pytest.mark.parametrize("version", [True, 0, 2, "1"])
def test_protocol_version_is_checked_before_native_fields_are_parsed(version):
    payload = _graph().to_dict()
    payload["protocol_version"] = version
    payload["fragments"] = ["this revision might have a different layout"]
    with pytest.raises(ValueError, match="protocol version"):
        FragmentGraph.from_dict(payload, expected_engine_identity="engine-test")


def test_foreign_engine_payload_is_rejected_before_native_fields_are_parsed():
    payload = _graph().to_dict()
    payload["engine_identity"] = "another-engine"
    payload["fragments"] = ["foreign native representation"]
    with pytest.raises(ValueError, match="engine identity"):
        FragmentGraph.from_dict(payload, expected_engine_identity="engine-test")


@pytest.mark.parametrize("native", ["", "not base64!", 123, "éé"])
def test_malformed_native_payloads_cannot_cross_the_plan_boundary(native):
    payload = _graph().to_dict()
    payload["fragments"][0]["native_plan"] = native
    with pytest.raises(ValueError, match="native_plan"):
        FragmentGraph.from_dict(payload, expected_engine_identity="engine-test")


def test_hash_partitioning_stays_native_and_changes_the_plan_identity():
    fragments = (
        _fragment("scan", partitions=2),
        _fragment("map", partitions=2, inputs=("in",)),
        _fragment("root", inputs=("in",)),
    )
    edges = (
        _edge("hash", "scan", "map", distribution=Distribution.HASH, partitioning=b"native-hash-expression"),
        _edge("gather", "map", "root"),
    )
    graph = _graph(fragments, edges)
    changed = _graph(fragments, (replace(edges[0], partitioning=b"other-native-expression"), edges[1]))
    assert graph.fingerprint() != changed.fingerprint()
    assert FragmentGraph.from_dict(graph.to_dict(), expected_engine_identity="engine-test") == graph
    with pytest.raises(ValueError, match="HASH partitioning"):
        replace(edges[0], partitioning=None)
    with pytest.raises(ValueError, match="only HASH"):
        replace(edges[1], partitioning=b"unexpected")


def test_duplicate_identities_are_rejected():
    graph = _graph()
    with pytest.raises(ValueError, match="duplicate fragment_id"):
        replace(graph, fragments=graph.fragments + (graph.fragments[0],))
    with pytest.raises(ValueError, match="duplicate exchange_id"):
        replace(graph, exchanges=graph.exchanges * 2)
    with pytest.raises(ValueError, match="duplicate inputs port"):
        _fragment("root", inputs=("in", "in"))


@pytest.mark.parametrize("field", ["producer_fragment_id", "consumer_fragment_id", "producer_port", "consumer_port"])
def test_exchanges_must_reference_real_ports(field):
    graph = _graph()
    with pytest.raises(ValueError, match="missing fragment or port"):
        replace(graph, exchanges=(replace(graph.exchanges[0], **{field: "missing"}),))


def test_schema_mismatch_and_duplicate_input_binding_are_rejected():
    with pytest.raises(ValueError, match="schema mismatch"):
        _graph((_fragment("scan"), _fragment("root", inputs=("in",), schema=b"incompatible-schema")))
    with pytest.raises(ValueError, match="more than one exchange"):
        _graph(
            (_fragment("scan"), _fragment("other"), _fragment("root", inputs=("in",))),
            (_edge("one", "scan", "root"), _edge("two", "other", "root")),
        )


def test_dangling_ports_cannot_produce_a_partial_graph():
    with pytest.raises(ValueError, match="input port must be connected"):
        _graph((_fragment("root", inputs=("missing",)),), ())
    with pytest.raises(ValueError, match="output port must feed"):
        _graph((_fragment("orphan"), _fragment("root")), ())


def test_cycle_is_rejected_even_when_it_has_an_edge_to_the_result():
    with pytest.raises(ValueError, match="contains a cycle"):
        _graph(
            (_fragment("a", inputs=("in",)), _fragment("b", inputs=("in",)), _fragment("root", inputs=("in",))),
            (_edge("ab", "a", "b"), _edge("ba", "b", "a"), _edge("br", "b", "root")),
        )


def test_gather_and_result_partition_rules_are_enforced():
    with pytest.raises(ValueError, match="one partition"):
        _graph((_fragment("root", partitions=2),), ())
    with pytest.raises(ValueError, match="GATHER requires"):
        _graph(
            (_fragment("scan"), _fragment("map", partitions=2, inputs=("in",)), _fragment("root", inputs=("in",))),
            (_edge("sm", "scan", "map"), _edge("mr", "map", "root")),
        )


@pytest.mark.parametrize("partitions", [True, 0, -1, 1.5, "2", 2**31])
def test_partition_count_cannot_be_coerced_or_overflow(partitions):
    with pytest.raises(ValueError, match="partition_count"):
        _fragment("root", partitions=partitions)


def test_unknown_plan_fields_are_not_silently_dropped():
    payload = _graph().to_dict()
    payload["legacy_runner"] = "fte"
    with pytest.raises(ValueError, match="exactly these fields"):
        FragmentGraph.from_dict(payload, expected_engine_identity="engine-test")
    del payload["legacy_runner"]
    payload["exchanges"][0]["files"] = ["old-manifest"]
    with pytest.raises(ValueError, match="exactly these fields"):
        FragmentGraph.from_dict(payload, expected_engine_identity="engine-test")
