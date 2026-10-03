# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Validated, mode-independent plan contracts for Ray execution.

Native plan, schema and hash-expression bytes are opaque here. Their decoding
and executable capability checks belong to the native compiler/runtime. This
module neither builds executable plans from SQL nor starts tasks. The public
local execution path does not use this distributed graph.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import heapq
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

PLAN_PROTOCOL_VERSION = 1


def _name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly these fields: {', '.join(sorted(expected))}")


def _payload(value: object, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)) or not value:
        raise ValueError(f"{label} must contain non-empty native bytes")
    return bytes(value)


def _decode(value: str, label: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be base64 text")
    try:
        return _payload(base64.b64decode(value, validate=True), label)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{label} must contain non-empty base64-encoded native bytes") from exc


def _encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _items(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{label} must be a sequence")
    return tuple(value)


class Distribution(str, Enum):
    GATHER = "gather"
    HASH = "hash"
    BROADCAST = "broadcast"


@dataclass(frozen=True)
class ScanSplitSpec:
    split_id: str
    payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _name(self.split_id, "split_id")
        object.__setattr__(self, "payload", _payload(self.payload, "split payload"))

    def to_dict(self) -> dict[str, Any]:
        return {"split_id": self.split_id, "payload": _encode(self.payload)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScanSplitSpec:
        _fields(value, {"split_id", "payload"}, cls.__name__)
        return cls(value["split_id"], _decode(value["payload"], "split payload"))


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    function_name: str
    capability: str
    codec: str
    requires_snapshot: bool
    splits: tuple[ScanSplitSpec, ...]

    def __post_init__(self) -> None:
        for name in ("source_id", "function_name", "capability", "codec"):
            _name(getattr(self, name), name)
        if type(self.requires_snapshot) is not bool:
            raise ValueError("requires_snapshot must be a boolean")
        splits = _items(self.splits, "splits")
        if any(not isinstance(split, ScanSplitSpec) for split in splits):
            raise ValueError("splits must contain ScanSplitSpec objects")
        if len({split.split_id for split in splits}) != len(splits):
            raise ValueError("duplicate split_id")
        # Enumeration order is native source state; preserve it for assignment.
        object.__setattr__(self, "splits", splits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "function_name": self.function_name,
            "capability": self.capability,
            "codec": self.codec,
            "requires_snapshot": self.requires_snapshot,
            "splits": [split.to_dict() for split in self.splits],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceSpec:
        _fields(
            value, {"source_id", "function_name", "capability", "codec", "requires_snapshot", "splits"}, cls.__name__
        )
        return cls(
            source_id=value["source_id"],
            function_name=value["function_name"],
            capability=value["capability"],
            codec=value["codec"],
            requires_snapshot=value["requires_snapshot"],
            splits=tuple(ScanSplitSpec.from_dict(split) for split in _items(value["splits"], "splits")),
        )


@dataclass(frozen=True)
class PortSpec:
    port_id: str
    schema: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _name(self.port_id, "port_id")
        object.__setattr__(self, "schema", _payload(self.schema, "schema"))

    def to_dict(self) -> dict[str, Any]:
        return {"port_id": self.port_id, "schema": _encode(self.schema)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PortSpec:
        _fields(value, {"port_id", "schema"}, cls.__name__)
        return cls(value["port_id"], _decode(value["schema"], "schema"))


@dataclass(frozen=True)
class FragmentSpec:
    fragment_id: str
    native_plan: bytes = field(repr=False)
    partition_count: int
    inputs: tuple[PortSpec, ...]
    outputs: tuple[PortSpec, ...]
    sources: tuple[SourceSpec, ...] = ()
    source_dependencies: tuple[SourceSpec, ...] = ()

    def __post_init__(self) -> None:
        _name(self.fragment_id, "fragment_id")
        object.__setattr__(self, "native_plan", _payload(self.native_plan, "native_plan"))
        if type(self.partition_count) is not int or not 1 <= self.partition_count <= 2**31 - 1:
            raise ValueError("partition_count must be an integer between 1 and 2147483647")
        for label in ("inputs", "outputs"):
            ports = _items(getattr(self, label), label)
            if any(not isinstance(port, PortSpec) for port in ports):
                raise ValueError(f"{label} must contain PortSpec objects")
            ids = [port.port_id for port in ports]
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate {label} port on fragment {self.fragment_id}")
            object.__setattr__(self, label, tuple(sorted(ports, key=lambda port: port.port_id)))
        if not self.outputs:
            raise ValueError("a read-only fragment must have an output port")
        for label in ("sources", "source_dependencies"):
            sources = _items(getattr(self, label), label)
            if any(not isinstance(source, SourceSpec) for source in sources):
                raise ValueError(f"{label} must contain SourceSpec objects")
            if len({source.source_id for source in sources}) != len(sources):
                raise ValueError(f"duplicate source_id in {label}")
            object.__setattr__(self, label, tuple(sorted(sources, key=lambda source: source.source_id)))

    @property
    def required_capabilities(self) -> tuple[str, ...]:
        return tuple(sorted({source.capability for source in self.sources + self.source_dependencies}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fragment_id": self.fragment_id,
            "native_plan": _encode(self.native_plan),
            "partition_count": self.partition_count,
            "inputs": [port.to_dict() for port in self.inputs],
            "outputs": [port.to_dict() for port in self.outputs],
            "sources": [source.to_dict() for source in self.sources],
            "source_dependencies": [source.to_dict() for source in self.source_dependencies],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FragmentSpec:
        _fields(
            value,
            {"fragment_id", "native_plan", "partition_count", "inputs", "outputs", "sources", "source_dependencies"},
            cls.__name__,
        )
        return cls(
            fragment_id=value["fragment_id"],
            native_plan=_decode(value["native_plan"], "native_plan"),
            partition_count=value["partition_count"],
            inputs=tuple(PortSpec.from_dict(port) for port in _items(value["inputs"], "inputs")),
            outputs=tuple(PortSpec.from_dict(port) for port in _items(value["outputs"], "outputs")),
            sources=tuple(SourceSpec.from_dict(source) for source in _items(value["sources"], "sources")),
            source_dependencies=tuple(
                SourceSpec.from_dict(source) for source in _items(value["source_dependencies"], "source_dependencies")
            ),
        )


@dataclass(frozen=True)
class ExchangeSpec:
    exchange_id: str
    producer_fragment_id: str
    producer_port: str
    consumer_fragment_id: str
    consumer_port: str
    distribution: Distribution
    partitioning: bytes | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name in ("exchange_id", "producer_fragment_id", "producer_port", "consumer_fragment_id", "consumer_port"):
            _name(getattr(self, name), name)
        try:
            object.__setattr__(self, "distribution", Distribution(self.distribution))
        except (ValueError, TypeError) as exc:
            raise ValueError("distribution must be 'gather', 'hash' or 'broadcast'") from exc
        if self.distribution is Distribution.HASH:
            object.__setattr__(self, "partitioning", _payload(self.partitioning, "HASH partitioning"))
        elif self.partitioning is not None:
            raise ValueError("only HASH accepts a native partitioning expression")

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_id": self.exchange_id,
            "producer_fragment_id": self.producer_fragment_id,
            "producer_port": self.producer_port,
            "consumer_fragment_id": self.consumer_fragment_id,
            "consumer_port": self.consumer_port,
            "distribution": self.distribution.value,
            "partitioning": _encode(self.partitioning) if self.partitioning is not None else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExchangeSpec:
        _fields(
            value,
            {
                "exchange_id",
                "producer_fragment_id",
                "producer_port",
                "consumer_fragment_id",
                "consumer_port",
                "distribution",
                "partitioning",
            },
            cls.__name__,
        )
        values = dict(value)
        if values["partitioning"] is not None:
            values["partitioning"] = _decode(values["partitioning"], "partitioning")
        return cls(**values)


@dataclass(frozen=True)
class ResultSpec:
    fragment_id: str
    output_port: str

    def __post_init__(self) -> None:
        _name(self.fragment_id, "result fragment_id")
        _name(self.output_port, "result output_port")

    def to_dict(self) -> dict[str, Any]:
        return {"fragment_id": self.fragment_id, "output_port": self.output_port}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResultSpec:
        _fields(value, {"fragment_id", "output_port"}, cls.__name__)
        return cls(**value)


@dataclass(frozen=True)
class FragmentGraph:
    query_id: str
    engine_identity: str
    fragments: tuple[FragmentSpec, ...]
    exchanges: tuple[ExchangeSpec, ...]
    result: ResultSpec
    protocol_version: int = PLAN_PROTOCOL_VERSION
    _order: tuple[str, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _name(self.query_id, "query_id")
        _name(self.engine_identity, "engine_identity")
        if type(self.protocol_version) is not int or self.protocol_version != PLAN_PROTOCOL_VERSION:
            raise ValueError(f"unsupported plan protocol version: {self.protocol_version!r}")
        if not isinstance(self.result, ResultSpec):
            raise ValueError("result must be a ResultSpec")
        fragments = _items(self.fragments, "fragments")
        exchanges = _items(self.exchanges, "exchanges")
        if not fragments or any(not isinstance(fragment, FragmentSpec) for fragment in fragments):
            raise ValueError("fragments must contain at least one FragmentSpec")
        if any(not isinstance(exchange, ExchangeSpec) for exchange in exchanges):
            raise ValueError("exchanges must contain ExchangeSpec objects")
        object.__setattr__(self, "fragments", tuple(sorted(fragments, key=lambda fragment: fragment.fragment_id)))
        object.__setattr__(self, "exchanges", tuple(sorted(exchanges, key=lambda exchange: exchange.exchange_id)))
        self._validate()

    def _validate(self) -> None:
        fragments = {fragment.fragment_id: fragment for fragment in self.fragments}
        if len(fragments) != len(self.fragments):
            raise ValueError("duplicate fragment_id")
        if len({edge.exchange_id for edge in self.exchanges}) != len(self.exchanges):
            raise ValueError("duplicate exchange_id")
        inputs = {(fragment.fragment_id, port.port_id): port for fragment in self.fragments for port in fragment.inputs}
        outputs = {
            (fragment.fragment_id, port.port_id): port for fragment in self.fragments for port in fragment.outputs
        }
        root = (self.result.fragment_id, self.result.output_port)
        if root not in outputs:
            raise ValueError("result references a missing output port")
        if fragments[self.result.fragment_id].partition_count != 1:
            raise ValueError("the root result fragment must have one partition; add a GATHER fragment")

        connected_inputs: set[tuple[str, str]] = set()
        connected_outputs = {root}
        parents: dict[str, set[str]] = {fragment_id: set() for fragment_id in fragments}
        children: dict[str, set[str]] = {fragment_id: set() for fragment_id in fragments}
        for edge in self.exchanges:
            producer = (edge.producer_fragment_id, edge.producer_port)
            consumer = (edge.consumer_fragment_id, edge.consumer_port)
            if producer not in outputs or consumer not in inputs:
                raise ValueError(f"exchange {edge.exchange_id} references a missing fragment or port")
            if consumer in connected_inputs:
                raise ValueError(f"input {consumer!r} has more than one exchange")
            if outputs[producer].schema != inputs[consumer].schema:
                raise ValueError(f"schema mismatch on exchange {edge.exchange_id}")
            if edge.distribution is Distribution.GATHER and fragments[edge.consumer_fragment_id].partition_count != 1:
                raise ValueError("GATHER requires exactly one consumer partition")
            connected_inputs.add(consumer)
            connected_outputs.add(producer)
            parents[edge.consumer_fragment_id].add(edge.producer_fragment_id)
            children[edge.producer_fragment_id].add(edge.consumer_fragment_id)

        if connected_inputs != set(inputs):
            raise ValueError("every declared input port must be connected")
        if connected_outputs != set(outputs):
            raise ValueError("every declared output port must feed an exchange or the result")
        if children[self.result.fragment_id]:
            raise ValueError("the result fragment must be terminal")

        remaining = {fragment_id: len(dependencies) for fragment_id, dependencies in parents.items()}
        ready = [fragment_id for fragment_id, count in remaining.items() if count == 0]
        heapq.heapify(ready)
        ordered: list[str] = []
        while ready:
            current = heapq.heappop(ready)
            ordered.append(current)
            for child in sorted(children[current]):
                remaining[child] -= 1
                if remaining[child] == 0:
                    heapq.heappush(ready, child)
        if len(ordered) != len(fragments):
            raise ValueError("fragment graph contains a cycle")

        reachable = {self.result.fragment_id}
        pending = [self.result.fragment_id]
        while pending:
            current = pending.pop()
            for parent in parents[current] - reachable:
                reachable.add(parent)
                pending.append(parent)
        if reachable != set(fragments):
            raise ValueError("every fragment must contribute to the root result")
        object.__setattr__(self, "_order", tuple(ordered))

    def topological_fragment_ids(self) -> tuple[str, ...]:
        return self._order

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "query_id": self.query_id,
            "engine_identity": self.engine_identity,
            "fragments": [fragment.to_dict() for fragment in self.fragments],
            "exchanges": [exchange.to_dict() for exchange in self.exchanges],
            "result": self.result.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, expected_engine_identity: str) -> FragmentGraph:
        _fields(
            value, {"protocol_version", "query_id", "engine_identity", "fragments", "exchanges", "result"}, cls.__name__
        )
        # Reject a different revision before attempting to interpret its layout.
        revision = value["protocol_version"]
        if type(revision) is not int or revision != PLAN_PROTOCOL_VERSION:
            raise ValueError(f"unsupported plan protocol version: {revision!r}")
        _name(expected_engine_identity, "expected_engine_identity")
        if value["engine_identity"] != expected_engine_identity:
            raise ValueError("fragment graph engine identity does not match this runtime")
        return cls(
            query_id=value["query_id"],
            engine_identity=value["engine_identity"],
            fragments=tuple(FragmentSpec.from_dict(item) for item in _items(value["fragments"], "fragments")),
            exchanges=tuple(ExchangeSpec.from_dict(item) for item in _items(value["exchanges"], "exchanges")),
            result=ResultSpec.from_dict(value["result"]),
            protocol_version=revision,
        )

    def fingerprint(self) -> str:
        """Digest this submission graph, not a complete execution cache key.

        A future cache must also include execution options and connection/source
        snapshots. Native payloads must be validated before executing a graph.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()
