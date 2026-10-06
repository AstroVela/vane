# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Immutable Ray query blueprints and worker plan preparation.

No task, exchange store, reservation or scheduler is created here. Worker
preparation takes an exclusive, query-owned connection. Local native queries
do not serialize this protocol and cannot enter this compiler.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vane.execution.compiler import FragmentCompileOptions, _native_fragment, _native_graph
from vane.execution.plan import (
    Distribution,
    FragmentGraph,
    _decode,
    _encode,
    _fields,
    _items,
    _name,
    _payload,
)
from vane.execution.query_options import DistributedMode, QueryExecutionOptions, RayExecution
from vane.execution.resource_demand import ResourceDemand

SUBMISSION_PROTOCOL_VERSION = 1
TYPE_PROFILE = "vane.analytical-types:4"
CONNECTION_PROFILE = "vane.builtin-session:1"


@dataclass(frozen=True)
class FragmentSourceSnapshot:
    fragment_id: str
    payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _name(self.fragment_id, "fragment_id")
        object.__setattr__(self, "payload", _payload(self.payload, "source snapshot"))

    def to_dict(self) -> dict[str, str]:
        return {"fragment_id": self.fragment_id, "payload": _encode(self.payload)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FragmentSourceSnapshot:
        _fields(value, {"fragment_id", "payload"}, cls.__name__)
        return cls(value["fragment_id"], _decode(value["payload"], "source snapshot"))


@dataclass(frozen=True)
class PlanCapabilities:
    """Native compiler/loader capabilities, not runtime or store readiness."""

    engine_identity: str
    protocol_version: int
    type_profile: str
    connection_profile: str
    distributions: tuple[Distribution, ...]
    scans: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in ("engine_identity", "type_profile", "connection_profile"):
            _name(getattr(self, name), name)
        if type(self.protocol_version) is not int or self.protocol_version < 1:
            raise ValueError("capability protocol_version must be a positive integer")
        distributions = tuple(Distribution(item) for item in _items(self.distributions, "distributions"))
        if len(set(distributions)) != len(distributions):
            raise ValueError("duplicate distribution capability")
        object.__setattr__(self, "distributions", tuple(sorted(distributions, key=lambda item: item.value)))
        scans = []
        for item in _items(self.scans, "scans"):
            pair = _items(item, "scan capability")
            if len(pair) != 2:
                raise ValueError("scan capability requires identity and codec")
            _name(pair[0], "scan capability")
            _name(pair[1], "scan codec")
            scans.append((pair[0], pair[1]))
        if len(set(scans)) != len(scans):
            raise ValueError("duplicate scan capability")
        object.__setattr__(self, "scans", tuple(sorted(scans)))


def native_plan_capabilities(connection: Any) -> PlanCapabilities:
    from vane._native import execution_plan

    return PlanCapabilities(**execution_plan.compiler_capabilities(connection))


@dataclass(frozen=True)
class RayQuerySpec:
    graph: FragmentGraph
    options: QueryExecutionOptions
    resources: ResourceDemand
    connection_snapshot: bytes = field(repr=False)
    source_snapshots: tuple[FragmentSourceSnapshot, ...]
    result_names: tuple[str, ...]
    protocol_version: int = SUBMISSION_PROTOCOL_VERSION
    type_profile: str = TYPE_PROFILE
    connection_profile: str = CONNECTION_PROFILE

    def __post_init__(self) -> None:
        if type(self.protocol_version) is not int or self.protocol_version != SUBMISSION_PROTOCOL_VERSION:
            raise ValueError("unsupported submission protocol version")
        if self.type_profile != TYPE_PROFILE or self.connection_profile != CONNECTION_PROFILE:
            raise ValueError("unsupported submission type or connection profile")
        if not isinstance(self.graph, FragmentGraph):
            raise ValueError("graph must be FragmentGraph")
        if not isinstance(self.options, QueryExecutionOptions) or not isinstance(self.options.target, RayExecution):
            raise ValueError("Ray submissions require RayExecution")
        if not isinstance(self.resources, ResourceDemand):
            raise ValueError("resources must be ResourceDemand")
        object.__setattr__(self, "connection_snapshot", _payload(self.connection_snapshot, "connection snapshot"))
        sources = _items(self.source_snapshots, "source_snapshots")
        if any(not isinstance(source, FragmentSourceSnapshot) for source in sources):
            raise ValueError("source_snapshots must contain FragmentSourceSnapshot objects")
        ids = [source.fragment_id for source in sources]
        if len(set(ids)) != len(ids) or set(ids) != {fragment.fragment_id for fragment in self.graph.fragments}:
            raise ValueError("each fragment requires exactly one source snapshot, including source-free fragments")
        object.__setattr__(self, "source_snapshots", tuple(sorted(sources, key=lambda source: source.fragment_id)))
        names = _items(self.result_names, "result_names")
        if not names or any(not isinstance(name, str) for name in names):
            raise ValueError("result_names must contain column names")
        object.__setattr__(self, "result_names", names)
        partitions = [fragment.partition_count for fragment in self.graph.fragments]
        contexts = max(partitions) if self.requires_replay else sum(partitions)
        if self.resources.task_contexts < contexts:
            raise ValueError(f"submission requires at least {contexts} task contexts")
        if not self.resources.memory.exchange_bytes or not self.resources.memory.staging_bytes:
            raise ValueError("Ray submissions require exchange and staging memory")
        if self.requires_replay and any(
            source.requires_snapshot
            for fragment in self.graph.fragments
            for source in fragment.sources + fragment.source_dependencies
        ):
            raise ValueError("FTE requires immutable source versions; ordinary Parquet files are not replayable")

    @property
    def query_id(self) -> str:
        return self.graph.query_id

    @property
    def requires_replay(self) -> bool:
        assert isinstance(self.options.target, RayExecution)
        return self.options.target.mode is DistributedMode.FTE

    @property
    def result_schema(self) -> bytes:
        root = next(
            fragment for fragment in self.graph.fragments if fragment.fragment_id == self.graph.result.fragment_id
        )
        return next(port.schema for port in root.outputs if port.port_id == self.graph.result.output_port)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "type_profile": self.type_profile,
            "connection_profile": self.connection_profile,
            "graph": self.graph.to_dict(),
            "options": self.options.to_dict(),
            "resources": self.resources.to_dict(),
            "connection_snapshot": _encode(self.connection_snapshot),
            "source_snapshots": [source.to_dict() for source in self.source_snapshots],
            "result_names": list(self.result_names),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, expected_engine_identity: str) -> RayQuerySpec:
        _fields(
            value,
            {
                "protocol_version",
                "type_profile",
                "connection_profile",
                "graph",
                "options",
                "resources",
                "connection_snapshot",
                "source_snapshots",
                "result_names",
            },
            cls.__name__,
        )
        if type(value["protocol_version"]) is not int or value["protocol_version"] != SUBMISSION_PROTOCOL_VERSION:
            raise ValueError("unsupported submission protocol version")
        return cls(
            graph=FragmentGraph.from_dict(value["graph"], expected_engine_identity=expected_engine_identity),
            options=QueryExecutionOptions.from_dict(value["options"]),
            resources=ResourceDemand.from_dict(value["resources"]),
            connection_snapshot=_decode(value["connection_snapshot"], "connection snapshot"),
            source_snapshots=tuple(
                FragmentSourceSnapshot.from_dict(item) for item in _items(value["source_snapshots"], "source_snapshots")
            ),
            result_names=value["result_names"],
            protocol_version=value["protocol_version"],
            type_profile=value["type_profile"],
            connection_profile=value["connection_profile"],
        )

    def cache_key(self) -> str:
        """Identity for a reusable plan blueprint, never cached results or attempts."""
        value = self.to_dict()
        del value["graph"]["query_id"]
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def prepare_ray_query(
    connection: Any,
    sql: str,
    *,
    query_id: str,
    options: QueryExecutionOptions,
    resources: ResourceDemand,
    compile_options: FragmentCompileOptions = FragmentCompileOptions(),
) -> RayQuerySpec:
    """Capture, bind and freeze a supported query without scheduling execution."""
    from vane._native import execution_plan

    if not isinstance(options, QueryExecutionOptions) or not isinstance(options.target, RayExecution):
        raise ValueError("Ray submissions require RayExecution")
    if not isinstance(resources, ResourceDemand):
        raise ValueError("resources must be ResourceDemand")
    if not isinstance(compile_options, FragmentCompileOptions):
        raise ValueError("compile_options must be FragmentCompileOptions")
    _name(sql, "sql")
    _name(query_id, "query_id")
    value = execution_plan.compile_submission(
        connection,
        sql,
        query_id,
        compile_options.partition_count,
        compile_options.hash_columns,
        options.target.mode is DistributedMode.FTE,
    )
    return _submission(value, options, resources)


def stage_ray_query(
    connection: Any,
    sql: str,
    *,
    query_id: str,
    options: QueryExecutionOptions,
    resources: ResourceDemand,
    source_directory: str,
    source_budget: int,
    compile_options: FragmentCompileOptions = FragmentCompileOptions(),
) -> tuple[RayQuerySpec, int]:
    """Copy file inputs before binding; the admitted query owns partial copies.

    The pure planning entry above never creates these files. The execution
    service must hold the store reservation and lease before entering here.
    """
    from vane._native import execution_plan

    if not isinstance(options, QueryExecutionOptions) or not isinstance(options.target, RayExecution):
        raise ValueError("FTE staging requires RayExecution")
    if options.target.mode is not DistributedMode.FTE:
        raise ValueError("file staging requires FTE execution")
    if not isinstance(resources, ResourceDemand) or not isinstance(compile_options, FragmentCompileOptions):
        raise ValueError("invalid FTE resources or compile options")
    _name(sql, "sql")
    _name(query_id, "query_id")
    value = execution_plan.stage_submission(
        connection,
        sql,
        query_id,
        compile_options.partition_count,
        compile_options.hash_columns,
        source_directory,
        source_budget,
    )
    return _submission(value, options, resources), value["source_bytes"]


def _submission(value: dict[str, Any], options: QueryExecutionOptions, resources: ResourceDemand) -> RayQuerySpec:
    return RayQuerySpec(
        graph=_native_graph(value["graph"]),
        options=options,
        resources=resources,
        connection_snapshot=value["connection_snapshot"],
        source_snapshots=tuple(FragmentSourceSnapshot(key, item) for key, item in value["source_snapshots"].items()),
        result_names=value["result_names"],
    )


def check_plan_capabilities(spec: RayQuerySpec, capabilities: PlanCapabilities) -> None:
    if not isinstance(spec, RayQuerySpec) or not isinstance(capabilities, PlanCapabilities):
        raise ValueError("expected RayQuerySpec and PlanCapabilities")
    if capabilities.protocol_version != spec.protocol_version:
        raise ValueError("worker submission protocol does not match")
    if capabilities.engine_identity != spec.graph.engine_identity:
        raise ValueError("worker engine identity does not match")
    if capabilities.type_profile != spec.type_profile or capabilities.connection_profile != spec.connection_profile:
        raise ValueError("worker type or connection profile does not match")
    if any(edge.distribution not in capabilities.distributions for edge in spec.graph.exchanges):
        raise ValueError("worker lacks an exchange distribution capability")
    if any(
        (source.capability, source.codec) not in capabilities.scans
        for fragment in spec.graph.fragments
        for source in fragment.sources + fragment.source_dependencies
    ):
        raise ValueError("worker lacks a source capability or split codec")


def prepare_worker_plan(connection: Any, spec: RayQuerySpec) -> None:
    """Validate native state on a dedicated worker connection, restoring its session.

    Call before creating each task/attempt. A failure retires this connection.
    This proves plan loadability and source preconditions, not that resources
    were reserved or an FTE exchange store has been provisioned.
    """
    from vane._native import execution_plan

    check_plan_capabilities(spec, native_plan_capabilities(connection))
    sources = {source.fragment_id: source.payload for source in spec.source_snapshots}
    for fragment in spec.graph.fragments:
        decoded = execution_plan.inspect_submitted_fragment(
            connection,
            fragment.native_plan,
            spec.connection_snapshot,
            sources[fragment.fragment_id],
            spec.requires_replay,
        )
        if _native_fragment(decoded) != fragment:
            raise ValueError("native fragment disagrees with submission graph")
        if fragment.fragment_id == spec.graph.result.fragment_id and tuple(decoded["names"]) != spec.result_names:
            raise ValueError("native result names disagree with submission")
    for exchange in spec.graph.exchanges:
        if exchange.distribution is Distribution.HASH:
            producer = next(
                fragment for fragment in spec.graph.fragments if fragment.fragment_id == exchange.producer_fragment_id
            )
            schema = next(port.schema for port in producer.outputs if port.port_id == exchange.producer_port)
            assert exchange.partitioning is not None
            execution_plan.validate_hash(connection, schema, exchange.partitioning)
