# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Internal Ray fragment compiler; independent of runners and execution modes.

This module does not submit queries. ``hash_columns`` requests physical output
partitioning by result-column positions; it does not implement SQL hash rules.
The native compiler binds the keys and owns their serialization and evaluation.
"""

from dataclasses import dataclass
from typing import Any

from vane.execution.plan import (
    Distribution,
    ExchangeSpec,
    FragmentGraph,
    FragmentSpec,
    PortSpec,
    ResultSpec,
    ScanSplitSpec,
    SourceSpec,
)


@dataclass(frozen=True)
class FragmentCompileOptions:
    partition_count: int = 1
    hash_columns: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.partition_count) is not int or not 1 <= self.partition_count <= 2**31 - 1:
            raise ValueError("partition_count must be an integer between 1 and 2147483647")
        if not isinstance(self.hash_columns, (tuple, list)):
            raise ValueError("hash_columns must be a sequence of result-column positions")
        columns = tuple(self.hash_columns)
        if any(type(column) is not int or not 0 <= column <= 2**31 - 1 for column in columns):
            raise ValueError("hash_columns must contain non-negative integer positions")
        if len(set(columns)) != len(columns):
            raise ValueError("hash_columns must not contain duplicates")
        object.__setattr__(self, "hash_columns", columns)


def _native_fragment(value: dict[str, Any]) -> FragmentSpec:
    return FragmentSpec(
        fragment_id=value["fragment_id"],
        native_plan=value["native_plan"],
        partition_count=value["partition_count"],
        inputs=tuple(PortSpec(**port) for port in value["inputs"]),
        outputs=tuple(PortSpec(**port) for port in value["outputs"]),
        sources=tuple(
            SourceSpec(
                source_id=source["source_id"],
                function_name=source["function_name"],
                capability=source["capability"],
                codec=source["codec"],
                requires_snapshot=source["requires_snapshot"],
                splits=tuple(ScanSplitSpec(**split) for split in source["splits"]),
            )
            for source in value["sources"]
        ),
    )


def compile_fragment_graph(
    connection: Any,
    sql: str,
    *,
    query_id: str,
    options: FragmentCompileOptions = FragmentCompileOptions(),
) -> FragmentGraph:
    """Bind and compile one supported read-only SELECT without starting tasks.

    The connection is locked while its native binding/optimizer settings are
    read. Use prepare_ray_query for a submission with connection/source
    snapshots; this lower-level compiler does not validate file versions.
    """
    from vane._native import execution_plan

    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("sql must be a non-empty string")
    if not isinstance(query_id, str) or not query_id.strip():
        raise ValueError("query_id must be a non-empty string")
    if not isinstance(options, FragmentCompileOptions):
        raise ValueError("options must be FragmentCompileOptions")
    value = execution_plan.compile(connection, sql, query_id, options.partition_count, options.hash_columns)
    return _native_graph(value)


def _native_graph(value: dict[str, Any]) -> FragmentGraph:
    return FragmentGraph(
        query_id=value["query_id"],
        engine_identity=value["engine_identity"],
        fragments=tuple(_native_fragment(fragment) for fragment in value["fragments"]),
        exchanges=tuple(ExchangeSpec(**exchange) for exchange in value["exchanges"]),
        result=ResultSpec(**value["result"]),
    )


def validate_native_graph(connection: Any, graph: FragmentGraph) -> None:
    """Decode native plans and reject disagreement with the Python graph view.

    This checks the compiler/loader contract. It does not admit a query, resolve
    credentials or prove that a file source is immutable enough for FTE replay.
    """
    from vane._native import execution_plan

    if graph.engine_identity != execution_plan.engine_identity():
        raise ValueError("fragment graph engine identity does not match this runtime")
    fragments = {fragment.fragment_id: fragment for fragment in graph.fragments}
    for fragment in graph.fragments:
        decoded = _native_fragment(execution_plan.inspect_fragment(connection, fragment.native_plan))
        if decoded != fragment:
            raise ValueError(f"native fragment {fragment.fragment_id} disagrees with its graph description")
    for exchange in graph.exchanges:
        if exchange.distribution is Distribution.HASH:
            assert exchange.partitioning is not None  # Guaranteed by ExchangeSpec.
            source = fragments[exchange.producer_fragment_id]
            port = next(port for port in source.outputs if port.port_id == exchange.producer_port)
            execution_plan.validate_hash(connection, port.schema, exchange.partitioning)
