# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Vane's Python DataSource API.

Inspired by Vane's DataSource/DataSourceTask pattern.
Enables streaming data ingestion with generator-based backpressure.

Usage::

    from vane.datasource import DataSource, DataSourceTask


    class MySource(DataSource):
        @property
        def schema(self): ...
        def get_tasks(self): ...


    class MyTask(DataSourceTask):
        def execute(self):
            yield pa.record_batch(...)


    rel = con.read_datasource(MySource(...))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyarrow as pa

    from vane import DuckDBPyConnection, DuckDBPyRelation
    from vane.sqltypes import DuckDBPyType


class DataSourceTask(ABC):
    """A serializable unit of work that produces data via a generator.

    Each task should represent an independently processable chunk of data
    (e.g., one file, one partition). Tasks are distributed across pipeline
    threads or Ray workers for parallel execution.
    """

    @abstractmethod
    def execute(self) -> Iterator[pa.RecordBatch]:
        """Execute this task, yielding record batches.

        This is a generator method. Each yield produces a small batch
        (recommended ~10MB) to enable streaming and backpressure.

        Yields:
            pa.RecordBatch: A batch of rows conforming to the DataSource schema.
        """
        ...


class DataSource(ABC):
    """Base class for streaming data sources.

    Subclasses define the schema and split the work into tasks.
    The engine handles scheduling, parallelism, and backpressure.
    """

    @property
    @abstractmethod
    def schema(self) -> dict[str, object]:
        """Column name → Vane SQL type string or structured type entry.

        This defines the output schema for all tasks produced by this source.
        """
        ...

    @abstractmethod
    def get_tasks(self) -> Iterator[DataSourceTask]:
        """Split this source into independently executable tasks.

        Each task should represent one logical unit of work (e.g., one file).
        Tasks can be executed in parallel across pipeline threads or workers.

        Yields:
            DataSourceTask: A task that can be pickled and sent to a worker.
        """
        ...


def _schema_to_arrow(schema: Mapping[str, object]) -> pa.Schema:
    """Convert DataSource schema dict to a PyArrow schema.

    Uses Vane's type parser to support all supported SQL types, including complex
    ones like ``INTEGER[]``, ``STRUCT(a INT, b VARCHAR)``, ``JSON[]``, etc.
    """
    import pyarrow as pa

    fields = []
    for name, type_spec in schema.items():
        arrow_type = _type_spec_to_arrow(type_spec)
        fields.append(pa.field(name, arrow_type))
    return pa.schema(fields)


def _type_spec_to_arrow(type_spec: object) -> pa.DataType:
    """Convert a DataSource schema value to a PyArrow DataType."""
    import pyarrow as pa

    if isinstance(type_spec, pa.DataType):
        return type_spec
    if isinstance(type_spec, dict):
        return _schema_entry_to_arrow(type_spec)
    if not isinstance(type_spec, str):
        raise TypeError(f"DataSource schema values must be SQL type strings or structured entries, got {type_spec!r}")
    return _duckdb_type_to_arrow(type_spec)


def _schema_entry_to_arrow(entry: Mapping[str, Any]) -> pa.DataType:
    import pyarrow as pa

    kind = str(entry.get("kind") or "").strip().lower()
    if kind == "tensor":
        dtype = _duckdb_type_to_arrow(str(entry.get("dtype") or ""))
        shape = tuple(int(dim) for dim in entry.get("shape") or ())
        if not shape:
            raise ValueError("DataSource tensor schema entry requires non-empty shape")
        if any(dim <= 0 for dim in shape):
            raise ValueError(f"DataSource tensor shape dimensions must be positive: {shape!r}")
        return pa.fixed_shape_tensor(dtype, shape)
    if not kind or kind == "duckdb_type":
        return _duckdb_type_to_arrow(str(entry.get("type") or ""))
    raise ValueError(f"Unsupported DataSource schema entry kind: {kind!r}")


def _duckdb_type_to_arrow(type_str: str) -> pa.DataType:
    """Convert a DuckDB type string to a PyArrow DataType.

    Leverages ``vane.type()`` to parse arbitrary type strings (including
    nested LIST, STRUCT, MAP) and recursively converts to PyArrow types.
    """
    import vane

    dt = vane.type(type_str)
    return _convert_duckdb_pytype(dt)


def _convert_duckdb_pytype(dt: DuckDBPyType) -> pa.DataType:
    """Recursively convert a DuckDBPyType to a PyArrow DataType."""
    import pyarrow as pa

    type_id = str(dt.id)

    # Basic scalar types
    _BASIC_MAP = {
        "varchar": pa.utf8,
        "integer": pa.int32,
        "bigint": pa.int64,
        "smallint": pa.int16,
        "tinyint": pa.int8,
        "uinteger": lambda: pa.uint32(),
        "ubigint": lambda: pa.uint64(),
        "usmallint": lambda: pa.uint16(),
        "utinyint": lambda: pa.uint8(),
        "float": pa.float32,
        "double": pa.float64,
        "boolean": pa.bool_,
        "blob": pa.binary,
        "timestamp": lambda: pa.timestamp("us"),
        "timestamp_s": lambda: pa.timestamp("s"),
        "timestamp_ms": lambda: pa.timestamp("ms"),
        "timestamp_ns": lambda: pa.timestamp("ns"),
        "date": pa.date32,
        "time": lambda: pa.time64("us"),
        "interval": lambda: pa.duration("us"),
        "json": pa.utf8,  # JSON stored as UTF-8 string in Arrow
        "hugeint": lambda: pa.decimal128(38, 0),
    }

    factory = _BASIC_MAP.get(type_id)
    if factory is not None:
        return factory()

    def child_type(value: object, *, field: str) -> DuckDBPyType:
        from vane.sqltypes import DuckDBPyType as RuntimeDuckDBPyType

        if not isinstance(value, RuntimeDuckDBPyType):
            raise TypeError(f"DuckDB type {dt} has invalid {field}: {value!r}")
        return value

    # Complex / nested types
    if type_id == "tensor":
        children = dict(dt.children)
        dtype = _convert_duckdb_pytype(child_type(children["dtype"], field="tensor dtype"))
        raw_shape = children["shape"]
        if not isinstance(raw_shape, (list, tuple)):
            raise TypeError(f"DuckDB type {dt} has invalid tensor shape: {raw_shape!r}")
        return pa.fixed_shape_tensor(dtype, tuple(int(dim) for dim in raw_shape))
    if type_id == "list":
        list_child_type = _convert_duckdb_pytype(child_type(dt.children[0][1], field="list child"))
        return pa.list_(list_child_type)
    if type_id == "array":
        children = dict(dt.children)
        array_child_type = _convert_duckdb_pytype(child_type(children["child"], field="array child"))
        raw_size = children["size"]
        if not isinstance(raw_size, int):
            raise TypeError(f"DuckDB type {dt} has invalid array size: {raw_size!r}")
        return pa.list_(array_child_type, list_size=raw_size)
    if type_id == "struct":
        fields = [
            (name, _convert_duckdb_pytype(child_type(child_dt, field=f"struct field {name!r}")))
            for name, child_dt in dt.children
        ]
        return pa.struct(fields)
    if type_id == "map":
        key_type = _convert_duckdb_pytype(child_type(dt.children[0][1], field="map key"))
        val_type = _convert_duckdb_pytype(child_type(dt.children[1][1], field="map value"))
        return pa.map_(key_type, val_type)
    raise ValueError(
        f"Unsupported DuckDB type '{dt}' (id={type_id}) for DataSource schema. Consider filing a feature request."
    )


_SENTINEL = object()


def read_datasource(
    source: DataSource,
    *,
    con: DuckDBPyConnection | None = None,
    limit: int = 0,
) -> DuckDBPyRelation:
    """Create a DuckDB Relation from a DataSource.

    Uses the ``datasource_scan`` C++ TableFunction for pipeline-level
    parallelism.  Each task becomes an independent ArrowArrayStream —
    DuckDB pipeline threads scan them in parallel (like parquet_scan).

    Args:
        source: A DataSource instance.
        con: DuckDB connection (required).
        limit: Optional row limit (0 = no limit).

    Returns:
        vane.DuckDBPyRelation: A scannable relation.
    """
    if con is None:
        raise ValueError("con is required for read_datasource()")

    to_udf_relation = getattr(source, "to_udf_relation", None)
    if callable(to_udf_relation):
        rel = to_udf_relation(con)
    else:
        rel = con.from_datasource(source)

    if limit > 0:
        rel = rel.limit(limit)

    from vane import DuckDBPyRelation as RuntimeDuckDBPyRelation

    if not isinstance(rel, RuntimeDuckDBPyRelation):
        raise TypeError(f"DataSource conversion must return DuckDBPyRelation, got {type(rel).__name__}")
    return rel
