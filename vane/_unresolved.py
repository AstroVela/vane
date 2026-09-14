# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Unbound requests shared by the Ray client and its driver session.

These objects describe API operations; they contain no client catalog bindings,
DuckDB connections, or executable native Relation objects. SQL is an unresolved
SQL node, so parsing options and extension syntax belong to the driver session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Only operations which construct native Relations belong in these sets.
# Terminal methods are admitted separately, after binding on the driver.
SOURCES = frozenset(
    "table view values table_function from_arrow from_df read_csv from_csv_auto "
    "read_json read_parquet from_parquet from_datasource _read_video_frames".split()
)
TRANSFORMS = frozenset(
    "filter project select select_types select_dtypes set_alias order limit distinct aggregate "
    "union except_ intersect join cross explode repartition local_exchange map map_batches flat_map "
    "any_value arg_max arg_min avg mean bit_and bit_or bit_xor bool_and bool_or count favg fsum "
    "geomean product histogram max min string_agg sum unique median mode quantile_cont quantile_disc "
    "quantile stddev_pop stddev_samp stddev std var_pop var_samp variance var row_number rank "
    "dense_rank rank_dense percent_rank cume_dist first_value n_tile lag last_value lead nth_value "
    "describe _mark_datasink".split()
)
CONNECTION_OPERATIONS = frozenset(
    "register unregister create_function remove_function _create_vane_function _create_vane_batch_function "
    "_remove_vane_function register_filesystem unregister_filesystem filesystem_is_registered list_filesystems "
    "install_extension load_extension get_table_names enable_profiling disable_profiling "
    "get_profiling_information query_progress type dtype sqltype array_type list_type tensor_type union_type "
    "string_type enum_type decimal_type struct_type row_type map_type".split()
)
RELATION_OPERATIONS = frozenset(
    "create_view to_view explain sql_query _arrow_schema _validate_datasink_transaction "
    "_validate_datasink_retry_input alias".split()
)
WRITES = {
    "write_parquet": "_unresolved_parquet_relation",
    "to_parquet": "_unresolved_parquet_relation",
    "write_csv": "_unresolved_csv_relation",
    "to_csv": "_unresolved_csv_relation",
    "write_file": "_unresolved_file_relation",
    "to_file": "_unresolved_file_relation",
    "insert_into": "_unresolved_insert_relation",
    "create": "_unresolved_create_relation",
    "to_table": "_unresolved_create_relation",
    "update": "_unresolved_update_relation",
    "delete": "_unresolved_delete_relation",
    "merge_into": "_unresolved_merge_relation",
}


@dataclass(frozen=True)
class UnresolvedInput:
    index: int


@dataclass(frozen=True)
class DriverStatement:
    query: str
    type: Any
    named_parameters: frozenset[str]
    expected_result_type: tuple[Any, ...]


@dataclass(frozen=True)
class DriverPlanReference:
    """An owned driver plan identity; the bound plan never visits the client."""

    session: str
    config: tuple[tuple[str, str], ...]
    query: str

    def session_id(self) -> str:
        return self.session

    def session_config(self) -> dict[str, str]:
        return dict(self.config)

    def idx(self) -> str:
        return self.query

    def copy_operation_id(self) -> str:
        return self.query


@dataclass(frozen=True)
class UnresolvedExpression:
    payload: bytes
    order: int
    null_order: int


@dataclass(frozen=True)
class UnresolvedType:
    payload: bytes


@dataclass(frozen=True)
class UnresolvedPlan:
    """A SQL leaf, connection source, or Relation operation and its inputs."""

    kind: str
    operation: str
    arguments: tuple[Any, ...] = ()
    keywords: tuple[tuple[str, Any], ...] = ()
    inputs: tuple[UnresolvedPlan, ...] = ()


@dataclass(frozen=True)
class UnresolvedRequest:
    session: str
    config: tuple[tuple[str, str], ...]
    query: str
    plan: UnresolvedPlan
    bootstrap: tuple[str, bool, tuple[tuple[str, Any], ...]]
    parent_session: str | None = None

    def session_id(self) -> str:
        return self.session

    def session_config(self) -> dict[str, str]:
        return dict(self.config)

    def idx(self) -> str:
        return self.query


def encode_argument(value: Any) -> Any:
    """Capture native expressions as parsed trees without resolving names."""
    from vane import _native
    from vane.sqltypes import DuckDBPyType

    if isinstance(value, _native.Expression):
        payload, order, null_order = _native.ray_cxx._serialize_unresolved_expression(value)
        return UnresolvedExpression(payload, order, null_order)
    if isinstance(value, DuckDBPyType):
        return UnresolvedType(_native.ray_cxx._serialize_unresolved_type(value))
    if isinstance(value, tuple):
        return tuple(encode_argument(item) for item in value)
    if isinstance(value, list):
        return [encode_argument(item) for item in value]
    if isinstance(value, dict):
        return {key: encode_argument(item) for key, item in value.items()}
    return value


def decode_argument(value: Any) -> Any:
    from vane import _native

    if isinstance(value, UnresolvedExpression):
        return _native.ray_cxx._deserialize_unresolved_expression(value.payload, value.order, value.null_order)
    if isinstance(value, UnresolvedType):
        return _native.ray_cxx._deserialize_unresolved_type(value.payload)
    if isinstance(value, tuple):
        return tuple(decode_argument(item) for item in value)
    if isinstance(value, list):
        return [decode_argument(item) for item in value]
    if isinstance(value, dict):
        return {key: decode_argument(item) for key, item in value.items()}
    return value


def _reduce_native_argument(value: Any) -> tuple[Any, tuple[Any, ...]]:
    return decode_argument, (encode_argument(value),)


# A function closure or DataSource may also contain a native expression/type.
# Use the same unresolved encoding when cloudpickle reaches one indirectly.
def register_argument_reducers() -> None:
    import copyreg

    from vane import _native
    from vane.sqltypes import DuckDBPyType

    copyreg.pickle(_native.Expression, _reduce_native_argument)
    copyreg.pickle(DuckDBPyType, _reduce_native_argument)
