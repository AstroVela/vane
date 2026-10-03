# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

from . import DuckDBPyConnection

def engine_identity() -> str: ...
def compile(
    connection: DuckDBPyConnection,
    sql: str,
    query_id: str,
    partition_count: int,
    hash_columns: Sequence[int],
) -> dict[str, Any]: ...
def inspect_fragment(connection: DuckDBPyConnection, payload: bytes) -> dict[str, Any]: ...
def compile_submission(
    connection: DuckDBPyConnection,
    sql: str,
    query_id: str,
    partition_count: int,
    hash_columns: Sequence[int],
    require_replay: bool,
) -> dict[str, Any]: ...
def compiler_capabilities(connection: DuckDBPyConnection) -> dict[str, Any]: ...
def inspect_submitted_fragment(
    connection: DuckDBPyConnection,
    payload: bytes,
    connection_snapshot: bytes,
    source_snapshot: bytes,
    require_replay: bool,
) -> dict[str, Any]: ...
def validate_hash(connection: DuckDBPyConnection, schema: bytes, partitioning: bytes) -> None: ...
def _execute_fragment_for_test(
    connection: DuckDBPyConnection,
    payload: bytes,
    inputs: dict[str, list[tuple[Any, ...]]],
    source_assignments: dict[str, list[str]],
) -> list[tuple[Any, ...]]: ...
def _hash_rows_for_test(
    connection: DuckDBPyConnection,
    schema: bytes,
    partitioning: bytes,
    rows: list[tuple[Any, ...]],
    partition_count: int,
) -> list[int]: ...
