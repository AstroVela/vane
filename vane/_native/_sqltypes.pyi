# SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import typing

import vane

class DuckDBPyType:
    def __eq__(self, other: object) -> bool: ...
    def __getattr__(self, name: str) -> DuckDBPyType: ...
    def __getitem__(self, name: str) -> DuckDBPyType: ...
    def __hash__(self) -> int: ...
    @typing.overload
    def __init__(self, type_str: str, connection: vane.DuckDBPyConnection | None, /) -> None: ...
    @typing.overload
    def __init__(self, obj: object, /) -> None: ...
    @property
    def children(self) -> list[tuple[str, DuckDBPyType | int | list[str] | tuple[int, ...]]]: ...
    @property
    def id(self) -> str: ...

BIGINT: DuckDBPyType
BIT: DuckDBPyType
BLOB: DuckDBPyType
BOOLEAN: DuckDBPyType
DATE: DuckDBPyType
DOUBLE: DuckDBPyType
FLOAT: DuckDBPyType
HUGEINT: DuckDBPyType
INTEGER: DuckDBPyType
INTERVAL: DuckDBPyType
SMALLINT: DuckDBPyType
SQLNULL: DuckDBPyType
TIME: DuckDBPyType
TIME_NS: DuckDBPyType
TIME_TZ: DuckDBPyType
TIMESTAMP: DuckDBPyType
TIMESTAMP_MS: DuckDBPyType
TIMESTAMP_NS: DuckDBPyType
TIMESTAMP_S: DuckDBPyType
TIMESTAMP_TZ: DuckDBPyType
TINYINT: DuckDBPyType
UBIGINT: DuckDBPyType
UHUGEINT: DuckDBPyType
UINTEGER: DuckDBPyType
USMALLINT: DuckDBPyType
UTINYINT: DuckDBPyType
UUID: DuckDBPyType
VARCHAR: DuckDBPyType
VARIANT: DuckDBPyType
