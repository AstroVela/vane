# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Python expression and discovery facade for the SQL FILE contract."""

from __future__ import annotations

import vane
from vane._expressions import as_expression


def _file_function(name: str, *arguments: object) -> vane.Expression:
    return vane.FunctionExpression(name, *(as_expression(argument) for argument in arguments))


def file(
    url: str | vane.Expression,
    content_type: str | vane.Expression | None = None,
    position: int | vane.Expression | None = None,
    size: int | vane.Expression | None = None,
    checksum: str | vane.Expression | None = None,
) -> vane.Expression:
    """Construct a FILE expression without accessing the referenced resource."""
    return vane.FunctionExpression(
        "file",
        as_expression(url),
        as_expression(content_type),
        as_expression(position),
        as_expression(size),
        as_expression(checksum),
    )


def to_file(path: str | vane.Expression) -> vane.Expression:
    """Convert a path to a FILE expression when the expression is executed."""
    return _file_function("to_file", path)


def try_to_file(path: str | vane.Expression) -> vane.Expression:
    """Convert a path to FILE, returning NULL for recoverable access failures."""
    return _file_function("try_to_file", path)


def file_enrich(value: vane.File | vane.Expression, fields: list[str] | vane.Expression) -> vane.Expression:
    """Enrich selected FILE fields when the expression is executed."""
    return _file_function("file_enrich", value, fields)


def file_path(value: vane.File | vane.Expression) -> vane.Expression:
    return _file_function("file_path", value)


def file_size(value: vane.File | vane.Expression) -> vane.Expression:
    return _file_function("file_size", value)


def file_exists(value: vane.File | vane.Expression) -> vane.Expression:
    return _file_function("file_exists", value)


def file_stat(value: vane.File | vane.Expression) -> vane.Expression:
    return _file_function("file_stat", value)


def file_mime_type(
    value: vane.File | vane.Expression,
    detect: str | vane.Expression = "metadata",
) -> vane.Expression:
    if isinstance(detect, str) and detect == "metadata":
        return _file_function("file_mime_type", value)
    return _file_function("file_mime_type", value, detect)


def guess_mime_type(value: bytes | vane.Expression) -> vane.Expression:
    return _file_function("guess_mime_type", value)


def file_same_location(
    left: vane.File | vane.Expression,
    right: vane.File | vane.Expression,
) -> vane.Expression:
    return _file_function("file_same_location", left, right)


def file_same_content(
    left: vane.File | vane.Expression,
    right: vane.File | vane.Expression,
) -> vane.Expression:
    return _file_function("file_same_content", left, right)


def file_locator_id(value: vane.File | vane.Expression) -> vane.Expression:
    return _file_function("file_locator_id", value)


def file_content_id(value: vane.File | vane.Expression) -> vane.Expression:
    return _file_function("file_content_id", value)


def list_files(
    path: str,
    recursive: bool = False,
    *,
    connection: vane.DuckDBPyConnection | None = None,
) -> vane.DuckDBPyRelation:
    """Return deterministic metadata rows from the SQL ``list_files`` function."""
    return vane.table_function("list_files", [path, recursive], connection=connection)


def from_files(
    path: str | list[str],
    *,
    connection: vane.DuckDBPyConnection | None = None,
) -> vane.DuckDBPyRelation:
    """Return a one-column relation of canonical FILE values."""
    parameter: object = path
    if isinstance(path, list):
        parameter = vane.Value(path, vane.list_type(vane.sqltypes.VARCHAR))
    parameters = [parameter]
    return vane.table_function("list_files", parameters, connection=connection).select(vane.ColumnExpression("file"))


__all__ = [
    "file",
    "file_content_id",
    "file_enrich",
    "file_exists",
    "file_locator_id",
    "file_mime_type",
    "file_path",
    "file_same_content",
    "file_same_location",
    "file_size",
    "file_stat",
    "from_files",
    "guess_mime_type",
    "list_files",
    "to_file",
    "try_to_file",
]
