# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types

from typing_extensions import assert_type

import vane
import vane.sqltypes as public_sqltypes
import vane.udf as public_udf
from vane import _native
from vane._native import _func, _sqltypes, ray_cxx
from vane._ray_cxx import require_ray_cxx_attr

statement_members: tuple[_native.StatementType, ...] = (
    _native.StatementType.INVALID,
    _native.StatementType.SELECT,
    _native.StatementType.INSERT,
    _native.StatementType.UPDATE,
    _native.StatementType.CREATE,
    _native.StatementType.DELETE,
    _native.StatementType.PREPARE,
    _native.StatementType.EXECUTE,
    _native.StatementType.ALTER,
    _native.StatementType.TRANSACTION,
    _native.StatementType.COPY,
    _native.StatementType.ANALYZE,
    _native.StatementType.VARIABLE_SET,
    _native.StatementType.CREATE_FUNC,
    _native.StatementType.EXPLAIN,
    _native.StatementType.DROP,
    _native.StatementType.EXPORT,
    _native.StatementType.PRAGMA,
    _native.StatementType.VACUUM,
    _native.StatementType.CALL,
    _native.StatementType.SET,
    _native.StatementType.LOAD,
    _native.StatementType.RELATION,
    _native.StatementType.EXTENSION,
    _native.StatementType.LOGICAL_PLAN,
    _native.StatementType.ATTACH,
    _native.StatementType.DETACH,
    _native.StatementType.MULTI,
    _native.StatementType.COPY_DATABASE,
    _native.StatementType.MERGE_INTO,
)
expected_result_members: tuple[_native.ExpectedResultType, ...] = (
    _native.ExpectedResultType.QUERY_RESULT,
    _native.ExpectedResultType.CHANGED_ROWS,
    _native.ExpectedResultType.NOTHING,
)
explain_members: tuple[_native.ExplainType, ...] = (
    _native.ExplainType.STANDARD,
    _native.ExplainType.ANALYZE,
)
line_terminator_members: tuple[_native.CSVLineTerminator, ...] = (
    _native.CSVLineTerminator.LINE_FEED,
    _native.CSVLineTerminator.CARRIAGE_RETURN_LINE_FEED,
)
exception_handling_members: tuple[_native.PythonExceptionHandling, ...] = (
    _native.PythonExceptionHandling.DEFAULT,
    _native.PythonExceptionHandling.RETURN_NULL,
)
render_mode_members: tuple[_native.RenderMode, ...] = (
    _native.RenderMode.ROWS,
    _native.RenderMode.COLUMNS,
)
token_members: tuple[_native.token_type, ...] = (
    _native.token_type.identifier,
    _native.token_type.numeric_const,
    _native.token_type.string_const,
    _native.token_type.operator,
    _native.token_type.keyword,
    _native.token_type.comment,
)
udf_type_members: tuple[_func.PythonUDFType, ...] = (
    _func.PythonUDFType.NATIVE,
    _func.PythonUDFType.ARROW,
)
null_handling_members: tuple[_func.FunctionNullHandling, ...] = (
    _func.FunctionNullHandling.DEFAULT,
    _func.FunctionNullHandling.SPECIAL,
)

assert_type(public_udf.NATIVE, public_udf.PythonUDFType)
assert_type(public_sqltypes.INTEGER, public_sqltypes.DuckDBPyType)
assert_type(vane.runners.get_or_create_runner(), vane.runners.runner.Runner)
assert_type(vane.sqltypes.FLOAT, public_sqltypes.DuckDBPyType)
assert_type(_func.NATIVE, _func.PythonUDFType)
assert_type(_sqltypes.INTEGER, _sqltypes.DuckDBPyType)
assert_type(_sqltypes.DuckDBPyType("INTEGER", None), _sqltypes.DuckDBPyType)

private_type: _sqltypes.DuckDBPyType = public_sqltypes.INTEGER
public_type: public_sqltypes.DuckDBPyType = private_type
assert_type(public_type, public_sqltypes.DuckDBPyType)

connection = vane.connect()
assert_type(connection.sql("SELECT 1").repartition(4, "1"), vane.DuckDBPyRelation)
assert_type(connection.sql("SELECT 1").repartition("1", num_partitions=4), vane.DuckDBPyRelation)
assert_type(
    connection.description,
    list[tuple[str, public_sqltypes.DuckDBPyType, None, None, None, None, None]] | None,
)
assert_type(
    vane.tensor_type(public_sqltypes.FLOAT, [2, 3]).children,
    list[tuple[str, public_sqltypes.DuckDBPyType | int | list[str] | tuple[int, ...]]],
)
file_value = vane.File("memory://typing", content_type="text/plain", position=0, size=1, checksum="sha256:a")
assert_type(file_value, vane.File)
assert_type(_native.File("memory://typing"), _native.File)
assert_type(file_value.url, str)
assert_type(file_value.content_type, str | None)
assert_type(file_value.position, int | None)
assert_type(file_value.size, int | None)
assert_type(file_value.checksum, str | None)
assert_type(vane.MediaType.image(), vane.MediaType)
assert_type(vane.ImageFile("memory://image"), vane.ImageFile)
assert_type(vane.AudioFile("memory://audio"), vane.AudioFile)
assert_type(vane.VideoFile("memory://video"), vane.VideoFile)
assert_type(vane.file_type(), public_sqltypes.DuckDBPyType)
assert_type(vane.file_type(vane.MediaType.image()), public_sqltypes.DuckDBPyType)
assert_type(_native.file_type(), _sqltypes.DuckDBPyType)
assert_type(_native.file_type(_native.MediaType.video()), _sqltypes.DuckDBPyType)
assert_type(vane.file_type().is_file(), bool)
image_value = vane.Image(b"\x00", 1, 1, "L")
assert_type(image_value, vane.Image)
assert_type(image_value.data, bytes)
assert_type(image_value.width, int)
assert_type(image_value.height, int)
assert_type(image_value.channels, int)
assert_type(image_value.mode, str)
assert_type(image_value.dtype, str)
assert_type(vane.image_type(), public_sqltypes.DuckDBPyType)
assert_type(vane.image_type().is_image(), bool)
assert_type(vane.file("memory://typing"), vane.Expression)
assert_type(vane.image_file("memory://image"), vane.Expression)
assert_type(vane.audio_file(file_value), vane.Expression)
assert_type(vane.audio_resample(vane.col("audio"), 16000), vane.Expression)
assert_type(vane.video_file(vane.col("file")), vane.Expression)
assert_type(vane.col("url").as_file(), vane.Expression)
assert_type(vane.col("url").as_file(vane.MediaType.image()), vane.Expression)
assert_type(vane.col("file").url, vane.Expression)

assert_type(_native._func, types.ModuleType)
assert_type(_native._sqltypes, types.ModuleType)
assert_type(_native.ray_cxx, types.ModuleType)
assert_type(vane.ray_cxx.PyLogicalPlan, type[ray_cxx.PyLogicalPlan])
assert_type(require_ray_cxx_attr("PyLogicalPlan"), type[ray_cxx.PyLogicalPlan])
assert_type(require_ray_cxx_attr("RayTaskResult"), type[ray_cxx.RayTaskResult])
cleanup_flight_shuffle = require_ray_cxx_attr("cleanup_flight_shuffle_for_query")
assert_type(cleanup_flight_shuffle("typing-query"), dict[str, int | str])
assert_type(ray_cxx.merge_scan_split_batches([b"batch"]), bytes)
assert_type(ray_cxx.split_scan_split_batch(b"batch"), list[tuple[str, bytes, int | None]])
split_scan_split_batch = require_ray_cxx_attr("split_scan_split_batch")
assert_type(split_scan_split_batch(b"batch"), list[tuple[str, bytes, int | None]])
assert_type(
    ray_cxx.split_exchange_source_task_by_partition(b"descriptor"),
    list[tuple[int, bytes, int, int, bool]],
)
assert_type(ray_cxx.RayTaskResult.no_output(), ray_cxx.RayTaskResult)
assert_type(ray_cxx.FteSplitQueue(), ray_cxx.FteSplitQueue)
