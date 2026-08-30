# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest

import vane

MEDIA_FILE_UDF_CASES = (
    (vane.MediaType.image(), "IMAGEFILE", vane.ImageFile, "image_file"),
    (vane.MediaType.audio(), "AUDIOFILE", vane.AudioFile, "audio_file"),
    (vane.MediaType.video(), "VIDEOFILE", vane.VideoFile, "video_file"),
)


def _file_arrow_type():
    import pyarrow as pa

    return pa.struct(
        [
            pa.field("url", pa.string()),
            pa.field("content_type", pa.string()),
            pa.field("position", pa.int64()),
            pa.field("size", pa.int64()),
            pa.field("checksum", pa.string()),
        ]
    )


def _file_record(
    url="memory://udf",
    content_type="application/octet-stream",
    position=0,
    size=3,
    checksum="sha256:abc",
):
    return {
        "url": url,
        "content_type": content_type,
        "position": position,
        "size": size,
        "checksum": checksum,
    }


def test_scalar_file_udf_materializes_and_returns_file_values():
    @vane.func(return_dtype=vane.file_type())
    def copy_file(identifier, value):
        assert identifier == 0
        assert isinstance(value, vane.File)
        return vane.File(value.url, value.content_type, value.position, value.size, value.checksum)

    connection = vane.connect()
    source = connection.sql(
        """
        SELECT *
        FROM (
            SELECT 0 AS id, file('memory://udf', 'text/plain', 1, 2, 'sha256:abc') AS value
            UNION ALL
            SELECT 1 AS id, NULL::FILE AS value
        )
        ORDER BY id
        """
    )
    result = source.select(vane.col("id"), copy_file(vane.col("id"), vane.col("value")).alias("value"))

    assert result.types[1].is_file()
    assert result.fetchall() == [
        (0, vane.File("memory://udf", "text/plain", 1, 2, "sha256:abc")),
        (1, None),
    ]


@pytest.mark.parametrize(("media_type", "type_name", "value_class", "constructor"), MEDIA_FILE_UDF_CASES)
def test_scalar_media_file_udf_preserves_exact_specialization(media_type, type_name, value_class, constructor):
    dtype = vane.file_type(media_type)

    @vane.func(return_dtype=dtype)
    def copy_media(value):
        assert type(value) is value_class
        return value_class(value.url, value.content_type, value.position, value.size, value.checksum)

    connection = vane.connect()
    source = connection.sql(
        f"""
        SELECT * FROM (
            VALUES (0, {constructor}('memory://media')), (1, NULL::{type_name})
        ) AS source(id, value)
        ORDER BY id
        """
    )
    result = source.select(vane.col("id"), copy_media(vane.col("value")).alias("value"))

    assert str(result.types[1]) == type_name
    assert result.fetchall() == [(0, value_class("memory://media")), (1, None)]


@pytest.mark.parametrize(("media_type", "type_name", "value_class", "constructor"), MEDIA_FILE_UDF_CASES)
def test_batch_media_file_udf_restores_declared_specialization(media_type, type_name, value_class, constructor):
    import pyarrow as pa

    @vane.func.batch(return_dtype=vane.file_type(media_type))
    def identity_media(values):
        assert isinstance(values, (pa.Array, pa.ChunkedArray))
        assert values.type.equals(_file_arrow_type())
        return values

    connection = vane.connect()
    source = connection.sql(
        f"""
        SELECT * FROM (
            VALUES (0, {constructor}('memory://media')), (1, NULL::{type_name})
        ) AS source(id, value)
        ORDER BY id
        """
    )
    result = source.select(vane.col("id"), identity_media(vane.col("value")).alias("value"))

    assert str(result.types[1]) == type_name
    assert result.fetchall() == [(0, value_class("memory://media")), (1, None)]


def test_empty_batch_media_file_udf_retains_declared_specialization():
    @vane.func.batch(return_dtype=vane.file_type(vane.MediaType.image()))
    def identity_image(values):
        return values

    connection = vane.connect()
    source = connection.sql("SELECT image_file('memory://empty') AS value WHERE FALSE")
    result = source.select(identity_image(vane.col("value")).alias("value"))

    assert str(result.types[0]) == "IMAGEFILE"
    assert result.fetchall() == []


def test_scalar_media_file_udf_rejects_generic_file_output():
    @vane.func(return_dtype=vane.file_type(vane.MediaType.image()))
    def invalid_media_output(_value):
        return vane.File("memory://generic")

    connection = vane.connect()
    source = connection.sql("SELECT 1 AS value")

    with pytest.raises(Exception, match=r"must be vane\.ImageFile or NULL"):
        source.select(invalid_media_output(vane.col("value"))).fetchall()


def test_scalar_media_file_udf_rejects_different_specialization_output():
    @vane.func(return_dtype=vane.file_type(vane.MediaType.video()))
    def invalid_media_output(_value):
        return vane.AudioFile("memory://audio")

    connection = vane.connect()
    source = connection.sql("SELECT 1 AS value")

    with pytest.raises(Exception, match=r"must be vane\.VideoFile or NULL"):
        source.select(invalid_media_output(vane.col("value"))).fetchall()


def test_scalar_file_udf_materializes_nested_files():
    nested_type = vane.list_type(vane.file_type())

    @vane.func(return_dtype=nested_type)
    def copy_files(values):
        assert isinstance(values[0], vane.File)
        assert values[1] is None
        return values

    connection = vane.connect()
    source = connection.sql("SELECT [file('memory://nested', NULL, NULL, NULL, NULL), NULL::FILE] AS values")
    result = source.select(copy_files(vane.col("values")).alias("values"))

    assert str(result.types[0]) == "FILE[]"
    assert result.fetchone() == ([vane.File("memory://nested"), None],)


def test_scalar_file_udf_reads_strict_file_view_on_worker(tmp_path):
    payload = b"prefix-worker-view-suffix"
    path = tmp_path / "udf-reader.bin"
    path.write_bytes(payload)
    escaped_path = str(path).replace("'", "''")

    @vane.func(return_dtype="BLOB")
    def read_view(value):
        assert isinstance(value, vane.File)
        with value.open() as reader:
            return reader.read()

    connection = vane.connect()
    source = connection.sql(f"SELECT file('{escaped_path}', NULL, 7, 11, NULL) AS value")

    assert source.select(read_view(vane.col("value"))).fetchone() == (payload[7:18],)


@pytest.mark.parametrize(
    "fallback",
    [
        _file_record(),
        ("memory://udf", "application/octet-stream", 0, 3, "sha256:abc"),
        b"memory://udf",
    ],
)
def test_scalar_file_udf_rejects_structural_output_fallbacks(fallback):
    @vane.func(return_dtype=vane.file_type())
    def invalid_output(_value):
        return fallback

    connection = vane.connect()
    source = connection.sql("SELECT 1 AS value")

    with pytest.raises(Exception, match=r"must be vane\.File or NULL"):
        source.select(invalid_output(vane.col("value"))).fetchall()


@pytest.mark.parametrize("logical_type", ["FILE", "IMAGEFILE"])
def test_file_udf_does_not_run_after_invalid_file_construction(tmp_path, logical_type):
    marker = tmp_path / "called"

    @vane.func(return_dtype="INTEGER")
    def observe(_value):
        Path(marker).write_text("called", encoding="utf-8")
        return 1

    connection = vane.connect()
    file_expression = "file(url, NULL, NULL, NULL, NULL)"
    if logical_type == "IMAGEFILE":
        file_expression = f"image_file({file_expression})"

    with pytest.raises(Exception, match=r"url cannot be NULL"):
        connection.sql(f"""
            SELECT {file_expression} AS value
            FROM (
                SELECT CASE WHEN i = 0 THEN NULL::VARCHAR ELSE 'memory://valid' END AS url
                FROM range(1) AS source(i)
            )
        """).select(observe(vane.col("value"))).fetchall()
    assert not marker.exists()


def test_batch_file_udf_receives_arrow_struct_and_restores_file_alias():
    import pyarrow as pa

    expected_type = _file_arrow_type()

    @vane.func.batch(return_dtype=vane.file_type(), batch_size=2)
    def identity(identifiers, values):
        assert len(identifiers) == len(values)
        assert isinstance(values, (pa.Array, pa.ChunkedArray))
        assert values.type.equals(expected_type)
        return values

    connection = vane.connect()
    source = connection.sql(
        """
        SELECT
            i,
            CASE
                WHEN i = 3 THEN NULL::FILE
                ELSE file('memory://' || i::VARCHAR, 'text/plain', i, 1, 'sha256:' || i::VARCHAR)
            END AS value
        FROM range(4) AS t(i)
        """
    )
    result = source.select(vane.col("i"), identity(vane.col("i"), vane.col("value")).alias("value"))

    assert result.types[1].is_file()
    assert result.fetchall() == [
        (0, vane.File("memory://0", "text/plain", 0, 1, "sha256:0")),
        (1, vane.File("memory://1", "text/plain", 1, 1, "sha256:1")),
        (2, vane.File("memory://2", "text/plain", 2, 1, "sha256:2")),
        (3, None),
    ]


def test_batch_file_udf_normalizes_each_worker_batch_before_concat():
    import pyarrow as pa

    @vane.func.batch(return_dtype=vane.file_type(), batch_size=2)
    def build_files(identifiers):
        string_type = pa.string() if identifiers[0].as_py() < 2 else pa.large_string()
        file_type = pa.struct(
            [
                pa.field("url", string_type),
                pa.field("content_type", string_type),
                pa.field("position", pa.int64()),
                pa.field("size", pa.int64()),
                pa.field("checksum", string_type),
            ]
        )
        return pa.array(
            [
                {
                    "url": f"memory://{identifier}",
                    "content_type": None,
                    "position": None,
                    "size": None,
                    "checksum": None,
                }
                for identifier in identifiers.to_pylist()
            ],
            type=file_type,
        )

    connection = vane.connect()
    result = connection.sql("SELECT i FROM range(4) AS t(i)").select(build_files(vane.col("i")).alias("value"))

    assert result.fetchall() == [(vane.File(f"memory://{identifier}"),) for identifier in range(4)]


def test_batch_file_udf_does_not_run_after_invalid_file_construction(tmp_path):
    marker = tmp_path / "called"

    @vane.func.batch(return_dtype="INTEGER")
    def observe(values):
        Path(marker).write_text("called", encoding="utf-8")
        import pyarrow as pa

        return pa.array([1] * len(values), type=pa.int32())

    connection = vane.connect()
    with pytest.raises(Exception, match=r"url cannot be NULL"):
        connection.sql("""
            SELECT file(url, NULL, NULL, NULL, NULL) AS value
            FROM (
                SELECT CASE WHEN i = 0 THEN NULL::VARCHAR ELSE 'memory://valid' END AS url
                FROM range(1) AS source(i)
            )
        """).select(observe(vane.col("value"))).fetchall()
    assert not marker.exists()


@pytest.mark.parametrize("mode", ["map_batches", "flat_map"])
def test_relation_table_file_udf_does_not_run_after_invalid_file_construction(tmp_path, mode):
    marker = tmp_path / "called"

    def observe_batch(table):
        Path(marker).write_text("called", encoding="utf-8")
        import pyarrow as pa

        return pa.table({"result": [1] * table.num_rows})

    def observe_row(_row):
        Path(marker).write_text("called", encoding="utf-8")
        return {"result": 1}

    connection = vane.connect()
    source = connection.sql("""
        SELECT file(url, NULL, NULL, NULL, NULL) AS value
        FROM (
            SELECT CASE WHEN i = 0 THEN NULL::VARCHAR ELSE 'memory://valid' END AS url
            FROM range(1) AS source(i)
        )
    """)
    if mode == "map_batches":
        result = source.map_batches(
            observe_batch,
            schema={"result": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
        )
    else:
        result = source.flat_map(
            observe_row,
            schema={"result": vane.sqltypes.INTEGER},
            execution_backend="subprocess_task",
        )

    with pytest.raises(Exception, match=r"url cannot be NULL"):
        result.fetchall()
    assert not marker.exists()


def test_flat_map_file_udf_materializes_and_returns_file_values():
    def copy_file(row):
        assert isinstance(row["value"], vane.File)
        return {"value": row["value"]}

    connection = vane.connect()
    source = connection.sql("SELECT file('memory://flat-map', NULL, NULL, NULL, NULL) AS value")
    result = source.flat_map(
        copy_file,
        schema={"value": vane.file_type()},
        execution_backend="subprocess_task",
    )

    assert result.types[0].is_file()
    assert result.fetchall() == [(vane.File("memory://flat-map"),)]


def test_flat_map_media_file_udf_preserves_specialization_in_subprocess():
    def copy_audio(row):
        assert type(row["value"]) is vane.AudioFile
        return {"value": row["value"]}

    connection = vane.connect()
    source = connection.sql("SELECT audio_file('memory://flat-map-audio') AS value")
    result = source.flat_map(
        copy_audio,
        schema={"value": vane.file_type(vane.MediaType.audio())},
        execution_backend="subprocess_task",
    )

    assert str(result.types[0]) == "AUDIOFILE"
    assert result.fetchall() == [(vane.AudioFile("memory://flat-map-audio"),)]


def test_flat_map_file_udf_rejects_structural_output_fallback():
    def invalid_output(_row):
        return {
            "value": {
                "url": "memory://udf",
                "content_type": "application/octet-stream",
                "position": 0,
                "size": 3,
                "checksum": "sha256:abc",
            }
        }

    connection = vane.connect()
    source = connection.sql("SELECT 1 AS value")
    result = source.flat_map(
        invalid_output,
        schema={"value": vane.file_type()},
        execution_backend="subprocess_task",
    )

    with pytest.raises(Exception, match=r"must be vane\.File or NULL"):
        result.fetchall()


def test_flat_map_file_udf_infers_non_file_composite_siblings():
    identifier = UUID("00112233-4455-6677-8899-aabbccddeeff")
    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    output_type = vane.type(
        "STRUCT(document FILE, id UUID, created_at TIMESTAMPTZ, empty_id UUID, empty_created_at TIMESTAMPTZ)"
    )

    def build_document(_row):
        return {
            "payload": {
                "document": vane.File("memory://composite"),
                "id": identifier,
                "created_at": created_at,
                "empty_id": None,
                "empty_created_at": None,
            }
        }

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").flat_map(
        build_document,
        schema={"payload": output_type},
        execution_backend="subprocess_task",
    )

    payload = result.fetchone()[0]
    assert dict(result.types[0].children)["document"].is_file()
    assert payload["document"] == vane.File("memory://composite")
    assert payload["id"] == identifier
    assert payload["created_at"].astimezone(timezone.utc) == created_at
    assert payload["empty_id"] is None
    assert payload["empty_created_at"] is None


def test_scalar_file_udf_preserves_non_file_composite_siblings():
    identifier = UUID("00112233-4455-6677-8899-aabbccddeeff")
    created_at = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    output_type = vane.type("STRUCT(document FILE, id UUID, created_at TIMESTAMPTZ)")

    @vane.func(return_dtype=output_type)
    def build_document(_value):
        return {
            "document": vane.File("memory://scalar-composite"),
            "id": identifier,
            "created_at": created_at,
        }

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))

    payload = result.fetchone()[0]
    assert dict(result.types[0].children)["document"].is_file()
    assert payload["document"] == vane.File("memory://scalar-composite")
    assert payload["id"] == identifier
    assert payload["created_at"].astimezone(timezone.utc) == created_at


def test_native_file_udfs_preserve_duckdb_sibling_coercions():
    identifier = UUID("00112233-4455-6677-8899-aabbccddeeff")
    output_type = vane.type("STRUCT(document FILE, id BIGINT, identifier UUID)")

    @vane.func(return_dtype=output_type)
    def build_document(_value):
        return {
            "document": vane.File("memory://coercion"),
            "id": "42",
            "identifier": str(identifier),
        }

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))

    assert result.fetchone() == ({"document": vane.File("memory://coercion"), "id": 42, "identifier": identifier},)

    def build_row(_row):
        return {
            "payload": {
                "document": vane.File("memory://flat-map-coercion"),
                "id": "43",
                "identifier": str(identifier),
            }
        }

    flat_map_result = connection.sql("SELECT 1 AS value").flat_map(
        build_row,
        schema={"payload": output_type},
        execution_backend="subprocess_task",
    )
    assert flat_map_result.fetchone() == (
        {"document": vane.File("memory://flat-map-coercion"), "id": 43, "identifier": identifier},
    )


def test_scalar_file_udf_accepts_positional_struct_output():
    output_type = vane.type("STRUCT(document FILE, id INTEGER)")

    @vane.func(return_dtype=output_type)
    def build_document(_value):
        return (vane.File("memory://positional-struct"), "42")

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))

    assert result.fetchone() == ({"document": vane.File("memory://positional-struct"), "id": 42},)


@pytest.mark.parametrize(
    "payload",
    [
        {"document": vane.File("memory://missing-field")},
        {"document": vane.File("memory://extra-field"), "id": 1, "extra": "discarded"},
    ],
    ids=["missing", "extra"],
)
def test_file_native_struct_outputs_require_exact_fields(payload):
    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "strict-struct",
            "method_return_type": "STRUCT(document FILE, id INTEGER)",
        }
    )

    with pytest.raises(vane.InvalidInputException, match="exactly the declared fields"):
        contract.scalar_outputs_to_array([payload])


@pytest.mark.parametrize(
    ("return_type", "nested_value"),
    [
        ("STRUCT(document FILE, meta STRUCT(id INTEGER, label VARCHAR))", {"id": 1}),
        ("STRUCT(document FILE, meta STRUCT(id INTEGER))", {"id": 1, "extra": "discarded"}),
        ("STRUCT(document FILE, meta STRUCT(id INTEGER)[])", [{"id": 1, "extra": "discarded"}]),
        (
            "STRUCT(document FILE, meta MAP(VARCHAR, STRUCT(id INTEGER)))",
            {"entry": {"id": 1, "extra": "discarded"}},
        ),
    ],
    ids=["missing", "extra", "list", "map"],
)
def test_file_native_nested_non_file_struct_outputs_require_exact_fields(return_type, nested_value):
    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "strict-nested-native-struct",
            "method_return_type": return_type,
        }
    )

    with pytest.raises(vane.InvalidInputException, match="exactly the declared fields"):
        contract.scalar_outputs_to_array(
            [
                {
                    "document": vane.File("memory://nested-struct"),
                    "meta": nested_value,
                }
            ]
        )


def test_file_native_non_file_struct_sibling_preserves_string_cast_input():
    output_type = vane.type("STRUCT(document FILE, meta STRUCT(id INTEGER))")

    @vane.func(return_dtype=output_type)
    def build_document(_value):
        return {
            "document": vane.File("memory://string-to-struct"),
            "meta": "{'id': 42}",
        }

    result = vane.connect().sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))

    assert result.project("payload.meta.id").fetchone() == (42,)


@pytest.mark.parametrize("malformed", ["missing", "extra"])
def test_file_arrow_struct_outputs_require_exact_fields(malformed):
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "strict-arrow-struct",
            "method_return_type": "STRUCT(document FILE, id INTEGER)",
        }
    )
    arrays = [pa.array([_file_record()], type=_file_arrow_type())]
    names = ["document"]
    if malformed == "extra":
        arrays.extend([pa.array([1]), pa.array(["discarded"])])
        names.extend(["id", "extra"])
    value = pa.StructArray.from_arrays(arrays, names=names)

    with pytest.raises(vane.InvalidInputException, match="exactly the declared fields"):
        contract.normalize_output_table(pa.table({"payload": value}))


@pytest.mark.parametrize("malformed", ["missing", "extra"])
def test_file_arrow_nested_non_file_struct_outputs_require_exact_fields(malformed):
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "strict-nested-arrow-struct",
            "method_return_type": "STRUCT(document FILE, meta STRUCT(id INTEGER, label VARCHAR))",
        }
    )
    meta_arrays = [pa.array([1])]
    meta_names = ["id"]
    if malformed == "extra":
        meta_arrays.extend([pa.array(["declared"]), pa.array(["discarded"])])
        meta_names.extend(["label", "extra"])
    meta = pa.StructArray.from_arrays(meta_arrays, names=meta_names)
    value = pa.StructArray.from_arrays(
        [pa.array([_file_record()], type=_file_arrow_type()), meta],
        names=["document", "meta"],
    )

    with pytest.raises(vane.InvalidInputException, match="exactly the declared fields"):
        contract.normalize_output_table(pa.table({"payload": value}))


def test_file_arrow_non_file_array_sibling_preserves_list_cast_input():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "list-to-array-sibling",
            "method_return_type": "STRUCT(document FILE, coords INTEGER[2])",
        }
    )
    coords = pa.array([[9, 8, 7], [1, 2]], type=pa.list_(pa.int32())).slice(1, 1)
    value = pa.StructArray.from_arrays(
        [
            pa.array([_file_record()], type=_file_arrow_type()),
            coords,
        ],
        names=["document", "coords"],
    )

    normalized = contract.normalize_output_table(pa.table({"payload": value}))

    assert normalized.column("payload").type.field("coords").type == pa.list_(pa.int32(), list_size=2)
    assert normalized.column("payload").to_pylist()[0]["coords"] == [1, 2]


def test_file_arrow_non_file_array_sibling_respects_list_view_offsets():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "list-view-to-array-sibling",
            "method_return_type": "STRUCT(document FILE, coords INTEGER[2])",
        }
    )
    coords = pa.ListViewArray.from_arrays(
        pa.array([2], type=pa.int32()),
        pa.array([2], type=pa.int32()),
        pa.array([9, 8, 1, 2], type=pa.int32()),
    )
    value = pa.StructArray.from_arrays(
        [pa.array([_file_record()], type=_file_arrow_type()), coords],
        names=["document", "coords"],
    )

    normalized = contract.normalize_output_table(pa.table({"payload": value}))

    assert normalized.column("payload").to_pylist()[0]["coords"] == [1, 2]


def test_file_arrow_non_file_array_sibling_rejects_uneven_list_rows():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "uneven-list-to-array-sibling",
            "method_return_type": "STRUCT(document FILE, coords INTEGER[2])",
        }
    )
    value = pa.StructArray.from_arrays(
        [
            pa.array([_file_record(), _file_record()], type=_file_arrow_type()),
            pa.array([[1], [2, 3, 4]], type=pa.list_(pa.int32())),
        ],
        names=["document", "coords"],
    )

    with pytest.raises(vane.InvalidInputException, match="could not normalize its declared storage"):
        contract.normalize_output_table(pa.table({"payload": value}))


@pytest.mark.parametrize("storage_kind", ["large_list", "large_list_view"])
def test_file_arrow_list_normalization_preserves_large_offsets(storage_kind):
    import pyarrow as pa

    from vane.execution.udf_file_contract import normalize_file_arrow_array

    files = pa.array([_file_record()], type=_file_arrow_type())
    if storage_kind == "large_list":
        source = pa.LargeListArray.from_arrays(pa.array([0, 1], type=pa.int64()), files)
    else:
        large_list_view_array = getattr(pa, "LargeListViewArray", None)
        is_large_list_view = getattr(pa.types, "is_large_list_view", None)
        if large_list_view_array is None or not callable(is_large_list_view):
            pytest.skip("PyArrow does not expose LargeListViewArray")
        source = large_list_view_array.from_arrays(
            pa.array([0], type=pa.int64()),
            pa.array([1], type=pa.int64()),
            files,
        )

    normalized = normalize_file_arrow_array(
        source,
        vane.list_type(vane.file_type()),
        boundary="large-list-output",
    )

    assert pa.types.is_large_list(normalized.type)
    assert normalized.to_pylist() == [[_file_record()]]


@pytest.mark.parametrize("storage_kind", ["list", "list_view"])
def test_chunked_file_list_normalization_promotes_every_chunk(monkeypatch, storage_kind):
    import pyarrow as pa

    import vane.execution.udf_file_contract as file_contract

    monkeypatch.setattr(file_contract, "_ARROW_LIST_OFFSET_MAX", 1)

    def make_chunk(records):
        files = pa.array(records, type=_file_arrow_type())
        if storage_kind == "list":
            return pa.ListArray.from_arrays(pa.array([0, len(records)], type=pa.int32()), files)
        list_view_array = getattr(pa, "ListViewArray", None)
        is_list_view = getattr(pa.types, "is_list_view", None)
        if list_view_array is None or not callable(is_list_view):
            pytest.skip("PyArrow does not expose ListViewArray")
        return list_view_array.from_arrays(
            pa.array([0], type=pa.int32()),
            pa.array([len(records)], type=pa.int32()),
            files,
        )

    source = pa.chunked_array(
        [
            make_chunk([_file_record(url="memory://small")]),
            make_chunk(
                [
                    _file_record(url="memory://large-1"),
                    _file_record(url="memory://large-2"),
                ]
            ),
        ]
    )

    normalized = file_contract.normalize_file_arrow_array(
        source,
        vane.list_type(vane.file_type()),
        boundary="chunked-large-file-list-output",
    )

    assert pa.types.is_large_list(normalized.type)
    assert all(pa.types.is_large_list(chunk.type) for chunk in normalized.chunks)
    assert normalized.to_pylist() == source.to_pylist()


def test_chunked_file_list_normalization_promotes_only_required_nested_paths(monkeypatch):
    import pyarrow as pa

    import vane.execution.udf_file_contract as file_contract

    monkeypatch.setattr(file_contract, "_ARROW_LIST_OFFSET_MAX", 1)
    list_type = pa.list_(_file_arrow_type())

    def make_chunk(long_records):
        return pa.StructArray.from_arrays(
            [
                pa.array([[_file_record(url="memory://short")]], type=list_type),
                pa.array([long_records], type=list_type),
                pa.array([0], type=pa.timestamp("ms")),
            ],
            names=["short", "long", "created_at"],
        )

    source = pa.chunked_array(
        [
            make_chunk([_file_record(url="memory://long-small")]),
            make_chunk(
                [
                    _file_record(url="memory://long-large-1"),
                    _file_record(url="memory://long-large-2"),
                ]
            ),
        ]
    )

    normalized = file_contract.normalize_file_arrow_array(
        source,
        vane.type("STRUCT(short FILE[], long FILE[], created_at TIMESTAMP)"),
        boundary="chunked-nested-large-file-list-output",
    )

    assert pa.types.is_list(normalized.type.field("short").type)
    assert pa.types.is_large_list(normalized.type.field("long").type)
    assert normalized.type.field("created_at").type == pa.timestamp("us")
    assert all(pa.types.is_list(chunk.type.field("short").type) for chunk in normalized.chunks)
    assert all(pa.types.is_large_list(chunk.type.field("long").type) for chunk in normalized.chunks)
    assert normalized.to_pylist() == source.to_pylist()


@pytest.mark.parametrize("declared_type", ["FILE[2]", "TENSOR(FILE, [2])"], ids=["array", "tensor"])
@pytest.mark.parametrize("storage_kind", ["list", "large_list", "list_view", "large_list_view"])
def test_file_fixed_sequence_normalization_accepts_variable_list_storage(declared_type, storage_kind):
    import pyarrow as pa

    from vane.execution.udf_file_contract import normalize_file_arrow_array

    records = [
        _file_record(url="memory://first"),
        _file_record(url="memory://second"),
    ]
    files = pa.array([_file_record(url="memory://ignored"), *records], type=_file_arrow_type())
    if storage_kind == "list":
        source = pa.ListArray.from_arrays(pa.array([0, 1, 3], type=pa.int32()), files).slice(1, 1)
    elif storage_kind == "large_list":
        source = pa.LargeListArray.from_arrays(pa.array([0, 1, 3], type=pa.int64()), files).slice(1, 1)
    else:
        class_name = "LargeListViewArray" if storage_kind == "large_list_view" else "ListViewArray"
        view_array = getattr(pa, class_name, None)
        predicate = getattr(pa.types, f"is_{storage_kind}", None)
        if view_array is None or not callable(predicate):
            pytest.skip(f"PyArrow does not expose {class_name}")
        offset_type = pa.int64() if storage_kind == "large_list_view" else pa.int32()
        source = view_array.from_arrays(
            pa.array([1], type=offset_type),
            pa.array([2], type=offset_type),
            files,
        )

    dtype = vane.tensor_type(vane.file_type(), (2,)) if declared_type.startswith("TENSOR") else vane.type(declared_type)
    normalized = normalize_file_arrow_array(source, dtype, boundary="variable-list-to-fixed-file-output")

    storage = normalized.storage if isinstance(normalized, pa.ExtensionArray) else normalized
    assert pa.types.is_fixed_size_list(storage.type)
    assert storage.type.list_size == 2
    assert normalized.to_pylist() == [records]


def test_file_array_normalization_validates_variable_list_lengths_and_values():
    import pyarrow as pa

    from vane.execution.udf_file_contract import normalize_file_arrow_array

    wrong_length = pa.array([[_file_record()]], type=pa.list_(_file_arrow_type()))
    with pytest.raises(vane.InvalidInputException, match="must have fixed size 2"):
        normalize_file_arrow_array(
            wrong_length,
            vane.type("FILE[2]"),
            boundary="wrong-length-file-array-output",
        )

    invalid_file = pa.array(
        [[_file_record(), _file_record(position=1, size=None)]],
        type=pa.list_(_file_arrow_type()),
    )
    with pytest.raises(vane.InvalidInputException, match="position and size"):
        normalize_file_arrow_array(
            invalid_file,
            vane.type("FILE[2]"),
            boundary="invalid-file-array-output",
        )

    hidden_invalid_file = pa.ListArray.from_arrays(
        pa.array([0, 2], type=pa.int32()),
        pa.array(
            [_file_record(), _file_record(position=1, size=None)],
            type=_file_arrow_type(),
        ),
        mask=pa.array([True], type=pa.bool_()),
    )
    normalized_null = normalize_file_arrow_array(
        hidden_invalid_file,
        vane.type("FILE[2]"),
        boundary="null-file-array-output",
    )
    assert normalized_null.to_pylist() == [None]


def test_file_list_normalization_accepts_fixed_size_list_storage():
    import pyarrow as pa

    from vane.execution.udf_file_contract import normalize_file_arrow_array

    records = [
        _file_record(url="memory://first"),
        _file_record(url="memory://second"),
    ]
    source = pa.FixedSizeListArray.from_arrays(
        pa.array(
            [_file_record(url="memory://ignored-1"), _file_record(url="memory://ignored-2"), *records],
            type=_file_arrow_type(),
        ),
        2,
    ).slice(1, 1)

    normalized = normalize_file_arrow_array(
        source,
        vane.list_type(vane.file_type()),
        boundary="fixed-to-variable-file-list-output",
    )

    assert pa.types.is_list(normalized.type)
    assert normalized.to_pylist() == [records]


def test_file_arrow_output_normalization_preserves_field_properties():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract, _map_array_from_offsets

    source_file_fields = [
        pa.field("url", pa.large_string(), nullable=False, metadata={b"file-url": b"preserved"}),
        pa.field("content_type", pa.large_string(), nullable=False, metadata={b"file-type": b"preserved"}),
        pa.field("position", pa.int64(), nullable=False, metadata={b"file-position": b"preserved"}),
        pa.field("size", pa.int64(), nullable=False, metadata={b"file-size": b"preserved"}),
        pa.field("checksum", pa.large_string(), nullable=False, metadata={b"file-checksum": b"preserved"}),
    ]
    source_file_type = pa.struct(source_file_fields)
    files = pa.array([_file_record()], type=source_file_type)
    documents = pa.ListArray.from_arrays(
        pa.array([0, 1], type=pa.int32()),
        files,
        type=pa.list_(
            pa.field("document_item", source_file_type, nullable=False, metadata={b"list-item": b"preserved"})
        ),
    )
    fixed = pa.FixedSizeListArray.from_arrays(
        files,
        type=pa.list_(
            pa.field("fixed_item", source_file_type, nullable=False, metadata={b"fixed-item": b"preserved"}),
            list_size=1,
        ),
    )
    lookup_type = pa.map_(
        pa.field("lookup_key", pa.string(), nullable=False, metadata={b"map-key": b"preserved"}),
        pa.field("lookup_value", source_file_type, metadata={b"map-value": b"preserved"}),
        keys_sorted=True,
    )
    # pa.array() discards MAP child metadata before normalization can see it.
    lookup = _map_array_from_offsets(
        [0, 1],
        pa.array(["document"]),
        files,
        mask=pa.array([False]),
        map_type=lookup_type,
    )
    assert lookup.type.key_field.metadata == {b"map-key": b"preserved"}
    assert lookup.type.item_field.metadata == {b"map-value": b"preserved"}
    payload_fields = [
        pa.field("document", source_file_type, nullable=False, metadata={b"document": b"preserved"}),
        pa.field("documents", documents.type, nullable=False, metadata={b"documents": b"preserved"}),
        pa.field("fixed", fixed.type, nullable=False, metadata={b"fixed": b"preserved"}),
        pa.field("lookup", lookup.type, nullable=False, metadata={b"lookup": b"preserved"}),
    ]
    payload = pa.StructArray.from_arrays([files, documents, fixed, lookup], fields=payload_fields)
    table = pa.Table.from_arrays(
        [payload],
        schema=pa.schema(
            [pa.field("payload", payload.type, nullable=False, metadata={b"top-field": b"preserved"})],
            metadata={b"schema": b"preserved"},
        ),
    )
    contract = FileUDFContract.from_payload(
        {
            "udf_name": "field-properties",
            "method_return_type": ("STRUCT(document FILE, documents FILE[], fixed FILE[1], lookup MAP(VARCHAR, FILE))"),
        }
    )

    normalized = contract.normalize_output_table(table)

    top_field = normalized.schema.field("payload")
    assert not top_field.nullable
    assert top_field.metadata == {b"top-field": b"preserved"}
    assert normalized.schema.metadata == {b"schema": b"preserved"}
    payload_type = top_field.type
    for name in ("document", "documents", "fixed", "lookup"):
        field = payload_type.field(name)
        assert not field.nullable
        assert field.metadata == {name.encode(): b"preserved"}

    list_field = payload_type.field("documents").type.value_field
    assert list_field.name == "document_item"
    assert not list_field.nullable
    assert list_field.metadata == {b"list-item": b"preserved"}

    fixed_field = payload_type.field("fixed").type.value_field
    assert fixed_field.name == "fixed_item"
    assert not fixed_field.nullable
    assert fixed_field.metadata == {b"fixed-item": b"preserved"}

    map_type = payload_type.field("lookup").type
    assert map_type.keys_sorted
    assert map_type.key_field.name == "lookup_key"
    assert map_type.key_field.metadata == {b"map-key": b"preserved"}
    assert map_type.item_field.name == "lookup_value"
    assert map_type.item_field.metadata == {b"map-value": b"preserved"}

    normalized_file_types = [
        payload_type.field("document").type,
        list_field.type,
        fixed_field.type,
        map_type.item_field.type,
    ]
    for normalized_file_type in normalized_file_types:
        for index, source_field in enumerate(source_file_fields):
            normalized_file_field = normalized_file_type.field(index)
            assert normalized_file_field.name == source_field.name
            assert normalized_file_field.nullable == source_field.nullable
            assert normalized_file_field.metadata == source_field.metadata
            expected_type = pa.string() if source_field.name in ("url", "content_type", "checksum") else pa.int64()
            assert normalized_file_field.type == expected_type


def test_file_map_builder_canonicalizes_an_all_valid_key_bitmap_without_aborting():
    import pyarrow as pa

    from vane.execution.udf_file_contract import _map_array_from_offsets

    keys = pa.ListArray.from_arrays(
        pa.array([0, 1], type=pa.int32()),
        pa.array([_file_record()], type=_file_arrow_type()),
        mask=pa.array([False]),
    )
    assert keys.null_count == 0
    assert keys.buffers()[0] is not None

    result = _map_array_from_offsets(
        [0, 1],
        keys,
        pa.array(["document"]),
        mask=pa.array([False]),
    )

    assert result.to_pylist() == [[([_file_record()], "document")]]
    assert result.keys.buffers()[0] is None

    with pytest.raises(ValueError, match="contain NULL"):
        _map_array_from_offsets(
            [0, 1],
            pa.array([None], type=pa.string()),
            pa.array(["document"]),
            mask=pa.array([False]),
        )


def test_file_map_normalization_drops_hidden_null_parent_entries_without_key_bitmap():
    import pyarrow as pa

    from vane.execution.udf_file_contract import _map_array_from_offsets, normalize_file_arrow_array

    source_file_type = pa.struct(
        [
            pa.field("url", pa.large_string()),
            pa.field("content_type", pa.large_string()),
            pa.field("position", pa.int64()),
            pa.field("size", pa.int64()),
            pa.field("checksum", pa.large_string()),
        ]
    )
    source = _map_array_from_offsets(
        [0, 1, 2],
        pa.array(["hidden", "visible"]),
        pa.array(
            [
                _file_record(position=1, size=None),
                _file_record(url="memory://visible"),
            ],
            type=source_file_type,
        ),
        mask=pa.array([True, False]),
    )

    normalized = normalize_file_arrow_array(
        source,
        vane.map_type(vane.sqltypes.VARCHAR, vane.file_type()),
        boundary="null-parent-file-map-output",
    )

    assert normalized.to_pylist() == [None, [("visible", _file_record(url="memory://visible"))]]
    assert normalized.keys.buffers()[0] is None


def test_file_tensor_normalization_uses_canonical_extension_storage():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    source_file_fields = [
        pa.field("url", pa.large_string(), nullable=False, metadata={b"file-url": b"preserved"}),
        pa.field("content_type", pa.large_string(), metadata={b"file-type": b"preserved"}),
        pa.field("position", pa.int64(), metadata={b"file-position": b"preserved"}),
        pa.field("size", pa.int64(), metadata={b"file-size": b"preserved"}),
        pa.field("checksum", pa.large_string(), metadata={b"file-checksum": b"preserved"}),
    ]
    source_file_type = pa.struct(source_file_fields)
    files = pa.array(
        [_file_record(url="memory://first"), _file_record(url="memory://second")],
        type=source_file_type,
    )
    storage = pa.FixedSizeListArray.from_arrays(
        files,
        type=pa.list_(
            pa.field("custom_item", source_file_type, nullable=False, metadata={b"item": b"source"}),
            list_size=2,
        ),
    )
    table = pa.Table.from_arrays(
        [storage],
        schema=pa.schema(
            [pa.field("documents", storage.type, nullable=False, metadata={b"field": b"preserved"})],
            metadata={b"schema": b"preserved"},
        ),
    )
    contract = FileUDFContract.from_payload(
        {
            "udf_name": "tensor-storage",
            "method_return_type": "TENSOR(FILE, [2])",
        }
    )

    normalized = contract.normalize_output_table(table)

    field = normalized.schema.field("documents")
    assert not field.nullable
    assert field.metadata == {b"field": b"preserved"}
    assert normalized.schema.metadata == {b"schema": b"preserved"}
    assert field.type.extension_name == "arrow.fixed_shape_tensor"
    assert tuple(field.type.shape) == (2,)
    assert field.type.storage_type.value_field.name == "item"
    normalized_file_type = field.type.storage_type.value_type
    for index, source_field in enumerate(source_file_fields):
        normalized_file_field = normalized_file_type.field(index)
        assert normalized_file_field.name == source_field.name
        assert normalized_file_field.nullable == source_field.nullable
        assert normalized_file_field.metadata == source_field.metadata
        expected_type = pa.string() if source_field.name in ("url", "content_type", "checksum") else pa.int64()
        assert normalized_file_field.type == expected_type
    assert normalized.column("documents").to_pylist() == [
        [_file_record(url="memory://first"), _file_record(url="memory://second")]
    ]


def test_file_arrow_non_file_map_sibling_preserves_string_cast_input():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "string-to-map-sibling",
            "method_return_type": "STRUCT(document FILE, attrs MAP(VARCHAR, INTEGER))",
        }
    )
    value = pa.StructArray.from_arrays(
        [
            pa.array([_file_record()], type=_file_arrow_type()),
            pa.array(["{a=1}"], type=pa.string()),
        ],
        names=["document", "attrs"],
    )

    normalized = contract.normalize_output_table(pa.table({"payload": value}))

    assert normalized.column("payload").type.field("attrs").type == pa.string()
    assert normalized.column("payload").to_pylist()[0]["attrs"] == "{a=1}"


@pytest.mark.parametrize(
    ("field_name", "declared_type", "source_value"),
    [
        ("items", "INTEGER[]", "[1, 2]"),
        ("meta", "STRUCT(id INTEGER)", "{'id': 1}"),
    ],
)
def test_file_arrow_non_file_composite_sibling_preserves_string_cast_input(
    field_name,
    declared_type,
    source_value,
):
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": f"string-to-{field_name}-sibling",
            "method_return_type": f"STRUCT(document FILE, {field_name} {declared_type})",
        }
    )
    value = pa.StructArray.from_arrays(
        [
            pa.array([_file_record()], type=_file_arrow_type()),
            pa.array([source_value], type=pa.string()),
        ],
        names=["document", field_name],
    )

    normalized = contract.normalize_output_table(pa.table({"payload": value}))

    assert normalized.column("payload").type.field(field_name).type == pa.string()
    assert normalized.column("payload").to_pylist()[0][field_name] == source_value


def test_scalar_file_udf_preserves_time_with_time_zone_offset():
    output_type = vane.type("STRUCT(document FILE, local_time TIME WITH TIME ZONE)")
    local_time = time(3, 4, 5, tzinfo=timezone(timedelta(hours=2, minutes=30)))

    @vane.func(return_dtype=output_type)
    def build_document(_value):
        return {
            "document": vane.File("memory://time-zone"),
            "local_time": local_time,
        }

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))

    assert result.project("payload.local_time::VARCHAR AS local_time").fetchone() == ("03:04:05+02:30",)


def test_scalar_file_udf_preserves_time_with_time_zone_storage_provenance():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "time-zone-provenance",
            "method_return_type": "STRUCT(document FILE, local_time TIME WITH TIME ZONE)",
        }
    )
    outputs = [
        {
            "document": vane.File("memory://time-zone-blob"),
            "local_time": b"03:04:05+02:00",
        },
        {
            "document": vane.File("memory://time-zone-text"),
            "local_time": "03:04:05+02:00",
        },
    ]

    arrays = contract.scalar_outputs_to_arrays(outputs)

    assert [array.type.field("local_time").type for array in arrays] == [pa.binary(), pa.string()]


@pytest.mark.parametrize(
    ("integer_type", "wide"),
    [
        ("HUGEINT", 2**127 - 1),
        ("UHUGEINT", 2**127),
    ],
)
def test_scalar_file_udf_preserves_full_128_bit_integer_sibling(integer_type, wide):
    output_type = vane.type(f"STRUCT(document FILE, wide {integer_type})")

    @vane.func(return_dtype=output_type)
    def build_document(_value):
        return {
            "document": vane.File("memory://wide-composite"),
            "wide": wide,
        }

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))

    document, returned_wide = result.project("payload.document AS document, payload.wide::VARCHAR AS wide").fetchone()
    assert document == vane.File("memory://wide-composite")
    assert int(returned_wide) == wide


@pytest.mark.parametrize("integer_type", ["HUGEINT", "UHUGEINT"])
def test_scalar_file_udf_accepts_textual_128_bit_integer_sibling(integer_type):
    output_type = vane.type(f"STRUCT(document FILE, wide {integer_type})")

    @vane.func(return_dtype=output_type)
    def build_document(_value):
        return {
            "document": vane.File("memory://textual-wide-composite"),
            "wide": "42",
        }

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))

    assert result.project("payload.wide::VARCHAR").fetchone() == ("42",)


@pytest.mark.parametrize("integer_type", ["HUGEINT", "UHUGEINT", "BIGNUM"])
def test_native_file_udfs_preserve_numeric_casts_for_wide_integer_siblings(integer_type):
    output_type = vane.type(f"STRUCT(document FILE, wide {integer_type})")

    @vane.func(return_dtype=output_type)
    def build_document(_value):
        return {
            "document": vane.File("memory://numeric-wide-scalar"),
            "wide": 1.0,
        }

    connection = vane.connect()
    scalar_result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))
    assert scalar_result.project("payload.wide::VARCHAR").fetchone() == ("1",)

    def build_row(_row):
        return {
            "payload": {
                "document": vane.File("memory://numeric-wide-flat-map"),
                "wide": 2.0,
            }
        }

    flat_map_result = connection.sql("SELECT 1 AS value").flat_map(
        build_row,
        schema={"payload": output_type},
        execution_backend="subprocess_task",
    )
    assert flat_map_result.project("payload.wide::VARCHAR").fetchone() == ("2",)


def test_native_file_udf_splits_mixed_wide_integer_storage():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "mixed-wide-storage",
            "method_return_type": "STRUCT(document FILE, wide HUGEINT)",
        }
    )
    outputs = [
        {"document": vane.File("memory://wide-integer"), "wide": 2**100},
        {"document": vane.File("memory://wide-double"), "wide": 1.0},
    ]

    arrays = contract.scalar_outputs_to_arrays(outputs)

    assert [array.type.field("wide").type for array in arrays] == [pa.string(), pa.float64()]


def test_file_udfs_roundtrip_parameterized_decimal_sibling_contract():
    output_type = vane.type("STRUCT(document FILE, amount DECIMAL(10,2))")

    @vane.func(return_dtype=output_type)
    def build_document(_value):
        return {
            "document": vane.File("memory://decimal-scalar"),
            "amount": Decimal("12.34"),
        }

    connection = vane.connect()
    scalar_result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))
    assert scalar_result.project("payload.amount::VARCHAR").fetchone() == ("12.34",)

    def build_row(_row):
        return {
            "payload": {
                "document": vane.File("memory://decimal-flat-map"),
                "amount": Decimal("56.78"),
            }
        }

    flat_map_result = connection.sql("SELECT 1 AS value").flat_map(
        build_row,
        schema={"payload": output_type},
        execution_backend="subprocess_task",
    )
    assert flat_map_result.project("payload.amount::VARCHAR").fetchone() == ("56.78",)


def test_file_udfs_preserve_arbitrary_precision_bignum_siblings():
    import pyarrow as pa

    wide = 10**100
    output_type = vane.type("STRUCT(document FILE, wide BIGNUM)")

    @vane.func(return_dtype=output_type)
    def build_document(_value):
        return {"document": vane.File("memory://bignum"), "wide": wide}

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))
    assert result.project("payload.wide::VARCHAR AS wide").fetchone() == (str(wide),)

    @vane.func.batch(return_dtype=output_type)
    def build_documents(values):
        return pa.StructArray.from_arrays(
            [
                pa.array([_file_record()] * len(values), type=_file_arrow_type()),
                pa.array([str(wide)] * len(values)),
            ],
            names=["document", "wide"],
        )

    batch_result = build_documents(pa.array([1], type=pa.int32()))
    assert batch_result.type.field("wide").type == pa.string()
    assert batch_result.to_pylist() == [{"document": _file_record(), "wide": str(wide)}]


def test_native_file_udfs_encode_special_leaves_inside_non_file_composites():
    wide = 10**100
    output_type = vane.type("STRUCT(document FILE, meta STRUCT(wide BIGNUM, label VARCHAR))")

    @vane.func(return_dtype=output_type)
    def build_document(_value):
        return {
            "document": vane.File("memory://nested-bignum-scalar"),
            "meta": {"wide": wide, "label": 42},
        }

    connection = vane.connect()
    scalar_result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))
    assert scalar_result.project("payload.meta.wide::VARCHAR, payload.meta.label").fetchone() == (str(wide), "42")

    def build_row(_row):
        return {
            "payload": {
                "document": vane.File("memory://nested-bignum-flat-map"),
                "meta": {"wide": wide, "label": 43},
            }
        }

    flat_map_result = connection.sql("SELECT 1 AS value").flat_map(
        build_row,
        schema={"payload": output_type},
        execution_backend="subprocess_task",
    )
    assert flat_map_result.project("payload.meta.wide::VARCHAR, payload.meta.label").fetchone() == (
        str(wide),
        "43",
    )


@pytest.mark.parametrize(
    ("nested_type", "nested_value"),
    [
        ("STRUCT(wide BIGNUM)", "{'wide': 42}"),
        ("BIGNUM[]", "[42]"),
        ("BIGNUM[1]", "[42]"),
        ("MAP(VARCHAR, BIGNUM)", "{key=42}"),
        ("TENSOR(BIGNUM, [1])", "[42]"),
    ],
    ids=["struct", "list", "array", "map", "tensor"],
)
def test_native_file_special_composite_siblings_preserve_cross_type_storage(nested_type, nested_value):
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "special-composite-cross-type",
            "method_return_type": f"STRUCT(document FILE, meta {nested_type})",
        }
    )

    result = contract.scalar_outputs_to_array(
        [
            {
                "document": vane.File("memory://special-composite-cross-type"),
                "meta": nested_value,
            }
        ]
    )

    assert result.type.field("meta").type == pa.string()
    assert result.to_pylist()[0]["meta"] == nested_value


def test_native_file_special_struct_sibling_cross_type_casts_and_splits():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    output_type = vane.type("STRUCT(document FILE, meta STRUCT(wide BIGNUM))")
    contract = FileUDFContract.from_payload(
        {
            "udf_name": "special-struct-cross-type",
            "method_return_type": str(output_type),
        }
    )
    pieces = contract.scalar_outputs_to_arrays(
        [
            {
                "document": vane.File("memory://special-struct-declared"),
                "meta": {"wide": 10**100},
            },
            {
                "document": vane.File("memory://special-struct-cross-type"),
                "meta": "{'wide': 42}",
            },
        ]
    )
    assert [piece.type.field("meta").type for piece in pieces] == [
        pa.struct([pa.field("wide", pa.string())]),
        pa.string(),
    ]

    @vane.func(return_dtype=output_type)
    def build_document(_value):
        return {
            "document": vane.File("memory://special-struct-scalar"),
            "meta": "{'wide': 42}",
        }

    connection = vane.connect()
    scalar_result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))
    assert scalar_result.project("payload.meta.wide::VARCHAR").fetchone() == ("42",)

    def build_row(_row):
        return {
            "payload": {
                "document": vane.File("memory://special-struct-flat-map"),
                "meta": "{'wide': 43}",
            }
        }

    flat_map_result = connection.sql("SELECT 1 AS value").flat_map(
        build_row,
        schema={"payload": output_type},
        execution_backend="subprocess_task",
    )
    assert flat_map_result.project("payload.meta.wide::VARCHAR").fetchone() == ("43",)


def test_native_file_udfs_split_mixed_sibling_storage_for_duckdb_casts():
    output_type = vane.type("STRUCT(document FILE, id BIGINT)")

    @vane.func(return_dtype=output_type)
    def build_document(value):
        return {
            "document": vane.File(f"memory://mixed-scalar-{value}"),
            "id": value if value == 1 else str(value),
        }

    connection = vane.connect()
    source = connection.sql("SELECT i AS value FROM range(1, 3) AS t(i)")
    scalar_result = source.select(build_document(vane.col("value")).alias("payload"))
    assert scalar_result.project("payload.document.url, payload.id").fetchall() == [
        ("memory://mixed-scalar-1", 1),
        ("memory://mixed-scalar-2", 2),
    ]

    def build_row(row):
        value = row["value"]
        return {
            "payload": {
                "document": vane.File(f"memory://mixed-flat-map-{value}"),
                "id": value if value == 1 else str(value),
            }
        }

    flat_map_result = source.flat_map(
        build_row,
        schema={"payload": output_type},
        execution_backend="subprocess_task",
    )
    assert flat_map_result.project("payload.document.url, payload.id").fetchall() == [
        ("memory://mixed-flat-map-1", 1),
        ("memory://mixed-flat-map-2", 2),
    ]


def test_empty_flat_map_file_udf_preserves_composite_output_type():
    output_type = vane.type("STRUCT(document FILE, id UUID, created_at TIMESTAMPTZ, wide UHUGEINT)")

    def emit_nothing(_row):
        return None

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").flat_map(
        emit_nothing,
        schema={"payload": output_type},
        execution_backend="subprocess_task",
    )

    assert dict(result.types[0].children)["document"].is_file()
    assert result.fetchall() == []


def test_flat_map_file_udf_preserves_map_keys_named_key_and_value():
    output_type = vane.map_type(vane.sqltypes.VARCHAR, vane.list_type(vane.file_type()))

    def build_map(_row):
        return {
            "files": {
                "key": [vane.File("memory://key")],
                "value": [vane.File("memory://value")],
            }
        }

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").flat_map(
        build_map,
        schema={"files": output_type},
        execution_backend="subprocess_task",
    )

    assert result.fetchone()[0] == {
        "key": [vane.File("memory://key")],
        "value": [vane.File("memory://value")],
    }


def test_flat_map_file_udf_roundtrips_parallel_map_representation():
    output_type = vane.map_type(vane.list_type(vane.file_type()), vane.sqltypes.VARCHAR)

    def copy_map(row):
        assert row["files"] == {
            "key": [[vane.File("memory://key")]],
            "value": ["document"],
        }
        return {"files": row["files"]}

    connection = vane.connect()
    source = connection.sql("SELECT map([[file('memory://key', NULL, NULL, NULL, NULL)]], ['document']) AS files")
    result = source.flat_map(
        copy_map,
        schema={"files": output_type},
        execution_backend="subprocess_task",
    )

    assert result.fetchone()[0] == {
        "key": [[vane.File("memory://key")]],
        "value": ["document"],
    }


def test_batch_file_udf_accepts_chunked_eager_output():
    import pyarrow as pa

    file_type = pa.struct(
        [
            pa.field("url", pa.large_string()),
            pa.field("content_type", pa.large_string()),
            pa.field("position", pa.int64()),
            pa.field("size", pa.int64()),
            pa.field("checksum", pa.large_string()),
        ]
    )

    @vane.func.batch(return_dtype=vane.file_type())
    def identity(values):
        return values

    values = pa.chunked_array(
        [
            pa.array([_file_record(url="memory://first")], type=file_type),
            pa.array([None, _file_record(url="memory://second")], type=file_type),
        ]
    )
    result = identity(values)

    assert isinstance(result, pa.ChunkedArray)
    assert result.type.equals(_file_arrow_type())
    assert result.to_pylist() == values.to_pylist()


def test_batch_file_udf_accepts_string_view_storage():
    import pyarrow as pa

    string_view = getattr(pa, "string_view", None)
    if string_view is None:
        pytest.skip("PyArrow does not expose string_view")
    file_type = pa.struct(
        [
            pa.field("url", string_view()),
            pa.field("content_type", string_view()),
            pa.field("position", pa.int64()),
            pa.field("size", pa.int64()),
            pa.field("checksum", string_view()),
        ]
    )

    @vane.func.batch(return_dtype=vane.file_type())
    def identity(values):
        return values

    values = pa.array([_file_record()], type=file_type)
    result = identity(values)

    assert result.type.equals(_file_arrow_type())
    assert result.to_pylist() == values.to_pylist()


def test_batch_file_udf_accepts_list_view_storage():
    import pyarrow as pa

    list_view_array = getattr(pa, "ListViewArray", None)
    if list_view_array is None:
        pytest.skip("PyArrow does not expose ListViewArray")
    files = pa.array(
        [
            _file_record(url="memory://first"),
            _file_record(url="memory://second"),
            _file_record(url="memory://third"),
        ],
        type=_file_arrow_type(),
    )
    views = list_view_array.from_arrays(
        pa.array([1, 0, 0], type=pa.int32()),
        pa.array([2, 2, 1], type=pa.int32()),
        files,
        mask=pa.array([False, True, False]),
    )

    @vane.func.batch(return_dtype=vane.list_type(vane.file_type()))
    def identity(_values):
        return views

    result = identity(pa.array([1, 2, 3], type=pa.int32()))

    assert result.type.equals(pa.list_(_file_arrow_type()))
    assert result.to_pylist() == [
        [_file_record(url="memory://second"), _file_record(url="memory://third")],
        None,
        [_file_record(url="memory://first")],
    ]


def test_batch_file_udf_accepts_binary_view_sibling_storage():
    import pyarrow as pa

    binary_view = getattr(pa, "binary_view", None)
    if binary_view is None:
        pytest.skip("PyArrow does not expose binary_view")
    output_type = vane.type("STRUCT(document FILE, payload BLOB)")

    @vane.func.batch(return_dtype=output_type)
    def build_document(values):
        return pa.StructArray.from_arrays(
            [
                pa.array([_file_record()] * len(values), type=_file_arrow_type()),
                pa.array([b"payload"] * len(values), type=binary_view()),
            ],
            names=["document", "payload"],
        )

    result = build_document(pa.array([1], type=pa.int32()))

    assert result.type.field("payload").type == pa.binary()
    assert result.to_pylist() == [{"document": _file_record(), "payload": b"payload"}]


def test_map_batches_normalizes_file_storage_across_output_batches():
    import cloudpickle
    import pyarrow as pa

    from vane.execution._udf_runtime import UDFExecutor

    def emit_files(_table):
        import pyarrow as pa

        for url, string_type in (("memory://regular", pa.string()), ("memory://large", pa.large_string())):
            file_type = pa.struct(
                [
                    pa.field("url", string_type),
                    pa.field("content_type", string_type),
                    pa.field("position", pa.int64()),
                    pa.field("size", pa.int64()),
                    pa.field("checksum", string_type),
                ]
            )
            yield pa.table({"value": pa.array([_file_record(url=url)], type=file_type)})

    executor = UDFExecutor(
        {
            "function_pickle": cloudpickle.dumps(emit_files),
            "call_mode": "map_batches",
            "execution_backend": "subprocess_task",
            "output_schema": [{"name": "value", "kind": "duckdb_type", "type": "FILE"}],
            "stream_output": True,
            "output_batch_size": 2,
        }
    )
    try:
        executor.submit(pa.table({"input": [1]}))
        output = executor.drain_outputs()
    finally:
        executor.close()

    assert len(output) == 1
    assert output[0].column("value").type.equals(_file_arrow_type())
    assert output[0].column("value").to_pylist() == [
        _file_record(url="memory://regular"),
        _file_record(url="memory://large"),
    ]


@pytest.mark.parametrize("nested", [False, True], ids=["columns", "composite"])
def test_map_batches_normalizes_file_siblings_across_output_batches(nested):
    import cloudpickle
    import pyarrow as pa

    from vane.execution._udf_runtime import UDFExecutor

    def emit_documents(_table):
        import pyarrow as pa

        for identifier, integer_type in ((1, pa.int32()), (2, pa.int64())):
            document = pa.array([_file_record(url=f"memory://{identifier}")], type=_file_arrow_type())
            identifiers = pa.array([identifier], type=integer_type)
            if not nested:
                yield pa.table({"document": document, "id": identifiers})
                continue
            payload_type = pa.struct(
                [
                    pa.field("document", _file_arrow_type()),
                    pa.field("id", integer_type),
                ]
            )
            yield pa.table({"payload": pa.StructArray.from_arrays([document, identifiers], type=payload_type)})

    output_schema = (
        [{"name": "payload", "kind": "duckdb_type", "type": "STRUCT(document FILE, id BIGINT)"}]
        if nested
        else [
            {"name": "document", "kind": "duckdb_type", "type": "FILE"},
            {"name": "id", "kind": "duckdb_type", "type": "BIGINT"},
        ]
    )
    executor = UDFExecutor(
        {
            "function_pickle": cloudpickle.dumps(emit_documents),
            "call_mode": "map_batches",
            "execution_backend": "subprocess_task",
            "output_schema": output_schema,
            "stream_output": True,
            "output_batch_size": 2,
        }
    )
    try:
        executor.submit(pa.table({"input": [1]}))
        output = executor.drain_outputs()
    finally:
        executor.close()

    assert len(output) == 1
    if nested:
        payload = output[0].column("payload")
        assert payload.type.field("id").type == pa.int64()
        assert payload.to_pylist() == [
            {"document": _file_record(url="memory://1"), "id": 1},
            {"document": _file_record(url="memory://2"), "id": 2},
        ]
    else:
        assert output[0].column("id").type == pa.int64()
        assert output[0].column("document").to_pylist() == [
            _file_record(url="memory://1"),
            _file_record(url="memory://2"),
        ]


def test_map_batches_normalizes_lossless_temporal_siblings_across_batches():
    import cloudpickle
    import pyarrow as pa

    from vane.execution._udf_runtime import UDFExecutor

    occurred_at = datetime(2024, 1, 2, 3, 4, 5, 123000)

    def emit_documents(_table):
        document = pa.array([_file_record()], type=_file_arrow_type())
        for unit in ("ms", "us"):
            yield pa.table(
                {
                    "document": document,
                    "occurred_at": pa.array([occurred_at], type=pa.timestamp(unit)),
                }
            )

    executor = UDFExecutor(
        {
            "function_pickle": cloudpickle.dumps(emit_documents),
            "call_mode": "map_batches",
            "execution_backend": "subprocess_task",
            "output_schema": [
                {"name": "document", "kind": "duckdb_type", "type": "FILE"},
                {"name": "occurred_at", "kind": "duckdb_type", "type": "TIMESTAMP"},
            ],
            "stream_output": True,
            "output_batch_size": 2,
        }
    )
    try:
        executor.submit(pa.table({"input": [1]}))
        output = executor.drain_outputs()
    finally:
        executor.close()

    assert len(output) == 1
    assert output[0].column("occurred_at").type == pa.timestamp("us")
    assert output[0].column("occurred_at").to_pylist() == [occurred_at, occurred_at]


def test_file_output_normalizes_chunked_temporal_siblings_atomically():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    logical_type = "STRUCT(document FILE, occurred_at TIMESTAMP)"
    contract = FileUDFContract.from_payload(
        {
            "udf_name": "chunked-temporal",
            "output_schema": [{"name": "payload", "kind": "duckdb_type", "type": logical_type}],
        }
    )

    file_type = pa.struct(
        [
            pa.field("url", pa.large_string()),
            pa.field("content_type", pa.large_string()),
            pa.field("position", pa.int64()),
            pa.field("size", pa.int64()),
            pa.field("checksum", pa.large_string()),
        ]
    )

    def make_chunk(nanoseconds):
        return pa.StructArray.from_arrays(
            [
                pa.array([_file_record()], type=file_type),
                pa.array([nanoseconds], type=pa.timestamp("ns")),
            ],
            names=["document", "occurred_at"],
        )

    payload = pa.chunked_array([make_chunk(1_000), make_chunk(1_001)])
    table = pa.table({"payload": payload})

    normalized = contract.normalize_output_table(table)

    assert normalized.column("payload").num_chunks == 2
    assert normalized.column("payload").type.field("occurred_at").type == pa.timestamp("ns")
    assert normalized.column("payload").type.field("document").type == _file_arrow_type()


def test_arrow_scalar_file_output_normalizes_temporal_batches_atomically():
    import pyarrow as pa

    output_type = vane.type("STRUCT(document FILE, occurred_at TIMESTAMP)")

    @vane.func.batch(return_dtype=output_type, batch_size=1)
    def build_document(identifiers):
        identifier = identifiers[0].as_py()
        nanoseconds = 1_000 if identifier == 0 else 1_001
        return pa.StructArray.from_arrays(
            [
                pa.array(
                    [_file_record(url=f"memory://temporal-{identifier}")],
                    type=_file_arrow_type(),
                ),
                pa.array([nanoseconds], type=pa.timestamp("ns")),
            ],
            names=["document", "occurred_at"],
        )

    connection = vane.connect()
    source = connection.sql("SELECT i FROM range(2) AS t(i)")
    result = source.select(build_document(vane.col("i")).alias("payload"))

    assert result.project("payload.document.url, epoch_ns(payload.occurred_at)").fetchall() == [
        ("memory://temporal-0", 1_000),
        ("memory://temporal-1", 1_000),
    ]


def test_arrow_scalar_file_output_keeps_cross_type_batches_and_null_rows_separate():
    import cloudpickle
    import pyarrow as pa

    from vane.execution._udf_runtime import UDFExecutor

    def build_document(identifiers):
        identifier = identifiers[0].as_py()
        text = pa.array([b"\xc3\xa9"], type=pa.binary()) if identifier == 0 else pa.array(["second"], type=pa.string())
        return pa.StructArray.from_arrays(
            [
                pa.array([_file_record(url=f"memory://heterogeneous-{identifier}")], type=_file_arrow_type()),
                text,
            ],
            names=["document", "text"],
        )

    executor = UDFExecutor(
        {
            "function_pickle": cloudpickle.dumps(build_document),
            "call_mode": "map",
            "execution_backend": "subprocess_task",
            "scalar_udf_type": "arrow",
            "batch_size": 1,
            "method_return_type": "STRUCT(document FILE, text VARCHAR)",
        }
    )
    try:
        executor.submit(pa.table({"identifier": pa.array([0, None, 1], type=pa.int64())}))
        outputs = executor.drain_outputs()
    finally:
        executor.close()

    assert [output.num_rows for output in outputs] == [2, 1]
    assert [output.column("value").type.field("text").type for output in outputs] == [
        pa.binary(),
        pa.string(),
    ]
    assert outputs[0].column("value").to_pylist() == [
        {
            "document": _file_record(url="memory://heterogeneous-0"),
            "text": b"\xc3\xa9",
        },
        None,
    ]
    assert outputs[1].column("value").to_pylist() == [
        {
            "document": _file_record(url="memory://heterogeneous-1"),
            "text": "second",
        }
    ]


def test_file_tensor_output_contract_normalizes_and_validates_file_elements():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "file-tensor",
            "output_schema": [
                {
                    "name": "documents",
                    "kind": "tensor",
                    "dtype": "FILE",
                    "shape": [2],
                }
            ],
        }
    )
    assert contract.has_file_outputs
    assert str(contract.output_types[0].id) == "tensor"

    file_type = pa.struct(
        [
            pa.field("url", pa.large_string()),
            pa.field("content_type", pa.large_string()),
            pa.field("position", pa.int64()),
            pa.field("size", pa.int64()),
            pa.field("checksum", pa.large_string()),
        ]
    )

    def file_tensor(records):
        files = pa.array(records, type=file_type)
        storage = pa.FixedSizeListArray.from_arrays(files, 2)
        return pa.ExtensionArray.from_storage(pa.fixed_shape_tensor(file_type, (2,)), storage)

    records = [
        _file_record(url="memory://first"),
        _file_record(url="memory://second"),
    ]
    normalized = contract.normalize_output_table(pa.table({"documents": file_tensor(records)}))

    assert normalized.column("documents").type == pa.fixed_shape_tensor(_file_arrow_type(), (2,))
    assert normalized.column("documents").to_pylist() == [records]

    invalid_records = [records[0], _file_record(url="memory://invalid", position=1, size=None)]
    with pytest.raises(vane.InvalidInputException, match="position and size"):
        contract.normalize_output_table(pa.table({"documents": file_tensor(invalid_records)}))

    files = pa.array(records, type=file_type)
    wrong_shape = pa.ExtensionArray.from_storage(
        pa.fixed_shape_tensor(file_type, (1, 2)),
        pa.FixedSizeListArray.from_arrays(files, 2),
    )
    with pytest.raises(vane.InvalidInputException, match="declared Arrow tensor metadata"):
        contract.normalize_output_table(pa.table({"documents": wrong_shape}))

    method_contract = FileUDFContract.from_payload(
        {
            "udf_name": "file-tensor-method",
            "method_return_type": "TENSOR(FILE, [2])",
        }
    )
    assert method_contract.has_file_outputs
    assert str(method_contract.output_types[0]) == "TENSOR(FILE, [2])"


@pytest.mark.parametrize("raw_uuid_bytes", [False, True], ids=["varchar", "blob"])
def test_map_batches_preserves_duckdb_casts_for_non_file_columns(raw_uuid_bytes):
    identifier = UUID("00112233-4455-6677-8899-aabbccddeeff")

    def build_output(table):
        import pyarrow as pa

        file_type = pa.struct(
            [
                pa.field("url", pa.string()),
                pa.field("content_type", pa.string()),
                pa.field("position", pa.int64()),
                pa.field("size", pa.int64()),
                pa.field("checksum", pa.string()),
            ]
        )
        return pa.table(
            {
                "document": pa.array(
                    [
                        {
                            "url": "memory://batch-coercion",
                            "content_type": None,
                            "position": None,
                            "size": None,
                            "checksum": None,
                        }
                    ]
                    * table.num_rows,
                    type=file_type,
                ),
                "identifier": pa.array(
                    [identifier.bytes if raw_uuid_bytes else str(identifier)] * table.num_rows,
                    type=pa.binary() if raw_uuid_bytes else pa.string(),
                ),
            }
        )

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").map_batches(
        build_output,
        schema={"document": vane.file_type(), "identifier": vane.sqltypes.UUID},
        execution_backend="subprocess_task",
    )

    assert result.fetchone() == (vane.File("memory://batch-coercion"), identifier)


def test_map_batches_preserves_invalid_blob_for_duckdb_uuid_cast():
    identifier = UUID("00112233-4455-6677-8899-aabbccddeeff")

    def build_output(table):
        import pyarrow as pa

        file_type = pa.struct(
            [
                pa.field("url", pa.string()),
                pa.field("content_type", pa.string()),
                pa.field("position", pa.int64()),
                pa.field("size", pa.int64()),
                pa.field("checksum", pa.string()),
            ]
        )
        record = {
            "url": "memory://invalid-uuid",
            "content_type": None,
            "position": None,
            "size": None,
            "checksum": None,
        }
        return pa.table(
            {
                "document": pa.array([record] * table.num_rows, type=file_type),
                "identifier": pa.array([str(identifier).encode()] * table.num_rows, type=pa.binary()),
            }
        )

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").map_batches(
        build_output,
        schema={"document": vane.file_type(), "identifier": vane.sqltypes.UUID},
        execution_backend="subprocess_task",
    )

    with pytest.raises(Exception, match="BLOB.*UUID|Could not convert"):
        result.fetchall()


def test_map_batches_stabilizes_uuid_sibling_transport_across_batches():
    import cloudpickle
    import pyarrow as pa

    from vane.execution._udf_runtime import UDFExecutor

    identifier = UUID("00112233-4455-6677-8899-aabbccddeeff")

    def emit_documents(_table):
        import pyarrow as pa

        document = pa.array([_file_record()], type=_file_arrow_type())
        yield pa.table({"document": document, "identifier": pa.array([identifier])})
        yield pa.table({"document": document, "identifier": pa.array([str(identifier)])})

    executor = UDFExecutor(
        {
            "function_pickle": cloudpickle.dumps(emit_documents),
            "call_mode": "map_batches",
            "execution_backend": "subprocess_task",
            "output_schema": [
                {"name": "document", "kind": "duckdb_type", "type": "FILE"},
                {"name": "identifier", "kind": "duckdb_type", "type": "UUID"},
            ],
            "stream_output": True,
            "output_batch_size": 2,
        }
    )
    try:
        executor.submit(pa.table({"input": [1]}))
        output = executor.drain_outputs()
    finally:
        executor.close()

    assert len(output) == 1
    assert output[0].column("identifier").type == pa.string()
    assert output[0].column("identifier").to_pylist() == [str(identifier), str(identifier)]


@pytest.mark.parametrize("stream_output", [False, True], ids=["non-stream", "stream"])
def test_map_batches_keeps_cross_type_file_sibling_batches_separate(stream_output):
    import cloudpickle
    import pyarrow as pa

    from vane.execution._udf_runtime import UDFExecutor

    def emit_documents(_table):
        import pyarrow as pa

        file_type = pa.struct(
            [
                pa.field("url", pa.string()),
                pa.field("content_type", pa.string()),
                pa.field("position", pa.int64()),
                pa.field("size", pa.int64()),
                pa.field("checksum", pa.string()),
            ]
        )
        document = pa.array(
            [
                {
                    "url": "memory://streamed-cast",
                    "content_type": None,
                    "position": None,
                    "size": None,
                    "checksum": None,
                }
            ],
            type=file_type,
        )
        yield pa.table({"document": document, "text": pa.array([b"first"], type=pa.binary())})
        yield pa.table({"document": document, "text": pa.array(["second"], type=pa.string())})

    executor = UDFExecutor(
        {
            "function_pickle": cloudpickle.dumps(emit_documents),
            "call_mode": "map_batches",
            "execution_backend": "subprocess_task",
            "output_schema": [
                {"name": "document", "kind": "duckdb_type", "type": "FILE"},
                {"name": "text", "kind": "duckdb_type", "type": "VARCHAR"},
            ],
            "stream_output": stream_output,
            "output_batch_size": 2,
        }
    )
    try:
        executor.submit(pa.table({"input": [1]}))
        output = executor.drain_outputs()
    finally:
        executor.close()

    assert [table.column("text").type for table in output] == [pa.binary(), pa.string()]
    assert [table.column("text").to_pylist() for table in output] == [[b"first"], ["second"]]


def test_eager_batch_file_udf_uses_stable_uuid_sibling_transport():
    import pyarrow as pa

    identifier = UUID("00112233-4455-6677-8899-aabbccddeeff")
    output_type = vane.type("STRUCT(document FILE, identifier UUID)")

    @vane.func.batch(return_dtype=output_type)
    def build_document(values):
        return pa.StructArray.from_arrays(
            [
                pa.array([_file_record()] * len(values), type=_file_arrow_type()),
                pa.array([identifier] * len(values)),
            ],
            names=["document", "identifier"],
        )

    result = build_document(pa.array([1], type=pa.int32()))

    assert result.type.field("identifier").type == pa.string()
    assert result.to_pylist() == [{"document": _file_record(), "identifier": str(identifier)}]


def test_map_batches_defers_non_file_cast_semantics_to_duckdb():
    import pyarrow as pa

    def emit_documents(_table):
        file_type = pa.struct(
            [
                pa.field("url", pa.string()),
                pa.field("content_type", pa.string()),
                pa.field("position", pa.int64()),
                pa.field("size", pa.int64()),
                pa.field("checksum", pa.string()),
            ]
        )
        document = pa.array(
            [
                {
                    "url": "memory://udf",
                    "content_type": "application/octet-stream",
                    "position": 0,
                    "size": 3,
                    "checksum": "sha256:abc",
                }
            ],
            type=file_type,
        )
        yield pa.table({"document": document, "text": pa.array([b"\xc3\xa9"], type=pa.binary())})
        yield pa.table({"document": document, "text": pa.array(["second"], type=pa.string())})

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").map_batches(
        emit_documents,
        schema={"document": vane.file_type(), "text": vane.sqltypes.VARCHAR},
        execution_backend="subprocess_task",
        output_batch_size=2,
    )

    assert result.project("document.url, text").fetchall() == [
        ("memory://udf", r"\xC3\xA9"),
        ("memory://udf", "second"),
    ]


def test_batch_expression_defers_file_sibling_cast_semantics_to_duckdb():
    import pyarrow as pa

    output_type = vane.type("STRUCT(document FILE, text VARCHAR)")

    @vane.func.batch(return_dtype=output_type)
    def emit_document(values):
        file_type = pa.struct(
            [
                pa.field("url", pa.string()),
                pa.field("content_type", pa.string()),
                pa.field("position", pa.int64()),
                pa.field("size", pa.int64()),
                pa.field("checksum", pa.string()),
            ]
        )
        documents = pa.array(
            [
                {
                    "url": "memory://batch-expression",
                    "content_type": None,
                    "position": None,
                    "size": None,
                    "checksum": None,
                }
            ]
            * len(values),
            type=file_type,
        )
        return pa.StructArray.from_arrays(
            [documents, pa.array([b"\xc3\xa9"] * len(values), type=pa.binary())],
            names=["document", "text"],
        )

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").select(emit_document(vane.col("value")).alias("payload"))

    assert result.project("payload.document.url, payload.text").fetchone() == (
        "memory://batch-expression",
        r"\xC3\xA9",
    )


def test_row_preserving_batch_file_output_fuses_heterogeneous_pieces():
    import pyarrow as pa

    output_type = vane.type("STRUCT(document FILE, text VARCHAR)")

    @vane.func.batch(return_dtype=output_type, batch_size=1)
    def emit_document(values):
        identifier = values[0].as_py()
        text = pa.array([b"\xc3\xa9"], type=pa.binary()) if identifier == 0 else pa.array(["second"])
        return pa.StructArray.from_arrays(
            [
                pa.array(
                    [_file_record(url=f"memory://row-preserving-{identifier}")],
                    type=_file_arrow_type(),
                ),
                text,
            ],
            names=["document", "text"],
        )

    connection = vane.connect()
    source = connection.sql("SELECT i FROM range(2) AS t(i)")
    result = source.select(emit_document(vane.col("i")).alias("payload"))

    assert result.project("payload.document.url, payload.text").fetchall() == [
        ("memory://row-preserving-0", r"\xC3\xA9"),
        ("memory://row-preserving-1", "second"),
    ]


def test_row_actor_rejects_file_inputs_and_outputs():
    import cloudpickle

    from vane.execution._udf_runtime import UDFExecutor

    class ReadFile:
        def __call__(self, value):
            return value.url

    with pytest.raises(vane.InvalidInputException, match=r"vane\.cls.*FILE outputs"):
        vane.cls(ReadFile, actor_number=1, return_dtype=vane.file_type())

    row_class = vane.cls(ReadFile, actor_number=1, return_dtype="VARCHAR")()
    connection = vane.connect()
    with pytest.raises(vane.InvalidInputException, match=r"vane\.cls.*FILE inputs"):
        vane.attach_function(
            row_class,
            connection=connection,
            alias="read_file_actor",
            parameters=[vane.file_type()],
        )

    actor = row_class.actor_class(["value"])
    with pytest.raises(ValueError, match=r"vane\.cls.*FILE inputs"):
        UDFExecutor(
            {
                "function_pickle": cloudpickle.dumps(actor),
                "call_mode": "map_batches_rows",
                "execution_backend": "subprocess_actor",
                "actor_number": 1,
                "input_types": ["FILE"],
                "output_schema": [{"name": "value", "kind": "duckdb_type", "type": "VARCHAR"}],
            }
        )


def test_file_arrow_validation_does_not_materialize_non_file_struct_siblings():
    import pyarrow as pa

    from vane.execution._udf_runtime import _build_valid_mask
    from vane.execution.udf_file_contract import FileUDFContract, validate_file_arrow_array

    class ExplodingScalar(pa.ExtensionScalar):
        def as_py(self, *args, **kwargs):
            raise AssertionError("unrelated field was materialized")

    class ExplodingType(pa.ExtensionType):
        def __init__(self):
            super().__init__(pa.binary(), "vane.test.file_validation_exploding")

        def __arrow_ext_serialize__(self):
            return b""

        @classmethod
        def __arrow_ext_deserialize__(cls, storage_type, serialized):
            return cls()

        def __arrow_ext_scalar_class__(self):
            return ExplodingScalar

    payload = pa.ExtensionArray.from_storage(ExplodingType(), pa.array([b"large-unrelated-payload"]))
    array = pa.StructArray.from_arrays(
        [pa.array([_file_record()], type=_file_arrow_type()), payload],
        names=["document", "payload"],
    )

    validate_file_arrow_array(
        array,
        vane.type("STRUCT(document FILE, payload BLOB)"),
        boundary="test output",
    )
    assert _build_valid_mask(pa.table({"value": array})) == [True]
    contract = FileUDFContract.from_payload(
        {
            "udf_name": "no-materialization",
            "output_schema": [{"name": "value", "kind": "duckdb_type", "type": "STRUCT(document FILE, payload BLOB)"}],
        }
    )
    normalized = contract.normalize_output_table(pa.table({"value": array}))
    assert normalized.column("value").type.field("payload").type == pa.binary()


def test_file_composite_arrow_fields_match_case_insensitively():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "case-insensitive",
            "output_schema": [{"name": "payload", "kind": "duckdb_type", "type": 'STRUCT("Document" FILE)'}],
        }
    )
    lower_case = pa.StructArray.from_arrays(
        [pa.array([_file_record()], type=_file_arrow_type())],
        names=["document"],
    )

    normalized = contract.normalize_output_table(pa.table({"payload": lower_case}))

    assert normalized.column("payload").type.field(0).name == "Document"
    assert normalized.column("payload").to_pylist() == [{"Document": _file_record()}]


def test_file_composite_normalization_masks_children_beneath_null_parents():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "null-parent",
            "output_schema": [{"name": "payload", "kind": "duckdb_type", "type": "STRUCT(document FILE, id BIGINT)"}],
        }
    )
    hidden_values = pa.StructArray.from_arrays(
        [
            pa.array([_file_record()], type=_file_arrow_type()),
            pa.array(["invalid"]),
        ],
        names=["document", "id"],
        mask=pa.array([True]),
    )

    normalized = contract.normalize_output_table(pa.table({"payload": hidden_values}))

    assert normalized.column("payload").type.field("id").type == pa.int64()
    assert normalized.column("payload").to_pylist() == [None]


def test_file_composite_native_values_match_fields_case_insensitively():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    logical_type = 'STRUCT("Document" FILE, "Label" VARCHAR)'
    input_contract = FileUDFContract.from_payload(
        {
            "udf_name": "case-insensitive-input",
            "input_types": [logical_type],
        }
    )
    lower_case = pa.StructArray.from_arrays(
        [
            pa.array([_file_record()], type=_file_arrow_type()),
            pa.array(["report"]),
        ],
        names=["document", "label"],
    )

    columns = input_contract.materialize_scalar_inputs(pa.table({"payload": lower_case}))

    assert columns == [[{"Document": vane.File(**_file_record()), "Label": "report"}]]

    output_contract = FileUDFContract.from_payload(
        {
            "udf_name": "case-insensitive-output",
            "output_schema": [{"name": "payload", "kind": "duckdb_type", "type": logical_type}],
        }
    )
    output = output_contract.scalar_outputs_to_array([{"document": vane.File(**_file_record()), "label": "report"}])

    assert output.type.names == ["Document", "Label"]
    assert output.to_pylist() == [{"Document": _file_record(), "Label": "report"}]


def test_batch_file_udf_types_all_null_output_as_file():
    import pyarrow as pa

    @vane.func.batch(return_dtype=vane.file_type())
    def nulls(values):
        return pa.array([None] * len(values))

    result = nulls(pa.array([1, 2], type=pa.int32()))

    assert result.type.equals(_file_arrow_type())
    assert result.to_pylist() == [None, None]


def test_batch_file_udf_reorders_named_struct_output_before_cast():
    import pyarrow as pa

    output_type = vane.type("STRUCT(document FILE, id INTEGER)")

    @vane.func.batch(return_dtype=output_type)
    def reordered(values):
        return pa.StructArray.from_arrays(
            [
                pa.array(range(len(values)), type=pa.int32()),
                pa.array([_file_record()] * len(values), type=_file_arrow_type()),
            ],
            names=["id", "document"],
        )

    result = reordered(pa.array([1, 2], type=pa.int32()))

    assert result.type.names == ["document", "id"]
    assert result.to_pylist() == [
        {"document": _file_record(), "id": 0},
        {"document": _file_record(), "id": 1},
    ]


def test_batch_file_udf_supports_time_ns_sibling():
    import pyarrow as pa

    output_type = vane.type("STRUCT(document FILE, precise TIME_NS)")
    precise = time(1, 2, 3, 456789)

    @vane.func.batch(return_dtype=output_type)
    def build_document(values):
        return pa.StructArray.from_arrays(
            [
                pa.array([_file_record()] * len(values), type=_file_arrow_type()),
                pa.array([precise] * len(values), type=pa.time64("ns")),
            ],
            names=["document", "precise"],
        )

    result = build_document(pa.array([1], type=pa.int32()))

    assert result.type.field("precise").type == pa.time64("ns")
    assert result.to_pylist() == [{"document": _file_record(), "precise": precise}]


def test_batch_file_udf_supports_bit_sibling():
    import pyarrow as pa

    output_type = vane.type("STRUCT(document FILE, flags BIT)")

    @vane.func.batch(return_dtype=output_type)
    def build_document(values):
        file_type = pa.struct(
            [
                pa.field("url", pa.string()),
                pa.field("content_type", pa.string()),
                pa.field("position", pa.int64()),
                pa.field("size", pa.int64()),
                pa.field("checksum", pa.string()),
            ]
        )
        document = {
            "url": "memory://bit-sibling",
            "content_type": None,
            "position": None,
            "size": None,
            "checksum": None,
        }
        return pa.StructArray.from_arrays(
            [
                pa.array([document] * len(values), type=file_type),
                pa.array(["101001"] * len(values), type=pa.string()),
            ],
            names=["document", "flags"],
        )

    bit_type = build_document.return_arrow_dtype.field("flags").type
    assert bit_type.extension_name == "arrow.opaque"
    assert bit_type.type_name == "bit"
    assert bit_type.vendor_name == "DuckDB"
    assert bit_type.storage_type == pa.binary()

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))

    assert result.project("payload.document.url, payload.flags::VARCHAR").fetchone() == (
        "memory://bit-sibling",
        "101001",
    )


def test_batch_file_udf_supports_enum_sibling():
    import pyarrow as pa

    output_type = vane.type("STRUCT(document FILE, status ENUM('open', 'closed'))")

    @vane.func.batch(return_dtype=output_type)
    def build_document(values):
        return pa.StructArray.from_arrays(
            [
                pa.array([_file_record()] * len(values), type=_file_arrow_type()),
                pa.array(["open"] * len(values), type=pa.string()),
            ],
            names=["document", "status"],
        )

    assert build_document.return_arrow_dtype.field("status").type == pa.string()

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))

    assert result.project("payload.document.url, payload.status::VARCHAR").fetchone() == (
        "memory://udf",
        "open",
    )


def test_batch_file_udf_supports_non_file_union_sibling():
    import pyarrow as pa

    output_type = vane.type("STRUCT(document FILE, choice UNION(s VARCHAR, i BIGINT))")

    @vane.func.batch(return_dtype=output_type)
    def build_document(values):
        return values

    choice_type = build_document.return_arrow_dtype.field("choice").type
    assert pa.types.is_union(choice_type)
    assert choice_type.mode == "sparse"
    assert [field.name for field in choice_type] == ["s", "i"]
    assert [field.type for field in choice_type] == [pa.string(), pa.int64()]


def test_file_arrow_non_file_union_sibling_rejects_nonordinal_type_codes():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "nonordinal-union-sibling",
            "method_return_type": "STRUCT(document FILE, choice UNION(s VARCHAR, i BIGINT))",
        }
    )
    choice = pa.UnionArray.from_sparse(
        pa.array([1], type=pa.int8()),
        [pa.array(["selected"]), pa.array([None], type=pa.int64())],
        field_names=["s", "i"],
        type_codes=[1, 0],
    )
    value = pa.StructArray.from_arrays(
        [pa.array([_file_record()], type=_file_arrow_type()), choice],
        names=["document", "choice"],
    )

    with pytest.raises(vane.InvalidInputException, match="type codes must match child ordinals"):
        contract.normalize_output_table(pa.table({"payload": value}))


def test_batch_file_udf_supports_sqlnull_sibling():
    import pyarrow as pa

    output_type = vane.struct_type(
        {
            "document": vane.file_type(),
            "missing": vane.sqltypes.SQLNULL,
        }
    )

    @vane.func.batch(return_dtype=output_type)
    def build_document(values):
        return pa.StructArray.from_arrays(
            [
                pa.array([_file_record()] * len(values), type=_file_arrow_type()),
                pa.nulls(len(values)),
            ],
            names=["document", "missing"],
        )

    assert build_document.return_arrow_dtype.field("missing").type == pa.null()
    assert build_document(pa.array([1], type=pa.int32())).to_pylist() == [{"document": _file_record(), "missing": None}]

    connection = vane.connect()
    result = connection.sql("SELECT 1 AS value").select(build_document(vane.col("value")).alias("payload"))
    assert result.project("payload.document.url, payload.missing").fetchone() == ("memory://udf", None)


def test_batch_file_udf_preserves_bit_identity_storage():
    import pyarrow as pa

    output_type = vane.type("STRUCT(document FILE, flags BIT)")

    @vane.func.batch(return_dtype=output_type)
    def identity(values):
        flags_type = values.type.field("flags").type
        assert flags_type.extension_name == "arrow.opaque"
        assert flags_type.type_name == "bit"
        assert flags_type.vendor_name == "DuckDB"
        assert flags_type.storage_type == pa.binary()
        return values

    connection = vane.connect()
    source = connection.sql(
        """
        SELECT struct_pack(
            document := file('memory://bit-identity', NULL, NULL, NULL, NULL),
            flags := '0101011'::BIT
        ) AS payload
        """
    )
    result = source.select(identity(vane.col("payload")).alias("payload"))

    assert result.project("payload.document.url, payload.flags::VARCHAR").fetchone() == (
        "memory://bit-identity",
        "0101011",
    )


def test_scalar_file_udf_materializes_bit_sibling_as_text():
    output_type = vane.type("STRUCT(document FILE, flags BIT)")

    @vane.func(return_dtype=output_type)
    def identity(value):
        assert isinstance(value["document"], vane.File)
        assert value["flags"] == "0101011"
        return value

    connection = vane.connect()
    source = connection.sql(
        """
        SELECT struct_pack(
            document := file('memory://scalar-bit-identity', NULL, NULL, NULL, NULL),
            flags := '0101011'::BIT
        ) AS payload
        """
    )
    result = source.select(identity(vane.col("payload")).alias("payload"))

    assert result.project("payload.document.url, payload.flags::VARCHAR").fetchone() == (
        "memory://scalar-bit-identity",
        "0101011",
    )


def test_file_contract_marks_sliced_nested_bit_inputs():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract, _map_array_from_offsets

    bit_storage = pa.array([b"\x01\xab"] * 4, type=pa.binary())
    flags = pa.ListArray.from_arrays(
        pa.array([0, 2, 4], type=pa.int32()),
        bit_storage,
        type=pa.list_(pa.field("flag", pa.binary(), nullable=False, metadata={b"flag-item": b"preserved"})),
    )
    fixed = pa.FixedSizeListArray.from_arrays(
        bit_storage,
        type=pa.list_(
            pa.field("fixed_flag", pa.binary(), nullable=False, metadata={b"fixed-item": b"preserved"}),
            list_size=2,
        ),
    )
    lookup_type = pa.map_(
        pa.field("lookup_key", pa.string(), nullable=False, metadata={b"map-key": b"preserved"}),
        pa.field("lookup_value", pa.binary(), metadata={b"map-value": b"preserved"}),
        keys_sorted=True,
    )
    # pa.array() discards MAP child metadata before BIT annotation can see it.
    lookup = _map_array_from_offsets(
        [0, 1, 2],
        pa.array(["first", "second"]),
        bit_storage.slice(0, 2),
        mask=pa.array([False, False]),
        map_type=lookup_type,
    )
    assert lookup.type.key_field.metadata == {b"map-key": b"preserved"}
    assert lookup.type.item_field.metadata == {b"map-value": b"preserved"}
    payload_fields = [
        pa.field("document", _file_arrow_type()),
        pa.field("flags", flags.type, nullable=False, metadata={b"flags": b"preserved"}),
        pa.field("fixed", fixed.type, nullable=False, metadata={b"fixed": b"preserved"}),
        pa.field("lookup", lookup.type, nullable=False, metadata={b"lookup": b"preserved"}),
    ]
    payload = pa.StructArray.from_arrays(
        [
            pa.array(
                [_file_record(url="memory://first"), _file_record(url="memory://second")], type=_file_arrow_type()
            ),
            flags,
            fixed,
            lookup,
        ],
        fields=payload_fields,
    )
    contract = FileUDFContract.from_payload(
        {
            "udf_name": "nested-bit-input",
            "input_types": ["STRUCT(document FILE, flags BIT[], fixed BIT[2], lookup MAP(VARCHAR, BIT))"],
        }
    )

    sliced = pa.table({"payload": payload}).slice(1, 1)
    prepared = contract.prepare_input_table(sliced).column("payload").chunk(0)

    for child in (prepared.field("flags").values, prepared.field("fixed").values, prepared.field("lookup").items):
        assert child.type.extension_name == "arrow.opaque"
        assert child.type.type_name == "bit"
        assert child.type.vendor_name == "DuckDB"
    for name in ("flags", "fixed", "lookup"):
        field = prepared.type.field(name)
        assert not field.nullable
        assert field.metadata == {name.encode(): b"preserved"}
    assert prepared.type.field("flags").type.value_field.metadata == {b"flag-item": b"preserved"}
    assert prepared.type.field("fixed").type.value_field.metadata == {b"fixed-item": b"preserved"}
    prepared_map = prepared.type.field("lookup").type
    assert prepared_map.keys_sorted
    assert prepared_map.key_field.metadata == {b"map-key": b"preserved"}
    assert prepared_map.item_field.metadata == {b"map-value": b"preserved"}
    assert contract.materialize_scalar_inputs(sliced) == [
        [
            {
                "document": vane.File(**_file_record(url="memory://second")),
                "flags": ["0101011", "0101011"],
                "fixed": ("0101011", "0101011"),
                "lookup": {"second": "0101011"},
            }
        ]
    ]


def test_map_batches_marks_top_level_bit_sibling_input():
    import pyarrow as pa

    def identity(table):
        flags_type = table.column("flags").type
        assert flags_type.extension_name == "arrow.opaque"
        assert flags_type.type_name == "bit"
        assert flags_type.vendor_name == "DuckDB"
        assert flags_type.storage_type == pa.binary()
        return table

    connection = vane.connect()
    source = connection.sql(
        """
        SELECT
            file('memory://batch-bit-identity', NULL, NULL, NULL, NULL) AS document,
            '0101011'::BIT AS flags
        """
    )
    result = source.map_batches(
        identity,
        schema={"document": vane.file_type(), "flags": vane.sqltypes.BIT},
        execution_backend="subprocess_task",
    )

    assert result.project("document.url, flags::VARCHAR").fetchone() == (
        "memory://batch-bit-identity",
        "0101011",
    )


def test_batch_file_output_preserves_bit_only_input_identity():
    import pyarrow as pa

    output_type = vane.type("STRUCT(document FILE, flags BIT)")

    @vane.func.batch(return_dtype=output_type)
    def attach_document(flags):
        assert flags.type.extension_name == "arrow.opaque"
        assert flags.type.type_name == "bit"
        assert flags.type.vendor_name == "DuckDB"
        documents = pa.array(
            [_file_record(url="memory://bit-only-input")] * len(flags),
            type=_file_arrow_type(),
        )
        return pa.StructArray.from_arrays([documents, flags], names=["document", "flags"])

    connection = vane.connect()
    source = connection.sql("SELECT '0101011'::BIT AS flags")
    result = source.select(attach_document(vane.col("flags")).alias("payload"))

    assert result.project("payload.document.url, payload.flags::VARCHAR").fetchone() == (
        "memory://bit-only-input",
        "0101011",
    )


def test_batch_file_output_uses_resolved_contract_for_connection_local_aliases():
    import pyarrow as pa

    output_type = vane.type("STRUCT(document FILE, flags BIT)")

    @vane.func.batch(return_dtype=output_type)
    def attach_document(flags, _moods):
        assert flags.type.extension_name == "arrow.opaque"
        assert flags.type.type_name == "bit"
        documents = pa.array(
            [_file_record(url="memory://local-aliases")] * len(flags),
            type=_file_arrow_type(),
        )
        return pa.StructArray.from_arrays([documents, flags], names=["document", "flags"])

    connection = vane.connect()
    connection.execute("CREATE TYPE local_flags AS BIT")
    connection.execute("CREATE TYPE mood AS ENUM ('happy', 'sad')")
    source = connection.sql("SELECT '0101011'::local_flags AS flags, 'happy'::mood AS mood")
    result = source.select(attach_document(vane.col("flags"), vane.col("mood")).alias("payload"))

    assert result.project("payload.document.url, payload.flags::VARCHAR").fetchone() == (
        "memory://local-aliases",
        "0101011",
    )


def test_file_output_contract_uses_resolved_type_instead_of_serialized_alias():
    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "resolved-output-alias",
            "method_return_type": "local_document_payload",
            "output_contract_types": ["STRUCT(document FILE, flags BIT)"],
        }
    )

    assert contract.has_file_outputs
    assert [(name, str(dtype)) for name, dtype in contract.output_types[0].children] == [
        ("document", "FILE"),
        ("flags", "BIT"),
    ]


def test_file_input_contract_precedes_catalog_local_alias_text():
    from vane.execution.udf_file_contract import FileUDFContract

    contract = FileUDFContract.from_payload(
        {
            "udf_name": "resolved-input-alias",
            "input_types": ['"source file"'],
            "input_contract_types": ["VARCHAR"],
            "method_return_type": "FILE",
            "output_contract_types": ["FILE"],
        }
    )

    assert not contract.has_file_inputs
    assert contract.has_file_outputs
    assert str(contract.input_types[0]) == "VARCHAR"


def test_scalar_file_output_supports_connection_local_file_bearing_alias():
    connection = vane.connect()
    connection.execute("CREATE TYPE local_document_payload AS STRUCT(document FILE, flags BIT)")
    output_type = vane.type("local_document_payload", connection=connection)

    @vane.func(return_dtype=output_type)
    def attach_document(_value):
        return {
            "document": vane.File("memory://local-output-alias"),
            "flags": "0101011",
        }

    result = connection.sql("SELECT 1 AS value").select(attach_document(vane.col("value")).alias("payload"))

    assert result.project("payload.document.url, payload.flags::VARCHAR").fetchone() == (
        "memory://local-output-alias",
        "0101011",
    )


def test_map_batches_file_output_supports_connection_local_sibling_alias():
    import pyarrow as pa

    def attach_document(table):
        return pa.table(
            {
                "document": pa.array(
                    [_file_record(url="memory://local-output-sibling")] * table.num_rows,
                    type=_file_arrow_type(),
                ),
                "mood": ["happy"] * table.num_rows,
            }
        )

    connection = vane.connect()
    connection.execute("CREATE TYPE mood AS ENUM ('happy', 'sad')")
    mood_type = vane.type("mood", connection=connection)
    result = connection.sql("SELECT 1 AS value").map_batches(
        attach_document,
        schema={"document": vane.file_type(), "mood": mood_type},
        execution_backend="subprocess_task",
    )

    assert result.project("document.url, mood::VARCHAR").fetchone() == (
        "memory://local-output-sibling",
        "happy",
    )

    empty_result = connection.sql("SELECT 1 AS value WHERE FALSE").map_batches(
        attach_document,
        schema={"document": vane.file_type(), "mood": mood_type},
        execution_backend="subprocess_task",
    )
    assert empty_result.fetchall() == []


def test_scalar_file_output_leaves_union_bit_input_unmaterialized():
    @vane.func(return_dtype=vane.file_type())
    def build_document(_value):
        return vane.File("memory://union-bit-input")

    connection = vane.connect()
    source = connection.sql("SELECT union_value(bits := '0101011'::BIT) AS value")
    result = source.select(build_document(vane.col("value")).alias("document"))

    assert result.fetchone() == (vane.File("memory://union-bit-input"),)


def test_scalar_file_output_materializes_bit_only_input_as_text():
    output_type = vane.type("STRUCT(document FILE, flags BIT)")

    @vane.func(return_dtype=output_type)
    def attach_document(flags):
        assert flags == "0101011"
        return {"document": vane.File("memory://scalar-bit-only-input"), "flags": flags}

    connection = vane.connect()
    source = connection.sql("SELECT '0101011'::BIT AS flags")
    result = source.select(attach_document(vane.col("flags")).alias("payload"))

    assert result.project("payload.document.url, payload.flags::VARCHAR").fetchone() == (
        "memory://scalar-bit-only-input",
        "0101011",
    )


def test_flat_map_file_output_materializes_bit_only_input_as_text():
    output_type = vane.type("STRUCT(document FILE, flags BIT)")

    def attach_document(row):
        assert row["flags"] == "0101011"
        return {
            "payload": {
                "document": vane.File("memory://flat-map-bit-only-input"),
                "flags": row["flags"],
            }
        }

    connection = vane.connect()
    source = connection.sql("SELECT '0101011'::BIT AS flags")
    result = source.flat_map(
        attach_document,
        schema={"payload": output_type},
        execution_backend="subprocess_task",
    )

    assert result.project("payload.document.url, payload.flags::VARCHAR").fetchone() == (
        "memory://flat-map-bit-only-input",
        "0101011",
    )


def test_batch_file_udf_preserves_blob_to_bit_cast_semantics():
    import pyarrow as pa

    output_type = vane.type("STRUCT(document FILE, flags BIT)")

    @vane.func.batch(return_dtype=output_type)
    def build_document(values):
        return pa.StructArray.from_arrays(
            [
                pa.array([_file_record()] * len(values), type=_file_arrow_type()),
                pa.array([b"\x01\xab"] * len(values), type=pa.binary()),
            ],
            names=["document", "flags"],
        )

    result = build_document(pa.array([1], type=pa.int32()))

    assert result.type.field("flags").type == pa.binary()
    connection = vane.connect()
    connection.register("file_bit_blob", pa.table({"payload": result}))
    assert connection.execute("SELECT payload.flags::BIT::VARCHAR FROM file_bit_blob").fetchone() == (
        "0000000110101011",
    )


def test_batch_file_udf_preserves_bit_to_blob_cast_semantics():
    import pyarrow as pa

    output_type = vane.type("STRUCT(document FILE, data BLOB)")

    @vane.func.batch(return_dtype=output_type)
    def attach_document(flags):
        assert flags.type.extension_name == "arrow.opaque"
        documents = pa.array(
            [_file_record(url="memory://bit-to-blob")] * len(flags),
            type=_file_arrow_type(),
        )
        return pa.StructArray.from_arrays([documents, flags], names=["document", "data"])

    connection = vane.connect()
    source = connection.sql("SELECT '0101011'::BIT AS flags")
    result = source.select(attach_document(vane.col("flags")).alias("payload"))

    assert result.project("payload.document.url, payload.data").fetchone() == (
        "memory://bit-to-blob",
        b"+",
    )


def test_batch_file_udf_uses_opaque_compat_without_pyarrow_opaque(monkeypatch):
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    connection = vane.connect()
    bit_storage = connection.execute("SELECT '0101011'::BIT AS flags").to_arrow_table().column("flags").chunk(0)
    monkeypatch.setattr(pa, "opaque", None)

    output_type = vane.type("STRUCT(document FILE, flags BIT)")

    @vane.func.batch(return_dtype=output_type)
    def identity(values):
        return values

    payload = pa.StructArray.from_arrays(
        [
            pa.array([_file_record()], type=_file_arrow_type()),
            bit_storage,
        ],
        names=["document", "flags"],
    )
    contract = FileUDFContract.from_payload({"input_types": [str(output_type)]})
    prepared = contract.prepare_input_table(pa.table({"payload": payload})).column("payload").chunk(0)
    result = identity(prepared)

    flags_type = identity.return_arrow_dtype.field("flags").type
    assert flags_type.extension_name == "arrow.opaque"
    assert flags_type.type_name == "bit"
    assert flags_type.vendor_name == "DuckDB"
    assert result.field("flags").type.equals(flags_type)
    assert result.field("flags").storage.to_pylist() == bit_storage.to_pylist()
    connection.register("file_bit_fallback", pa.table({"payload": result}))
    assert connection.execute("SELECT payload.flags::BIT::VARCHAR FROM file_bit_fallback").fetchone() == ("0101011",)


def test_file_output_normalization_preserves_full_intervals():
    import pyarrow as pa

    from vane.execution.udf_file_contract import FileUDFContract

    interval = pa.MonthDayNano((2, 3, 4_000))
    contract = FileUDFContract.from_payload(
        {
            "udf_name": "interval",
            "output_schema": [
                {"name": "document", "kind": "duckdb_type", "type": "FILE"},
                {"name": "span", "kind": "duckdb_type", "type": "INTERVAL"},
            ],
        }
    )
    table = pa.table(
        {
            "document": pa.array([_file_record()], type=_file_arrow_type()),
            "span": pa.array([interval], type=pa.month_day_nano_interval()),
        }
    )

    normalized = contract.normalize_output_table(table)

    assert normalized.column("span").type == pa.month_day_nano_interval()
    assert normalized.column("span").to_pylist() == [interval]


def test_map_batches_file_schema_preserves_calendar_interval_semantics():
    def identity(table):
        return table

    connection = vane.connect()
    source = connection.sql(
        "SELECT file('memory://interval', NULL, NULL, NULL, NULL) AS document, INTERVAL '1 month' AS span"
    )
    result = source.map_batches(
        identity,
        schema={"document": vane.file_type(), "span": vane.sqltypes.INTERVAL},
        execution_backend="subprocess_task",
    )

    assert result.project("DATE '2000-01-31' + span AS shifted").fetchone() == (datetime(2000, 2, 29),)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (_file_record(url=None), r"File\.url must be str"),
        (_file_record(url="memory://bad\0url"), "NUL"),
        (_file_record(position=0, size=None), "position and size"),
        (_file_record(position=-1, size=1), "non-negative"),
        (_file_record(position=2**63 - 1, size=1), "exceeds BIGINT"),
        (_file_record(checksum="invalid"), "<algorithm>:<digest>"),
    ],
)
def test_batch_file_udf_validates_every_output(record, message):
    import pyarrow as pa

    @vane.func.batch(return_dtype=vane.file_type())
    def invalid_output(values):
        return pa.array([record] * len(values), type=_file_arrow_type())

    with pytest.raises(vane.InvalidInputException, match=message):
        invalid_output(pa.array([1], type=pa.int32()))


def test_file_udf_validation_errors_do_not_expose_file_values():
    import pyarrow as pa

    sentinel = "sensitive-file-token-9274"
    record = _file_record(url=f"memory://{sentinel}", position=-1)

    @vane.func.batch(return_dtype=vane.file_type())
    def invalid_output(values):
        return pa.array([record] * len(values), type=_file_arrow_type())

    with pytest.raises(vane.InvalidInputException, match="non-negative") as captured:
        invalid_output(pa.array([1], type=pa.int32()))
    assert sentinel not in str(captured.value)


def test_batch_file_udf_validates_worker_output_values():
    import pyarrow as pa

    record = _file_record(position=-1)
    file_type = _file_arrow_type()

    @vane.func.batch(return_dtype=vane.file_type())
    def invalid_output(values):
        return pa.array([record] * len(values), type=file_type)

    connection = vane.connect()
    source = connection.sql("SELECT i FROM range(2) AS t(i)")

    with pytest.raises(Exception, match="non-negative"):
        source.select(invalid_output(vane.col("i"))).fetchall()


@pytest.mark.parametrize("shape", ["wrong_type", "missing_field", "extra_field", "blob"])
def test_batch_file_udf_rejects_castable_wrong_arrow_shape(shape):
    import pyarrow as pa

    if shape == "blob":
        wrong_type = pa.binary()
        record = b"not-a-file"
    else:
        file_type = _file_arrow_type()
        fields = [file_type.field(index) for index in range(len(file_type))]
        record = _file_record()
        if shape == "wrong_type":
            fields[2] = pa.field("position", pa.int32())
            fields[3] = pa.field("size", pa.int32())
        elif shape == "missing_field":
            fields.pop()
        else:
            fields.append(pa.field("version", pa.string()))
            record["version"] = "v1"
        wrong_type = pa.struct(fields)

    @vane.func.batch(return_dtype=vane.file_type())
    def wrong_shape(values):
        return pa.array([record] * len(values), type=wrong_type)

    with pytest.raises(vane.InvalidInputException, match=r"canonical five-field Arrow STRUCT"):
        wrong_shape(pa.array([1], type=pa.int32()))


def test_registered_scalar_file_udf_preserves_type_and_nulls():
    @vane.func(return_dtype=vane.file_type())
    def identity(value):
        assert isinstance(value, vane.File)
        return value

    connection = vane.connect()
    vane.attach_function(identity, connection=connection, alias="identity_file_sql", parameters=[vane.file_type()])

    rows = connection.sql(
        """
        SELECT typeof(identity_file_sql(value)), identity_file_sql(value)
        FROM (
            SELECT 0 AS id, file('memory://sql', NULL, NULL, NULL, NULL) AS value
            UNION ALL
            SELECT 1 AS id, NULL::FILE
        )
        ORDER BY id
        """
    ).fetchall()

    assert rows == [("FILE", vane.File("memory://sql")), ("FILE", None)]


def test_registered_batch_file_udf_preserves_type():
    @vane.func.batch(return_dtype=vane.file_type())
    def identity(values):
        return values

    connection = vane.connect()
    vane.attach_function(identity, connection=connection, alias="identity_file_batch_sql", parameters=["FILE"])

    result = connection.sql(
        "SELECT identity_file_batch_sql(file('memory://batch-sql', NULL, NULL, NULL, NULL)) AS value"
    )

    assert result.types[0].is_file()
    assert result.fetchone() == (vane.File("memory://batch-sql"),)


def test_registered_media_file_udfs_preserve_exact_types_and_reject_mismatches():
    image_type = vane.file_type(vane.MediaType.image())
    audio_type = vane.file_type(vane.MediaType.audio())

    @vane.func(return_dtype=image_type)
    def image_identity(value):
        assert type(value) is vane.ImageFile
        return value

    @vane.func.batch(return_dtype=audio_type)
    def audio_identity(values):
        return values

    connection = vane.connect()
    vane.attach_function(image_identity, connection=connection, alias="identity_image_sql", parameters=[image_type])
    vane.attach_function(audio_identity, connection=connection, alias="identity_audio_sql", parameters=[audio_type])

    row = connection.sql(
        """
        SELECT
            typeof(identity_image_sql(image_file('memory://image'))),
            identity_image_sql(image_file('memory://image')),
            typeof(identity_audio_sql(audio_file('memory://audio'))),
            identity_audio_sql(audio_file('memory://audio'))
        """
    ).fetchone()

    assert row == (
        "IMAGEFILE",
        vane.ImageFile("memory://image"),
        "AUDIOFILE",
        vane.AudioFile("memory://audio"),
    )
    with pytest.raises(vane.BinderException, match=r"No function matches"):
        connection.sql("SELECT identity_image_sql(audio_file('memory://audio'))")


@pytest.mark.parametrize(
    "argument",
    [
        """
        struct_pack(
            url := 'memory://struct',
            content_type := NULL::VARCHAR,
            "position" := NULL::BIGINT,
            size := NULL::BIGINT,
            checksum := NULL::VARCHAR
        )
        """,
        "'memory://blob'::BLOB",
    ],
)
def test_registered_file_udf_rejects_struct_and_blob_arguments_at_bind(argument):
    @vane.func(return_dtype=vane.file_type())
    def identity(value):
        return value

    connection = vane.connect()
    vane.attach_function(identity, connection=connection, alias="strict_file_sql", parameters=["FILE"])

    with pytest.raises(vane.BinderException, match=r"No function matches"):
        connection.sql(f"SELECT strict_file_sql({argument})")


def test_same_shaped_generic_struct_udf_remains_unaffected():
    @vane.func(return_dtype="VARCHAR")
    def extract_url(value):
        assert isinstance(value, dict)
        return value["url"]

    connection = vane.connect()
    source = connection.sql(
        """
        SELECT struct_pack(
            url := 'memory://plain-struct',
            content_type := NULL::VARCHAR,
            "position" := NULL::BIGINT,
            size := NULL::BIGINT,
            checksum := NULL::VARCHAR
        ) AS value
        """
    )

    assert source.select(extract_url(vane.col("value"))).fetchone() == ("memory://plain-struct",)


@pytest.mark.parametrize("mode", ["scalar", "batch"])
def test_empty_file_udf_relation_retains_declared_type(mode):
    @vane.func(return_dtype=vane.file_type())
    def scalar_identity(value):
        return value

    @vane.func.batch(return_dtype=vane.file_type())
    def batch_identity(value):
        return value

    identity = scalar_identity if mode == "scalar" else batch_identity

    connection = vane.connect()
    source = connection.sql("SELECT file('memory://empty', NULL, NULL, NULL, NULL) AS value WHERE FALSE")
    result = source.select(identity(vane.col("value")).alias("value"))

    assert result.types[0].is_file()
    assert result.fetchall() == []
