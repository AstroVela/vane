# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

import vane


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


def test_file_udf_rejects_invalid_input_before_user_code(tmp_path):
    marker = tmp_path / "called"

    @vane.func(return_dtype="INTEGER")
    def observe(_value):
        Path(marker).write_text("called", encoding="utf-8")
        return 1

    connection = vane.connect()
    connection.execute("CREATE TABLE invalid_file_udf_input(value FILE)")
    connection.execute(
        """
        INSERT INTO invalid_file_udf_input
        SELECT struct_pack(
            url := NULL::VARCHAR,
            content_type := NULL::VARCHAR,
            "position" := NULL::BIGINT,
            size := NULL::BIGINT,
            checksum := NULL::VARCHAR
        )
        """
    )

    with pytest.raises(Exception, match=r"invalid FILE value"):
        connection.table("invalid_file_udf_input").select(observe(vane.col("value"))).fetchall()
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


def test_batch_file_udf_rejects_invalid_input_before_user_code(tmp_path):
    marker = tmp_path / "called"

    @vane.func.batch(return_dtype="INTEGER")
    def observe(values):
        Path(marker).write_text("called", encoding="utf-8")
        import pyarrow as pa

        return pa.array([1] * len(values), type=pa.int32())

    connection = vane.connect()
    connection.execute("CREATE TABLE invalid_batch_file_udf_input(value FILE)")
    connection.execute(
        """
        INSERT INTO invalid_batch_file_udf_input
        SELECT struct_pack(
            url := NULL::VARCHAR,
            content_type := NULL::VARCHAR,
            "position" := NULL::BIGINT,
            size := NULL::BIGINT,
            checksum := NULL::VARCHAR
        )
        """
    )

    with pytest.raises(Exception, match=r"invalid FILE value"):
        connection.table("invalid_batch_file_udf_input").select(observe(vane.col("value"))).fetchall()
    assert not marker.exists()


@pytest.mark.parametrize("mode", ["map_batches", "flat_map"])
def test_relation_table_file_udf_rejects_invalid_input_before_user_code(tmp_path, mode):
    marker = tmp_path / "called"

    def observe_batch(table):
        Path(marker).write_text("called", encoding="utf-8")
        import pyarrow as pa

        return pa.table({"result": [1] * table.num_rows})

    def observe_row(_row):
        Path(marker).write_text("called", encoding="utf-8")
        return {"result": 1}

    connection = vane.connect()
    connection.execute("CREATE TABLE invalid_relation_file_udf_input(value FILE)")
    connection.execute(
        """
        INSERT INTO invalid_relation_file_udf_input
        SELECT struct_pack(
            url := NULL::VARCHAR,
            content_type := NULL::VARCHAR,
            "position" := NULL::BIGINT,
            size := NULL::BIGINT,
            checksum := NULL::VARCHAR
        )
        """
    )
    source = connection.table("invalid_relation_file_udf_input")
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

    with pytest.raises(Exception, match=r"invalid FILE value"):
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
