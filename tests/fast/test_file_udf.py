# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

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


def test_batch_file_udf_types_all_null_output_as_file():
    import pyarrow as pa

    @vane.func.batch(return_dtype=vane.file_type())
    def nulls(values):
        return pa.array([None] * len(values))

    result = nulls(pa.array([1, 2], type=pa.int32()))

    assert result.type.equals(_file_arrow_type())
    assert result.to_pylist() == [None, None]


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
