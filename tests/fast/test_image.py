# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pickle

import pandas as pd
import pyarrow as pa
import pytest

import vane

IMAGE_FIELDS = ("data", "width", "height", "channels", "mode")


def test_image_value_and_type_contract():
    image = vane.Image(bytes(range(12)), 2, 2, "RGB")
    dtype = vane.image_type()

    assert vane.Image is vane._native.Image
    assert tuple(getattr(image, field) for field in IMAGE_FIELDS) == (bytes(range(12)), 2, 2, 3, "RGB")
    assert image.dtype == "uint8"
    assert image == pickle.loads(pickle.dumps(image))
    assert hash(image) == hash(pickle.loads(pickle.dumps(image)))
    assert repr(image) == "Image(data=<12 bytes>, width=2, height=2, mode='RGB')"

    assert str(dtype) == "IMAGE"
    assert dtype.id == "struct"
    assert dtype.is_image()
    assert not dtype.is_file()
    assert dtype == vane.sqltype("IMAGE")
    assert dtype.children == [
        ("data", vane.sqltypes.BLOB),
        ("width", vane.sqltypes.UINTEGER),
        ("height", vane.sqltypes.UINTEGER),
        ("channels", vane.sqltypes.UTINYINT),
        ("mode", vane.sqltypes.VARCHAR),
    ]
    assert pickle.loads(pickle.dumps(dtype)).is_image()


def test_image_value_is_immutable_and_final():
    image = vane.Image(b"\x00", 1, 1, "L")
    for field in (*IMAGE_FIELDS, "dtype"):
        with pytest.raises(AttributeError):
            setattr(image, field, None)
    with pytest.raises(TypeError):

        class DerivedImage(vane.Image):
            pass


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((b"", 0, 1, "L"), "positive"),
        ((b"", 1, 0, "L"), "positive"),
        ((b"\x00", 1, 1, "CMYK"), "one of L, LA, RGB, or RGBA"),
        ((b"\x00", 1, 1, "RGB"), "has 1 bytes, expected 3"),
    ],
)
def test_image_value_rejects_invalid_layout(args, error):
    with pytest.raises(ValueError, match=error):
        vane.Image(*args)


@pytest.mark.parametrize("field", ["width", "height"])
def test_image_value_rejects_invalid_dimensions(field):
    args = {"data": b"\x00", "width": 1, "height": 1, "mode": "L"}
    args[field] = True
    with pytest.raises(TypeError, match=rf"Image\.{field} must be int"):
        vane.Image(**args)

    args[field] = 2**32
    with pytest.raises(OverflowError, match="unsigned 32-bit"):
        vane.Image(**args)


def test_image_parameter_fetch_and_arrow_storage(duckdb_cursor):
    image = vane.Image(b"\x00\x01\x02\x03", 2, 2, "L")
    result = duckdb_cursor.execute("SELECT typeof($1), $1", [image])

    assert result.fetchone() == ("IMAGE", image)

    arrow = duckdb_cursor.execute("SELECT $1 AS image", [image]).to_arrow_table()
    assert arrow.schema.field("image").type == pa.struct(
        [
            pa.field("data", pa.binary()),
            pa.field("width", pa.uint32()),
            pa.field("height", pa.uint32()),
            pa.field("channels", pa.uint8()),
            pa.field("mode", pa.string()),
        ]
    )
    assert arrow.column("image").to_pylist() == [
        {"data": image.data, "width": 2, "height": 2, "channels": 1, "mode": "L"}
    ]


def test_image_comparison_requires_the_image_logical_type(duckdb_cursor):
    image = vane.Image(b"\x00", 1, 1, "L")
    other = vane.Image(b"\x01", 1, 1, "L")

    assert duckdb_cursor.execute(
        "SELECT $1 = $2, $1 != $3, $1 = NULL::IMAGE",
        [image, image, other],
    ).fetchone() == (True, True, None)

    with pytest.raises(vane.BinderException, match="Cannot compare values of type IMAGE"):
        duckdb_cursor.execute("SELECT $1 = struct_pack(data := $2)", [image, b"\x00"])


def test_declared_nested_image_type_round_trips(duckdb_cursor):
    image = vane.Image(b"\x00\x01", 2, 1, "L")
    dtype = vane.struct_type({"images": vane.list_type(vane.image_type())})

    type_name, value = duckdb_cursor.execute(
        "SELECT typeof($1), $1",
        [vane.Value({"images": [image, None]}, dtype)],
    ).fetchone()

    assert type_name == "STRUCT(images IMAGE[])"
    assert value == {"images": [image, None]}


def test_pandas_image_columns_preserve_logical_values(duckdb_cursor):
    image = vane.Image(b"\x00\x01", 1, 1, "LA")
    relation = duckdb_cursor.from_df(pd.DataFrame({"image": [image, None]}))

    assert relation.types[0].is_image()
    assert relation.fetchall() == [(image,), (None,)]
    assert relation.df()["image"].tolist()[0] == image


def test_image_type_rejects_structural_fallbacks(duckdb_cursor):
    record = {"data": b"\x00", "width": 1, "height": 1, "channels": 1, "mode": "L"}

    with pytest.raises(vane.InvalidInputException, match="Only vane.Image or NULL"):
        vane.ConstantExpression(vane.Value(record, vane.image_type()))
    with pytest.raises(vane.BinderException, match="governed values require an exact logical type match"):
        duckdb_cursor.sql(
            "SELECT struct_pack(data := '\\x00'::BLOB, width := 1::UINTEGER, height := 1::UINTEGER, "
            "channels := 1::UTINYINT, mode := 'L')::IMAGE"
        )


def test_registered_scalar_image_udf_preserves_type_and_nulls():
    @vane.func(return_dtype=vane.image_type())
    def identity(value):
        assert isinstance(value, vane.Image)
        return value

    connection = vane.connect()
    vane.attach_function(identity, connection=connection, alias="identity_image_value", parameters=[vane.image_type()])
    image = vane.Image(bytes(range(6)), 2, 1, "RGB")

    rows = connection.execute(
        "SELECT typeof(identity_image_value(value)), identity_image_value(value) "
        "FROM (VALUES ($1), (NULL::IMAGE)) images(value)",
        [image],
    ).fetchall()

    assert rows == [("IMAGE", image), ("IMAGE", None)]


def test_batch_image_udf_validates_physical_contract():
    invalid_type = pa.struct(
        [
            pa.field("data", pa.binary()),
            pa.field("width", pa.uint32()),
            pa.field("height", pa.uint32()),
            pa.field("channels", pa.uint8()),
            pa.field("mode", pa.string()),
        ]
    )

    @vane.func.batch(return_dtype=vane.image_type())
    def invalid_image(values):
        return pa.array(
            [{"data": b"\x00", "width": 1, "height": 1, "channels": 3, "mode": "RGB"}] * len(values),
            type=invalid_type,
        )

    with pytest.raises(vane.InvalidInputException, match="expected 3"):
        invalid_image(pa.array([1], type=pa.int32()))


def test_registered_batch_image_udf_preserves_type():
    @vane.func.batch(return_dtype=vane.image_type())
    def identity(values):
        return values

    connection = vane.connect()
    vane.attach_function(identity, connection=connection, alias="identity_image_batch", parameters=[vane.image_type()])
    image = vane.Image(bytes(range(8)), 2, 1, "RGBA")

    assert connection.execute(
        "SELECT typeof(identity_image_batch($1)), identity_image_batch($1)", [image]
    ).fetchone() == (
        "IMAGE",
        image,
    )
