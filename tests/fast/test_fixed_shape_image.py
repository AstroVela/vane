# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pickle

import pyarrow as pa
import pytest

import vane


def test_fixed_image_type_identity_and_serialization():
    dtype = vane.image_type("RGB", 2, 3)
    assert str(dtype) == "IMAGE('RGB', 2, 3)"
    assert dtype.is_image()
    assert dtype == vane.sqltype("IMAGE('RGB', 2, 3)")
    assert dtype == pickle.loads(pickle.dumps(dtype))
    assert dtype != vane.image_type()
    assert dtype != vane.image_type("RGB", 3, 2)
    assert dtype != vane.image_type("RGBA", 2, 3)
    assert dtype.children == vane.image_type().children
    nested = vane.struct_type({"images": vane.list_type(dtype)})
    assert pickle.loads(pickle.dumps(nested)) == nested


@pytest.mark.parametrize("args", [("RGB",), (None, 2, 3), ("RGB", True, 3), ("RGB", 2, 3.5)])
def test_fixed_image_type_requires_complete_typed_layout(args):
    with pytest.raises(TypeError, match="mode.*height.*width"):
        vane.image_type(*args)


@pytest.mark.parametrize(
    "sql",
    ["IMAGE('RGB', 0, 1)", "IMAGE('CMYK', 1, 1)", "IMAGE('RGB', 1)", "IMAGE('RGB', NULL, 1)", "IMAGE('RGB', 1.5, 1)"],
)
def test_fixed_image_sql_type_rejects_invalid_layout(sql):
    with vane.connect() as con, pytest.raises(vane.Error):
        con.sql(f"SELECT NULL::{sql}")


def test_fixed_image_cast_and_typed_python_value():
    image = vane.Image(bytes(range(18)), 3, 2, "RGB")
    dtype = vane.image_type("RGB", 2, 3)
    with vane.connect() as con:
        assert con.execute("SELECT typeof($1), $1", [vane.Value(image, dtype)]).fetchone() == (str(dtype), image)
        rendered = str(vane.ConstantExpression(vane.Value(image, dtype)))
        assert con.execute(f"SELECT typeof({rendered}), {rendered}").fetchone() == (str(dtype), image)
        assert con.execute("SELECT CAST($1 AS IMAGE('RGB', 2, 3))", [image]).fetchone() == (image,)
        assert con.execute("SELECT CAST($1 AS IMAGE)", [vane.Value(image, dtype)]).fetchone() == (image,)
        with pytest.raises(vane.InvalidInputException, match="does not match"):
            con.execute("SELECT CAST($1 AS IMAGE('RGB', 3, 2))", [image])
        with pytest.raises(vane.InvalidInputException, match="does not match"):
            vane.ConstantExpression(vane.Value(image, vane.image_type("RGB", 3, 2)))


def test_fixed_image_try_cast_checks_each_selected_row_and_null():
    image = vane.Image(b"\x01\x02\x03", 1, 1, "RGB")
    wrong = vane.Image(b"\x04", 1, 1, "L")
    with vane.connect() as con:
        con.execute("CREATE TABLE images(i INTEGER, value IMAGE)")
        con.executemany("INSERT INTO images VALUES (?, ?)", [(0, image), (1, wrong), (2, None), (3, image)])
        rows = con.execute(
            "SELECT i, TRY_CAST(value AS IMAGE('RGB', 1, 1)) FROM images WHERE i != 0 ORDER BY i"
        ).fetchall()
        assert rows == [(1, None), (2, None), (3, image)]
        assert con.execute("SELECT CAST(NULL::IMAGE AS IMAGE('RGB', 1, 1)) FROM range(3)").fetchall() == [(None,)] * 3
        assert con.execute("SELECT CAST($1 AS IMAGE('RGB', 1, 1)) FROM range(5)", [image]).fetchall() == [(image,)] * 5
        assert con.execute("SELECT CAST($1 AS IMAGE('RGB', 1, 1)) = $1", [image]).fetchone() == (True,)
        assert con.execute(
            "SELECT CAST($1 AS IMAGE('RGB', 1, 1)) = CAST($2 AS IMAGE('L', 1, 1))", [image, wrong]
        ).fetchone() == (False,)


def test_fixed_image_nested_storage_roundtrip(tmp_path):
    image = vane.Image(b"\x01\x02\x03", 1, 1, "RGB")
    dtype = vane.struct_type({"images": vane.list_type(vane.image_type("RGB", 1, 1))})
    value = {"images": [image, None]}
    database = str(tmp_path / "fixed-images.db")
    with vane.connect(database) as con:
        con.execute("CREATE TABLE images(value STRUCT(images IMAGE('RGB', 1, 1)[]))")
        con.execute("INSERT INTO images VALUES (?)", [vane.Value(value, dtype)])
    with vane.connect(database) as con:
        relation = con.table("images")
        assert relation.types == [dtype]
        assert relation.fetchall() == [(value,)]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize(
    "query",
    [
        "SELECT CASE WHEN i = 0 THEN $1 ELSE $2 END AS value FROM range(2) t(i)",
        "SELECT * FROM (VALUES ($1), ($2)) t(value)",
        "SELECT $1 AS value UNION ALL SELECT $2",
        "SELECT COALESCE(CASE WHEN i = 0 THEN $1 END, $2) AS value FROM range(2) t(i)",
        "SELECT unnest([$1, $2]) AS value",
    ],
)
def test_mixed_image_layouts_have_order_independent_common_type(query, generic, reverse):
    left = vane.Image(b"abc", 1, 1, "RGB")
    right = vane.Image(b"xy", 2, 1, "L")
    values = [
        vane.Value(left, vane.image_type("RGB", 1, 1)),
        vane.Value(right, vane.image_type() if generic else vane.image_type("L", 1, 2)),
    ]
    expected = [left, right]
    if reverse:
        values.reverse()
        expected.reverse()
    with vane.connect() as con:
        relation = con.sql(query, params=values)
        assert relation.types == [vane.image_type()]
        assert relation.fetchall() == [(image,) for image in expected]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "wrap_type,wrap_value",
    [
        (vane.list_type, lambda image: [image, None]),
        (lambda dtype: vane.array_type(dtype, 2), lambda image: (image, None)),
        (
            lambda dtype: vane.struct_type({"image": dtype, "label": vane.sqltypes.VARCHAR}),
            lambda image: {"image": image, "label": "kept"},
        ),
        (lambda dtype: vane.map_type(vane.sqltypes.VARCHAR, dtype), lambda image: {"kept": image}),
    ],
    ids=["list", "array", "struct", "map"],
)
def test_nested_mixed_image_layouts_widen_without_losing_pixels(wrap_type, wrap_value, reverse):
    left = vane.Image(b"abc", 1, 1, "RGB")
    right = vane.Image(b"abcdef", 2, 1, "RGB")
    types = [vane.image_type("RGB", 1, 1), vane.image_type("RGB", 1, 2)]
    values = [left, right]
    parameters = [vane.Value(wrap_value(value), wrap_type(dtype)) for value, dtype in zip(values, types, strict=True)]
    expected = [wrap_value(value) for value in values]
    if reverse:
        parameters.reverse()
        expected.reverse()
    with vane.connect() as con:
        relation = con.sql("SELECT $1 AS value UNION ALL SELECT $2", params=parameters)
        assert relation.types == [wrap_type(vane.image_type())]
        assert relation.fetchall() == [(value,) for value in expected]


def test_common_type_keeps_equal_image_constraints_and_requires_explicit_narrowing():
    image = vane.Image(b"abc", 1, 1, "RGB")
    fixed = vane.image_type("RGB", 1, 1)
    with vane.connect() as con:
        for other in [vane.Value(image, fixed), None]:
            relation = con.sql("SELECT $1 AS value UNION ALL SELECT $2", params=[vane.Value(image, fixed), other])
            assert relation.types == [fixed]
        vane.attach_function(
            vane.func(return_dtype=fixed)(lambda value: value),
            connection=con,
            alias="fixed_input",
            parameters=[fixed],
        )
        with pytest.raises(vane.BinderException):
            con.execute("SELECT fixed_input($1)", [image])
        assert con.execute("SELECT fixed_input(CAST($1 AS IMAGE('RGB', 1, 1)))", [image]).fetchone() == (image,)


def test_image_map_display_does_not_allow_sql_to_erase_the_logical_value():
    image = vane.Image(b"abc", 1, 1, "RGB")
    value = vane.Value({"kept": image}, vane.map_type(vane.sqltypes.VARCHAR, vane.image_type("RGB", 1, 1)))
    with vane.connect() as con:
        assert con.sql("SELECT $1 AS value", params=[value]).fetchall() == [({"kept": image},)]
        for target in ["VARCHAR", "MAP(VARCHAR, VARCHAR)", "STRUCT(key VARCHAR, value VARCHAR)[]"]:
            with pytest.raises(vane.BinderException, match="governed"):
                con.execute(f"SELECT CAST($1 AS {target})", [value])


def test_fixed_image_rejects_raw_struct_construction():
    with vane.connect() as con, pytest.raises(vane.BinderException, match="exact logical type"):
        con.sql(
            "SELECT struct_pack(data := 'abc'::BLOB, width := 1::UINTEGER, height := 1::UINTEGER, "
            "channels := 3::UTINYINT, mode := 'RGB')::IMAGE('RGB', 1, 1)"
        )


@pytest.mark.parametrize("batch", [False, True])
def test_fixed_image_registered_udf_preserves_layout(batch):
    dtype = vane.image_type("RGB", 1, 1)

    def identity(value):
        if not batch:
            assert isinstance(value, vane.Image)
        return value

    function = (vane.func.batch if batch else vane.func)(return_dtype=dtype)(identity)
    with vane.connect() as con:
        vane.attach_function(function, connection=con, alias="fixed_identity", parameters=[dtype])
        image = vane.Image(b"\x01\x02\x03", 1, 1, "RGB")
        relation = con.sql(
            "SELECT fixed_identity(value) AS image FROM (VALUES (CAST($1 AS IMAGE('RGB', 1, 1))), "
            "(NULL::IMAGE('RGB', 1, 1))) t(value)",
            params=[image],
        )
        assert relation.types == [dtype]
        assert relation.fetchall() == [(image,), (None,)]


@pytest.mark.parametrize("batch", [False, True])
def test_fixed_image_udf_rejects_valid_pixels_with_wrong_shape(batch):
    dtype = vane.image_type("RGB", 1, 2)
    wrong = vane.Image(bytes(range(6)), 1, 2, "RGB")
    if batch:
        storage = pa.struct(
            [
                ("data", pa.binary()),
                ("width", pa.uint32()),
                ("height", pa.uint32()),
                ("channels", pa.uint8()),
                ("mode", pa.string()),
            ]
        )

        @vane.func.batch(return_dtype=dtype)
        def invalid(values):
            return pa.array(
                [{name: getattr(wrong, name) for name in ("data", "width", "height", "channels", "mode")}]
                * len(values),
                type=storage,
            )

        with pytest.raises(vane.InvalidInputException, match="layout"):
            invalid(pa.array([1]))
    else:

        @vane.func(return_dtype=dtype)
        def invalid(_value):
            return wrong

        with vane.connect() as con:
            vane.attach_function(invalid, connection=con, alias="invalid_fixed_image", parameters=["INTEGER"])
            with pytest.raises(vane.InvalidInputException, match="layout"):
                con.execute("SELECT invalid_fixed_image(1)")


def test_fixed_image_batch_ignores_inactive_nested_rows():
    storage = pa.struct(
        [
            ("data", pa.binary()),
            ("width", pa.uint32()),
            ("height", pa.uint32()),
            ("channels", pa.uint8()),
            ("mode", pa.string()),
        ]
    )
    images = pa.array(
        [
            {"data": b"abcdef", "width": 1, "height": 2, "channels": 3, "mode": "RGB"},
            {"data": b"abcdef", "width": 2, "height": 1, "channels": 3, "mode": "RGB"},
        ],
        type=storage,
    )
    payload = pa.StructArray.from_arrays([images], names=["image"], mask=pa.array([True, False]))

    @vane.func.batch(return_dtype=vane.struct_type({"image": vane.image_type("RGB", 1, 2)}))
    def output(_values):
        return payload

    assert output(pa.array([1, 2])).to_pylist() == [None, {"image": images[1].as_py()}]


_IMAGE_CONTAINERS = [
    ("{}", lambda image: image),
    ("{}[]", lambda image: [image, None]),
    ("{}[2]", lambda image: (image, None)),
    ("STRUCT(image {})", lambda image: {"image": image}),
    ("MAP(VARCHAR, {})", lambda image: {"kept": image}),
    ("STRUCT(images {}[])", lambda image: {"images": [image, None]}),
]


@pytest.mark.parametrize("container,wrap", _IMAGE_CONTAINERS)
@pytest.mark.parametrize("source_fixed", [False, True])
def test_fixed_image_assignment_requires_explicit_layout_cast(container, wrap, source_fixed):
    image = vane.Image(b"abcdef" if source_fixed else b"abc", 2 if source_fixed else 1, 1, "RGB")
    source_type = container.format("IMAGE('RGB', 1, 2)" if source_fixed else "IMAGE")
    target_type = container.format("IMAGE('RGB', 1, 1)")
    parameter = vane.Value(wrap(image), vane.sqltype(source_type))
    with vane.connect() as con:
        con.execute(f"CREATE TABLE source_images(value {source_type})")
        con.execute("INSERT INTO source_images VALUES (?)", [parameter])
        con.execute(f"CREATE TABLE fixed_images(value {target_type})")
        con.execute("INSERT INTO fixed_images VALUES (NULL)")
        for query in [
            "INSERT INTO fixed_images SELECT value FROM source_images",
            "UPDATE fixed_images SET value = (SELECT value FROM source_images)",
        ]:
            with pytest.raises(vane.BinderException, match="governed"):
                con.execute(query)
        with pytest.raises(vane.BinderException, match="governed"):
            con.execute("INSERT INTO fixed_images VALUES (?)", [parameter])
        assert con.execute("SELECT value FROM fixed_images").fetchall() == [(None,)]
        if not source_fixed:
            con.execute(f"INSERT INTO fixed_images SELECT CAST(value AS {target_type}) FROM source_images")
            assert con.execute("SELECT value FROM fixed_images WHERE value IS NOT NULL").fetchall() == [(wrap(image),)]


@pytest.mark.parametrize("container,wrap", _IMAGE_CONTAINERS)
def test_explicit_nested_image_cast_validates_each_leaf(container, wrap):
    good = vane.Image(b"abc", 1, 1, "RGB")
    bad = vane.Image(b"abcdef", 2, 1, "RGB")
    source_type = container.format("IMAGE")
    target_type = container.format("IMAGE('RGB', 1, 1)")
    with vane.connect() as con:
        con.execute(f"CREATE TABLE images(i INTEGER, value {source_type})")
        con.executemany(
            "INSERT INTO images VALUES (?, ?)",
            [
                (0, vane.Value(wrap(good), vane.sqltype(source_type))),
                (1, vane.Value(wrap(bad), vane.sqltype(source_type))),
                (2, None),
            ],
        )
        relation = con.sql(f"SELECT CAST(value AS {target_type}) FROM images WHERE i != 1 ORDER BY i")
        assert relation.types == [vane.sqltype(target_type)]
        assert relation.fetchall() == [(wrap(good),), (None,)]
        with pytest.raises(vane.InvalidInputException, match="does not match"):
            con.execute(f"SELECT CAST(value AS {target_type}) FROM images ORDER BY i")
        assert con.execute(f"SELECT TRY_CAST(value AS {target_type}) FROM images ORDER BY i").fetchall() == [
            (wrap(good),),
            (wrap(None),),
            (None,),
        ]
        expression = con.table("images").filter("i = 0").select(vane.col("value").cast(vane.sqltype(target_type)))
        assert expression.types == [vane.sqltype(target_type)]
        assert expression.fetchall() == [(wrap(good),)]


def test_fixed_image_cast_validation_survives_predicate_rewrites():
    image = vane.Image(b"abc", 1, 1, "RGB")
    with vane.connect() as con:
        con.execute("CREATE TABLE images(value IMAGE)")
        con.execute("INSERT INTO images VALUES (?)", [image])
        for predicate in ["= $1", "IN ($1)"]:
            with pytest.raises(vane.InvalidInputException, match="does not match"):
                con.execute(f"SELECT value FROM images WHERE CAST(value AS IMAGE('RGB', 1, 2)) {predicate}", [image])


def test_image_map_try_cast_nulls_invalid_keys_and_preserves_other_rows():
    good = vane.Image(b"abc", 1, 1, "RGB")
    bad = vane.Image(b"abcdef", 2, 1, "RGB")
    source_type = vane.map_type(vane.image_type(), vane.sqltypes.INTEGER)
    target = "MAP(IMAGE('RGB', 1, 1), INTEGER)"
    originals = [{good: 1}, {bad: 2}, {good: 1, bad: 2}, {}, None]
    with vane.connect() as con:
        con.execute("CREATE TABLE maps(i INTEGER, value MAP(IMAGE, INTEGER))")
        con.executemany(
            "INSERT INTO maps VALUES (?, ?)",
            [(i, vane.Value(value, source_type)) for i, value in enumerate(originals)],
        )
        assert con.execute(f"SELECT TRY_CAST(value AS {target}) FROM maps ORDER BY i").fetchall() == [
            ({good: 1},),
            (None,),
            (None,),
            ({},),
            (None,),
        ]
        rows = con.execute(f"SELECT TRY_CAST(value AS {target}), value FROM maps ORDER BY i").fetchall()
        assert [row[0] for row in rows] == [{good: 1}, None, None, {}, None]
        assert [row[1] for row in rows] == originals
        assert (
            con.execute(
                f"SELECT TRY_CAST($1 AS {target}) FROM range(5)", [vane.Value({bad: 1}, source_type)]
            ).fetchall()
            == [(None,)] * 5
        )
        assert con.execute(f"SELECT CAST(value AS {target}) FROM maps WHERE i = 0").fetchone() == ({good: 1},)
        with pytest.raises(vane.InvalidInputException, match="does not match"):
            con.execute(f"SELECT CAST(value AS {target}) FROM maps WHERE i = 1")
        arrow = con.execute(f"SELECT TRY_CAST(value AS {target}) AS value FROM maps ORDER BY i").fetch_arrow_table()
        assert arrow.column(0).is_null().to_pylist() == [False, True, True, False, True]


@pytest.mark.parametrize("bad_layout", [False, True])
def test_image_map_key_cast_rejects_colliding_nested_keys(bad_layout):
    image = vane.Image(b"abcdef" if bad_layout else b"abc", 2 if bad_layout else 1, 1, "RGB")
    keys = "[{'image': $1, 'label': '01'}, {'image': $1, 'label': '1'}]"
    target = "MAP(STRUCT(image IMAGE('RGB', 1, 1), label INTEGER), INTEGER)"
    with vane.connect() as con:
        # The two keys are initially distinct; casting the label merges them.
        assert con.execute(f"SELECT TRY_CAST(MAP({keys}, [1, 2]) AS {target})", [image]).fetchone() == (None,)
        with pytest.raises(vane.InvalidInputException, match="does not match|unique"):
            con.execute(f"SELECT CAST(MAP({keys}, [1, 2]) AS {target})", [image])


def test_image_map_key_failure_inside_a_list_nulls_only_that_map():
    good = vane.Image(b"abc", 1, 1, "RGB")
    bad = vane.Image(b"abcdef", 2, 1, "RGB")
    with vane.connect() as con:
        value = con.execute(
            "SELECT TRY_CAST([MAP([$1], [1]), MAP([$2], [2])] AS MAP(IMAGE('RGB', 1, 1), INTEGER)[])",
            [good, bad],
        ).fetchone()[0]
        assert value == [{good: 1}, None]
