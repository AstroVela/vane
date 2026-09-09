# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pytest

import vane
from tests.fast.test_image_operators import _check_png
from tests.fast.test_image_to_tensor import _assert_cell, _types
from tests.fast.test_ray_native_media_extensions import _load_provider
from vane._image import image_arrow_type


@pytest.mark.real_ray
@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("fixed", [False, True])
def test_ray_crop_png_and_udf_preserve_image_contract(ray_local, backend, fixed):
    pytest.importorskip("PIL.Image")
    from vane.runners.ray.runner import RayRunner

    source_type = "IMAGE('RGBA', 2, 3)" if fixed else "IMAGE"
    result_type = vane.image_type("RGBA") if fixed else vane.image_type()

    @vane.func.batch(return_dtype=result_type)
    def identity(images):
        assert images.type.extension_name == "vane.image"
        return images

    with vane.connect(config={"image_backend": backend}) as con:
        if backend == "native":
            _load_provider(con, "image")
        vane.attach_function(identity, connection=con, alias="image_identity", parameters=[result_type])
        relation = con.sql(
            f"""WITH images AS (
                SELECT i, (CASE WHEN i % 5 = 0 THEN NULL ELSE
                  image(repeat(chr((65+i)::INTEGER),24)::BLOB,3,2,4,'RGBA') END)::{source_type} AS image
                FROM range(18) t(i)
            ), cropped AS (
                SELECT i, image_identity(crop(image, [-1,0,3,1])) AS image FROM images
            ) SELECT a.i, a.image, encode_image(a.image, 'PNG') AS png
              FROM cropped a JOIN range(18) b(i) ON a.i = b.i ORDER BY a.i"""
        )
        assert relation.columns == ["i", "image", "png"]
        assert relation.types == [vane.sqltypes.BIGINT, result_type, vane.sqltypes.BLOB]
        runner = RayRunner(address=None, max_task_backlog=None)
        try:
            parts = list(runner.run_iter_tables(relation))
            table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
        finally:
            runner.close()
        # run_iter_tables exposes physical output names; the relation above
        # owns the public names. Inspect the Image's positional output field.
        assert table.num_columns == 3
        assert table.schema.field(1).type.equals(image_arrow_type(result_type))
        rows = con.from_arrow(table).fetchall()
    assert len(rows) == 18
    for index, image, encoded in rows:
        if index % 5 == 0:
            assert image is encoded is None
        else:
            expected = np.full((1, 3, 4), 65 + index, dtype=np.uint8)
            expected[:, 0] = 0
            np.testing.assert_array_equal(image, expected)
            _check_png(encoded, expected, "RGBA")


@pytest.mark.real_ray
@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("fixed", [False, True])
def test_ray_resize_convert_and_arrow_udf_preserve_shapes(ray_local, backend, fixed):
    from vane.runners.ray.runner import RayRunner

    input_type = "IMAGE('RGB',2,2)" if fixed else "IMAGE"
    result_type = vane.image_type("RGBA", 3, 3) if fixed else vane.image_type("RGBA")

    @vane.func.batch(return_dtype=result_type)
    def identity(images):
        assert images.type.equals(image_arrow_type(result_type))
        return images

    with vane.connect(config={"image_backend": backend}) as con:
        if backend == "native":
            _load_provider(con, "image")
        vane.attach_function(identity, connection=con, alias="transform_identity", parameters=[result_type])
        relation = con.sql(f"""WITH images AS (
            SELECT i, (CASE WHEN i%5=0 THEN NULL ELSE
                image(repeat(chr((65+i)::INTEGER),12)::BLOB,2,2,3,'RGB') END)::{input_type} AS image
            FROM range(18) t(i)
        ), transformed AS (
            SELECT i, transform_identity(convert_image(resize(image,3,3),'RGBA')) AS image FROM images
        ) SELECT a.i, a.image FROM transformed a JOIN range(18) b(i) ON a.i=b.i ORDER BY a.i""")
        assert relation.types == [vane.sqltypes.BIGINT, result_type]
        runner = RayRunner(address=None, max_task_backlog=None)
        try:
            parts = list(runner.run_iter_tables(relation))
            table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
        finally:
            runner.close()
        assert table.schema.field(1).type.equals(image_arrow_type(result_type))
        rows = con.from_arrow(table).fetchall()
    assert len(rows) == 18
    for index, pixels in rows:
        if index % 5 == 0:
            assert pixels is None
        else:
            expected = np.full((3, 3, 4), 65 + index, dtype=np.uint8)
            expected[:, :, -1] = 255
            np.testing.assert_array_equal(pixels, expected)


@pytest.mark.real_ray
@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("form", ["generic", "mode", "fixed"])
def test_ray_image_to_tensor_udf_and_flight_shuffle_without_extension(ray_local, backend, form):
    from vane.runners.ray.runner import RayRunner

    image_type, tensor_type, arrow_type = _types("RGB", form, 1, 2)

    @vane.func.batch(return_dtype=tensor_type)
    def identity(tensors):
        assert tensors.type.equals(arrow_type)
        return tensors

    with vane.connect(config={"image_backend": backend}) as con:
        vane.attach_function(identity, connection=con, alias="image_tensor_identity", parameters=[tensor_type])
        relation = con.sql(f"""WITH images AS (
            SELECT i, (CASE WHEN i%5=0 THEN NULL ELSE
                image(from_hex(repeat(lpad(to_hex(i+200),2,'0'),6)),2,1,3,'RGB') END)::{image_type} AS image
            FROM range(18) t(i)
        ), tensors AS (
            SELECT i, image_tensor_identity(image_to_tensor(image)) AS tensor FROM images
        ) SELECT a.i, a.tensor FROM tensors a JOIN range(18) b(i) ON a.i=b.i ORDER BY a.i""")
        assert relation.types == [vane.sqltypes.BIGINT, tensor_type]
        runner = RayRunner(address=None, max_task_backlog=None)
        try:
            parts = list(runner.run_iter_tables(relation))
            table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
        finally:
            runner.close()
        assert table.schema.field(1).type.equals(arrow_type)
        rows = con.from_arrow(table).fetchall()
    assert len(rows) == 18
    for index, tensor in rows:
        expected = None if index % 5 == 0 else np.full((1, 2, 3), index + 200, dtype=np.uint8)
        _assert_cell(tensor, expected, form == "fixed", form == "generic")


@pytest.mark.real_ray
@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("mode,pixel_type", [("RGB16", np.uint16), ("RGB32F", np.float32)])
def test_ray_wide_codec_hash_and_udf_keep_logical_types(ray_local, backend, mode, pixel_type):
    from tests.fast.test_image_modes import assert_pixels
    from vane.runners.ray.runner import RayRunner

    dtype = vane.image_type(mode)
    pixels = np.arange(27).reshape(3, 3, 3).astype(pixel_type)
    if pixel_type == np.float32:
        pixels /= 26
    else:
        pixels *= 2000

    @vane.func.batch(return_dtype=dtype)
    def identity(images):
        assert images.type.equals(image_arrow_type(dtype))
        return images

    with vane.connect(config={"image_backend": backend}) as con:
        if backend == "native":
            _load_provider(con, "image")
        vane.attach_function(identity, connection=con, alias="wide_identity", parameters=[dtype])
        # Use a SQL literal so the complete source value traverses the plan serializer.
        literal = str(vane.ConstantExpression(vane.Value(pixels, dtype)))
        relation = con.sql(f"""WITH images AS (
            SELECT i, wide_identity(decode_image(encode_image(
                CASE WHEN i%5=0 THEN NULL ELSE {literal} END,'TIFF'), mode=>'{mode}')) AS image
            FROM range(18) t(i)
        ) SELECT a.i, a.image, image_hash(a.image) AS hash
          FROM images a JOIN range(18) b(i) ON a.i=b.i ORDER BY a.i""")
        expected = con.sql("SELECT image_hash($1)", params=[vane.Value(pixels, dtype)]).fetchone()[0]
        runner = RayRunner(address=None, max_task_backlog=None)
        try:
            parts = list(runner.run_iter_tables(relation))
            table = pa.concat_tables([part.to_arrow() if hasattr(part, "to_arrow") else part for part in parts])
        finally:
            runner.close()
        assert table.schema.field(1).type.equals(image_arrow_type(dtype))
        assert table.schema.field(2).type == pa.binary(8)
        rows = con.from_arrow(table).fetchall()
    assert len(rows) == 18
    for index, image, hashed in rows:
        if index % 5 == 0:
            assert image is hashed is None
        else:
            assert_pixels(image, pixels)
            assert hashed == expected
