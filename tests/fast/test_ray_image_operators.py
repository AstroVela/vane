# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pyarrow as pa
import pytest

import vane
from tests.fast.test_image_operators import _check_png
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
