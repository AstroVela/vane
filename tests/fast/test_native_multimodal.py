# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import pickle

import numpy as np
import pytest

import vane
from vane import image

pa = pytest.importorskip("pyarrow", "18.0.0")

PILLOW_11_3_RGB_3X2_COMPRESS_LEVEL_2 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000030000000208020000001216f14d"
    "0000001449444154785e63646064620603164e4e4e080b00028e0048f186b7bb"
    "0000000049454e44ae426082"
)


def _pillow_crop_encode_png(frames: np.ndarray, boxes: np.ndarray) -> list[bytes]:
    image_module = pytest.importorskip("PIL.Image")
    output = []
    buffer = io.BytesIO()
    for frame, box in zip(frames, boxes, strict=True):
        cropped = image_module.fromarray(frame).crop(tuple(map(int, box)))
        buffer.seek(0)
        buffer.truncate(0)
        cropped.save(buffer, format="PNG", compress_level=2)
        output.append(buffer.getvalue())
    return output


def _relation_for_frames(duckdb_cursor, frames: np.ndarray, boxes: object):
    frame_array = pa.FixedShapeTensorArray.from_numpy_ndarray(np.ascontiguousarray(frames))
    bbox_array = pa.array(boxes, type=pa.list_(pa.float64()))
    return duckdb_cursor.from_arrow(pa.table({"frame": frame_array, "bbox": bbox_array}))


def _encode_expression(duckdb_cursor, frames: np.ndarray, boxes: object) -> list[bytes | None]:
    relation = _relation_for_frames(duckdb_cursor, frames, boxes)
    expression = image.encode(image.crop(vane.col("frame"), vane.col("bbox")), format="png")
    return [row[0] for row in relation.select(expression.alias("object")).fetchall()]


def test_native_image_expression_api_only_exposes_crop_and_encode():
    assert "image" in vane.__all__
    assert image.__all__ == ["crop", "encode"]
    assert image.crop is vane.image.crop
    assert image.encode is vane.image.encode
    assert not hasattr(image, "crop_encode_png")


def test_fused_crop_encode_function_is_not_registered(duckdb_cursor):
    frames = np.zeros((1, 1, 1, 3), dtype=np.uint8)
    relation = _relation_for_frames(duckdb_cursor, frames, [[0.0, 0.0, 1.0, 1.0]])
    fused = vane.FunctionExpression("image_crop_encode_png", vane.col("frame"), vane.col("bbox"))

    with pytest.raises(vane.CatalogException, match="image_crop_encode_png"):
        relation.select(fused)


def test_crop_and_encode_expressions_match_pillow_bytes(duckdb_cursor):
    frames = np.arange(3 * 5 * 6 * 3, dtype=np.uint8).reshape((3, 5, 6, 3))
    boxes = np.array(
        [
            [1.9, 0.2, 5.8, 3.9],
            [0.0, 2.0, 2.0, 5.0],
            [2.0, 1.0, 6.0, 5.0],
        ],
        dtype=np.float64,
    )

    reference = _pillow_crop_encode_png(frames, boxes)
    encoded = _encode_expression(duckdb_cursor, frames, boxes.tolist())

    assert encoded == reference


def test_crop_expression_exposes_padded_rgb_image_value(duckdb_cursor):
    frames = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((1, 2, 3, 3))
    relation = _relation_for_frames(duckdb_cursor, frames, [[-1.0, -1.0, 2.0, 2.0]])
    cropped_relation = relation.select(image.crop(vane.col("frame"), vane.col("bbox")).alias("image"))

    [(cropped,)] = cropped_relation.fetchall()

    expected = np.zeros((3, 3, 3), dtype=np.uint8)
    expected[1:, 1:] = frames[0, :, :2]
    assert cropped_relation.types == [vane.image_type()]
    assert cropped == {"width": 3, "height": 3, "pixels": expected.tobytes()}


def test_crop_expression_allows_empty_pillow_image_until_encoding(duckdb_cursor):
    frames = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((1, 2, 3, 3))
    relation = _relation_for_frames(duckdb_cursor, frames, [[1.0, 0.0, 1.0, 2.0]])
    cropped = image.crop(vane.col("frame"), vane.col("bbox"))

    assert relation.select(cropped.alias("image")).fetchall() == [({"width": 0, "height": 2, "pixels": b""},)]
    with pytest.raises(vane.InvalidInputException, match="tile cannot extend outside image"):
        relation.select(image.encode(cropped).alias("encoded")).fetchall()


def test_crop_expressions_accept_fixed_size_bbox_array(duckdb_cursor):
    frames = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((1, 2, 3, 3))
    frame_array = pa.FixedShapeTensorArray.from_numpy_ndarray(frames)
    bbox_array = pa.array([[0.0, 0.0, 3.0, 2.0]], type=pa.list_(pa.float64(), 4))
    relation = duckdb_cursor.from_arrow(pa.table({"frame": frame_array, "bbox": bbox_array}))

    expected = _pillow_crop_encode_png(frames, np.array([[0.0, 0.0, 3.0, 2.0]]))
    cropped = image.crop(vane.col("frame"), vane.col("bbox"))
    assert relation.select(image.encode(cropped)).fetchall() == [(expected[0],)]


def test_crop_and_encode_match_pillow_padding_and_large_idat_output(duckdb_cursor):
    rng = np.random.default_rng(20260810)
    frames = rng.integers(0, 256, size=(4, 256, 320, 3), dtype=np.uint8)
    boxes = np.array(
        [
            [-3.9, -2.1, 319.8, 255.9],
            [317.2, 253.8, 325.7, 260.4],
            [325.0, 260.0, 330.0, 264.0],
            [-8.0, -6.0, -2.0, -1.0],
        ],
        dtype=np.float64,
    )

    reference = _pillow_crop_encode_png(frames, boxes)
    encoded = _encode_expression(duckdb_cursor, frames, boxes.tolist())

    assert encoded == reference
    assert len(reference[0]) > 65536


def test_crop_and_encode_match_pillow_width_scaled_idat_chunks(duckdb_cursor):
    rng = np.random.default_rng(17000)
    frames = rng.integers(0, 256, size=(1, 2, 17_000, 3), dtype=np.uint8)
    boxes = np.array([[0.0, 0.0, 17_000.0, 2.0]], dtype=np.float64)

    reference = _pillow_crop_encode_png(frames, boxes)
    assert _encode_expression(duckdb_cursor, frames, boxes.tolist()) == reference
    assert len(reference[0]) > 68_000


def test_crop_and_encode_match_pillow_across_random_boxes(duckdb_cursor):
    rng = np.random.default_rng(20260811)
    frames = rng.integers(0, 256, size=(32, 11, 13, 3), dtype=np.uint8)
    integer_boxes = []
    for _ in range(len(frames)):
        left = int(rng.integers(-7, 15))
        top = int(rng.integers(-6, 13))
        integer_boxes.append([left, top, left + int(rng.integers(1, 18)), top + int(rng.integers(1, 16))])
    boxes = np.asarray(integer_boxes, dtype=np.float64)
    boxes += np.where(boxes < 0, -0.75, 0.75)

    reference = _pillow_crop_encode_png(frames, boxes)
    assert _encode_expression(duckdb_cursor, frames, boxes.tolist()) == reference


def test_crop_and_encode_have_pillow_11_3_golden_bytes_without_runtime_pillow(duckdb_cursor):
    frames = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((1, 2, 3, 3))

    assert _encode_expression(duckdb_cursor, frames, [[0.0, 0.0, 3.0, 2.0]]) == [PILLOW_11_3_RGB_3X2_COMPRESS_LEVEL_2]


def test_crop_and_encode_run_after_unnest_without_a_python_udf(duckdb_cursor):
    frames = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((1, 2, 3, 3))
    frame_array = pa.FixedShapeTensorArray.from_numpy_ndarray(frames)
    feature_type = pa.struct([("bbox", pa.list_(pa.float64())), ("label", pa.int64())])
    features = pa.array(
        [
            [
                {"bbox": [0.0, 0.0, 2.0, 2.0], "label": 1},
                {"bbox": [1.0, 0.0, 3.0, 2.0], "label": 2},
            ]
        ],
        type=pa.list_(feature_type),
    )
    relation = duckdb_cursor.from_arrow(pa.table({"frame": frame_array, "features": features}))
    exploded = relation.select(
        vane.col("frame"),
        vane.FunctionExpression("unnest", vane.col("features")).alias("features"),
    )
    bbox = vane.FunctionExpression("struct_extract", vane.col("features"), vane.lit("bbox"))
    cropped = image.crop(vane.col("frame"), bbox)
    encoded = exploded.select(image.encode(cropped).alias("object"))

    expected_frames = np.repeat(frames, 2, axis=0)
    expected_boxes = np.array([[0.0, 0.0, 2.0, 2.0], [1.0, 0.0, 3.0, 2.0]])
    assert [row[0] for row in encoded.fetchall()] == _pillow_crop_encode_png(expected_frames, expected_boxes)
    native_plan = exploded.select(image.encode(cropped)).explain().lower()
    assert "image_crop" in native_plan
    assert "image_encode" in native_plan
    assert "image_crop_encode_png" not in native_plan
    assert "streaming_udf" not in native_plan


@pytest.mark.parametrize(
    "box",
    [
        [2.0, 1.0, 2.0, 3.0],
        [2.0, 1.0, 4.0, 1.0],
        [3.0, 1.0, 2.0, 3.0],
        [1.0, 3.0, 4.0, 2.0],
        [np.nan, 0.0, 1.0, 1.0],
        [np.inf, 0.0, 1.0, 1.0],
        [-np.inf, 0.0, 1.0, 1.0],
    ],
)
def test_crop_expressions_match_pillow_coordinate_error_messages(duckdb_cursor, box):
    frames = np.arange(4 * 5 * 3, dtype=np.uint8).reshape((1, 4, 5, 3))
    boxes = np.asarray([box], dtype=np.float64)

    with pytest.raises(Exception) as reference_error:
        _pillow_crop_encode_png(frames, boxes)
    with pytest.raises(Exception) as native_error:
        _encode_expression(duckdb_cursor, frames, boxes.tolist())

    assert str(reference_error.value) in str(native_error.value)


def test_crop_and_encode_propagate_null_rows(duckdb_cursor):
    frames = np.arange(3 * 2 * 3 * 3, dtype=np.uint8).reshape((3, 2, 3, 3))
    boxes = [[0.0, 0.0, 3.0, 2.0], None, [1.0, 0.0, 3.0, 2.0]]
    expected = _pillow_crop_encode_png(
        frames[[0, 2]],
        np.array([[0.0, 0.0, 3.0, 2.0], [1.0, 0.0, 3.0, 2.0]]),
    )

    assert _encode_expression(duckdb_cursor, frames, boxes) == [expected[0], None, expected[1]]


def test_crop_and_encode_reset_validity_across_vector_boundaries(duckdb_cursor):
    row_count = 2051
    frame = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((1, 2, 3, 3))
    frames = np.repeat(frame, row_count, axis=0)
    full_box = [0.0, 0.0, 3.0, 2.0]
    boxes = [None] * row_count
    boxes[2047] = full_box
    boxes[2049] = full_box
    expected = [None] * row_count
    expected[2047] = PILLOW_11_3_RGB_3X2_COMPRESS_LEVEL_2
    expected[2049] = PILLOW_11_3_RGB_3X2_COMPRESS_LEVEL_2

    assert _encode_expression(duckdb_cursor, frames, boxes) == expected


def test_crop_rejects_rgb_values_larger_than_the_image_blob_limit(duckdb_cursor):
    frames = np.zeros((1, 1, 1, 3), dtype=np.uint8)
    relation = _relation_for_frames(duckdb_cursor, frames, [[0.0, 0.0, 50_000.0, 50_000.0]])

    with pytest.raises(vane.OutOfRangeException, match="BLOB size limit"):
        relation.select(image.crop(vane.col("frame"), vane.col("bbox"))).fetchall()


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (np.zeros((1, 2, 3, 3), dtype=np.float32), "elements must be UTINYINT"),
        (np.zeros((1, 2, 3, 4), dtype=np.uint8), r"shape \[height, width, 3\]"),
    ],
)
def test_crop_rejects_non_rgb_uint8_tensor_types(duckdb_cursor, frames, message):
    relation = _relation_for_frames(duckdb_cursor, frames, [[0.0, 0.0, 1.0, 1.0]])

    with pytest.raises(vane.BinderException, match=message):
        relation.select(image.crop(vane.col("frame"), vane.col("bbox")))


def test_crop_rejects_bbox_with_wrong_runtime_length(duckdb_cursor):
    frames = np.zeros((1, 2, 3, 3), dtype=np.uint8)

    with pytest.raises(vane.InvalidInputException, match="exactly four coordinates"):
        _encode_expression(duckdb_cursor, frames, [[0.0, 0.0, 1.0]])


def test_image_encode_format_is_a_constant_option(duckdb_cursor):
    frames = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((1, 2, 3, 3))
    relation = _relation_for_frames(duckdb_cursor, frames, [[0.0, 0.0, 3.0, 2.0]])
    cropped = image.crop(vane.col("frame"), vane.col("bbox"))

    uppercase = relation.select(image.encode(cropped, format="PNG").alias("object"))
    assert uppercase.fetchall() == [(PILLOW_11_3_RGB_3X2_COMPRESS_LEVEL_2,)]
    with pytest.raises(vane.NotImplementedException, match="supported formats: png"):
        relation.select(image.encode(cropped, format="jpeg"))

    relation_with_format = relation.select(vane.col("frame"), vane.col("bbox"), vane.lit("png").alias("format"))
    dynamic_format = vane.FunctionExpression("image_encode", cropped, vane.col("format"))
    with pytest.raises(vane.BinderException, match="format must be a constant string"):
        relation_with_format.select(dynamic_format)


def test_image_encode_python_api_rejects_non_string_format():
    with pytest.raises(TypeError, match="format must be a string"):
        image.encode(vane.col("image"), format=1)  # type: ignore[arg-type]


def test_image_type_and_encode_survive_distributed_plan_serialization(duckdb_cursor):
    relation = duckdb_cursor.sql(
        """
        SELECT struct_pack(
            width := 3::UINTEGER,
            height := 2::UINTEGER,
            pixels := from_hex('000102030405060708090a0b0c0d0e0f1011')
        )::IMAGE('RGB8') AS image
        """
    ).select(
        image.encode(vane.col("image")).alias("encoded"),
        vane.col("image"),
    )
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "native-image-serialization")
    restored_logical = pickle.loads(pickle.dumps(logical))
    target = vane.connect()
    try:
        physical = restored_logical.to_physical_plan(target)
        restored_physical = pickle.loads(pickle.dumps(physical))
        result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(
            target.cursor(),
            restored_physical,
        )

        assert result.completion_status == "ok"
        assert result.result_schema == {
            "names": ["col_0", "col_1"],
            "types": ["BLOB", "IMAGE(RGB8)"],
        }
        assert len(result.partition_payloads) == 1
        payload = result.partition_payloads[0]
        assert payload.to_pylist() == [
            {
                "c0": PILLOW_11_3_RGB_3X2_COMPRESS_LEVEL_2,
                "c1": {"width": 3, "height": 2, "pixels": bytes(range(18))},
            }
        ]
        assert payload.schema.field("c1").metadata[b"ARROW:extension:name"] == b"vane.image"
    finally:
        target.close()


def test_crop_and_encode_survive_distributed_plan_serialization(duckdb_cursor):
    frames = np.arange(2 * 3 * 3, dtype=np.uint8).reshape((1, 2, 3, 3))
    _relation_for_frames(duckdb_cursor, frames, [[0.0, 0.0, 3.0, 2.0]]).create("native_image_plan_input")
    cropped = image.crop(vane.col("frame"), vane.col("bbox"))
    relation = duckdb_cursor.table("native_image_plan_input").select(image.encode(cropped).alias("encoded"))
    logical = vane.ray_cxx.PyLogicalPlan.from_duckdb_relation(relation, "native-image-crop-serialization")
    restored_logical = pickle.loads(pickle.dumps(logical))
    target = duckdb_cursor.cursor()
    try:
        physical = restored_logical.to_physical_plan(target)
        restored_physical = pickle.loads(pickle.dumps(physical))
        result = vane.ray_cxx.DistributedPhysicalPlanRunner().execute_native(
            target,
            restored_physical,
        )

        assert result.completion_status == "ok"
        assert result.result_schema == {"names": ["col_0"], "types": ["BLOB"]}
        assert len(result.partition_payloads) == 1
        assert result.partition_payloads[0].to_pylist() == [{"c0": PILLOW_11_3_RGB_3X2_COMPRESS_LEVEL_2}]
    finally:
        target.close()
