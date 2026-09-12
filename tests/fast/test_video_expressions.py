# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

import vane
from tests.fast import test_read_video_frames as streaming
from tests.image_helpers import assert_image_equal

video_path = streaming.video_path
video_connection = streaming.video_connection


@pytest.fixture(
    params=[
        ("libx264", "yuv420p", None),
        ("libx264", "yuv420p", (1, 1, 1, 1)),
        ("libx264", "yuv420p", (1, 1, 1, 2)),
        ("libx264", "yuv444p", (1, 1, 1, 1)),
        ("ffv1", "yuv420p10le", (9, 9, 16, 1)),
        ("ffv1", "bgr0", None),
    ],
    ids=["420", "709-limited", "709-full", "444", "2020-10bit", "rgb"],
)
def detailed_video(request, tmp_path):
    return _detailed_video(tmp_path, *request.param)


def _detailed_video(tmp_path, codec, pixel_format, color, *, interlaced=False):
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL.Image")
    pytest.importorskip("psutil")
    path = tmp_path / ("details.mp4" if codec == "libx264" else "details.mkv")
    with av.open(str(path), "w") as output:
        stream = output.add_stream(codec, rate=8)
        stream.width, stream.height, stream.pix_fmt = 64, 48, pixel_format
        if codec == "libx264":
            stream.codec_context.gop_size = 3
            stream.codec_context.max_b_frames = 2
            stream.options = {"crf": "18", "sc_threshold": "0", "threads": "1"}
            if interlaced:
                stream.options["x264-params"] = "interlaced=1:tff=1"
        if color is not None:
            context = stream.codec_context
            context.colorspace, context.color_primaries, context.color_trc, context.color_range = color
        y, x = np.indices((48, 64))
        for index in range(6):
            # Spatial color detail exposes chroma interpolation differences that
            # the small, uniform grayscale streaming fixture cannot detect.
            pixels = np.stack(
                ((x * 7 + index * 19) % 256, (y * 11 + index * 31) % 256, ((x ^ y) * 13 + index * 47) % 256),
                axis=-1,
            ).astype("uint8")
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return vane.VideoFile(str(path))


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("width,height", [(64, 48), (23, 17), (97, 65), (1, 1), (1, 17), (23, 1)])
def test_video_pixels_match_across_backends(detailed_video, width, height):
    _assert_video_pixels_match_across_backends(detailed_video, width, height)


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize("width,height", [(64, 48), (23, 17), (97, 65), (1, 17)])
def test_video_interlaced_pixels_match_across_backends(tmp_path, width, height):
    file = _detailed_video(tmp_path, "libx264", "yuv420p", (1, 1, 1, 1), interlaced=True)
    import av

    with av.open(file.url) as container:
        assert all(frame.interlaced_frame for frame in container.decode(video=0))
    _assert_video_pixels_match_across_backends(file, width, height)


def _assert_video_pixels_match_across_backends(detailed_video, width, height):
    from tests.fast import test_native_media_extensions as media

    def pixels(image):
        assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
        return "RGB", image.shape[1], image.shape[0], image.tobytes()

    # A backend change must preserve all RGB bytes, including on odd and narrow
    # output sizes. DTS is decoder provenance and is tested separately.
    with media._connect("video") as native, vane.connect(config={"video_backend": "python"}) as python:
        frames = list(detailed_video.frames(width=width, height=height))
        expected = [
            (frame.frame_index, frame.frame_pts, ("RGB", width, height, frame.data.tobytes())) for frame in frames
        ]
        assert len(expected) == 6
        query = "SELECT video_frames($1, width => $2, height => $3, index => $4)"
        index = native.execute("SELECT build_video_index($1)", [detailed_video]).fetchone()[0]
        for connection, seek_index in [(python, None), (native, None), (native, index)]:
            actual = connection.execute(query, [detailed_video, width, height, seek_index]).fetchone()[0]
            assert [(frame["frame_index"], frame["frame_pts"], pixels(frame["data"])) for frame in actual] == expected
            if (width, height) == (64, 48):
                for target in (0, 5):
                    image = connection.execute(
                        "SELECT get_video_frame_by_idx($1, $2, index => $3)",
                        [detailed_video, target, seek_index],
                    ).fetchone()[0]
                    assert pixels(image) == expected[target][2]
            keyframes = connection.execute(
                "SELECT video_keyframes($1, width => $2, height => $3, index => $4)",
                [detailed_video, width, height, seek_index],
            ).fetchone()[0]
            assert [pixels(image) for image in keyframes] == [
                expected[i][2] for i, frame in enumerate(frames) if frame.is_key_frame
            ]
            relation = vane.read_video_frames(
                detailed_video,
                height,
                width,
                connection=connection,
                indexes=[seek_index] if seek_index is not None else None,
            )
            actual = relation.project("frame_index, frame_pts, data").fetchall()
            assert [(ordinal, pts, pixels(image)) for ordinal, pts, image in actual] == expected


def test_native_video_scalar_requires_the_loaded_extension():
    with vane.connect(config={"video_backend": "native"}) as con:
        for expression in ["video_frames(NULL)", "video_keyframes(NULL)", "get_video_frame_by_idx(NULL, 0)"]:
            with pytest.raises(vane.BinderException, match="requires the video extension"):
                con.sql(f"SELECT {expression}")


@pytest.mark.parametrize(
    "source",
    ["'unopened://missing'", "file('unopened://missing', NULL, NULL, NULL, NULL)", "image_file('unopened://missing')"],
)
def test_video_scalar_requires_exact_videofile_logical_type(video_connection, source):
    with pytest.raises(vane.BinderException, match="requires VIDEOFILE"):
        video_connection.sql(f"SELECT video_frames({source})")


@pytest.mark.usefixtures("ray_query")
def test_video_frame_list_has_image_pixels_and_source_metadata(video_connection, video_path):
    file = vane.VideoFile(str(video_path), "video/mp4")
    relation = video_connection.sql(
        "SELECT video_frames($1, start_time => 0.5, end_time => 2, width => 8, height => 6, "
        "sample_interval_seconds => 0.5) AS frames",
        params=[file],
    )
    assert relation.types[0].children[0][1].children[-1] == ("data", vane.image_type("RGB"))
    frames = relation.fetchone()[0]
    assert_image_equal([frame["frame_index"] for frame in frames], [2, 4, 6, 8])
    assert_image_equal([frame["frame_time"] for frame in frames], [0.5, 1, 1.5, 2])
    for frame in frames:
        assert frame["file"] == file
        assert frame["frame_time_base_numerator"] == 1
        assert frame["frame_time_base_denominator"] > 0
        assert isinstance(frame["is_key_frame"], bool)
        assert frame["frame_pts"] is not None
        image = frame["data"]
        assert isinstance(image, np.ndarray)
        assert_image_equal(image.shape, (6, 8, 3))
        assert_image_equal(image.dtype, np.uint8)
        assert_image_equal(image.nbytes, 144)


@pytest.mark.usefixtures("ray_query")
def test_video_function_and_expression_forms_share_options(video_connection, video_path):
    source = video_connection.sql("SELECT video_file($1) AS file", params=[str(video_path)])
    options = dict(start_time=0.5, end_time=2, width=8, height=6, sample_interval_seconds=0.5)
    functional = source.select(vane.video_frames(vane.col("file"), **options)).fetchone()[0]
    method = source.select(vane.col("file").video_frames(**options)).fetchone()[0]
    assert_image_equal(functional, method)
    functional_keys = source.select(vane.video_keyframes(vane.col("file"), **options)).fetchone()[0]
    method_keys = source.select(vane.col("file").video_keyframes(**options)).fetchone()[0]
    assert_image_equal(functional_keys, method_keys)


@pytest.mark.usefixtures("ray_query")
def test_video_keyframes_and_exact_index_match_the_same_backend(video_connection, video_path):
    con = video_connection
    file = vane.VideoFile(str(video_path))
    frames = con.execute("SELECT video_frames($1)", [file]).fetchone()[0]
    keys = con.execute("SELECT video_keyframes($1)", [file]).fetchone()[0]
    assert_image_equal(keys, [frame["data"] for frame in frames if frame["is_key_frame"]])
    assert keys and len(keys) < len(frames)
    for index in [0, 5, 11]:
        image = con.execute("SELECT get_video_frame_by_idx($1, $2)", [file, index]).fetchone()[0]
        assert_image_equal(image, frames[index]["data"])
        assert_image_equal(
            con.sql("SELECT $1 AS file", params=[file])
            .select(vane.get_video_frame_by_idx(vane.col("file"), index))
            .fetchone(),
            (image,),
        )


@pytest.mark.usefixtures("ray_query")
def test_video_frame_expressions_keep_governed_byte_windows(video_connection, video_path, tmp_path):
    payload = video_path.read_bytes()
    prefix = b"outside logical view\0" * 13
    path = tmp_path / "frames'bounded.bin"
    path.write_bytes(prefix + payload + b"outside suffix")
    file = vane.VideoFile(str(path), "video/mp4", len(prefix), len(payload), "sha256:opaque")
    frames = video_connection.execute("SELECT video_frames($1, end_time => 0.25)", [file]).fetchone()[0]
    assert_image_equal([frame["frame_index"] for frame in frames], [0, 1])
    assert all(frame["file"] == file for frame in frames)


@pytest.mark.usefixtures("ray_query")
def test_video_scalar_backend_dispatch_and_lazy_construction(video_connection, video_path, monkeypatch):
    import vane._video_expressions as helpers

    con = video_connection
    backend = con.execute("SELECT current_setting('video_backend')").fetchone()[0]
    calls = []
    original = helpers._scalar_video_frames

    def observe(file, options, execution_context, reserve, index=None):
        assert backend == "python", "native scalar invoked a Python codec helper"
        assert execution_context is not None
        calls.append(file.url)
        return original(file, options, execution_context, reserve, index)

    monkeypatch.setattr(helpers, "_scalar_video_frames", observe)
    query = "SELECT get_video_frame_by_idx(video_file(path), 0) FROM (SELECT 'unopened://missing' path)"
    relation = con.sql(query)
    assert calls == []
    assert relation.types == [vane.image_type("RGB")]
    plan = con.execute("EXPLAIN (FORMAT JSON) " + query).fetchone()[1].lower()
    assert ("native_get_video_frame_by_idx" if backend == "native" else "_vane_get_video_frame_by_idx") in plan
    assert calls == []
    image = con.execute("SELECT get_video_frame_by_idx(video_file($1), 0)", [str(video_path)]).fetchone()[0]
    assert isinstance(image, np.ndarray)
    assert bool(calls) == (backend == "python")


@pytest.mark.usefixtures("ray_query")
def test_video_frame_expression_null_and_empty_selections(video_connection, video_path):
    con = video_connection
    assert_image_equal(
        con.execute(
            "SELECT video_frames(NULL), video_keyframes(NULL), get_video_frame_by_idx(NULL, 0), "
            "get_video_frame_by_idx(video_file('unopened://missing'), NULL)"
        ).fetchone(),
        (None, None, None, None),
    )
    assert con.execute(
        "SELECT video_frames(video_file($1), start_time => 99), video_keyframes(video_file($1), start_time => 99)",
        [str(video_path)],
    ).fetchone() == ([], [])


@pytest.mark.usefixtures("ray_query")
def test_video_scalar_io_obeys_the_query_connection(video_connection, video_path):
    video_connection.execute("SET enable_external_access = false")
    with pytest.raises(vane.PermissionException, match="[Dd]isabled|[Ee]xternal access"):
        video_connection.execute(
            "SELECT get_video_frame_by_idx(video_file($1), 0, on_error => 'null')", [str(video_path)]
        )
    assert_image_equal(video_connection.execute("SELECT 42").fetchone(), (42,))


@pytest.mark.usefixtures("ray_query")
def test_python_video_scalar_revokes_escaped_query_capabilities(video_path, monkeypatch):
    import vane._video_expressions as helpers

    retained = []
    original = helpers._scalar_video_frames

    def observe(file, options, execution_context, reserve, index=None):
        retained.append((execution_context, reserve))
        return original(file, options, execution_context, reserve, index)

    monkeypatch.setattr(helpers, "_scalar_video_frames", observe)
    with vane.connect() as con:
        assert isinstance(
            con.execute("SELECT get_video_frame_by_idx(video_file($1), 0)", [str(video_path)]).fetchone()[0], np.ndarray
        )
        assert len(retained) == 1
        token, reserve = retained[0]
        with pytest.raises(vane.InvalidInputException, match="no longer active"):
            token._check_interrupted()
        with pytest.raises(vane.InvalidInputException, match="no longer active"):
            token._capture_video_error(RuntimeError("expired"))
        with pytest.raises(vane.InvalidInputException, match="no longer active"):
            reserve(8, 6)


@pytest.mark.usefixtures("ray_query")
def test_video_scalar_mixed_null_and_valid_rows(video_connection, video_path):
    rows = video_connection.execute(
        "SELECT i, get_video_frame_by_idx(video_file($1), CASE WHEN i = 0 THEN NULL ELSE 1 END) "
        "FROM range(3) t(i) ORDER BY i",
        [str(video_path)],
    ).fetchall()
    assert_image_equal(rows[0], (0, None))
    assert isinstance(rows[1][1], np.ndarray)
    assert_image_equal(rows[1][1], rows[2][1])


@pytest.mark.usefixtures("ray_query")
def test_video_scalar_format_policy_keeps_io_and_limits_visible(video_connection, video_path, tmp_path):
    con = video_connection
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not a video")
    for expression in ["video_frames", "video_keyframes", "get_video_frame_by_idx"]:
        index = ", 0" if expression == "get_video_frame_by_idx" else ""
        query = f"SELECT {expression}(video_file($1){index}, on_error => 'null')"
        assert con.execute(query, [str(invalid)]).fetchone() == (None,)
        with pytest.raises(vane.Error):
            con.execute(query, [str(tmp_path / "missing.mp4")])
        with pytest.raises(vane.Error):
            con.execute(query.replace("'null'", "'raise'"), [str(invalid)])
    for option in [
        "max_input_bytes => 1",
        "max_output_bytes => 1",
        "max_output_frames => 1",
        "max_decoded_frames => 1",
    ]:
        with pytest.raises(vane.Error):
            con.execute(f"SELECT video_frames(video_file($1), on_error => 'null', {option})", [str(video_path)])
    assert con.execute(
        "SELECT get_video_frame_by_idx(video_file($1), 12, on_error => 'null')", [str(video_path)]
    ).fetchone() == (None,)
    with pytest.raises(vane.Error, match="out of range"):
        con.execute("SELECT get_video_frame_by_idx(video_file($1), 12)", [str(video_path)])
    with pytest.raises(vane.Error, match="max_decoded_frames"):
        con.execute(
            "SELECT get_video_frame_by_idx(video_file($1), 5, max_decoded_frames => 5, on_error => 'null')",
            [str(video_path)],
        )


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize(
    "options",
    [
        "width => 8",
        "width => 0, height => 6",
        "start_time => -1",
        "start_time => 2, end_time => 1",
        "sample_interval_seconds => 0",
        "max_pixels => 33554433",
        "max_output_bytes => 268435457",
        "max_output_frames => 100001",
        "on_error => 'skip'",
        "unknown_option => 1",
    ],
)
def test_video_scalar_rejects_invalid_options(video_connection, options):
    with pytest.raises(vane.Error):
        video_connection.execute(f"SELECT video_frames(video_file('unopened://missing'), {options})")


@pytest.mark.usefixtures("ray_query")
def test_video_scalar_preserves_nested_image_udf_contract(video_connection, video_path):
    con = video_connection
    dtype = vane.list_type(vane.image_type("RGB"))

    @vane.func(return_dtype=dtype)
    def identity(images):
        assert all(isinstance(image, np.ndarray) for image in images)
        return images

    vane.attach_function(identity, connection=con, alias="frame_identity", parameters=[dtype])
    keys = con.execute("SELECT video_keyframes(video_file($1))", [str(video_path)]).fetchone()[0]
    result = con.execute("SELECT frame_identity(video_keyframes(video_file($1)))", [str(video_path)]).fetchone()[0]
    assert_image_equal(result, keys)


@pytest.mark.usefixtures("ray_query")
@pytest.mark.parametrize(
    "failure,error_type",
    [
        (MemoryError("pixel allocation"), vane.OutOfMemoryException),
        (OSError("reader failed"), vane.IOException),
        (KeyboardInterrupt(), vane.InterruptException),
        (vane.InterruptException("reader interrupted"), vane.InterruptException),
        (vane.OutOfRangeException("reader limit"), vane.OutOfRangeException),
        (vane.NotImplementedException("reader cannot seek"), vane.NotImplementedException),
        (vane.InvalidInputException("reader options"), vane.InvalidInputException),
    ],
)
def test_python_video_scalar_format_policy_preserves_system_failures(monkeypatch, failure, error_type):
    import vane._video_expressions as helpers

    def fail(*args):
        raise failure

    monkeypatch.setattr(helpers, "_scalar_video_frames", fail)
    with vane.connect() as con, pytest.raises(error_type):
        con.execute("SELECT video_frames(video_file('unopened://source'), on_error => 'null')")


@pytest.mark.usefixtures("ray_query")
def test_python_video_scalar_observes_pending_interrupt_before_null_policy(monkeypatch):
    import vane._video_expressions as helpers

    with vane.connect() as con:

        def fail(*args):
            con.interrupt()
            raise vane.VideoFileFormatError("competing format error")

        monkeypatch.setattr(helpers, "_scalar_video_frames", fail)
        with pytest.raises(vane.InterruptException):
            con.execute("SELECT video_frames(video_file('unopened://source'), on_error => 'null')")


@pytest.mark.usefixtures("ray_query")
def test_python_video_scalar_cleanup_cannot_mask_pixel_allocation_failure(monkeypatch):
    from types import SimpleNamespace

    import vane._video_expressions as helpers

    class BrokenFrames:
        def __iter__(self):
            return self

        def __next__(self):
            return SimpleNamespace(data=SimpleNamespace(mode="RGB", size=(1, 1), tobytes=self.fail))

        def fail(self):
            raise MemoryError("pixel allocation")

        def close(self):
            raise vane.VideoFileFormatError("cleanup failed")

    monkeypatch.setattr(helpers, "_iter_video_frames", lambda *args: BrokenFrames())
    monkeypatch.setattr(helpers, "_load_av", lambda: None)
    monkeypatch.setattr(helpers, "_load_pillow", lambda: None)
    with vane.connect() as con, pytest.raises(vane.OutOfMemoryException, match="ran out of memory"):
        con.execute("SELECT video_frames(video_file('unopened://source'), on_error => 'null')")


@pytest.mark.usefixtures("ray_query")
def test_video_scalar_bounds_total_output_for_a_chunk(video_connection, video_path):
    with pytest.raises(vane.OutOfRangeException, match="batch exceeds 256 MiB"):
        video_connection.execute(
            "SELECT sum(len(video_frames(video_file($1), width => 128, height => 128))) FROM range(512)",
            [str(video_path)],
        )
    assert_image_equal(video_connection.execute("SELECT 42").fetchone(), (42,))


@pytest.mark.usefixtures("ray_query")
def test_video_scalar_list_vectors_reset_between_chunks(video_connection, video_path):
    rows = video_connection.execute(
        "SELECT count(*), sum(len(frames)), sum(frames[1].frame_index) FROM ("
        "SELECT video_frames(CASE WHEN i % 3 = 0 THEN NULL ELSE video_file($1) END, end_time => 0) AS frames "
        "FROM range(2049) t(i)) videos",
        [str(video_path)],
    ).fetchone()
    assert_image_equal(rows, (2049, 1366, 0))
