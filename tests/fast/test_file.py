# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import functools
import http.server
import os
import pickle
import socket
import threading

import pandas as pd
import pytest

import vane

FILE_FIELDS = ("url", "content_type", "position", "size", "checksum")
FILE_METHODS = ("exists", "mime_type", "open", "stat", "to_tempfile")
IMAGE_FILE_METHODS = FILE_METHODS + ("decode", "metadata")
AUDIO_FILE_METHODS = FILE_METHODS + ("metadata", "to_numpy")
MEDIA_FILE_CASES = (
    ("image", "IMAGEFILE", vane.ImageFile, vane.image_file),
    ("audio", "AUDIOFILE", vane.AudioFile, vane.audio_file),
    ("video", "VIDEOFILE", vane.VideoFile, vane.video_file),
)
FILE_METHODS_BY_CLASS = {
    vane.File: FILE_METHODS,
    vane.ImageFile: IMAGE_FILE_METHODS,
    vane.AudioFile: AUDIO_FILE_METHODS,
    vane.VideoFile: FILE_METHODS,
}


def test_file_value_contract():
    minimal = vane.File("s3://bucket/object")
    assert vane.File is vane._native.File
    assert tuple(getattr(minimal, field) for field in FILE_FIELDS) == (
        "s3://bucket/object",
        None,
        None,
        None,
        None,
    )
    assert str(minimal) == "s3://bucket/object"
    assert repr(minimal) == "File(url='s3://bucket/object', content_type=None, position=None, size=None, checksum=None)"
    assert {name for name in dir(minimal) if not name.startswith("_")} == set(FILE_FIELDS + FILE_METHODS)

    complete = vane.File("memory://part", "text/plain", 2, 4, "sha256:abcd")
    assert tuple(getattr(complete, field) for field in FILE_FIELDS) == (
        "memory://part",
        "text/plain",
        2,
        4,
        "sha256:abcd",
    )
    assert complete == vane.File("memory://part", "text/plain", 2, 4, "sha256:abcd")
    assert complete != minimal
    assert complete != "memory://part"
    assert hash(complete) == hash(pickle.loads(pickle.dumps(complete)))
    assert pickle.loads(pickle.dumps(complete)) == complete


def test_file_value_is_immutable_and_final():
    value = vane.File("memory://value")
    for field in FILE_FIELDS:
        with pytest.raises(AttributeError):
            setattr(value, field, None)
    with pytest.raises(AttributeError):
        value.secret = "credential"
    with pytest.raises(TypeError):

        class DerivedFile(vane.File):
            pass


@pytest.mark.parametrize("url", [None, 42, b"path"])
def test_file_value_requires_string_url(url):
    with pytest.raises(TypeError, match=r"File\.url must be str"):
        vane.File(url)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"content_type": 1}, "content_type"),
        ({"checksum": b"sha256:a"}, "checksum"),
        ({"position": True, "size": 0}, "position"),
        ({"position": 0, "size": False}, "size"),
        ({"position": 0.0, "size": 1}, "position"),
        ({"position": 0, "size": "1"}, "size"),
    ],
)
def test_file_value_requires_exact_optional_types(kwargs, field):
    with pytest.raises(TypeError, match=rf"File\.{field} must be"):
        vane.File("memory://value", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"position": 0},
        {"size": 0},
        {"position": -1, "size": 0},
        {"position": 0, "size": -1},
        {"position": 2**63 - 1, "size": 1},
    ],
)
def test_file_value_rejects_invalid_ranges(kwargs):
    with pytest.raises(ValueError):
        vane.File("memory://value", **kwargs)


@pytest.mark.parametrize("field", ["position", "size"])
def test_file_value_rejects_bigint_overflow(field):
    kwargs = {"position": 0, "size": 0, field: 2**63}
    with pytest.raises(OverflowError, match="signed 64-bit"):
        vane.File("memory://value", **kwargs)


@pytest.mark.parametrize("checksum", ["", "sha256", ":digest", "sha256:", "sha256:a:b", "sha256:a\0b"])
def test_file_value_rejects_invalid_checksums(checksum):
    with pytest.raises(ValueError, match=r"<algorithm>:<digest>"):
        vane.File("memory://value", checksum=checksum)


def test_file_value_rejects_nul_url():
    with pytest.raises(ValueError, match="NUL"):
        vane.File("memory://bad\0url")


def test_file_data_type_contract():
    dtype = vane.file_type()
    plain_struct = vane.struct_type(
        {
            "url": vane.sqltypes.VARCHAR,
            "content_type": vane.sqltypes.VARCHAR,
            "position": vane.sqltypes.BIGINT,
            "size": vane.sqltypes.BIGINT,
            "checksum": vane.sqltypes.VARCHAR,
        }
    )

    assert str(dtype) == "FILE"
    assert dtype.id == "struct"
    assert dtype.is_file()
    assert dtype == vane.sqltype("FILE")
    assert not plain_struct.is_file()
    assert dtype.children == [
        ("url", vane.sqltypes.VARCHAR),
        ("content_type", vane.sqltypes.VARCHAR),
        ("position", vane.sqltypes.BIGINT),
        ("size", vane.sqltypes.BIGINT),
        ("checksum", vane.sqltypes.VARCHAR),
    ]
    assert dtype.url == vane.sqltypes.VARCHAR
    assert pickle.loads(pickle.dumps(dtype)).is_file()


def test_media_type_contract():
    values = (
        vane.MediaType.unknown(),
        vane.MediaType.image(),
        vane.MediaType.audio(),
        vane.MediaType.video(),
    )

    assert [repr(value) for value in values] == [
        "MediaType.unknown()",
        "MediaType.image()",
        "MediaType.audio()",
        "MediaType.video()",
    ]
    assert len(set(values)) == 4
    assert vane.MediaType.image() == vane.MediaType.image()
    with pytest.raises(TypeError):
        vane.MediaType()


@pytest.mark.parametrize(("media", "type_name", "value_class", "_constructor"), MEDIA_FILE_CASES)
def test_media_file_data_type_contract(media, type_name, value_class, _constructor):
    media_type = getattr(vane.MediaType, media)()
    dtype = vane.file_type(media_type)

    assert str(dtype) == type_name
    assert dtype.id == "struct"
    assert dtype.is_file()
    assert dtype == vane.sqltype(type_name)
    assert dtype != vane.file_type()
    assert pickle.loads(pickle.dumps(dtype)) == dtype
    assert issubclass(value_class, vane.File)


@pytest.mark.parametrize(("media", "type_name", "value_class", "_constructor"), MEDIA_FILE_CASES)
def test_media_file_python_value_contract(media, type_name, value_class, _constructor):
    value = value_class(
        f"memory://{media}",
        f"{media}/test",
        1,
        2,
        f"sha256:{media}",
    )

    assert isinstance(value, vane.File)
    assert type(value) is value_class
    assert tuple(getattr(value, field) for field in FILE_FIELDS) == (
        f"memory://{media}",
        f"{media}/test",
        1,
        2,
        f"sha256:{media}",
    )
    assert repr(value).startswith(f"{value_class.__name__}(url=")
    assert pickle.loads(pickle.dumps(value)) == value
    assert type(pickle.loads(pickle.dumps(value))) is value_class
    generic = vane.File(value.url, value.content_type, value.position, value.size, value.checksum)
    assert value != generic
    assert len({value, generic}) == 2
    expected_methods = FILE_METHODS_BY_CLASS[value_class]
    assert {name for name in dir(value) if not name.startswith("_")} == set(FILE_FIELDS + expected_methods)


def test_media_file_value_inherits_reader_and_metadata_behavior(tmp_path):
    path = tmp_path / "image.bin"
    path.write_bytes(b"prefix-image-suffix")
    value = vane.ImageFile(str(path), "image/png", 7, 5, "sha256:image")

    assert value.exists()
    assert value.stat()["url"] == str(path)
    assert value.mime_type() == "image/png"
    with value.open(buffer_size=2) as reader:
        assert reader.read() == b"image"
    with value.to_tempfile(buffer_size=2) as temporary:
        assert temporary.read() == b"image"


@pytest.mark.parametrize(("media", "type_name", "value_class", "constructor"), MEDIA_FILE_CASES)
def test_media_file_expression_constructor_is_pure_and_materializes_exact_type(
    duckdb_cursor,
    media,
    type_name,
    value_class,
    constructor,
):
    value = (
        duckdb_cursor.sql("SELECT 1").select(constructor("memory://missing", verify=False).alias("value")).fetchone()[0]
    )

    assert type(value) is value_class
    assert value == value_class("memory://missing")
    assert str(duckdb_cursor.sql("SELECT 1").select(constructor("memory://missing")).types[0]) == type_name

    from_method = (
        duckdb_cursor.sql("SELECT 'memory://method' AS url")
        .select(vane.col("url").as_file(getattr(vane.MediaType, media)()))
        .fetchone()[0]
    )
    assert type(from_method) is value_class
    assert from_method == value_class("memory://method")


def test_media_file_constructor_accepts_bound_string_parameter(duckdb_cursor):
    row = duckdb_cursor.execute("SELECT image_file(?)", ["memory://parameter"]).fetchone()

    assert row == (vane.ImageFile("memory://parameter"),)


def test_file_family_values_render_in_relation_boxes(duckdb_cursor, capsys):
    query = """
        SELECT
            file('generic-' || i, NULL, NULL, NULL, NULL) AS generic_file,
            image_file('image-' || i) AS image_file,
            [audio_file('audio-' || i)] AS nested_audio_files
        FROM range(3) AS source(i)
    """

    rendered = str(duckdb_cursor.sql(query))
    assert "generic-0" in rendered
    assert "generic-2" in rendered
    assert "image-0" in rendered
    assert "image-2" in rendered
    assert "audio-0" in rendered
    assert "audio-2" in rendered

    duckdb_cursor.sql(query).show(max_rows=2)
    shown = capsys.readouterr().out
    assert "generic-0" in shown
    assert "generic-2" in shown
    assert "image-0" in shown
    assert "image-2" in shown
    assert "audio-0" in shown
    assert "audio-2" in shown


def test_media_file_expression_and_generic_file_functions(duckdb_cursor):
    image = vane.ImageFile("memory://image.png", "image/png", 2, 4, "sha256:image")
    row = duckdb_cursor.sql(
        """
        SELECT
            image_file(file('memory://image.png', 'image/png', 2, 4, 'sha256:image')) AS image,
            [audio_file('memory://audio.mp3'), NULL::AUDIOFILE] AS audio,
            struct_pack(video := video_file('memory://video.mp4')) AS nested,
            typeof(file_enrich(image_file('memory://image.png'), [])) AS enriched_type,
            file_path(image_file('memory://image.png')) AS path,
            file_same_location(image_file('memory://same'), video_file('memory://same')) AS same_location
        """
    ).fetchone()

    assert row == (
        image,
        [vane.AudioFile("memory://audio.mp3"), None],
        {"video": vane.VideoFile("memory://video.mp4")},
        "IMAGEFILE",
        "memory://image.png",
        True,
    )


def test_file_family_functions_keep_untyped_null_calls_unambiguous(duckdb_cursor):
    row = duckdb_cursor.sql(
        """
        SELECT
            typeof(file_enrich(NULL, [])),
            file_path(NULL),
            file_mime_type(NULL),
            file_same_location(NULL, image_file('memory://image')),
            file_same_content(video_file('memory://video'), NULL)
        """
    ).fetchone()

    assert row == ("FILE", None, None, None, None)


@pytest.mark.parametrize(
    ("value", "dtype", "expected_type"),
    [
        (
            [vane.ImageFile("memory://image"), None],
            vane.list_type(vane.file_type(vane.MediaType.image())),
            "IMAGEFILE[]",
        ),
        (
            [vane.AudioFile("memory://audio")],
            vane.array_type(vane.file_type(vane.MediaType.audio()), 1),
            "AUDIOFILE[1]",
        ),
        (
            {"media": vane.VideoFile("memory://video")},
            vane.struct_type({"media": vane.file_type(vane.MediaType.video())}),
            "STRUCT(media VIDEOFILE)",
        ),
        (
            {"image": vane.ImageFile("memory://image")},
            vane.map_type(vane.sqltypes.VARCHAR, vane.file_type(vane.MediaType.image())),
            "MAP(VARCHAR, IMAGEFILE)",
        ),
        (
            [vane.AudioFile("memory://audio")],
            vane.tensor_type(vane.file_type(vane.MediaType.audio()), (1,)),
            "TENSOR(AUDIOFILE, [1])",
        ),
    ],
)
def test_declared_nested_media_file_types_preserve_specialization(duckdb_cursor, value, dtype, expected_type):
    actual_type, actual = duckdb_cursor.execute("SELECT typeof($1), $1", [vane.Value(value, dtype)]).fetchone()

    assert actual_type == expected_type
    if isinstance(value, list) and expected_type.endswith("[1]"):
        assert actual == tuple(value)
    elif expected_type.startswith("TENSOR"):
        assert actual == tuple(value)
    else:
        assert actual == value


def test_declared_media_file_value_requires_matching_python_subclass(duckdb_cursor):
    image_type = vane.file_type(vane.MediaType.image())

    for value in (vane.File("memory://generic"), vane.AudioFile("memory://audio")):
        with pytest.raises(vane.InvalidInputException, match="cannot be converted to IMAGEFILE"):
            duckdb_cursor.execute("SELECT $1", [vane.Value(value, image_type)]).fetchone()


def test_media_file_direct_comparison_requires_matching_specialization(duckdb_cursor):
    with pytest.raises(vane.BinderException, match="IMAGEFILE.*AUDIOFILE"):
        duckdb_cursor.sql("SELECT image_file('x') = audio_file('x')")
    with pytest.raises(vane.BinderException, match="IMAGEFILE.*FILE"):
        duckdb_cursor.sql("SELECT image_file('x') = file('x', NULL, NULL, NULL, NULL)")
    with pytest.raises(vane.BinderException, match=r"image_file\(\).*AUDIOFILE"):
        duckdb_cursor.sql("SELECT image_file(audio_file('x'))")
    with pytest.raises(vane.BinderException, match=r"image_file\(\).*JSON"):
        duckdb_cursor.sql("SELECT image_file(json '\"memory://image\"')")


@pytest.mark.parametrize(
    ("constructor", "value_class", "suffix", "payload", "content_type"),
    [
        (vane.image_file, vane.ImageFile, ".png", bytes.fromhex("89504e470d0a1a0a"), "image/png"),
        (vane.audio_file, vane.AudioFile, ".mp3", bytes.fromhex("fffb9064"), "audio/mpeg"),
        (vane.video_file, vane.VideoFile, ".mpg", bytes.fromhex("000001ba"), "video/mpeg"),
    ],
)
def test_media_file_verify_uses_bounded_content_detection(
    duckdb_cursor,
    tmp_path,
    constructor,
    value_class,
    suffix,
    payload,
    content_type,
):
    path = tmp_path / f"media{suffix}"
    prefix = b"prefix"
    path.write_bytes(prefix + payload + b"suffix")

    source = vane.file(str(path), position=len(prefix), size=len(payload))
    value = duckdb_cursor.sql("SELECT 1").select(constructor(source, verify=True)).fetchone()[0]

    assert type(value) is value_class
    assert value.content_type == content_type
    assert (value.position, value.size) == (len(prefix), len(payload))


def test_pandas_media_file_columns_preserve_specialization(duckdb_cursor):
    values = [vane.ImageFile("memory://first"), None, vane.ImageFile("memory://second", "image/png")]
    relation = duckdb_cursor.from_df(pd.DataFrame({"value": values}))

    assert str(relation.types[0]) == "IMAGEFILE"
    assert relation.types[0].is_file()
    assert [row[0] for row in relation.fetchall()] == values


def test_local_file_results_materialize_recursively(duckdb_cursor):
    value = vane.File("memory://nested", "text/plain", 1, 2, "sha256:nested")
    row = duckdb_cursor.sql(
        """
        SELECT
            file('memory://nested', 'text/plain', 1, 2, 'sha256:nested'),
            [file('memory://nested', 'text/plain', 1, 2, 'sha256:nested'), NULL::FILE],
            struct_pack(item := file('memory://nested', 'text/plain', 1, 2, 'sha256:nested')),
            map(['item'], [file('memory://nested', 'text/plain', 1, 2, 'sha256:nested')]),
            union_value(item := file('memory://nested', 'text/plain', 1, 2, 'sha256:nested'))
        """
    ).fetchone()

    assert row == (value, [value, None], {"item": value}, {"item": value}, value)


def test_file_dataframe_results_use_file_values(duckdb_cursor):
    frame = duckdb_cursor.sql(
        """
        SELECT value
        FROM (
            SELECT 0 AS sort_key, file('memory://a', 'text/plain', 0, 4, 'sha256:a') AS value
            UNION ALL
            SELECT 1, NULL::FILE
        )
        ORDER BY sort_key
        """
    ).df()

    values = frame["value"].tolist()
    assert values[0] == vane.File("memory://a", "text/plain", 0, 4, "sha256:a")
    assert pd.isna(values[1])


def test_pandas_file_columns_preserve_file_values(duckdb_cursor):
    values = [vane.File("memory://first"), None, vane.File("memory://second", "text/plain")]
    relation = duckdb_cursor.from_df(pd.DataFrame({"value": values}))

    assert relation.types[0].is_file()
    assert [row[0] for row in relation.fetchall()] == values


def test_plain_file_shaped_struct_remains_a_dictionary(duckdb_cursor):
    query = """
        SELECT struct_pack(
            url := 'memory://struct',
            content_type := NULL::VARCHAR,
            "position" := NULL::BIGINT,
            size := NULL::BIGINT,
            checksum := NULL::VARCHAR
        ) AS value
    """
    expected = {
        "url": "memory://struct",
        "content_type": None,
        "position": None,
        "size": None,
        "checksum": None,
    }

    value = duckdb_cursor.sql(query).fetchone()[0]
    columnar_value = duckdb_cursor.sql(query).df()["value"].iloc[0]
    assert value == expected
    assert columnar_value == expected
    assert not isinstance(value, vane.File)
    assert not isinstance(columnar_value, vane.File)


@pytest.mark.parametrize(
    ("value", "dtype", "expected"),
    [
        (None, vane.file_type(), None),
        (None, vane.list_type(vane.file_type()), None),
        ([], vane.list_type(vane.file_type()), []),
        ([None], vane.list_type(vane.file_type()), [None]),
        ([None], vane.array_type(vane.file_type(), 1), (None,)),
        ({"file": None}, vane.struct_type({"file": vane.file_type()}), {"file": None}),
        ({}, vane.map_type(vane.sqltypes.VARCHAR, vane.file_type()), {}),
        (
            {"key": ["one"], "value": [None]},
            vane.map_type(vane.sqltypes.VARCHAR, vane.file_type()),
            {"one": None},
        ),
        ([[]], vane.list_type(vane.list_type(vane.file_type())), [[]]),
    ],
)
def test_declared_types_preserve_file_through_empty_and_null_values(duckdb_cursor, value, dtype, expected):
    actual_type, actual = duckdb_cursor.execute("SELECT typeof($1), $1", [vane.Value(value, dtype)]).fetchone()
    assert actual_type == str(dtype)
    assert actual == expected


def test_file_values_cross_explicit_python_boundaries(duckdb_cursor):
    value = vane.File("memory://parameter", "text/plain", 0, 3, "sha256:parameter")

    relation = duckdb_cursor.values(vane.ConstantExpression(value))
    assert relation.types[0].is_file()
    assert relation.fetchone() == (value,)

    duckdb_cursor.execute("CREATE TABLE file_parameters(value FILE)")
    duckdb_cursor.execute("INSERT INTO file_parameters VALUES (?)", [value])
    assert duckdb_cursor.sql("SELECT value FROM file_parameters").fetchone() == (value,)
    assert duckdb_cursor.execute("SELECT ?", [[value, None]]).fetchone() == ([value, None],)


def test_typed_union_selects_file_and_composite_members(duckdb_cursor):
    value = vane.File("memory://union")
    plain_struct_type = vane.struct_type({"item": vane.file_type()})
    list_type = vane.list_type(vane.file_type())
    array_type = vane.array_type(vane.file_type(), 1)
    map_type = vane.map_type(vane.file_type(), vane.sqltypes.INTEGER)
    cases = [
        (
            value,
            vane.union_type({"text": vane.sqltypes.VARCHAR, "file": vane.file_type()}),
            "file",
            value,
        ),
        (
            [value],
            vane.union_type({"text": vane.sqltypes.VARCHAR, "array": array_type}),
            "array",
            (value,),
        ),
        (
            [value, None],
            vane.union_type({"text": vane.sqltypes.VARCHAR, "files": list_type}),
            "files",
            [value, None],
        ),
        (
            {"item": value},
            vane.union_type({"text": vane.sqltypes.VARCHAR, "record": plain_struct_type}),
            "record",
            {"item": value},
        ),
        (
            {value: 7},
            vane.union_type({"text": vane.sqltypes.VARCHAR, "mapping": map_type}),
            "mapping",
            {value: 7},
        ),
    ]

    for python_value, dtype, expected_tag, expected_value in cases:
        actual_type, actual_tag, actual_value = duckdb_cursor.execute(
            "SELECT typeof($1), union_tag($1), $1", [vane.Value(python_value, dtype)]
        ).fetchone()
        assert actual_type == str(dtype)
        assert actual_tag == expected_tag
        assert actual_value == expected_value


def test_file_map_keys_materialize_as_hashable_values(duckdb_cursor):
    key = vane.File("memory://map-key", "text/plain", 0, 3, "sha256:key")
    value = vane.Value({key: 42}, vane.map_type(vane.file_type(), vane.sqltypes.BIGINT))

    fetched = duckdb_cursor.execute("SELECT $1", [value]).fetchone()[0]
    columnar = duckdb_cursor.execute("SELECT $1", [value]).df().iloc[0, 0]
    assert fetched == {key: 42}
    assert columnar == {key: 42}


@pytest.mark.parametrize(
    "fallback",
    [
        {
            "url": "memory://dict",
            "content_type": None,
            "position": None,
            "size": None,
            "checksum": None,
        },
        ("memory://tuple", None, None, None, None),
        b"memory://blob",
    ],
)
def test_explicit_file_conversion_rejects_structural_fallbacks(fallback):
    with pytest.raises(vane.InvalidInputException, match="Only vane.File or NULL"):
        vane.ConstantExpression(vane.Value(fallback, vane.file_type()))
    with pytest.raises(vane.InvalidInputException, match="Only vane.File or NULL"):
        vane.ConstantExpression(vane.Value([fallback], vane.list_type(vane.file_type())))


def test_file_expression_metadata_facade(duckdb_cursor, tmp_path):
    path = tmp_path / "facade.txt"
    path.write_text("hello", encoding="utf-8")
    missing = tmp_path / "missing.txt"

    source = duckdb_cursor.sql(
        "SELECT file($1, NULL, NULL, NULL, NULL) AS value",
        params=[str(path)],
    )
    value = vane.col("value")
    row = source.select(
        vane.file_path(value),
        vane.file_size(value),
        vane.file_exists(value),
        vane.file_stat(value),
        vane.file_mime_type(value),
        value.file_path(),
        value.file_size(),
        value.file_exists(),
        value.file_stat(),
        value.file_mime_type(),
    ).fetchone()

    assert row[0:3] == (str(path), 5, True)
    assert row[3]["object_size"] == 5
    assert row[3]["content_type"] == "text/plain"
    assert row[4] == "text/plain"
    assert row[5:] == row[:5]
    assert duckdb_cursor.values(vane.try_to_file(str(missing))).fetchone() == (None,)
    assert duckdb_cursor.values(vane.to_file(str(path))).fetchone() == (vane.File(str(path), "text/plain", 0, 5),)


def test_file_identity_and_enrichment_facade(duckdb_cursor, tmp_path):
    path = tmp_path / "identity.bin"
    path.write_bytes(b"abc")
    value = vane.file(str(path), None, 0, 3)
    enriched = vane.file_enrich(value, ["checksum"])

    row = duckdb_cursor.values(
        vane.file_same_location(value, enriched),
        vane.file_same_content(enriched, enriched),
        vane.file_locator_id(value),
        vane.file_content_id(enriched),
        vane.guess_mime_type(b"\x89PNG\r\n\x1a\n"),
    ).fetchone()

    assert row[0:2] == (True, True)
    assert row[2].startswith("file-locator-v1:sha256:")
    assert row[3].startswith("file-content-v1:checksum:sha256:")
    assert row[4] == "image/png"


def test_concrete_file_metadata_methods_use_sql_contract(tmp_path):
    path = tmp_path / "concrete.json"
    path.write_text("{}", encoding="utf-8")

    value = vane.File(str(path))
    stat = value.stat()

    assert value.exists() is True
    assert value.mime_type() == "application/json"
    assert stat["url"] == str(path)
    assert stat["object_size"] == 2
    assert stat["last_modified"] is not None
    assert stat["version"] is None
    assert stat["etag"] is None
    assert stat["content_type"] == "application/json"
    assert vane.File(str(tmp_path / "missing")).exists() is False


def test_concrete_file_metadata_methods_accept_connection(tmp_path):
    scoped_home = tmp_path / "scoped-home"
    scoped_home.mkdir()
    path = scoped_home / "connection.bin"
    path.write_bytes(b"\x89PNG\r\n\x1a\n")

    connection = vane.connect("")
    try:
        connection.execute("SET home_directory = ?", [str(scoped_home)])
        value = vane.File("~/connection.bin")

        assert value.exists(connection=connection) is True
        assert value.mime_type("content", connection=connection) == "image/png"
        assert value.stat(connection=connection)["object_size"] == 8
    finally:
        connection.close()


def test_list_files_and_from_files_delegate_to_sql(duckdb_cursor, tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    nested_dir = first_dir / "nested"
    first_dir.mkdir()
    second_dir.mkdir()
    nested_dir.mkdir()
    first = first_dir / "a.txt"
    second = first_dir / "b.json"
    nested = nested_dir / "c.txt"
    last = second_dir / "d.txt"
    for path, content in ((first, "a"), (second, "{}"), (nested, "c"), (last, "d")):
        path.write_text(content, encoding="utf-8")

    listed = vane.list_files(str(first_dir), connection=duckdb_cursor).fetchall()
    assert [row[0] for row in listed] == [str(first), str(second)]
    assert [row[6] for row in listed] == [
        vane.File(str(first), "text/plain", 0, 1),
        vane.File(str(second), "application/json", 0, 2),
    ]

    recursive = vane.list_files(str(first_dir), recursive=True, connection=duckdb_cursor).fetchall()
    assert [row[0] for row in recursive] == [str(first), str(second), str(nested)]

    values = vane.from_files(
        [str(last), str(first)],
        connection=duckdb_cursor,
    ).fetchall()
    assert values == [
        (vane.File(str(last), "text/plain", 0, 1),),
        (vane.File(str(first), "text/plain", 0, 1),),
    ]


def test_list_files_empty_glob_and_directory(duckdb_cursor, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    assert vane.list_files(str(empty), connection=duckdb_cursor).fetchall() == []
    assert vane.list_files(str(tmp_path / "*.missing"), connection=duckdb_cursor).fetchall() == []
    assert vane.from_files([], connection=duckdb_cursor).fetchall() == []


def test_list_files_accepts_literal_glob_filename(duckdb_cursor, tmp_path):
    if os.name == "nt":
        pytest.skip("Windows filenames cannot contain a literal asterisk")

    directory = tmp_path / "literal_glob"
    directory.mkdir()
    literal = directory / "*"
    literal.write_text("value", encoding="utf-8")

    rows = vane.list_files(str(directory), connection=duckdb_cursor).fetchall()

    assert [row[0] for row in rows] == [str(literal)]
    assert rows[0][6] == vane.File(str(literal), None, 0, 5)


@pytest.mark.parametrize(
    ("literal_name", "matching_name"),
    [
        ("literal*directory", "literalXdirectory"),
        ("literal?directory", "literalYdirectory"),
        ("literal[d]irectory", "literalddirectory"),
    ],
)
def test_list_files_accepts_literal_glob_directory(duckdb_cursor, tmp_path, literal_name, matching_name):
    if os.name == "nt":
        pytest.skip("Windows directory names cannot contain literal glob characters")

    directory = tmp_path / literal_name
    empty_directory = tmp_path / f"empty-{literal_name}"
    matching_directory = tmp_path / matching_name
    nested = directory / "nested"
    directory.mkdir()
    empty_directory.mkdir()
    matching_directory.mkdir()
    nested.mkdir()
    child = directory / "child.txt"
    descendant = nested / "descendant.txt"
    (matching_directory / "excluded.txt").write_text("excluded", encoding="utf-8")
    child.write_text("child", encoding="utf-8")
    descendant.write_text("descendant", encoding="utf-8")

    rows = vane.list_files(str(directory), connection=duckdb_cursor).fetchall()
    recursive = vane.list_files(str(directory), recursive=True, connection=duckdb_cursor).fetchall()

    assert [row[0] for row in rows] == [str(child)]
    assert [row[0] for row in recursive] == [str(child), str(descendant)]
    assert vane.list_files(str(empty_directory), connection=duckdb_cursor).fetchall() == []


def test_list_files_treats_ipv6_authority_as_literal(duckdb_cursor, tmp_path):
    class IPv6HTTPServer(http.server.ThreadingHTTPServer):
        address_family = socket.AF_INET6

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    path = tmp_path / "value.txt"
    path.write_text("value", encoding="utf-8")
    handler = functools.partial(QuietHandler, directory=str(tmp_path))
    try:
        server = IPv6HTTPServer(("::1", 0), handler)
    except OSError as error:
        pytest.skip(f"IPv6 loopback is unavailable: {error}")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://[::1]:{server.server_port}/value.txt?token=value?.txt"
    try:
        duckdb_cursor.execute("SET http_proxy = ''")
        rows = duckdb_cursor.execute("SELECT url, object_size FROM list_files(?)", [url]).fetchall()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert rows == [(url, 5)]


@pytest.mark.skipif(os.name == "nt", reason="symlink traversal semantics differ on Windows")
def test_list_files_recursive_preserves_file_symlinks_without_following_directory_symlinks(duckdb_cursor, tmp_path):
    root = tmp_path / "root"
    nested = root / "nested"
    outside = tmp_path / "outside"
    root.mkdir()
    nested.mkdir()
    outside.mkdir()
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    regular = root / "regular.txt"
    nested_regular = nested / "regular.txt"
    regular.write_text("regular", encoding="utf-8")
    nested_regular.write_text("nested", encoding="utf-8")
    (outside / "excluded.txt").write_text("excluded", encoding="utf-8")
    direct_link = root / "direct-link.txt"
    nested_link = nested / "nested-link.txt"
    directory_link = root / "directory-link"
    try:
        direct_link.symlink_to(target)
        nested_link.symlink_to(target)
        directory_link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")

    non_recursive = duckdb_cursor.execute("SELECT url FROM list_files(?)", [str(root)]).fetchall()
    recursive = duckdb_cursor.execute("SELECT url FROM list_files(?, TRUE)", [str(root)]).fetchall()

    assert [row[0] for row in non_recursive] == sorted([str(direct_link), str(regular)])
    assert [row[0] for row in recursive] == sorted(
        [str(direct_link), str(nested_link), str(nested_regular), str(regular)]
    )


def test_list_files_honors_file_search_path(duckdb_cursor, tmp_path, monkeypatch):
    search_path = tmp_path / "search"
    directory = search_path / "directory"
    empty = search_path / "empty"
    literal_directory = search_path / "literal[d]irectory"
    matching_directory = search_path / "literalddirectory"
    direct_directory = tmp_path / "direct"
    directory.mkdir(parents=True)
    empty.mkdir()
    literal_directory.mkdir()
    matching_directory.mkdir()
    direct_directory.mkdir()
    concrete = search_path / "concrete.txt"
    child = directory / "child.txt"
    literal_child = literal_directory / "literal.txt"
    direct_child = direct_directory / "direct.txt"
    concrete.write_text("concrete", encoding="utf-8")
    child.write_text("child", encoding="utf-8")
    literal_child.write_text("literal", encoding="utf-8")
    (matching_directory / "excluded.txt").write_text("excluded", encoding="utf-8")
    direct_child.write_text("direct", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    duckdb_cursor.execute("SET file_search_path = ?", [str(search_path)])

    assert [row[0] for row in vane.list_files("concrete.txt", connection=duckdb_cursor).fetchall()] == [str(concrete)]
    assert [row[0] for row in vane.list_files("directory", connection=duckdb_cursor).fetchall()] == [str(child)]
    assert [row[0] for row in vane.list_files("literal[d]irectory", connection=duckdb_cursor).fetchall()] == [
        str(literal_child)
    ]
    direct_rows = vane.list_files("direct", connection=duckdb_cursor).fetchall()
    assert len(direct_rows) == 1
    assert os.path.samefile(direct_rows[0][0], direct_child)
    assert vane.list_files("empty", connection=duckdb_cursor).fetchall() == []
    with pytest.raises(vane.IOException, match="does not exist or is not listable"):
        vane.list_files("missing", connection=duckdb_cursor).fetchall()


@pytest.mark.skipif(os.name == "nt", reason="colon is not valid in a Windows path component")
def test_list_files_treats_embedded_scheme_delimiter_as_local_search_path_text(duckdb_cursor, tmp_path, monkeypatch):
    search_path = tmp_path / "search"
    directory = search_path / "root" / "http:"
    directory.mkdir(parents=True)
    child = directory / "value.txt"
    child.write_text("value", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    duckdb_cursor.execute("SET file_search_path = ?", [str(search_path)])

    rows = duckdb_cursor.execute("SELECT url FROM list_files('root/http://value.txt')").fetchall()

    assert len(rows) == 1
    assert os.path.samefile(rows[0][0], child)


@pytest.mark.skipif(os.name == "nt", reason="symlink traversal semantics differ on Windows")
def test_list_files_search_path_preserves_direct_directory_symlink(duckdb_cursor, tmp_path, monkeypatch):
    search_path = tmp_path / "search"
    target = tmp_path / "target"
    link = tmp_path / "link"
    search_path.mkdir()
    target.mkdir()
    (target / "value.txt").write_text("value", encoding="utf-8")
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")

    monkeypatch.chdir(tmp_path)
    duckdb_cursor.execute("SET file_search_path = ?", [str(search_path)])

    rows = duckdb_cursor.execute("SELECT url FROM list_files('link')").fetchall()

    assert rows == [("link/value.txt",)]


@pytest.mark.parametrize("mode", [0, 0o400])
def test_list_files_search_path_reports_inaccessible_recursive_subdirectory(duckdb_cursor, tmp_path, monkeypatch, mode):
    if os.name == "nt":
        pytest.skip("POSIX directory permissions are required")

    search_path = tmp_path / "search"
    root = search_path / "root"
    inaccessible = root / "inaccessible"
    inaccessible.mkdir(parents=True)
    (root / "visible.txt").write_text("visible", encoding="utf-8")
    (inaccessible / "hidden.txt").write_text("hidden", encoding="utf-8")
    inaccessible.chmod(mode)
    monkeypatch.chdir(tmp_path)
    duckdb_cursor.execute("SET file_search_path = ?", [str(search_path)])
    try:
        if os.access(inaccessible, os.R_OK | os.X_OK):
            pytest.skip("test process can bypass directory permissions")
        with pytest.raises(vane.IOException, match="exists but is not accessible"):
            vane.list_files("root", recursive=True, connection=duckdb_cursor).fetchall()
    finally:
        inaccessible.chmod(0o700)


def test_list_files_preserves_posix_backslashes_in_concrete_directories(duckdb_cursor, tmp_path):
    if os.name == "nt":
        pytest.skip("backslash is a path separator on Windows")

    root = tmp_path / "literal\\directory"
    nested = root / "nested\\directory"
    outside = tmp_path / "outside"
    root.mkdir()
    nested.mkdir()
    outside.mkdir()
    direct = root / "direct.txt"
    nested_file = nested / "nested.txt"
    outside_file = outside / "outside.txt"
    direct.write_text("direct", encoding="utf-8")
    nested_file.write_text("nested", encoding="utf-8")
    outside_file.write_text("outside", encoding="utf-8")

    file_link = root / "file-link.txt"
    directory_link = root / "directory-link"
    try:
        file_link.symlink_to(outside_file)
        directory_link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")

    non_recursive = duckdb_cursor.execute("SELECT url FROM list_files(?)", [str(root)]).fetchall()
    recursive = duckdb_cursor.execute("SELECT url FROM list_files(?, TRUE)", [str(root)]).fetchall()

    assert [row[0] for row in non_recursive] == sorted([str(direct), str(file_link)])
    assert [row[0] for row in recursive] == sorted([str(direct), str(file_link), str(nested_file)])


def test_list_files_preserves_hash_in_local_directory_url(duckdb_cursor, tmp_path):
    directory = tmp_path / "literal#directory"
    nested = directory / "nested"
    directory.mkdir()
    nested.mkdir()
    direct = directory / "direct.txt"
    descendant = nested / "descendant.txt"
    direct.write_text("direct", encoding="utf-8")
    descendant.write_text("descendant", encoding="utf-8")
    directory_url = directory.resolve().as_uri().replace("%23", "#")

    rows = duckdb_cursor.execute("SELECT url FROM list_files(?)", [directory_url]).fetchall()
    trailing = duckdb_cursor.execute("SELECT url FROM list_files(?)", [f"{directory_url}/"]).fetchall()
    recursive = duckdb_cursor.execute("SELECT url FROM list_files(?, TRUE)", [directory_url]).fetchall()

    assert [row[0] for row in rows] == [f"{directory_url}/direct.txt"]
    assert trailing == rows
    assert [row[0] for row in recursive] == sorted(
        [f"{directory_url}/direct.txt", f"{directory_url}/nested/descendant.txt"]
    )


def test_list_files_normalizes_file_url_identity_across_directory_and_glob(duckdb_cursor, tmp_path):
    directory = tmp_path / "identity"
    directory.mkdir()
    child = directory / "value.txt"
    child.write_text("value", encoding="utf-8")
    directory_url = directory.as_uri()

    direct = duckdb_cursor.execute(
        "SELECT url, file_locator_id(file), file FROM list_files(?)", [directory_url]
    ).fetchone()
    glob = duckdb_cursor.execute(
        "SELECT url, file_locator_id(file), file FROM list_files(?)", [f"{directory_url}/*"]
    ).fetchone()

    assert direct[0] == glob[0] == child.as_uri()
    assert direct[1] == glob[1]
    assert duckdb_cursor.execute("SELECT file_same_location(?, ?)", [direct[2], glob[2]]).fetchone() == (True,)


@pytest.mark.parametrize("mode", [0, 0o400])
def test_list_files_reports_inaccessible_directory(duckdb_cursor, tmp_path, mode):
    if os.name == "nt":
        pytest.skip("POSIX directory permissions are required")

    directory = tmp_path / "inaccessible"
    directory.mkdir()
    (directory / "value.txt").write_text("value", encoding="utf-8")
    directory.chmod(mode)
    try:
        if os.access(directory, os.R_OK | os.X_OK):
            pytest.skip("test process can bypass directory permissions")
        with pytest.raises(vane.IOException, match="exists but is not accessible"):
            vane.list_files(str(directory), connection=duckdb_cursor).fetchall()
    finally:
        directory.chmod(0o700)


@pytest.mark.parametrize("mode", [0, 0o400])
def test_list_files_reports_inaccessible_recursive_subdirectory(duckdb_cursor, tmp_path, mode):
    if os.name == "nt":
        pytest.skip("POSIX directory permissions are required")

    root = tmp_path / "root"
    inaccessible = root / "inaccessible"
    root.mkdir()
    inaccessible.mkdir()
    (root / "visible.txt").write_text("visible", encoding="utf-8")
    (inaccessible / "hidden.txt").write_text("hidden", encoding="utf-8")
    inaccessible.chmod(mode)
    try:
        if os.access(inaccessible, os.R_OK | os.X_OK):
            pytest.skip("test process can bypass directory permissions")
        with pytest.raises(vane.IOException, match="exists but is not accessible"):
            vane.list_files(str(root), recursive=True, connection=duckdb_cursor).fetchall()
    finally:
        inaccessible.chmod(0o700)


def test_list_files_reports_missing_and_unsupported_http_listing(duckdb_cursor, tmp_path):
    with pytest.raises(vane.IOException, match="does not exist"):
        vane.list_files(str(tmp_path / "missing"), connection=duckdb_cursor).fetchall()

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    handler = functools.partial(QuietHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        duckdb_cursor.execute("SET http_proxy = ''")
        duckdb_cursor.execute("SET allow_asterisks_in_http_paths = true")
        url = f"http://127.0.0.1:{server.server_port}/*.txt"
        with pytest.raises(vane.NotImplementedException, match="does not support glob listing"):
            vane.list_files(url, connection=duckdb_cursor).fetchall()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_list_files_reports_authoritatively_missing_path_before_unsupported_listing(duckdb_cursor):
    fsspec = pytest.importorskip("fsspec", minversion="2022.11.0")

    class ExistenceOnlyFileSystem(fsspec.AbstractFileSystem):
        protocol = "exists-only"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.glob_calls = 0

        def isfile(self, _path):
            return False

        def isdir(self, _path):
            return False

        def glob(self, *_args, **_kwargs):
            self.glob_calls += 1
            raise NotImplementedError("globbing is unavailable")

        def ls(self, *_args, **_kwargs):
            raise NotImplementedError("directory listing is unavailable")

    filesystem = ExistenceOnlyFileSystem(skip_instance_cache=True)
    duckdb_cursor.register_filesystem(filesystem)
    try:
        with pytest.raises(vane.IOException, match="does not exist"):
            vane.list_files("exists-only://missing", connection=duckdb_cursor).fetchall()
        assert filesystem.glob_calls == 1
    finally:
        duckdb_cursor.unregister_filesystem("exists-only")


def test_list_files_registered_filesystem_filters_directories_and_preserves_unknown_metadata(duckdb_cursor):
    fsspec = pytest.importorskip("fsspec", minversion="2022.11.0")
    memory = fsspec.filesystem("memory", skip_instance_cache=True)
    memory.store = {}
    memory.pseudo_dirs = [""]
    memory.makedirs("root/nested")
    memory.pipe("root/value.txt", b"value")
    memory.pipe("root/nested/child.txt", b"child")
    duckdb_cursor.register_filesystem(memory)
    try:
        rows = vane.list_files("memory://root/*", connection=duckdb_cursor).fetchall()
    finally:
        duckdb_cursor.unregister_filesystem("memory")

    assert len(rows) == 1
    assert rows[0][0].endswith("/root/value.txt")
    assert rows[0][1] is None
    assert rows[0][6] == vane.File(rows[0][0], "text/plain")


def test_list_files_registered_filesystem_accepts_literal_glob_key(duckdb_cursor):
    fsspec = pytest.importorskip("fsspec", minversion="2022.11.0")
    memory = fsspec.filesystem("memory", skip_instance_cache=True)
    memory.store = {}
    memory.pseudo_dirs = [""]
    memory.pipe("root/literal*", b"value")
    duckdb_cursor.register_filesystem(memory)
    try:
        rows = duckdb_cursor.execute("SELECT url FROM list_files(?)", ["memory:///root/literal*"]).fetchall()
    finally:
        duckdb_cursor.unregister_filesystem("memory")

    assert rows == [("memory:///root/literal*",)]


def test_list_files_accepts_literal_glob_key_without_glob_support(duckdb_cursor, tmp_path):
    (tmp_path / "literal*.txt").write_text("value", encoding="utf-8")

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    handler = functools.partial(QuietHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        duckdb_cursor.execute("SET http_proxy = ''")
        duckdb_cursor.execute("SET allow_asterisks_in_http_paths = true")
        url = f"http://127.0.0.1:{server.server_port}/literal*.txt"
        rows = duckdb_cursor.execute("SELECT url, object_size, file FROM list_files(?)", [url]).fetchall()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert rows == [(url, None, vane.File(url, "text/plain"))]


def test_list_files_registered_filesystem_preserves_question_mark_globs(duckdb_cursor):
    fsspec = pytest.importorskip("fsspec", minversion="2022.11.0")
    memory = fsspec.filesystem("memory", skip_instance_cache=True)
    memory.store = {}
    memory.pseudo_dirs = [""]
    memory.pipe("root/value1.txt", b"one")
    memory.pipe("root1/fixed.txt", b"fixed")
    duckdb_cursor.register_filesystem(memory)
    try:
        rows = duckdb_cursor.execute("SELECT url FROM list_files(?)", ["memory:///root/value?.txt"]).fetchall()
        empty = duckdb_cursor.execute("SELECT url FROM list_files(?)", ["memory:///root/missing?.txt"]).fetchall()
        authority_match = duckdb_cursor.execute(
            "SELECT url FROM list_files(?)", ["memory://root?/fixed.txt"]
        ).fetchall()
        authority_empty = duckdb_cursor.execute(
            "SELECT url FROM list_files(?)", ["memory://missing?/fixed.txt"]
        ).fetchall()
    finally:
        duckdb_cursor.unregister_filesystem("memory")

    assert rows == [("memory:///root/value1.txt",)]
    assert empty == []
    assert authority_match == [("memory:///root1/fixed.txt",)]
    assert authority_empty == []


def test_list_files_registered_filesystem_preserves_hash_in_directory_key(duckdb_cursor):
    fsspec = pytest.importorskip("fsspec", minversion="2022.11.0")

    class DirectoryMemoryFileSystem(fsspec.implementations.memory.MemoryFileSystem):
        vane_directory_semantics = True

    memory = DirectoryMemoryFileSystem(skip_instance_cache=True)
    memory.store = {}
    memory.pseudo_dirs = [""]
    memory.makedirs("root/literal#directory")
    memory.pipe("root/literal#directory/value.txt", b"value")
    duckdb_cursor.register_filesystem(memory)
    try:
        rows = duckdb_cursor.execute("SELECT url FROM list_files(?)", ["memory://root/literal#directory"]).fetchall()
    finally:
        duckdb_cursor.unregister_filesystem("memory")

    assert rows == [("memory://root/literal#directory/value.txt",)]


def test_list_files_registered_filesystem_accepts_empty_directory(duckdb_cursor):
    fsspec = pytest.importorskip("fsspec", minversion="2022.11.0")

    class DirectoryMemoryFileSystem(fsspec.implementations.memory.MemoryFileSystem):
        vane_directory_semantics = True

    memory = DirectoryMemoryFileSystem(skip_instance_cache=True)
    memory.store = {}
    memory.pseudo_dirs = [""]
    memory.makedirs("empty")
    duckdb_cursor.register_filesystem(memory)
    try:
        rows = vane.list_files("memory://empty", connection=duckdb_cursor).fetchall()
    finally:
        duckdb_cursor.unregister_filesystem("memory")

    assert rows == []


def test_list_files_registered_directory_filesystem_does_not_require_glob(duckdb_cursor):
    fsspec = pytest.importorskip("fsspec", minversion="2022.11.0")

    class ListOnlyDirectoryMemoryFileSystem(fsspec.implementations.memory.MemoryFileSystem):
        vane_directory_semantics = True

        def glob(self, *_args, **_kwargs):
            raise AssertionError("list_files() must not glob a confirmed directory")

    memory = ListOnlyDirectoryMemoryFileSystem(skip_instance_cache=True)
    memory.store = {}
    memory.pseudo_dirs = [""]
    memory.makedirs("root/nested")
    memory.pipe("root/direct.txt", b"direct")
    memory.pipe("root/nested/child.txt", b"child")
    duckdb_cursor.register_filesystem(memory)
    try:
        rows = duckdb_cursor.execute("SELECT url FROM list_files('memory://root')").fetchall()
        recursive = duckdb_cursor.execute("SELECT url FROM list_files('memory://root', TRUE)").fetchall()
    finally:
        duckdb_cursor.unregister_filesystem("memory")

    assert rows == [("memory://root/direct.txt",)]
    assert recursive == [("memory://root/direct.txt",), ("memory://root/nested/child.txt",)]


def test_list_files_normalizes_registered_url_identity_across_directory_and_glob(duckdb_cursor):
    pytest.importorskip("fsspec", minversion="2022.11.0")
    memory_module = pytest.importorskip("fsspec.implementations.memory")

    class DirectoryMemoryFileSystem(memory_module.MemoryFileSystem):
        vane_directory_semantics = True

    memory = DirectoryMemoryFileSystem(skip_instance_cache=True)
    memory.store = {}
    memory.pseudo_dirs = [""]
    memory.makedirs("root")
    memory.pipe("root/value.txt", b"value")
    duckdb_cursor.register_filesystem(memory)
    try:
        direct = duckdb_cursor.execute(
            "SELECT url, file_locator_id(file), file FROM list_files('memory://root')"
        ).fetchone()
        glob = duckdb_cursor.execute(
            "SELECT url, file_locator_id(file), file FROM list_files('memory://root/*')"
        ).fetchone()
    finally:
        duckdb_cursor.unregister_filesystem("memory")

    assert direct[0] == glob[0] == "memory://root/value.txt"
    assert direct[1] == glob[1]
    assert duckdb_cursor.execute("SELECT file_same_location(?, ?)", [direct[2], glob[2]]).fetchone() == (True,)


def test_list_files_normalizes_registered_protocol_alias_identity_across_directory_and_glob(duckdb_cursor, tmp_path):
    fsspec = pytest.importorskip("fsspec", minversion="2022.11.0")
    directory = tmp_path / "alias-identity"
    directory.mkdir()
    child = directory / "value.txt"
    child.write_text("value", encoding="utf-8")
    directory_url = directory.as_uri().replace("file://", "local://", 1)

    filesystem = fsspec.filesystem("file", skip_instance_cache=True)
    duckdb_cursor.register_filesystem(filesystem)
    try:
        direct = duckdb_cursor.execute(
            "SELECT url, file_locator_id(file), file FROM list_files(?)", [directory_url]
        ).fetchone()
        glob = duckdb_cursor.execute(
            "SELECT url, file_locator_id(file), file FROM list_files(?)", [f"{directory_url}/*"]
        ).fetchone()
    finally:
        duckdb_cursor.unregister_filesystem("file")

    assert direct[0] == glob[0] == child.as_uri().replace("file://", "local://", 1)
    assert direct[1] == glob[1]
    assert duckdb_cursor.execute("SELECT file_same_location(?, ?)", [direct[2], glob[2]]).fetchone() == (True,)


def test_list_files_recognizes_registered_protocol_identifier_with_underscore(duckdb_cursor):
    fsspec = pytest.importorskip("fsspec", minversion="2022.11.0")
    memory_module = pytest.importorskip("fsspec.implementations.memory")

    class UnderscoreProtocolMemoryFileSystem(memory_module.MemoryFileSystem):
        protocol = "custom_protocol"
        vane_directory_semantics = True
        _strip_protocol = classmethod(fsspec.AbstractFileSystem._strip_protocol.__func__)

    filesystem = UnderscoreProtocolMemoryFileSystem(skip_instance_cache=True)
    filesystem.store = {}
    filesystem.pseudo_dirs = [""]
    filesystem.makedirs("root")
    filesystem.pipe("root/value.txt", b"value")
    duckdb_cursor.register_filesystem(filesystem)
    try:
        rows = duckdb_cursor.execute("SELECT url FROM list_files('custom_protocol://root')").fetchall()
    finally:
        duckdb_cursor.unregister_filesystem("custom_protocol")

    assert rows == [("custom_protocol://root/value.txt",)]


def test_list_files_normalizes_registered_authority_identity_across_directory_and_glob(duckdb_cursor):
    pytest.importorskip("fsspec", minversion="2022.11.0")
    ftp_module = pytest.importorskip("fsspec.implementations.ftp")
    memory_module = pytest.importorskip("fsspec.implementations.memory")

    class DirectoryFTPFileSystem(memory_module.MemoryFileSystem):
        protocol = "ftp"
        vane_directory_semantics = True
        _strip_protocol = classmethod(ftp_module.FTPFileSystem._strip_protocol.__func__)

    filesystem = DirectoryFTPFileSystem(skip_instance_cache=True)
    filesystem.store = {}
    filesystem.pseudo_dirs = [""]
    filesystem.pipe("/root/value.txt", b"value")
    duckdb_cursor.register_filesystem(filesystem)
    try:
        direct = duckdb_cursor.execute(
            "SELECT url, file_locator_id(file), file FROM list_files('ftp://host/root')"
        ).fetchone()
        glob = duckdb_cursor.execute(
            "SELECT url, file_locator_id(file), file FROM list_files('ftp://host/root/*')"
        ).fetchone()
    finally:
        duckdb_cursor.unregister_filesystem("ftp")

    assert direct[0] == glob[0] == "ftp://host/root/value.txt"
    assert direct[1] == glob[1]
    assert duckdb_cursor.execute("SELECT file_same_location(?, ?)", [direct[2], glob[2]]).fetchone() == (True,)


def test_list_files_preserves_registered_glob_result_with_different_authority(duckdb_cursor):
    pytest.importorskip("fsspec", minversion="2022.11.0")
    ftp_module = pytest.importorskip("fsspec.implementations.ftp")
    memory_module = pytest.importorskip("fsspec.implementations.memory")

    class DifferentAuthorityFTPFileSystem(memory_module.MemoryFileSystem):
        protocol = "ftp"
        _strip_protocol = classmethod(ftp_module.FTPFileSystem._strip_protocol.__func__)

        def glob(self, _path, **_kwargs):
            return ["ftp://other/root/value.txt"]

    filesystem = DifferentAuthorityFTPFileSystem(skip_instance_cache=True)
    filesystem.store = {}
    filesystem.pseudo_dirs = [""]
    filesystem.pipe("/root/value.txt", b"value")
    duckdb_cursor.register_filesystem(filesystem)
    try:
        rows = duckdb_cursor.execute("SELECT url FROM list_files('ftp://host/root/*')").fetchall()
    finally:
        duckdb_cursor.unregister_filesystem("ftp")

    assert rows == [("ftp://other/root/value.txt",)]
    assert filesystem.isfile(rows[0][0])


def test_list_files_treats_embedded_scheme_delimiter_as_registered_path_text(duckdb_cursor):
    pytest.importorskip("fsspec", minversion="2022.11.0")
    memory_module = pytest.importorskip("fsspec.implementations.memory")

    class EmbeddedSchemeMemoryFileSystem(memory_module.MemoryFileSystem):
        def glob(self, _path, **_kwargs):
            return ["/root/http://value.txt"]

    filesystem = EmbeddedSchemeMemoryFileSystem(skip_instance_cache=True)
    filesystem.store = {}
    filesystem.pseudo_dirs = [""]
    filesystem.pipe("/root/http://value.txt", b"value")
    duckdb_cursor.register_filesystem(filesystem)
    try:
        rows = duckdb_cursor.execute("SELECT url FROM list_files('memory://root/*')").fetchall()
    finally:
        duckdb_cursor.unregister_filesystem("memory")

    assert rows == [("memory:///root/http://value.txt",)]
    assert filesystem.isfile(rows[0][0])


def test_list_files_preserves_unrelated_registered_glob_urls(duckdb_cursor):
    fsspec = pytest.importorskip("fsspec", minversion="2022.11.0")

    class ForeignResultFileSystem(fsspec.AbstractFileSystem):
        protocol = "source"

        def glob(self, path, **kwargs):
            return ["other://bucket/value.txt"]

    filesystem = ForeignResultFileSystem(skip_instance_cache=True)
    duckdb_cursor.register_filesystem(filesystem)
    try:
        rows = duckdb_cursor.execute("SELECT url FROM list_files('source://root/*')").fetchall()
    finally:
        duckdb_cursor.unregister_filesystem("source")

    assert rows == [("other://bucket/value.txt",)]


def test_list_files_registered_directory_filesystem_accepts_trailing_directory_separator(duckdb_cursor):
    pytest.importorskip("fsspec", minversion="2022.11.0")

    memory_module = pytest.importorskip("fsspec.implementations.memory")

    class TrailingDirectoryMemoryFileSystem(memory_module.MemoryFileSystem):
        vane_directory_semantics = True

        def ls(self, path, detail=True, **kwargs):
            entries = super().ls(path, detail=detail, **kwargs)
            if not detail:
                return entries
            result = []
            for entry in entries:
                entry = dict(entry)
                if entry["type"] == "directory":
                    entry["name"] = f"{entry['name'].rstrip('/')}/"
                result.append(entry)
            return result

    memory = TrailingDirectoryMemoryFileSystem(skip_instance_cache=True)
    memory.store = {}
    memory.pseudo_dirs = [""]
    memory.makedirs("root/nested")
    memory.pipe("root/direct.txt", b"direct")
    memory.pipe("root/nested/child.txt", b"child")
    duckdb_cursor.register_filesystem(memory)
    try:
        rows = duckdb_cursor.execute("SELECT url FROM list_files('memory://root', TRUE)").fetchall()
    finally:
        duckdb_cursor.unregister_filesystem("memory")

    assert rows == [("memory://root/direct.txt",), ("memory://root/nested/child.txt",)]


def test_list_files_registered_directory_filesystem_falls_back_when_ls_is_not_implemented(duckdb_cursor):
    fsspec = pytest.importorskip("fsspec", minversion="2022.11.0")

    class GlobOnlyDirectoryMemoryFileSystem(fsspec.implementations.memory.MemoryFileSystem):
        vane_directory_semantics = True

        def ls(self, *_args, **_kwargs):
            raise NotImplementedError("directory listing is unavailable")

        def glob(self, path, **_kwargs):
            if path == "memory://root/*":
                return ["/root/value.txt"]
            return []

    memory = GlobOnlyDirectoryMemoryFileSystem(skip_instance_cache=True)
    memory.store = {}
    memory.pseudo_dirs = [""]
    memory.makedirs("root")
    memory.pipe("root/value.txt", b"value")
    duckdb_cursor.register_filesystem(memory)
    try:
        rows = duckdb_cursor.execute("SELECT url FROM list_files('memory://root')").fetchall()
    finally:
        duckdb_cursor.unregister_filesystem("memory")

    assert rows == [("memory://root/value.txt",)]


def test_file_does_not_implicitly_convert_to_plain_struct():
    plain_struct = vane.struct_type(
        {
            "url": vane.sqltypes.VARCHAR,
            "content_type": vane.sqltypes.VARCHAR,
            "position": vane.sqltypes.BIGINT,
            "size": vane.sqltypes.BIGINT,
            "checksum": vane.sqltypes.VARCHAR,
        }
    )
    with pytest.raises(vane.InvalidInputException, match="cannot be converted to STRUCT"):
        vane.ConstantExpression(vane.Value(vane.File("memory://value"), plain_struct))


def test_plain_struct_does_not_select_file_union_member():
    dtype = vane.union_type({"file": vane.file_type(), "text": vane.sqltypes.VARCHAR})
    value = {
        "url": "memory://struct",
        "content_type": None,
        "position": None,
        "size": None,
        "checksum": None,
    }
    with pytest.raises(vane.InvalidInputException, match="no compatible composite member"):
        vane.ConstantExpression(vane.Value(value, dtype))


def test_plain_struct_cannot_be_inserted_as_file(duckdb_cursor):
    duckdb_cursor.execute("CREATE TABLE invalid_file(value FILE)")
    with pytest.raises(
        vane.BinderException,
        match="FILE-family casts require an exact logical type match",
    ):
        duckdb_cursor.execute(
            """
            INSERT INTO invalid_file
            SELECT ROW(NULL::VARCHAR, NULL::VARCHAR, NULL::BIGINT, NULL::BIGINT, NULL::VARCHAR)
            """
        )


def test_file_expression_api_is_pure_and_defers_validation(duckdb_cursor):
    expression = vane.file(
        "s3://bucket/not-accessed",
        content_type="application/octet-stream",
        position=5,
        size=7,
        checksum="sha256:expression",
    )
    assert duckdb_cursor.values(expression).fetchone() == (
        vane.File("s3://bucket/not-accessed", "application/octet-stream", 5, 7, "sha256:expression"),
    )
    assert duckdb_cursor.values(
        expression.url,
        expression.content_type,
        expression.position,
        expression.size,
        expression.checksum,
    ).fetchone() == ("s3://bucket/not-accessed", "application/octet-stream", 5, 7, "sha256:expression")

    relation = duckdb_cursor.values(vane.ConstantExpression("memory://column").alias("source"))
    assert relation.select(vane.col("source").as_file()).fetchone() == (vane.File("memory://column"),)

    invalid = vane.file("memory://invalid-range", position=0)
    assert isinstance(invalid, vane.Expression)
    with pytest.raises(vane.InvalidInputException, match="position and size"):
        duckdb_cursor.values(invalid).fetchone()
