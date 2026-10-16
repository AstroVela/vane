# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Installed SQL/Python parity through real embedding executors."""

import pickle

import pytest

import vane
from tests.fast.test_ai_multimodal_embedding import ContentDescriptor, ContentProvider, OptionalContentProvider
from vane.ai import AudioInput, ImageInput, MultimodalInput, TextInput, VideoInput, embed
from vane.ai import provider as registry


@pytest.fixture(autouse=True)
def providers(monkeypatch):
    monkeypatch.setitem(registry.PROVIDERS, "content_fixture", lambda name=None: ContentProvider())
    monkeypatch.setitem(registry.PROVIDERS, "optional_fixture", lambda name=None: OptionalContentProvider())


PARTS = "[{kind: 'text', mime_type: NULL}, {kind: 'video', mime_type: 'video/mp4'}]"


@pytest.mark.parametrize("family", ["sglang", "vllm"])
@pytest.mark.parametrize("kind", ["text", "image", "video", "audio"])
@pytest.mark.parametrize("condition", [None, "FALSE", "NULL"])
def test_scalar_null_skips_gpu_execution_after_validation(monkeypatch, family, kind, condition):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    options = {"gpus_per_actor": 1, "input_modalities": [kind]}
    if kind == "audio":
        options.update(sample_rate=16000, max_samples=16000)
    wrapper = {"text": TextInput, "image": ImageInput, "video": VideoInput, "audio": AudioInput}[kind]
    predicate = "" if condition is None else f', "where" := {condition}'
    with vane.connect() as conn:
        python = conn.sql("SELECT 1").select(
            embed(wrapper(vane.ConstantExpression(None)), provider=family, model="fixture", dimensions=2, **options)
        )
        sql = conn.sql(
            f"""SELECT ai_embed(NULL, input_spec := {{kind: '{kind}'}}, provider := '{family}',
                model := 'fixture', dimensions := 2, options := $1{predicate})""",
            params=[options],
        )
        assert sql.types == python.types == [vane.array_type(vane.sqltypes.FLOAT, 2)]
        assert sql.fetchall() == python.fetchall() == [(None,)]


@pytest.mark.parametrize("value", ["NULL::IMAGE", "CASE WHEN TRUE THEN NULL::IMAGE ELSE NULL::IMAGE END", "$1::IMAGE"])
def test_cosmos_foldable_null_image_requires_no_gpu(monkeypatch, value):
    monkeypatch.setenv("VANE_RUNNER", "local-fast")
    with vane.connect() as conn:
        result = conn.sql(
            f"""SELECT ai_embed({value}, input_spec := {{kind: 'image'}},
                provider := 'transformers', model := 'nvidia/Cosmos-Embed1-224p',
                options := {{revision: '{"a" * 40}', trust_remote_code: true, device: 'cuda', dtype: 'float16'}})""",
            params=[None] if "$1" in value else None,
        )
        assert result.fetchall() == [(None,)]


@pytest.mark.parametrize("kind", ["text", "image", "video", "audio"])
def test_scalar_null_does_not_skip_type_or_predicate_validation(kind):
    options = {"input_modalities": [kind], "gpus_per_actor": 1}
    if kind == "audio":
        options.update(sample_rate=16000, max_samples=16000)
    with vane.connect() as conn:
        for value, predicate, message in [("NULL::INTEGER", "FALSE", "requires|must be"), ("NULL", "1", "BOOLEAN")]:
            with pytest.raises(Exception, match=message):
                conn.sql(
                    f"""SELECT ai_embed({value}, input_spec := {{kind: '{kind}'}}, provider := 'sglang',
                        model := 'fixture', dimensions := 2, options := $1, "where" := {predicate})""",
                    params=[options],
                )


def test_joint_sql_python_selection_and_plan_round_trip():
    from tests.ai.test_expression_ai_sql import _execute_ai_physical_plan, _round_trip_ai_plan

    with vane.connect() as conn:
        conn.sql("""SELECT * FROM (VALUES (0, 'ok', 'ab'::BLOB, TRUE),
            (1, ' ', 'c'::BLOB, TRUE), (2, NULL, NULL, FALSE),
            (3, repeat('x', 32), ''::BLOB, NULL), (4, NULL, 'last'::BLOB, TRUE))
            t(id, text, video, selected)""").create_view("source")
        sql = conn.sql(f"""SELECT id, ai_embed({{transcript: text, clip: video}},
            input_spec := {PARTS}, "where" := selected,
            provider := 'content_fixture', options := {{batch_size: 2, max_retries: 0}}) AS vector FROM source""")
        python = conn.table("source").select(
            "id",
            embed(
                MultimodalInput([TextInput(vane.col("text")), VideoInput(vane.col("video"), "video/mp4")]),
                where=vane.col("selected"),
                provider=ContentProvider(),
                batch_size=2,
                max_retries=0,
            ).alias("vector"),
        )
        expected = [(0, (4, 2)), (1, (1, 1)), (2, None), (3, None), (4, (4, 1))]
        assert sql.types[-1] == vane.array_type(vane.sqltypes.FLOAT, 2)
        assert sql.order("id").fetchall() == python.order("id").fetchall() == expected
        target, physical, serialized = _round_trip_ai_plan(sql)
        try:
            assert pickle.loads(serialized)
            result = _execute_ai_physical_plan(target, physical)
            assert sorted(zip(result.column(0).to_pylist(), result.column(1).to_pylist())) == [
                (index, list(value) if value else None) for index, value in expected
            ]
        finally:
            target.close()


@pytest.mark.parametrize(
    "kind,mime",
    [
        ("text", None),
        ("image", None),
        ("video", None),
        ("audio", None),
        ("image", "image/png"),
        ("video", "video/mp4"),
        ("audio", "audio/wav"),
    ],
)
def test_constant_null_optional_parts_are_not_folded(kind, mime):
    mime_sql = "NULL" if mime is None else f"'{mime}'"
    with vane.connect() as conn:
        assert conn.sql(f"""SELECT ai_embed({{absent: NULL, present: 'ok'}},
            input_spec := [{{kind: '{kind}', mime_type: {mime_sql}}}, {{kind: 'text', mime_type: NULL}}],
            provider := 'optional_fixture')""").fetchall() == [((2, 1),)]


@pytest.mark.parametrize("on_error", ["raise", "ignore"])
@pytest.mark.parametrize("joint", [False, True])
def test_all_absent_selected_content_fails(on_error, joint):
    value, spec = ("{text: NULL, video: NULL}", PARTS) if joint else ("NULL", "{kind: 'video', mime_type: 'video/mp4'}")
    with vane.connect() as conn:
        with pytest.raises(Exception, match="no nonempty parts"):
            conn.sql(f"""SELECT ai_embed({value}, input_spec := {spec}, provider := 'content_fixture',
                on_error := '{on_error}')""").fetchall()


@pytest.mark.parametrize("condition", ["FALSE", "NULL"])
def test_unselected_rows_skip_conversion_and_instantiation(monkeypatch, condition):
    monkeypatch.setattr(ContentDescriptor, "instantiate", lambda _: (_ for _ in ()).throw(AssertionError("created")))
    with vane.connect() as conn:
        assert conn.sql(f"""SELECT ai_embed({{text: repeat('x', 100), video: ''::BLOB}}, input_spec := {PARTS},
            provider := 'content_fixture', "where" := {condition})""").fetchall() == [(None,)]


@pytest.mark.parametrize("condition", ["FALSE", "NULL", "TRUE"])
@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_encoded_media_types_are_checked_even_for_unselected_batches(condition, kind):
    mime = {"image": "image/png", "video": "video/mp4", "audio": "audio/wav"}[kind]
    with vane.connect() as conn:
        with pytest.raises(Exception, match="BLOB"):
            conn.sql(f"""SELECT ai_embed('not bytes', input_spec := {{kind: '{kind}', mime_type: '{mime}'}},
                provider := 'optional_fixture', "where" := {condition})""")


@pytest.mark.parametrize(
    "spec",
    [
        "[]",
        "'video'",
        "{kind: 'path'}",
        "{kind: 'text', extra: NULL}",
        "{kind: NULL}",
        "{kind: 'video', mime_type: 'video/'}",
        "{kind: 'video', mime_type: 'audio/wav'}",
    ],
)
def test_invalid_layout_fails_at_planning(spec):
    with vane.connect() as conn:
        with pytest.raises(Exception, match="input_spec|MIME"):
            conn.sql(f"SELECT ai_embed(NULL, input_spec := {spec}, provider := 'content_fixture')")


@pytest.mark.parametrize("value", ["NULL", "'x'", "{one_part: 'x'}", "{a: 'x', b: NULL, c: NULL}"])
def test_joint_field_count_must_match_spec(value):
    with vane.connect() as conn:
        with pytest.raises(Exception, match="one field per part"):
            conn.sql(f"SELECT ai_embed({value}, input_spec := {PARTS}, provider := 'content_fixture')")


def test_row_varying_layout_and_non_boolean_predicate_rejected():
    with vane.connect() as conn:
        with pytest.raises(Exception, match="foldable|constant"):
            conn.sql(
                "SELECT ai_embed('x', input_spec := {kind: kind}, provider := 'content_fixture') FROM (VALUES ('text')) t(kind)"
            )
        with pytest.raises(Exception, match="BOOLEAN"):
            conn.sql(
                """SELECT ai_embed({t: 'x'}, input_spec := [{kind: 'text'}], provider := 'content_fixture', "where" := 1)"""
            )


@pytest.mark.parametrize("bad", ["''::BLOB", "repeat('x', 17)::BLOB"])
def test_input_budgets_are_not_ignored(bad):
    with vane.connect() as conn:
        with pytest.raises(Exception, match="empty|byte budget"):
            conn.sql(f"""SELECT ai_embed({bad}, input_spec := {{kind: 'video', mime_type: 'video/mp4'}},
                provider := 'content_fixture', on_error := 'ignore')""").fetchall()


def test_prepared_layout_and_selection():
    with vane.connect() as conn:
        conn.execute("""PREPARE embedding AS SELECT ai_embed($1, input_spec := $2, "where" := $3,
            provider := 'content_fixture')""")
        assert conn.execute(
            "EXECUTE embedding('abc'::BLOB, {kind: 'video', mime_type: 'video/mp4'}, TRUE)"
        ).fetchall() == [((3, 1),)]
        assert conn.execute(
            "EXECUTE embedding(''::BLOB, {kind: 'video', mime_type: 'video/mp4'}, FALSE)"
        ).fetchall() == [(None,)]


def test_obsolete_sql_entrypoints_are_removed():
    with vane.connect() as conn:
        assert (
            conn.sql("""SELECT function_name FROM duckdb_functions()
            WHERE function_name IN ('ai_embed_image', 'ai_embed_video', 'ai_embed_audio',
                '__vane_ai_embed_image', '__vane_ai_embed_video', '__vane_ai_embed_audio')""").fetchall()
            == []
        )


@pytest.mark.parametrize(
    "options", [{"max_chunk_chars": 2}, {"chunk_overlap_chars": 0}, {"execution_backend": "subprocess_task"}]
)
def test_relation_only_options_are_rejected_by_sql_and_python_expressions(monkeypatch, options):
    from tests.ai.test_expression_ai_sql import MockProvider

    monkeypatch.setitem(registry.PROVIDERS, "mock_ai_sql", lambda name=None: MockProvider())
    with vane.connect() as conn:
        with pytest.raises(Exception, match="Unsupported Embed option"):
            conn.sql("SELECT ai_embed('abcde', provider := 'mock_ai_sql', options := $1)", params=[options])
        with pytest.raises(TypeError, match="Unsupported Embed option"):
            embed(vane.col("text"), provider=MockProvider(), **options)


def test_joint_parts_reach_the_encoder_in_declared_order(monkeypatch):
    class OrderedEncoder:
        def embed_multimodal(self, rows):
            return [[len(row.parts[0].value), len(row.parts[1].value)] for row in rows]

    class OrderedDescriptor(ContentDescriptor):
        def instantiate(self):
            return OrderedEncoder()

    class OrderedProvider(ContentProvider):
        def get_multimodal_embedder(self, model=None, dimensions=None, *, options=None):
            return OrderedDescriptor()

    monkeypatch.setitem(registry.PROVIDERS, "ordered_fixture", lambda name=None: OrderedProvider())
    with vane.connect() as conn:
        assert conn.sql(f"""SELECT ai_embed({{z: 'abc', a: 'd'::BLOB}}, input_spec := {PARTS},
            provider := 'ordered_fixture')""").fetchall() == [((3, 1),)]


def test_joint_struct_is_evaluated_once_per_row():
    with vane.connect() as conn:
        conn.execute("CREATE SEQUENCE counter START 1")
        assert conn.sql("""SELECT ai_embed({a: nextval('counter')::VARCHAR, b: nextval('counter')::VARCHAR},
            input_spec := [{kind: 'text'}, {kind: 'text'}], provider := 'content_fixture')""").fetchall() == [((2, 2),)]
        assert conn.sql("SELECT currval('counter')").fetchone() == (2,)


@pytest.mark.real_ray
def test_joint_sql_selection_through_default_ray_actor(ray_local, monkeypatch):
    monkeypatch.delenv("VANE_RUNNER", raising=False)
    with vane.connect() as conn:
        result = conn.sql(f"""SELECT i, ai_embed({{text: 'x', clip: NULL}}, input_spec := {PARTS},
            "where" := i % 2 = 0, provider := 'content_fixture', options := {{batch_size: 2}}) AS vector
            FROM range(5) t(i)""")
        assert "ray_actor" in result.explain()
        assert result.order("i").fetchall() == [(0, (1, 1)), (1, None), (2, (1, 1)), (3, None), (4, (1, 1))]
