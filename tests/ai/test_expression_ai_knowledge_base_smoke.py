from __future__ import annotations

import pyarrow as pa
import pytest

import vane
from tests.ai.test_expression_ai_functions import MockProvider


def test_python_expression_knowledge_base_shape() -> None:
    conn = vane.connect()
    rel = conn.sql(
        """
        select *
        from (
            values
                (1, 'u1', 'en', 'search systems scale retrieval quickly'),
                (2, 'u2', 'zh', repeat('搜索系统 ', 30)),
                (3, 'u3', 'en', 'short')
        ) as docs(id, url, language, text)
        """
    )

    @vane.func(return_dtype="VARCHAR[]")
    def chunk_text(text: str) -> list[str]:
        words = text.split()
        return [" ".join(words[start : start + 2]) for start in range(0, len(words), 2)]

    @vane.cls(actor_number=1, return_dtype="VARCHAR", name="kb_embed", gpus=0.25)
    class MockEmbedder:
        def __init__(self, marker: float) -> None:
            self.marker = marker
            self.calls = 0

        def __call__(self, chunk: str) -> str:
            self.calls += 1
            return f"len={len(chunk)};call={self.calls};marker={self.marker}"

    filtered = rel.filter("language = 'en' and length(text) > 20")
    with_chunk_lists = filtered.select(
        vane.col("id"),
        vane.col("url"),
        chunk_text(vane.col("text")).alias("chunks"),
    )
    conn.execute("DROP VIEW IF EXISTS kb_actor_chunks")
    with_chunk_lists.to_view("kb_actor_chunks")
    chunks = conn.sql(
        """
        select id, url, unnest(chunks) as chunk
        from kb_actor_chunks
        order by id, chunk
        """
    )
    embedder = MockEmbedder(7.0)
    out = chunks.select(
        vane.col("id"),
        vane.col("chunk"),
        embedder(vane.col("chunk")).alias("embedding"),
        vane.ai.prompt(vane.col("chunk"), provider=MockProvider()).alias("topic"),
    ).order("chunk")

    with pytest.warns(
        RuntimeWarning,
        match="gpus is not reserved by the local subprocess actor backend",
    ):
        rows = out.fetchall()
    plan = out.explain()

    assert [row[0] for row in rows] == [1, 1, 1]
    assert [row[1] for row in rows] == ["quickly", "scale retrieval", "search systems"]
    assert [row[2] for row in rows] == [
        "len=7;call=1;marker=7.0",
        "len=15;call=2;marker=7.0",
        "len=14;call=3;marker=7.0",
    ]
    assert [row[3] for row in rows] == [
        "topic:quickly",
        "topic:scale retrieval",
        "topic:search systems",
    ]
    assert "subprocess_actor" in plan
    assert "gpus:" in plan


def test_batch_preprocessing_feeds_ai_embed_and_prompt() -> None:
    """Keep the row-preserving batch-to-AI expression composition covered."""
    conn = vane.connect()
    rel = conn.sql(
        """
        select *
        from (
            values
                (1, '  Search systems scale retrieval quickly  '),
                (2, 'Vector databases need careful evaluation'),
                (3, 'too short')
        ) as docs(id, text)
        """
    )

    def normalize_chunk(table: pa.Table) -> pa.Table:
        chunks = [value.strip().lower() for value in table.column("text").to_pylist()]
        return pa.table({"chunk": chunks})

    chunk = vane.func.batch(
        normalize_chunk,
        inputs={"text": vane.col("text")},
        schema={"chunk": "VARCHAR"},
        row_preserving=True,
    ).alias("chunk")
    filtered = rel.filter("length(trim(text)) > 20")
    with_chunks = filtered.select(vane.col("id"), chunk)
    out = with_chunks.select(
        vane.col("id"),
        vane.col("chunk"),
        vane.ai.embed(
            vane.col("chunk"),
            provider=MockProvider(),
            dimensions=4,
        ).alias("embedding"),
        vane.ai.prompt(vane.col("chunk"), provider=MockProvider()).alias("topic"),
    ).order("id")

    rows = out.fetchall()

    assert [row[0] for row in rows] == [1, 2]
    assert [row[1] for row in rows] == [
        "search systems scale retrieval quickly",
        "vector databases need careful evaluation",
    ]
    assert [list(row[2]) for row in rows] == [
        [38.0, 38.0, 38.0, 38.0],
        [40.0, 40.0, 40.0, 40.0],
    ]
    assert [row[3] for row in rows] == [
        "topic:search systems scale retrieval quickly",
        "topic:vector databases need careful evaluation",
    ]
