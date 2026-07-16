import pytest


def test_relation_audio_preprocess_is_not_public():
    import duckdb

    rel = duckdb.sql("select ''::BLOB as audio_bytes")

    assert not hasattr(rel, "audio_preprocess")


def test_duckdb_audio_preprocess_table_function_is_not_registered():
    import duckdb

    con = duckdb.connect()

    with pytest.raises(duckdb.CatalogException, match="duckdb_audio_preprocess"):
        con.execute(
            """
            select *
            from duckdb_audio_preprocess(
                (select ''::BLOB as audio_bytes),
                struct_pack(dummy := true)
            )
            """
        ).fetchall()
