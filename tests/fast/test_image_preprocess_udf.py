import pytest


def test_relation_image_preprocess_is_not_public():
    import duckdb

    rel = duckdb.sql("select 'local.png' as image_url")

    assert not hasattr(rel, "image_preprocess")


def test_duckdb_image_preprocess_table_function_is_not_registered():
    import duckdb

    con = duckdb.connect()

    with pytest.raises(duckdb.CatalogException, match="duckdb_image_preprocess"):
        con.execute(
            """
            select *
            from duckdb_image_preprocess(
                (select 'local.png' as image_url),
                struct_pack(dummy := true)
            )
            """
        ).fetchall()
