import pytest


def test_image_crop_encode_is_not_registered():
    import duckdb

    con = duckdb.connect()

    with pytest.raises(duckdb.CatalogException, match="image_crop_encode"):
        con.execute(
            """
            select image_crop_encode(
                ''::BLOB,
                1,
                1,
                0,
                0,
                1,
                1
            )
            """
        ).fetchall()
