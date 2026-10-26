import pandas as pd


class TestGeometry:
    def test_fetchall(self, duckdb_cursor):
        duckdb_cursor.execute("SELECT 'POINT(1 2)'::GEOMETRY AS geom")
        results = duckdb_cursor.fetchall()
        assert isinstance(results[0][0], bytes)

    def test_fetchnumpy(self, duckdb_cursor):
        duckdb_cursor.execute("SELECT 'POINT(1 2)'::GEOMETRY AS geom")
        results = duckdb_cursor.fetchnumpy()
        assert isinstance(results["geom"][0], (bytes, bytearray))

    def test_df(self, duckdb_cursor):
        duckdb_cursor.execute("SELECT 'POINT(1 2)'::GEOMETRY AS geom")
        df = duckdb_cursor.df()
        assert isinstance(df["geom"].iloc[0], (bytes, bytearray))

    def test_null(self, duckdb_cursor):
        duckdb_cursor.execute("SELECT NULL::GEOMETRY AS geom")
        results = duckdb_cursor.fetchall()
        assert results[0][0] is None

    def test_multiple_rows(self, duckdb_cursor):
        duckdb_cursor.execute("SELECT 'POINT(1 2)'::GEOMETRY AS geom UNION ALL SELECT NULL::GEOMETRY")
        df = duckdb_cursor.df()
        assert df.shape[0] == 2
        assert isinstance(df["geom"].iloc[0], (bytes, bytearray))
        assert pd.isna(df["geom"].iloc[1])
