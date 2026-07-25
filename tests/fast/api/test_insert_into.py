import pytest
from pandas import DataFrame

import duckdb


class TestInsertInto:
    def test_insert_into_schema(self, duckdb_cursor):
        # open connection
        con = duckdb.connect()
        con.execute("CREATE SCHEMA s")
        con.execute("CREATE TABLE s.t (id INTEGER PRIMARY KEY)")

        # make relation
        df = DataFrame([1], columns=["id"])
        rel = con.from_df(df)

        rel.insert_into("s.t")

        assert con.execute("select * from s.t").fetchall() == [(1,)]

        # This should fail since this will go to default schema
        with pytest.raises(duckdb.CatalogException):
            rel.insert_into("t")

        # If we add t in the default schema it should work.
        con.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        rel.insert_into("t")
        assert con.execute("select * from t").fetchall() == [(1,)]

    def test_insert_into_catalog_qualification(self):
        """insert_into must preserve catalog qualification when writing to a catalog-qualified table name."""
        import pandas as pd

        con = duckdb.connect()
        con.execute("ATTACH ':memory:' AS cat_a")
        con.execute("ATTACH ':memory:' AS cat_b")
        con.execute("CREATE TABLE cat_a.main.t (x INT)")
        con.execute("CREATE TABLE cat_b.main.t (x INT)")
        con.execute("INSERT INTO cat_a.main.t VALUES (42)")

        rel = con.from_df(pd.DataFrame({"x": [99]}))
        rel.insert_into("cat_b.main.t")

        results_a = con.execute("SELECT * FROM cat_a.main.t").fetchall()
        results_b = con.execute("SELECT * FROM cat_b.main.t").fetchall()
        assert results_a == [(42,)], f"cat_a.main.t should be [(42,)] but got {results_a}"
        assert results_b == [(99,)], f"cat_b.main.t should be [(99,)] but got {results_b}"

    def test_insert_into_catalog_quoted_identifiers(self):
        """insert_into must preserve catalog qualification with quoted identifiers."""
        import pandas as pd

        con = duckdb.connect()
        con.execute("ATTACH ':memory:' AS \"my.catalog\"")
        con.execute("ATTACH ':memory:' AS other_cat")
        con.execute('CREATE TABLE "my.catalog".main.t (x INT)')
        con.execute("CREATE TABLE other_cat.main.t (x INT)")

        rel = con.from_df(pd.DataFrame({"x": [77]}))
        rel.insert_into('"my.catalog".main.t')

        results_a = con.execute('SELECT * FROM "my.catalog".main.t').fetchall()
        results_b = con.execute("SELECT * FROM other_cat.main.t").fetchall()
        assert results_a == [(77,)], f'"my.catalog".main.t should be [(77,)] but got {results_a}'
        assert results_b == [], f"other_cat.main.t should be [] but got {results_b}"

    def test_create_catalog_qualification(self):
        """create must preserve catalog qualification when creating a table with a catalog-qualified name."""
        import pandas as pd

        con = duckdb.connect()
        con.execute("ATTACH ':memory:' AS cat_a")
        con.execute("ATTACH ':memory:' AS cat_b")
        con.execute("CREATE TABLE cat_a.main.t (x INT)")
        con.execute("INSERT INTO cat_a.main.t VALUES (42)")

        rel = con.from_df(pd.DataFrame({"x": [55]}))
        rel.create("cat_b.main.t")

        results_a = con.execute("SELECT * FROM cat_a.main.t").fetchall()
        results_b = con.execute("SELECT * FROM cat_b.main.t").fetchall()
        assert results_a == [(42,)], f"cat_a.main.t should be [(42,)] but got {results_a}"
        assert results_b == [(55,)], f"cat_b.main.t should be [(55,)] but got {results_b}"
