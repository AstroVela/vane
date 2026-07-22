# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

from decimal import Decimal

import pytest

import vane


# A closed connection should invalidate all relation's methods
class TestRAPICloseConnRel:
    def test_close_conn_rel(self, duckdb_cursor):
        con = vane.connect()
        con.execute("CREATE TABLE items(item VARCHAR, value DECIMAL(10,2), count INTEGER)")
        con.execute("INSERT INTO items VALUES ('jeans', 20.0, 1), ('hammer', 42.2, 2)")
        rel = con.table("items")
        con.close()

        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            print(rel)
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            len(rel)
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.aggregate("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.any_value("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.apply("", "")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.arg_max("", "")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.arg_min("", "")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.to_arrow_table()
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.avg("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.bit_and("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.bit_or("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.bit_xor("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.bitstring_agg("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.bool_and("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.bool_or("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.count("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.create("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.create_view("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.cume_dist("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.dense_rank("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.describe()
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.df()
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.distinct()
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.execute()
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.favg("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.fetchall()
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.fetchnumpy()
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.fetchone()
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.filter("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.first("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.first_value("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.fsum("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.geomean("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.histogram("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.insert("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.insert_into("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.lag("", "")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.last("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.last_value("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.lead("", "")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            print(rel.limit(1))
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.list("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.map(
                lambda item, value, count: count,
                return_type=vane.sqltypes.INTEGER,
            )
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.max("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.mean("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.median("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.min("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.mode("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.n_tile("", 1)
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.nth_value("", "", 1)
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.order("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.percent_rank("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.product("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.project("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.quantile("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.quantile_cont("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.quantile_disc("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.query("", "")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.rank("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.rank_dense("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.row_number("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.std("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.stddev("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.stddev_pop("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.stddev_samp("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.string_agg("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.sum("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.to_arrow_table()
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.to_df()
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.var("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.var_pop("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.var_samp("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.variance("")
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            rel.write_csv("")

        con = vane.connect()
        con.execute("CREATE TABLE items(item VARCHAR, value DECIMAL(10,2), count INTEGER)")
        con.execute("INSERT INTO items VALUES ('jeans', 20.0, 1), ('hammer', 42.2, 2)")
        valid_rel = con.table("items")

        # Test these bad boys when left relation is valid
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            valid_rel.union(rel)
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            valid_rel.except_(rel)
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            valid_rel.intersect(rel)
        with pytest.raises(vane.ConnectionException, match="Connection has already been closed"):
            valid_rel.join(rel.set_alias("rel"), "rel.items = valid_rel.items")

    def test_del_conn(self, duckdb_cursor):
        con = vane.connect()
        con.execute("CREATE TABLE items(item VARCHAR, value DECIMAL(10,2), count INTEGER)")
        con.execute("INSERT INTO items VALUES ('jeans', 20.0, 1), ('hammer', 42.2, 2)")
        rel = con.table("items")
        del con
        # Relation keeps the connection alive via connection_owner
        res = rel.fetchall()
        assert res == [("jeans", Decimal("20.00"), 1), ("hammer", Decimal("42.20"), 2)]
