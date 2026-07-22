# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import warnings

import pytest

import vane

pytest.importorskip("pyarrow")


class TestArrowDeprecation:
    @pytest.fixture(autouse=True)
    def setup(self, duckdb_cursor):
        self.con = duckdb_cursor
        self.con.execute("CREATE TABLE t AS SELECT 1 AS a")

    def test_relation_fetch_arrow_table_deprecated(self):
        rel = self.con.table("t")
        with pytest.warns(
            DeprecationWarning, match="fetch_arrow_table\\(\\) is deprecated, use to_arrow_table\\(\\) instead"
        ):
            rel.fetch_arrow_table()

    def test_relation_fetch_record_batch_deprecated(self):
        rel = self.con.table("t")
        with pytest.warns(
            DeprecationWarning, match="fetch_record_batch\\(\\) is deprecated, use to_arrow_reader\\(\\) instead"
        ):
            rel.fetch_record_batch()

    def test_relation_fetch_arrow_reader_deprecated(self):
        rel = self.con.table("t")
        with pytest.warns(
            DeprecationWarning, match="fetch_arrow_reader\\(\\) is deprecated, use to_arrow_reader\\(\\) instead"
        ):
            rel.fetch_arrow_reader()

    def test_connection_fetch_arrow_table_deprecated(self):
        self.con.execute("SELECT 1")
        with pytest.warns(
            DeprecationWarning, match="fetch_arrow_table\\(\\) is deprecated, use to_arrow_table\\(\\) instead"
        ):
            self.con.fetch_arrow_table()

    def test_connection_fetch_record_batch_deprecated(self):
        self.con.execute("SELECT 1")
        with pytest.warns(
            DeprecationWarning, match="fetch_record_batch\\(\\) is deprecated, use to_arrow_reader\\(\\) instead"
        ):
            self.con.fetch_record_batch()

    def test_module_fetch_arrow_table_deprecated(self):
        vane.execute("SELECT 1")
        with pytest.warns(
            DeprecationWarning, match="fetch_arrow_table\\(\\) is deprecated, use to_arrow_table\\(\\) instead"
        ):
            vane.fetch_arrow_table()

    def test_module_fetch_record_batch_deprecated(self):
        vane.execute("SELECT 1")
        with pytest.warns(
            DeprecationWarning, match="fetch_record_batch\\(\\) is deprecated, use to_arrow_reader\\(\\) instead"
        ):
            vane.fetch_record_batch()

    def test_relation_to_arrow_table_works(self):
        rel = self.con.table("t")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = rel.to_arrow_table()
        assert result.num_rows == 1

    def test_relation_to_arrow_reader_works(self):
        rel = self.con.table("t")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reader = rel.to_arrow_reader()
        assert reader.read_all().num_rows == 1

    def test_relation_arrow_no_warning(self):
        """relation.arrow() should NOT emit a deprecation warning (soft deprecated)."""
        rel = self.con.table("t")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reader = rel.arrow()
        assert reader.read_all().num_rows == 1

    def test_connection_to_arrow_table_works(self):
        self.con.execute("SELECT 1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = self.con.to_arrow_table()
        assert result.num_rows == 1

    def test_connection_to_arrow_reader_works(self):
        self.con.execute("SELECT 1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reader = self.con.to_arrow_reader()
        assert reader.read_all().num_rows == 1

    def test_connection_arrow_no_warning(self):
        """connection.arrow() should NOT emit a deprecation warning (soft deprecated)."""
        self.con.execute("SELECT 1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reader = self.con.arrow()
        assert reader.read_all().num_rows == 1

    def test_module_to_arrow_table_works(self):
        vane.execute("SELECT 1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = vane.to_arrow_table()
        assert result.num_rows == 1

    def test_module_to_arrow_reader_works(self):
        vane.execute("SELECT 1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            reader = vane.to_arrow_reader()
        assert reader.read_all().num_rows == 1

    def test_module_arrow_no_warning(self):
        """vane.arrow(rows_per_batch) should NOT emit a deprecation warning (soft deprecated)."""
        vane.execute("SELECT 1")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = vane.arrow()
        assert result.read_all().num_rows == 1

    def test_from_arrow_not_deprecated(self):
        """vane.arrow(arrow_object) should NOT emit a deprecation warning."""
        import pyarrow as pa

        table = pa.table({"a": [1, 2, 3]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rel = vane.arrow(table)
        assert rel.fetchall() == [(1,), (2,), (3,)]
