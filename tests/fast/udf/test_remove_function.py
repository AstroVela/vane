# SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: MIT AND Apache-2.0
#
# Modified by Vane contributors.

import pytest

import vane
from vane.sqltypes import BIGINT, VARCHAR

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")


class TestRemoveFunction:
    def test_not_created(self):
        con = vane.connect()
        with pytest.raises(
            vane.InvalidInputException,
            match="No function by the name of 'not_a_registered_function' was found in the list of "
            "registered functions",
        ):
            con.remove_function("not_a_registered_function")

    def test_double_remove(self):
        def func(x: int) -> int:
            return x

        con = vane.connect()
        con.create_function("func", func)
        con.sql("select func(42)")
        con.remove_function("func")
        with pytest.raises(
            vane.InvalidInputException,
            match="No function by the name of 'func' was found in the list of registered functions",
        ):
            con.remove_function("func")

        with pytest.raises(vane.CatalogException, match="Scalar Function with name func does not exist!"):
            con.sql("select func(42)")

    def test_use_after_remove(self):
        def func(x: int) -> int:
            return x

        con = vane.connect()
        con.create_function("func", func)
        rel = con.sql("select func(42)")
        con.remove_function("func")
        """
            Error: Catalog Error: Scalar Function with name func does not exist!
        """
        with pytest.raises(vane.CatalogException, match="Scalar Function with name func does not exist!"):
            rel.fetchall()

    def test_use_after_remove_and_recreation(self):
        def func(x: str) -> str:
            return x

        con = vane.connect()
        con.create_function("func", func)

        with pytest.raises(vane.BinderException, match="No function matches the given name"):
            con.sql("select func(42)")

        rel2 = con.sql("select func('test'::VARCHAR)")
        con.remove_function("func")

        def also_func(x: int) -> int:
            return x

        con.create_function("func", also_func)
        with pytest.raises(vane.BinderException, match="No function matches the given name"):
            rel2.fetchall()

    def test_overwrite_name(self):
        def func(x):
            return x

        con = vane.connect()
        # create first version of the function
        con.create_function("func", func, [BIGINT], BIGINT)

        # create relation that uses the function
        rel1 = con.sql("select func('3')")

        def other_func(x):
            return x

        with pytest.raises(
            vane.NotImplementedException,
            match="A function by the name of 'func' is already created, creating multiple functions with the "
            "same name is not supported yet, please remove it first",
        ):
            con.create_function("func", other_func, [VARCHAR], VARCHAR)

        con.remove_function("func")

        with pytest.raises(
            vane.CatalogException, match="Catalog Error: Scalar Function with name func does not exist!"
        ):
            # Attempted to execute the relation using the 'func' function, but it was deleted
            rel1.fetchall()

        con.create_function("func", other_func, [VARCHAR], VARCHAR)
        # create relation that uses the new version
        rel2 = con.sql("select func('test')")

        # execute both relations
        res1 = rel1.fetchall()
        res2 = rel2.fetchall()
        # This has been converted to string, because the previous version of the function no longer exists
        assert res1 == [("3",)]
        assert res2 == [("test",)]
