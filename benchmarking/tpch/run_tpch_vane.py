#!/usr/bin/env python3
"""Run Daft TPC-H benchmarks with daft 0.6.2 against local parquet data.

Usage:
    python benchmarking/tpch/run_tpch_vane.py \
        --parquet_folder data/tpch10 \
        --questions 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
"""

from __future__ import annotations

import argparse
import datetime
import os
import statistics
import time
from collections.abc import Callable

import daft
from daft import DataFrame, DataType, col, lit

print(f"daft version: {daft.__version__}")
print(f"daft location: {daft.__file__}")


# ── helpers ──────────────────────────────────────────────────────────────────
GetDFFunc = Callable[[str], DataFrame]


def get_df_with_parquet_folder(parquet_folder: str) -> Callable[[str], DataFrame]:
    def _get_df(table_name: str) -> DataFrame:
        df = daft.read_parquet(os.path.join(parquet_folder, table_name, "*.parquet"))
        # Rename lowercase columns to uppercase, and cast Decimal to Float64
        # to avoid Decimal precision overflow in daft 0.6.2
        exprs = []
        for c in df.column_names:
            e = col(c)
            if "Decimal" in str(df.schema()[c]):
                e = e.cast(DataType.float64())
            exprs.append(e.alias(c.upper()))
        return df.select(*exprs)

    return _get_df


# ── TPC-H queries (adapted from Daft/benchmarking/tpch/answers.py for 0.6.2) ──


def q1(get_df: GetDFFunc) -> DataFrame:
    lineitem = get_df("lineitem")
    discounted_price = col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))
    taxed_discounted_price = discounted_price * (1 + col("L_TAX"))
    return (
        lineitem.where(col("L_SHIPDATE") <= datetime.date(1998, 9, 2))
        .groupby(col("L_RETURNFLAG"), col("L_LINESTATUS"))
        .agg(
            col("L_QUANTITY").sum().alias("sum_qty"),
            col("L_EXTENDEDPRICE").sum().alias("sum_base_price"),
            discounted_price.sum().alias("sum_disc_price"),
            taxed_discounted_price.sum().alias("sum_charge"),
            col("L_QUANTITY").mean().alias("avg_qty"),
            col("L_EXTENDEDPRICE").mean().alias("avg_price"),
            col("L_DISCOUNT").mean().alias("avg_disc"),
            col("L_QUANTITY").count().alias("count_order"),
        )
        .sort(["L_RETURNFLAG", "L_LINESTATUS"])
    )


def q2(get_df: GetDFFunc) -> DataFrame:
    region = get_df("region")
    nation = get_df("nation")
    supplier = get_df("supplier")
    partsupp = get_df("partsupp")
    part = get_df("part")

    europe = (
        region.where(col("R_NAME") == "EUROPE")
        .join(nation, left_on=col("R_REGIONKEY"), right_on=col("N_REGIONKEY"))
        .join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))
        .join(partsupp, left_on=col("S_SUPPKEY"), right_on=col("PS_SUPPKEY"))
    )
    brass = part.where((col("P_SIZE") == 15) & col("P_TYPE").endswith("BRASS")).join(
        europe,
        left_on=col("P_PARTKEY"),
        right_on=col("PS_PARTKEY"),
    )
    min_cost = brass.groupby(col("P_PARTKEY")).agg(col("PS_SUPPLYCOST").min().alias("min"))
    return (
        brass.join(min_cost, on=col("P_PARTKEY"))
        .where(col("PS_SUPPLYCOST") == col("min"))
        .select("S_ACCTBAL", "S_NAME", "N_NAME", "P_PARTKEY", "P_MFGR", "S_ADDRESS", "S_PHONE", "S_COMMENT")
        .sort(by=["S_ACCTBAL", "N_NAME", "S_NAME", "P_PARTKEY"], desc=[True, False, False, False])
        .limit(100)
    )


def q3(get_df: GetDFFunc) -> DataFrame:
    customer = get_df("customer").where(col("C_MKTSEGMENT") == "BUILDING")
    orders = get_df("orders").where(col("O_ORDERDATE") < datetime.date(1995, 3, 15))
    lineitem = get_df("lineitem").where(col("L_SHIPDATE") > datetime.date(1995, 3, 15))
    return (
        customer.join(orders, left_on=col("C_CUSTKEY"), right_on=col("O_CUSTKEY"))
        .select(col("O_ORDERKEY"), col("O_ORDERDATE"), col("O_SHIPPRIORITY"))
        .join(lineitem, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
        .select(
            col("O_ORDERKEY"),
            (col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))).alias("volume"),
            col("O_ORDERDATE"),
            col("O_SHIPPRIORITY"),
        )
        .groupby(col("O_ORDERKEY"), col("O_ORDERDATE"), col("O_SHIPPRIORITY"))
        .agg(col("volume").sum().alias("revenue"))
        .sort(by=["revenue", "O_ORDERDATE"], desc=[True, False])
        .limit(10)
        .select("O_ORDERKEY", "revenue", "O_ORDERDATE", "O_SHIPPRIORITY")
    )


def q4(get_df: GetDFFunc) -> DataFrame:
    orders = get_df("orders").where(
        (col("O_ORDERDATE") >= datetime.date(1993, 7, 1)) & (col("O_ORDERDATE") < datetime.date(1993, 10, 1))
    )
    lineitems = get_df("lineitem").where(col("L_COMMITDATE") < col("L_RECEIPTDATE"))
    return (
        orders.join(lineitems, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"), how="semi")
        .groupby(col("O_ORDERPRIORITY"))
        .agg(col("O_ORDERKEY").count().alias("order_count"))
        .sort(col("O_ORDERPRIORITY"))
    )


def q5(get_df: GetDFFunc) -> DataFrame:
    orders = get_df("orders").where(
        (col("O_ORDERDATE") >= datetime.date(1994, 1, 1)) & (col("O_ORDERDATE") < datetime.date(1995, 1, 1))
    )
    region = get_df("region").where(col("R_NAME") == "ASIA")
    nation = get_df("nation")
    supplier = get_df("supplier")
    lineitem = get_df("lineitem")
    customer = get_df("customer")
    return (
        region.join(nation, left_on=col("R_REGIONKEY"), right_on=col("N_REGIONKEY"))
        .join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))
        .join(lineitem, left_on=col("S_SUPPKEY"), right_on=col("L_SUPPKEY"))
        .select(col("N_NAME"), col("L_EXTENDEDPRICE"), col("L_DISCOUNT"), col("L_ORDERKEY"), col("N_NATIONKEY"))
        .join(orders, left_on=col("L_ORDERKEY"), right_on=col("O_ORDERKEY"))
        .join(customer, left_on=[col("O_CUSTKEY"), col("N_NATIONKEY")], right_on=[col("C_CUSTKEY"), col("C_NATIONKEY")])
        .select(col("N_NAME"), (col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))).alias("value"))
        .groupby(col("N_NAME"))
        .agg(col("value").sum().alias("revenue"))
        .sort(col("revenue"), desc=True)
    )


def q6(get_df: GetDFFunc) -> DataFrame:
    lineitem = get_df("lineitem")
    return lineitem.where(
        (col("L_SHIPDATE") >= datetime.date(1994, 1, 1))
        & (col("L_SHIPDATE") < datetime.date(1995, 1, 1))
        & (col("L_DISCOUNT") >= 0.05)
        & (col("L_DISCOUNT") <= 0.07)
        & (col("L_QUANTITY") < 24)
    ).sum(col("L_EXTENDEDPRICE") * col("L_DISCOUNT"))


def q7(get_df: GetDFFunc) -> DataFrame:
    lineitem = get_df("lineitem").where(
        (col("L_SHIPDATE") >= datetime.date(1995, 1, 1)) & (col("L_SHIPDATE") <= datetime.date(1996, 12, 31))
    )
    nation = get_df("nation").where((col("N_NAME") == "FRANCE") | (col("N_NAME") == "GERMANY"))
    supplier = get_df("supplier")
    customer = get_df("customer")
    orders = get_df("orders")

    supNation = (
        nation.join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))
        .join(lineitem, left_on=col("S_SUPPKEY"), right_on=col("L_SUPPKEY"))
        .select(
            col("N_NAME").alias("supp_nation"),
            col("L_ORDERKEY"),
            col("L_EXTENDEDPRICE"),
            col("L_DISCOUNT"),
            col("L_SHIPDATE"),
        )
    )
    return (
        nation.join(customer, left_on=col("N_NATIONKEY"), right_on=col("C_NATIONKEY"))
        .join(orders, left_on=col("C_CUSTKEY"), right_on=col("O_CUSTKEY"))
        .select(col("N_NAME").alias("cust_nation"), col("O_ORDERKEY"))
        .join(supNation, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
        .where(
            ((col("supp_nation") == "FRANCE") & (col("cust_nation") == "GERMANY"))
            | ((col("supp_nation") == "GERMANY") & (col("cust_nation") == "FRANCE"))
        )
        .select(
            col("supp_nation"),
            col("cust_nation"),
            col("L_SHIPDATE").year().alias("l_year"),
            (col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))).alias("volume"),
        )
        .groupby(col("supp_nation"), col("cust_nation"), col("l_year"))
        .agg(col("volume").sum().alias("revenue"))
        .sort(by=["supp_nation", "cust_nation", "l_year"])
    )


def q8(get_df: GetDFFunc) -> DataFrame:
    region = get_df("region").where(col("R_NAME") == "AMERICA")
    orders = get_df("orders").where(
        (col("O_ORDERDATE") <= datetime.date(1996, 12, 31)) & (col("O_ORDERDATE") >= datetime.date(1995, 1, 1))
    )
    part = get_df("part").where(col("P_TYPE") == "ECONOMY ANODIZED STEEL")
    nation = get_df("nation")
    supplier = get_df("supplier")
    lineitem = get_df("lineitem")
    customer = get_df("customer")

    nat = nation.join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))
    line = (
        lineitem.select(
            col("L_PARTKEY"),
            col("L_SUPPKEY"),
            col("L_ORDERKEY"),
            (col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))).alias("volume"),
        )
        .join(part, left_on=col("L_PARTKEY"), right_on=col("P_PARTKEY"))
        .join(nat, left_on=col("L_SUPPKEY"), right_on=col("S_SUPPKEY"))
    )
    return (
        nation.join(region, left_on=col("N_REGIONKEY"), right_on=col("R_REGIONKEY"))
        .select(col("N_NATIONKEY"))
        .join(customer, left_on=col("N_NATIONKEY"), right_on=col("C_NATIONKEY"))
        .select(col("C_CUSTKEY"))
        .join(orders, left_on=col("C_CUSTKEY"), right_on=col("O_CUSTKEY"))
        .select(col("O_ORDERKEY"), col("O_ORDERDATE"))
        .join(line, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
        .select(
            col("O_ORDERDATE").year().alias("o_year"),
            col("volume"),
            (col("N_NAME") == "BRAZIL").if_else(col("volume"), lit(0.0)).alias("case_volume"),
        )
        .groupby(col("o_year"))
        .agg(col("case_volume").sum().alias("case_volume_sum"), col("volume").sum().alias("volume_sum"))
        .select(col("o_year"), col("case_volume_sum") / col("volume_sum"))
        .sort(col("o_year"))
    )


def q9(get_df: GetDFFunc) -> DataFrame:
    lineitem = get_df("lineitem")
    part = get_df("part")
    nation = get_df("nation")
    supplier = get_df("supplier")
    partsupp = get_df("partsupp")
    orders = get_df("orders")

    linepart = part.where(col("P_NAME").contains("green")).join(
        lineitem, left_on=col("P_PARTKEY"), right_on=col("L_PARTKEY")
    )
    natsup = nation.join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))
    return (
        linepart.join(natsup, left_on=col("L_SUPPKEY"), right_on=col("S_SUPPKEY"))
        .join(partsupp, left_on=[col("L_SUPPKEY"), col("P_PARTKEY")], right_on=[col("PS_SUPPKEY"), col("PS_PARTKEY")])
        .join(orders, left_on=col("L_ORDERKEY"), right_on=col("O_ORDERKEY"))
        .select(
            col("N_NAME"),
            col("O_ORDERDATE").year().alias("o_year"),
            (col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT")) - col("PS_SUPPLYCOST") * col("L_QUANTITY")).alias(
                "amount"
            ),
        )
        .groupby(col("N_NAME"), col("o_year"))
        .agg(col("amount").sum())
        .sort(by=["N_NAME", "o_year"], desc=[False, True])
    )


def q10(get_df: GetDFFunc) -> DataFrame:
    lineitem = get_df("lineitem").where(col("L_RETURNFLAG") == "R")
    orders = get_df("orders")
    nation = get_df("nation")
    customer = get_df("customer")
    return (
        orders.where(
            (col("O_ORDERDATE") < datetime.date(1994, 1, 1)) & (col("O_ORDERDATE") >= datetime.date(1993, 10, 1))
        )
        .join(customer, left_on=col("O_CUSTKEY"), right_on=col("C_CUSTKEY"))
        .join(nation, left_on=col("C_NATIONKEY"), right_on=col("N_NATIONKEY"))
        .join(lineitem, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
        .select(
            col("O_CUSTKEY"),
            col("C_NAME"),
            (col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))).alias("volume"),
            col("C_ACCTBAL"),
            col("N_NAME"),
            col("C_ADDRESS"),
            col("C_PHONE"),
            col("C_COMMENT"),
        )
        .groupby("O_CUSTKEY", "C_NAME", "C_ACCTBAL", "C_PHONE", "N_NAME", "C_ADDRESS", "C_COMMENT")
        .agg(col("volume").sum().alias("revenue"))
        .sort(col("revenue"), desc=True)
        .select("O_CUSTKEY", "C_NAME", "revenue", "C_ACCTBAL", "N_NAME", "C_ADDRESS", "C_PHONE", "C_COMMENT")
        .limit(20)
    )


def q11(get_df: GetDFFunc) -> DataFrame:
    partsupp = get_df("partsupp")
    supplier = get_df("supplier")
    nation = get_df("nation")
    var_1 = "GERMANY"
    var_2 = 0.0001 / 1

    res_1 = (
        partsupp.join(supplier, left_on=col("PS_SUPPKEY"), right_on=col("S_SUPPKEY"))
        .join(nation, left_on=col("S_NATIONKEY"), right_on=col("N_NATIONKEY"))
        .where(col("N_NAME") == var_1)
    )
    res_2 = res_1.agg((col("PS_SUPPLYCOST") * col("PS_AVAILQTY")).sum().alias("tmp")).select(
        col("tmp") * var_2, lit(1).alias("lit")
    )
    return (
        res_1.groupby("PS_PARTKEY")
        .agg((col("PS_SUPPLYCOST") * col("PS_AVAILQTY")).sum().alias("value"))
        .with_column("lit", lit(1))
        .join(res_2, on="lit")
        .where(col("value") > col("tmp"))
        .select(col("PS_PARTKEY"), col("value").round(2))
        .sort(col("value"), desc=True)
    )


def q12(get_df: GetDFFunc) -> DataFrame:
    orders = get_df("orders")
    lineitem = get_df("lineitem")
    return (
        orders.join(lineitem, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
        .where(
            col("L_SHIPMODE").is_in(["MAIL", "SHIP"])
            & (col("L_COMMITDATE") < col("L_RECEIPTDATE"))
            & (col("L_SHIPDATE") < col("L_COMMITDATE"))
            & (col("L_RECEIPTDATE") >= datetime.date(1994, 1, 1))
            & (col("L_RECEIPTDATE") < datetime.date(1995, 1, 1))
        )
        .groupby(col("L_SHIPMODE"))
        .agg(
            col("O_ORDERPRIORITY").is_in(["1-URGENT", "2-HIGH"]).if_else(lit(1), lit(0)).sum().alias("high_line_count"),
            (~col("O_ORDERPRIORITY").is_in(["1-URGENT", "2-HIGH"]))
            .if_else(lit(1), lit(0))
            .sum()
            .alias("low_line_count"),
        )
        .sort(col("L_SHIPMODE"))
    )


def q13(get_df: GetDFFunc) -> DataFrame:
    customers = get_df("customer")
    orders = get_df("orders")
    return (
        customers.join(
            orders.where(~col("O_COMMENT").regexp(".*special.*requests.*")),
            left_on="C_CUSTKEY",
            right_on="O_CUSTKEY",
            how="left",
        )
        .groupby(col("C_CUSTKEY"))
        .agg(col("O_ORDERKEY").count().alias("c_count"))
        .sort("C_CUSTKEY")
        .groupby("c_count")
        .agg(col("c_count").count().alias("custdist"))
        .sort(["custdist", "c_count"], desc=[True, True])
    )


def q14(get_df: GetDFFunc) -> DataFrame:
    lineitem = get_df("lineitem")
    part = get_df("part")
    return (
        lineitem.join(part, left_on=col("L_PARTKEY"), right_on=col("P_PARTKEY"))
        .where((col("L_SHIPDATE") >= datetime.date(1995, 9, 1)) & (col("L_SHIPDATE") < datetime.date(1995, 10, 1)))
        .agg(
            col("P_TYPE")
            .startswith("PROMO")
            .if_else(col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT")), lit(0))
            .sum()
            .alias("tmp_1"),
            (col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))).sum().alias("tmp_2"),
        )
        .select(100.00 * (col("tmp_1") / col("tmp_2")).alias("promo_revenue"))
    )


def q15(get_df: GetDFFunc) -> DataFrame:
    lineitem = get_df("lineitem")
    revenue = (
        lineitem.where(
            (col("L_SHIPDATE") >= datetime.date(1996, 1, 1)) & (col("L_SHIPDATE") < datetime.date(1996, 4, 1))
        )
        .groupby(col("L_SUPPKEY"))
        .agg((col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))).sum().alias("total_revenue"))
        .select(col("L_SUPPKEY").alias("supplier_no"), "total_revenue")
    )
    revenue = revenue.join(revenue.max("total_revenue"), on="total_revenue")
    supplier = get_df("supplier")
    return (
        supplier.join(revenue, left_on=col("S_SUPPKEY"), right_on=col("supplier_no"))
        .select("S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_PHONE", "total_revenue")
        .sort("S_SUPPKEY")
    )


def q16(get_df: GetDFFunc) -> DataFrame:
    part = get_df("part")
    partsupp = get_df("partsupp")
    supplier = get_df("supplier")
    suppkeys = supplier.where(col("S_COMMENT").regexp(".*Customer.*Complaints.*")).select(
        col("S_SUPPKEY"), col("S_SUPPKEY").alias("PS_SUPPKEY_RIGHT")
    )
    return (
        part.join(partsupp, left_on=col("P_PARTKEY"), right_on=col("PS_PARTKEY"))
        .where(
            (col("P_BRAND") != "Brand#45")
            & ~col("P_TYPE").startswith("MEDIUM POLISHED")
            & (col("P_SIZE").is_in([49, 14, 23, 45, 19, 3, 36, 9]))
        )
        .join(suppkeys, left_on="PS_SUPPKEY", right_on="S_SUPPKEY", how="left")
        .where(col("PS_SUPPKEY_RIGHT").is_null())
        .select("P_BRAND", "P_TYPE", "P_SIZE", "PS_SUPPKEY")
        .distinct()
        .groupby("P_BRAND", "P_TYPE", "P_SIZE")
        .agg(col("PS_SUPPKEY").count().alias("supplier_cnt"))
        .sort(["supplier_cnt", "P_BRAND", "P_TYPE", "P_SIZE"], desc=[True, False, False, False])
    )


def q17(get_df: GetDFFunc) -> DataFrame:
    lineitem = get_df("lineitem")
    part = get_df("part")
    res_1 = part.where((col("P_BRAND") == "Brand#23") & (col("P_CONTAINER") == "MED BOX")).join(
        lineitem, left_on="P_PARTKEY", right_on="L_PARTKEY", how="left"
    )
    return (
        res_1.groupby("P_PARTKEY")
        .agg((0.2 * col("L_QUANTITY")).mean().alias("avg_quantity"))
        .select(col("P_PARTKEY").alias("key"), col("avg_quantity"))
        .join(res_1, left_on="key", right_on="P_PARTKEY")
        .where(col("L_QUANTITY") < col("avg_quantity"))
        .agg((col("L_EXTENDEDPRICE") / 7.0).sum().alias("avg_yearly"))
    )


def q18(get_df: GetDFFunc) -> DataFrame:
    customer = get_df("customer")
    orders = get_df("orders")
    lineitem = get_df("lineitem")
    res_1 = lineitem.groupby("L_ORDERKEY").agg(col("L_QUANTITY").sum().alias("sum_qty")).where(col("sum_qty") > 300)
    return (
        orders.join(res_1, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
        .join(customer, left_on=col("O_CUSTKEY"), right_on=col("C_CUSTKEY"))
        .join(lineitem, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
        .groupby("C_NAME", "C_CUSTKEY", "O_ORDERKEY", "O_ORDERDATE", "O_TOTALPRICE")
        .agg(col("L_QUANTITY").sum().alias("sum"))
        .select("C_NAME", "C_CUSTKEY", "O_ORDERKEY", col("O_ORDERDATE").alias("O_ORDERDAT"), "O_TOTALPRICE", "sum")
        .sort(["O_TOTALPRICE", "O_ORDERDAT"], desc=[True, False])
        .limit(100)
    )


def q19(get_df: GetDFFunc) -> DataFrame:
    lineitem = get_df("lineitem")
    part = get_df("part")
    return (
        lineitem.join(part, left_on=col("L_PARTKEY"), right_on=col("P_PARTKEY"))
        .where(
            (
                (col("P_BRAND") == "Brand#12")
                & col("P_CONTAINER").is_in(["SM CASE", "SM BOX", "SM PACK", "SM PKG"])
                & (col("L_QUANTITY") >= 1)
                & (col("L_QUANTITY") <= 11)
                & (col("P_SIZE") >= 1)
                & (col("P_SIZE") <= 5)
                & col("L_SHIPMODE").is_in(["AIR", "AIR REG"])
                & (col("L_SHIPINSTRUCT") == "DELIVER IN PERSON")
            )
            | (
                (col("P_BRAND") == "Brand#23")
                & col("P_CONTAINER").is_in(["MED BAG", "MED BOX", "MED PKG", "MED PACK"])
                & (col("L_QUANTITY") >= 10)
                & (col("L_QUANTITY") <= 20)
                & (col("P_SIZE") >= 1)
                & (col("P_SIZE") <= 10)
                & col("L_SHIPMODE").is_in(["AIR", "AIR REG"])
                & (col("L_SHIPINSTRUCT") == "DELIVER IN PERSON")
            )
            | (
                (col("P_BRAND") == "Brand#34")
                & col("P_CONTAINER").is_in(["LG CASE", "LG BOX", "LG PACK", "LG PKG"])
                & (col("L_QUANTITY") >= 20)
                & (col("L_QUANTITY") <= 30)
                & (col("P_SIZE") >= 1)
                & (col("P_SIZE") <= 15)
                & col("L_SHIPMODE").is_in(["AIR", "AIR REG"])
                & (col("L_SHIPINSTRUCT") == "DELIVER IN PERSON")
            )
        )
        .agg((col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))).sum().alias("revenue"))
    )


def q20(get_df: GetDFFunc) -> DataFrame:
    supplier = get_df("supplier")
    nation = get_df("nation")
    part = get_df("part")
    partsupp = get_df("partsupp")
    lineitem = get_df("lineitem")

    res_1 = (
        lineitem.where(
            (col("L_SHIPDATE") >= datetime.date(1994, 1, 1)) & (col("L_SHIPDATE") < datetime.date(1995, 1, 1))
        )
        .groupby("L_PARTKEY", "L_SUPPKEY")
        .agg(((col("L_QUANTITY") * 0.5).sum()).alias("sum_quantity"))
    )
    res_2 = nation.where(col("N_NAME") == "CANADA")
    res_3 = supplier.join(res_2, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
    return (
        part.where(col("P_NAME").startswith("forest"))
        .select("P_PARTKEY")
        .distinct()
        .join(partsupp, left_on="P_PARTKEY", right_on="PS_PARTKEY")
        .join(res_1, left_on=["PS_SUPPKEY", "P_PARTKEY"], right_on=["L_SUPPKEY", "L_PARTKEY"])
        .where(col("PS_AVAILQTY") > col("sum_quantity"))
        .select("PS_SUPPKEY")
        .distinct()
        .join(res_3, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
        .select("S_NAME", "S_ADDRESS")
        .sort("S_NAME")
    )


def q21(get_df: GetDFFunc) -> DataFrame:
    supplier = get_df("supplier")
    nation = get_df("nation")
    lineitem = get_df("lineitem")
    orders = get_df("orders")

    res_1 = (
        lineitem.select("L_SUPPKEY", "L_ORDERKEY")
        .groupby("L_ORDERKEY")
        .agg(col("L_SUPPKEY").count().alias("nunique_col"))
        .where(col("nunique_col") > 1)
        .join(lineitem.where(col("L_RECEIPTDATE") > col("L_COMMITDATE")), on="L_ORDERKEY")
    )
    return (
        res_1.select("L_SUPPKEY", "L_ORDERKEY")
        .groupby("L_ORDERKEY")
        .agg(col("L_SUPPKEY").count().alias("nunique_col"))
        .join(res_1, on="L_ORDERKEY")
        .join(supplier, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
        .join(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
        .join(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
        .where((col("nunique_col") == 1) & (col("N_NAME") == "SAUDI ARABIA") & (col("O_ORDERSTATUS") == "F"))
        .groupby("S_NAME")
        .agg(col("O_ORDERKEY").count().alias("numwait"))
        .sort(["numwait", "S_NAME"], desc=[True, False])
        .limit(100)
    )


def q22(get_df: GetDFFunc) -> DataFrame:
    orders = get_df("orders")
    customer = get_df("customer")

    res_1 = (
        customer.with_column("cntrycode", col("C_PHONE").left(2))
        .where(col("cntrycode").is_in(["13", "31", "23", "29", "30", "18", "17"]))
        .select("C_ACCTBAL", "C_CUSTKEY", "cntrycode")
    )
    res_2 = (
        res_1.where(col("C_ACCTBAL") > 0).agg(col("C_ACCTBAL").mean().alias("avg_acctbal")).with_column("lit", lit(1))
    )
    return (
        res_1.join(orders, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="anti")
        .with_column("lit", lit(1))
        .join(res_2, on="lit")
        .where(col("C_ACCTBAL") > col("avg_acctbal"))
        .groupby("cntrycode")
        .agg(
            col("C_ACCTBAL").count().alias("numcust"),
            col("C_ACCTBAL").sum().alias("totacctbal"),
        )
        .sort("cntrycode")
    )


# ── query registry ──
QUERIES = {
    1: q1,
    2: q2,
    3: q3,
    4: q4,
    5: q5,
    6: q6,
    7: q7,
    8: q8,
    9: q9,
    10: q10,
    11: q11,
    12: q12,
    13: q13,
    14: q14,
    15: q15,
    16: q16,
    17: q17,
    18: q18,
    19: q19,
    20: q20,
    21: q21,
    22: q22,
}


# ── main ──
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Daft TPC-H benchmarks (daft 0.6.2 compatible)")
    parser.add_argument("--parquet_folder", required=True, help="Path to parquet data root (e.g. data/tpch10)")
    parser.add_argument("--questions", default=None, help="Comma-separated query numbers (default: all 1-22)")
    parser.add_argument(
        "--runner",
        default="native",
        choices=["native", "ray"],
        help="Daft runner: 'native' (default) or 'ray' (local Ray cluster)",
    )
    parser.add_argument("--iterations", default=3, type=int, help="Number of iterations per query (default: 3)")
    args = parser.parse_args()

    if args.runner == "ray":
        import ray

        ray.init()
        daft.set_runner_ray()
        print(f"Runner: ray (nodes={len(ray.nodes())}, cpus={int(ray.cluster_resources().get('CPU', 0))})")
    else:
        daft.set_runner_native()
        print("Runner: native")

    if args.questions:
        questions = sorted({int(q) for q in args.questions.split(",")})
    else:
        questions = list(range(1, 23))

    get_df = get_df_with_parquet_folder(args.parquet_folder)
    iterations = args.iterations

    print(f"\nRunning TPC-H queries {questions} on data: {args.parquet_folder}")
    print(f"Iterations per query: {iterations}\n")
    print(f"{'Query':<10} {'Status':<10} {'Min (s)':<12} {'Avg (s)':<12} {'Med (s)':<12} {'Rows':<12}")
    print("-" * 74)

    total_time = 0.0
    failed = []
    for qnum in questions:
        query_fn = QUERIES[qnum]
        iter_times = []
        rows = 0
        query_failed = False
        for _ in range(iterations):
            t0 = time.time()
            try:
                result = query_fn(get_df)
                pdf = result.collect().to_pandas()
                elapsed = time.time() - t0
                iter_times.append(elapsed)
                rows = len(pdf)
            except Exception as e:
                elapsed = time.time() - t0
                query_failed = True
                failed.append(qnum)
                print(f"Q{qnum:<9} {'FAIL':<10} {'—':<12} {'—':<12} {'—':<12} {str(e)[:40]}")
                break
        if not query_failed and iter_times:
            min_t = min(iter_times)
            avg_t = sum(iter_times) / len(iter_times)
            med_t = statistics.median(iter_times)
            total_time += sum(iter_times)
            times_str = ", ".join(f"{t:.2f}" for t in iter_times)
            print(f"Q{qnum:<9} {'OK':<10} {min_t:<12.2f} {avg_t:<12.2f} {med_t:<12.2f} {rows:<12}")
            print(f"{'':>10} iters: [{times_str}]")

    print("-" * 74)
    print(
        f"Total: {total_time:.2f}s | Passed: {len(questions) - len(failed)}/{len(questions)} | Iterations: {iterations}"
    )
    if failed:
        print(f"Failed queries: {failed}")
