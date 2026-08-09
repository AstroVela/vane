# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest

import vane

pa = pytest.importorskip("pyarrow")

ROW_COUNT = 70_000
ORDER_CASES = (
    pytest.param(1, 1, id="threads-1-batch-1"),
    pytest.param(2, 10, id="threads-2-batch-10"),
    pytest.param(4, 2048, id="threads-4-batch-2048"),
    pytest.param(8, 4097, id="threads-8-batch-4097"),
)


def make_batched_table(batch_size):
    table = pa.table({"value": range(ROW_COUNT)})
    return pa.Table.from_batches(table.to_batches(max_chunksize=batch_size))


@pytest.mark.parametrize("threads,batch_size", ORDER_CASES)
@pytest.mark.parametrize(
    "preserve_insertion_order",
    [None, True],
    ids=["default", "explicit-true"],
)
def test_arrow_scan_preserves_insertion_order(threads, batch_size, preserve_insertion_order):
    config = {"threads": threads}
    if preserve_insertion_order is not None:
        config["preserve_insertion_order"] = preserve_insertion_order

    with vane.connect(config=config) as connection:
        setting = connection.execute("SELECT current_setting('preserve_insertion_order')").fetchone()[0]
        actual = connection.from_arrow(make_batched_table(batch_size)).execute().fetchall()

    assert setting is True
    assert actual == [(value,) for value in range(ROW_COUNT)]


@pytest.mark.parametrize("threads,batch_size", ORDER_CASES)
def test_arrow_scan_can_disable_insertion_order_preservation(threads, batch_size):
    with vane.connect(config={"threads": threads, "preserve_insertion_order": False}) as connection:
        setting = connection.execute("SELECT current_setting('preserve_insertion_order')").fetchone()[0]
        actual = connection.from_arrow(make_batched_table(batch_size)).execute().fetchall()

    assert setting is False
    assert len(actual) == ROW_COUNT
    assert Counter(actual) == Counter((value,) for value in range(ROW_COUNT))
