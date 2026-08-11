# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

try:
    import ray
except Exception:
    ray = None

import vane
from vane import runners


def _sql_string(value):
    return value.replace("'", "''")


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
@pytest.mark.parametrize(
    ("function_name", "suffix", "payloads"),
    [
        ("read_blob", ".blob", [b"\x00\xffbinary-a", b"binary-b\x00"]),
        ("read_text", ".txt", ["plain text", "unicode \u6587\u672c"]),
    ],
    ids=["blob", "text"],
)
def test_read_file_functions_run_through_ray(tmp_path, function_name, suffix, payloads):
    pa = pytest.importorskip("pyarrow")

    expected = []
    for index, payload in enumerate(payloads):
        path = tmp_path / f"input-{index}{suffix}"
        if isinstance(payload, bytes):
            path.write_bytes(payload)
            size = len(payload)
        else:
            path.write_text(payload, encoding="utf-8")
            size = len(payload.encode("utf-8"))
        expected.append((str(path), payload, size))

    pattern = _sql_string(str(tmp_path / f"*{suffix}"))
    connection = vane.connect()
    try:
        relation = connection.sql(f"SELECT filename, content, size FROM {function_name}('{pattern}')")
        runners.set_runner_ray(noop_if_initialized=True)
        runner = runners.get_or_create_runner()

        partitions = list(runner.run_iter_tables(relation))
        tables = [partition.to_arrow() if hasattr(partition, "to_arrow") else partition for partition in partitions]
        result = pa.concat_tables(tables)
        actual = list(
            zip(
                result.column(0).to_pylist(),
                result.column(1).to_pylist(),
                result.column(2).to_pylist(),
                strict=True,
            )
        )

        assert sorted(actual) == sorted(expected)
    finally:
        connection.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
@pytest.mark.parametrize("function_name", ["read_blob", "read_text"], ids=["blob", "text"])
def test_read_file_functions_allow_empty_glob_through_ray(tmp_path, function_name):
    pattern = _sql_string(str(tmp_path / "missing-*"))
    connection = vane.connect()
    try:
        assert connection.sql(f"SELECT filename FROM {function_name}('{pattern}')").fetchall() == []

        relation = connection.sql(f"SELECT filename FROM {function_name}('{pattern}')")
        runners.set_runner_ray(noop_if_initialized=True)
        runner = runners.get_or_create_runner()

        partitions = list(runner.run_iter_tables(relation))
        tables = [partition.to_arrow() if hasattr(partition, "to_arrow") else partition for partition in partitions]
        assert sum(table.num_rows for table in tables) == 0
    finally:
        connection.close()


@pytest.mark.skipif(ray is None, reason="ray not installed")
@pytest.mark.usefixtures("ray_local")
def test_unsupported_table_function_reports_user_error():
    connection = vane.connect()
    try:
        relation = connection.sql("SELECT name FROM duckdb_settings()")
        runners.set_runner_ray(noop_if_initialized=True)
        runner = runners.get_or_create_runner()

        with pytest.raises(
            vane.InvalidInputException,
            match=r'Ray runner.*table function "duckdb_settings".*bind data is missing',
        ):
            list(runner.run_iter_tables(relation))
    finally:
        connection.close()
