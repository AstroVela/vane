# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vane.execution.udf_arrow_parquet_sink import write_terminal_arrow_parquet_output


def _sink_payload(output_directory, *, write_empty_file=True):
    return {
        "terminal_arrow_parquet_sink": True,
        "terminal_arrow_parquet_output_directory": str(output_directory),
        "terminal_arrow_parquet_file_extension": "parquet",
        "terminal_arrow_parquet_writer_options": {
            "compression": "zstd",
            "compression_level": 3,
            "data_page_version": "1.0",
            "dictionary_pagesize_limit": 1 << 20,
            "row_group_size": 2,
            "version": "1.0",
        },
        "terminal_arrow_parquet_expected_names": ["x", "label"],
        "terminal_arrow_parquet_expected_types": ["INTEGER", "VARCHAR"],
        "terminal_arrow_parquet_write_empty_file": write_empty_file,
    }


def test_terminal_arrow_parquet_sink_writes_one_file_and_copy_statistics(tmp_path):
    payload = _sink_payload(tmp_path / "output")
    statistics = write_terminal_arrow_parquet_output(
        payload,
        (
            pa.table({"x": pa.array([1], type=pa.int32()), "label": ["a"]}),
            pa.table({"x": pa.array([2], type=pa.int32()), "label": ["b"]}),
            pa.table({"x": pa.array([3], type=pa.int32()), "label": ["c"]}),
        ),
        invocation_id="lease:1",
    )

    assert statistics.schema.names == [
        "filename",
        "count",
        "file_size_bytes",
        "footer_size_bytes",
        "column_statistics",
        "partition_keys",
    ]
    assert statistics.column("count").to_pylist() == [3]
    output_path = statistics.column("filename")[0].as_py()
    assert statistics.column("file_size_bytes")[0].as_py() > 0
    assert statistics.column("footer_size_bytes")[0].as_py() > 0
    assert statistics.column("column_statistics")[0].as_py() == []
    assert statistics.column("partition_keys")[0].as_py() == []
    assert pq.read_table(output_path).to_pydict() == {"x": [1, 2, 3], "label": ["a", "b", "c"]}
    metadata = pq.ParquetFile(output_path).metadata
    assert "parquet-cpp-arrow" in str(metadata.created_by).lower()
    assert metadata.format_version == "1.0"
    assert metadata.num_row_groups == 2
    assert [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)] == [2, 1]


def test_terminal_arrow_parquet_sink_honors_no_empty_file(tmp_path):
    output_directory = tmp_path / "empty-output"
    payload = _sink_payload(output_directory, write_empty_file=False)
    empty = pa.table(
        {
            "x": pa.array([], type=pa.int32()),
            "label": pa.array([], type=pa.string()),
        }
    )

    statistics = write_terminal_arrow_parquet_output(payload, (empty,), invocation_id="lease:empty")

    assert statistics.num_rows == 0
    assert not output_directory.exists()


def test_terminal_arrow_parquet_sink_writes_schema_only_file_for_empty_input(tmp_path):
    output_directory = tmp_path / "empty-output"
    payload = _sink_payload(output_directory)

    statistics = write_terminal_arrow_parquet_output(payload, (), invocation_id="lease:empty")

    assert statistics.column("count").to_pylist() == [0]
    output_path = statistics.column("filename")[0].as_py()
    assert pq.read_schema(output_path) == pa.schema([("x", pa.int32()), ("label", pa.string())])
    assert pq.read_table(output_path).num_rows == 0


def test_terminal_arrow_parquet_sink_rejects_schema_mismatch(tmp_path):
    payload = _sink_payload(tmp_path / "output")

    with pytest.raises(ValueError, match="output columns do not match COPY schema"):
        write_terminal_arrow_parquet_output(
            payload,
            (pa.table({"wrong": [1], "label": ["a"]}),),
            invocation_id="lease:bad-schema",
        )


def test_terminal_arrow_parquet_sink_renames_internal_row_preserving_columns(tmp_path):
    payload = _sink_payload(tmp_path / "output")
    payload.update({"call_mode": "map", "scalar_arg_count": 1})

    statistics = write_terminal_arrow_parquet_output(
        payload,
        (pa.table({"c1": pa.array([1], type=pa.int32()), "value": ["a"]}),),
        invocation_id="lease:row-preserving",
    )

    output_path = statistics.column("filename")[0].as_py()
    assert pq.read_table(output_path).to_pydict() == {"x": [1], "label": ["a"]}


def test_terminal_arrow_parquet_sink_honors_file_extension(tmp_path):
    payload = _sink_payload(tmp_path / "output")
    payload["terminal_arrow_parquet_file_extension"] = "pq"

    statistics = write_terminal_arrow_parquet_output(
        payload,
        (pa.table({"x": pa.array([1], type=pa.int32()), "label": ["a"]}),),
        invocation_id="lease:extension",
    )

    output_path = statistics.column("filename")[0].as_py()
    assert output_path.endswith(".pq")
    assert pq.read_table(output_path).to_pydict() == {"x": [1], "label": ["a"]}
