// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "vane_python/pyconnection/pyconnection.hpp"

namespace duckdb {

Optional<py::list> DuckDBPyConnection::GetDescription() {
	auto query_lock = LockForQuery();
	if (!con.HasResult()) {
		return py::none();
	}
	auto &result = con.GetResult();
	return result.Description();
}

int DuckDBPyConnection::GetRowcount() {
	return -1;
}

// these should be functions on the result but well
Optional<py::tuple> DuckDBPyConnection::FetchOne() {
	auto query_lock = LockForQuery();
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchOne();
}

py::list DuckDBPyConnection::FetchMany(idx_t size) {
	auto query_lock = LockForQuery();
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchMany(size);
}

py::list DuckDBPyConnection::FetchAll() {
	auto query_lock = LockForQuery();
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchAll();
}

py::dict DuckDBPyConnection::FetchNumpy() {
	auto query_lock = LockForQuery();
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchNumpyInternal();
}

PandasDataFrame DuckDBPyConnection::FetchDF(bool date_as_object) {
	auto query_lock = LockForQuery();
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchDF(date_as_object);
}

PandasDataFrame DuckDBPyConnection::FetchDFChunk(const idx_t vectors_per_chunk, bool date_as_object) {
	auto query_lock = LockForQuery();
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchDFChunk(vectors_per_chunk, date_as_object);
}

duckdb::pyarrow::Table DuckDBPyConnection::FetchArrow(idx_t rows_per_batch) {
	auto query_lock = LockForQuery();
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.ToArrowTable(rows_per_batch);
}

py::dict DuckDBPyConnection::FetchPyTorch() {
	auto query_lock = LockForQuery();
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchPyTorch();
}

py::dict DuckDBPyConnection::FetchTF() {
	auto query_lock = LockForQuery();
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchTF();
}

PolarsDataFrame DuckDBPyConnection::FetchPolars(idx_t rows_per_batch, bool lazy) {
	auto query_lock = LockForQuery();
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.ToPolars(rows_per_batch, lazy);
}

duckdb::pyarrow::RecordBatchReader DuckDBPyConnection::FetchRecordBatchReader(const idx_t rows_per_batch) {
	auto query_lock = LockForQuery();
	if (!con.HasResult()) {
		throw InvalidInputException("No open result set");
	}
	auto &result = con.GetResult();
	return result.FetchRecordBatchReader(rows_per_batch);
}

} // namespace duckdb
