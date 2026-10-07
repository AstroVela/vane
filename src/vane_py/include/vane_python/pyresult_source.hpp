// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "vane_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

struct DuckDBPyResultMetadata {
	vector<string> names;
	vector<LogicalType> types;
	ClientProperties client_properties;
};

//! Native result ownership shared by local cursor and Relation conversions.
class DuckDBPyResultSource {
public:
	virtual ~DuckDBPyResultSource() = default;

	virtual const DuckDBPyResultMetadata &Metadata() const = 0;
	virtual unique_ptr<DataChunk> FetchChunk(bool raw = false) = 0;
	virtual ArrowArrayStream TakeArrowStream(idx_t rows_per_batch) = 0;
	//! Only live native streams drive the source connection while being fetched.
	virtual bool RequiresConnectionLock() const = 0;
	virtual optional_idx KnownRowCount() const = 0;
	virtual bool IsClosed() const = 0;
	virtual void Close() = 0;
};

unique_ptr<DuckDBPyResultSource> MakeLocalPyResultSource(unique_ptr<QueryResult> result);

} // namespace duckdb
