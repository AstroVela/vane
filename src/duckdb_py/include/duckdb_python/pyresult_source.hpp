// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"

namespace duckdb {

struct DuckDBPyResultMetadata {
	vector<string> names;
	vector<LogicalType> types;
	ClientProperties client_properties;
};

//! A backend-neutral source consumed by DuckDBPyResult.
//!
//! Local execution wraps DuckDB's QueryResult. Distributed execution exposes
//! the runner's Arrow partition stream through the same chunk/Arrow surfaces,
//! so Python result conversion and cursor state remain shared.
class DuckDBPyResultSource {
public:
	virtual ~DuckDBPyResultSource() = default;

	virtual const DuckDBPyResultMetadata &Metadata() const = 0;
	virtual unique_ptr<DataChunk> FetchChunk(bool raw = false) = 0;
	virtual ArrowArrayStream TakeArrowStream(idx_t rows_per_batch) = 0;
	virtual optional_idx KnownRowCount() const = 0;
	virtual bool IsClosed() const = 0;
	virtual void Close() = 0;
};

unique_ptr<DuckDBPyResultSource> MakeLocalPyResultSource(unique_ptr<QueryResult> result);

unique_ptr<DuckDBPyResultSource> MakeDistributedArrowPyResultSource(py::object table_iterator, vector<string> names,
                                                                    vector<LogicalType> types,
                                                                    const shared_ptr<ClientContext> &context);

} // namespace duckdb
