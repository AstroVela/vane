// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "vane_python/pyresult_source.hpp"

#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_query_result.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/arrow/result_arrow_wrapper.hpp"
#include "duckdb/common/enums/stream_execution_result.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/stream_query_result.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/pytype.hpp"

#include <initializer_list>

namespace duckdb {

namespace {

class LocalQueryResultSource : public DuckDBPyResultSource {
public:
	explicit LocalQueryResultSource(unique_ptr<QueryResult> result_p) : result(std::move(result_p)) {
		if (!result) {
			throw InternalException("LocalQueryResultSource created without a result");
		}
		metadata.names = result->names;
		metadata.types = result->types;
		metadata.client_properties = result->client_properties;
	}

	const DuckDBPyResultMetadata &Metadata() const override {
		return metadata;
	}

	unique_ptr<DataChunk> FetchChunk(bool raw) override {
		if (!result) {
			throw InvalidInputException("result closed");
		}
		if (closed) {
			return nullptr;
		}
		if (!closed && result->type == QueryResultType::STREAM_RESULT && !result->Cast<StreamQueryResult>().IsOpen()) {
			closed = true;
			return nullptr;
		}

		if (!raw && result->type == QueryResultType::STREAM_RESULT) {
			auto &stream_result = result->Cast<StreamQueryResult>();
			StreamExecutionResult execution_result;
			while (!StreamQueryResult::IsChunkReady(execution_result = stream_result.ExecuteTask())) {
				{
					PythonGILWrapper gil;
					if (PyErr_CheckSignals() != 0) {
						throw std::runtime_error("Query interrupted");
					}
				}
				if (execution_result == StreamExecutionResult::BLOCKED) {
					stream_result.WaitForTask();
				}
			}
			if (execution_result == StreamExecutionResult::EXECUTION_CANCELLED) {
				throw InvalidInputException("The execution of the query was cancelled before it could finish, likely "
				                            "caused by executing a different query");
			}
			if (execution_result == StreamExecutionResult::EXECUTION_ERROR) {
				stream_result.ThrowError();
			}
		}

		auto chunk = raw ? result->FetchRaw() : result->Fetch();
		if (result->HasError()) {
			result->ThrowError();
		}
		if (!chunk || chunk->size() == 0) {
			closed = true;
		}
		return chunk;
	}

	ArrowArrayStream TakeArrowStream(idx_t rows_per_batch) override;

	bool RequiresConnectionLock() const override {
		return result && result->type == QueryResultType::STREAM_RESULT;
	}

	optional_idx KnownRowCount() const override {
		if (result && result->type == QueryResultType::MATERIALIZED_RESULT) {
			return result->Cast<MaterializedQueryResult>().RowCount();
		}
		return optional_idx();
	}

	bool IsClosed() const override {
		return closed || !result;
	}

	void Close() override {
		if (result && result->type == QueryResultType::STREAM_RESULT) {
			result->Cast<StreamQueryResult>().Close();
		}
		result.reset();
		closed = true;
	}

private:
	DuckDBPyResultMetadata metadata;
	unique_ptr<QueryResult> result;
	bool closed = false;
};

//! ArrowQueryResult stores already-exported arrays and cannot be consumed via
//! QueryResult::Fetch(). This owner exposes those arrays as an Arrow C stream.
struct ArrowQueryResultStreamOwner {
	explicit ArrowQueryResultStreamOwner(unique_ptr<QueryResult> result_p) : result(std::move(result_p)) {
		auto &arrow_result = result->Cast<ArrowQueryResult>();
		arrays = arrow_result.ConsumeArrays();
		types = result->types;
		names = result->names;
		client_properties = result->client_properties;

		stream.private_data = this;
		stream.get_schema = GetSchema;
		stream.get_next = GetNext;
		stream.release = Release;
		stream.get_last_error = GetLastError;
	}

	static int GetSchema(ArrowArrayStream *stream, ArrowSchema *out) {
		if (!stream || !stream->release) {
			return -1;
		}
		auto self = reinterpret_cast<ArrowQueryResultStreamOwner *>(stream->private_data);
		out->release = nullptr;
		try {
			ArrowConverter::ToArrowSchema(out, self->types, self->names, self->client_properties);
			return 0;
		} catch (std::exception &ex) {
			self->last_error = ex.what();
			return -1;
		}
	}

	static int GetNext(ArrowArrayStream *stream, ArrowArray *out) {
		if (!stream || !stream->release) {
			return -1;
		}
		auto self = reinterpret_cast<ArrowQueryResultStreamOwner *>(stream->private_data);
		if (self->index >= self->arrays.size()) {
			out->release = nullptr;
			return 0;
		}
		*out = self->arrays[self->index]->arrow_array;
		self->arrays[self->index]->arrow_array.release = nullptr;
		self->index++;
		return 0;
	}

	static void Release(ArrowArrayStream *stream) {
		if (!stream || !stream->release) {
			return;
		}
		stream->release = nullptr;
		delete reinterpret_cast<ArrowQueryResultStreamOwner *>(stream->private_data);
	}

	static const char *GetLastError(ArrowArrayStream *stream) {
		if (!stream || !stream->release) {
			return "stream was released";
		}
		auto self = reinterpret_cast<ArrowQueryResultStreamOwner *>(stream->private_data);
		return self->last_error.c_str();
	}

	ArrowArrayStream stream;
	unique_ptr<QueryResult> result;
	vector<unique_ptr<ArrowArrayWrapper>> arrays;
	vector<LogicalType> types;
	vector<string> names;
	ClientProperties client_properties;
	idx_t index = 0;
	string last_error;
};

ArrowArrayStream LocalQueryResultSource::TakeArrowStream(idx_t rows_per_batch) {
	if (!result) {
		throw InvalidInputException("result closed");
	}
	closed = true;
	if (result->type == QueryResultType::ARROW_RESULT) {
		auto owner = new ArrowQueryResultStreamOwner(std::move(result));
		return owner->stream;
	}
	auto owner = new ResultArrowArrayStreamWrapper(std::move(result), rows_per_batch);
	return owner->stream;
}

} // namespace

unique_ptr<DuckDBPyResultSource> MakeLocalPyResultSource(unique_ptr<QueryResult> result) {
	return make_uniq<LocalQueryResultSource>(std::move(result));
}

} // namespace duckdb
