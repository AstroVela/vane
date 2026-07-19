// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "duckdb_python/pyresult_source.hpp"

#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_query_result.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/arrow/result_arrow_wrapper.hpp"
#include "duckdb/common/enums/stream_execution_result.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/stream_query_result.hpp"
#include "duckdb_python/pybind11/gil_wrapper.hpp"
#include "ray/safe_pyobject.hpp"

namespace duckdb {

using distributed::python::ray::SafePyObject;

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

static bool ResultPythonRuntimeUsable() {
	return distributed::python::ray::SafePyObjectCanDecRef();
}

//! Flattens the iterator of pyarrow.Table partitions into one Arrow C stream.
//! The external schema is derived from the relation rather than partition
//! field names (distributed partitions currently use positional c0/c1 names).
struct DistributedArrowStreamOwner {
	DistributedArrowStreamOwner(py::object iterator_p, vector<string> names_p, vector<LogicalType> types_p,
	                            const shared_ptr<ClientContext> &context_p, idx_t rows_per_batch_p)
	    : iterator(SafePyObject(py::iter(iterator_p))), names(std::move(names_p)), types(std::move(types_p)),
	      context(context_p), client_properties(context_p->GetClientProperties()), rows_per_batch(rows_per_batch_p) {
		stream.private_data = this;
		stream.get_schema = GetSchema;
		stream.get_next = GetNext;
		stream.release = Release;
		stream.get_last_error = GetLastError;
	}

	~DistributedArrowStreamOwner() {
		Close();
	}

	static int GetSchema(ArrowArrayStream *stream, ArrowSchema *out) {
		if (!stream || !stream->release) {
			return -1;
		}
		auto self = reinterpret_cast<DistributedArrowStreamOwner *>(stream->private_data);
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
		auto self = reinterpret_cast<DistributedArrowStreamOwner *>(stream->private_data);
		out->release = nullptr;
		try {
			return self->Next(out);
		} catch (py::error_already_set &ex) {
			self->last_error = ex.what();
			return -1;
		} catch (std::exception &ex) {
			self->last_error = ex.what();
			return -1;
		} catch (...) {
			self->last_error = "unknown distributed result stream error";
			return -1;
		}
	}

	static void Release(ArrowArrayStream *stream) {
		if (!stream || !stream->release) {
			return;
		}
		stream->release = nullptr;
		delete reinterpret_cast<DistributedArrowStreamOwner *>(stream->private_data);
	}

	static const char *GetLastError(ArrowArrayStream *stream) {
		if (!stream || !stream->release) {
			return "stream was released";
		}
		auto self = reinterpret_cast<DistributedArrowStreamOwner *>(stream->private_data);
		return self->last_error.c_str();
	}

	int Next(ArrowArray *out) {
		while (true) {
			if (current_stream.release) {
				PythonGILWrapper gil;
				if (current_stream.get_next(&current_stream, out)) {
					auto error = current_stream.get_last_error ? current_stream.get_last_error(&current_stream)
					                                           : "unknown Arrow stream error";
					throw InvalidInputException("Distributed result Arrow stream failed: %s", error);
				}
				if (out->release) {
					return 0;
				}
				current_stream.release(&current_stream);
			}

			if (exhausted) {
				out->release = nullptr;
				return 0;
			}
			OpenNextPartition();
		}
	}

	void OpenNextPartition() {
		PythonGILWrapper gil;
		auto iterator_obj = iterator.get();
		PyObject *next_ptr = PyIter_Next(iterator_obj.ptr());
		if (!next_ptr) {
			if (PyErr_Occurred()) {
				throw py::error_already_set();
			}
			exhausted = true;
			return;
		}

		auto table = NormalizeTable(py::reinterpret_steal<py::object>(next_ptr));
		py::object arrow_source = table;
		if (rows_per_batch > 0 && py::hasattr(table, "to_reader")) {
			arrow_source = table.attr("to_reader")(py::arg("max_chunksize") = rows_per_batch);
		}
		auto capsule_obj = arrow_source.attr("__arrow_c_stream__")();
		auto capsule = py::reinterpret_borrow<py::capsule>(capsule_obj);
		auto exported_stream = capsule.get_pointer<ArrowArrayStream>();
		if (!exported_stream || !exported_stream->release) {
			throw InvalidInputException("Distributed result returned a released Arrow stream");
		}
		current_stream = *exported_stream;
		exported_stream->release = nullptr;

		ArrowSchema actual_schema;
		actual_schema.release = nullptr;
		if (current_stream.get_schema(&current_stream, &actual_schema)) {
			auto error = current_stream.get_last_error ? current_stream.get_last_error(&current_stream)
			                                           : "unknown Arrow schema error";
			throw InvalidInputException("Failed to read distributed result schema: %s", error);
		}
		if (!actual_schema.release) {
			throw InvalidInputException("Distributed result returned a released Arrow schema");
		}
		try {
			ArrowTableSchema actual;
			ArrowTableFunction::PopulateArrowTableSchema(*context, actual, actual_schema);
			auto actual_types = actual.GetTypes();
			if (actual_types.size() != types.size()) {
				throw InvalidInputException("Distributed result partition %d has %d columns, expected %d",
				                            partition_index, actual_types.size(), types.size());
			}
			for (idx_t col_idx = 0; col_idx < types.size(); col_idx++) {
				if (actual_types[col_idx] != types[col_idx]) {
					throw InvalidInputException("Distributed result partition %d column %d has type %s, expected %s",
					                            partition_index, col_idx, actual_types[col_idx].ToString(),
					                            types[col_idx].ToString());
				}
			}
		} catch (...) {
			actual_schema.release(&actual_schema);
			throw;
		}
		actual_schema.release(&actual_schema);
		partition_index++;
	}

	py::object NormalizeTable(py::object table) {
		auto column_count = py::cast<idx_t>(table.attr("num_columns"));
		if (column_count != types.size()) {
			throw InvalidInputException("Distributed result partition %d has %d columns, expected %d", partition_index,
			                            column_count, types.size());
		}

		ArrowSchema expected_arrow_schema;
		expected_arrow_schema.release = nullptr;
		ArrowConverter::ToArrowSchema(&expected_arrow_schema, types, names, client_properties);
		py::object schema;
		try {
			auto pyarrow_lib = py::module_::import("pyarrow").attr("lib");
			schema = pyarrow_lib.attr("Schema").attr("_import_from_c")((uint64_t)&expected_arrow_schema); // NOLINT
		} catch (...) {
			if (expected_arrow_schema.release) {
				expected_arrow_schema.release(&expected_arrow_schema);
			}
			throw;
		}
		py::list columns;
		for (idx_t col_idx = 0; col_idx < types.size(); col_idx++) {
			auto column = table.attr("column")(col_idx);
			py::object expected_type = schema.attr("field")(col_idx).attr("type");
			if (!py::cast<bool>(column.attr("type").attr("equals")(expected_type))) {
				try {
					column = column.attr("cast")(expected_type, py::arg("safe") = true);
				} catch (py::error_already_set &ex) {
					throw InvalidInputException(
					    "Distributed result partition %d column %d cannot be safely cast to %s: %s", partition_index,
					    col_idx, types[col_idx].ToString(), ex.what());
				}
			}
			columns.append(column);
		}
		return py::module_::import("pyarrow").attr("Table").attr("from_arrays")(columns, py::arg("schema") = schema);
	}

	void Close() {
		if (closed) {
			return;
		}
		closed = true;
		if (current_stream.release) {
			if (ResultPythonRuntimeUsable()) {
				PythonGILWrapper gil;
				current_stream.release(&current_stream);
			} else {
				current_stream.release = nullptr;
			}
		}
		if (!iterator.empty() && ResultPythonRuntimeUsable()) {
			PythonGILWrapper gil;
			auto iterator_obj = iterator.get();
			try {
				if (py::hasattr(iterator_obj, "close")) {
					iterator_obj.attr("close")();
				}
			} catch (py::error_already_set &ex) {
				ex.discard_as_unraisable("closing distributed result stream");
			}
		}
		iterator.reset_with_gil();
	}

	ArrowArrayStream stream;
	SafePyObject iterator;
	vector<string> names;
	vector<LogicalType> types;
	shared_ptr<ClientContext> context;
	ClientProperties client_properties;
	idx_t rows_per_batch;
	ArrowArrayStream current_stream {};
	idx_t partition_index = 0;
	bool exhausted = false;
	bool closed = false;
	string last_error;
};

class DistributedArrowResultSource : public DuckDBPyResultSource {
public:
	DistributedArrowResultSource(py::object table_iterator, vector<string> names, vector<LogicalType> types,
	                             const shared_ptr<ClientContext> &context_p)
	    : iterator(SafePyObject(std::move(table_iterator))), context(context_p) {
		if (!context) {
			throw InternalException("DistributedArrowResultSource created without a context");
		}
		metadata.names = std::move(names);
		metadata.types = std::move(types);
		metadata.client_properties = context->GetClientProperties();
	}

	~DistributedArrowResultSource() override {
		Close();
	}

	const DuckDBPyResultMetadata &Metadata() const override {
		return metadata;
	}

	unique_ptr<DataChunk> FetchChunk(bool raw) override {
		(void)raw;
		EnsureStream(STANDARD_VECTOR_SIZE);
		if (closed) {
			return nullptr;
		}

		while (!current_scan ||
		       current_scan->chunk_offset >= NumericCast<idx_t>(current_scan->chunk->arrow_array.length)) {
			current_scan.reset();
			auto array = stream->GetNextChunk();
			if (!array || !array->arrow_array.release) {
				stream.reset();
				closed = true;
				return nullptr;
			}
			if (array->arrow_array.length == 0) {
				continue;
			}
			auto owned_array = make_uniq<ArrowArrayWrapper>();
			owned_array->arrow_array = array->arrow_array;
			array->arrow_array.release = nullptr;
			current_scan = make_uniq<ArrowScanLocalState>(std::move(owned_array), *context);
			for (idx_t col_idx = 0; col_idx < metadata.types.size(); col_idx++) {
				current_scan->column_ids.push_back(col_idx);
			}
		}

		auto remaining = NumericCast<idx_t>(current_scan->chunk->arrow_array.length) - current_scan->chunk_offset;
		auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining);
		auto output = make_uniq<DataChunk>();
		output->Initialize(*context, metadata.types, count);
		output->SetCardinality(count);
		ArrowTableFunction::ArrowToDuckDB(*current_scan, arrow_table.GetColumns(), *output, false);
		current_scan->chunk_offset += output->size();
		output->Verify();
		chunk_consumption_started = true;
		return output;
	}

	ArrowArrayStream TakeArrowStream(idx_t rows_per_batch) override {
		if (chunk_consumption_started || current_scan) {
			throw InvalidInputException("Cannot switch a partially consumed distributed row result to an Arrow stream");
		}
		EnsureStream(rows_per_batch);
		if (!stream) {
			throw InvalidInputException("result closed");
		}
		auto result = stream->arrow_array_stream;
		stream->arrow_array_stream.release = nullptr;
		stream.reset();
		closed = true;
		return result;
	}

	optional_idx KnownRowCount() const override {
		return optional_idx();
	}

	bool IsClosed() const override {
		return closed;
	}

	void Close() override {
		if (closed && iterator.empty() && !stream) {
			return;
		}
		current_scan.reset();
		stream.reset();
		if (!iterator.empty() && ResultPythonRuntimeUsable()) {
			PythonGILWrapper gil;
			auto iterator_obj = iterator.get();
			try {
				if (py::hasattr(iterator_obj, "close")) {
					iterator_obj.attr("close")();
				}
			} catch (py::error_already_set &ex) {
				ex.discard_as_unraisable("closing distributed result iterator");
			}
		}
		iterator.reset_with_gil();
		closed = true;
	}

private:
	void EnsureStream(idx_t rows_per_batch) {
		if (stream || closed) {
			return;
		}
		if (iterator.empty()) {
			throw InvalidInputException("result closed");
		}

		PythonGILWrapper gil;
		auto owner =
		    new DistributedArrowStreamOwner(iterator.get(), metadata.names, metadata.types, context, rows_per_batch);
		iterator.reset_with_gil();
		stream = make_uniq<ArrowArrayStreamWrapper>();
		stream->arrow_array_stream = owner->stream;

		ArrowSchemaWrapper schema;
		stream->GetSchema(schema);
		ArrowTableFunction::PopulateArrowTableSchema(*context, arrow_table, schema.arrow_schema);
	}

	DuckDBPyResultMetadata metadata;
	SafePyObject iterator;
	shared_ptr<ClientContext> context;
	unique_ptr<ArrowArrayStreamWrapper> stream;
	ArrowTableSchema arrow_table;
	unique_ptr<ArrowScanLocalState> current_scan;
	bool chunk_consumption_started = false;
	bool closed = false;
};

} // namespace

unique_ptr<DuckDBPyResultSource> MakeLocalPyResultSource(unique_ptr<QueryResult> result) {
	return make_uniq<LocalQueryResultSource>(std::move(result));
}

unique_ptr<DuckDBPyResultSource> MakeDistributedArrowPyResultSource(py::object table_iterator, vector<string> names,
                                                                    vector<LogicalType> types,
                                                                    const shared_ptr<ClientContext> &context) {
	return make_uniq<DistributedArrowResultSource>(std::move(table_iterator), std::move(names), std::move(types),
	                                               context);
}

} // namespace duckdb
