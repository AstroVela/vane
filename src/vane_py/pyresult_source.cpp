// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

#include "vane_python/pyresult_source.hpp"

#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_query_result.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/arrow/result_arrow_wrapper.hpp"
#include "duckdb/common/enums/stream_execution_result.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/stream_query_result.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"
#include "ray/safe_pyobject.hpp"

#include <initializer_list>

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

//! Arrow transports the FILE family using canonical STRUCT storage. Restore a
//! Vane-owned FILE alias only when relation metadata declares its exact type.
static bool IsFileStorageType(const LogicalType &type) {
	if (type.id() != LogicalTypeId::STRUCT || type.HasAlias() ||
	    StructType::GetChildCount(type) != FileLogicalType::FIELD_COUNT) {
		return false;
	}
	auto file_type = FileLogicalType::Create();
	for (idx_t index = 0; index < FileLogicalType::FIELD_COUNT; index++) {
		if (StructType::GetChildName(type, index) != StructType::GetChildName(file_type, index) ||
		    StructType::GetChildType(type, index) != StructType::GetChildType(file_type, index)) {
			return false;
		}
	}
	return true;
}

static bool DistributedResultTypeMatches(const LogicalType &actual, const LogicalType &expected) {
	if (actual == expected) {
		return true;
	}
	if (FileLogicalType::IsFile(expected)) {
		return IsFileStorageType(actual);
	}
	if (TensorType::IsTensor(expected)) {
		return TensorType::IsTensor(actual) && TensorType::GetShape(actual) == TensorType::GetShape(expected) &&
		       DistributedResultTypeMatches(TensorType::GetChildType(actual), TensorType::GetChildType(expected));
	}
	if (actual.HasAlias() || expected.HasAlias() || actual.id() != expected.id()) {
		return false;
	}

	switch (expected.id()) {
	case LogicalTypeId::LIST:
		return DistributedResultTypeMatches(ListType::GetChildType(actual), ListType::GetChildType(expected));
	case LogicalTypeId::ARRAY:
		return ArrayType::GetSize(actual) == ArrayType::GetSize(expected) &&
		       DistributedResultTypeMatches(ArrayType::GetChildType(actual), ArrayType::GetChildType(expected));
	case LogicalTypeId::MAP:
		return DistributedResultTypeMatches(MapType::KeyType(actual), MapType::KeyType(expected)) &&
		       DistributedResultTypeMatches(MapType::ValueType(actual), MapType::ValueType(expected));
	case LogicalTypeId::STRUCT: {
		if (StructType::GetChildCount(actual) != StructType::GetChildCount(expected)) {
			return false;
		}
		for (idx_t index = 0; index < StructType::GetChildCount(expected); index++) {
			if (StructType::GetChildName(actual, index) != StructType::GetChildName(expected, index) ||
			    !DistributedResultTypeMatches(StructType::GetChildType(actual, index),
			                                  StructType::GetChildType(expected, index))) {
				return false;
			}
		}
		return true;
	}
	case LogicalTypeId::UNION: {
		if (UnionType::GetMemberCount(actual) != UnionType::GetMemberCount(expected)) {
			return false;
		}
		for (idx_t index = 0; index < UnionType::GetMemberCount(expected); index++) {
			if (UnionType::GetMemberName(actual, index) != UnionType::GetMemberName(expected, index) ||
			    !DistributedResultTypeMatches(UnionType::GetMemberType(actual, index),
			                                  UnionType::GetMemberType(expected, index))) {
				return false;
			}
		}
		return true;
	}
	default:
		return false;
	}
}

//! Flattens the iterator of pyarrow.Table partitions into one Arrow C stream.
//! The external schema is derived from the relation rather than partition
//! field names (distributed partitions currently use positional c0/c1 names).
struct DistributedArrowStreamOwner {
	DistributedArrowStreamOwner(py::object iterator_p, py::object prefetched_partition_p, bool has_prefetched_partition,
	                            bool iterator_exhausted, vector<string> names_p, vector<LogicalType> types_p,
	                            const shared_ptr<ClientContext> &context_p, idx_t rows_per_batch_p)
	    : iterator(SafePyObject(py::iter(iterator_p))), names(std::move(names_p)), types(std::move(types_p)),
	      context(context_p), client_properties(context_p->GetClientProperties()), rows_per_batch(rows_per_batch_p),
	      exhausted(iterator_exhausted) {
		if (has_prefetched_partition) {
			prefetched_partition = SafePyObject(std::move(prefetched_partition_p));
		}
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
		if (self->failed) {
			return -1;
		}
		try {
			ArrowConverter::ToArrowSchema(out, self->types, self->names, self->client_properties);
			return 0;
		} catch (std::exception &ex) {
			self->Fail(ex.what());
			return -1;
		}
	}

	static int GetNext(ArrowArrayStream *stream, ArrowArray *out) {
		if (!stream || !stream->release) {
			return -1;
		}
		auto self = reinterpret_cast<DistributedArrowStreamOwner *>(stream->private_data);
		out->release = nullptr;
		if (self->failed) {
			return -1;
		}
		// DuckDB can invoke this callback after releasing the GIL. Keep it held
		// through the catch blocks so Python exceptions are also formatted and
		// destroyed while the Python runtime is protected.
		PythonGILWrapper gil;
		try {
			return self->Next(out);
		} catch (py::error_already_set &ex) {
			self->Fail(ex.what());
			return -1;
		} catch (std::exception &ex) {
			self->Fail(ex.what());
			return -1;
		} catch (...) {
			self->Fail("unknown distributed result stream error");
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

	static string StreamErrorOr(ArrowArrayStream &stream, const char *fallback) {
		if (!stream.get_last_error) {
			return fallback;
		}
		const char *error = stream.get_last_error(&stream);
		return error && error[0] != '\0' ? error : fallback;
	}

	int Next(ArrowArray *out) {
		while (true) {
			if (current_stream.release) {
				PythonGILWrapper gil;
				if (current_stream.get_next(&current_stream, out)) {
					auto error = StreamErrorOr(current_stream, "unknown Arrow stream error");
					if (out->release) {
						out->release(out);
					}
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
		py::object table;
		if (!prefetched_partition.empty()) {
			table = prefetched_partition.get();
			prefetched_partition.reset_with_gil();
		} else {
			auto iterator_obj = iterator.get();
			PyObject *next_ptr = PyIter_Next(iterator_obj.ptr());
			if (!next_ptr) {
				if (PyErr_Occurred()) {
					throw py::error_already_set();
				}
				exhausted = true;
				return;
			}
			table = py::reinterpret_steal<py::object>(next_ptr);
		}

		table = NormalizeTable(std::move(table));
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
		if (!exported_stream->get_schema || !exported_stream->get_next) {
			throw InvalidInputException("Distributed result returned an Arrow stream without required callbacks");
		}
		current_stream = *exported_stream;
		exported_stream->release = nullptr;

		ArrowSchema actual_schema;
		actual_schema.release = nullptr;
		if (current_stream.get_schema(&current_stream, &actual_schema)) {
			auto error = StreamErrorOr(current_stream, "unknown Arrow schema error");
			if (actual_schema.release) {
				actual_schema.release(&actual_schema);
			}
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
				if (!DistributedResultTypeMatches(actual_types[col_idx], types[col_idx])) {
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

	static bool ArrowTypeMatchesAny(const py::object &type_predicates, const py::object &type,
	                                std::initializer_list<const char *> predicate_names) {
		for (const auto *predicate_name : predicate_names) {
			if (py::hasattr(type_predicates, predicate_name) &&
			    py::cast<bool>(type_predicates.attr(predicate_name)(type))) {
				return true;
			}
		}
		return false;
	}

	static bool CanNormalizeArrowExtensionStorage(const py::object &actual_type, const py::object &expected_type,
	                                              const py::object &type_predicates) {
		if (!py::hasattr(actual_type, "extension_name") || !py::hasattr(expected_type, "extension_name") ||
		    !py::hasattr(actual_type, "storage_type") || !py::hasattr(expected_type, "storage_type")) {
			return false;
		}
		auto extension_name = py::cast<string>(actual_type.attr("extension_name"));
		if (extension_name != py::cast<string>(expected_type.attr("extension_name"))) {
			return false;
		}
		if (extension_name == "arrow.opaque") {
			if (!py::hasattr(actual_type, "type_name") || !py::hasattr(expected_type, "type_name") ||
			    !py::hasattr(actual_type, "vendor_name") || !py::hasattr(expected_type, "vendor_name") ||
			    py::cast<string>(actual_type.attr("type_name")) != py::cast<string>(expected_type.attr("type_name")) ||
			    py::cast<string>(actual_type.attr("vendor_name")) !=
			        py::cast<string>(expected_type.attr("vendor_name"))) {
				return false;
			}
		} else if (extension_name == "arrow.fixed_shape_tensor") {
			for (const auto *attribute : {"shape", "permutation", "dim_names"}) {
				if (!py::hasattr(actual_type, attribute) || !py::hasattr(expected_type, attribute) ||
				    !py::cast<bool>(actual_type.attr(attribute).attr("__eq__")(expected_type.attr(attribute)))) {
					return false;
				}
			}
		} else if (extension_name != "arrow.json") {
			return false;
		}
		return CanNormalizeArrowType(actual_type.attr("storage_type"), expected_type.attr("storage_type"),
		                             type_predicates);
	}

	static bool CanNormalizeArrowType(const py::object &actual_type, const py::object &expected_type,
	                                  const py::object &type_predicates) {
		if (py::cast<bool>(actual_type.attr("equals")(expected_type))) {
			return true;
		}
		if (CanNormalizeArrowExtensionStorage(actual_type, expected_type, type_predicates)) {
			return true;
		}
		auto same_family = [&](std::initializer_list<const char *> predicate_names) {
			return ArrowTypeMatchesAny(type_predicates, actual_type, predicate_names) &&
			       ArrowTypeMatchesAny(type_predicates, expected_type, predicate_names);
		};
		if (same_family({"is_string", "is_large_string", "is_string_view"}) ||
		    same_family({"is_binary", "is_large_binary", "is_binary_view"})) {
			return true;
		}
		if (same_family({"is_timestamp"})) {
			auto actual_timezone = actual_type.attr("tz");
			auto expected_timezone = expected_type.attr("tz");
			return !py::none().is(actual_timezone) && !py::none().is(expected_timezone) &&
			       !py::cast<string>(actual_timezone).empty() && !py::cast<string>(expected_timezone).empty() &&
			       py::cast<string>(actual_type.attr("unit")) == py::cast<string>(expected_type.attr("unit"));
		}
		if (same_family({"is_list", "is_large_list", "is_list_view", "is_large_list_view"})) {
			return CanNormalizeArrowType(actual_type.attr("value_type"), expected_type.attr("value_type"),
			                             type_predicates);
		}
		if (same_family({"is_fixed_size_list"})) {
			return py::cast<idx_t>(actual_type.attr("list_size")) == py::cast<idx_t>(expected_type.attr("list_size")) &&
			       CanNormalizeArrowType(actual_type.attr("value_type"), expected_type.attr("value_type"),
			                             type_predicates);
		}
		if (same_family({"is_struct"})) {
			auto field_count = py::cast<idx_t>(actual_type.attr("num_fields"));
			if (field_count != py::cast<idx_t>(expected_type.attr("num_fields"))) {
				return false;
			}
			for (idx_t field_idx = 0; field_idx < field_count; field_idx++) {
				auto actual_field = actual_type.attr("field")(field_idx);
				auto expected_field = expected_type.attr("field")(field_idx);
				if (py::cast<string>(actual_field.attr("name")) != py::cast<string>(expected_field.attr("name")) ||
				    py::cast<bool>(actual_field.attr("nullable")) != py::cast<bool>(expected_field.attr("nullable")) ||
				    !CanNormalizeArrowType(actual_field.attr("type"), expected_field.attr("type"), type_predicates)) {
					return false;
				}
			}
			return true;
		}
		if (same_family({"is_union"})) {
			if (py::cast<string>(actual_type.attr("mode")) != "sparse" ||
			    py::cast<string>(expected_type.attr("mode")) != "sparse" ||
			    py::cast<vector<int64_t>>(actual_type.attr("type_codes")) !=
			        py::cast<vector<int64_t>>(expected_type.attr("type_codes"))) {
				return false;
			}
			auto field_count = py::cast<idx_t>(actual_type.attr("num_fields"));
			if (field_count != py::cast<idx_t>(expected_type.attr("num_fields"))) {
				return false;
			}
			for (idx_t field_idx = 0; field_idx < field_count; field_idx++) {
				auto actual_field = actual_type.attr("field")(field_idx);
				auto expected_field = expected_type.attr("field")(field_idx);
				if (py::cast<string>(actual_field.attr("name")) != py::cast<string>(expected_field.attr("name")) ||
				    py::cast<bool>(actual_field.attr("nullable")) != py::cast<bool>(expected_field.attr("nullable")) ||
				    !CanNormalizeArrowType(actual_field.attr("type"), expected_field.attr("type"), type_predicates)) {
					return false;
				}
			}
			return true;
		}
		if (same_family({"is_map"})) {
			return py::cast<bool>(actual_type.attr("keys_sorted")) ==
			           py::cast<bool>(expected_type.attr("keys_sorted")) &&
			       CanNormalizeArrowType(actual_type.attr("key_type"), expected_type.attr("key_type"),
			                             type_predicates) &&
			       CanNormalizeArrowType(actual_type.attr("item_type"), expected_type.attr("item_type"),
			                             type_predicates);
		}
		return false;
	}

	static py::object NormalizeArrowArray(const py::object &array, const py::object &expected_type,
	                                      const py::object &pyarrow, const py::object &type_predicates) {
		auto actual_type = array.attr("type");
		if (py::cast<bool>(actual_type.attr("equals")(expected_type))) {
			return array;
		}
		if (CanNormalizeArrowExtensionStorage(actual_type, expected_type, type_predicates)) {
			auto storage = NormalizeArrowArray(array.attr("storage"), expected_type.attr("storage_type"), pyarrow,
			                                   type_predicates);
			return pyarrow.attr("ExtensionArray").attr("from_storage")(expected_type, storage);
		}
		auto same_family = [&](std::initializer_list<const char *> predicate_names) {
			return ArrowTypeMatchesAny(type_predicates, actual_type, predicate_names) &&
			       ArrowTypeMatchesAny(type_predicates, expected_type, predicate_names);
		};
		if (same_family({"is_string", "is_large_string", "is_string_view"}) ||
		    same_family({"is_binary", "is_large_binary", "is_binary_view"})) {
			return array.attr("cast")(expected_type, py::arg("safe") = true);
		}
		if (same_family({"is_timestamp"})) {
			return array.attr("cast")(expected_type, py::arg("safe") = true);
		}
		if (same_family({"is_list", "is_large_list", "is_list_view", "is_large_list_view"})) {
			auto values = NormalizeArrowArray(array.attr("flatten")(), expected_type.attr("value_type"), pyarrow,
			                                  type_predicates);
			auto large_offsets =
			    ArrowTypeMatchesAny(type_predicates, expected_type, {"is_large_list", "is_large_list_view"});
			auto int64_type = pyarrow.attr("int64")();
			auto offset_type = large_offsets ? int64_type : pyarrow.attr("int32")();
			auto sizes =
			    array.attr("value_lengths")().attr("fill_null")(0).attr("cast")(int64_type, py::arg("safe") = true);
			auto compute = py::module_::import("pyarrow.compute");
			py::list offset_arrays;
			offset_arrays.append(pyarrow.attr("array")(py::make_tuple(0), py::arg("type") = int64_type));
			offset_arrays.append(compute.attr("cumulative_sum")(sizes));
			auto offsets = pyarrow.attr("concat_arrays")(offset_arrays);
			if (!large_offsets) {
				sizes = sizes.attr("cast")(offset_type, py::arg("safe") = true);
				offsets = offsets.attr("cast")(offset_type, py::arg("safe") = true);
			}
			auto mask = array.attr("is_null")();
			if (ArrowTypeMatchesAny(type_predicates, expected_type, {"is_list"})) {
				return pyarrow.attr("ListArray")
				    .attr("from_arrays")(offsets, values, py::arg("type") = expected_type, py::arg("mask") = mask);
			}
			if (ArrowTypeMatchesAny(type_predicates, expected_type, {"is_large_list"})) {
				return pyarrow.attr("LargeListArray")
				    .attr("from_arrays")(offsets, values, py::arg("type") = expected_type, py::arg("mask") = mask);
			}
			auto view_offsets = offsets.attr("slice")(0, py::len(array));
			if (ArrowTypeMatchesAny(type_predicates, expected_type, {"is_list_view"})) {
				return pyarrow.attr("ListViewArray")
				    .attr("from_arrays")(view_offsets, sizes, values, py::arg("type") = expected_type,
				                         py::arg("mask") = mask);
			}
			return pyarrow.attr("LargeListViewArray")
			    .attr("from_arrays")(view_offsets, sizes, values, py::arg("type") = expected_type,
			                         py::arg("mask") = mask);
		}
		if (same_family({"is_fixed_size_list"})) {
			auto list_size = py::cast<idx_t>(actual_type.attr("list_size"));
			auto array_offset = py::cast<idx_t>(array.attr("offset"));
			auto array_length = NumericCast<idx_t>(py::len(array));
			auto values = array.attr("values").attr("slice")(array_offset * list_size, array_length * list_size);
			values = NormalizeArrowArray(values, expected_type.attr("value_type"), pyarrow, type_predicates);
			return pyarrow.attr("FixedSizeListArray")
			    .attr("from_arrays")(values, py::arg("type") = expected_type,
			                         py::arg("mask") = array.attr("is_null")());
		}
		if (same_family({"is_struct"})) {
			py::list fields;
			auto field_count = py::cast<idx_t>(expected_type.attr("num_fields"));
			for (idx_t field_idx = 0; field_idx < field_count; field_idx++) {
				fields.append(NormalizeArrowArray(array.attr("field")(field_idx),
				                                  expected_type.attr("field")(field_idx).attr("type"), pyarrow,
				                                  type_predicates));
			}
			return pyarrow.attr("StructArray")
			    .attr("from_arrays")(fields, py::arg("type") = expected_type,
			                         py::arg("mask") = array.attr("is_null")());
		}
		if (same_family({"is_union"})) {
			py::list children;
			auto field_count = py::cast<idx_t>(expected_type.attr("num_fields"));
			for (idx_t field_idx = 0; field_idx < field_count; field_idx++) {
				children.append(NormalizeArrowArray(array.attr("field")(field_idx),
				                                    expected_type.attr("field")(field_idx).attr("type"), pyarrow,
				                                    type_predicates));
			}

			py::list type_id_buffers;
			type_id_buffers.append(py::none());
			type_id_buffers.append(array.attr("buffers")().attr("__getitem__")(1));
			auto type_ids = pyarrow.attr("Array").attr("from_buffers")(
			    pyarrow.attr("int8")(), py::len(array), type_id_buffers, py::arg("offset") = array.attr("offset"));
			py::list type_id_chunks;
			type_id_chunks.append(type_ids);
			type_ids = pyarrow.attr("concat_arrays")(type_id_chunks);

			py::list union_buffers;
			union_buffers.append(py::none());
			union_buffers.append(type_ids.attr("buffers")().attr("__getitem__")(1));
			return pyarrow.attr("Array").attr("from_buffers")(expected_type, py::len(array), union_buffers,
			                                                  py::arg("children") = children);
		}
		if (same_family({"is_map"})) {
			auto offsets = array.attr("offsets");
			auto start = py::cast<idx_t>(offsets.attr("__getitem__")(0).attr("as_py")());
			auto end = py::cast<idx_t>(offsets.attr("__getitem__")(-1).attr("as_py")());
			auto entry_count = end - start;
			auto compute = py::module_::import("pyarrow.compute");
			auto normalized_offsets =
			    compute.attr("subtract")(offsets, start).attr("cast")(pyarrow.attr("int32")(), py::arg("safe") = true);
			auto keys = array.attr("keys").attr("slice")(start, entry_count);
			keys = NormalizeArrowArray(keys, expected_type.attr("key_type"), pyarrow, type_predicates);
			auto items = array.attr("items").attr("slice")(start, entry_count);
			items = NormalizeArrowArray(items, expected_type.attr("item_type"), pyarrow, type_predicates);
			return pyarrow.attr("MapArray")
			    .attr("from_arrays")(normalized_offsets, keys, items, py::arg("type") = expected_type,
			                         py::arg("mask") = array.attr("is_null")());
		}
		throw InternalException("Unsupported Arrow normalization from %s to %s", py::cast<string>(py::str(actual_type)),
		                        py::cast<string>(py::str(expected_type)));
	}

	static py::object NormalizeArrowColumn(const py::object &column, const py::object &expected_type,
	                                       const py::object &pyarrow, const py::object &type_predicates) {
		py::list normalized_chunks;
		for (auto chunk_handle : column.attr("chunks")) {
			auto chunk = py::reinterpret_borrow<py::object>(chunk_handle);
			normalized_chunks.append(NormalizeArrowArray(chunk, expected_type, pyarrow, type_predicates));
		}
		return pyarrow.attr("chunked_array")(normalized_chunks, py::arg("type") = expected_type);
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
		auto pyarrow = py::module_::import("pyarrow");
		auto type_predicates = pyarrow.attr("types");
		for (idx_t col_idx = 0; col_idx < types.size(); col_idx++) {
			auto column = table.attr("column")(col_idx);
			py::object expected_type = schema.attr("field")(col_idx).attr("type");
			auto actual_type = column.attr("type");
			if (!py::cast<bool>(actual_type.attr("equals")(expected_type))) {
				if (!CanNormalizeArrowType(actual_type, expected_type, type_predicates)) {
					throw InvalidInputException(
					    "Distributed result partition %d column %d has Arrow type %s, expected %s for DuckDB type %s",
					    partition_index, col_idx, py::cast<string>(py::str(actual_type)),
					    py::cast<string>(py::str(expected_type)), types[col_idx].ToString());
				}
				try {
					column = NormalizeArrowColumn(column, expected_type, pyarrow, type_predicates);
				} catch (py::error_already_set &ex) {
					throw InvalidInputException("Distributed result partition %d column %d failed Arrow type "
					                            "normalization from %s to %s: %s",
					                            partition_index, col_idx, py::cast<string>(py::str(actual_type)),
					                            py::cast<string>(py::str(expected_type)), ex.what());
				}
			}
			columns.append(column);
		}
		return pyarrow.attr("Table").attr("from_arrays")(columns, py::arg("schema") = schema);
	}

	void Fail(string error) {
		if (!failed) {
			last_error = std::move(error);
			failed = true;
		}
		Close();
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
		prefetched_partition.reset_with_gil();
		iterator.reset_with_gil();
	}

	ArrowArrayStream stream;
	SafePyObject iterator;
	SafePyObject prefetched_partition;
	vector<string> names;
	vector<LogicalType> types;
	shared_ptr<ClientContext> context;
	ClientProperties client_properties;
	idx_t rows_per_batch;
	ArrowArrayStream current_stream {};
	idx_t partition_index = 0;
	bool exhausted = false;
	bool closed = false;
	bool failed = false;
	string last_error;
};

class DistributedArrowResultSource : public DuckDBPyResultSource {
public:
	DistributedArrowResultSource(py::object table_iterator, py::object prefetched_partition,
	                             bool has_prefetched_partition_p, bool iterator_exhausted_p, vector<string> names,
	                             vector<LogicalType> types, const shared_ptr<ClientContext> &context_p)
	    : iterator(SafePyObject(std::move(table_iterator))), has_prefetched_partition(has_prefetched_partition_p),
	      iterator_exhausted(iterator_exhausted_p), context(context_p) {
		if (!context) {
			throw InternalException("DistributedArrowResultSource created without a context");
		}
		if (has_prefetched_partition) {
			this->prefetched_partition = SafePyObject(std::move(prefetched_partition));
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
		if (closed && iterator.empty() && prefetched_partition.empty() && !stream) {
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
		prefetched_partition.reset_with_gil();
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
		auto new_stream = make_uniq<ArrowArrayStreamWrapper>();
		auto owner = make_uniq<DistributedArrowStreamOwner>(iterator.get(), prefetched_partition.get(),
		                                                    has_prefetched_partition, iterator_exhausted,
		                                                    metadata.names, metadata.types, context, rows_per_batch);
		new_stream->arrow_array_stream = owner->stream;
		owner.release(); // The ArrowArrayStream release callback now owns the stream owner.
		prefetched_partition.reset_with_gil();
		iterator.reset_with_gil();

		ArrowSchemaWrapper schema;
		new_stream->GetSchema(schema);
		ArrowTableSchema new_arrow_table;
		ArrowTableFunction::PopulateArrowTableSchema(*context, new_arrow_table, schema.arrow_schema);

		stream = std::move(new_stream);
		arrow_table = std::move(new_arrow_table);
	}

	DuckDBPyResultMetadata metadata;
	SafePyObject iterator;
	SafePyObject prefetched_partition;
	bool has_prefetched_partition;
	bool iterator_exhausted;
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

unique_ptr<DuckDBPyResultSource>
MakeDistributedArrowPyResultSource(py::object table_iterator, py::object prefetched_partition,
                                   bool has_prefetched_partition, bool iterator_exhausted, vector<string> names,
                                   vector<LogicalType> types, const shared_ptr<ClientContext> &context) {
	return make_uniq<DistributedArrowResultSource>(std::move(table_iterator), std::move(prefetched_partition),
	                                               has_prefetched_partition, iterator_exhausted, std::move(names),
	                                               std::move(types), context);
}

} // namespace duckdb
