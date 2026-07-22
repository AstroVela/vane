// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_py/udf_executor.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb_python/udf_executor.hpp"

#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_appender.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/arrow_aux_data.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/udf_executor.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/parallel/interrupt.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb_python/arrow/arrow_export_utils.hpp"
#include "duckdb_python/pybind11/gil_wrapper.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb_python/pybind11/registered_py_object.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/python_objects.hpp"

#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>

#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

namespace duckdb {

namespace {

static bool PythonRuntimeUsableForUDFTeardown() {
	if (!Py_IsInitialized()) {
		return false;
	}
	if (PythonIsFinalizing()) {
		return false;
	}
	return true;
}

static bool UDFDebugEnabled() {
	static const bool enabled = [] {
		const char *value = std::getenv("DUCKDB_DISTRIBUTED_DEBUG");
		if (!value || !*value) {
			return false;
		}
		auto normalized = StringUtil::Lower(string(value));
		if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") {
			return false;
		}
		if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on") {
			return true;
		}
		throw InvalidInputException("DUCKDB_DISTRIBUTED_DEBUG must be a boolean value");
	}();
	return enabled;
}

static void UDFDebugLog(const string &message) {
	if (!UDFDebugEnabled()) {
		return;
	}
	std::cerr << "[vane-udf-executor pid=" << getpid() << " tid=" << std::this_thread::get_id() << "] " << message
	          << std::endl;
}

// Dispatcher-delivered callbacks can synchronously destroy a query and call
// Unregister on this same thread. Such a call must be retired by a later loop
// iteration; waiting here would make the dispatcher join itself.
static thread_local bool g_on_udf_dispatcher_thread = false;

static std::chrono::milliseconds UDFUnregisterTimeout() {
	constexpr long DEFAULT_TIMEOUT_MS = 5000;
	const char *value = std::getenv("VANE_UDF_UNREGISTER_TIMEOUT_MS");
	if (!value || !*value) {
		return std::chrono::milliseconds(DEFAULT_TIMEOUT_MS);
	}
	size_t parsed_characters = 0;
	long long parsed = 0;
	try {
		parsed = std::stoll(value, &parsed_characters, 10);
	} catch (const std::exception &ex) {
		throw InvalidInputException("VANE_UDF_UNREGISTER_TIMEOUT_MS must be a positive integer: %s", ex.what());
	}
	if (parsed_characters != std::strlen(value) || parsed <= 0) {
		throw InvalidInputException("VANE_UDF_UNREGISTER_TIMEOUT_MS must be a positive integer");
	}
	return std::chrono::milliseconds(parsed);
}

static const char *UDFOutputEventKindName(UDFOutputEventKind kind) {
	switch (kind) {
	case UDFOutputEventKind::DATA:
		return "DATA";
	case UDFOutputEventKind::COMPLETE:
		return "COMPLETE";
	case UDFOutputEventKind::ERROR:
		return "ERROR";
	case UDFOutputEventKind::FINISHED:
		return "FINISHED";
	default:
		return "UNKNOWN";
	}
}

static void ReleaseOutputLeaseCallback(std::function<void()> &callback) {
	if (!callback) {
		return;
	}
	auto release = std::move(callback);
	callback = nullptr;
	release();
}

static atomic<uint64_t> g_udf_debug_drain_tick {0};
static atomic<uint64_t> g_udf_debug_dispatcher_tick {0};
static atomic<uint64_t> g_udf_distributed_ref_bundle_data_events {0};
static atomic<uint64_t> g_udf_distributed_direct_table_rejected_events {0};
static atomic<uint64_t> g_udf_direct_arrow_table_conversion_count {0};
static atomic<uint64_t> g_udf_direct_output_arrow_table_conversion_count {0};
static atomic<uint64_t> g_udf_python_export_under_client_context_lock_count {0};

// Global mutex protecting ClientContext access from multiple threads.
// DuckDB's ClientContext is NOT thread-safe.  With VANE_REPARTITION_COUNT>1,
// multiple pipeline threads call Submit() (which reads from ClientContext),
// and the dispatcher thread calls ConvertArrowTableToDataChunk() (heavy
// ClientContext use).  Without this mutex, concurrent access corrupts the
// heap (manifests as SIGABRT in AllocatedData::Reset → cfree).
static mutex g_client_context_mutex;
static thread_local idx_t g_client_context_mutex_depth = 0;

static bool ClientContextMutexHeldByCurrentThread() {
	return g_client_context_mutex_depth > 0;
}

class ScopedClientContextLock {
public:
	ScopedClientContextLock() : lock(g_client_context_mutex) {
		g_client_context_mutex_depth++;
	}

	ScopedClientContextLock(const ScopedClientContextLock &) = delete;
	ScopedClientContextLock &operator=(const ScopedClientContextLock &) = delete;

	~ScopedClientContextLock() {
		D_ASSERT(g_client_context_mutex_depth > 0);
		g_client_context_mutex_depth--;
		lock.unlock();
	}

private:
	std::unique_lock<mutex> lock;
};

static idx_t ArrowTableScanCapacity(const py::object &table) {
	if (!py::hasattr(table, "num_rows")) {
		throw InvalidInputException("direct Arrow table result must expose num_rows");
	}
	auto rows = py::cast<idx_t>(table.attr("num_rows"));
	if (rows > 0 && rows < STANDARD_VECTOR_SIZE) {
		return rows;
	}
	return STANDARD_VECTOR_SIZE;
}

class OwnedArrowTableStreamFactory {
public:
	OwnedArrowTableStreamFactory() {
		std::memset(&stream, 0, sizeof(stream));
		std::memset(&schema, 0, sizeof(schema));
	}

	OwnedArrowTableStreamFactory(const OwnedArrowTableStreamFactory &) = delete;
	OwnedArrowTableStreamFactory &operator=(const OwnedArrowTableStreamFactory &) = delete;

	~OwnedArrowTableStreamFactory() {
		if (stream.release) {
			stream.release(&stream);
			D_ASSERT(!stream.release);
		}
		if (schema.release) {
			schema.release(&schema);
			D_ASSERT(!schema.release);
		}
	}

	ArrowArrayStream stream;
	ArrowSchema schema;
};

static unique_ptr<ArrowArrayStreamWrapper> ProduceOwnedArrowTableStream(uintptr_t factory_ptr,
                                                                        ArrowStreamParameters &parameters) {
	if (parameters.filters && !parameters.filters->filters.empty()) {
		throw InvalidInputException("direct Arrow table conversion does not support filter pushdown");
	}
	auto factory = reinterpret_cast<OwnedArrowTableStreamFactory *>(factory_ptr);
	if (!factory->stream.release) {
		throw InvalidInputException("direct Arrow table stream has already been consumed");
	}
	auto res = make_uniq<ArrowArrayStreamWrapper>();
	res->arrow_array_stream = factory->stream;
	factory->stream.release = nullptr;
	return res;
}

static void GetOwnedArrowTableStreamSchema(ArrowArrayStream *factory_ptr, ArrowSchema &schema) {
	auto factory = reinterpret_cast<OwnedArrowTableStreamFactory *>(factory_ptr);
	if (!factory->schema.release) {
		throw InvalidInputException("direct Arrow table stream schema is not available");
	}
	schema = factory->schema;
	schema.release = nullptr;
}

static unique_ptr<OwnedArrowTableStreamFactory> ExportOwnedArrowTableStream(const py::object &table) {
	D_ASSERT(py::gil_check());
	if (ClientContextMutexHeldByCurrentThread()) {
		g_udf_python_export_under_client_context_lock_count.fetch_add(1, std::memory_order_relaxed);
		throw InternalException(
		    "direct Arrow table conversion attempted Python export while holding ClientContext lock");
	}
	auto factory = make_uniq<OwnedArrowTableStreamFactory>();
	auto capsule_obj = table.attr("__arrow_c_stream__")();
	auto capsule = py::reinterpret_borrow<py::capsule>(capsule_obj);
	auto stream = capsule.get_pointer<struct ArrowArrayStream>();
	if (!stream || !stream->release) {
		throw InvalidInputException("The __arrow_c_stream__() method returned a released stream");
	}
	factory->stream = *stream;
	stream->release = nullptr;
	if (factory->stream.get_schema(&factory->stream, &factory->schema)) {
		auto error =
		    factory->stream.get_last_error ? factory->stream.get_last_error(&factory->stream) : "unknown error";
		throw InvalidInputException("Failed to get Arrow schema from direct Arrow table stream: %s", error);
	}
	if (!factory->schema.release) {
		throw InvalidInputException("direct Arrow table stream returned a released schema");
	}
	return factory;
}

class ScopedGILReleaseIfHeld {
public:
	ScopedGILReleaseIfHeld() {
		if (PythonRuntimeUsableForUDFTeardown() && PyGILState_Check()) {
			release = make_uniq<py::gil_scoped_release>();
		}
	}

	ScopedGILReleaseIfHeld(const ScopedGILReleaseIfHeld &) = delete;
	ScopedGILReleaseIfHeld &operator=(const ScopedGILReleaseIfHeld &) = delete;

private:
	unique_ptr<py::gil_scoped_release> release;
};

static void AreExtensionsRegistered(const LogicalType &arrow_type, const LogicalType &duckdb_type) {
	if (arrow_type != duckdb_type) {
		if (arrow_type.id() == LogicalTypeId::BLOB && duckdb_type.id() == LogicalTypeId::UUID) {
			throw InvalidConfigurationException(
			    "Mismatch on return type from Arrow object (%s) and DuckDB (%s). It seems that you are using the UUID "
			    "arrow canonical extension, but the same is not yet registered. Make sure to register it first with "
			    "e.g., pa.register_extension_type(UUIDType()). ",
			    arrow_type.ToString(), duckdb_type.ToString());
		}
		if (!arrow_type.IsJSONType() && duckdb_type.IsJSONType()) {
			throw InvalidConfigurationException(
			    "Mismatch on return type from Arrow object (%s) and DuckDB (%s). It seems that you are using the JSON "
			    "arrow canonical extension, but the same is not yet registered. Make sure to register it first with "
			    "e.g., pa.register_extension_type(JSONType()). ",
			    arrow_type.ToString(), duckdb_type.ToString());
		}
	}
}

static unique_ptr<DataChunk> ConvertArrowTableToDataChunk(const py::object &table, ClientContext &context,
                                                          const vector<LogicalType> &expected_types) {
	g_udf_direct_arrow_table_conversion_count.fetch_add(1, std::memory_order_relaxed);
	D_ASSERT(py::gil_check());
	auto scan_capacity = ArrowTableScanCapacity(table);
	auto stream_factory = ExportOwnedArrowTableStream(table);
	py::gil_scoped_release gil;
	ScopedClientContextLock ctx_g;

	auto stream_factory_produce = ProduceOwnedArrowTableStream;
	auto stream_factory_get_schema = GetOwnedArrowTableStreamSchema;

	vector<Value> children;
	children.reserve(3);
	children.push_back(Value::POINTER(CastPointerToValue(stream_factory.get())));
	children.push_back(Value::POINTER(CastPointerToValue(stream_factory_produce)));
	children.push_back(Value::POINTER(CastPointerToValue(stream_factory_get_schema)));

	named_parameter_map_t named_params;
	vector<LogicalType> input_types;
	vector<string> input_names;

	TableFunctionRef empty;
	TableFunction dummy_table_function;
	dummy_table_function.name = "ConvertArrowTableToDataChunk";
	TableFunctionBindInput bind_input(children, named_params, input_types, input_names, nullptr, nullptr,
	                                  dummy_table_function, empty);
	vector<LogicalType> return_types;
	vector<string> return_names;

	auto bind_data = ArrowTableFunction::ArrowScanBind(context, bind_input, return_types, return_names);

	bool needs_cast = false;
	if (!expected_types.empty()) {
		if (return_types.size() != expected_types.size()) {
			throw InvalidInputException("Arrow result column count %d does not match expected %d", return_types.size(),
			                            expected_types.size());
		}
		for (idx_t i = 0; i < return_types.size(); i++) {
			AreExtensionsRegistered(return_types[i], expected_types[i]);
			if (return_types[i] != expected_types[i]) {
				needs_cast = true;
			}
		}
	}

	vector<column_t> column_ids;
	column_ids.reserve(return_types.size());
	for (idx_t i = 0; i < return_types.size(); i++) {
		column_ids.push_back(i);
	}

	TableFunctionInitInput input(bind_data.get(), column_ids, vector<idx_t>(), nullptr);
	auto global_state = ArrowTableFunction::ArrowScanInitGlobal(context, input);
	auto local_state = ArrowTableFunction::ArrowScanInitLocalInternal(context, input, global_state.get());

	TableFunctionInput function_input(bind_data.get(), local_state.get(), global_state.get());

	// Scan first batch
	DataChunk first_chunk;
	first_chunk.Initialize(context, return_types, scan_capacity);
	ArrowTableFunction::ArrowScanFunction(context, function_input, first_chunk);

	if (first_chunk.size() == 0) {
		// Empty table
		auto output = make_uniq<DataChunk>();
		output->Initialize(context, return_types, 0);
		output->SetCardinality(0);
		return output;
	}

	// Peek for a second batch to decide fast-path vs slow-path
	DataChunk peek_chunk;
	peek_chunk.Initialize(context, return_types, scan_capacity);
	ArrowTableFunction::ArrowScanFunction(context, function_input, peek_chunk);

	if (peek_chunk.size() == 0) {
		// ── Single-batch fast path: skip Flatten+Append copy ──
		// DirectConversion already set vector data pointers directly into Arrow
		// buffers (zero-copy). ArrowAuxiliaryData on vector buffers keeps the
		// Arrow memory alive after scan state is destroyed.
		if (needs_cast) {
			DataChunk cast_chunk;
			cast_chunk.Initialize(context, expected_types, first_chunk.size());
			cast_chunk.SetCardinality(first_chunk.size());
			for (idx_t i = 0; i < first_chunk.ColumnCount(); i++) {
				VectorOperations::Cast(context, first_chunk.data[i], cast_chunk.data[i], first_chunk.size());
				cast_chunk.data[i].Verify(first_chunk.size());
			}
			auto output = make_uniq<DataChunk>();
			output->Move(cast_chunk);
			return output;
		}
		auto output = make_uniq<DataChunk>();
		output->Move(first_chunk);
		return output;
	}

	// ── Multi-batch slow path: Flatten+Append (copies data) ──
	DataChunk result;
	result.Initialize(context, return_types, scan_capacity);
	result.SetCardinality(0);

	first_chunk.Flatten();
	result.Append(first_chunk, true);
	peek_chunk.Flatten();
	result.Append(peek_chunk, true);

	DataChunk scan_chunk;
	scan_chunk.Initialize(context, return_types, scan_capacity);
	while (true) {
		scan_chunk.Reset();
		ArrowTableFunction::ArrowScanFunction(context, function_input, scan_chunk);
		if (scan_chunk.size() == 0) {
			break;
		}
		scan_chunk.Flatten();
		result.Append(scan_chunk, true);
	}

	if (needs_cast) {
		DataChunk cast_chunk;
		cast_chunk.Initialize(context, expected_types, result.size());
		cast_chunk.SetCardinality(result.size());
		for (idx_t i = 0; i < result.ColumnCount(); i++) {
			VectorOperations::Cast(context, result.data[i], cast_chunk.data[i], result.size());
			cast_chunk.data[i].Verify(result.size());
		}
		auto output = make_uniq<DataChunk>();
		output->Move(cast_chunk);
		return output;
	}

	auto output = make_uniq<DataChunk>();
	output->Move(result);
	return output;
}

static unique_ptr<DataChunk> MakeEmptyDataChunk() {
	auto output = make_uniq<DataChunk>();
	vector<LogicalType> types;
	output->Initialize(Allocator::DefaultAllocator(), types, 0);
	output->SetCardinality(0);
	return output;
}

struct PythonReadyResult {
	py::object output;
	bool submit_complete = true;
	idx_t submit_id = 0;
};

static PythonReadyResult DecodePythonReadyResult(const py::handle &value) {
	PythonReadyResult result;
	result.output = py::reinterpret_borrow<py::object>(value);
	if (!py::isinstance<py::tuple>(value)) {
		return result;
	}
	auto tuple = py::reinterpret_borrow<py::tuple>(value);
	if (py::len(tuple) == 3 && py::isinstance<py::str>(tuple[0])) {
		auto marker = tuple[0].cast<string>();
		if (marker == "__vane_submit_result__") {
			result.submit_id = tuple[1].cast<idx_t>();
			result.output = py::reinterpret_borrow<py::object>(tuple[2]);
			return result;
		}
	}
	if (py::len(tuple) != 2 || !py::isinstance<py::bool_>(tuple[1])) {
		return result;
	}
	result.output = py::reinterpret_borrow<py::object>(tuple[0]);
	result.submit_complete = tuple[1].cast<bool>();
	return result;
}

static bool IsPythonExceptionObject(const py::handle &value) {
	return py::hasattr(value, "__traceback__");
}

static constexpr const char *REF_BUNDLE_RESULT_MARKER = "__vane_ref_bundle_result__";

static shared_ptr<void> MakePythonObjectHolder(py::object obj) {
	auto *boxed = new py::object(std::move(obj));
	return shared_ptr<void>(boxed, [](void *ptr) {
		auto *boxed_obj = static_cast<py::object *>(ptr);
		if (!boxed_obj) {
			return;
		}
		if (!PythonRuntimeUsableForUDFTeardown()) {
			boxed_obj->release();
			delete boxed_obj;
			return;
		}
		try {
			PythonGILWrapper gil;
			delete boxed_obj;
		} catch (...) {
			// Avoid throwing from a shared_ptr deleter. The process is already
			// tearing down Python state, so leaking the small holder is safer.
		}
	});
}

static py::object BorrowPythonObjectHolder(const shared_ptr<void> &holder) {
	if (!holder) {
		return py::none();
	}
	auto *boxed = static_cast<py::object *>(holder.get());
	if (!boxed) {
		return py::none();
	}
	return py::reinterpret_borrow<py::object>(*boxed);
}

struct CollectorOutputLeaseCallbacks {
	std::function<void()> handoff;
	std::function<void()> release;
};

static CollectorOutputLeaseCallbacks
MakeCollectorOutputLeaseCallbacks(const py::object &collector, const string &request_id, const string &lease_id);

static bool PyDictContains(const py::dict &dict, const char *key) {
	return dict.contains(py::str(key));
}

static idx_t PyDictIdx(const py::dict &dict, const char *key, idx_t default_value = 0) {
	if (!PyDictContains(dict, key)) {
		return default_value;
	}
	auto value = dict[py::str(key)];
	if (value.is_none()) {
		return default_value;
	}
	return value.cast<idx_t>();
}

static string PyDictString(const py::dict &dict, const char *key) {
	if (!PyDictContains(dict, key)) {
		return string();
	}
	auto value = dict[py::str(key)];
	if (value.is_none()) {
		return string();
	}
	return py::str(value).cast<string>();
}

static bool IsPythonRefBundleResult(const py::handle &value) {
	if (!py::isinstance<py::tuple>(value)) {
		return false;
	}
	auto tuple = py::reinterpret_borrow<py::tuple>(value);
	return py::len(tuple) >= 4 && py::isinstance<py::str>(tuple[0]) &&
	       tuple[0].cast<string>() == REF_BUNDLE_RESULT_MARKER;
}

static unique_ptr<LazyRefDataChunk> ConvertPythonRefBundleResult(const py::handle &value,
                                                                 const vector<LogicalType> &expected_types) {
	if (!IsPythonRefBundleResult(value)) {
		throw InvalidInputException("udf result is not a ref bundle result");
	}
	auto tuple = py::reinterpret_borrow<py::tuple>(value);
	auto refs = py::reinterpret_borrow<py::sequence>(tuple[1]);
	auto metadata = py::reinterpret_borrow<py::sequence>(tuple[2]);
	if (py::len(refs) != py::len(metadata)) {
		throw InvalidInputException("ref bundle result has mismatched refs and metadata lengths");
	}

	auto bundle = make_uniq<LazyRefDataChunk>();
	bundle->logical_types = expected_types;
	if (!tuple[3].is_none()) {
		auto names = py::reinterpret_borrow<py::sequence>(tuple[3]);
		for (auto item : names) {
			bundle->names.push_back(py::str(item).cast<string>());
		}
	}
	if (bundle->names.empty()) {
		for (idx_t i = 0; i < expected_types.size(); i++) {
			bundle->names.push_back(StringUtil::Format("c%d", i));
		}
	}

	for (py::size_t i = 0; i < py::len(refs); i++) {
		RefBlockDescriptor block;
		block.object_ref = MakePythonObjectHolder(py::reinterpret_borrow<py::object>(refs[i]));
		if (!py::isinstance<py::dict>(metadata[i])) {
			throw InvalidInputException("ref bundle metadata entry must be a dict");
		}
		auto meta = py::reinterpret_borrow<py::dict>(metadata[i]);
		block.metadata.num_rows = PyDictIdx(meta, "num_rows");
		block.metadata.size_bytes = PyDictIdx(meta, "size_bytes");
		block.metadata.query_id = PyDictString(meta, "query_id");
		block.metadata.operator_id = PyDictString(meta, "producer_stage_id");
		block.metadata.attempt_id = PyDictString(meta, "attempt_id");
		if (PyDictContains(meta, "slice_start") && PyDictContains(meta, "slice_end")) {
			block.has_slice = true;
			block.slice.start_offset = PyDictIdx(meta, "slice_start");
			block.slice.end_offset = PyDictIdx(meta, "slice_end");
			if (block.slice.end_offset < block.slice.start_offset || block.slice.end_offset > block.metadata.num_rows) {
				throw InvalidInputException("ref bundle metadata has invalid slice bounds");
			}
		}
		bundle->blocks.push_back(std::move(block));
	}
	bundle->RecomputeCardinality();
	return bundle;
}

static unique_ptr<LazyRefDataChunk> CopyLazyRefDataChunk(const LazyRefDataChunk &input) {
	auto copy = make_uniq<LazyRefDataChunk>();
	copy->blocks = input.blocks;
	copy->logical_types = input.logical_types;
	copy->names = input.names;
	copy->cardinality = input.cardinality;
	copy->wrap_columns_as_struct = input.wrap_columns_as_struct;
	return copy;
}

static vector<LogicalType> StructChildTypes(const LogicalType &struct_type) {
	vector<LogicalType> child_types;
	auto child_count = StructType::GetChildCount(struct_type);
	child_types.reserve(child_count);
	for (idx_t i = 0; i < child_count; i++) {
		child_types.push_back(StructType::GetChildType(struct_type, i));
	}
	return child_types;
}

static unique_ptr<DataChunk> WrapDataChunkColumnsAsStruct(ClientContext &context, unique_ptr<DataChunk> raw,
                                                          const LogicalType &struct_type) {
	if (!raw) {
		throw InternalException("cannot wrap null DataChunk as STRUCT");
	}
	auto child_count = StructType::GetChildCount(struct_type);
	if (raw->ColumnCount() != child_count) {
		throw InvalidInputException("lazy block materialization expected %d STRUCT fields but block has %d columns",
		                            child_count, raw->ColumnCount());
	}
	auto row_count = raw->size();
	auto output = make_uniq<DataChunk>();
	vector<LogicalType> output_types;
	output_types.push_back(struct_type);
	output->Initialize(context, output_types, row_count);
	auto &struct_vector = output->data[0];
	auto &struct_entries = StructVector::GetEntries(struct_vector);
	for (idx_t i = 0; i < child_count; i++) {
		auto &declared_type = StructType::GetChildType(struct_type, i);
		if (raw->data[i].GetType() == declared_type) {
			struct_entries[i]->Reference(raw->data[i]);
		} else {
			VectorOperations::Cast(context, raw->data[i], *struct_entries[i], row_count);
		}
	}
	output->SetCardinality(row_count);
	return output;
}

class PythonRayExternalBlockBackend : public ExternalBlockBackendInterface {
public:
	bool CanMaterialize(const ExternalBlockDescriptor &desc) override {
		return desc.object_ref != nullptr;
	}

	unique_ptr<DataChunk> Materialize(ClientContext &context, const LazyDataChunk &chunk) override {
		if (chunk.blocks.empty()) {
			auto output = make_uniq<DataChunk>();
			output->Initialize(context, chunk.logical_types, 0);
			output->SetCardinality(0);
			return output;
		}

		PythonGILWrapper gil;
		py::list refs;
		py::list slices;
		py::list metadata;
		py::list names;
		if (!chunk.wrap_columns_as_struct) {
			for (auto &name : chunk.names) {
				names.append(py::str(name));
			}
		}
		for (auto &block : chunk.blocks) {
			refs.append(BorrowPythonObjectHolder(block.object_ref));
			if (block.has_slice) {
				slices.append(py::make_tuple(block.slice.start_offset, block.slice.end_offset));
			} else {
				slices.append(py::none());
			}
			py::dict meta;
			meta[py::str("num_rows")] = py::int_(block.metadata.num_rows);
			meta[py::str("size_bytes")] = py::int_(block.metadata.size_bytes);
			if (!block.metadata.query_id.empty()) {
				meta[py::str("query_id")] = py::str(block.metadata.query_id);
			}
			if (!block.metadata.operator_id.empty()) {
				meta[py::str("operator_id")] = py::str(block.metadata.operator_id);
			}
			if (!block.metadata.attempt_id.empty()) {
				meta[py::str("attempt_id")] = py::str(block.metadata.attempt_id);
			}
			if (!block.column_ids.empty()) {
				py::list column_ids;
				for (auto column_id : block.column_ids) {
					column_ids.append(py::int_(column_id));
				}
				meta[py::str("column_ids")] = std::move(column_ids);
			}
			metadata.append(std::move(meta));
		}

		py::object table;
		try {
			auto helper = py::module_::import("vane.execution.ref_bundle").attr("materialize_ref_bundle");
			table = helper(std::move(refs), std::move(slices), std::move(metadata), std::move(names));
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("external block materialization failed: %s", ex.what());
		}

		if (chunk.wrap_columns_as_struct) {
			if (chunk.logical_types.size() != 1 || chunk.logical_types[0].id() != LogicalTypeId::STRUCT) {
				throw InvalidInputException("lazy block STRUCT wrapping requires exactly one STRUCT logical type");
			}
			auto raw = ConvertArrowTableToDataChunk(table, context, StructChildTypes(chunk.logical_types[0]));
			py::gil_scoped_release release;
			ScopedClientContextLock ctx_g;
			return WrapDataChunkColumnsAsStruct(context, std::move(raw), chunk.logical_types[0]);
		}
		return ConvertArrowTableToDataChunk(table, context, chunk.logical_types);
	}
};

static std::pair<bool, string> GetStructStringField(const Value &payload, const string &name) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return std::make_pair(false, string());
	}
	auto &children = StructValue::GetChildren(payload);
	auto child_count = StructType::GetChildCount(payload.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) != name) {
			continue;
		}
		if (i >= children.size() || children[i].IsNull()) {
			return std::make_pair(false, string());
		}
		auto &child_type = StructType::GetChildType(payload.type(), i);
		if (child_type.id() == LogicalTypeId::VARCHAR) {
			return std::make_pair(true, StringValue::Get(children[i]));
		}
		return std::make_pair(true, children[i].ToString());
	}
	return std::make_pair(false, string());
}

static bool GetStructValueField(const Value &payload, const string &name, Value &result);

static std::pair<bool, idx_t> GetStructIdxField(const Value &payload, const string &name) {
	Value value;
	if (!GetStructValueField(payload, name, value)) {
		return std::make_pair(false, idx_t(0));
	}
	auto parsed = value.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
	if (parsed < 0) {
		throw InvalidInputException("UDF payload field '%s' must be a non-negative integer", name);
	}
	return std::make_pair(true, static_cast<idx_t>(parsed));
}

static bool GetStructBoolFlag(const Value &payload, const string &name) {
	auto value = GetStructStringField(payload, name);
	return value.first && (StringUtil::CIEquals(value.second, "true") || value.second == "1");
}

static bool PayloadUsesTaskAdmission(const Value &payload) {
	auto backend = GetStructStringField(payload, "execution_backend");
	return backend.first && (backend.second == "ray_task" || backend.second == "ray_actor" ||
	                         backend.second == "subprocess_task" || backend.second == "subprocess_actor");
}

static string GetStructStringOrDefault(const Value &payload, const string &name, const string &default_value = "") {
	auto value = GetStructStringField(payload, name);
	if (!value.first || value.second.empty()) {
		return default_value;
	}
	return value.second;
}

static bool RequiresDistributedRefBundleOutput(const Value &payload) {
	return GetStructBoolFlag(payload, "produce_ref_bundle_output") ||
	       GetStructBoolFlag(payload, "produce_ray_block_stream");
}

static bool IsDistributedUDFOutputContractPayload(const Value &payload) {
	return RequiresDistributedRefBundleOutput(payload);
}

static void ValidateDistributedRefBundlePayload(const Value &payload) {
	if (!IsDistributedUDFOutputContractPayload(payload)) {
		return;
	}
	auto backend = GetStructStringField(payload, "execution_backend");
	if (!backend.first || backend.second.empty()) {
		throw InvalidInputException("distributed UDF ref-bundle output contract requires execution_backend");
	}
	const bool ray_backend = backend.second == "ray_task" || backend.second == "ray_actor";
	const bool subprocess_backend = backend.second == "subprocess_task" || backend.second == "subprocess_actor";
	if (!ray_backend && !subprocess_backend) {
		throw InvalidInputException(
		    "distributed UDF ref-bundle output contract does not support execution_backend='%s'", backend.second);
	}
	if (ray_backend) {
		if (!GetStructBoolFlag(payload, "produce_ray_block_stream")) {
			throw InvalidInputException(
			    "distributed Ray UDF output requires produce_ray_block_stream=true; udf_name=%s execution_backend=%s",
			    GetStructStringOrDefault(payload, "udf_name", "<unknown>"), backend.second);
		}
		return;
	}
	if (!GetStructBoolFlag(payload, "produce_ref_bundle_output")) {
		throw InvalidInputException(
		    "distributed UDF ref-bundle output contract requires produce_ref_bundle_output=true; udf_name=%s "
		    "execution_backend=%s streaming_breaker=%s stream_output=%s",
		    GetStructStringOrDefault(payload, "udf_name", "<unknown>"), backend.second,
		    GetStructBoolFlag(payload, "streaming_breaker") ? "true" : "false",
		    GetStructBoolFlag(payload, "stream_output") ? "true" : "false");
	}
	auto mode = GetStructStringField(payload, "streaming_output_mode");
	if (!mode.first || mode.second.empty()) {
		throw InvalidInputException(
		    "distributed UDF ref-bundle output contract requires streaming_output_mode=%s; udf_name=%s "
		    "execution_backend=%s",
		    "local_shm_ref_bundle", GetStructStringOrDefault(payload, "udf_name", "<unknown>"), backend.second);
	}
	if (mode.second != "local_shm_ref_bundle") {
		throw InvalidInputException(
		    "distributed UDF ref-bundle output contract expected streaming_output_mode=%s for execution_backend=%s; "
		    "udf_name=%s got=%s",
		    "local_shm_ref_bundle", backend.second, GetStructStringOrDefault(payload, "udf_name", "<unknown>"),
		    mode.second);
	}
}

static string PythonTypeName(const py::handle &value) {
	try {
		return py::str(py::type::of(value)).cast<string>();
	} catch (...) {
		return "<unknown>";
	}
}

static string DistributedRefBundleContractError(const Value &payload, idx_t submit_id, const py::handle &value) {
	return StringUtil::Format(
	    "distributed UDF DATA output must be a direct block-stream ref-bundle; udf_name=%s execution_backend=%s "
	    "submit_id=%llu got=%s",
	    GetStructStringOrDefault(payload, "udf_name", "<unknown>"),
	    GetStructStringOrDefault(payload, "execution_backend", "<unknown>"), static_cast<unsigned long long>(submit_id),
	    PythonTypeName(value));
}

static bool GetStructValueField(const Value &payload, const string &name, Value &result) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return false;
	}
	auto &children = StructValue::GetChildren(payload);
	auto child_count = StructType::GetChildCount(payload.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) != name) {
			continue;
		}
		if (i >= children.size() || children[i].IsNull()) {
			return false;
		}
		result = children[i];
		return true;
	}
	return false;
}

static bool GetStructShapeField(const Value &payload, const string &name, vector<idx_t> &shape) {
	Value child;
	if (!GetStructValueField(payload, name, child)) {
		return false;
	}
	if (child.IsNull() || child.type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("udf output_schema tensor field '%s' must be a LIST<BIGINT>", name);
	}
	shape.clear();
	auto &children = ListValue::GetChildren(child);
	shape.reserve(children.size());
	for (auto &entry : children) {
		auto dim = entry.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
		if (dim <= 0) {
			throw InvalidInputException("udf output_schema tensor shape dimensions must be positive");
		}
		shape.push_back(NumericCast<idx_t>(dim));
	}
	if (shape.empty()) {
		throw InvalidInputException("udf output_schema tensor shape must be non-empty");
	}
	return true;
}

static vector<LogicalType> ParseOutputSchemaField(const Value &payload) {
	vector<LogicalType> output_types;
	Value output_schema;
	if (!GetStructValueField(payload, "output_schema", output_schema)) {
		return output_types;
	}
	auto &entries = ListValue::GetChildren(output_schema);
	output_types.reserve(entries.size());
	for (auto &entry : entries) {
		if (entry.IsNull() || entry.type().id() != LogicalTypeId::STRUCT) {
			throw InvalidInputException("udf output_schema entries must be STRUCT values");
		}
		auto entry_kind = GetStructStringField(entry, "kind");
		if (!entry_kind.first || entry_kind.second.empty() || StringUtil::CIEquals(entry_kind.second, "duckdb_type")) {
			auto entry_type = GetStructStringField(entry, "type");
			if (!entry_type.first || entry_type.second.empty()) {
				throw InvalidInputException("udf output_schema duckdb_type entry is missing type");
			}
			output_types.push_back(DBConfig::ParseLogicalType(entry_type.second));
			continue;
		}
		if (StringUtil::CIEquals(entry_kind.second, "tensor")) {
			auto entry_dtype = GetStructStringField(entry, "dtype");
			if (!entry_dtype.first || entry_dtype.second.empty()) {
				throw InvalidInputException("udf output_schema tensor entry is missing dtype");
			}
			vector<idx_t> shape;
			if (!GetStructShapeField(entry, "shape", shape)) {
				throw InvalidInputException("udf output_schema tensor entry is missing shape");
			}
			output_types.push_back(TensorType::Create(DBConfig::ParseLogicalType(entry_dtype.second), shape));
			continue;
		}
		throw InvalidInputException("Unsupported udf output_schema kind '%s'", entry_kind.second);
	}
	return output_types;
}

static bool HasStructField(const Value &payload, const string &name) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return false;
	}
	auto child_count = StructType::GetChildCount(payload.type());
	auto &children = StructValue::GetChildren(payload);
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) != name) {
			continue;
		}
		return i < children.size() && !children[i].IsNull();
	}
	return false;
}

static vector<LogicalType> ParseStringListTypesField(const Value &payload, const string &name);

static vector<LogicalType> ParseExpectedOutputTypes(const Value &payload) {
	auto return_type = GetStructStringField(payload, "method_return_type");
	if (return_type.first && !return_type.second.empty()) {
		vector<LogicalType> output_types;
		output_types.push_back(DBConfig::ParseLogicalType(return_type.second));
		return output_types;
	}
	auto structured_output_types = ParseOutputSchemaField(payload);
	if (!structured_output_types.empty()) {
		return structured_output_types;
	}
	throw InvalidInputException("udf payload is missing output type information");
}

static vector<LogicalType> ParseStringListTypesField(const Value &payload, const string &name) {
	Value field;
	if (!GetStructValueField(payload, name, field)) {
		return {};
	}
	if (field.IsNull() || field.type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("UDF payload field '%s' must be a LIST<VARCHAR>", name);
	}
	vector<LogicalType> types;
	auto &entries = ListValue::GetChildren(field);
	types.reserve(entries.size());
	for (auto &entry : entries) {
		if (entry.IsNull()) {
			throw InvalidInputException("UDF payload field '%s' cannot contain NULL", name);
		}
		types.push_back(DBConfig::ParseLogicalType(StringValue::Get(entry)));
	}
	return types;
}

static bool IsSubprocessExecutionBackend(const string &backend) {
	return backend == "subprocess_task" || backend == "subprocess_actor";
}

// ─── Deep-copy helper (no GIL needed) ────────────────────────────────────────

static unique_ptr<DataChunk> DeepCopyDataChunk(DataChunk &input) {
	auto copy = make_uniq<DataChunk>();
	copy->Initialize(Allocator::DefaultAllocator(), input.GetTypes(), input.size());
	input.Copy(*copy);
	return copy;
}

// ─── Centralized Dispatcher data structures ──────────────────────────────────

struct DispatcherSubmitTask {
	idx_t submit_id = 0;
	vector<unique_ptr<ArrowSchemaWrapper>> arrow_schemas;
	vector<unique_ptr<ArrowArrayWrapper>> arrow_arrays;
	vector<LogicalType> types;
	vector<string> names;
	unique_ptr<DataChunk> rows;
	ClientProperties options;
};

struct DispatcherRefSubmitTask {
	idx_t submit_id = 0;
	unique_ptr<LazyRefDataChunk> bundle;
	unique_ptr<DataChunk> rows;
	ClientProperties options;
};

struct DispatcherCommand {
	enum Type { REQUEST_TASK_ADMISSION, SUBMIT, SUBMIT_REF_BUNDLE, FINISHED_SUBMITTING };
	Type type;
	idx_t retained_input_bytes = 0;          // only used for REQUEST_TASK_ADMISSION
	DispatcherSubmitTask submit_task;        // only used for SUBMIT
	DispatcherRefSubmitTask ref_submit_task; // only used for SUBMIT_REF_BUNDLE
};

struct ExecutorSlot {
	uint64_t id = 0;
	Value payload;
	shared_ptr<void> actor_handles; // opaque boxed py::object; decode only under GIL
	vector<LogicalType> expected_output_types;
	vector<LogicalType> expected_ref_output_types;
	ClientContext *context = nullptr;

	// Python executor — created lazily by dispatcher thread (only thread with GIL)
	unique_ptr<RegisteredObject> py_executor;
	bool py_executor_initialized = false;

	// Command queue: pipeline threads push, dispatcher pops
	mutex cmd_lock;
	std::deque<DispatcherCommand> cmd_queue;

	// One scheduling lookahead for distributed Ray task admission. Active task
	// concurrency is owned by QueryResourceManager task leases, not this slot.
	mutex task_admission_lock;
	bool task_admission_request_pending = false;
	bool task_admission_available = false;
	bool task_admission_reserved = false;
	idx_t task_admission_retained_input_bytes = 0;

	// Result queue: dispatcher pushes, pipeline threads pop
	mutex result_lock;
	std::deque<UDFResult> result_queue;

	// Optional push consumer used by queue-driven streaming breaker sources.
	mutex consumer_lock;
	bool has_output_consumer = false;
	UDFOutputConsumer output_consumer;

	// Rows passthrough: only touched by dispatcher thread
	std::unordered_map<idx_t, unique_ptr<DataChunk>> rows_by_submit_id;
	std::unordered_set<idx_t> terminal_submit_ids;
	// Lazy input descriptors carry upstream output-lease ownership tokens. Keep
	// them alive until the downstream submit reaches a terminal event; retaining
	// them only through remote() submission releases the physical input credit
	// while the actor may still be consuming the ObjectRefs.
	std::unordered_map<idx_t, unique_ptr<LazyRefDataChunk>> ref_inputs_by_submit_id;

	// State flags
	atomic<bool> finished_submitting_acked {false};
	atomic<bool> all_tasks_finished {false};
	atomic<bool> pending_shutdown {false};
	atomic<bool> cleanup_cancel_requested {false};
	atomic<bool> collector_cancel_requested {false};

	// Error propagation
	mutex error_lock;
	atomic<bool> has_error {false};
	atomic<bool> abort_requested {false};
	string error;

	// Wakeup
	mutex wakeup_lock;
	bool has_interrupt_state = false;
	InterruptState interrupt_state;
	std::function<void()> wakeup_callback;

	// Inflight tracking (C++ side — avoids GIL for empty result checks)
	atomic<int> inflight_count {0};
	bool is_table_udf = false;
	atomic<bool> udf_stats_valid {false};
	atomic<idx_t> udf_running_task_count {0};
	atomic<idx_t> udf_queued_task_count {0};
	atomic<idx_t> udf_max_running_tasks {0};
	atomic<bool> udf_output_budget_stats_valid {false};
	atomic<bool> udf_output_budget_available {true};
	atomic<idx_t> udf_output_budget_estimated_bytes {0};
	atomic<idx_t> udf_output_budget_limit_bytes {0};
	atomic<idx_t> udf_output_budget_usage_bytes {0};

	// Synchronous cleanup: signaled by CleanupSlot, waited on by Unregister
	mutex cleanup_lock;
	std::condition_variable cleanup_cv;
	bool cleanup_complete {false};
};

static ClientContext &RequireActiveSlotContext(ExecutorSlot &slot) {
	if (slot.pending_shutdown.load() || !slot.context) {
		throw InvalidInputException("udf executor slot is shutting down");
	}
	return *slot.context;
}

// ─── GlobalPythonDispatcher (singleton) ──────────────────────────────────────
//
// All Python/GIL interactions for UDF executors are routed through a
// single background thread owned by this singleton.  Pipeline threads never
// acquire the GIL — they push commands and pop results via per-slot queues.

class GlobalPythonDispatcher {
public:
	static constexpr size_t MAX_RESULT_QUEUE_PER_SLOT = 4;

	static GlobalPythonDispatcher &Instance() {
		static GlobalPythonDispatcher instance;
		return instance;
	}

	// Register a new executor slot.  Returns slot ID and stores the raw
	// pointer in *out_slot (valid until Unregister).
	uint64_t Register(Value payload, vector<LogicalType> output_types, vector<LogicalType> ref_output_types,
	                  ClientContext *ctx, shared_ptr<void> actor_handles, ExecutorSlot **out_slot) {
		ThrowIfDispatcherError();
		lock_guard<mutex> g(global_lock);
		slot_generation++;
		if (pending_async_shutdown.exchange(false)) {
			{
				lock_guard<mutex> shutdown_lock(async_shutdown_lock);
				async_shutdown_complete = true;
			}
			async_shutdown_cv.notify_all();
		}
		auto id = next_slot_id.fetch_add(1);
		auto slot = make_shared_ptr<ExecutorSlot>();
		slot->id = id;
		slot->payload = std::move(payload);
		slot->actor_handles = std::move(actor_handles);
		slot->expected_output_types = std::move(output_types);
		slot->expected_ref_output_types = std::move(ref_output_types);
		slot->context = ctx;
		const auto payload_async_mode = GetStructBoolFlag(slot->payload, "async_mode");
		if (payload_async_mode && GetStructBoolFlag(slot->payload, "streaming_breaker")) {
			throw InvalidInputException("streaming_breaker=True does not support async_mode=True");
		}
		slot->is_table_udf = ClassifyUDFMode(slot->payload) == UDFMode::RESULT_ONLY_BATCH;
		auto *raw = slot.get();
		slots[id] = std::move(slot);
		*out_slot = raw;
		UDFDebugLog(StringUtil::Format("register slot=%llu total_slots=%llu", static_cast<unsigned long long>(id),
		                               static_cast<unsigned long long>(slots.size())));
		StartThreadLocked();
		return id;
	}

	void Unregister(uint64_t id) {
		shared_ptr<ExecutorSlot> slot;
		const auto unregister_timeout = UDFUnregisterTimeout();
		{
			lock_guard<mutex> g(global_lock);
			auto it = slots.find(id);
			if (it == slots.end()) {
				return;
			}
			slot = it->second;
			slot->pending_shutdown.store(true);
			UDFDebugLog(StringUtil::Format(
			    "unregister_request slot=%llu inflight=%lld slots=%llu", static_cast<unsigned long long>(id),
			    static_cast<long long>(slot->inflight_count.load()), static_cast<unsigned long long>(slots.size())));
		}
		work_pending.store(true);
		work_cv.notify_one();
		if (g_on_udf_dispatcher_thread) {
			// The owning ClientContext may be destroyed as soon as this nested
			// destructor returns. Every subsequent dispatcher phase skips pending
			// slots, so clear the non-owning pointer immediately on this thread.
			slot->context = nullptr;
			// The slot map owns the state until CleanupSlot removes it on the next
			// dispatcher iteration. There is no external destructor to join here.
			UDFDebugLog(
			    StringUtil::Format("unregister_deferred_on_dispatcher slot=%llu", static_cast<unsigned long long>(id)));
			return;
		}
		// Wait for the dispatcher thread to finish cleaning up this slot.
		// This prevents the executor destructor from completing while the
		// dispatcher is still accessing slot data (race → SIGSEGV).
		{
			std::unique_lock<mutex> lk(slot->cleanup_lock);
			if (!slot->cleanup_complete) {
				// CleanupSlot destroys Python-owned objects on the dispatcher
				// thread. If this destructor is running from a pybind entrypoint,
				// holding the GIL while waiting would deadlock that cleanup.
				ScopedGILReleaseIfHeld release_gil;
				if (!slot->cleanup_cv.wait_for(lk, unregister_timeout, [&slot] { return slot->cleanup_complete; })) {
					auto msg =
					    StringUtil::Format("udf unregister cleanup exceeded hard deadline of %llu ms for slot=%llu",
					                       static_cast<unsigned long long>(unregister_timeout.count()),
					                       static_cast<unsigned long long>(id));
					RecordDispatcherError(msg);
					slot->abort_requested.store(true);
					return;
				}
			}
		}
		// CleanupSlot erases the map entry. This local shared_ptr keeps the slot
		// alive through the condition-variable wait and lock destruction.
		{
			std::unique_lock<mutex> shutdown_lk(async_shutdown_lock);
			ScopedGILReleaseIfHeld release_gil;
			if (!async_shutdown_complete && !async_shutdown_cv.wait_for(shutdown_lk, unregister_timeout,
			                                                            [this] { return async_shutdown_complete; })) {
				auto msg = StringUtil::Format("udf async collector shutdown exceeded hard deadline of %llu ms",
				                              static_cast<unsigned long long>(unregister_timeout.count()));
				RecordDispatcherError(msg);
				return;
			}
		}
	}

	void NotifyWork() {
		work_pending.store(true);
		work_cv.notify_one();
	}

	void EnqueueOutputLeaseRelease(std::function<void()> release) {
		if (!release) {
			return;
		}
		{
			lock_guard<mutex> lock(output_lease_release_lock);
			output_lease_releases.push_back(std::move(release));
		}
		NotifyWork();
	}

	void EnqueueDeferredWakeup(std::function<void()> callback) {
		if (!callback) {
			return;
		}
		{
			lock_guard<mutex> lock(deferred_wakeup_lock);
			deferred_wakeups.push_back(std::move(callback));
		}
		NotifyWork();
	}

	~GlobalPythonDispatcher() {
		StopThread();
	}

	void Shutdown() {
		// Python atexit callbacks still own a valid interpreter and normally hold
		// the GIL.  Release it while joining so the dispatcher can acquire the GIL
		// for its final Python-object cleanup before module/static destruction.
		ScopedGILReleaseIfHeld release_gil;
		StopThread();
	}

private:
	GlobalPythonDispatcher() = default;
	GlobalPythonDispatcher(const GlobalPythonDispatcher &) = delete;
	GlobalPythonDispatcher &operator=(const GlobalPythonDispatcher &) = delete;

	void StartThreadLocked() {
		if (thread_running) {
			return;
		}
		stop.store(false);
		dispatcher_thread = std::thread([this]() { DispatcherLoop(); });
		thread_running = true;
	}

	void StopThread() {
		if (!thread_running) {
			return;
		}
		stop.store(true);
		work_pending.store(true);
		work_cv.notify_one();
		if (dispatcher_thread.joinable()) {
			dispatcher_thread.join();
		}
		thread_running = false;
	}

	void RecordDispatcherError(const string &msg) {
		{
			lock_guard<mutex> lock(dispatcher_error_lock);
			if (!has_dispatcher_error.load()) {
				dispatcher_error = msg;
				has_dispatcher_error.store(true);
			}
		}
		work_pending.store(true);
		work_cv.notify_one();
	}

	void ThrowIfDispatcherError() {
		if (!has_dispatcher_error.load()) {
			return;
		}
		lock_guard<mutex> lock(dispatcher_error_lock);
		throw InvalidInputException("udf dispatcher error: %s", dispatcher_error);
	}

	bool HasPendingOutputLeaseReleases() {
		lock_guard<mutex> lock(output_lease_release_lock);
		return !output_lease_releases.empty();
	}

	bool DrainDeferredWakeups() {
		std::deque<std::function<void()>> callbacks;
		{
			lock_guard<mutex> lock(deferred_wakeup_lock);
			callbacks.swap(deferred_wakeups);
		}
		for (auto &callback : callbacks) {
			if (!callback) {
				continue;
			}
			try {
				callback();
			} catch (const std::exception &ex) {
				RecordDispatcherError(StringUtil::Format("deferred UDF wakeup failed: %s", ex.what()));
			} catch (...) {
				RecordDispatcherError("deferred UDF wakeup failed with an unknown exception");
			}
		}
		return !callbacks.empty();
	}

	bool DrainOutputLeaseReleases_WithGIL() {
		std::deque<std::function<void()>> releases;
		{
			lock_guard<mutex> lock(output_lease_release_lock);
			releases.swap(output_lease_releases);
		}
		for (auto &release : releases) {
			if (release) {
				release();
			}
		}
		if (!releases.empty()) {
			UDFDebugLog(StringUtil::Format("output_lease_releases drained=%llu",
			                               static_cast<unsigned long long>(releases.size())));
		}
		return !releases.empty();
	}

	// ── Dispatcher main loop (runs on the single dispatcher thread) ──────

	void DispatcherLoop() {
		g_on_udf_dispatcher_thread = true;
		while (!stop.load()) {
			// Consume event flags at the start of the loop. Producers set these
			// flags before notifying work_cv; clearing them later in the loop can
			// clobber a wakeup that races with async result draining and put the
			// dispatcher to sleep while results are already queued.
			work_pending.exchange(false);
			auto python_wakeup_fired = wakeup_fired.exchange(false);

			// Snapshot active slot pointers
			vector<ExecutorSlot *> active;
			{
				lock_guard<mutex> g(global_lock);
				active.reserve(slots.size());
				for (auto &kv : slots) {
					active.push_back(kv.second.get());
				}
			}

			// Phase 1: drain all command queues (no GIL) and handle shutdowns.
			// A slot may be unregistered immediately after a pipeline finalizer
			// enqueues its last submit. Since inflight_count is only incremented
			// once the dispatcher runs the Python submit, cleanup must not run
			// before already-queued commands are dispatched.
			bool did_work = DrainDeferredWakeups();
			struct SlotCommands {
				ExecutorSlot *slot;
				std::deque<DispatcherCommand> commands;
			};
			vector<unique_ptr<SlotCommands>> pending_commands;
			bool has_pending_cleanup_cancel = false;
			bool has_output_lease_releases = HasPendingOutputLeaseReleases();

			for (auto &slot : active) {
				if (stop.load()) {
					break;
				}
				std::deque<DispatcherCommand> commands;
				{
					lock_guard<mutex> sg(slot->cmd_lock);
					commands.swap(slot->cmd_queue);
				}
				const bool had_commands = !commands.empty();
				if (had_commands) {
					auto sc = make_uniq<SlotCommands>();
					sc->slot = slot;
					sc->commands = std::move(commands);
					pending_commands.push_back(std::move(sc));
				}
				if (slot->pending_shutdown.load()) {
					if (had_commands) {
						continue;
					}
					if (!slot->cleanup_complete) {
						int inflight = slot->inflight_count.load();
						if (inflight > 0 && !slot->cleanup_cancel_requested.load()) {
							has_pending_cleanup_cancel = true;
							continue;
						}
						CleanupSlot(slot->id);
						did_work = true;
					}
					slot = nullptr; // prevent dangling pointer in subsequent loops
					continue;
				}
			}

			// Check if any slot has completed all tasks (C++ side only, no GIL needed)
			for (auto *slot : active) {
				if (!slot)
					continue; // skip cleaned-up slots
				if (!slot->pending_shutdown.load() && slot->finished_submitting_acked.load() &&
				    slot->inflight_count.load() == 0 && !slot->all_tasks_finished.load()) {
					slot->all_tasks_finished.store(true);
					NotifySlotFinished(*slot);
					did_work = true;
				}
			}

			// Drain async sources whenever there are in-flight slots or tracked ObjectRefs in the async collector.
			bool has_async_inflight = false;
			bool pending_commands_need_async_collector = false;
			for (auto *slot : active) {
				if (!slot) {
					continue;
				}
				if (!slot->pending_shutdown.load() && slot->inflight_count.load() > 0) {
					has_async_inflight = true;
				}
			}
			for (auto &sc : pending_commands) {
				if (!sc || !sc->slot) {
					continue;
				}
				auto execution_backend = GetStructStringField(sc->slot->payload, "execution_backend");
				if (execution_backend.first &&
				    (execution_backend.second == "ray_task" || execution_backend.second == "ray_actor")) {
					pending_commands_need_async_collector = true;
					break;
				}
			}
			bool drain_async = (bool)async_collector || has_async_inflight;

			bool needs_gil = !pending_commands.empty() || has_pending_cleanup_cancel || drain_async ||
			                 has_output_lease_releases || pending_async_shutdown.load();
			needs_gil = needs_gil || python_wakeup_fired;
			auto debug_loop_tick = g_udf_debug_dispatcher_tick.fetch_add(1, std::memory_order_relaxed) + 1;
			idx_t debug_active_slots = 0;
			idx_t debug_inflight_slots = 0;
			idx_t debug_inflight_total = 0;
			idx_t debug_pending_shutdown_slots = 0;
			idx_t debug_command_count = 0;
			for (auto *slot : active) {
				if (!slot) {
					continue;
				}
				debug_active_slots++;
				auto inflight = slot->inflight_count.load();
				if (inflight > 0) {
					debug_inflight_slots++;
					debug_inflight_total += static_cast<idx_t>(inflight);
				}
				if (slot->pending_shutdown.load()) {
					debug_pending_shutdown_slots++;
				}
			}
			for (auto &sc : pending_commands) {
				if (sc) {
					debug_command_count += static_cast<idx_t>(sc->commands.size());
				}
			}
			if (UDFDebugEnabled() &&
			    (!pending_commands.empty() || has_pending_cleanup_cancel || has_output_lease_releases ||
			     pending_async_shutdown.load() || debug_loop_tick % 200 == 0)) {
				UDFDebugLog(StringUtil::Format(
				    "dispatcher_loop tick=%llu active_slots=%llu inflight_slots=%llu inflight_total=%llu "
				    "pending_shutdown_slots=%llu command_slots=%llu command_count=%llu async_collector=%s "
				    "has_async_inflight=%s drain_async=%s needs_gil=%s work_pending=%s wakeup_fired=%s",
				    static_cast<unsigned long long>(debug_loop_tick),
				    static_cast<unsigned long long>(debug_active_slots),
				    static_cast<unsigned long long>(debug_inflight_slots),
				    static_cast<unsigned long long>(debug_inflight_total),
				    static_cast<unsigned long long>(debug_pending_shutdown_slots),
				    static_cast<unsigned long long>(pending_commands.size()),
				    static_cast<unsigned long long>(debug_command_count), async_collector ? "true" : "false",
				    has_async_inflight ? "true" : "false", drain_async ? "true" : "false", needs_gil ? "true" : "false",
				    work_pending.load() ? "true" : "false", python_wakeup_fired ? "true" : "false"));
			}
			// Phase 2: single GIL acquisition for all submits + async drain
			if (needs_gil) {
				PythonGILWrapper gil;

				// Ensure async collector exists for ObjectRef-based paths.
				if ((bool)async_collector || pending_commands_need_async_collector) {
					try {
						EnsureAsyncCollector();
					} catch (const std::exception &ex) {
						auto msg = StringUtil::Format("udf async collector initialization failed: %s", ex.what());
						bool marked_slot = false;
						for (auto &sc : pending_commands) {
							if (sc && sc->slot && !sc->slot->abort_requested.load()) {
								SetSlotError(*sc->slot, msg);
								marked_slot = true;
							}
						}
						for (auto *slot : active) {
							if (!slot || slot->abort_requested.load() || slot->inflight_count.load() <= 0) {
								continue;
							}
							SetSlotError(*slot, msg);
							marked_slot = true;
						}
						if (!marked_slot) {
							RecordDispatcherError(msg);
						}
						did_work = true;
						continue;
					}
				}

				if (has_output_lease_releases) {
					try {
						did_work |= DrainOutputLeaseReleases_WithGIL();
					} catch (const std::exception &ex) {
						auto msg = StringUtil::Format("udf output lease release failed: %s", ex.what());
						bool marked_slot = false;
						for (auto *slot : active) {
							if (!slot || slot->abort_requested.load()) {
								continue;
							}
							SetSlotError(*slot, msg);
							marked_slot = true;
						}
						if (!marked_slot) {
							RecordDispatcherError(msg);
						}
						did_work = true;
					}
				}

				// Process all pending commands
				for (auto &sc : pending_commands) {
					for (auto &cmd : sc->commands) {
						if (stop.load())
							break;
						if (sc->slot->abort_requested.load() || sc->slot->pending_shutdown.load()) {
							did_work = true;
							continue;
						}
						try {
							switch (cmd.type) {
							case DispatcherCommand::REQUEST_TASK_ADMISSION:
								EnsurePythonExecutor_WithGIL(*sc->slot);
								DoRequestTaskAdmission_WithGIL(*sc->slot, cmd.retained_input_bytes);
								did_work = true;
								break;
							case DispatcherCommand::SUBMIT:
								EnsurePythonExecutor_WithGIL(*sc->slot);
								DoSubmit_WithGIL(*sc->slot, cmd.submit_task);
								did_work = true;
								break;
							case DispatcherCommand::SUBMIT_REF_BUNDLE:
								EnsurePythonExecutor_WithGIL(*sc->slot);
								DoSubmitRefBundle_WithGIL(*sc->slot, cmd.ref_submit_task);
								did_work = true;
								break;
							case DispatcherCommand::FINISHED_SUBMITTING:
								EnsurePythonExecutor_WithGIL(*sc->slot);
								DoFinishedSubmitting_WithGIL(*sc->slot);
								did_work = true;
								break;
							}
						} catch (const std::exception &ex) {
							SetSlotError(*sc->slot, ex.what());
						}
					}
				}

				if (has_pending_cleanup_cancel) {
					for (auto *slot : active) {
						if (!slot || !slot->pending_shutdown.load() || slot->cleanup_cancel_requested.load() ||
						    slot->inflight_count.load() <= 0) {
							continue;
						}
						CancelSlotForCleanup_WithGIL(*slot);
						did_work = true;
					}
				}

				// Drain completed results from async collector
				if (drain_async) {
					did_work |= DrainAsyncResults_WithGIL(active);
				}
				for (auto *slot : active) {
					if (!slot || slot->abort_requested.load() || slot->pending_shutdown.load() || !slot->py_executor) {
						continue;
					}
					auto wake_submitter = RefreshTaskAdmission_WithGIL(*slot);
					wake_submitter |= UpdateSlotUDFStats_WithGIL(*slot);
					if (wake_submitter) {
						WakeupPipelineThread(*slot);
						did_work = true;
					}
				}
				// A last-slot shutdown is generation-fenced against concurrent Register.
				// Hold global_lock through collector shutdown so a new slot either cancels
				// the pending generation first or registers after a fully joined shutdown.
				if (pending_async_shutdown.load()) {
					std::unique_lock<mutex> slots_lock(global_lock);
					const bool should_shutdown = pending_async_shutdown.load() && slots.empty() &&
					                             pending_async_shutdown_generation == slot_generation;
					if (should_shutdown && async_collector) {
						try {
							async_collector->obj.attr("shutdown")();
							async_collector.reset();
						} catch (const py::error_already_set &ex) {
							RecordDispatcherError(
							    StringUtil::Format("udf async collector shutdown failed: %s", ex.what()));
							async_collector.reset();
						} catch (const std::exception &ex) {
							RecordDispatcherError(
							    StringUtil::Format("udf async collector shutdown failed: %s", ex.what()));
							async_collector.reset();
						}
					}
					pending_async_shutdown.store(false);
					slots_lock.unlock();
					{
						lock_guard<mutex> shutdown_lock(async_shutdown_lock);
						async_shutdown_complete = true;
					}
					async_shutdown_cv.notify_all();
					did_work = true;
				}
			}

			if (!did_work) {
				std::unique_lock<mutex> lk(global_lock);
				auto predicate = [this] {
					return stop.load() || work_pending.load() || wakeup_fired.load();
				};
				if (UDFDebugEnabled() && (!has_async_inflight || debug_loop_tick % 200 == 0)) {
					UDFDebugLog(StringUtil::Format(
					    "dispatcher_wait tick=%llu has_async_inflight=%s active_slots=%llu inflight_slots=%llu "
					    "inflight_total=%llu work_pending=%s wakeup_fired=%s",
					    static_cast<unsigned long long>(debug_loop_tick), has_async_inflight ? "true" : "false",
					    static_cast<unsigned long long>(debug_active_slots),
					    static_cast<unsigned long long>(debug_inflight_slots),
					    static_cast<unsigned long long>(debug_inflight_total), work_pending.load() ? "true" : "false",
					    wakeup_fired.load() ? "true" : "false"));
				}
				// Pure event-driven wait: only wake on actual events. All producers
				// (NotifyWork, Unregister, StopThread, Python executor wakeup, async
				// collector wakeup) set work_pending/wakeup_fired before signaling.
				work_cv.wait(lk, predicate);
			}
		}

		// Drain every owned wakeup before destroying query/executor state. A
		// callback may enqueue another callback while resolving an error, so run
		// to a fixed point.
		while (DrainDeferredWakeups()) {
		}

		// Final cleanup: only touch Python while the interpreter is still alive.
		// Move slot ownership out of the dispatcher while holding global_lock,
		// then destroy slots after releasing it for the same reason as
		// Unregister(): queued Arrow/PyArrow buffers can acquire the GIL.
		std::unordered_map<uint64_t, shared_ptr<ExecutorSlot>> cleanup_slots;
		unique_ptr<RegisteredObject> cleanup_async_collector;
		{
			lock_guard<mutex> g(global_lock);
			if (slots.empty() && !async_collector) {
				return;
			}
			cleanup_slots.swap(slots);
			cleanup_async_collector = std::move(async_collector);
		}
		if (!PythonRuntimeUsableForUDFTeardown()) {
			for (auto &kv : cleanup_slots) {
				if (kv.second->py_executor) {
					kv.second->py_executor.release();
				}
				kv.second->actor_handles.reset();
				MarkSlotCleanupComplete(*kv.second);
			}
			if (cleanup_async_collector) {
				cleanup_async_collector.release();
			}
			return;
		}
		try {
			PythonGILWrapper gil;
			for (auto &kv : cleanup_slots) {
				if (kv.second->py_executor) {
					kv.second->py_executor.reset();
				}
				kv.second->actor_handles.reset();
			}
			if (cleanup_async_collector) {
				cleanup_async_collector->obj.attr("shutdown")();
				cleanup_async_collector.reset();
			}
		} catch (const py::error_already_set &ex) {
			RecordDispatcherError(StringUtil::Format("udf dispatcher final cleanup failed: %s", ex.what()));
			if (cleanup_async_collector) {
				cleanup_async_collector.reset();
			}
		} catch (const std::exception &ex) {
			RecordDispatcherError(StringUtil::Format("udf dispatcher final cleanup failed: %s", ex.what()));
			if (cleanup_async_collector) {
				cleanup_async_collector.reset();
			}
		}
		for (auto &kv : cleanup_slots) {
			MarkSlotCleanupComplete(*kv.second);
		}
	}

	void MarkSlotCleanupComplete(ExecutorSlot &slot) {
		slot.context = nullptr;
		{
			lock_guard<mutex> cleanup_guard(slot.cleanup_lock);
			slot.cleanup_complete = true;
		}
		slot.cleanup_cv.notify_all();
	}

	void CleanupSlot(uint64_t id) {
		shared_ptr<ExecutorSlot> slot;
		{
			lock_guard<mutex> g(global_lock);
			auto it = slots.find(id);
			if (it == slots.end()) {
				return;
			}
			slot = it->second;
		}
		auto *slot_ptr = slot.get();
		UDFDebugLog(StringUtil::Format("cleanup_start slot=%llu inflight=%lld pending_shutdown=%s",
		                               static_cast<unsigned long long>(id),
		                               static_cast<long long>(slot_ptr->inflight_count.load()),
		                               slot_ptr->pending_shutdown.load() ? "true" : "false"));
		// Cancel any collector-side records for this slot before destroying the
		// Python executor. Normal cleanup has no records left; timeout cleanup
		// uses this as the explicit retire path for late generator events.
		//
		// Destroy all Python objects under GIL (slot still alive in slots map).
		// Both py_executor and actor_handles are py::objects that must not be
		// destroyed without GIL — their implicit dtors call Py_DECREF, which
		// crashes in PyList_AsTuple/PyType_FromModuleAndSpec without GIL.
		{
			PythonGILWrapper gil;
			CancelAsyncCollectorSlot_WithGIL(*slot_ptr);
			if (slot_ptr->py_executor) {
				try {
					slot_ptr->py_executor->obj.attr("close")();
				} catch (const py::error_already_set &ex) {
					SetSlotError(*slot_ptr, StringUtil::Format("udf executor close failed: %s", ex.what()));
				} catch (const std::exception &ex) {
					SetSlotError(*slot_ptr, StringUtil::Format("udf executor close failed: %s", ex.what()));
				}
				slot_ptr->py_executor.reset();
			}
			slot_ptr->actor_handles.reset();
		}
		{
			std::deque<UDFResult> retired_results;
			{
				lock_guard<mutex> rg(slot_ptr->result_lock);
				retired_results.swap(slot_ptr->result_queue);
			}
			slot_ptr->rows_by_submit_id.clear();
			slot_ptr->ref_inputs_by_submit_id.clear();
			slot_ptr->inflight_count.store(0);
		}
		{
			lock_guard<mutex> g(global_lock);
			auto it = slots.find(id);
			if (it != slots.end() && it->second.get() == slot_ptr) {
				slots.erase(it);
				if (slots.empty() && (bool)async_collector) {
					{
						lock_guard<mutex> shutdown_lock(async_shutdown_lock);
						async_shutdown_complete = false;
					}
					pending_async_shutdown.store(true);
					pending_async_shutdown_generation = slot_generation;
				}
			}
		}
		// External unregister callers retain a shared_ptr while waiting. A
		// dispatcher-thread unregister has no waiter, so this local shared_ptr is
		// the final owner after the map entry is erased.
		MarkSlotCleanupComplete(*slot_ptr);
		UDFDebugLog(StringUtil::Format("cleanup_done slot=%llu", static_cast<unsigned long long>(id)));
	}

	// ── Submit/ready-result helpers (caller must hold GIL) ──────────────────────

	bool GetOutputConsumer(ExecutorSlot &slot, UDFOutputConsumer &consumer) {
		lock_guard<mutex> cg(slot.consumer_lock);
		if (!slot.has_output_consumer) {
			return false;
		}
		consumer = slot.output_consumer;
		return true;
	}

	struct SlotResultCapacity {
		idx_t rows = 0;
		idx_t bytes = 0;
		idx_t item_bytes = 0;
		bool has_byte_capacity = false;
		bool has_item_byte_capacity = false;
	};

	SlotResultCapacity GetSlotResultCapacity(ExecutorSlot &slot) {
		UDFOutputConsumer output_consumer;
		if (GetOutputConsumer(slot, output_consumer)) {
			ScopedGILReleaseIfHeld release_gil;
			if (!output_consumer.data_capacity || !output_consumer.data_byte_capacity ||
			    !output_consumer.data_item_byte_capacity) {
				throw InvalidInputException(
				    "registered udf output consumer requires event, byte, and item-byte capacity callbacks");
			}
			SlotResultCapacity result;
			result.rows = output_consumer.data_capacity();
			result.bytes = output_consumer.data_byte_capacity();
			result.item_bytes = output_consumer.data_item_byte_capacity();
			result.has_byte_capacity = true;
			result.has_item_byte_capacity = true;
			return result;
		}
		lock_guard<mutex> rg(slot.result_lock);
		SlotResultCapacity result;
		if (slot.result_queue.size() >= MAX_RESULT_QUEUE_PER_SLOT) {
			return result;
		}
		result.rows = static_cast<idx_t>(MAX_RESULT_QUEUE_PER_SLOT - slot.result_queue.size());
		return result;
	}

	bool ShouldDrainSyncResults(ExecutorSlot &slot) {
		return GetSlotResultCapacity(slot).rows > 0;
	}

	void DoRequestTaskAdmission_WithGIL(ExecutorSlot &slot, idx_t retained_input_bytes) {
		if (!PayloadUsesTaskAdmission(slot.payload)) {
			throw InternalException("task admission requested for an unsupported UDF executor");
		}
		if (!slot.py_executor || !py::hasattr(slot.py_executor->obj, "request_task_admission")) {
			throw InvalidInputException("UDF executor is missing request_task_admission()");
		}
		slot.py_executor->obj.attr("request_task_admission")(py::int_(retained_input_bytes));
	}

	bool RefreshTaskAdmission_WithGIL(ExecutorSlot &slot) {
		if (!PayloadUsesTaskAdmission(slot.payload)) {
			return false;
		}
		if (!slot.py_executor || !py::hasattr(slot.py_executor->obj, "task_admission_state")) {
			throw InvalidInputException("distributed Ray UDF executor is missing task_admission_state()");
		}
		auto raw_state = slot.py_executor->obj.attr("task_admission_state")();
		if (!py::isinstance<py::dict>(raw_state)) {
			throw InvalidInputException("distributed Ray UDF task_admission_state() must return a dict");
		}
		auto state = raw_state.cast<py::dict>();
		auto state_name = py::str(state[py::str("state")]).cast<string>();
		if (state_name == "failed") {
			auto detail = state.contains(py::str("error")) ? py::str(state[py::str("error")]).cast<string>()
			                                               : string("unknown task-admission failure");
			throw InvalidInputException("distributed Ray UDF task admission failed: %s", detail);
		}
		const bool ready = state_name == "ready";
		idx_t retained_input_bytes = 0;
		if (ready) {
			retained_input_bytes = state[py::str("retained_input_bytes")].cast<idx_t>();
		}
		lock_guard<mutex> guard(slot.task_admission_lock);
		if (state_name == "idle" || state_name == "closed") {
			slot.task_admission_reserved = false;
			slot.task_admission_available = false;
			slot.task_admission_retained_input_bytes = 0;
			slot.task_admission_request_pending = false;
			return false;
		}
		const bool available = ready && !slot.task_admission_reserved;
		const bool became_available = available && !slot.task_admission_available;
		slot.task_admission_available = available;
		slot.task_admission_retained_input_bytes = available ? retained_input_bytes : 0;
		if (ready) {
			slot.task_admission_request_pending = false;
		}
		return became_available;
	}

	void ConsumeTaskAdmissionReservation(ExecutorSlot &slot) {
		if (!PayloadUsesTaskAdmission(slot.payload)) {
			return;
		}
		{
			lock_guard<mutex> guard(slot.task_admission_lock);
			if (!slot.task_admission_reserved) {
				throw InternalException("UDF submit consumed task admission without a reserved lease");
			}
			// The Python authority consumed the concrete lease inside submit_with_id().
			// Do not wait for a transient Python idle snapshot: the next request may
			// already become requested before the dispatcher polls state again.
			slot.task_admission_reserved = false;
			slot.task_admission_available = false;
			slot.task_admission_request_pending = false;
			slot.task_admission_retained_input_bytes = 0;
		}
		WakeupPipelineThread(slot);
	}

	void CancelAsyncCollectorSlot_WithGIL(ExecutorSlot &slot) {
		if (!async_collector) {
			return;
		}
		if (slot.collector_cancel_requested.exchange(true)) {
			return;
		}
		try {
			async_collector->obj.attr("cancel_slot")(py::int_(slot.id));
		} catch (const py::error_already_set &ex) {
			SetSlotError(slot, StringUtil::Format("udf async collector cancel_slot failed: %s", ex.what()));
			return;
		} catch (const std::exception &ex) {
			SetSlotError(slot, StringUtil::Format("udf async collector cancel_slot failed: %s", ex.what()));
			return;
		}
		try {
			auto collector_pending = async_collector->obj.attr("slot_has_pending")(py::int_(slot.id)).cast<bool>();
			if (collector_pending) {
				SetSlotError(slot, "udf async collector still has pending records after slot cancel");
			}
		} catch (const py::error_already_set &ex) {
			SetSlotError(slot, StringUtil::Format("udf async collector slot_has_pending failed: %s", ex.what()));
		} catch (const std::exception &ex) {
			SetSlotError(slot, StringUtil::Format("udf async collector slot_has_pending failed: %s", ex.what()));
		}
	}

	void CancelSlotForCleanup_WithGIL(ExecutorSlot &slot) {
		if (slot.cleanup_cancel_requested.exchange(true)) {
			return;
		}
		CancelAsyncCollectorSlot_WithGIL(slot);
		if (slot.py_executor) {
			try {
				slot.py_executor->obj.attr("close")();
			} catch (const py::error_already_set &ex) {
				SetSlotError(slot, StringUtil::Format("udf executor close failed: %s", ex.what()));
			} catch (const std::exception &ex) {
				SetSlotError(slot, StringUtil::Format("udf executor close failed: %s", ex.what()));
			}
		}
		{
			lock_guard<mutex> rg(slot.result_lock);
			slot.result_queue.clear();
		}
		slot.rows_by_submit_id.clear();
		slot.ref_inputs_by_submit_id.clear();
		slot.inflight_count.store(0);
		slot.finished_submitting_acked.store(true);
		slot.all_tasks_finished.store(true);
	}

	void AbortSlotForError(ExecutorSlot &slot) {
		const bool first_abort = !slot.abort_requested.exchange(true);
		slot.inflight_count.store(0);
		slot.finished_submitting_acked.store(true);
		slot.all_tasks_finished.store(true);
		{
			lock_guard<mutex> rg(slot.result_lock);
			slot.result_queue.clear();
		}
		slot.rows_by_submit_id.clear();
		slot.ref_inputs_by_submit_id.clear();
		if (first_abort && PythonRuntimeUsableForUDFTeardown()) {
			try {
				PythonGILWrapper gil;
				CancelAsyncCollectorSlot_WithGIL(slot);
			} catch (const py::error_already_set &ex) {
				SetSlotError(slot, StringUtil::Format("udf async collector cancel_slot failed: %s", ex.what()));
			} catch (const std::exception &ex) {
				SetSlotError(slot, StringUtil::Format("udf async collector cancel_slot failed: %s", ex.what()));
			}
		}
	}

	void MaybeMarkSlotFinished(ExecutorSlot &slot) {
		if (slot.abort_requested.load() || slot.has_error.load()) {
			return;
		}
		if (!slot.finished_submitting_acked.load() || slot.inflight_count.load() != 0) {
			return;
		}
		bool expected = false;
		if (!slot.all_tasks_finished.compare_exchange_strong(expected, true)) {
			return;
		}
		NotifySlotFinished(slot);
	}

	static bool ExtractPythonStatsIdx_WithGIL(const py::dict &stats, const char *key, idx_t &out) {
		auto key_obj = py::str(key);
		if (!stats.contains(key_obj)) {
			return false;
		}
		try {
			auto value = stats[key_obj];
			if (value.is_none()) {
				return false;
			}
			auto parsed = py::int_(value).cast<unsigned long long>();
			out = parsed > static_cast<unsigned long long>(std::numeric_limits<idx_t>::max())
			          ? std::numeric_limits<idx_t>::max()
			          : static_cast<idx_t>(parsed);
			return true;
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("udf stats field '%s' parse failed: %s", key, ex.what());
		} catch (const std::exception &ex) {
			throw InvalidInputException("udf stats field '%s' parse failed: %s", key, ex.what());
		}
	}

	bool UpdateSlotUDFStats_WithGIL(ExecutorSlot &slot) {
		if (!slot.py_executor || !py::hasattr(slot.py_executor->obj, "stats")) {
			return false;
		}
		try {
			auto raw_stats = slot.py_executor->obj.attr("stats")();
			if (!py::isinstance<py::dict>(raw_stats)) {
				return false;
			}
			auto stats = raw_stats.cast<py::dict>();
			idx_t running = 0;
			idx_t queued = 0;
			idx_t max_running = 0;
			idx_t output_budget_available = 1;
			idx_t output_budget_estimated_bytes = 0;
			idx_t output_budget_limit_bytes = 0;
			idx_t output_budget_usage_bytes = 0;
			bool has_running = ExtractPythonStatsIdx_WithGIL(stats, "udf_running_task_count", running);
			bool has_queued = ExtractPythonStatsIdx_WithGIL(stats, "udf_queued_task_count", queued);
			bool has_max = ExtractPythonStatsIdx_WithGIL(stats, "udf_max_running_tasks", max_running);
			bool has_output_budget_available =
			    ExtractPythonStatsIdx_WithGIL(stats, "udf_output_budget_available", output_budget_available);
			bool has_output_budget_estimated = ExtractPythonStatsIdx_WithGIL(stats, "udf_output_budget_estimated_bytes",
			                                                                 output_budget_estimated_bytes);
			bool has_output_budget_limit =
			    ExtractPythonStatsIdx_WithGIL(stats, "udf_output_budget_limit_bytes", output_budget_limit_bytes);
			bool has_output_budget_usage =
			    ExtractPythonStatsIdx_WithGIL(stats, "udf_output_budget_usage_bytes", output_budget_usage_bytes);
			if (!has_running && !has_queued && !has_max && !has_output_budget_available &&
			    !has_output_budget_estimated && !has_output_budget_limit && !has_output_budget_usage) {
				return false;
			}
			if (has_running) {
				slot.udf_running_task_count.store(running);
			}
			if (has_queued) {
				slot.udf_queued_task_count.store(queued);
			}
			if (has_max) {
				slot.udf_max_running_tasks.store(max_running);
			}
			bool wake_submitter = false;
			if (has_output_budget_available) {
				auto available = output_budget_available != 0;
				auto was_available = slot.udf_output_budget_available.exchange(available);
				wake_submitter = !was_available && available;
				slot.udf_output_budget_stats_valid.store(true);
			}
			if (has_output_budget_estimated) {
				slot.udf_output_budget_estimated_bytes.store(output_budget_estimated_bytes);
			}
			if (has_output_budget_limit) {
				slot.udf_output_budget_limit_bytes.store(output_budget_limit_bytes);
			}
			if (has_output_budget_usage) {
				slot.udf_output_budget_usage_bytes.store(output_budget_usage_bytes);
			}
			slot.udf_stats_valid.store(true);
			return wake_submitter;
		} catch (const py::error_already_set &ex) {
			SetSlotError(slot, StringUtil::Format("udf stats update failed: %s", ex.what()));
			return true;
		} catch (const std::exception &ex) {
			SetSlotError(slot, StringUtil::Format("udf stats update failed: %s", ex.what()));
			return true;
		}
	}

	static UDFResult OutputEventToResult(UDFOutputEvent &&event) {
		UDFResult result;
		result.outputs = std::move(event.outputs);
		result.ref_outputs = std::move(event.ref_outputs);
		result.rows = std::move(event.rows);
		result.submit_complete = event.submit_complete;
		result.submit_id = event.submit_id;
		result.handoff_output_lease = std::move(event.handoff_output_lease);
		result.release_output_lease = std::move(event.release_output_lease);
		return result;
	}

	void ReleaseTerminalSubmit(ExecutorSlot &slot, idx_t submit_id) {
		auto inflight_before = slot.inflight_count.load();
		if (submit_id > 0 && !slot.terminal_submit_ids.insert(submit_id).second) {
			SetSlotError(slot, StringUtil::Format("duplicate terminal UDF event for submit_id %llu",
			                                      static_cast<unsigned long long>(submit_id)));
			return;
		}
		if (inflight_before <= 0) {
			SetSlotError(slot, StringUtil::Format("terminal UDF event underflow for submit_id %llu",
			                                      static_cast<unsigned long long>(submit_id)));
			return;
		}
		slot.inflight_count--;
		if (submit_id > 0) {
			slot.rows_by_submit_id.erase(submit_id);
			slot.ref_inputs_by_submit_id.erase(submit_id);
		}
		UDFDebugLog(StringUtil::Format(
		    "release_terminal slot=%llu submit=%llu inflight_before=%lld inflight_after=%lld",
		    static_cast<unsigned long long>(slot.id), static_cast<unsigned long long>(submit_id),
		    static_cast<long long>(inflight_before), static_cast<long long>(slot.inflight_count.load())));
	}

	void QueueTerminalControlResult(ExecutorSlot &slot, idx_t submit_id) {
		UDFResult result;
		result.outputs = MakeEmptyDataChunk();
		result.rows = MakeEmptyDataChunk();
		result.submit_complete = true;
		result.submit_id = submit_id;
		{
			lock_guard<mutex> rg(slot.result_lock);
			slot.result_queue.push_back(std::move(result));
		}
		ReleaseTerminalSubmit(slot, submit_id);
		MaybeMarkSlotFinished(slot);
		WakeupPipelineThread(slot);
	}

	UDFOutputConsumer ResolveOutputConsumer(ExecutorSlot &slot) {
		UDFOutputConsumer output_consumer;
		if (GetOutputConsumer(slot, output_consumer)) {
			return output_consumer;
		}

		UDFOutputConsumer queue_consumer;
		queue_consumer.data_capacity = [&slot]() {
			lock_guard<mutex> rg(slot.result_lock);
			if (slot.result_queue.size() >= MAX_RESULT_QUEUE_PER_SLOT) {
				return idx_t(0);
			}
			return static_cast<idx_t>(MAX_RESULT_QUEUE_PER_SLOT - slot.result_queue.size());
		};
		queue_consumer.accept_event = [this, &slot](UDFOutputEvent &&event) {
			if (event.kind == UDFOutputEventKind::ERROR) {
				throw InvalidInputException("%s", event.error);
			}
			if (event.kind != UDFOutputEventKind::DATA) {
				return;
			}
			auto result = OutputEventToResult(std::move(event));
			{
				lock_guard<mutex> rg(slot.result_lock);
				slot.result_queue.push_back(std::move(result));
			}
			WakeupPipelineThread(slot);
		};
		queue_consumer.accept_error = [this, &slot](const string &msg) {
			SetSlotError(slot, msg);
		};
		queue_consumer.notify_finished = [this, &slot]() {
			NotifySlotFinished(slot);
		};
		return queue_consumer;
	}

	bool DeliverOutputEvent(ExecutorSlot &slot, UDFOutputEvent &&event) {
		const auto event_kind = event.kind;
		const bool submit_complete = event.submit_complete;
		const auto submit_id_for_log = event.submit_id;
		const auto error_for_log = event.error;
		const bool terminal_submit_event =
		    event_kind == UDFOutputEventKind::COMPLETE || event_kind == UDFOutputEventKind::ERROR || submit_complete;
		bool released_terminal_submit = false;
		auto release_terminal_submit = [&]() {
			if (!terminal_submit_event || released_terminal_submit) {
				return;
			}
			released_terminal_submit = true;
			ReleaseTerminalSubmit(slot, submit_id_for_log);
		};
		UDFOutputConsumer registered_consumer;
		const bool has_registered_output_consumer = GetOutputConsumer(slot, registered_consumer);
		if (UDFDebugEnabled() &&
		    (event_kind != UDFOutputEventKind::DATA || terminal_submit_event || has_registered_output_consumer)) {
			UDFDebugLog(StringUtil::Format(
			    "deliver_event slot=%llu submit=%llu kind=%s submit_complete=%s registered_consumer=%s inflight=%lld",
			    static_cast<unsigned long long>(slot.id), static_cast<unsigned long long>(submit_id_for_log),
			    UDFOutputEventKindName(event_kind), submit_complete ? "true" : "false",
			    has_registered_output_consumer ? "true" : "false", static_cast<long long>(slot.inflight_count.load())));
		}

		if (event_kind == UDFOutputEventKind::FINISHED) {
			MaybeMarkSlotFinished(slot);
			return true;
		}

		auto consumer = has_registered_output_consumer ? registered_consumer : ResolveOutputConsumer(slot);
		if (!consumer.accept_event) {
			release_terminal_submit();
			ReleaseOutputLeaseCallback(event.release_output_lease);
			SetSlotError(slot, "udf output consumer is missing accept_event callback");
			return true;
		}

		if (event_kind == UDFOutputEventKind::DATA && !has_registered_output_consumer) {
			bool capacity_available = true;
			if (consumer.data_capacity) {
				ScopedGILReleaseIfHeld release_gil;
				capacity_available = consumer.data_capacity() > 0;
			}
			if (!capacity_available) {
				release_terminal_submit();
				ReleaseOutputLeaseCallback(event.release_output_lease);
				SetSlotError(slot, "udf output consumer capacity contract was violated");
				return false;
			}
		}

		auto release_on_consumer_error = event.release_output_lease;
		try {
			// Registered streaming consumers can wake the downstream source
			// synchronously from accept_event(). Release submit capacity first
			// so the awakened source does not immediately block on stale
			// inflight accounting.
			if (has_registered_output_consumer) {
				release_terminal_submit();
			}
			ScopedGILReleaseIfHeld release_gil;
			consumer.accept_event(std::move(event));
		} catch (const std::exception &ex) {
			release_terminal_submit();
			ReleaseOutputLeaseCallback(release_on_consumer_error);
			SetSlotError(slot, ex.what());
			return true;
		}

		release_terminal_submit();
		if (event_kind == UDFOutputEventKind::ERROR) {
			SetSlotError(slot, error_for_log.empty() ? "udf async error" : error_for_log);
			return true;
		}
		MaybeMarkSlotFinished(slot);
		if (!has_registered_output_consumer) {
			WakeupPipelineThread(slot);
		}
		if (event_kind != UDFOutputEventKind::DATA) {
			return true;
		}
		return true;
	}

	bool TryDeliverResult(ExecutorSlot &slot, UDFResult &&result) {
		UDFOutputEvent event;
		event.kind = UDFOutputEventKind::DATA;
		event.submit_id = result.submit_id;
		event.outputs = std::move(result.outputs);
		event.ref_outputs = std::move(result.ref_outputs);
		event.rows = std::move(result.rows);
		event.submit_complete = result.submit_complete;
		event.handoff_output_lease = std::move(result.handoff_output_lease);
		event.release_output_lease = std::move(result.release_output_lease);
		return DeliverOutputEvent(slot, std::move(event));
	}

	const vector<LogicalType> &ExpectedDirectOutputTypes(ExecutorSlot &slot, idx_t submit_id) {
		if (UDFModePreservesRows(ClassifyUDFMode(slot.payload)) &&
		    slot.expected_ref_output_types.size() > slot.expected_output_types.size()) {
			auto row_entry = slot.rows_by_submit_id.find(submit_id);
			if (row_entry != slot.rows_by_submit_id.end() && row_entry->second &&
			    row_entry->second->ColumnCount() == 0) {
				return slot.expected_ref_output_types;
			}
		}
		return slot.expected_output_types;
	}

	bool DrainSyncResults(ExecutorSlot &slot) {
		// Caller holds GIL. Drain at most one queued result per dispatcher
		// iteration; Python executors wake the dispatcher when new items arrive.
		auto raw_result = slot.py_executor->obj.attr("take_ready_result")();
		if (raw_result.is_none()) {
			return false;
		}

		auto ready_result = DecodePythonReadyResult(raw_result);
		auto &result = ready_result.output;
		idx_t submit_id = ready_result.submit_id;
		if (submit_id == 0) {
			SetSlotError(slot, "udf async result queue: missing submit_id (submit/result mismatch)");
			return true;
		}

		if (IsPythonExceptionObject(result)) {
			auto msg = py::str(result).cast<string>();
			if (ready_result.submit_complete) {
				ReleaseTerminalSubmit(slot, submit_id);
				MaybeMarkSlotFinished(slot);
			}
			SetSlotError(slot, msg);
			return true;
		}

		if (slot.rows_by_submit_id.find(submit_id) == slot.rows_by_submit_id.end()) {
			SetSlotError(slot, "udf async result queue: submit_id rows missing (submit/result mismatch)");
			return true;
		}

		unique_ptr<DataChunk> outputs_chunk;
		unique_ptr<LazyRefDataChunk> ref_outputs;
		bool none_result = result.is_none();
		if (none_result) {
			outputs_chunk = MakeEmptyDataChunk();
		} else if (IsPythonRefBundleResult(result)) {
			if (RequiresDistributedRefBundleOutput(slot.payload)) {
				g_udf_distributed_ref_bundle_data_events.fetch_add(1, std::memory_order_relaxed);
			}
			ref_outputs = ConvertPythonRefBundleResult(result, slot.expected_ref_output_types);
		} else if (RequiresDistributedRefBundleOutput(slot.payload)) {
			g_udf_distributed_direct_table_rejected_events.fetch_add(1, std::memory_order_relaxed);
			ReleaseTerminalSubmit(slot, submit_id);
			SetSlotError(slot, DistributedRefBundleContractError(slot.payload, submit_id, result));
			return true;
		} else {
			g_udf_direct_output_arrow_table_conversion_count.fetch_add(1, std::memory_order_relaxed);
			outputs_chunk = ConvertArrowTableToDataChunk(result, RequireActiveSlotContext(slot),
			                                             ExpectedDirectOutputTypes(slot, submit_id));
		}

		unique_ptr<DataChunk> rows_chunk;
		if (slot.is_table_udf) {
			if (ready_result.submit_complete) {
				auto erased = slot.rows_by_submit_id.erase(submit_id);
				if (erased == 0) {
					SetSlotError(slot, "udf async result queue: submit_id rows missing (submit/result mismatch)");
					return true;
				}
			}
			rows_chunk = MakeEmptyDataChunk();
		} else {
			if (!ready_result.submit_complete) {
				SetSlotError(slot, "udf async result queue: partial non-table results are not supported");
				return true;
			}
			auto row_entry = slot.rows_by_submit_id.find(submit_id);
			if (row_entry == slot.rows_by_submit_id.end()) {
				SetSlotError(slot, "udf async result queue: submit_id rows missing (submit/result mismatch)");
				return true;
			}
			rows_chunk = std::move(row_entry->second);
			slot.rows_by_submit_id.erase(row_entry);
		}

		UDFResult class_result;
		class_result.outputs = std::move(outputs_chunk);
		class_result.ref_outputs = std::move(ref_outputs);
		class_result.rows = std::move(rows_chunk);
		class_result.submit_complete = ready_result.submit_complete;
		class_result.submit_id = submit_id;
		if (!TryDeliverResult(slot, std::move(class_result))) {
			SetSlotError(slot, "udf async result queue: output consumer rejected capacity-aware delivery");
			return true;
		}
		return true;
	}

	bool DrainSyncSlotSafely(ExecutorSlot &slot) {
		auto old_error = slot.has_error.load();
		try {
			return DrainSyncResults(slot) || slot.has_error.load() != old_error;
		} catch (const py::error_already_set &ex) {
			SetSlotError(slot, StringUtil::Format("udf async collector failed: %s", ex.what()));
			return true;
		} catch (const std::exception &ex) {
			SetSlotError(slot, StringUtil::Format("udf async result drain failed: %s", ex.what()));
			return true;
		}
	}

	void DoSubmit_WithGIL(ExecutorSlot &slot, DispatcherSubmitTask &task) {
		try {
			// Wrap pre-computed Arrow C data into PyArrow Table (caller holds GIL)
			if (task.arrow_schemas.size() != task.arrow_arrays.size() || task.arrow_arrays.empty()) {
				throw InternalException("udf submit task has invalid Arrow batch envelope");
			}
			py::list batches;
			for (idx_t batch_idx = 0; batch_idx < task.arrow_arrays.size(); batch_idx++) {
				TransformDuckToArrowChunk(task.arrow_schemas[batch_idx]->arrow_schema,
				                          task.arrow_arrays[batch_idx]->arrow_array, batches);
			}
			auto py_args = pyarrow::ToArrowTable(task.types, task.names, batches, task.options);

			// submit_with_id() lets Python async executors return out-of-order while
			// preserving the C++ submit_id -> input rows association.
			py::object ref = slot.py_executor->obj.attr("submit_with_id")(py::int_(task.submit_id), py_args);
			ConsumeTaskAdmissionReservation(slot);

			TrackSubmittedPythonRef_WithGIL(slot, task.submit_id, ref, std::move(task.rows));
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("udf submit failed: %s", ex.what());
		}
	}

	py::tuple BuildPythonRefBundleArgs_WithGIL(LazyRefDataChunk &bundle) {
		py::list refs;
		py::list slices;
		py::list metadata;
		py::list names;
		for (auto &name : bundle.names) {
			names.append(py::str(name));
		}
		for (auto &block : bundle.blocks) {
			refs.append(BorrowPythonObjectHolder(block.object_ref));
			if (block.has_slice) {
				slices.append(py::make_tuple(block.slice.start_offset, block.slice.end_offset));
			} else {
				slices.append(py::none());
			}
			py::dict meta;
			meta[py::str("num_rows")] = py::int_(block.metadata.num_rows);
			meta[py::str("size_bytes")] = py::int_(block.metadata.size_bytes);
			if (!block.metadata.query_id.empty()) {
				meta[py::str("query_id")] = py::str(block.metadata.query_id);
			}
			if (!block.metadata.operator_id.empty()) {
				meta[py::str("operator_id")] = py::str(block.metadata.operator_id);
			}
			if (!block.metadata.attempt_id.empty()) {
				meta[py::str("attempt_id")] = py::str(block.metadata.attempt_id);
			}
			if (!block.column_ids.empty()) {
				py::list column_ids;
				for (auto column_id : block.column_ids) {
					column_ids.append(py::int_(column_id));
				}
				meta[py::str("column_ids")] = std::move(column_ids);
			}
			metadata.append(std::move(meta));
		}
		return py::make_tuple(std::move(refs), std::move(slices), std::move(metadata), std::move(names));
	}

	void TrackSubmittedPythonRef_WithGIL(ExecutorSlot &slot, idx_t submit_id, py::object &ref,
	                                     unique_ptr<DataChunk> rows, bool require_generator_ref = false) {
		if (ref.is_none()) {
			auto inflight_before = slot.inflight_count.load();
			slot.inflight_count++;
			slot.rows_by_submit_id.emplace(submit_id, std::move(rows));
			UDFDebugLog(StringUtil::Format(
			    "track_event_queue slot=%llu submit=%llu require_generator=%s inflight_before=%lld inflight_after=%lld",
			    static_cast<unsigned long long>(slot.id), static_cast<unsigned long long>(submit_id),
			    require_generator_ref ? "true" : "false", static_cast<long long>(inflight_before),
			    static_cast<long long>(slot.inflight_count.load())));
			return;
		}
		if (!async_collector) {
			throw InvalidInputException("udf async collector is unavailable");
		}
		py::object error_context = py::none();
		if (py::hasattr(slot.py_executor->obj, "error_context")) {
			error_context = slot.py_executor->obj.attr("error_context")();
		}
		async_collector->obj.attr("track_generator_ref")(py::int_(slot.id), py::int_(submit_id), ref, error_context);
		auto inflight_before = slot.inflight_count.load();
		slot.inflight_count++;
		slot.rows_by_submit_id.emplace(submit_id, std::move(rows));
		UDFDebugLog(StringUtil::Format(
		    "track_generator slot=%llu submit=%llu require_generator=%s inflight_before=%lld inflight_after=%lld",
		    static_cast<unsigned long long>(slot.id), static_cast<unsigned long long>(submit_id),
		    require_generator_ref ? "true" : "false", static_cast<long long>(inflight_before),
		    static_cast<long long>(slot.inflight_count.load())));
	}

	void DoSubmitRefBundle_WithGIL(ExecutorSlot &slot, DispatcherRefSubmitTask &task) {
		try {
			if (!task.bundle) {
				throw InvalidInputException("udf ref bundle submit received null bundle");
			}
			auto bundle_args = BuildPythonRefBundleArgs_WithGIL(*task.bundle);
			auto refs = bundle_args[0];
			auto slices = bundle_args[1];
			auto metadata = bundle_args[2];
			auto names = bundle_args[3];

			py::object ref = slot.py_executor->obj.attr("submit_ref_bundle_with_id")(py::int_(task.submit_id), refs,
			                                                                         slices, metadata, names);
			ConsumeTaskAdmissionReservation(slot);
			auto inserted = slot.ref_inputs_by_submit_id.emplace(task.submit_id, std::move(task.bundle));
			if (!inserted.second) {
				throw InternalException("duplicate retained ref-bundle submit id");
			}

			TrackSubmittedPythonRef_WithGIL(slot, task.submit_id, ref, std::move(task.rows), true);
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("udf ref bundle submit failed: %s", ex.what());
		}
	}

	bool DrainAsyncResults_WithGIL(vector<ExecutorSlot *> &active) {
		bool did_work = false;
		std::unordered_map<uint64_t, ExecutorSlot *> active_by_id;
		active_by_id.reserve(active.size());
		for (auto *slot : active) {
			if (slot && !slot->pending_shutdown.load()) {
				active_by_id.emplace(slot->id, slot);
			}
		}
		auto fail_active_slots = [&](const string &msg) {
			for (auto *slot : active) {
				if (!slot || !slot->py_executor || slot->abort_requested.load() || slot->pending_shutdown.load()) {
					continue;
				}
				SetSlotError(*slot, msg);
			}
			did_work = true;
		};
		auto handle_output = [&](ExecutorSlot &slot, idx_t submit_id, const py::handle &output) -> bool {
			auto ready_result = DecodePythonReadyResult(output);
			if (ready_result.submit_id > 0) {
				submit_id = ready_result.submit_id;
			}
			if (submit_id == 0) {
				SetSlotError(slot, "udf async: missing submit_id (submit/result mismatch)");
				did_work = true;
				return true;
			}
			auto &output_obj = ready_result.output;
			// Check if result is an exception
			if (IsPythonExceptionObject(output_obj)) {
				auto msg = py::str(output_obj).cast<string>();
				if (ready_result.submit_complete) {
					ReleaseTerminalSubmit(slot, submit_id);
				}
				SetSlotError(slot, msg);
				did_work = true;
				return true;
			}

			// Convert Arrow result to DataChunk
			if (output_obj.is_none()) {
				unique_ptr<DataChunk> rows_chunk;
				if (slot.is_table_udf) {
					if (ready_result.submit_complete) {
						auto erased = slot.rows_by_submit_id.erase(submit_id);
						if (erased == 0) {
							SetSlotError(slot, "udf async: submit_id rows missing (submit/result mismatch)");
							return true;
						}
					}
					rows_chunk = MakeEmptyDataChunk();
				} else {
					if (!ready_result.submit_complete) {
						SetSlotError(slot, "udf async: partial non-table results are not supported");
						return true;
					}
					auto row_entry = slot.rows_by_submit_id.find(submit_id);
					if (row_entry == slot.rows_by_submit_id.end()) {
						SetSlotError(slot, "udf async: submit_id rows missing (submit/result mismatch)");
						return true;
					}
					rows_chunk = std::move(row_entry->second);
					slot.rows_by_submit_id.erase(row_entry);
				}
				auto empty_outputs = MakeEmptyDataChunk();
				UDFResult empty_result;
				empty_result.outputs = std::move(empty_outputs);
				empty_result.rows = std::move(rows_chunk);
				empty_result.submit_complete = ready_result.submit_complete;
				empty_result.submit_id = submit_id;
				if (!TryDeliverResult(slot, std::move(empty_result))) {
					SetSlotError(slot, "udf async: output consumer rejected capacity-aware delivery");
					return false;
				}
				did_work = true;
				return true;
			}

			unique_ptr<DataChunk> outputs_chunk;
			unique_ptr<LazyRefDataChunk> ref_outputs;
			if (IsPythonRefBundleResult(output_obj)) {
				if (RequiresDistributedRefBundleOutput(slot.payload)) {
					g_udf_distributed_ref_bundle_data_events.fetch_add(1, std::memory_order_relaxed);
				}
				ref_outputs = ConvertPythonRefBundleResult(output_obj, slot.expected_ref_output_types);
			} else if (RequiresDistributedRefBundleOutput(slot.payload)) {
				g_udf_distributed_direct_table_rejected_events.fetch_add(1, std::memory_order_relaxed);
				ReleaseTerminalSubmit(slot, submit_id);
				SetSlotError(slot, DistributedRefBundleContractError(slot.payload, submit_id, output_obj));
				return true;
			} else {
				g_udf_direct_output_arrow_table_conversion_count.fetch_add(1, std::memory_order_relaxed);
				outputs_chunk = ConvertArrowTableToDataChunk(output_obj, RequireActiveSlotContext(slot),
				                                             ExpectedDirectOutputTypes(slot, submit_id));
			}
			unique_ptr<DataChunk> rows_chunk;
			if (slot.is_table_udf) {
				if (ready_result.submit_complete) {
					auto erased = slot.rows_by_submit_id.erase(submit_id);
					if (erased == 0) {
						ReleaseTerminalSubmit(slot, submit_id);
						SetSlotError(slot, "udf async: submit_id rows missing (submit/result mismatch)");
						return true;
					}
				}
				rows_chunk = MakeEmptyDataChunk();
			} else {
				if (!ready_result.submit_complete) {
					ReleaseTerminalSubmit(slot, submit_id);
					SetSlotError(slot, "udf async: partial non-table results are not supported");
					return true;
				}
				auto row_entry = slot.rows_by_submit_id.find(submit_id);
				if (row_entry == slot.rows_by_submit_id.end()) {
					ReleaseTerminalSubmit(slot, submit_id);
					SetSlotError(slot, "udf async: submit_id rows missing (submit/result mismatch)");
					return true;
				}
				rows_chunk = std::move(row_entry->second);
				slot.rows_by_submit_id.erase(row_entry);
			}

			UDFResult result;
			result.outputs = std::move(outputs_chunk);
			result.ref_outputs = std::move(ref_outputs);
			result.rows = std::move(rows_chunk);
			result.submit_complete = ready_result.submit_complete;
			result.submit_id = submit_id;
			if (!TryDeliverResult(slot, std::move(result))) {
				SetSlotError(slot, "udf async: output consumer rejected capacity-aware delivery");
				return false;
			}
			did_work = true;
			return true;
		};

		if (async_collector) {
			try {
				py::dict capacities;
				idx_t debug_capacity_total = 0;
				idx_t debug_capacity_positive_slots = 0;
				idx_t debug_inflight_total = 0;
				idx_t debug_active_slots = 0;
				auto debug_probe_tick = g_udf_debug_drain_tick.load(std::memory_order_relaxed);
				bool debug_probe_slots = UDFDebugEnabled() && (debug_probe_tick >= 650 || debug_probe_tick % 200 == 0);
				for (auto *slot : active) {
					if (!slot || slot->abort_requested.load() || slot->pending_shutdown.load()) {
						continue;
					}
					const bool has_inflight = slot->inflight_count.load() > 0;
					if (!has_inflight) {
						continue;
					}
					debug_active_slots++;
					debug_inflight_total += static_cast<idx_t>(slot->inflight_count.load());
					if (debug_probe_slots) {
						UDFDebugLog(StringUtil::Format(
						    "capacity_probe_begin drain_tick=%llu slot=%llu inflight=%lld has_consumer=%s "
						    "pending_shutdown=%s",
						    static_cast<unsigned long long>(debug_probe_tick),
						    static_cast<unsigned long long>(slot->id),
						    static_cast<long long>(slot->inflight_count.load()),
						    slot->has_output_consumer ? "true" : "false",
						    slot->pending_shutdown.load() ? "true" : "false"));
					}
					auto capacity = GetSlotResultCapacity(*slot);
					if (debug_probe_slots) {
						UDFDebugLog(StringUtil::Format(
						    "capacity_probe_end drain_tick=%llu slot=%llu rows=%llu bytes=%llu "
						    "item_bytes=%llu has_bytes=%s has_item_bytes=%s",
						    static_cast<unsigned long long>(debug_probe_tick),
						    static_cast<unsigned long long>(slot->id), static_cast<unsigned long long>(capacity.rows),
						    static_cast<unsigned long long>(capacity.bytes),
						    static_cast<unsigned long long>(capacity.item_bytes),
						    capacity.has_byte_capacity ? "true" : "false",
						    capacity.has_item_byte_capacity ? "true" : "false"));
					}
					debug_capacity_total += capacity.rows;
					if (capacity.rows > 0) {
						debug_capacity_positive_slots++;
					}
					py::dict capacity_obj;
					capacity_obj[py::str("rows")] = py::int_(capacity.rows);
					if (capacity.has_byte_capacity) {
						capacity_obj[py::str("bytes")] = py::int_(capacity.bytes);
					}
					if (capacity.has_item_byte_capacity) {
						capacity_obj[py::str("item_bytes")] = py::int_(capacity.item_bytes);
					}
					capacities[py::int_(slot->id)] = std::move(capacity_obj);
				}

				auto results_list = async_collector->obj.attr("drain_results")(capacities);
				auto results = results_list.cast<py::list>();
				auto debug_result_count = static_cast<idx_t>(py::len(results));
				auto debug_tick = g_udf_debug_drain_tick.fetch_add(1, std::memory_order_relaxed) + 1;
				if (UDFDebugEnabled() && (debug_result_count > 0 || debug_tick % 200 == 0)) {
					UDFDebugLog(StringUtil::Format(
					    "drain_results tick=%llu active_slots=%llu inflight_total=%llu capacity_total=%llu "
					    "capacity_positive_slots=%llu result_count=%llu",
					    static_cast<unsigned long long>(debug_tick),
					    static_cast<unsigned long long>(debug_active_slots),
					    static_cast<unsigned long long>(debug_inflight_total),
					    static_cast<unsigned long long>(debug_capacity_total),
					    static_cast<unsigned long long>(debug_capacity_positive_slots),
					    static_cast<unsigned long long>(debug_result_count)));
				}

				for (auto &item : results) {
					ExecutorSlot *slot = nullptr;
					try {
						auto tup = item.cast<py::tuple>();
						uint64_t slot_id = 0;
						idx_t submit_id = 0;
						string event_kind;
						string output_request_id;
						string output_lease_id;
						py::object payload;
						auto tuple_len = py::len(tup);
						if (tuple_len == 4 || tuple_len == 6) {
							slot_id = tup[0].cast<uint64_t>();
							submit_id = tup[1].cast<idx_t>();
							event_kind = tup[2].cast<string>();
							payload = py::reinterpret_borrow<py::object>(tup[3]);
							if (tuple_len == 6) {
								if (event_kind != "data") {
									throw InvalidInputException(
									    "udf async collector returned lease identity for non-data event");
								}
								output_request_id = tup[4].cast<string>();
								output_lease_id = tup[5].cast<string>();
							}
						} else {
							throw InvalidInputException("udf async collector returned invalid result tuple");
						}
						std::function<void()> handoff_output_lease;
						std::function<void()> release_output_lease;
						if (event_kind == "data") {
							if (tuple_len != 6 || output_request_id.empty() || output_lease_id.empty()) {
								throw InvalidInputException(
								    "udf async collector data event is missing output lease identity");
							}
							auto callbacks = MakeCollectorOutputLeaseCallbacks(async_collector->obj, output_request_id,
							                                                   output_lease_id);
							handoff_output_lease = std::move(callbacks.handoff);
							release_output_lease = std::move(callbacks.release);
						}
						auto slot_entry = active_by_id.find(slot_id);
						if (slot_entry != active_by_id.end()) {
							slot = slot_entry->second;
						}
						if (!slot) {
							if (release_output_lease) {
								release_output_lease();
							}
							continue;
						}
						if (slot->abort_requested.load() || slot->pending_shutdown.load()) {
							if (release_output_lease) {
								release_output_lease();
							}
							did_work = true;
							continue;
						}

						UDFOutputConsumer output_consumer;
						if (GetOutputConsumer(*slot, output_consumer)) {
							UDFOutputEvent event;
							event.submit_id = submit_id;
							event.submit_complete = false;
							if (event_kind == "data") {
								event.kind = UDFOutputEventKind::DATA;
								try {
									if (!IsPythonRefBundleResult(payload)) {
										g_udf_distributed_direct_table_rejected_events.fetch_add(
										    1, std::memory_order_relaxed);
										event.kind = UDFOutputEventKind::ERROR;
										event.submit_complete = true;
										event.error =
										    DistributedRefBundleContractError(slot->payload, submit_id, payload);
										if (release_output_lease) {
											release_output_lease();
											release_output_lease = nullptr;
										}
									} else {
										g_udf_distributed_ref_bundle_data_events.fetch_add(1,
										                                                   std::memory_order_relaxed);
										event.ref_outputs =
										    ConvertPythonRefBundleResult(payload, slot->expected_ref_output_types);
										event.handoff_output_lease = std::move(handoff_output_lease);
										event.release_output_lease = std::move(release_output_lease);
									}
								} catch (const std::exception &ex) {
									if (release_output_lease) {
										release_output_lease();
										release_output_lease = nullptr;
									}
									event.kind = UDFOutputEventKind::ERROR;
									event.submit_complete = true;
									event.error = ex.what();
								}
							} else if (event_kind == "complete") {
								event.kind = UDFOutputEventKind::COMPLETE;
								event.submit_complete = true;
							} else if (event_kind == "error") {
								event.kind = UDFOutputEventKind::ERROR;
								event.submit_complete = true;
								event.error = py::str(payload).cast<string>();
							} else {
								event.kind = UDFOutputEventKind::ERROR;
								event.submit_complete = true;
								event.error = StringUtil::Format(
								    "udf async collector returned unknown output event '%s'", event_kind);
							}
							if (!DeliverOutputEvent(*slot, std::move(event))) {
								SetSlotError(*slot, "udf async: output consumer rejected capacity-aware delivery");
							}
							did_work = true;
							continue;
						}

						if (event_kind == "data") {
							unique_ptr<DataChunk> outputs_chunk;
							unique_ptr<LazyRefDataChunk> ref_outputs;
							if (IsPythonRefBundleResult(payload)) {
								if (RequiresDistributedRefBundleOutput(slot->payload)) {
									g_udf_distributed_ref_bundle_data_events.fetch_add(1, std::memory_order_relaxed);
								}
								ref_outputs = ConvertPythonRefBundleResult(payload, slot->expected_ref_output_types);
							} else if (RequiresDistributedRefBundleOutput(slot->payload)) {
								g_udf_distributed_direct_table_rejected_events.fetch_add(1, std::memory_order_relaxed);
								if (release_output_lease) {
									release_output_lease();
									release_output_lease = nullptr;
								}
								UDFOutputEvent event;
								event.kind = UDFOutputEventKind::ERROR;
								event.submit_id = submit_id;
								event.submit_complete = true;
								event.error = DistributedRefBundleContractError(slot->payload, submit_id, payload);
								DeliverOutputEvent(*slot, std::move(event));
								did_work = true;
								continue;
							} else {
								g_udf_direct_output_arrow_table_conversion_count.fetch_add(1,
								                                                           std::memory_order_relaxed);
								outputs_chunk =
								    ConvertArrowTableToDataChunk(payload, RequireActiveSlotContext(*slot),
								                                 ExpectedDirectOutputTypes(*slot, submit_id));
							}
							unique_ptr<DataChunk> rows_chunk;
							if (slot->is_table_udf) {
								rows_chunk = MakeEmptyDataChunk();
							} else {
								auto row_entry = slot->rows_by_submit_id.find(submit_id);
								if (row_entry == slot->rows_by_submit_id.end()) {
									if (release_output_lease) {
										release_output_lease();
										release_output_lease = nullptr;
									}
									SetSlotError(*slot, "udf async: submit_id rows missing (submit/result mismatch)");
									did_work = true;
									continue;
								}
								if (!row_entry->second) {
									if (release_output_lease) {
										release_output_lease();
										release_output_lease = nullptr;
									}
									SetSlotError(*slot, "udf async: non-table generator produced multiple data events");
									did_work = true;
									continue;
								}
								rows_chunk = std::move(row_entry->second);
							}

							UDFResult result;
							result.outputs = std::move(outputs_chunk);
							result.ref_outputs = std::move(ref_outputs);
							result.rows = std::move(rows_chunk);
							result.submit_complete = false;
							result.submit_id = submit_id;
							result.handoff_output_lease = std::move(handoff_output_lease);
							result.release_output_lease = std::move(release_output_lease);
							if (!TryDeliverResult(*slot, std::move(result))) {
								SetSlotError(*slot, "udf async: output consumer rejected capacity-aware delivery");
							}
							did_work = true;
							continue;
						}
						if (event_kind == "complete") {
							QueueTerminalControlResult(*slot, submit_id);
							did_work = true;
							continue;
						}
						if (event_kind == "error") {
							UDFOutputEvent event;
							event.kind = UDFOutputEventKind::ERROR;
							event.submit_id = submit_id;
							event.submit_complete = true;
							event.error = py::str(payload).cast<string>();
							DeliverOutputEvent(*slot, std::move(event));
							did_work = true;
							continue;
						}

						SetSlotError(*slot, StringUtil::Format("udf async collector returned unknown output event '%s'",
						                                       event_kind));
						did_work = true;
					} catch (const py::error_already_set &ex) {
						if (slot && !slot->abort_requested.load()) {
							SetSlotError(*slot, StringUtil::Format("udf async collector result failed: %s", ex.what()));
						} else {
							fail_active_slots(StringUtil::Format("udf async collector failed: %s", ex.what()));
						}
						did_work = true;
					} catch (const std::exception &ex) {
						if (slot && !slot->abort_requested.load()) {
							SetSlotError(*slot, StringUtil::Format("udf async collector result failed: %s", ex.what()));
						} else {
							fail_active_slots(StringUtil::Format("udf async result drain failed: %s", ex.what()));
						}
						did_work = true;
					}
				}

				for (auto *slot : active) {
					if (!slot || slot->abort_requested.load() || slot->pending_shutdown.load() || !slot->py_executor ||
					    slot->inflight_count.load() <= 0) {
						continue;
					}
					if (!ShouldDrainSyncResults(*slot)) {
						continue;
					}
					if (DrainSyncSlotSafely(*slot)) {
						did_work = true;
					}
				}
			} catch (const py::error_already_set &ex) {
				fail_active_slots(StringUtil::Format("udf async collector failed: %s", ex.what()));
			} catch (const std::exception &ex) {
				fail_active_slots(StringUtil::Format("udf async result drain failed: %s", ex.what()));
			}
		}

		if (!async_collector) {
			for (auto *slot : active) {
				if (!slot || slot->abort_requested.load() || slot->pending_shutdown.load() || !slot->py_executor ||
				    slot->inflight_count.load() <= 0) {
					continue;
				}
				if (!ShouldDrainSyncResults(*slot)) {
					continue;
				}
				if (DrainSyncSlotSafely(*slot)) {
					did_work = true;
				}
			}
		}

		return did_work;
	}

	void DoFinishedSubmitting_WithGIL(ExecutorSlot &slot) {
		// Caller holds GIL
		try {
			slot.py_executor->obj.attr("finished_submitting")();
			slot.finished_submitting_acked.store(true);
			MaybeMarkSlotFinished(slot);
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("udf finished_submitting failed: %s", ex.what());
		}
	}

	void SetSlotError(ExecutorSlot &slot, const string &msg) {
		bool should_notify = false;
		{
			lock_guard<mutex> eg(slot.error_lock);
			if (!slot.has_error.load()) {
				slot.error = msg;
				slot.has_error.store(true);
				should_notify = true;
			}
		}
		AbortSlotForError(slot);
		if (!should_notify) {
			return;
		}
		WakeupPipelineThread(slot);
		UDFOutputConsumer output_consumer;
		if (GetOutputConsumer(slot, output_consumer) && output_consumer.accept_error) {
			try {
				ScopedGILReleaseIfHeld release_gil;
				output_consumer.accept_error(msg);
			} catch (const std::exception &ex) {
				lock_guard<mutex> eg(slot.error_lock);
				slot.error += StringUtil::Format("; udf output consumer accept_error failed: %s", ex.what());
			}
		}
	}

	void NotifySlotFinished(ExecutorSlot &slot) {
		WakeupPipelineThread(slot);
		UDFOutputConsumer output_consumer;
		if (GetOutputConsumer(slot, output_consumer) && output_consumer.notify_finished) {
			try {
				ScopedGILReleaseIfHeld release_gil;
				output_consumer.notify_finished();
			} catch (const std::exception &ex) {
				SetSlotError(slot, StringUtil::Format("udf output consumer notify_finished failed: %s", ex.what()));
			}
		}
	}

	void RecordWakeupFailure(ExecutorSlot &slot, const string &msg) {
		bool first_error = false;
		{
			lock_guard<mutex> eg(slot.error_lock);
			if (!slot.has_error.load()) {
				slot.error = msg;
				slot.has_error.store(true);
				first_error = true;
			}
		}
		if (first_error) {
			AbortSlotForError(slot);
		}
		RecordDispatcherError(msg);
	}

	void WakeupPipelineThread(ExecutorSlot &slot) {
		// Fire the interrupt callback for operators that returned TASK_BLOCKED.
		// The callback is one-shot so a stale callback cannot reschedule an
		// already-running task.
		std::function<void()> callback;
		InterruptState interrupt_state;
		bool should_callback = false;
		{
			lock_guard<mutex> wg(slot.wakeup_lock);
			if (slot.has_interrupt_state) {
				interrupt_state = slot.interrupt_state;
				slot.has_interrupt_state = false;
				should_callback = true;
			}
			callback = slot.wakeup_callback;
		}
		if (should_callback) {
			try {
				interrupt_state.Callback();
			} catch (const std::exception &ex) {
				RecordWakeupFailure(slot, StringUtil::Format("udf interrupt callback failed: %s", ex.what()));
			}
		}
		if (callback) {
			try {
				ScopedGILReleaseIfHeld release_gil;
				callback();
			} catch (const std::exception &ex) {
				RecordWakeupFailure(slot, StringUtil::Format("udf wakeup callback failed: %s", ex.what()));
			}
		}
	}

	// ── Helpers (caller holds GIL) ───────────────────────────────────────

	void EnsurePythonExecutor_WithGIL(ExecutorSlot &slot) {
		if (slot.py_executor_initialized) {
			return;
		}
		ClientProperties cached_options;
		{
			py::gil_scoped_release release;
			ScopedClientContextLock ctx_g;
			cached_options = RequireActiveSlotContext(slot).GetClientProperties();
		}
		py::object payload_obj = PythonObject::FromValue(slot.payload, slot.payload.type(), cached_options);

		py::object module;
		try {
			module = py::module_::import("vane.execution.unified_executor");
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("Failed to import vane.execution.unified_executor: %s", ex.what());
		}
		try {
			py::object options = py::none();
			if (slot.actor_handles) {
				auto *boxed_options = static_cast<py::object *>(slot.actor_handles.get());
				if (!py::isinstance<py::dict>(*boxed_options)) {
					throw InvalidInputException(
					    "udf executor options must be a dict; actor_handles list path has been removed");
				}
				options = py::reinterpret_borrow<py::dict>(*boxed_options);
			}
			py::object executor_obj = module.attr("build_unified_executor")(payload_obj, options);
			slot.py_executor = make_uniq<RegisteredObject>(std::move(executor_obj));
			auto wakeup_fn = py::cpp_function([this]() {
				wakeup_fired.store(true);
				work_cv.notify_one();
			});
			slot.py_executor->obj.attr("register_wakeup")(wakeup_fn);
			slot.py_executor_initialized = true;
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("Failed to build udf executor: %s", ex.what());
		}
	}

	void EnsureAsyncCollector() {
		if (async_collector) {
			return;
		}
		// Caller holds GIL
		try {
			auto module = py::module_::import("vane.execution.udf_stream_result_collector");
			auto collector_obj = module.attr("AsyncResultCollector")();
			// Wire wakeup callback: when async collector completes an await,
			// it calls this lambda which signals work_cv so the dispatcher
			// wakes up immediately to drain results (no GIL spin).
			auto wakeup_fn = py::cpp_function([this]() {
				wakeup_fired.store(true);
				work_cv.notify_one();
			});
			collector_obj.attr("set_wakeup_callback")(wakeup_fn);
			async_collector = make_uniq<RegisteredObject>(std::move(collector_obj));
		} catch (const py::error_already_set &ex) {
			throw InvalidInputException("udf async collector initialization failed: %s", ex.what());
		} catch (const std::exception &ex) {
			throw InvalidInputException("udf async collector initialization failed: %s", ex.what());
		}
	}

	// ── Members ──────────────────────────────────────────────────────────
	mutex global_lock;
	std::condition_variable work_cv;
	std::unordered_map<uint64_t, shared_ptr<ExecutorSlot>> slots;
	atomic<uint64_t> next_slot_id {1};
	std::thread dispatcher_thread;
	atomic<bool> stop {false};
	atomic<bool> wakeup_fired {false};           // set by async collector wakeup callback
	atomic<bool> work_pending {false};           // set by all notify paths before signaling work_cv
	atomic<bool> pending_async_shutdown {false}; // deferred async_collector shutdown
	uint64_t slot_generation = 0;
	uint64_t pending_async_shutdown_generation = 0;
	mutex async_shutdown_lock;
	std::condition_variable async_shutdown_cv;
	bool async_shutdown_complete {true};
	mutex dispatcher_error_lock;
	atomic<bool> has_dispatcher_error {false};
	string dispatcher_error;
	bool thread_running = false;
	mutex output_lease_release_lock;
	std::deque<std::function<void()>> output_lease_releases;
	mutex deferred_wakeup_lock;
	std::deque<std::function<void()>> deferred_wakeups;
	unique_ptr<RegisteredObject> async_collector; // Python AsyncResultCollector
};

struct CollectorOutputLeaseCallbackState {
	mutex lock;
	bool handed_off = false;
	bool released = false;
};

static CollectorOutputLeaseCallbacks
MakeCollectorOutputLeaseCallbacks(const py::object &collector, const string &request_id, const string &lease_id) {
	auto collector_holder = MakePythonObjectHolder(py::reinterpret_borrow<py::object>(collector));
	auto state = std::make_shared<CollectorOutputLeaseCallbackState>();
	CollectorOutputLeaseCallbacks callbacks;
	callbacks.handoff = [collector_holder, state, request_id, lease_id]() {
		{
			lock_guard<mutex> guard(state->lock);
			if (state->released || state->handed_off) {
				return;
			}
			state->handed_off = true;
		}
		if (!PythonRuntimeUsableForUDFTeardown()) {
			return;
		}
		GlobalPythonDispatcher::Instance().EnqueueOutputLeaseRelease([collector_holder, request_id, lease_id]() {
			auto collector_obj = BorrowPythonObjectHolder(collector_holder);
			if (collector_obj.is_none()) {
				return;
			}
			collector_obj.attr("handoff_output_block_lease")(py::str(request_id), py::str(lease_id));
		});
	};
	callbacks.release = [collector_holder, state, request_id, lease_id]() {
		{
			lock_guard<mutex> guard(state->lock);
			if (state->released) {
				return;
			}
			state->released = true;
		}
		if (!PythonRuntimeUsableForUDFTeardown()) {
			return;
		}
		GlobalPythonDispatcher::Instance().EnqueueOutputLeaseRelease([collector_holder, request_id, lease_id]() {
			auto collector_obj = BorrowPythonObjectHolder(collector_holder);
			if (collector_obj.is_none()) {
				return;
			}
			collector_obj.attr("release_output_block_lease")(py::str(request_id), py::str(lease_id));
		});
	};
	return callbacks;
}

// ─── DispatchedUDFPythonExecutor ────────────────────────────────────────
//
// Thin UDFExecutor wrapper.  Pipeline threads interact with this class.
// It NEVER acquires the GIL — all Python work is delegated to the global
// dispatcher thread via per-slot command/result queues.

class DispatchedUDFPythonExecutor : public UDFExecutor {
public:
	DispatchedUDFPythonExecutor(Value payload, vector<LogicalType> output_types, vector<LogicalType> ref_output_types,
	                            shared_ptr<void> actor_handles = nullptr)
	    : payload_(std::move(payload)), output_types_(std::move(output_types)),
	      ref_output_types_(std::move(ref_output_types)), actor_handles_(std::move(actor_handles)) {
	}

	~DispatchedUDFPythonExecutor() override {
		if (registered_) {
			GlobalPythonDispatcher::Instance().Unregister(slot_id_);
		}
		// actor_handles_ is moved into the dispatcher slot during Register.
		// If registration never happened, reset the opaque holder here.
		actor_handles_.reset();
	}

	idx_t Submit(DataChunk &args, DataChunk &rows, ClientContext &context) override {
		idx_t submit_id = 0;
		TrySubmit(args, rows, context, submit_id);
		return submit_id;
	}

	bool TrySubmit(DataChunk &args, DataChunk &rows, ClientContext &context, idx_t &submit_id) override {
		return TrySubmitWithRetainedBytes(args, rows, context, args.GetAllocationSize(), submit_id);
	}

	bool TrySubmitWithRetainedBytes(DataChunk &args, DataChunk &rows, ClientContext &context,
	                                idx_t retained_input_bytes, idx_t &submit_id) override {
		lock_guard<mutex> submit_guard(submit_lock_);
		EnsureRegistered(context);

		ThrowIfSlotError();
		if (!TryReserveTaskAdmission(retained_input_bytes)) {
			submit_id = 0;
			return false;
		}

		{
			DispatcherSubmitTask task;
			task.submit_id = next_submit_id_++;
			task.rows = DeepCopyDataChunk(rows);

			{
				ScopedClientContextLock ctx_g;
				InitializeSubmitTask(task, args, context);
				AppendSubmitTaskArrowBatch(task, args, context);
			}

			submit_id = task.submit_id;
			EnqueueSubmitTask(std::move(task));
			return true;
		}
	}

	bool TrySubmitEnvelope(vector<unique_ptr<DataChunk>> &args, DataChunk &rows, ClientContext &context,
	                       idx_t &submit_id) override {
		idx_t retained_input_bytes = 0;
		for (auto &arg : args) {
			if (arg) {
				auto chunk_bytes = arg->GetAllocationSize();
				if (chunk_bytes > std::numeric_limits<idx_t>::max() - retained_input_bytes) {
					throw InvalidInputException("udf materialized envelope input size overflow");
				}
				retained_input_bytes += chunk_bytes;
			}
		}
		return TrySubmitEnvelopeWithRetainedBytes(args, rows, context, retained_input_bytes, submit_id);
	}

	bool TrySubmitEnvelopeWithRetainedBytes(vector<unique_ptr<DataChunk>> &args, DataChunk &rows,
	                                        ClientContext &context, idx_t retained_input_bytes,
	                                        idx_t &submit_id) override {
		lock_guard<mutex> submit_guard(submit_lock_);
		EnsureRegistered(context);
		ThrowIfSlotError();
		if (args.empty()) {
			throw InvalidInputException("udf materialized envelope submit requires at least one input chunk");
		}
		if (!args[0]) {
			throw InvalidInputException("udf materialized envelope submit received a null input chunk");
		}
		for (auto &arg : args) {
			if (!arg) {
				throw InvalidInputException("udf materialized envelope submit received a null input chunk");
			}
		}
		if (!TryReserveTaskAdmission(retained_input_bytes)) {
			submit_id = 0;
			return false;
		}

		DispatcherSubmitTask task;
		task.submit_id = next_submit_id_++;
		task.rows = DeepCopyDataChunk(rows);

		{
			ScopedClientContextLock ctx_g;
			InitializeSubmitTask(task, *args[0], context);
			for (auto &arg : args) {
				if (!arg) {
					throw InvalidInputException("udf materialized envelope submit received a null input chunk");
				}
				AppendSubmitTaskArrowBatch(task, *arg, context);
			}
		}

		submit_id = task.submit_id;
		EnqueueSubmitTask(std::move(task));
		return true;
	}

	bool SupportsRefBundleInput() override {
		return true;
	}

	idx_t SubmitRefBundle(LazyRefDataChunk &bundle, DataChunk &rows, ClientContext &context) override {
		idx_t submit_id = 0;
		TrySubmitRefBundle(bundle, rows, context, submit_id);
		return submit_id;
	}

	bool TrySubmitRefBundle(LazyRefDataChunk &bundle, DataChunk &rows, ClientContext &context,
	                        idx_t &submit_id) override {
		return TrySubmitRefBundleWithRetainedBytes(bundle, rows, context, 0, submit_id);
	}

	bool TrySubmitRefBundleWithRetainedBytes(LazyRefDataChunk &bundle, DataChunk &rows, ClientContext &context,
	                                         idx_t retained_input_bytes, idx_t &submit_id) override {
		lock_guard<mutex> submit_guard(submit_lock_);
		EnsureRegistered(context);

		if (!TryReserveTaskAdmission(retained_input_bytes)) {
			submit_id = 0;
			return false;
		}

		{
			lock_guard<mutex> eg(slot_->error_lock);
			if (slot_->has_error.load()) {
				auto msg = slot_->error;
				throw InvalidInputException("udf async error: %s", msg);
			}
		}

		{
			DispatcherRefSubmitTask task;
			task.submit_id = next_submit_id_++;
			task.rows = DeepCopyDataChunk(rows);
			task.bundle = CopyLazyRefDataChunk(bundle);
			{
				ScopedClientContextLock ctx_g;
				task.options = context.GetClientProperties();
			}

			submit_id = task.submit_id;
			{
				lock_guard<mutex> sg(slot_->cmd_lock);
				DispatcherCommand cmd;
				cmd.type = DispatcherCommand::SUBMIT_REF_BUNDLE;
				cmd.ref_submit_task = std::move(task);
				slot_->cmd_queue.push_back(std::move(cmd));
			}
			GlobalPythonDispatcher::Instance().NotifyWork();
			return true;
		}
	}

	std::pair<bool, UDFResult> TakeReadyResult(ClientContext &) override {
		if (!registered_) {
			return std::make_pair(false, UDFResult());
		}

		// Check errors first
		{
			lock_guard<mutex> eg(slot_->error_lock);
			if (slot_->has_error.load()) {
				auto msg = slot_->error;
				throw InvalidInputException("udf async error: %s", msg);
			}
		}

		// Try to pop a result (no GIL — just result_lock)
		{
			lock_guard<mutex> rg(slot_->result_lock);
			if (!slot_->result_queue.empty()) {
				auto result = std::move(slot_->result_queue.front());
				slot_->result_queue.pop_front();
				GlobalPythonDispatcher::Instance().NotifyWork();
				return std::make_pair(true, std::move(result));
			}
		}
		return std::make_pair(false, UDFResult());
	}

	void FinishedSubmitting(ClientContext &) override {
		if (!registered_) {
			return;
		}
		ThrowIfSlotError();
		{
			lock_guard<mutex> sg(slot_->cmd_lock);
			DispatcherCommand cmd;
			cmd.type = DispatcherCommand::FINISHED_SUBMITTING;
			slot_->cmd_queue.push_back(std::move(cmd));
		}
		GlobalPythonDispatcher::Instance().NotifyWork();
	}

	bool AllTasksFinished(ClientContext &) override {
		if (!registered_) {
			return true;
		}
		ThrowIfSlotError();
		return slot_->all_tasks_finished.load();
	}

	bool SupportsAsyncWakeup() override {
		return true;
	}

	bool SupportsOutputConsumer() override {
		return true;
	}

	void RegisterOutputConsumer(UDFOutputConsumer consumer) override {
		output_consumer_ = std::move(consumer);
		has_output_consumer_ = true;
		if (!registered_ || !slot_) {
			return;
		}
		{
			lock_guard<mutex> cg(slot_->consumer_lock);
			slot_->output_consumer = output_consumer_;
			slot_->has_output_consumer = true;
		}
		GlobalPythonDispatcher::Instance().NotifyWork();
	}

	void NotifyOutputConsumerSpaceAvailable() override {
		if (!registered_ || !slot_) {
			return;
		}
		GlobalPythonDispatcher::Instance().NotifyWork();
	}

	idx_t DebugSlotId() override {
		if (!registered_ || !slot_) {
			return 0;
		}
		return slot_id_;
	}

	InsertionOrderPreservingMap<string> Stats() override {
		InsertionOrderPreservingMap<string> result;
		if (!registered_ || !slot_ || !slot_->udf_stats_valid.load()) {
			return result;
		}
		result["udf_running_task_count"] =
		    std::to_string(slot_->udf_running_task_count.load(std::memory_order_relaxed));
		result["udf_queued_task_count"] = std::to_string(slot_->udf_queued_task_count.load(std::memory_order_relaxed));
		result["udf_max_running_tasks"] = std::to_string(slot_->udf_max_running_tasks.load(std::memory_order_relaxed));
		if (slot_->udf_output_budget_stats_valid.load(std::memory_order_relaxed)) {
			result["udf_output_budget_available"] =
			    slot_->udf_output_budget_available.load(std::memory_order_relaxed) ? "1" : "0";
			result["udf_output_budget_estimated_bytes"] =
			    std::to_string(slot_->udf_output_budget_estimated_bytes.load(std::memory_order_relaxed));
			result["udf_output_budget_limit_bytes"] =
			    std::to_string(slot_->udf_output_budget_limit_bytes.load(std::memory_order_relaxed));
			result["udf_output_budget_usage_bytes"] =
			    std::to_string(slot_->udf_output_budget_usage_bytes.load(std::memory_order_relaxed));
		}
		return result;
	}

	UDFWakeupRegistrationResult RegisterWakeup(InterruptState &interrupt_state) override {
		if (!registered_ || !slot_) {
			return UDFWakeupRegistrationResult::UNSUPPORTED;
		}
		lock_guard<mutex> wg(slot_->wakeup_lock);
		// Check readiness while holding wakeup_lock so result delivery cannot
		// miss an unarmed InterruptState between the ready check and arm step.
		if (WakeupReady()) {
			slot_->has_interrupt_state = false;
			return UDFWakeupRegistrationResult::READY;
		}
		slot_->has_interrupt_state = true;
		slot_->interrupt_state = interrupt_state;
		return UDFWakeupRegistrationResult::ARMED;
	}

	void RegisterWakeupCallback(std::function<void()> callback) override {
		wakeup_callback_ = std::move(callback);
		if (!registered_ || !slot_) {
			return;
		}
		lock_guard<mutex> wg(slot_->wakeup_lock);
		slot_->wakeup_callback = wakeup_callback_;
	}

	void EnqueueDeferredWakeup(std::function<void()> callback) override {
		if (!registered_ || !slot_) {
			throw InvalidInputException("cannot enqueue a deferred wakeup for an unregistered UDF executor");
		}
		GlobalPythonDispatcher::Instance().EnqueueDeferredWakeup(std::move(callback));
	}

private:
	void InitializeSubmitTask(DispatcherSubmitTask &task, DataChunk &args, ClientContext &context) {
		task.types = args.GetTypes();
		task.names.clear();
		task.names.reserve(task.types.size());
		for (idx_t i = 0; i < task.types.size(); i++) {
			task.names.push_back(StringUtil::Format("c%d", i));
		}
		task.options = context.GetClientProperties();
	}

	void AppendSubmitTaskArrowBatch(DispatcherSubmitTask &task, DataChunk &args, ClientContext &context) {
		if (args.GetTypes() != task.types) {
			throw InvalidInputException("udf materialized envelope submit received incompatible input chunk types");
		}
		auto schema = make_uniq<ArrowSchemaWrapper>();
		ArrowConverter::ToArrowSchema(&schema->arrow_schema, task.types, task.names, task.options);

		auto append_capacity = args.size() > 0 ? args.size() : idx_t(1);
		ArrowAppender appender(task.types, append_capacity, task.options,
		                       ArrowTypeExtensionData::GetExtensionTypes(context, task.types));
		appender.Append(args, 0, args.size(), args.size());

		auto array = make_uniq<ArrowArrayWrapper>();
		array->arrow_array = appender.Finalize();
		task.arrow_schemas.push_back(std::move(schema));
		task.arrow_arrays.push_back(std::move(array));
	}

	void EnqueueSubmitTask(DispatcherSubmitTask &&task) {
		{
			lock_guard<mutex> sg(slot_->cmd_lock);
			DispatcherCommand cmd;
			cmd.type = DispatcherCommand::SUBMIT;
			cmd.submit_task = std::move(task);
			slot_->cmd_queue.push_back(std::move(cmd));
		}
		GlobalPythonDispatcher::Instance().NotifyWork();
	}

	void ThrowIfSlotError() {
		if (!slot_) {
			return;
		}
		lock_guard<mutex> eg(slot_->error_lock);
		if (slot_->has_error.load()) {
			throw InvalidInputException("udf async error: %s", slot_->error);
		}
	}

	bool WakeupReady() {
		if (PayloadUsesTaskAdmission(payload_)) {
			lock_guard<mutex> guard(slot_->task_admission_lock);
			if (slot_->task_admission_available) {
				return true;
			}
		}
		{
			lock_guard<mutex> rg(slot_->result_lock);
			if (!slot_->result_queue.empty()) {
				return true;
			}
		}
		if (slot_->all_tasks_finished.load()) {
			return true;
		}
		return slot_->has_error.load();
	}

	bool TryReserveTaskAdmission(idx_t retained_input_bytes) {
		if (!PayloadUsesTaskAdmission(payload_)) {
			return true;
		}
		bool request_admission = false;
		{
			lock_guard<mutex> guard(slot_->task_admission_lock);
			if (slot_->task_admission_available) {
				if (slot_->task_admission_retained_input_bytes != retained_input_bytes) {
					throw InvalidInputException(
					    "UDF task admission input changed while blocked: granted=%llu current=%llu",
					    static_cast<unsigned long long>(slot_->task_admission_retained_input_bytes),
					    static_cast<unsigned long long>(retained_input_bytes));
				}
				slot_->task_admission_available = false;
				slot_->task_admission_reserved = true;
				slot_->task_admission_retained_input_bytes = 0;
				return true;
			}
			if (slot_->task_admission_reserved) {
				// One concrete lease is already queued for Python consumption.  The
				// dispatcher wakes this submitter immediately after that handoff so
				// the next request cannot overtake the consumed-state transition.
				return false;
			}
			if (!slot_->task_admission_request_pending) {
				slot_->task_admission_request_pending = true;
				request_admission = true;
			}
		}
		if (request_admission) {
			lock_guard<mutex> command_guard(slot_->cmd_lock);
			DispatcherCommand command;
			command.type = DispatcherCommand::REQUEST_TASK_ADMISSION;
			command.retained_input_bytes = retained_input_bytes;
			slot_->cmd_queue.push_back(std::move(command));
			GlobalPythonDispatcher::Instance().NotifyWork();
		}
		return false;
	}

	void EnsureRegistered(ClientContext &context) {
		if (registered_) {
			return;
		}
		slot_id_ = GlobalPythonDispatcher::Instance().Register(payload_, output_types_, ref_output_types_, &context,
		                                                       std::move(actor_handles_), &slot_);
		registered_ = true;
		if (wakeup_callback_) {
			lock_guard<mutex> wg(slot_->wakeup_lock);
			slot_->wakeup_callback = wakeup_callback_;
		}
		if (has_output_consumer_) {
			lock_guard<mutex> cg(slot_->consumer_lock);
			slot_->output_consumer = output_consumer_;
			slot_->has_output_consumer = true;
		}
	}

	Value payload_;
	vector<LogicalType> output_types_;
	vector<LogicalType> ref_output_types_;
	shared_ptr<void> actor_handles_;
	uint64_t slot_id_ = 0;
	ExecutorSlot *slot_ = nullptr; // raw pointer, valid from Register to Unregister
	bool registered_ = false;
	mutex submit_lock_;
	idx_t next_submit_id_ = 1;
	std::function<void()> wakeup_callback_;
	bool has_output_consumer_ = false;
	UDFOutputConsumer output_consumer_;
};

// ─── Factory (no GIL needed) ─────────────────────────────────────────────────

static unique_ptr<UDFExecutor> CreatePythonUDFExecutor(ClientContext &context, const Value &payload, UDFConfig &,
                                                       shared_ptr<void> actor_handles) {
	(void)context;
	if (payload.IsNull()) {
		throw InvalidInputException("UDF payload is required");
	}

	auto execution_payload = payload;
	vector<LogicalType> output_types = ParseExpectedOutputTypes(execution_payload);
	vector<LogicalType> ref_output_types = ParseStringListTypesField(execution_payload, "ref_output_types");
	if (ref_output_types.empty()) {
		throw InvalidInputException("UDF payload is missing ref_output_types");
	}
	auto execution_backend = GetStructStringField(execution_payload, "execution_backend");
	if (!execution_backend.first || execution_backend.second.empty()) {
		throw InvalidInputException("UDF payload is missing execution_backend");
	}
	if (!IsSubprocessExecutionBackend(execution_backend.second) && execution_backend.second != "ray_task" &&
	    execution_backend.second != "ray_actor") {
		throw InvalidInputException("Unsupported UDF execution_backend '%s'", execution_backend.second);
	}
	ValidateDistributedRefBundlePayload(execution_payload);

	// Subprocess/Ray paths: keep executor options opaque until the dispatcher thread
	// holds the GIL, then decode them inside EnsurePythonExecutor_WithGIL().
	return make_uniq<DispatchedUDFPythonExecutor>(std::move(execution_payload), std::move(output_types),
	                                              std::move(ref_output_types), std::move(actor_handles));
}

} // namespace

py::dict GetUDFExecutorDebugCounters() {
	py::dict result;
	result["udf_distributed_ref_bundle_data_events"] =
	    py::int_(g_udf_distributed_ref_bundle_data_events.load(std::memory_order_relaxed));
	result["udf_distributed_direct_table_rejected_events"] =
	    py::int_(g_udf_distributed_direct_table_rejected_events.load(std::memory_order_relaxed));
	result["udf_direct_arrow_table_conversion_count"] =
	    py::int_(g_udf_direct_arrow_table_conversion_count.load(std::memory_order_relaxed));
	result["udf_direct_output_arrow_table_conversion_count"] =
	    py::int_(g_udf_direct_output_arrow_table_conversion_count.load(std::memory_order_relaxed));
	result["udf_python_export_under_client_context_lock_count"] =
	    py::int_(g_udf_python_export_under_client_context_lock_count.load(std::memory_order_relaxed));
	return result;
}

void ResetUDFExecutorDebugCounters() {
	g_udf_distributed_ref_bundle_data_events.store(0, std::memory_order_relaxed);
	g_udf_distributed_direct_table_rejected_events.store(0, std::memory_order_relaxed);
	g_udf_direct_arrow_table_conversion_count.store(0, std::memory_order_relaxed);
	g_udf_direct_output_arrow_table_conversion_count.store(0, std::memory_order_relaxed);
	g_udf_python_export_under_client_context_lock_count.store(0, std::memory_order_relaxed);
}

void ShutdownUDFExecutorDispatcher() {
	GlobalPythonDispatcher::Instance().Shutdown();
}

void RegisterUDFExecutorFactory() {
	SetUDFExecutorFactory(CreatePythonUDFExecutor);
	SetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE, make_shared_ptr<PythonRayExternalBlockBackend>());
}

} // namespace duckdb
