// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "task.hpp"
#include "duckdb_python/pybind11/gil_wrapper.hpp"

#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include <duckdb/function/table/arrow.hpp>
#include <duckdb/function/table_function.hpp>
#include <duckdb/parser/tableref/table_function_ref.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/execution/operator/exchange/physical_remote_exchange_sink.hpp>
#include <duckdb/execution/distributed/plan/exchange_sink_instance_task.hpp>
#include <duckdb/execution/distributed/utils/channel.hpp>
#include <duckdb/common/arrow/arrow_converter.hpp>
#include <duckdb/common/arrow/arrow.hpp>
#include <duckdb/common/arrow/arrow_type_extension.hpp>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <cstdint>

namespace py = pybind11;
using namespace duckdb::distributed::python::ray;

namespace {

bool RayTaskPythonRuntimeUsable() {
	if (!Py_IsInitialized()) {
		return false;
	}
	if (duckdb::PythonIsFinalizing()) {
		return false;
	}
	return true;
}

class ScopedGILReleaseIfHeld {
public:
	ScopedGILReleaseIfHeld() {
		if (RayTaskPythonRuntimeUsable() && PyGILState_Check()) {
			release_ = std::make_unique<py::gil_scoped_release>();
		}
	}

	ScopedGILReleaseIfHeld(const ScopedGILReleaseIfHeld &) = delete;
	ScopedGILReleaseIfHeld &operator=(const ScopedGILReleaseIfHeld &) = delete;

private:
	std::unique_ptr<py::gil_scoped_release> release_;
};

py::object GetLoadedRayModuleOrNone() {
	auto modules = py::module_::import("sys").attr("modules");
	if (!py::isinstance<py::dict>(modules)) {
		return py::none();
	}
	auto modules_dict = py::reinterpret_borrow<py::dict>(modules);
	if (!modules_dict.contains("ray")) {
		return py::none();
	}
	auto ray_mod = py::reinterpret_borrow<py::object>(modules_dict["ray"]);
	if (ray_mod.is_none()) {
		return py::none();
	}
	return ray_mod;
}

const duckdb::PhysicalRemoteExchangeSink *FindRemoteExchangeSink(const duckdb::PhysicalOperator &op) {
	if (op.type == duckdb::PhysicalOperatorType::EXCHANGE_SINK) {
		return dynamic_cast<const duckdb::PhysicalRemoteExchangeSink *>(&op);
	}
	for (auto &child : op.children) {
		if (auto *sink = FindRemoteExchangeSink(child.get())) {
			return sink;
		}
	}
	return nullptr;
}

bool ParseExchangeSinkInstanceObject(py::object obj, duckdb::distributed::ExchangeSinkInstanceHandle &out) {
	if (obj.is_none()) {
		return false;
	}
	if (py::isinstance<py::bytes>(obj)) {
		auto bytes = obj.cast<string>();
		out = duckdb::distributed::ExchangeSinkInstanceTaskDescriptor::DeserializeFromBytes(bytes).sink_instance;
		return true;
	}
	if (!py::isinstance<py::dict>(obj)) {
		return false;
	}
	auto d = obj.cast<py::dict>();
	py::object sink_handle_obj = py::none();
	if (d.contains("sink_handle")) {
		sink_handle_obj = py::reinterpret_borrow<py::object>(d["sink_handle"]);
	}
	if (py::isinstance<py::dict>(sink_handle_obj)) {
		auto sink_handle = sink_handle_obj.cast<py::dict>();
		if (sink_handle.contains("task_partition_id")) {
			out.sink_handle.task_partition_id = py::int_(sink_handle["task_partition_id"]).cast<duckdb::idx_t>();
		} else if (sink_handle.contains("partition_id")) {
			out.sink_handle.task_partition_id = py::int_(sink_handle["partition_id"]).cast<duckdb::idx_t>();
		}
	} else if (d.contains("task_partition_id")) {
		out.sink_handle.task_partition_id = py::int_(d["task_partition_id"]).cast<duckdb::idx_t>();
	} else if (d.contains("partition_id")) {
		out.sink_handle.task_partition_id = py::int_(d["partition_id"]).cast<duckdb::idx_t>();
	}
	if (d.contains("attempt_id")) {
		out.attempt_id = py::int_(d["attempt_id"]).cast<duckdb::idx_t>();
	}
	if (d.contains("output_partition_count")) {
		out.output_partition_count = py::int_(d["output_partition_count"]).cast<duckdb::idx_t>();
	}
	if (d.contains("output_location")) {
		out.output_location = py::str(d["output_location"]).cast<string>();
	} else if (d.contains("attempt_path")) {
		out.output_location = py::str(d["attempt_path"]).cast<string>();
	}
	return true;
}

void ApplyExchangeSinkInstanceToMaterializedOutput(::duckdb::distributed::MaterializedOutput &output, py::object obj) {
	duckdb::distributed::ExchangeSinkInstanceHandle instance;
	if (ParseExchangeSinkInstanceObject(std::move(obj), instance)) {
		output.set_exchange_sink_instance(std::move(instance));
	}
}

duckdb::distributed::WorkerId WorkerIdFromPythonHandle(const py::object &handle,
                                                       const duckdb::distributed::WorkerId &fallback) {
	try {
		if (py::hasattr(handle, "exchange_node_id")) {
			auto value = handle.attr("exchange_node_id");
			if (!value.is_none()) {
				auto worker_id = value.cast<std::string>();
				if (!worker_id.empty()) {
					return duckdb::distributed::make_worker_id(worker_id);
				}
			}
		}
		if (py::hasattr(handle, "worker_id")) {
			auto value = handle.attr("worker_id");
			if (!value.is_none()) {
				auto worker_id = value.cast<std::string>();
				if (!worker_id.empty()) {
					return duckdb::distributed::make_worker_id(worker_id);
				}
			}
		}
	} catch (...) {
	}
	return fallback;
}

duckdb::distributed::TaskContext TaskContextFromPythonHandle(const py::object &handle) {
	if (!py::hasattr(handle, "task_context_info")) {
		throw duckdb::InternalException("FTE result handle must provide task_context_info");
	}
	auto info_obj = handle.attr("task_context_info");
	if (info_obj.is_none() || !py::isinstance<py::dict>(info_obj)) {
		throw duckdb::InternalException("FTE result handle task_context_info must be a dict");
	}
	auto info = info_obj.cast<py::dict>();
	for (auto key : {"query_idx", "last_node_id", "task_id", "node_ids"}) {
		if (!info.contains(key)) {
			throw duckdb::InternalException(std::string("FTE result handle task_context_info missing ") + key);
		}
	}

	uint16_t query_idx = static_cast<uint16_t>(info["query_idx"].cast<uint64_t>());
	auto last_node_id = static_cast<duckdb::distributed::NodeID>(info["last_node_id"].cast<uint64_t>());
	auto original_task_id = static_cast<duckdb::distributed::TaskID>(info["task_id"].cast<uint64_t>());
	std::vector<duckdb::distributed::NodeID> node_ids;
	for (auto node_id : info["node_ids"]) {
		node_ids.push_back(
		    static_cast<duckdb::distributed::NodeID>(py::reinterpret_borrow<py::object>(node_id).cast<uint64_t>()));
	}
	if (node_ids.empty()) {
		throw duckdb::InternalException("FTE result handle task_context_info node_ids must not be empty");
	}
	return duckdb::distributed::TaskContext(query_idx, last_node_id, original_task_id, std::move(node_ids));
}

std::string FteTaskIdStringFromPythonHandle(const py::object &handle) {
	if (!py::hasattr(handle, "task_id")) {
		throw duckdb::InternalException("FTE result handle must provide task_id");
	}
	auto task_id = handle.attr("task_id");
	if (task_id.is_none()) {
		throw duckdb::InternalException("FTE result handle task_id must not be None");
	}
	if (!(py::hasattr(task_id, "query_id") && py::hasattr(task_id, "fragment_execution_id") &&
	      py::hasattr(task_id, "partition_id") && py::hasattr(task_id, "attempt_id"))) {
		throw duckdb::InternalException(
		    "FTE result handle task_id must expose query_id, fragment_execution_id, partition_id, and attempt_id");
	}

	auto query_id = py::str(task_id.attr("query_id")).cast<std::string>();
	if (query_id.empty()) {
		throw duckdb::InternalException("FTE result handle task_id query_id must be non-empty");
	}
	auto fragment_execution_id = task_id.attr("fragment_execution_id").cast<uint64_t>();
	auto partition_id = task_id.attr("partition_id").cast<uint64_t>();
	auto attempt_id = task_id.attr("attempt_id").cast<uint64_t>();
	return query_id + "." + std::to_string(fragment_execution_id) + "." + std::to_string(partition_id) + "." +
	       std::to_string(attempt_id);
}

} // namespace

static std::shared_ptr<duckdb::ColumnDataCollection> ArrowObjectToCollection(const py::object &obj,
                                                                             duckdb::ClientContext &context) {
	auto ptr = obj.ptr();
	auto stream_factory = duckdb::make_uniq<duckdb::PythonTableArrowArrayStreamFactory>(
	    ptr, context.GetClientProperties(), duckdb::PyArrowObjectType::Table);
	auto stream_factory_produce = duckdb::PythonTableArrowArrayStreamFactory::Produce;
	auto stream_factory_get_schema = duckdb::PythonTableArrowArrayStreamFactory::GetSchema;

	duckdb::vector<duckdb::Value> children;
	children.reserve(3);
	children.push_back(duckdb::Value::POINTER(duckdb::CastPointerToValue(stream_factory.get())));
	children.push_back(duckdb::Value::POINTER(duckdb::CastPointerToValue(stream_factory_produce)));
	children.push_back(duckdb::Value::POINTER(duckdb::CastPointerToValue(stream_factory_get_schema)));

	duckdb::named_parameter_map_t named_params;
	duckdb::vector<duckdb::LogicalType> input_types;
	duckdb::vector<string> input_names;

	duckdb::TableFunctionRef empty;
	duckdb::TableFunction dummy_table_function;
	dummy_table_function.name = "ArrowObjectToCollection";
	duckdb::TableFunctionBindInput bind_input(children, named_params, input_types, input_names, nullptr, nullptr,
	                                          dummy_table_function, empty);
	duckdb::vector<duckdb::LogicalType> return_types;
	duckdb::vector<string> return_names;

	auto bind_data = duckdb::ArrowTableFunction::ArrowScanBind(context, bind_input, return_types, return_names);

	auto collection =
	    std::make_shared<duckdb::ColumnDataCollection>(duckdb::Allocator::DefaultAllocator(), return_types);
	duckdb::ColumnDataAppendState append_state;
	collection->InitializeAppend(append_state);

	duckdb::DataChunk scan_chunk;
	scan_chunk.Initialize(context, return_types, STANDARD_VECTOR_SIZE);

	duckdb::vector<duckdb::column_t> column_ids;
	column_ids.reserve(return_types.size());
	for (duckdb::idx_t i = 0; i < return_types.size(); i++) {
		column_ids.push_back(i);
	}

	duckdb::TableFunctionInitInput input(bind_data.get(), column_ids, duckdb::vector<duckdb::idx_t>(), nullptr);
	auto global_state = duckdb::ArrowTableFunction::ArrowScanInitGlobal(context, input);
	auto local_state = duckdb::ArrowTableFunction::ArrowScanInitLocalInternal(context, input, global_state.get());

	duckdb::TableFunctionInput function_input(bind_data.get(), local_state.get(), global_state.get());
	while (true) {
		scan_chunk.Reset();
		duckdb::ArrowTableFunction::ArrowScanFunction(context, function_input, scan_chunk);
		if (scan_chunk.size() == 0) {
			break;
		}
		collection->Append(append_state, scan_chunk);
	}

	return collection;
}

static py::object NormalizeArrowCompatiblePayload(py::object obj) {
	if (py::hasattr(obj, "to_arrow")) {
		obj = obj.attr("to_arrow")();
	}
	if (py::hasattr(obj, "_table")) {
		obj = obj.attr("_table");
	}
	return obj;
}

static std::shared_ptr<duckdb::ColumnDataCollection>
MaterializeArrowPayloadToCollection(const py::object &obj, duckdb::ClientContext *context) {
	if (context) {
		return ArrowObjectToCollection(obj, *context);
	}

	auto conn = duckdb::DuckDBPyConnection::DefaultConnection();
	if (!conn || !conn->con.GetConnection().context) {
		throw duckdb::InternalException(
		    "Failed to materialize Arrow payload into ColumnDataCollection: no default DuckDB connection");
	}
	return ArrowObjectToCollection(obj, *conn->con.GetConnection().context);
}

std::shared_ptr<duckdb::ColumnDataCollection>
duckdb::distributed::python::ray::MaterializePyPayloadToCollection(const py::object &obj,
                                                                   duckdb::ClientContext *context) {
	if (obj.is_none()) {
		return nullptr;
	}

	duckdb::PythonGILWrapper gil;
	py::object resolved = obj;
	if (py::hasattr(resolved, "object_ref")) {
		resolved = resolved.attr("object_ref");
	}

	auto ray_mod = GetLoadedRayModuleOrNone();
	if (!ray_mod.is_none()) {
		try {
			auto object_ref_cls = ray_mod.attr("ObjectRef");
			auto safe_get = py::module_::import("duckdb.runners.ray.safe_get").attr("resolve_object_refs_blocking");
			for (int depth = 0; depth < 5 && py::isinstance(resolved, object_ref_cls); ++depth) {
				resolved = safe_get(resolved);
			}
		} catch (const py::error_already_set &e) {
			throw duckdb::InternalException(
			    string("Failed to resolve ray.ObjectRef while materializing Python payload: ") + e.what());
		}
	}

	resolved = NormalizeArrowCompatiblePayload(resolved);
	if (duckdb::DuckDBPyConnection::GetArrowType(resolved) == duckdb::PyArrowObjectType::Invalid) {
		string repr_str = "unknown";
		try {
			repr_str = py::str(py::type::of(resolved)).cast<string>();
		} catch (...) {
		}
		throw duckdb::InternalException("Failed to materialize Python payload into ColumnDataCollection. "
		                                "Only Arrow-compatible payloads are supported. Got: " +
		                                repr_str);
	}
	return MaterializeArrowPayloadToCollection(resolved, context);
}

RayBackedResultPartition::RayBackedResultPartition(py::object object_ref, size_t num_rows, size_t size_bytes,
                                                   py::object lease_owner)
    : object_ref_(std::move(object_ref)), lease_owner_(std::move(lease_owner)), num_rows_(num_rows),
      size_bytes_(size_bytes) {
}

RayBackedResultPartition::~RayBackedResultPartition() = default;

duckdb::distributed::DuckDBResult<size_t> RayBackedResultPartition::size_bytes() const {
	if (size_bytes_ != 0) {
		return duckdb::distributed::DuckDBResult<size_t>::ok(size_bytes_);
	}

	auto collection = materialized_collection_.load();
	return duckdb::distributed::DuckDBResult<size_t>::ok(collection ? collection->SizeInBytes() : 0);
}

duckdb::distributed::DuckDBResult<size_t> RayBackedResultPartition::num_rows() const {
	if (num_rows_ != 0) {
		return duckdb::distributed::DuckDBResult<size_t>::ok(num_rows_);
	}

	auto collection = materialized_collection_.load();
	return duckdb::distributed::DuckDBResult<size_t>::ok(collection ? collection->Count() : 0);
}

py::object RayBackedResultPartition::GetObjectRef() const {
	return object_ref_.get();
}

py::object RayBackedResultPartition::GetLeaseOwner() const {
	return lease_owner_.get();
}

std::shared_ptr<duckdb::ColumnDataCollection> RayBackedResultPartition::to_column_data() const {
	// Python callers may enter with the GIL. Release it before call_once waits
	// so the materializing thread can acquire it.
	ScopedGILReleaseIfHeld release_gil;

	std::call_once(materialize_once_, [this] {
		try {
			duckdb::PythonGILWrapper gil;
			try {
				auto object_ref = object_ref_.get();
				materialized_collection_.store(MaterializePyPayloadToCollection(object_ref, nullptr));
			} catch (const py::error_already_set &ex) {
				throw duckdb::InvalidInputException("Failed to materialize Ray result partition: %s", ex.what());
			}
		} catch (...) {
			materialize_error_ = std::current_exception();
		}
	});

	if (materialize_error_) {
		std::rethrow_exception(materialize_error_);
	}
	return materialized_collection_.load();
}

py::object duckdb::distributed::python::ray::ResultPartitionToPyObject(
    const std::shared_ptr<duckdb::distributed::ResultPartition> &part) {
	if (!part) {
		return py::none();
	}

	if (auto *ray_part = dynamic_cast<duckdb::distributed::python::ray::RayBackedResultPartition *>(part.get())) {
		auto lease_owner = ray_part->GetLeaseOwner();
		if (lease_owner.is_none() || !py::hasattr(lease_owner, "transition_to")) {
			throw duckdb::InternalException("Ray result partition is missing its output lease owner");
		}
		lease_owner.attr("transition_to")(py::str("downstream_input"));
		return py::cast(duckdb::distributed::python::ray::RayResultPartitionRef(
		    ray_part->GetObjectRef(), ray_part->GetNumRowsMetadata(), ray_part->GetSizeBytesMetadata(),
		    std::move(lease_owner)));
	}

	auto collection = part->to_column_data();
	if (!collection) {
		return py::none();
	}

	duckdb::ClientProperties props;
	auto conn = duckdb::DuckDBPyConnection::DefaultConnection();
	if (conn && conn->con.GetConnection().context) {
		props = conn->con.GetConnection().context->GetClientProperties();
	}

	const auto &types = collection->Types();
	duckdb::vector<string> names;
	names.reserve(types.size());
	for (duckdb::idx_t i = 0; i < types.size(); i++) {
		names.push_back("c" + std::to_string(i));
	}

	ArrowSchema arrow_schema;
	duckdb::ArrowConverter::ToArrowSchema(&arrow_schema, types, names, props);
	auto pyarrow_lib = py::module::import("pyarrow").attr("lib");
	auto schema_import_func = pyarrow_lib.attr("Schema").attr("_import_from_c");
	auto pa_schema = schema_import_func(reinterpret_cast<uint64_t>(&arrow_schema));

	py::list batches;
	auto batch_import_func = pyarrow_lib.attr("RecordBatch").attr("_import_from_c");
	std::unordered_map<duckdb::idx_t, const duckdb::shared_ptr<duckdb::ArrowTypeExtensionData>> ext_map;
	for (auto &chunk : collection->Chunks()) {
		ArrowArray arrow_array;
		duckdb::ArrowConverter::ToArrowArray(chunk, &arrow_array, props, ext_map);
		ArrowSchema batch_schema;
		duckdb::ArrowConverter::ToArrowSchema(&batch_schema, types, names, props);
		batches.append(
		    batch_import_func(reinterpret_cast<uint64_t>(&arrow_array), reinterpret_cast<uint64_t>(&batch_schema)));
	}

	auto pa = py::module::import("pyarrow");
	return pa.attr("Table").attr("from_batches")(batches, pa_schema);
}

static ::duckdb::distributed::ResultPartitionRef BuildResultPartitionFromPyObject(py::object part) {
	size_t rows = 0;
	size_t size_bytes = 0;
	if (py::hasattr(part, "num_rows")) {
		rows = part.attr("num_rows").cast<size_t>();
	}
	if (py::hasattr(part, "size_bytes")) {
		size_bytes = part.attr("size_bytes").cast<size_t>();
	}
	if (py::hasattr(part, "object_ref")) {
		if (!py::hasattr(part, "lease_owner")) {
			throw duckdb::InternalException("RayResultPartitionRef is missing output lease ownership");
		}
		auto lease_owner = part.attr("lease_owner");
		if (lease_owner.is_none() || !py::hasattr(lease_owner, "release")) {
			throw duckdb::InternalException("RayResultPartitionRef has an invalid output lease owner");
		}
		return std::make_shared<RayBackedResultPartition>(part.attr("object_ref"), rows, size_bytes,
		                                                  std::move(lease_owner));
	}

	py::object resolved = NormalizeArrowCompatiblePayload(part);

	if (duckdb::DuckDBPyConnection::GetArrowType(resolved) != duckdb::PyArrowObjectType::Invalid) {
		auto collection = MaterializePyPayloadToCollection(resolved, nullptr);
		return std::make_shared<::duckdb::distributed::ColumnDataResultPartition>(std::move(collection));
	}

	string repr_str = "unknown";
	try {
		repr_str = py::str(py::type::of(resolved)).cast<string>();
	} catch (...) {
	}
	throw duckdb::InternalException(
	    "Failed to convert Python object to ResultPartition. "
	    "Only metadata-aware Ray fragments or Arrow-compatible objects are supported. Got: " +
	    repr_str);
}

namespace duckdb {
namespace distributed {
namespace python {
namespace ray {

struct RayTaskPollState {
	uint64_t id = 0;
	duckdb::distributed::python::ray::SafePyObject handle;
	duckdb::distributed::python::ray::RayTaskResultHandle::WorkerId worker_id;
	duckdb::distributed::python::ray::RayTaskResultHandle::TaskContext task_context;
	std::shared_ptr<duckdb::distributed::UnboundedChannelState<
	    duckdb::distributed::DuckDBResult<std::pair<bool, duckdb::distributed::MaterializedOutput>>>>
	    result_state;
	std::atomic<bool> done_sent {false};
};

namespace {

class RayTaskResultPoller {
public:
	static RayTaskResultPoller &Get() {
		static RayTaskResultPoller instance;
		return instance;
	}

	uint64_t Register(const std::shared_ptr<RayTaskPollState> &state) {
		lock_guard<mutex> guard(mutex_);
		state->id = next_id_++;
		tasks_[state->id] = state;
		EnsureStartedLocked();
		return state->id;
	}

	void Unregister(uint64_t id) {
		lock_guard<mutex> guard(mutex_);
		tasks_.erase(id);
	}

	void Shutdown() {
		stop_.store(true);
		if (thread_.joinable()) {
			thread_.join();
		}
	}

	~RayTaskResultPoller() {
		Shutdown();
	}

private:
	RayTaskResultPoller() = default;

	void EnsureStartedLocked() {
		if (thread_.joinable()) {
			return;
		}
		stop_.store(false);
		thread_ = std::thread([this]() { Run(); });
	}

	void Run() {
		while (!stop_.load()) {
			if (!RayTaskPythonRuntimeUsable()) {
				stop_.store(true);
				break;
			}
			std::vector<std::shared_ptr<RayTaskPollState>> snapshot;
			{
				lock_guard<mutex> guard(mutex_);
				snapshot.reserve(tasks_.size());
				for (auto &kv : tasks_) {
					snapshot.push_back(kv.second);
				}
			}
			if (snapshot.empty()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
				continue;
			}

			bool had_progress = false;

			try {
				// Single GIL acquisition per polling cycle (was: N times per cycle)
				PythonGILWrapper gil;

				// Batch-check which handles are ready using Python helper
				// This replaces N individual done() calls with one batch call
				py::list handles_list;
				std::vector<size_t> active_indices;
				for (size_t i = 0; i < snapshot.size(); ++i) {
					auto &state = snapshot[i];
					if (!state || state->done_sent.load()) {
						continue;
					}
					if (!state->handle.has_value()) {
						SendError(state, "RayTaskResultHandle missing handle");
						continue;
					}
					handles_list.append(state->handle.get());
					active_indices.push_back(i);
				}

				if (!active_indices.empty()) {
					// Call batch_wait_ready() - single Python call replaces N done() calls
					auto driver_mod = py::module_::import("duckdb.runners.ray.driver");
					py::object batch_wait_fn = driver_mod.attr("batch_wait_ready");
					py::list ready_indices = batch_wait_fn(handles_list);

					// Process only the ready (done) handles
					for (auto py_idx : ready_indices) {
						size_t ready_pos = py_idx.cast<size_t>();
						if (ready_pos >= active_indices.size())
							continue;
						size_t snap_idx = active_indices[ready_pos];
						auto &state = snapshot[snap_idx];
						if (state && !state->done_sent.load()) {
							ProcessDoneUnderGIL(state);
							had_progress = true;
						}
					}
				}
			} catch (const py::error_already_set &) {
				// Best-effort done notification; continue polling other tasks.
			} catch (const std::exception &) {
			}

			// Clean up done tasks
			{
				lock_guard<mutex> guard(mutex_);
				for (auto &state : snapshot) {
					if (state && state->done_sent.load()) {
						tasks_.erase(state->id);
					}
				}
			}

			// Adaptive sleep: no sleep if progress was made
			if (had_progress) {
				// Immediate next cycle
			} else if (snapshot.size() > 50) {
				std::this_thread::sleep_for(std::chrono::microseconds(500));
			} else {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		}
	}

	void SendError(const std::shared_ptr<RayTaskPollState> &state, const string &msg) {
		if (state->done_sent.exchange(true)) {
			return;
		}
		// ALWAYS log task errors to stderr
		duckdb::distributed::DuckDBError err(msg);
		duckdb::distributed::DuckDBResult<std::pair<bool, duckdb::distributed::MaterializedOutput>> res =
		    duckdb::distributed::DuckDBResult<std::pair<bool, duckdb::distributed::MaterializedOutput>>::err(err);
		if (state->result_state) {
			state->result_state->send(std::move(res));
		}
	}

	void SendOutput(const std::shared_ptr<RayTaskPollState> &state, duckdb::distributed::MaterializedOutput output) {
		if (state->done_sent.exchange(true)) {
			return;
		}
		// Final task completion is routed through the FTE query result path.
		if (state->result_state) {
			state->result_state->send(
			    duckdb::distributed::DuckDBResult<std::pair<bool, duckdb::distributed::MaterializedOutput>>::ok(
			        std::make_pair(true, std::move(output))));
		}
	}

	void SendNoOutput(const std::shared_ptr<RayTaskPollState> &state) {
		if (state->done_sent.exchange(true)) {
			return;
		}
		if (state->result_state) {
			state->result_state->send(
			    duckdb::distributed::DuckDBResult<std::pair<bool, duckdb::distributed::MaterializedOutput>>::ok(
			        std::make_pair(false, duckdb::distributed::MaterializedOutput())));
		}
	}

	// ProcessDoneUnderGIL: handles a task that batch_wait_ready() reported as done.
	// GIL is already held by the caller.
	void ProcessDoneUnderGIL(const std::shared_ptr<RayTaskPollState> &state) {
		if (!state || state->done_sent.load()) {
			return;
		}
		try {
			if (!state->handle.has_value()) {
				SendError(state, "RayTaskResultHandle missing handle");
				return;
			}
			py::object handle_obj = state->handle.get();

			py::object result = handle_obj.attr("get_result_sync")();
			RayTaskResult task_result = result.cast<RayTaskResult>();
			if (task_result.tag == RayTaskResult::Tag::Success) {
				std::vector<::duckdb::distributed::ResultPartitionRef> partitions;
				partitions.reserve(task_result.parts.size());
				for (auto &part_obj : task_result.parts) {
					py::object part = part_obj.get();
					partitions.push_back(BuildResultPartitionFromPyObject(part));
				}
				auto current_worker_id = WorkerIdFromPythonHandle(handle_obj, state->worker_id);
				std::vector<duckdb::distributed::NodeID> node_ids(state->task_context.node_ids().begin(),
				                                                  state->task_context.node_ids().end());
				::duckdb::distributed::MaterializedOutput mat_output(std::move(partitions), current_worker_id,
				                                                     std::move(node_ids));
				mat_output.set_flight_port(task_result.flight_port);
				ApplyExchangeSinkInstanceToMaterializedOutput(mat_output, task_result.ExchangeSinkInstanceObject());
				SendOutput(state, std::move(mat_output));
			} else if (task_result.tag == RayTaskResult::Tag::NoOutput) {
				SendNoOutput(state);
			} else if (task_result.tag == RayTaskResult::Tag::WorkerDied) {
				SendError(state, "worker died");
			} else {
				SendError(state, "worker unavailable");
			}
		} catch (const py::error_already_set &e) {
			SendError(state, e.what());
		} catch (const std::exception &e) {
			SendError(state, e.what());
		}
	}

	mutex mutex_;
	std::unordered_map<uint64_t, std::shared_ptr<RayTaskPollState>> tasks_;
	std::atomic<bool> stop_ {false};
	std::atomic<uint64_t> next_id_ {1};
	std::thread thread_;
};

} // namespace
} // namespace ray
} // namespace python
} // namespace distributed
} // namespace duckdb

RayTaskResultHandle::RayTaskResultHandle(TaskContext task_context, py::object handle, WorkerId worker_id,
                                         std::string fte_task_id)
    : task_context_(task_context), fte_task_id_(std::move(fte_task_id)),
      poll_result_cache_(std::make_shared<PollResultCache>()) {
	poll_state_ = std::make_shared<RayTaskPollState>();
	poll_state_->handle = SafePyObject(std::move(handle));
	poll_state_->worker_id = worker_id;
	poll_state_->task_context = task_context;
	poll_state_->result_state = std::make_shared<duckdb::distributed::UnboundedChannelState<
	    duckdb::distributed::DuckDBResult<std::pair<bool, duckdb::distributed::MaterializedOutput>>>>();
	RayTaskResultPoller::Get().Register(poll_state_);
}

RayTaskResultHandle::~RayTaskResultHandle() {
	if (poll_state_) {
		poll_state_->done_sent.store(true);
		RayTaskResultPoller::Get().Unregister(poll_state_->id);
	}
}

RayTaskResultHandle::TaskContext RayTaskResultHandle::GetTaskContext() const {
	return task_context_;
}

const std::string &RayTaskResultHandle::GetFteTaskId() const {
	return fte_task_id_;
}

std::pair<bool, ::duckdb::distributed::DuckDBResult<std::pair<bool, ::duckdb::distributed::MaterializedOutput>>>
RayTaskResultHandle::poll() {
	using ResultType = ::duckdb::distributed::DuckDBResult<std::pair<bool, ::duckdb::distributed::MaterializedOutput>>;
	if (!poll_result_cache_) {
		return std::make_pair(false, ResultType::err(::duckdb::distributed::DuckDBError("no poll result cache")));
	}
	std::lock_guard<std::mutex> guard(poll_result_cache_->mutex);
	if (poll_result_cache_->result.has_value()) {
		return std::make_pair(true, poll_result_cache_->result.value());
	}
	if (!poll_state_ || !poll_state_->result_state) {
		return std::make_pair(false, ResultType::err(::duckdb::distributed::DuckDBError("no poll state")));
	}
	auto opt = poll_state_->result_state->try_recv();
	if (!opt.first) {
		return std::make_pair(false, ResultType::err(::duckdb::distributed::DuckDBError("not ready")));
	}
	poll_result_cache_->result = std::move(opt.second);
	return std::make_pair(true, poll_result_cache_->result.value());
}

void RayTaskResultHandle::AckPollResult() {
	if (!poll_result_cache_) {
		return;
	}
	std::lock_guard<std::mutex> guard(poll_result_cache_->mutex);
	poll_result_cache_->result.reset();
}

void RayTaskResultHandle::ReleasePollResult() {
	if (!poll_result_cache_) {
		return;
	}
	bool should_release = false;
	{
		std::lock_guard<std::mutex> guard(poll_result_cache_->mutex);
		if (!released_) {
			released_ = true;
			should_release = true;
		}
	}
	if (!should_release) {
		return;
	}
	if (poll_state_) {
		// Cleanup is terminal for this handle. Stop the shared poller before
		// releasing Python payload ownership so a stale poll snapshot cannot
		// revive the handle after cleanup returns.
		poll_state_->done_sent.store(true);
		RayTaskResultPoller::Get().Unregister(poll_state_->id);
	}
	try {
		PythonGILWrapper gil;
		if (!poll_state_ || !poll_state_->handle.has_value()) {
			throw duckdb::InternalException("RayTaskResultHandle missing handle for release_result_payload");
		}
		py::object handle_obj = poll_state_->handle.get();
		handle_obj.attr("release_result_payload")();
	} catch (...) {
		std::lock_guard<std::mutex> guard(poll_result_cache_->mutex);
		released_ = false;
		throw;
	}
}

PythonTaskResultHandle::PythonTaskResultHandle(TaskContext task_context, py::object handle, WorkerId worker_id,
                                               std::string fte_task_id)
    : task_context_(task_context), worker_id_(std::move(worker_id)), fte_task_id_(std::move(fte_task_id)),
      handle_(std::move(handle)), poll_result_cache_(std::make_shared<PollResultCache>()) {
}

PythonTaskResultHandle::TaskContext PythonTaskResultHandle::GetTaskContext() const {
	return task_context_;
}

const std::string &PythonTaskResultHandle::GetFteTaskId() const {
	return fte_task_id_;
}

std::pair<bool, PythonTaskResultHandle::PollResult> PythonTaskResultHandle::poll() {
	using ResultType = PythonTaskResultHandle::PollResult;
	if (!poll_result_cache_) {
		return std::make_pair(false, ResultType::err(duckdb::distributed::DuckDBError("no poll result cache")));
	}
	{
		std::lock_guard<std::mutex> guard(poll_result_cache_->mutex);
		if (poll_result_cache_->result.has_value()) {
			return std::make_pair(true, poll_result_cache_->result.value());
		}
	}
	std::optional<ResultType> terminal_result;
	try {
		PythonGILWrapper gil;
		if (!handle_.has_value()) {
			return std::make_pair(
			    false, ResultType::err(duckdb::distributed::DuckDBError("Python task result handle missing handle")));
		}
		py::object handle_obj = handle_.get();
		if (py::hasattr(handle_obj, "done") && !handle_obj.attr("done")().cast<bool>()) {
			return std::make_pair(false, ResultType::err(duckdb::distributed::DuckDBError("not ready")));
		}

		py::object result = handle_obj.attr("get_result_sync")();
		RayTaskResult task_result = result.cast<RayTaskResult>();
		if (task_result.tag == RayTaskResult::Tag::Success) {
			std::vector<duckdb::distributed::ResultPartitionRef> partitions;
			partitions.reserve(task_result.parts.size());
			for (auto &part_obj : task_result.parts) {
				py::object part = part_obj.get();
				partitions.push_back(BuildResultPartitionFromPyObject(part));
			}
			auto current_worker_id = WorkerIdFromPythonHandle(handle_obj, worker_id_);
			std::vector<duckdb::distributed::NodeID> node_ids(task_context_.node_ids().begin(),
			                                                  task_context_.node_ids().end());
			duckdb::distributed::MaterializedOutput output(std::move(partitions), current_worker_id,
			                                               std::move(node_ids));
			output.set_flight_port(task_result.flight_port);
			ApplyExchangeSinkInstanceToMaterializedOutput(output, task_result.ExchangeSinkInstanceObject());
			terminal_result = ResultType::ok(std::make_pair(true, std::move(output)));
		} else if (task_result.tag == RayTaskResult::Tag::NoOutput) {
			terminal_result = ResultType::ok(std::make_pair(false, duckdb::distributed::MaterializedOutput()));
		} else if (task_result.tag == RayTaskResult::Tag::WorkerDied) {
			terminal_result = ResultType::err(duckdb::distributed::DuckDBError("worker died"));
		} else {
			terminal_result = ResultType::err(duckdb::distributed::DuckDBError("worker unavailable"));
		}
	} catch (const py::error_already_set &e) {
		terminal_result = ResultType::err(duckdb::distributed::DuckDBError(e.what()));
	} catch (const std::exception &e) {
		terminal_result = ResultType::err(duckdb::distributed::DuckDBError(e.what()));
	}
	std::lock_guard<std::mutex> guard(poll_result_cache_->mutex);
	if (!poll_result_cache_->result.has_value()) {
		poll_result_cache_->result = std::move(terminal_result.value());
	}
	return std::make_pair(true, poll_result_cache_->result.value());
}

void PythonTaskResultHandle::AckPollResult() {
	if (!poll_result_cache_) {
		return;
	}
	bool should_ack = false;
	{
		std::lock_guard<std::mutex> guard(poll_result_cache_->mutex);
		poll_result_cache_->result.reset();
		if (!acked_) {
			acked_ = true;
			should_ack = true;
		}
	}
	if (!should_ack) {
		return;
	}
	try {
		PythonGILWrapper gil;
		if (!handle_.has_value()) {
			return;
		}
		py::object handle_obj = handle_.get();
		if (py::hasattr(handle_obj, "ack")) {
			handle_obj.attr("ack")();
		}
	} catch (const py::error_already_set &) {
	} catch (const std::exception &) {
	}
}

void PythonTaskResultHandle::ReleasePollResult() {
	if (!poll_result_cache_) {
		return;
	}
	bool should_release = false;
	{
		std::lock_guard<std::mutex> guard(poll_result_cache_->mutex);
		if (!released_) {
			released_ = true;
			should_release = true;
		}
	}
	if (!should_release) {
		return;
	}
	try {
		PythonGILWrapper gil;
		if (!handle_.has_value()) {
			throw duckdb::InternalException("PythonTaskResultHandle missing handle for release_result_payload");
		}
		py::object handle_obj = handle_.get();
		handle_obj.attr("release_result_payload")();
	} catch (...) {
		std::lock_guard<std::mutex> guard(poll_result_cache_->mutex);
		released_ = false;
		throw;
	}
}

duckdb::distributed::python::ray::PythonTaskResultHandle
duckdb::distributed::python::ray::MakePythonTaskResultHandle(py::object handle) {
	auto task_context = TaskContextFromPythonHandle(handle);
	auto worker_id = WorkerIdFromPythonHandle(handle, duckdb::distributed::make_worker_id(""));
	if (!worker_id || worker_id->empty()) {
		throw duckdb::InternalException("FTE result handle must provide non-empty worker_id");
	}
	auto fte_task_id = FteTaskIdStringFromPythonHandle(handle);
	return PythonTaskResultHandle(task_context, std::move(handle), worker_id, std::move(fte_task_id));
}

RayWorkerTask::RayWorkerTask(duckdb::distributed::WorkerTask task) : task_(std::move(task)) {
}

std::unordered_map<string, string> RayWorkerTask::Context() const {
	return task_.context();
}

py::dict RayWorkerTask::TaskContextInfo() const {
	auto task_context = task_.task_context();
	py::dict info;
	info["query_idx"] = task_context.query_idx();
	info["last_node_id"] = task_context.last_node_id();
	info["task_id"] = task_context.task_id();
	py::list node_ids;
	for (auto node_id : task_context.node_ids()) {
		node_ids.append(node_id);
	}
	info["node_ids"] = std::move(node_ids);
	return info;
}

string RayWorkerTask::Name() const {
	return task_.name();
}

py::object RayWorkerTask::Plan() const {

	// Return the underlying PhysicalPlan wrapped in PyPhysicalPlanWrapper (DistributedPhysicalPlan)
	auto plan_ref = task_.plan();
	if (!plan_ref) {
		return py::none();
	}

	// Create PyPhysicalPlanWrapper by passing the plan via capsule
	duckdb::PythonGILWrapper gil;
	auto ray_cxx = py::module_::import("_duckdb.ray_cxx");
	auto task_context = task_.context();
	auto query_id_entry = task_context.find("query_id");
	if (query_id_entry == task_context.end() || query_id_entry->second.empty()) {
		throw duckdb::InternalException("RayWorkerTask::Plan requires non-empty task context query_id");
	}
	py::object query_id_obj = py::str(query_id_entry->second);
	auto udf_registrations_obj = ray_cxx.attr("_lookup_query_udf_registrations")(query_id_obj);
	auto udf_actor_handles_obj = ray_cxx.attr("_lookup_query_udf_actor_handles")(query_id_obj);
	auto connection_snapshot_obj = ray_cxx.attr("_lookup_query_connection_snapshot")(query_id_obj);

	// The capsule owns this shared_ptr copy if any later Python or C++ step
	// raises, so materialization errors can propagate without leaking it.
	auto *plan_copy = new std::shared_ptr<duckdb::PhysicalPlan>(plan_ref);
	py::capsule plan_capsule(plan_copy,
	                         [](void *ptr) { delete static_cast<std::shared_ptr<duckdb::PhysicalPlan> *>(ptr); });
	auto create_fn = ray_cxx.attr("_create_physical_plan_from_capsule");
	return create_fn(plan_capsule, query_id_obj, udf_registrations_obj, udf_actor_handles_obj, connection_snapshot_obj);
}

py::dict RayWorkerTask::Inputs() const {
	duckdb::PythonGILWrapper gil;
	py::dict result;
	for (const auto &kv : task_.inputs()) {
		py::dict entry;
		if (kv.second.kind == duckdb::distributed::TaskInput::Kind::ScanTask) {
			entry["kind"] = "scan_task";
			// scan_task_bytes is raw binary data — use py::bytes, NOT py::str
			// (py::str would trigger UTF-8 decode and fail on arbitrary bytes)
			entry["data"] = py::bytes(kv.second.scan_task_bytes);
		} else if (kv.second.kind == duckdb::distributed::TaskInput::Kind::ExchangeSourceTask) {
			entry["kind"] = "exchange_source_task";
			entry["data"] = py::bytes(kv.second.exchange_source_task_bytes);
		}
		result[py::str(std::to_string(kv.first))] = entry;
	}
	return result;
}

py::object RayWorkerTask::ExchangeSinkInstance() const {
	duckdb::PythonGILWrapper gil;
	auto plan_ref = task_.plan();
	if (!plan_ref || !plan_ref->HasRoot()) {
		return py::none();
	}
	auto *sink = FindRemoteExchangeSink(plan_ref->Root());
	if (!sink) {
		return py::none();
	}
	const auto &instance = sink->SinkHandle();
	py::dict sink_handle;
	sink_handle["task_partition_id"] = instance.sink_handle.task_partition_id;
	sink_handle["partition_id"] = instance.sink_handle.task_partition_id;

	py::dict result;
	result["sink_handle"] = sink_handle;
	result["task_partition_id"] = instance.sink_handle.task_partition_id;
	result["partition_id"] = instance.sink_handle.task_partition_id;
	result["attempt_id"] = instance.attempt_id;
	result["output_partition_count"] = instance.output_partition_count;
	if (!instance.output_location.empty()) {
		result["output_location"] = instance.output_location;
		result["attempt_path"] = instance.output_location;
	}
	return result;
}
