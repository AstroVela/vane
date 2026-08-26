// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "task.hpp"
#include "worker.hpp"
#include "worker_manager.hpp"
#include "bounded_diagnostics.hpp"
#include "python_bounded_diagnostics.hpp"
#include "safe_pyobject.hpp"
#include "datasource_function.hpp"

#include "vane_python/pyrelation.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/python_objects.hpp"
#include "vane_python/arrow/arrow_array_stream.hpp"
#include "vane_python/arrow/arrow_export_utils.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"

#include <duckdb/execution/distributed/plan/distributed_physical_plan.hpp>
#include <duckdb/execution/distributed/plan/plan_config.hpp>
#include <duckdb/execution/distributed/plan/runner.hpp>
#include <duckdb/execution/distributed/common_types.hpp>
#include <duckdb/execution/distributed/copy_to_file.hpp>
#include <duckdb/execution/distributed/copy_finalize.hpp>
#include <duckdb/execution/distributed/extension_write_task_provider.hpp>
#include <duckdb/execution/distributed/plan/exchange_sink_instance_task.hpp>
#include <duckdb/execution/distributed/plan/exchange_source_task.hpp>
#include <duckdb/execution/distributed/plan/fte_split_queue.hpp>
#include <duckdb/execution/distributed/plan/scan_split.hpp>
#include <duckdb/execution/distributed/pipeline_node/pipeline_node.hpp>
#include <duckdb/execution/distributed/pipeline_node/sink.hpp>
#include <duckdb/execution/distributed/pipeline_node/streaming_udf_passthrough.hpp>
#include <duckdb/execution/distributed/pipeline_node/table_inout.hpp>
#include <duckdb/execution/distributed/pipeline_node/translator.hpp>
#include <duckdb/execution/distributed/pipeline_node/translator_api.hpp>
#include <duckdb/execution/distributed/pipeline_node/vllm.hpp>
#include <duckdb/execution/distributed/utils/channel.hpp>
#include <duckdb/planner/planner.hpp>
#include <duckdb/common/file_system.hpp>
#include <duckdb/common/limits.hpp>
#include <duckdb/common/local_file_system.hpp>
#include <duckdb/common/map.hpp>
#include <duckdb/common/set.hpp>
#include <duckdb/common/types/uuid.hpp>
#include <duckdb/optimizer/optimizer.hpp>

#include <exception>
#include <optional>
#include <tuple>

static inline int DuckdbGetEnvIntMs(const char *name) {
	const char *val = std::getenv(name);
	if (!val || !*val) {
		return 0;
	}
	char *end = nullptr;
	long parsed = std::strtol(val, &end, 10);
	if (end == val || parsed <= 0) {
		return 0;
	}
	if (parsed > std::numeric_limits<int>::max()) {
		return std::numeric_limits<int>::max();
	}
	return static_cast<int>(parsed);
}

#include <duckdb/execution/physical_plan_generator.hpp>
#include <duckdb/execution/executor.hpp>
#include <duckdb/parser/statement/logical_plan_statement.hpp>
#include <duckdb/parser/statement/relation_statement.hpp>
#include <duckdb/common/serializer/memory_stream.hpp>
#include <duckdb/common/serializer/binary_serializer.hpp>
#include <duckdb/common/serializer/binary_deserializer.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/client_data.hpp>
#include <duckdb/main/config.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/main/database_manager.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/main/distributed_extension_manager.hpp>
#include <duckdb/main/extension_helper.hpp>
#include <duckdb/parser/keyword_helper.hpp>
#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/types/value.hpp>
#include <duckdb/common/vector_size.hpp>
#include <duckdb/common/algorithm.hpp>
#include <duckdb/parallel/interrupt.hpp>
#include <duckdb/parallel/thread_context.hpp>
#include <duckdb/parallel/task_scheduler.hpp>
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/main/relation/data_sink_relation.hpp>
#include <duckdb/execution/operator/helper/physical_materialized_collector.hpp>
#include <duckdb/execution/operator/helper/physical_data_sink.hpp>
#include <duckdb/execution/operator/exchange/physical_remote_exchange_sink.hpp>
#include <duckdb/execution/operator/exchange/physical_remote_exchange_source.hpp>
#include <duckdb/execution/operator/persistent/physical_batch_copy_to_file.hpp>
#include <duckdb/execution/operator/persistent/physical_copy_to_file.hpp>
#include <duckdb/execution/operator/persistent/physical_distributed_extension_write.hpp>
#include <duckdb/execution/operator/scan/physical_column_data_scan.hpp>
#include <duckdb/execution/operator/scan/physical_dummy_scan.hpp>
#include <duckdb/execution/operator/scan/physical_table_scan.hpp>
#include <duckdb/planner/filter/constant_filter.hpp>
#include <duckdb/planner/filter/in_filter.hpp>
#include <duckdb/execution/operator/projection/physical_tableinout_function.hpp>
#include <duckdb/execution/operator/projection/physical_udf_inout.hpp>
#include <duckdb/function/scalar/udf_functions.hpp>
#include <duckdb/function/table/datasource_scan.hpp>
#include <duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp>
#include <duckdb/execution/operator/helper/physical_result_collector.hpp>
#include <duckdb/execution/operator/projection/physical_projection.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/common/arrow/arrow_converter.hpp>
#include <duckdb/common/arrow/arrow.hpp>
#include <duckdb/common/arrow/arrow_type_extension.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/execution/distributed/exchange/flight_exchange_manager.hpp>
#include <duckdb/execution/distributed/exchange/shuffle_cache_registry.hpp>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <limits>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

#include "native_fragment_executor.cpp"

namespace duckdb {

#include "result_stream_bindings.cpp"
#include "logical_plan_bindings.cpp"
#include "distributed_plan_bindings.cpp"

class PhysicalNonSerializableOperatorForTest : public PhysicalOperator {
public:
	explicit PhysicalNonSerializableOperatorForTest(PhysicalPlan &physical_plan)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::DUMMY_SCAN, {LogicalType::INTEGER}, 1) {
	}

	string GetName() const override {
		return "INTENTIONALLY_NON_SERIALIZABLE";
	}

protected:
	void SerializeOperatorData(Serializer &) const override {
		throw NotImplementedException("INTENTIONALLY_NON_SERIALIZABLE operator cannot be serialized");
	}
};

class CoordinatorOnlyWriteGlobalStateForTest final : public DistributedWriteGlobalState {};

class CoordinatorOnlyWriteLocalStateForTest final : public DistributedWriteLocalState {};

static unique_ptr<DistributedWriteGlobalState>
CoordinatorOnlyWriteInitializeGlobalForTest(ClientContext &, const DistributedExtensionWriteInfo &,
                                            const DistributedWriteTaskContext &) {
	return make_uniq<CoordinatorOnlyWriteGlobalStateForTest>();
}

static unique_ptr<DistributedWriteLocalState>
CoordinatorOnlyWriteInitializeLocalForTest(ExecutionContext &, const DistributedExtensionWriteInfo &,
                                           const DistributedWriteTaskContext &, DistributedWriteGlobalState &) {
	return make_uniq<CoordinatorOnlyWriteLocalStateForTest>();
}

static void CoordinatorOnlyWriteSinkForTest(ExecutionContext &, const DistributedExtensionWriteInfo &,
                                            const DistributedWriteTaskContext &, DistributedWriteGlobalState &,
                                            DistributedWriteLocalState &, DataChunk &) {
}

static void CoordinatorOnlyWriteCombineForTest(ExecutionContext &, const DistributedExtensionWriteInfo &,
                                               const DistributedWriteTaskContext &, DistributedWriteGlobalState &,
                                               DistributedWriteLocalState &) {
}

static vector<DistributedWriteFragment> CoordinatorOnlyWriteFinalizeForTest(ClientContext &,
                                                                            const DistributedExtensionWriteInfo &,
                                                                            const DistributedWriteTaskContext &,
                                                                            DistributedWriteGlobalState &) {
	return {};
}

static void RegisterCoordinatorOnlyExtensionWriteForTest(ClientContext &context) {
	DistributedExtensionManifest manifest;
	manifest.extension_name = "vane_test";
	manifest.capabilities.push_back({DistributedExtensionCapabilityKind::WRITE_OPERATOR, "coordinator_only_write", 1});

	DistributedWriteOperatorExtension write_operator;
	write_operator.name = "coordinator_only_write";
	write_operator.protocol_version = 1;
	write_operator.mode = DistributedWriteMode::CALLBACK;
	write_operator.fragment_codec = {"vane_test.coordinator_only_write", 1};
	write_operator.callbacks.initialize_global = CoordinatorOnlyWriteInitializeGlobalForTest;
	write_operator.callbacks.initialize_local = CoordinatorOnlyWriteInitializeLocalForTest;
	write_operator.callbacks.sink = CoordinatorOnlyWriteSinkForTest;
	write_operator.callbacks.combine = CoordinatorOnlyWriteCombineForTest;
	write_operator.callbacks.finalize = CoordinatorOnlyWriteFinalizeForTest;

	DistributedExtensionManager::Get(context).RegisterExtension(
	    manifest, {make_shared_ptr<const DistributedWriteOperatorExtension>(std::move(write_operator))});
}

class PhysicalCoordinatorOnlyExtensionWriteForTest final : public PhysicalOperator,
                                                           public distributed::ExtensionWriteTaskProvider {
public:
	explicit PhysicalCoordinatorOnlyExtensionWriteForTest(PhysicalPlan &physical_plan, bool fail_finalize_p = false)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, {LogicalType::BIGINT}, 1),
	      fail_finalize(fail_finalize_p) {
		plan.extension_name = "vane_test";
		plan.operator_name = "coordinator_only_write";
	}

	optional_ptr<distributed::ExtensionWriteTaskProvider> GetExtensionWriteTaskProvider() override {
		return this;
	}

	const distributed::DistributedExtensionWritePlan &WritePlan() const override {
		return plan;
	}

	void ValidateDistributedWrite(ClientContext &) const override {
	}

	idx_t FinalizeDistributedWrite(ClientContext &, const vector<DistributedWriteTaskResult> &results) const override {
		if (fail_finalize) {
			throw IOException("planned coordinator-only extension finalization failure");
		}
		idx_t rows = 0;
		for (const auto &result : results) {
			rows += result.RowCount();
		}
		return rows;
	}

	void AbortDistributedWrite(ClientContext &, const vector<DistributedWriteTaskResult> &) const override {
	}

protected:
	void SerializeOperatorData(Serializer &) const override {
		throw NotImplementedException("COORDINATOR_ONLY_EXTENSION_WRITE root cannot be serialized");
	}

private:
	distributed::DistributedExtensionWritePlan plan;
	bool fail_finalize;
};

class CountingResultCollectorForTest {
public:
	CountingResultCollectorForTest() : calls(make_shared_ptr<std::atomic<idx_t>>(0)) {
	}

	unique_ptr<PhysicalOperator> operator()(ClientContext &context, PreparedStatementData &data) const {
		calls->fetch_add(1, std::memory_order_relaxed);
		return PhysicalResultCollector::GetResultCollector(context, data);
	}

	shared_ptr<std::atomic<idx_t>> calls;
};

class ScopedShuffleCacheRegistrationForTest {
public:
	explicit ScopedShuffleCacheRegistrationForTest(string exchange_id) : exchange_id_(std::move(exchange_id)) {
	}

	~ScopedShuffleCacheRegistrationForTest() {
		distributed::ShuffleCacheRegistry::Instance().Remove(exchange_id_);
	}

	ScopedShuffleCacheRegistrationForTest(const ScopedShuffleCacheRegistrationForTest &) = delete;
	ScopedShuffleCacheRegistrationForTest &operator=(const ScopedShuffleCacheRegistrationForTest &) = delete;

private:
	string exchange_id_;
};

static py::object ResolveFlightShuffleCleanupConnection(py::object cleanup_connection,
                                                        const py::object &connection_snapshot,
                                                        const py::object &effective_session_config,
                                                        bool apply_snapshot_s3_credentials) {
	auto resolved_connection = connection_snapshot.is_none()
	                               ? cleanup_connection
	                               : ResolveConnectionForSnapshot(cleanup_connection, connection_snapshot);
	auto &resolved_wrapper = ExtractPyConnectionWrapper(resolved_connection);
	if (resolved_wrapper.con.ConnectionIsClosed()) {
		throw std::runtime_error("resolved shuffle cleanup connection is closed");
	}
	ConnectionSnapshotApplyOptions snapshot_options;
	snapshot_options.apply_session_config = false;
	snapshot_options.apply_s3_credentials = apply_snapshot_s3_credentials;
	ValidateConnectionSnapshotExtensions(resolved_connection, connection_snapshot,
	                                     snapshot_options.enforce_extension_security);
	if (ConnectionSnapshotDeclaresExtension(connection_snapshot, "httpfs")) {
		ApplyEffectiveVaneSessionConfig(resolved_wrapper, effective_session_config);
	}
	ApplyConnectionSnapshot(resolved_connection, connection_snapshot, snapshot_options);
	return resolved_connection;
}

static string QueryConnectionSettingForTest(DuckDBPyConnection &connection, const string &name) {
	auto result = ExecuteSnapshotQuery(connection.con.GetConnection(),
	                                   "SELECT current_setting(" + QuoteSQLStringLiteral(name) + ")");
	for (auto &row : result->Collection().Rows()) {
		auto value = row.GetValue(0);
		return value.IsNull() ? string() : value.ToString();
	}
	throw std::runtime_error("connection setting query returned no rows: " + name);
}

template <typename CALLBACK>
static auto WithCopyRecoveryContext(py::object conn_obj, CALLBACK callback) {
	auto run_callback = [&](duckdb::ClientContext &context) {
		using Result = decltype(callback(context));
		std::optional<Result> result;
		std::exception_ptr callback_error;
		{
			py::gil_scoped_release release;
			context.RunFunctionInTransaction(
			    [&]() {
				    // Match normal query startup after a failed query leaves a stale interrupt flag behind.
				    context.ClearInterrupt();
				    try {
					    result.emplace(callback(context));
				    } catch (...) {
					    callback_error = std::current_exception();
				    }
			    },
			    false);
		}
		if (callback_error) {
			std::rethrow_exception(callback_error);
		}
		D_ASSERT(result.has_value());
		return std::move(result.value());
	};
	if (!conn_obj.is_none()) {
		auto &conn_wrapper = ExtractPyConnectionWrapper(std::move(conn_obj));
		return run_callback(*conn_wrapper.con.GetConnection().context);
	}
	duckdb::DuckDB db(nullptr);
	duckdb::Connection conn(db);
	return run_callback(*conn.context);
}

static py::list ReadIntegerExchangeSourceForTest(duckdb::ClientContext &context,
                                                 duckdb::distributed::FlightExchangeConfig config,
                                                 duckdb::distributed::ExchangeSourceHandle handle) {
	using namespace duckdb::distributed;
	config.expected_types = {LogicalType::INTEGER};
	FlightExchangeSource source(config, &context);
	source.AddSourceHandles({std::move(handle)});
	vector<LogicalType> output_types = {LogicalType::INTEGER};
	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), output_types);
	py::list values;
	while (source.ReadChunk(output)) {
		for (idx_t row = 0; row < output.size(); row++) {
			values.append(output.GetValue(0, row).GetValue<int32_t>());
		}
	}
	source.Close();
	return values;
}

void register_ray_bindings(py::module_ &mod) {
	auto m = mod.def_submodule("ray_cxx");
	m.doc() = "C++ Ray execution bindings (experimental)";
	m.def(
	    "_install_counting_result_collector_for_test",
	    [](py::object conn_obj) {
		    auto &conn_wrapper = ExtractPyConnectionWrapper(std::move(conn_obj));
		    auto &context = *conn_wrapper.con.GetConnection().context;
		    ClientConfig::GetConfig(context).get_result_collector = CountingResultCollectorForTest();
	    },
	    py::arg("conn"), "Install a connection-level result collector that counts materialized queries.");
	m.def(
	    "_connection_result_collector_calls_for_test",
	    [](py::object conn_obj) {
		    auto &conn_wrapper = ExtractPyConnectionWrapper(std::move(conn_obj));
		    auto &context = *conn_wrapper.con.GetConnection().context;
		    auto &callback = ClientConfig::GetConfig(context).get_result_collector;
		    auto collector = callback.target<CountingResultCollectorForTest>();
		    if (!collector) {
			    throw InternalException("The counting result collector is no longer installed");
		    }
		    return collector->calls->load(std::memory_order_relaxed);
	    },
	    py::arg("conn"), "Return the number of materialized queries observed by the test collector.");
	m.def(
	    "_execute_materialized_int64_for_test",
	    [](py::object conn_obj, const string &query) {
		    auto &conn_wrapper = ExtractPyConnectionWrapper(std::move(conn_obj));
		    auto result = conn_wrapper.con.GetConnection().Query(query);
		    if (result->HasError()) {
			    throw InvalidInputException(result->GetError());
		    }
		    return result->GetValue<int64_t>(0, 0);
	    },
	    py::arg("conn"), py::arg("query"), "Execute a materialized query and return its first BIGINT value.");
	m.def("shutdown_local_flight_service", []() {
		auto result = []() {
			py::gil_scoped_release release;
			return duckdb::distributed::FlightExchangeManager::ShutdownLocalFlightServer();
		}();
		if (result.is_err()) {
			throw std::runtime_error(result.error().what());
		}
	});

	py::class_<RayResultPartitionRef>(m, "RayResultPartitionRef")
	    .def(py::init<py::object, size_t, size_t, py::object>())
	    .def_property_readonly("object_ref", &RayResultPartitionRef::GetObjectRef)
	    .def_property_readonly("num_rows", &RayResultPartitionRef::GetNumRows)
	    .def_property_readonly("size_bytes", &RayResultPartitionRef::GetSizeBytes)
	    .def_property_readonly("lease_owner", &RayResultPartitionRef::GetLeaseOwner);

	py::class_<NativePartitionMetadata>(m, "NativePartitionMetadata")
	    .def(py::init<size_t, size_t>())
	    .def_readonly("num_rows", &NativePartitionMetadata::num_rows)
	    .def_readonly("size_bytes", &NativePartitionMetadata::size_bytes);

	using RayBackedResultPartition = duckdb::distributed::python::ray::RayBackedResultPartition;
	py::class_<RayBackedResultPartition, std::shared_ptr<RayBackedResultPartition>>(m,
	                                                                                "_RayBackedResultPartitionForTest")
	    .def(py::init([](py::object payload) {
		    return std::make_shared<RayBackedResultPartition>(std::move(payload), 0, 0, py::none());
	    }))
	    .def("materialize", [](const RayBackedResultPartition &partition) {
		    auto collection = partition.to_column_data();
		    return collection ? collection->Count() : 0;
	    });

	py::class_<NativeDistributedTaskResult>(m, "NativeDistributedTaskResult")
	    .def(py::init<py::iterable, py::iterable, py::object, py::object, std::string, int, py::object, py::object>(),
	         py::arg("partition_payloads"), py::arg("partition_metadatas"), py::arg("result_schema"), py::arg("stats"),
	         py::arg("completion_status"), py::arg("flight_port") = 0, py::arg("exchange_sink_instance") = py::none(),
	         py::arg("task_stats") = py::none())
	    .def_property_readonly("partition_payloads", &NativeDistributedTaskResult::PartitionPayloads)
	    .def_property_readonly("partition_metadatas", &NativeDistributedTaskResult::PartitionMetadatas)
	    .def_property_readonly("result_schema", &NativeDistributedTaskResult::ResultSchema)
	    .def_property_readonly("stats", &NativeDistributedTaskResult::Stats)
	    .def_property_readonly("task_stats", &NativeDistributedTaskResult::TaskStats)
	    .def_readonly("completion_status", &NativeDistributedTaskResult::completion_status)
	    .def_readonly("flight_port", &NativeDistributedTaskResult::flight_port)
	    .def_property_readonly("exchange_sink_instance", &NativeDistributedTaskResult::ExchangeSinkInstance);

	py::class_<RayTaskResult>(m, "RayTaskResult")
	    .def_static("success", &RayTaskResult::Success, py::arg("ray_part_refs"), py::arg("stats"),
	                py::arg("result_schema") = py::none(), py::arg("flight_port") = 0,
	                py::arg("exchange_sink_instance") = py::none())
	    .def_static("no_output", &RayTaskResult::NoOutput)
	    .def_static("update_ack", &RayTaskResult::NoOutput)
	    .def_static("worker_died", &RayTaskResult::WorkerDied)
	    .def_static("worker_unavailable", &RayTaskResult::WorkerUnavailable)
	    .def_property_readonly("ok",
	                           [](const RayTaskResult &self) {
		                           return self.tag == RayTaskResult::Tag::Success ||
		                                  self.tag == RayTaskResult::Tag::NoOutput;
	                           })
	    .def_property_readonly("has_output",
	                           [](const RayTaskResult &self) { return self.tag == RayTaskResult::Tag::Success; })
	    .def_property_readonly("result_schema", [](const RayTaskResult &self) { return self.ResultSchema(); })
	    .def_property_readonly("flight_port", [](const RayTaskResult &self) { return self.flight_port; })
	    .def_property_readonly("exchange_sink_instance",
	                           [](const RayTaskResult &self) { return self.ExchangeSinkInstanceObject(); });

	py::class_<RayWorkerTask>(m, "RayWorkerTask")
	    .def(py::init<duckdb::distributed::WorkerTask>())
	    .def("context", &RayWorkerTask::Context)
	    .def("task_context", &RayWorkerTask::TaskContextInfo)
	    .def("name", &RayWorkerTask::Name)
	    .def("plan", &RayWorkerTask::Plan)
	    .def("exchange_sink_config", &RayWorkerTask::ExchangeSinkConfig)
	    .def("Inputs", &RayWorkerTask::Inputs);

	py::class_<RayWorkerRuntime, std::shared_ptr<RayWorkerRuntime>>(m, "RayWorkerRuntime")
	    .def(py::init<string, py::object, double, double, size_t>())
	    .def_property_readonly("worker_id",
	                           [](const RayWorkerRuntime &self) -> string {
		                           const auto &id = self.Id();
		                           return id ? *id : "";
	                           })
	    .def_property_readonly("num_cpus", &RayWorkerRuntime::TotalNumCpus)
	    .def_property_readonly("num_gpus", &RayWorkerRuntime::TotalNumGpus)
	    .def_property_readonly("total_memory_bytes", &RayWorkerRuntime::TotalMemoryBytes);
	// Note: submit_task is not exposed - only used internally from C++

	py::class_<duckdb::distributed::FteSplitQueue, std::shared_ptr<duckdb::distributed::FteSplitQueue>>(m,
	                                                                                                    "FteSplitQueue")
	    .def(py::init<>())
	    .def(
	        "add_scan_split",
	        [](duckdb::distributed::FteSplitQueue &self, py::bytes bytes) {
		        self.AddSplit(duckdb::distributed::TaskInput::make_scan_split_batch(bytes.cast<string>()));
	        },
	        py::arg("bytes"))
	    .def(
	        "add_exchange_source_split",
	        [](duckdb::distributed::FteSplitQueue &self, py::bytes bytes) {
		        self.AddSplit(duckdb::distributed::TaskInput::make_exchange_source_task(bytes.cast<string>()));
	        },
	        py::arg("bytes"))
	    .def("no_more_splits", &duckdb::distributed::FteSplitQueue::NoMoreSplits)
	    .def("cancel", &duckdb::distributed::FteSplitQueue::Cancel)
	    .def("buffered_splits", &duckdb::distributed::FteSplitQueue::BufferedSplits)
	    .def("buffered_bytes", &duckdb::distributed::FteSplitQueue::BufferedBytes)
	    .def("submitted_splits", &duckdb::distributed::FteSplitQueue::SubmittedSplits)
	    .def("consumed_splits", &duckdb::distributed::FteSplitQueue::ConsumedSplits)
	    .def("completed_splits", &duckdb::distributed::FteSplitQueue::CompletedSplits)
	    .def("submitted_rows", &duckdb::distributed::FteSplitQueue::SubmittedRows)
	    .def("submitted_input_bytes", &duckdb::distributed::FteSplitQueue::SubmittedInputBytes)
	    .def("consumed_rows", &duckdb::distributed::FteSplitQueue::ConsumedRows)
	    .def("consumed_input_bytes", &duckdb::distributed::FteSplitQueue::ConsumedInputBytes)
	    .def("completed_rows", &duckdb::distributed::FteSplitQueue::CompletedRows)
	    .def("completed_input_bytes", &duckdb::distributed::FteSplitQueue::CompletedInputBytes)
	    .def("queue_wait_ms", &duckdb::distributed::FteSplitQueue::QueueWaitMillis)
	    .def("exchange_source_partition_count", &duckdb::distributed::FteSplitQueue::ExchangeSourcePartitionCount)
	    .def("exchange_source_task_count", &duckdb::distributed::FteSplitQueue::ExchangeSourceTaskCount)
	    .def("complete_consumed_splits", &duckdb::distributed::FteSplitQueue::CompleteConsumedSplits)
	    .def("try_get_next",
	         [](duckdb::distributed::FteSplitQueue &self) {
		         auto result = self.TryGetNext();
		         return FteSplitQueueResultToDict(result);
	         })
	    .def("wait_for_next", [](duckdb::distributed::FteSplitQueue &self) {
		    duckdb::distributed::FteSplitQueue::GetNextResult result;
		    {
			    py::gil_scoped_release release;
			    result = self.WaitForNext();
		    }
		    return FteSplitQueueResultToDict(result);
	    });

	py::class_<RayWorkerManager, std::shared_ptr<RayWorkerManager>>(m, "RayWorkerManager")
	    .def(py::init<>())
	    .def("worker_snapshots",
	         [](RayWorkerManager &self) {
		         auto snaps_res = self.worker_snapshots();
		         if (snaps_res.is_err()) {
			         throw duckdb::InternalException(string("worker_snapshots failed: ") + snaps_res.error().what());
		         }
		         auto snaps = std::move(snaps_res.value());
		         py::list out;
		         for (auto &s : snaps) {
			         py::dict d;
			         // worker_id is optional
			         try {
				         if (s.worker_id())
					         d["worker_id"] = *s.worker_id();
				         else
					         d["worker_id"] = py::none();
			         } catch (...) {
				         d["worker_id"] = py::none();
			         }
			         d["num_cpus"] = s.total_num_cpus();
			         d["num_gpus"] = s.total_num_gpus();
			         d["available_num_cpus"] = s.available_num_cpus();
			         d["available_num_gpus"] = s.available_num_gpus();
			         d["total_memory_bytes"] = s.total_memory_bytes();
			         d["available_memory_bytes"] = s.available_memory_bytes();
			         out.append(d);
		         }
		         return out;
	         })
	    .def("shutdown",
	         [](RayWorkerManager &self) {
		         auto res = [&]() {
			         py::gil_scoped_release release;
			         return self.shutdown();
		         }();
		         if (res.is_err()) {
			         throw duckdb::InternalException(res.error().what());
		         }
	         })
	    .def("register_query_owner", &RayWorkerManager::register_query_owner, py::arg("query_id"),
	         py::arg("owner_query_id"))
	    .def("drop_query_fragments", &RayWorkerManager::drop_query_fragments, py::arg("query_id"))
	    .def(
	        "wait_fte_query",
	        [](RayWorkerManager &self, const string &query_id, double timeout_s) {
		        auto res = [&]() {
			        py::gil_scoped_release release;
			        return self.wait_fte_query(query_id, timeout_s);
		        }();
		        if (res.is_err()) {
			        throw duckdb::InternalException(res.error().what());
		        }
	        },
	        py::arg("query_id"), py::arg("timeout_s") = 0.0)
	    .def(
	        "_wait_fte_query_scoped_for_test",
	        [](RayWorkerManager &self, const string &query_id, double timeout_s) {
		        std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash>
		            task_contexts {duckdb::distributed::TaskContext::from_node_context(1, 2, 3)};
		        auto res = [&]() {
			        py::gil_scoped_release release;
			        return self.wait_fte_query(query_id, timeout_s, task_contexts, {});
		        }();
		        if (res.is_err()) {
			        throw duckdb::InternalException(res.error().what());
		        }
	        },
	        py::arg("query_id"), py::arg("timeout_s") = 0.0)
	    .def(
	        "_wait_fte_query_streaming_for_test",
	        [](RayWorkerManager &self, const string &query_id, double timeout_s, int64_t fail_after,
	           int64_t throw_after) {
		        size_t output_count = 0;
		        auto on_output = [&output_count, fail_after,
		                          throw_after](const duckdb::distributed::MaterializedOutput &) {
			        if (throw_after >= 0 && output_count >= static_cast<size_t>(throw_after)) {
				        throw std::runtime_error("planned streaming callback exception");
			        }
			        if (fail_after >= 0 && output_count >= static_cast<size_t>(fail_after)) {
				        return duckdb::distributed::DuckDBResult<void>::err(
				            duckdb::distributed::DuckDBError::external_error("planned streaming callback failure"));
			        }
			        output_count++;
			        return duckdb::distributed::DuckDBResult<void>::ok();
		        };
		        auto res = [&]() {
			        py::gil_scoped_release release;
			        return self.wait_fte_query_streaming(query_id, timeout_s, std::move(on_output));
		        }();
		        if (res.is_err()) {
			        throw duckdb::InternalException(res.error().what());
		        }
		        return output_count;
	        },
	        py::arg("query_id"), py::arg("timeout_s") = 0.0, py::arg("fail_after") = -1, py::arg("throw_after") = -1)
	    .def("fragment_stats",
	         [](RayWorkerManager &self) { return BuildFragmentStatsSummary(self.fragment_stats_by_worker()); })
	    .def("try_autoscale", [](RayWorkerManager &self, py::object bundles_obj) {
		    if (!py::isinstance<py::iterable>(bundles_obj)) {
			    throw py::type_error("bundles must be an iterable of resource dicts");
		    }
		    std::vector<duckdb::distributed::TaskResourceRequest> reqs;
		    for (auto it : bundles_obj) {
			    py::object obj = py::reinterpret_borrow<py::object>(it);
			    py::dict d = obj.cast<py::dict>();
			    duckdb::distributed::ResourceRequest rr;
			    if (py::bool_(d.contains("CPU")))
				    rr.set_num_cpus(py::float_(d["CPU"]));
			    if (py::bool_(d.contains("GPU")))
				    rr.set_num_gpus(py::float_(d["GPU"]));
			    if (py::bool_(d.contains("memory")))
				    rr.set_memory_bytes(py::int_(d["memory"]));
			    reqs.emplace_back(rr);
		    }
		    auto res = self.try_autoscale(reqs);
		    if (res.is_err()) {
			    throw duckdb::InternalException(res.error().what());
		    }
	    });

	// Register the higher-level distributed plan / runner / stream stubs
	py::class_<ResultPartitionStream, std::shared_ptr<ResultPartitionStream>>(m, "ResultPartitionStream")
	    .def("next_nowait", &ResultPartitionStream::next_nowait)
	    .def("set_ready_callback", &ResultPartitionStream::set_ready_callback)
	    .def("arm_ready_notification", &ResultPartitionStream::arm_ready_notification)
	    .def("clear_ready_callback", &ResultPartitionStream::clear_ready_callback);

	// Helper function to create PyPhysicalPlanWrapper from capsule (used by task.cpp)
	// Wraps a raw PhysicalPlan in a DistributedPhysicalPlan for unified execution.
	m.def(
	    "_create_physical_plan_from_capsule",
	    [](py::capsule capsule, py::object query_id_obj, py::object resource_query_id_obj,
	       py::object udf_registrations_obj, py::object udf_actor_handles_obj, py::object connection_snapshot_obj) {
		    auto *plan_ptr = static_cast<std::shared_ptr<duckdb::PhysicalPlan> *>(capsule.get_pointer());
		    if (!plan_ptr || !*plan_ptr) {
			    return PyPhysicalPlanWrapper();
		    }
		    // Wrap raw PhysicalPlan in DistributedPhysicalPlan
		    uint16_t idx = duckdb::distributed::get_query_idx_counter().fetch_add(1);
		    if (query_id_obj.is_none()) {
			    throw duckdb::InternalException("_create_physical_plan_from_capsule requires a non-empty query_id");
		    }
		    auto query_id = query_id_obj.cast<string>();
		    if (query_id.empty()) {
			    throw duckdb::InternalException("_create_physical_plan_from_capsule requires a non-empty query_id");
		    }
		    if (resource_query_id_obj.is_none()) {
			    throw duckdb::InternalException(
			        "_create_physical_plan_from_capsule requires a non-empty resource_query_id");
		    }
		    auto resource_query_id = resource_query_id_obj.cast<string>();
		    if (resource_query_id.empty()) {
			    throw duckdb::InternalException(
			        "_create_physical_plan_from_capsule requires a non-empty resource_query_id");
		    }
		    auto cfg = std::make_shared<duckdb::distributed::DuckDBExecutionConfig>(
		        duckdb::distributed::DuckDBExecutionConfig::from_env());
		    auto distributed_plan =
		        std::make_shared<duckdb::distributed::DistributedPhysicalPlan>(idx, query_id, *plan_ptr, cfg);
		    auto result = PyPhysicalPlanWrapper(distributed_plan, resource_query_id);
		    result.query_id_ = query_id;
		    result.udf_registrations_ = udf_registrations_obj;
		    result.udf_actor_handles_ = udf_actor_handles_obj;
		    result.connection_snapshot_ = connection_snapshot_obj;
		    (void)VaneSessionIdFromSnapshot(result.connection_snapshot_);
		    (void)VaneSessionConfigFromSnapshot(result.connection_snapshot_);
		    return result;
	    },
	    py::arg("capsule"), py::arg("query_id") = py::none(), py::arg("resource_query_id") = py::none(),
	    py::arg("udf_registrations") = py::none(), py::arg("udf_actor_handles") = py::none(),
	    py::arg("connection_snapshot") = py::none(),
	    "Internal helper to create PyPhysicalPlanWrapper from C++ capsule");

	m.def(
	    "_lookup_query_udf_registrations", [](const string &query_id) { return LookupQueryUDFRegistrations(query_id); },
	    py::arg("query_id"));

	m.def(
	    "_lookup_query_udf_actor_handles", [](const string &query_id) { return LookupQueryUDFActorHandles(query_id); },
	    py::arg("query_id"));

	m.def(
	    "_lookup_query_connection_snapshot",
	    [](const string &query_id) { return LookupQueryConnectionSnapshot(query_id); }, py::arg("query_id"));

	m.def(
	    "_prepare_query_snapshot_connection",
	    [](const string &query_id) {
		    auto snapshot = LookupQueryConnectionSnapshot(query_id);
		    if (snapshot.is_none()) {
			    throw std::runtime_error("query connection snapshot is unavailable: " + query_id);
		    }
		    // Create and fully prepare an isolated DatabaseInstance before the
		    // worker admits any task that can deserialize this physical plan.
		    auto connection =
		        CreateConnectionFromBootstrapSnapshot(LookupBootstrapSnapshot(snapshot), false, true, true);
		    PrepareConnectionSnapshotExtensions(connection, snapshot);
		    return connection;
	    },
	    py::arg("query_id"));

	m.def(
	    "_validate_query_snapshot_connection",
	    [](py::object connection, const string &query_id) {
		    auto snapshot = LookupQueryConnectionSnapshot(query_id);
		    if (snapshot.is_none()) {
			    throw std::runtime_error("query connection snapshot is unavailable: " + query_id);
		    }
		    // Admission is verification-only for dynamic artifacts. Provider
		    // discovery and dynamic loading are confined to preparation above.
		    ValidateConnectionSnapshotExtensions(connection, snapshot, true);
	    },
	    py::arg("connection"), py::arg("query_id"));

	m.def(
	    "_register_query_python_replay_state",
	    [](const string &query_id, PyPhysicalPlanWrapper &plan) {
		    if (query_id.empty()) {
			    throw duckdb::InternalException("Query Python replay registration requires a non-empty query_id");
		    }
		    // Worker resource settings are owned by the Ray actor allocation, not
		    // by the source connection. Remove them from every worker-side replay
		    // before the snapshot is registered or used to resolve a database.
		    plan.connection_snapshot_ = PrepareWorkerConnectionSnapshot(plan.connection_snapshot_);
		    // The resource query owns this lifecycle. A retried FTE task can carry a
		    // physical plan created under a different source plan identifier.
		    return RegisterQueryPythonReplayState(query_id, plan.udf_registrations_, plan.udf_actor_handles_,
		                                          plan.connection_snapshot_);
	    },
	    py::arg("query_id"), py::arg("plan"));

	m.def(
	    "_cleanup_query_python_replay_state", [](const string &query_id) { CleanupQueryPythonReplayState(query_id); },
	    py::arg("query_id"));

	m.def(
	    "begin_flight_shuffle_query_execution",
	    [](const string &query_id) {
		    auto result = [&]() {
			    py::gil_scoped_release release;
			    return duckdb::distributed::ShuffleCacheRegistry::Instance().BeginQueryExecution(query_id);
		    }();
		    if (result.is_err()) {
			    throw std::runtime_error(result.error().what());
		    }
	    },
	    py::arg("query_id"));

	m.def(
	    "end_flight_shuffle_query_execution",
	    [](const string &query_id) {
		    auto result = [&]() {
			    py::gil_scoped_release release;
			    return duckdb::distributed::ShuffleCacheRegistry::Instance().EndQueryExecution(query_id);
		    }();
		    if (result.is_err()) {
			    throw std::runtime_error(result.error().what());
		    }
	    },
	    py::arg("query_id"));

	m.def(
	    "close_flight_shuffle_query",
	    [](const string &query_id) {
		    auto result = [&]() {
			    py::gil_scoped_release release;
			    return duckdb::distributed::ShuffleCacheRegistry::Instance().CloseQuery(query_id);
		    }();
		    if (result.is_err()) {
			    throw std::runtime_error(result.error().what());
		    }
	    },
	    py::arg("query_id"));

	m.def(
	    "flight_shuffle_query_status",
	    [](const string &query_id) {
		    auto status = [&]() {
			    py::gil_scoped_release release;
			    return duckdb::distributed::ShuffleCacheRegistry::Instance().QueryStatus(query_id);
		    }();
		    py::dict out;
		    out["cleanup_pending"] = status.cleanup_pending;
		    out["active_executions"] = status.active_executions;
		    return out;
	    },
	    py::arg("query_id"));

	m.def(
	    "cleanup_flight_shuffle_for_query",
	    [](const string &query_id, py::object cleanup_connection, const string &connection_snapshot_query_id,
	       bool apply_snapshot_s3_credentials, py::object effective_session_config) {
		    py::dict out;
		    if (query_id.empty()) {
			    out["registry_entries_removed"] = 0;
			    out["storage_entries_removed"] = 0;
			    out["cleanup_errors"] = 0;
			    out["cleanup_storage_required"] = 0;
			    out["cleanup_pending"] = 0;
			    out["active_executions"] = 0;
			    out["last_error"] = "";
			    return out;
		    }
		    py::object resolved_cleanup_connection = cleanup_connection;
		    // Keep the resolved DatabaseInstance alive until after storage drops
		    // its FileSystem and FileOpener references.
		    std::shared_ptr<duckdb::distributed::ShuffleStorage> cleanup_storage;
		    if (!cleanup_connection.is_none()) {
			    auto &conn_wrapper = ExtractPyConnectionWrapper(cleanup_connection);
			    if (conn_wrapper.con.ConnectionIsClosed()) {
				    throw std::runtime_error("shuffle cleanup connection is closed");
			    }
			    py::object snapshot = py::none();
			    if (!connection_snapshot_query_id.empty()) {
				    snapshot = LookupQueryConnectionSnapshot(connection_snapshot_query_id);
				    if (!snapshot.is_none()) {
					    resolved_cleanup_connection = ResolveFlightShuffleCleanupConnection(
					        cleanup_connection, snapshot, effective_session_config, apply_snapshot_s3_credentials);
				    } else {
					    ApplyEffectiveVaneSessionConfig(conn_wrapper, effective_session_config);
				    }
				    // Concurrent idempotent teardown can retire replay state after
				    // this cleanup cursor was configured. Continue with its
				    // sanitized session baseline: remaining storage failures stay
				    // visible and retryable in the registry.
			    } else {
				    ApplyEffectiveVaneSessionConfig(conn_wrapper, effective_session_config);
			    }
			    auto &resolved_wrapper = ExtractPyConnectionWrapper(resolved_cleanup_connection);
			    auto &context = *resolved_wrapper.con.GetConnection().context;
			    auto &fs = *duckdb::DBConfig::GetConfig(context).file_system;
			    auto *opener = duckdb::ClientData::Get(context).file_opener.get();
			    cleanup_storage = duckdb::distributed::MakeDuckDBFileSystemShuffleStorage(fs, opener);
		    } else if (!connection_snapshot_query_id.empty()) {
			    throw std::runtime_error("query connection snapshot requires a shuffle cleanup connection");
		    }
		    auto cleanup_result = [&]() {
			    py::gil_scoped_release release;
			    return duckdb::distributed::ShuffleCacheRegistry::Instance().RemoveAndCleanupByQuery(query_id,
			                                                                                         cleanup_storage);
		    }();
		    out["registry_entries_removed"] = cleanup_result.registry_entries_removed;
		    out["storage_entries_removed"] = cleanup_result.storage_entries_removed;
		    out["cleanup_errors"] = cleanup_result.cleanup_errors;
		    out["cleanup_storage_required"] = cleanup_result.cleanup_storage_required;
		    out["cleanup_pending"] = cleanup_result.cleanup_pending;
		    out["active_executions"] = cleanup_result.active_executions;
		    out["last_error"] = cleanup_result.last_error;
		    return out;
	    },
	    py::arg("query_id"), py::arg("cleanup_connection") = py::none(), py::arg("connection_snapshot_query_id") = "",
	    py::arg("apply_snapshot_s3_credentials") = true, py::arg("effective_session_config") = py::none());

	m.def(
	    "_resolve_flight_shuffle_cleanup_connection_for_test",
	    [](py::object cleanup_connection, const string &connection_snapshot_query_id,
	       py::object effective_session_config, bool apply_snapshot_s3_credentials) {
		    auto snapshot = LookupQueryConnectionSnapshot(connection_snapshot_query_id);
		    if (snapshot.is_none()) {
			    throw std::runtime_error("query connection snapshot is unavailable");
		    }
		    auto resolved_connection = ResolveFlightShuffleCleanupConnection(
		        cleanup_connection, snapshot, effective_session_config, apply_snapshot_s3_credentials);
		    auto &resolved_wrapper = ExtractPyConnectionWrapper(resolved_connection);
		    py::dict settings;
		    for (const auto &name :
		         {"s3_endpoint", "s3_region", "s3_access_key_id", "s3_secret_access_key", "s3_session_token"}) {
			    settings[py::str(name)] = py::str(QueryConnectionSettingForTest(resolved_wrapper, name));
		    }
		    settings[py::str("reused_input")] = py::bool_(resolved_connection.ptr() == cleanup_connection.ptr());
		    return settings;
	    },
	    py::arg("cleanup_connection"), py::arg("connection_snapshot_query_id"),
	    py::arg("effective_session_config") = py::none(), py::arg("apply_snapshot_s3_credentials") = true);

	m.def(
	    "retire_flight_shuffle_query",
	    [](const string &query_id) {
		    auto result = [&]() {
			    py::gil_scoped_release release;
			    return duckdb::distributed::ShuffleCacheRegistry::Instance().RetireQuery(query_id);
		    }();
		    if (result.is_err()) {
			    throw std::runtime_error(result.error().what());
		    }
	    },
	    py::arg("query_id"));

	try {
		py::module_ atexit = py::module_::import("atexit");
		atexit.attr("register")(py::cpp_function([]() { CleanupAllQueryPythonReplayState(); }));
	} catch (const py::error_already_set &) {
		// Best-effort cleanup hook. The maps are still explicitly cleared by
		// execution paths that consume the replay state.
	}

	py::class_<PyPhysicalPlanWrapper>(m, "DistributedPhysicalPlan")
	    .def("idx", &PyPhysicalPlanWrapper::idx)
	    .def("resource_query_id", &PyPhysicalPlanWrapper::resource_query_id)
	    .def("session_id", &PyPhysicalPlanWrapper::session_id)
	    .def("session_config", &PyPhysicalPlanWrapper::session_config)
	    .def("has_explicit_s3_credentials", &PyPhysicalPlanWrapper::has_explicit_s3_credentials)
	    .def("has_root", &PyPhysicalPlanWrapper::has_root)
	    .def("clone", &PyPhysicalPlanWrapper::clone, py::arg("conn") = py::none())
	    .def("num_partitions", &PyPhysicalPlanWrapper::num_partitions)
	    .def("repr_ascii", &PyPhysicalPlanWrapper::repr_ascii)
	    .def("repr_mermaid", &PyPhysicalPlanWrapper::repr_mermaid)
	    .def("scan_split_batch_map", &PyPhysicalPlanWrapper::scan_split_batch_map)
	    .def("collect_query_resource_graph_metadata", &PyPhysicalPlanWrapper::collect_query_resource_graph_metadata,
	         py::arg("conn") = py::none())
	    .def("collect_udf_nodes", &PyPhysicalPlanWrapper::collect_udf_nodes, py::arg("conn") = py::none())
	    .def("collect_vllm_nodes", &PyPhysicalPlanWrapper::collect_vllm_nodes, py::arg("conn") = py::none())
	    .def("set_udf_actor_handles", &PyPhysicalPlanWrapper::set_udf_actor_handles, py::arg("handles_map"),
	         py::arg("conn") = py::none())
	    .def(
	        "_validate_serializable_for_submission",
	        [](const PyPhysicalPlanWrapper &p) { p.validate_serializable_for_submission(); },
	        "Validate worker-executable physical roots before distributed query registration.")
	    .def(py::pickle(
	        // __getstate__: serialize the plan
	        [](const PyPhysicalPlanWrapper &p) {
		        // An absent plan is an explicit optional state. A materialized
		        // or deferred root must always carry a non-empty payload.
		        if (!p.has_root()) {
			        if (!p.serialized_root_.empty()) {
				        return py::make_tuple(true, py::bytes(p.serialized_root_), p.query_id_, p.resource_query_id_,
				                              p.udf_registrations_, p.udf_actor_handles_, p.connection_snapshot_);
			        }
			        return py::make_tuple(false, py::bytes(""), p.query_id_, p.resource_query_id_, p.udf_registrations_,
			                              p.udf_actor_handles_, p.connection_snapshot_);
		        }

		        auto serialized_root = p.serialize_root_for_clone();
		        return py::make_tuple(true, py::bytes(serialized_root), p.query_id_, p.resource_query_id_,
		                              p.udf_registrations_, p.udf_actor_handles_, p.connection_snapshot_);
	        },
	        // __setstate__: store deferred bytes for later deserialization
	        [](py::tuple t) {
		        if (t.size() != 7) {
			        throw duckdb::InternalException("Invalid state for PyPhysicalPlanWrapper pickle");
		        }
		        bool has_data = t[0].cast<bool>();
		        py::bytes serialized = t[1].cast<py::bytes>();
		        string data_str = serialized;
		        if (has_data && data_str.empty()) {
			        throw duckdb::InternalException(
			            "Invalid PyPhysicalPlanWrapper pickle: serialized root is marked present but empty");
		        }
		        if (!has_data && !data_str.empty()) {
			        throw duckdb::InternalException(
			            "Invalid PyPhysicalPlanWrapper pickle: absent root carries serialized data");
		        }
		        string query_id = t[2].cast<string>();

		        PyPhysicalPlanWrapper result;
		        result.query_id_ = std::move(query_id);
		        result.resource_query_id_ = t[3].cast<string>();
		        if (result.resource_query_id_.empty()) {
			        throw duckdb::InternalException(
			            "Invalid PyPhysicalPlanWrapper pickle: resource_query_id must not be empty");
		        }
		        result.udf_registrations_ = t[4];
		        result.udf_actor_handles_ = t[5];
		        result.connection_snapshot_ = t[6];
		        (void)VaneSessionIdFromSnapshot(result.connection_snapshot_);
		        (void)VaneSessionConfigFromSnapshot(result.connection_snapshot_);
		        result.ensure_plan_identity();

		        if (has_data) {
			        result.serialized_root_ = std::move(data_str);
		        }
		        return result;
	        }));

	m.def(
	    "_make_non_serializable_physical_plan_for_test",
	    [](const string &query_id) {
		    if (query_id.empty()) {
			    throw duckdb::InternalException("test physical plan requires a non-empty query_id");
		    }
		    auto physical_plan = std::make_shared<duckdb::PhysicalPlan>(duckdb::Allocator::DefaultAllocator());
		    auto &root = physical_plan->Make<PhysicalNonSerializableOperatorForTest>();
		    physical_plan->SetRoot(root);
		    uint16_t idx = duckdb::distributed::get_query_idx_counter().fetch_add(1);
		    auto config = std::make_shared<duckdb::distributed::DuckDBExecutionConfig>(
		        duckdb::distributed::DuckDBExecutionConfig::from_env());
		    auto distributed_plan = std::make_shared<duckdb::distributed::DistributedPhysicalPlan>(
		        idx, query_id, std::move(physical_plan), std::move(config));
		    return PyPhysicalPlanWrapper(std::move(distributed_plan));
	    },
	    py::arg("query_id"), "Create an intentionally non-serializable physical plan for native tests.");

	m.def(
	    "_make_coordinator_only_extension_write_plan_for_test",
	    [](const string &query_id, bool fail_finalize, py::object conn_obj) {
		    if (query_id.empty()) {
			    throw duckdb::InternalException("test extension write plan requires a non-empty query_id");
		    }
		    auto physical_plan = std::make_shared<duckdb::PhysicalPlan>(duckdb::Allocator::DefaultAllocator());
		    vector<LogicalType> child_types {LogicalType::BIGINT};
		    auto &child = physical_plan->Make<PhysicalDummyScan>(std::move(child_types), 1);
		    auto &root = physical_plan->Make<PhysicalCoordinatorOnlyExtensionWriteForTest>(fail_finalize);
		    root.children.push_back(child);
		    physical_plan->SetRoot(root);
		    uint16_t idx = duckdb::distributed::get_query_idx_counter().fetch_add(1);
		    auto config = std::make_shared<duckdb::distributed::DuckDBExecutionConfig>(
		        duckdb::distributed::DuckDBExecutionConfig::from_env());
		    auto distributed_plan = std::make_shared<duckdb::distributed::DistributedPhysicalPlan>(
		        idx, query_id, std::move(physical_plan), std::move(config));
		    PyPhysicalPlanWrapper result(std::move(distributed_plan));
		    if (!conn_obj.is_none()) {
			    if (py::hasattr(conn_obj, "c")) {
				    conn_obj = conn_obj.attr("c");
			    }
			    auto &py_conn = conn_obj.cast<DuckDBPyConnection &>();
			    result.connection_snapshot_ = CaptureConnectionSnapshot(py_conn, conn_obj);
		    }
		    return result;
	    },
	    py::arg("query_id"), py::arg("fail_finalize") = false, py::arg("conn") = py::none(),
	    "Create a coordinator-only extension write plan for submission-boundary tests.");

	m.def(
	    "_register_coordinator_only_extension_write_for_test",
	    [](py::object conn_obj) {
		    if (py::hasattr(conn_obj, "c")) {
			    conn_obj = conn_obj.attr("c");
		    }
		    auto &py_conn = conn_obj.cast<DuckDBPyConnection &>();
		    auto &db_conn = py_conn.con.GetConnection();
		    if (!db_conn.context) {
			    throw py::value_error("test extension write registration requires an open DuckDB connection");
		    }
		    RegisterCoordinatorOnlyExtensionWriteForTest(*db_conn.context);
	    },
	    py::arg("conn"), "Register the coordinator-only callback write contract for native tests.");

	m.def(
	    "_make_worker_task_for_test",
	    [](bool has_plan, py::object query_id_obj) {
		    duckdb::distributed::DuckPhysicalPlanRef physical_plan;
		    if (has_plan) {
			    physical_plan = std::make_shared<duckdb::PhysicalPlan>(duckdb::Allocator::DefaultAllocator());
		    }
		    auto config = std::make_shared<duckdb::distributed::DuckDBExecutionConfig>(
		        duckdb::distributed::DuckDBExecutionConfig::from_env());
		    std::unordered_map<string, string> context;
		    if (!query_id_obj.is_none()) {
			    context["query_id"] = query_id_obj.cast<string>();
		    }
		    duckdb::distributed::WorkerTask task(duckdb::distributed::TaskContext(), std::move(physical_plan),
		                                         std::move(config), std::move(context), "WorkerTaskForTest");
		    return RayWorkerTask(std::move(task));
	    },
	    py::arg("has_plan") = false, py::arg("query_id") = py::none(),
	    "Create a worker task with explicit plan/query-id presence for native tests.");

	m.def(
	    "_make_worker_task_from_plan_for_test",
	    [](const PyPhysicalPlanWrapper &plan, const string &execution_query_id, const string &resource_query_id) {
		    if (!plan.plan_ || !plan.plan_->physical_plan() || !plan.plan_->physical_plan()->HasRoot()) {
			    throw duckdb::InternalException("test worker task requires a physical plan with a root");
		    }
		    if (execution_query_id.empty() || resource_query_id.empty()) {
			    throw duckdb::InternalException(
			        "test worker task requires non-empty execution_query_id and resource_query_id");
		    }
		    auto config = std::make_shared<duckdb::distributed::DuckDBExecutionConfig>(
		        duckdb::distributed::DuckDBExecutionConfig::from_env());
		    std::unordered_map<string, string> context {
		        {"query_id", execution_query_id},
		        {"resource_query_id", resource_query_id},
		    };
		    duckdb::distributed::WorkerTask task(duckdb::distributed::TaskContext(), plan.plan_->physical_plan(),
		                                         std::move(config), std::move(context), "WorkerTaskFromPlanForTest");
		    return RayWorkerTask(std::move(task));
	    },
	    py::arg("plan"), py::arg("execution_query_id"), py::arg("resource_query_id"),
	    "Create a worker task from a physical plan with distinct execution and resource query IDs.");

	m.def(
	    "_submission_error_owner_query_id_for_test",
	    [](const string &execution_query_id, py::object resource_query_id_obj) {
		    std::unordered_map<string, string> context {{"query_id", execution_query_id}};
		    if (!resource_query_id_obj.is_none()) {
			    context["resource_query_id"] = resource_query_id_obj.cast<string>();
		    }
		    auto config = std::make_shared<duckdb::distributed::DuckDBExecutionConfig>(
		        duckdb::distributed::DuckDBExecutionConfig::from_env());
		    std::vector<duckdb::distributed::WorkerTask> tasks;
		    tasks.emplace_back(duckdb::distributed::TaskContext(), duckdb::distributed::DuckPhysicalPlanRef(),
		                       std::move(config), std::move(context), "SubmissionErrorOwnerForTest");
		    return duckdb::distributed::python::ray::SubmissionErrorOwnerQueryId(tasks, execution_query_id);
	    },
	    py::arg("execution_query_id"), py::arg("resource_query_id") = py::none(),
	    "Resolve the query that owns a retained submission exception.");

	py::class_<PyLogicalPlan>(m, "PyLogicalPlan")
	    .def_static("from_duckdb_relation",
	                [](py::object relation_obj, py::object query_id_obj) {
		                try {
			                return LogicalPlanFromDuckDBRelation(std::move(relation_obj), std::move(query_id_obj),
			                                                     DuckDBRelationPlanKind::READ);
		                } catch (const py::type_error &) {
			                throw;
		                } catch (const py::error_already_set &) {
			                throw;
		                } catch (const std::exception &ex) {
			                throw py::value_error(ex.what());
		                }
	                })
	    .def_static("from_duckdb_write_relation",
	                [](py::object relation_obj, py::object query_id_obj) {
		                try {
			                return LogicalPlanFromDuckDBRelation(std::move(relation_obj), std::move(query_id_obj),
			                                                     DuckDBRelationPlanKind::WRITE);
		                } catch (const py::type_error &) {
			                throw;
		                } catch (const py::error_already_set &) {
			                throw;
		                } catch (const std::exception &ex) {
			                throw py::value_error(ex.what());
		                }
	                })
	    .def_static("from_duckdb_datasink_relation",
	                [](py::object relation_obj, py::object query_id_obj) {
		                try {
			                return LogicalPlanFromDuckDBRelation(std::move(relation_obj), std::move(query_id_obj),
			                                                     DuckDBRelationPlanKind::DATA_SINK);
		                } catch (const py::type_error &) {
			                throw;
		                } catch (const py::error_already_set &) {
			                throw;
		                } catch (const std::exception &ex) {
			                throw py::value_error(ex.what());
		                }
	                })
	    .def("idx", &PyLogicalPlan::idx)
	    .def("session_id", &PyLogicalPlan::session_id)
	    .def("session_config", &PyLogicalPlan::session_config)
	    .def("has_explicit_s3_credentials", &PyLogicalPlan::has_explicit_s3_credentials)
	    .def("to_physical_plan", &PyLogicalPlan::to_physical_plan, py::arg("conn") = py::none(),
	         py::arg("effective_session_config") = py::none())
	    .def(py::pickle(
	        [](const PyLogicalPlan &p) {
		        if (p.serialized_logical_plan_.empty()) {
			        throw duckdb::InternalException("PyLogicalPlan missing serialized logical plan");
		        }
		        (void)DecodeLogicalPlanEnvelope(p.serialized_logical_plan_);
		        return py::make_tuple(p.query_id_, py::bytes(p.serialized_logical_plan_), p.udf_registrations_,
		                              p.connection_snapshot_);
	        },
	        [](py::tuple t) {
		        if (t.size() != 4)
			        throw duckdb::InternalException("Invalid state for PyLogicalPlan");
		        string query_id = py::cast<string>(t[0]);
		        py::bytes serialized_bytes = py::cast<py::bytes>(t[1]);
		        string serialized_plan = serialized_bytes;
		        if (serialized_plan.empty()) {
			        throw duckdb::InternalException("PyLogicalPlan deserialization failed: empty logical plan payload");
		        }
		        (void)DecodeLogicalPlanEnvelope(serialized_plan);
		        PyLogicalPlan plan;
		        plan.query_id_ = std::move(query_id);
		        plan.serialized_logical_plan_ = std::move(serialized_plan);
		        plan.udf_registrations_ = t[2];
		        plan.connection_snapshot_ = t[3];
		        (void)VaneSessionIdFromSnapshot(plan.connection_snapshot_);
		        (void)VaneSessionConfigFromSnapshot(plan.connection_snapshot_);
		        return plan;
	        }));

	m.def(
	    "describe_native_progress",
	    [](py::object conn_obj, py::object plan_obj) {
		    if (!py::isinstance<PyPhysicalPlanWrapper>(plan_obj)) {
			    throw py::type_error("plan must be DistributedPhysicalPlan (PyPhysicalPlanWrapper)");
		    }
		    const auto *plan_ptr = plan_obj.cast<PyPhysicalPlanWrapper *>();
		    return DescribeNativeProgress(conn_obj, *plan_ptr);
	    },
	    py::arg("conn"), py::arg("plan"));

	py::class_<PyPhysicalPlanWrapperRunner>(m, "DistributedPhysicalPlanRunner")
	    .def(py::init<>())
	    .def(py::init<py::object>(), py::arg("backend"))
	    .def("_fail_next_extension_result_marshalling_for_test",
	         &PyPhysicalPlanWrapperRunner::fail_next_extension_result_marshalling_for_test)
	    .def(
	        "run_plan",
	        [](PyPhysicalPlanWrapperRunner &self, py::object plan_obj, py::object conn_obj) -> py::object {
		        if (!py::isinstance<PyPhysicalPlanWrapper>(plan_obj)) {
			        throw py::type_error("plan must be DistributedPhysicalPlan (PyPhysicalPlanWrapper)");
		        }
		        const auto *plan_ptr = plan_obj.cast<PyPhysicalPlanWrapper *>();
		        if (!plan_ptr->IsInitialized()) {
			        throw py::value_error(
			            "DistributedPhysicalPlan is uninitialized; construct it via constructor/helper APIs");
		        }

		        auto get_client_context = [plan_ptr](py::object conn_obj,
		                                             duckdb::distributed::python::ray::SafePyObject &keepalive)
		            -> duckdb::shared_ptr<duckdb::ClientContext> {
			        if (!plan_ptr->worker_connection_.is_none()) {
				        try {
					        auto &py_conn = ExtractPyConnectionWrapper(plan_ptr->worker_connection_);
					        auto &db_conn = py_conn.con.GetConnection();
					        if (db_conn.context) {
						        keepalive =
						            duckdb::distributed::python::ray::SafePyObject(plan_ptr->worker_connection_);
						        return db_conn.context;
					        }
				        } catch (...) {
				        }
			        }
			        if (plan_ptr->client_context_) {
				        return plan_ptr->client_context_;
			        }
			        if (conn_obj.is_none()) {
				        return nullptr;
			        }
			        py::object deser_conn = conn_obj;
			        if (py::hasattr(conn_obj, "c")) {
				        deser_conn = conn_obj.attr("c");
			        }
			        try {
				        auto &py_conn = deser_conn.cast<duckdb::DuckDBPyConnection &>();
				        auto &db_conn = py_conn.con.GetConnection();
				        if (!db_conn.context) {
					        return nullptr;
				        }
				        keepalive = duckdb::distributed::python::ray::SafePyObject(deser_conn);
				        return db_conn.context;
			        } catch (...) {
				        return nullptr;
			        }
		        };

		        duckdb::distributed::python::ray::SafePyObject keepalive;
		        auto client_context = get_client_context(conn_obj, keepalive);
		        auto ptr = self.run_plan(*plan_ptr, client_context, std::move(keepalive));
		        return py::cast(ptr, py::return_value_policy::reference_internal);
	        },
	        py::arg("plan"), py::arg("conn") = py::none())
	    .def(
	        "run_copy_plan",
	        [](PyPhysicalPlanWrapperRunner &self, py::object plan_obj, py::object conn_obj,
	           py::object on_execution_started_obj) -> py::object {
		        if (!py::isinstance<PyPhysicalPlanWrapper>(plan_obj)) {
			        throw py::type_error("plan must be DistributedPhysicalPlan (PyPhysicalPlanWrapper)");
		        }
		        const auto *plan_ptr = plan_obj.cast<PyPhysicalPlanWrapper *>();
		        if (!plan_ptr->IsInitialized()) {
			        throw py::value_error(
			            "DistributedPhysicalPlan is uninitialized; construct it via constructor/helper APIs");
		        }

		        auto get_client_context = [plan_ptr](py::object conn_obj,
		                                             duckdb::distributed::python::ray::SafePyObject &keepalive)
		            -> duckdb::shared_ptr<duckdb::ClientContext> {
			        if (!plan_ptr->worker_connection_.is_none()) {
				        try {
					        auto &py_conn = ExtractPyConnectionWrapper(plan_ptr->worker_connection_);
					        auto &db_conn = py_conn.con.GetConnection();
					        if (db_conn.context) {
						        keepalive =
						            duckdb::distributed::python::ray::SafePyObject(plan_ptr->worker_connection_);
						        return db_conn.context;
					        }
				        } catch (...) {
				        }
			        }
			        if (plan_ptr->client_context_) {
				        return plan_ptr->client_context_;
			        }
			        if (conn_obj.is_none()) {
				        return nullptr;
			        }
			        py::object deser_conn = conn_obj;
			        if (py::hasattr(conn_obj, "c")) {
				        deser_conn = conn_obj.attr("c");
			        }
			        try {
				        auto &py_conn = deser_conn.cast<duckdb::DuckDBPyConnection &>();
				        auto &db_conn = py_conn.con.GetConnection();
				        if (!db_conn.context) {
					        return nullptr;
				        }
				        keepalive = duckdb::distributed::python::ray::SafePyObject(deser_conn);
				        return db_conn.context;
			        } catch (...) {
				        return nullptr;
			        }
		        };

		        duckdb::distributed::python::ray::SafePyObject keepalive;
		        auto client_context = get_client_context(conn_obj, keepalive);
		        duckdb::distributed::python::ray::SafePyObject on_execution_started;
		        if (!on_execution_started_obj.is_none()) {
			        on_execution_started =
			            duckdb::distributed::python::ray::SafePyObject(std::move(on_execution_started_obj));
		        }
		        return self.run_copy_plan(*plan_ptr, client_context, std::move(keepalive),
		                                  std::move(on_execution_started));
	        },
	        py::arg("plan"), py::arg("conn") = py::none(), py::arg("on_execution_started") = py::none())
	    .def(
	        "run_datasink_plan",
	        [](PyPhysicalPlanWrapperRunner &self, py::object plan_obj, py::object conn_obj,
	           py::object on_execution_started_obj) -> py::dict {
		        if (!py::isinstance<PyPhysicalPlanWrapper>(plan_obj)) {
			        throw py::type_error("plan must be DistributedPhysicalPlan (PyPhysicalPlanWrapper)");
		        }
		        const auto *plan_ptr = plan_obj.cast<PyPhysicalPlanWrapper *>();
		        if (!plan_ptr->IsInitialized()) {
			        throw py::value_error(
			            "DistributedPhysicalPlan is uninitialized; construct it via constructor/helper APIs");
		        }

		        auto get_client_context = [plan_ptr](py::object connection_obj,
		                                             duckdb::distributed::python::ray::SafePyObject &keepalive)
		            -> duckdb::shared_ptr<duckdb::ClientContext> {
			        if (!plan_ptr->worker_connection_.is_none()) {
				        try {
					        auto &py_conn = ExtractPyConnectionWrapper(plan_ptr->worker_connection_);
					        auto &db_conn = py_conn.con.GetConnection();
					        if (db_conn.context) {
						        keepalive =
						            duckdb::distributed::python::ray::SafePyObject(plan_ptr->worker_connection_);
						        return db_conn.context;
					        }
				        } catch (...) {
				        }
			        }
			        if (plan_ptr->client_context_) {
				        return plan_ptr->client_context_;
			        }
			        if (connection_obj.is_none()) {
				        return nullptr;
			        }
			        py::object resolved_conn = connection_obj;
			        if (py::hasattr(connection_obj, "c")) {
				        resolved_conn = connection_obj.attr("c");
			        }
			        try {
				        auto &py_conn = resolved_conn.cast<duckdb::DuckDBPyConnection &>();
				        auto &db_conn = py_conn.con.GetConnection();
				        if (!db_conn.context) {
					        return nullptr;
				        }
				        keepalive = duckdb::distributed::python::ray::SafePyObject(resolved_conn);
				        return db_conn.context;
			        } catch (...) {
				        return nullptr;
			        }
		        };

		        duckdb::distributed::python::ray::SafePyObject keepalive;
		        auto client_context = get_client_context(conn_obj, keepalive);
		        duckdb::distributed::python::ray::SafePyObject on_execution_started;
		        if (!on_execution_started_obj.is_none()) {
			        on_execution_started =
			            duckdb::distributed::python::ray::SafePyObject(std::move(on_execution_started_obj));
		        }
		        return self.run_datasink_plan(*plan_ptr, client_context, std::move(keepalive),
		                                      std::move(on_execution_started));
	        },
	        py::arg("plan"), py::arg("conn") = py::none(), py::arg("on_execution_started") = py::none())
	    .def(
	        "finalize_copy",
	        [](PyPhysicalPlanWrapperRunner &self, py::list file_infos, py::str copy_spec_key, py::str staging_root,
	           py::object conn_obj) -> py::object {
		        auto get_client_context = [](py::object conn_obj,
		                                     duckdb::distributed::python::ray::SafePyObject &keepalive)
		            -> duckdb::shared_ptr<duckdb::ClientContext> {
			        if (conn_obj.is_none()) {
				        return nullptr;
			        }
			        py::object deser_conn = conn_obj;
			        if (py::hasattr(conn_obj, "c")) {
				        deser_conn = conn_obj.attr("c");
			        }
			        try {
				        auto &py_conn = deser_conn.cast<duckdb::DuckDBPyConnection &>();
				        auto &db_conn = py_conn.con.GetConnection();
				        if (!db_conn.context) {
					        return nullptr;
				        }
				        keepalive = duckdb::distributed::python::ray::SafePyObject(deser_conn);
				        return db_conn.context;
			        } catch (...) {
				        return nullptr;
			        }
		        };

		        duckdb::distributed::python::ray::SafePyObject keepalive;
		        auto client_context = get_client_context(conn_obj, keepalive);
		        return self.finalize_copy_impl(file_infos, copy_spec_key, staging_root, client_context);
	        },
	        py::arg("file_infos"), py::arg("copy_spec"), py::arg("staging_root"), py::arg("conn") = py::none())
	    .def("drop_query_fragments", &PyPhysicalPlanWrapperRunner::drop_query_fragments, py::arg("query_id"))
	    .def(
	        "_register_query_owner_for_test",
	        [](PyPhysicalPlanWrapperRunner &self, const string &query_id, const string &owner_query_id) {
		        self.register_query_owner(query_id, owner_query_id);
	        },
	        py::arg("query_id"), py::arg("owner_query_id"))
	    .def("close_session", &PyPhysicalPlanWrapperRunner::close_session, py::arg("session_id"))
	    .def("warm_up", &PyPhysicalPlanWrapperRunner::warm_up)
	    .def("shutdown",
	         [](PyPhysicalPlanWrapperRunner &self) {
		         py::gil_scoped_release release;
		         self.shutdown();
	         })
	    .def("fragment_stats",
	         [](PyPhysicalPlanWrapperRunner &self) {
		         return BuildFragmentStatsSummary(self.fragment_stats_by_worker());
	         })
	    // Use a single dispatcher for execute_native to avoid pybind11 overload resolution issues
	    .def(
	        "execute_native",
	        [](PyPhysicalPlanWrapperRunner &self, py::object conn_obj, py::object plan_obj,
	           py::object scan_split_batch_obj, py::object exchange_source_task_obj, py::object copy_output_info_obj,
	           py::object exchange_sink_instance_obj, py::object fte_scan_source_queues_obj,
	           py::object fte_exchange_source_queues_obj, py::object dynamic_filter_domains_obj,
	           py::object native_progress_callback_obj, py::object runtime_context_obj,
	           py::object effective_session_config_obj) {
		        string plan_type_name = py::str(py::type::of(plan_obj).attr("__name__")).cast<string>();
		        auto parse_node_id = [](py::handle key, const char *map_name) -> idx_t {
			        if (!py::isinstance<py::str>(key)) {
				        throw py::value_error(string(map_name) + " node_id must be a decimal string");
			        }
			        auto key_str = py::str(key).cast<string>();
			        if (key_str.empty()) {
				        throw py::value_error(string(map_name) + " node_id must not be empty");
			        }
			        if (!std::all_of(key_str.begin(), key_str.end(),
			                         [](char value) { return value >= '0' && value <= '9'; })) {
				        throw py::value_error(string(map_name) + " node_id must contain only decimal digits");
			        }
			        if (key_str.size() > 1 && key_str.front() == '0') {
				        throw py::value_error(string(map_name) + " node_id must use canonical decimal form");
			        }
			        try {
				        auto value = std::stoull(key_str);
				        if (value >= NumericLimits<idx_t>::Maximum()) {
					        throw std::out_of_range("node_id exceeds the valid idx_t range");
				        }
				        return static_cast<idx_t>(value);
			        } catch (const std::exception &ex) {
				        throw py::value_error(string("Invalid ") + map_name + " node_id: " + ex.what());
			        }
		        };
		        std::unordered_map<idx_t, duckdb::distributed::ScanSplitBatch> scan_split_batch_map;
		        bool has_scan_split_batch_map = false;
		        if (!scan_split_batch_obj.is_none()) {
			        if (py::isinstance<py::dict>(scan_split_batch_obj)) {
				        auto dict_obj = scan_split_batch_obj.cast<py::dict>();
				        for (auto item : dict_obj) {
					        auto node_id = parse_node_id(item.first, "scan_split_batch");
					        // Values are raw bytes (py::bytes) from driver context
					        auto val_obj = py::reinterpret_borrow<py::object>(item.second);
					        if (!py::isinstance<py::bytes>(val_obj)) {
						        throw py::value_error("scan_split_batch values must be raw bytes");
					        }
					        auto val_bytes = val_obj.cast<string>();
					        if (val_bytes.empty()) {
						        throw py::value_error("scan_split_batch must not be empty");
					        }
					        try {
						        auto inserted =
						            scan_split_batch_map
						                .emplace(node_id,
						                         duckdb::distributed::ScanSplitBatch::DeserializeFromBytes(val_bytes))
						                .second;
						        if (!inserted) {
							        throw std::invalid_argument("duplicate normalized node_id");
						        }
					        } catch (const std::exception &ex) {
						        throw py::value_error(string("Invalid scan split batch map entry: ") + ex.what());
					        }
				        }
				        has_scan_split_batch_map = !scan_split_batch_map.empty();
			        } else {
				        throw py::value_error("scan_split_batch must be a dict mapping node_id to raw bytes");
			        }
		        }
		        const std::unordered_map<idx_t, duckdb::distributed::ScanSplitBatch> *scan_split_batch_map_ptr =
		            has_scan_split_batch_map ? &scan_split_batch_map : nullptr;

		        std::unordered_map<idx_t, duckdb::distributed::ExchangeSourceTaskDescriptor> exchange_source_task_map;
		        bool has_exchange_source_task_map = false;
		        if (!exchange_source_task_obj.is_none()) {
			        if (py::isinstance<py::dict>(exchange_source_task_obj)) {
				        auto dict_obj = exchange_source_task_obj.cast<py::dict>();
				        for (auto item : dict_obj) {
					        auto node_id = parse_node_id(item.first, "exchange_source_task");
					        auto val_obj = py::reinterpret_borrow<py::object>(item.second);
					        if (!py::isinstance<py::bytes>(val_obj)) {
						        throw py::value_error("exchange_source_task values must be raw bytes");
					        }
					        auto val_bytes = val_obj.cast<string>();
					        if (val_bytes.empty()) {
						        throw py::value_error("exchange_source_task descriptor must not be empty");
					        }
					        try {
						        auto inserted =
						            exchange_source_task_map
						                .emplace(
						                    node_id,
						                    duckdb::distributed::ExchangeSourceTaskDescriptor::DeserializeFromBytes(
						                        val_bytes))
						                .second;
						        if (!inserted) {
							        throw std::invalid_argument("duplicate normalized node_id");
						        }
					        } catch (const std::exception &ex) {
						        throw py::value_error(string("Invalid exchange source task map entry: ") + ex.what());
					        }
				        }
				        has_exchange_source_task_map = !exchange_source_task_map.empty();
			        } else {
				        throw py::value_error("exchange_source_task must be a dict mapping node_id to raw bytes");
			        }
		        }
		        const std::unordered_map<idx_t, duckdb::distributed::ExchangeSourceTaskDescriptor>
		            *exchange_source_task_map_ptr = has_exchange_source_task_map ? &exchange_source_task_map : nullptr;

		        std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>>
		            fte_scan_source_queue_map;
		        bool has_fte_scan_source_queue_map = false;
		        if (!fte_scan_source_queues_obj.is_none()) {
			        if (!py::isinstance<py::dict>(fte_scan_source_queues_obj)) {
				        throw py::value_error("fte_scan_source_queues must be a dict mapping node_id to FteSplitQueue");
			        }
			        auto dict_obj = fte_scan_source_queues_obj.cast<py::dict>();
			        for (auto item : dict_obj) {
				        auto node_id = parse_node_id(item.first, "fte_scan_source_queues");
				        auto value_obj = py::reinterpret_borrow<py::object>(item.second);
				        if (!py::isinstance<duckdb::distributed::FteSplitQueue>(value_obj)) {
					        throw py::value_error("fte_scan_source_queues values must be FteSplitQueue instances");
				        }
				        try {
					        auto queue = value_obj.cast<std::shared_ptr<duckdb::distributed::FteSplitQueue>>();
					        if (!fte_scan_source_queue_map.emplace(node_id, std::move(queue)).second) {
						        throw std::invalid_argument("duplicate normalized node_id");
					        }
				        } catch (const std::exception &ex) {
					        throw py::value_error(string("Invalid FTE scan source queue map entry: ") + ex.what());
				        }
			        }
			        has_fte_scan_source_queue_map = !fte_scan_source_queue_map.empty();
		        }
		        const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>>
		            *fte_scan_source_queue_map_ptr =
		                has_fte_scan_source_queue_map ? &fte_scan_source_queue_map : nullptr;

		        std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>>
		            fte_exchange_source_queue_map;
		        bool has_fte_exchange_source_queue_map = false;
		        if (!fte_exchange_source_queues_obj.is_none()) {
			        if (!py::isinstance<py::dict>(fte_exchange_source_queues_obj)) {
				        throw py::value_error(
				            "fte_exchange_source_queues must be a dict mapping node_id to FteSplitQueue");
			        }
			        auto dict_obj = fte_exchange_source_queues_obj.cast<py::dict>();
			        for (auto item : dict_obj) {
				        auto node_id = parse_node_id(item.first, "fte_exchange_source_queues");
				        auto value_obj = py::reinterpret_borrow<py::object>(item.second);
				        if (!py::isinstance<duckdb::distributed::FteSplitQueue>(value_obj)) {
					        throw py::value_error("fte_exchange_source_queues values must be FteSplitQueue instances");
				        }
				        try {
					        auto queue = value_obj.cast<std::shared_ptr<duckdb::distributed::FteSplitQueue>>();
					        if (!fte_exchange_source_queue_map.emplace(node_id, std::move(queue)).second) {
						        throw std::invalid_argument("duplicate normalized node_id");
					        }
				        } catch (const std::exception &ex) {
					        throw py::value_error(string("Invalid FTE exchange source queue map entry: ") + ex.what());
				        }
			        }
			        has_fte_exchange_source_queue_map = !fte_exchange_source_queue_map.empty();
		        }
		        const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>>
		            *fte_exchange_source_queue_map_ptr =
		                has_fte_exchange_source_queue_map ? &fte_exchange_source_queue_map : nullptr;

		        duckdb::distributed::ExchangeSinkInstanceTaskDescriptor exchange_sink_instance_task;
		        bool has_exchange_sink_instance_task = false;
		        if (!exchange_sink_instance_obj.is_none()) {
			        if (py::isinstance<py::bytes>(exchange_sink_instance_obj)) {
				        auto bytes = exchange_sink_instance_obj.cast<string>();
				        exchange_sink_instance_task =
				            duckdb::distributed::ExchangeSinkInstanceTaskDescriptor::DeserializeFromBytes(bytes);
				        has_exchange_sink_instance_task = true;
			        } else if (py::isinstance<py::dict>(exchange_sink_instance_obj)) {
				        auto d = exchange_sink_instance_obj.cast<py::dict>();
				        py::object sink_handle_obj = py::none();
				        if (d.contains("sink_handle")) {
					        sink_handle_obj = py::reinterpret_borrow<py::object>(d["sink_handle"]);
				        }
				        if (py::isinstance<py::dict>(sink_handle_obj)) {
					        auto sink_handle = sink_handle_obj.cast<py::dict>();
					        if (sink_handle.contains("query_id")) {
						        exchange_sink_instance_task.sink_instance.query_id =
						            py::str(sink_handle["query_id"]).cast<string>();
					        }
					        if (sink_handle.contains("task_partition_id")) {
						        exchange_sink_instance_task.sink_instance.sink_handle.task_partition_id =
						            py::int_(sink_handle["task_partition_id"]).cast<idx_t>();
					        } else if (sink_handle.contains("partition_id")) {
						        exchange_sink_instance_task.sink_instance.sink_handle.task_partition_id =
						            py::int_(sink_handle["partition_id"]).cast<idx_t>();
					        }
				        } else if (d.contains("task_partition_id")) {
					        exchange_sink_instance_task.sink_instance.sink_handle.task_partition_id =
					            py::int_(d["task_partition_id"]).cast<idx_t>();
				        } else if (d.contains("partition_id")) {
					        exchange_sink_instance_task.sink_instance.sink_handle.task_partition_id =
					            py::int_(d["partition_id"]).cast<idx_t>();
				        }

				        if (d.contains("attempt_id")) {
					        exchange_sink_instance_task.sink_instance.attempt_id =
					            py::int_(d["attempt_id"]).cast<idx_t>();
				        }
				        if (d.contains("output_partition_count")) {
					        exchange_sink_instance_task.sink_instance.output_partition_count =
					            py::int_(d["output_partition_count"]).cast<idx_t>();
				        }
				        if (d.contains("output_location")) {
					        exchange_sink_instance_task.sink_instance.output_location =
					            py::str(d["output_location"]).cast<string>();
				        } else if (d.contains("attempt_path")) {
					        exchange_sink_instance_task.sink_instance.output_location =
					            py::str(d["attempt_path"]).cast<string>();
				        }
				        if (d.contains("flight_server_epoch")) {
					        exchange_sink_instance_task.sink_instance.flight_server_epoch =
					            py::str(d["flight_server_epoch"]).cast<string>();
				        }
				        if (d.contains("flight_host")) {
					        exchange_sink_instance_task.sink_instance.flight_host =
					            py::str(d["flight_host"]).cast<string>();
				        }
				        if (d.contains("query_id")) {
					        exchange_sink_instance_task.sink_instance.query_id = py::str(d["query_id"]).cast<string>();
				        }
				        if (d.contains("mark_join_build_summary_valid")) {
					        exchange_sink_instance_task.sink_instance.mark_join_build_summary.valid =
					            py::bool_(d["mark_join_build_summary_valid"]).cast<bool>();
				        }
				        if (d.contains("mark_join_build_has_rows")) {
					        exchange_sink_instance_task.sink_instance.mark_join_build_summary.has_rows =
					            py::bool_(d["mark_join_build_has_rows"]).cast<bool>();
				        }
				        if (d.contains("mark_join_build_has_null")) {
					        exchange_sink_instance_task.sink_instance.mark_join_build_summary.has_null =
					            py::bool_(d["mark_join_build_has_null"]).cast<bool>();
				        }
				        if (!exchange_sink_instance_task.sink_instance.mark_join_build_summary.IsConsistent()) {
					        throw py::value_error("exchange_sink_instance has an invalid MARK join build summary");
				        }
				        has_exchange_sink_instance_task = true;
			        } else {
				        throw py::value_error("exchange_sink_instance must be bytes or dict");
			        }
		        }
		        const duckdb::distributed::ExchangeSinkInstanceTaskDescriptor *exchange_sink_instance_task_ptr =
		            has_exchange_sink_instance_task ? &exchange_sink_instance_task : nullptr;

		        CopyOutputInfo copy_output_info;
		        bool has_copy_output_info = false;
		        if (!copy_output_info_obj.is_none()) {
			        if (!py::isinstance<py::dict>(copy_output_info_obj)) {
				        throw py::value_error("copy_output_info must be a dict with base, run_id, and remote_base");
			        }
			        auto d = copy_output_info_obj.cast<py::dict>();
			        if (d.contains("base")) {
				        copy_output_info.base = py::str(d["base"]).cast<string>();
			        }
			        if (d.contains("run_id")) {
				        copy_output_info.run_id = py::str(d["run_id"]).cast<string>();
			        }
			        if (d.contains("remote_base")) {
				        copy_output_info.remote_base = py::str(d["remote_base"]).cast<string>();
			        }
			        has_copy_output_info = true;
		        }
		        const CopyOutputInfo *copy_output_info_ptr = has_copy_output_info ? &copy_output_info : nullptr;

		        // Check if it's a PyPhysicalPlanWrapper
		        if (py::isinstance<PyPhysicalPlanWrapper>(plan_obj)) {
			        auto &plan = plan_obj.cast<PyPhysicalPlanWrapper &>();
			        PyPhysicalPlanWrapper *exec_plan = &plan;
			        PyPhysicalPlanWrapper deferred_exec_plan;

			        try {
				        // A materialized physical plan owns objects bound to the
				        // DatabaseInstance retained by worker_connection_. Replacing that
				        // connection here can destroy the bound instance while operators
				        // still reference it. Deferred worker plans have no retained
				        // connection and are materialized against conn_obj below.
				        py::object exec_conn = plan.resolve_execution_connection(conn_obj);
				        const bool apply_snapshot_session_config = effective_session_config_obj.is_none();
				        // Establish the exact extension set before applying refreshed AWS
				        // settings. Replay the full snapshot only after the
				        // environment/profile baseline so explicit source-connection
				        // settings retain normal precedence.
				        ValidateConnectionSnapshotExtensions(exec_conn, plan.connection_snapshot_, true);
				        if (ConnectionSnapshotDeclaresExtension(plan.connection_snapshot_, "httpfs")) {
					        ApplyEffectiveVaneSessionConfig(ExtractPyConnectionWrapper(exec_conn),
					                                        effective_session_config_obj);
				        }
				        plan.ensure_connection_snapshot(exec_conn, apply_snapshot_session_config);
				        // Handle deferred deserialization (from pickle round-trip)
				        if (!plan.has_root() && !plan.serialized_root_.empty()) {
					        // Materialize into a temporary wrapper so any physical-plan
					        // state created during execution is destroyed before the
					        // caller closes the cursor/client context. Mutating the
					        // caller-owned deferred wrapper in place can leave it
					        // holding an executed plan whose destructor runs after the
					        // cursor is closed, causing use-after-free crashes.
					        deferred_exec_plan.query_id_ = plan.query_id_;
					        deferred_exec_plan.resource_query_id_ = plan.resource_query_id_;
					        deferred_exec_plan.udf_registrations_ = plan.udf_registrations_;
					        deferred_exec_plan.udf_actor_handles_ = plan.udf_actor_handles_;
					        deferred_exec_plan.connection_snapshot_ = plan.connection_snapshot_;
					        deferred_exec_plan.serialized_root_ = plan.serialized_root_;
					        deferred_exec_plan.worker_connection_ = exec_conn;
					        deferred_exec_plan.materialize_deferred_root(exec_conn, apply_snapshot_session_config);
					        exec_plan = &deferred_exec_plan;
				        }

				        if (!exec_plan->has_root()) {
					        throw py::value_error("PyPhysicalPlanWrapper has no root after deserialization");
				        }
				        exec_plan->worker_connection_ = exec_conn;
				        exec_plan->client_context_ = ExtractPyConnectionWrapper(exec_conn).con.GetConnection().context;
				        exec_plan->ensure_connection_snapshot(exec_conn, apply_snapshot_session_config);
				        exec_plan->apply_udf_actor_handles();
				        auto result = self.execute_native_impl(
				            exec_conn, exec_plan->plan_->physical_plan(), plan.idx(), plan.resource_query_id_,
				            scan_split_batch_map_ptr, exchange_source_task_map_ptr, exchange_sink_instance_task_ptr,
				            fte_scan_source_queue_map_ptr, fte_exchange_source_queue_map_ptr, copy_output_info_ptr,
				            dynamic_filter_domains_obj, native_progress_callback_obj, runtime_context_obj);
				        return result;
			        } catch (...) {
				        throw;
			        }
		        }

		        // Unknown type
		        throw py::type_error("execute_native expects PyPhysicalPlanWrapper (DistributedPhysicalPlan), got " +
		                             plan_type_name);
	        },
	        py::arg("conn"), py::arg("plan"), py::arg("scan_split_batch") = py::none(),
	        py::arg("exchange_source_task") = py::none(), py::arg("copy_output_info") = py::none(),
	        py::arg("exchange_sink_instance") = py::none(), py::arg("fte_scan_source_queues") = py::none(),
	        py::arg("fte_exchange_source_queues") = py::none(), py::arg("dynamic_filter_domains") = py::none(),
	        py::arg("native_progress_callback") = py::none(), py::arg("runtime_context") = py::none(),
	        py::arg("effective_session_config") = py::none(), "Execute physical plan using DuckDB's native Executor");

	// Merge the independently scheduled splits assigned to one worker attempt
	// into its transport batch.
	m.def(
	    "merge_scan_split_batches",
	    [](const py::list &bytes_list) -> py::bytes {
		    using namespace duckdb::distributed;
		    if (py::len(bytes_list) == 0) {
			    throw py::value_error("merge_scan_split_batches requires at least one batch");
		    }
		    ScanSplitBatch merged;
		    bool has_batch = false;
		    for (auto item : bytes_list) {
			    py::bytes b = item.cast<py::bytes>();
			    string raw(b);
			    auto batch = ScanSplitBatch::DeserializeFromBytes(raw);
			    if (!has_batch) {
				    merged = std::move(batch);
				    has_batch = true;
			    } else {
				    merged.Merge(std::move(batch));
			    }
		    }
		    auto result = merged.SerializeToBytes();
		    return py::bytes(result);
	    },
	    py::arg("bytes_list"), "Merge compatible serialized scan split batches into one worker assignment.");

	m.def(
	    "split_scan_split_batch",
	    [](py::bytes bytes_obj) -> py::list {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto batch = ScanSplitBatch::DeserializeFromBytes(raw);
		    py::list result;
		    for (auto &singleton : batch.Explode()) {
			    const auto &split = singleton.splits[0];
			    py::object estimated_bytes = py::none();
			    if (split.estimated_bytes.IsValid()) {
				    estimated_bytes = py::int_(split.estimated_bytes.GetIndex());
			    }
			    result.append(py::make_tuple(py::str(split.split_id), py::bytes(singleton.SerializeToBytes()),
			                                 std::move(estimated_bytes)));
		    }
		    return result;
	    },
	    py::arg("bytes"),
	    "Explode a serialized scan split batch into (split_id, singleton_batch, estimated_bytes) tuples.");

	m.def(
	    "exchange_source_task_partition_indices",
	    [](py::bytes bytes_obj) {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto desc = ExchangeSourceTaskDescriptor::DeserializeFromBytes(raw);
		    return desc.partition_indices;
	    },
	    py::arg("bytes"), "Return partition_indices from a raw ExchangeSourceTaskDescriptor.");

	m.def(
	    "exchange_source_task_replicated",
	    [](py::bytes bytes_obj) {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto desc = ExchangeSourceTaskDescriptor::DeserializeFromBytes(raw);
		    return desc.replicated;
	    },
	    py::arg("bytes"), "Return whether a raw ExchangeSourceTaskDescriptor is replicated.");

	m.def(
	    "exchange_source_task_mark_join_build_summary_for_test",
	    [](py::bytes bytes_obj) {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto desc = ExchangeSourceTaskDescriptor::DeserializeFromBytes(raw);
		    py::dict out;
		    out["valid"] = desc.mark_join_build_summary.valid;
		    out["has_rows"] = desc.mark_join_build_summary.has_rows;
		    out["has_null"] = desc.mark_join_build_summary.has_null;
		    return out;
	    },
	    py::arg("bytes"), "Return MARK build summary metadata from an exchange source descriptor.");

	m.def(
	    "exchange_source_task_logical_identity",
	    [](py::bytes bytes_obj) {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto desc = ExchangeSourceTaskDescriptor::DeserializeFromBytes(raw);
		    std::set<idx_t> source_task_partition_ids;
		    for (const auto &handle : desc.source_handles) {
			    if (handle.source_task_partition_id == DConstants::INVALID_INDEX) {
				    throw py::value_error(
				        "exchange source handle is missing its stable source task partition identity");
			    }
			    source_task_partition_ids.insert(handle.source_task_partition_id);
		    }
		    if (source_task_partition_ids.empty()) {
			    throw py::value_error("exchange source task has no source task partition identities");
		    }

		    py::dict out;
		    out["partition_indices"] = desc.partition_indices;
		    out["source_task_partition_ids"] =
		        vector<idx_t>(source_task_partition_ids.begin(), source_task_partition_ids.end());
		    out["source_partition_count"] = desc.source_partition_count;
		    out["source_task_count"] = desc.source_task_count;
		    out["replicated"] = desc.replicated;
		    return out;
	    },
	    py::arg("bytes"),
	    "Return completion-order-independent logical identity metadata from an exchange source task.");

	m.def(
	    "read_committed_copy_direct_write_result",
	    [](const std::string &base_path, const std::string &run_id, py::object conn_obj) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    auto read_res = WithCopyRecoveryContext(std::move(conn_obj), [&](ClientContext &context) {
			    auto &fs = FileSystem::GetFileSystem(context);
			    return ReadCommittedDistributedCopyDirectWriteResult(fs, base_path, run_id);
		    });
		    if (read_res.is_err()) {
			    throw py::value_error(read_res.error().what());
		    }
		    auto result = std::move(read_res).value();

		    py::dict out;
		    out["rows_copied"] = result.rows_copied;
		    py::list files;
		    for (const auto &info : result.files) {
			    py::dict entry;
			    entry["staging_path"] = info.staging_path;
			    entry["worker_output_path"] = info.staging_path;
			    entry["final_path"] = info.final_path;
			    entry["row_count"] = info.row_count;
			    entry["file_size_bytes"] = info.file_size_bytes;
			    files.append(entry);
		    }
		    out["files"] = files;
		    AppendDistributedCopyResultMetadata(out, result);
		    return out;
	    },
	    py::arg("base_path"), py::arg("run_id"), py::arg("conn") = py::none(),
	    "Read a committed direct-write distributed COPY result through its committed manifest.");

	m.def(
	    "cleanup_uncommitted_copy_direct_write_run",
	    [](const std::string &base_path, const std::string &run_id) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    auto cleanup_res = CleanupDistributedCopyUncommittedDirectWriteRun(fs, base_path, run_id);
		    if (cleanup_res.is_err()) {
			    throw py::value_error(cleanup_res.error().what());
		    }
		    auto cleanup = std::move(cleanup_res).value();
		    py::dict out;
		    out["skipped_committed"] = cleanup.skipped_committed;
		    out["data_run_dir_existed"] = cleanup.data_run_dir_existed;
		    out["data_run_dir_removed"] = cleanup.data_run_dir_removed;
		    out["commit_dir_existed"] = cleanup.commit_dir_existed;
		    out["commit_dir_removed"] = cleanup.commit_dir_removed;
		    return out;
	    },
	    py::arg("base_path"), py::arg("run_id"),
	    "Cleanup a lifecycle-registered uncommitted direct-write distributed COPY run.");

	m.def(
	    "inspect_copy_direct_write_run",
	    [](const std::string &base_path, const std::string &run_id, py::object conn_obj) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    auto inspection_res = WithCopyRecoveryContext(std::move(conn_obj), [&](ClientContext &context) {
			    auto &fs = FileSystem::GetFileSystem(context);
			    return InspectDistributedCopyDirectWriteRun(fs, base_path, run_id);
		    });
		    if (inspection_res.is_err()) {
			    throw py::value_error(inspection_res.error().what());
		    }
		    auto inspection = std::move(inspection_res).value();
		    py::dict out;
		    switch (inspection.state) {
		    case DistributedCopyDirectWriteRunState::COMMITTED:
			    out["state"] = "COMMITTED";
			    break;
		    case DistributedCopyDirectWriteRunState::UNCOMMITTED:
			    out["state"] = "UNCOMMITTED";
			    break;
		    case DistributedCopyDirectWriteRunState::UNKNOWN:
			    out["state"] = "UNKNOWN";
			    break;
		    }
		    out["safe_to_retry"] = false;
		    out["error"] = inspection.error;
		    out["copy_output_base_path"] = inspection.base_path;
		    out["copy_output_run_id"] = inspection.run_id;
		    out["copy_output_commit_dir"] = inspection.paths.commit_dir;
		    out["copy_output_manifest_path"] = inspection.paths.manifest_path;
		    out["copy_output_committed_marker_path"] = inspection.paths.committed_marker_path;
		    out["copy_output_lifecycle_path"] = inspection.paths.lifecycle_path;
		    py::list files;
		    if (inspection.state == DistributedCopyDirectWriteRunState::COMMITTED) {
			    out["rows_copied"] = inspection.committed_result.rows_copied;
			    for (const auto &info : inspection.committed_result.files) {
				    py::dict entry;
				    entry["staging_path"] = info.staging_path;
				    entry["worker_output_path"] = info.staging_path;
				    entry["final_path"] = info.final_path;
				    entry["row_count"] = info.row_count;
				    entry["file_size_bytes"] = info.file_size_bytes;
				    files.append(entry);
			    }
		    } else {
			    out["rows_copied"] = py::none();
		    }
		    out["files"] = files;
		    return out;
	    },
	    py::arg("base_path"), py::arg("run_id"), py::arg("conn") = py::none(),
	    "Inspect a direct-write COPY run without making an uncommitted run safe to retry.");

	m.def(
	    "force_abort_copy_direct_write_run",
	    [](const std::string &base_path, const std::string &run_id, py::object conn_obj) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    auto abort_res = WithCopyRecoveryContext(std::move(conn_obj), [&](ClientContext &context) {
			    auto &fs = FileSystem::GetFileSystem(context);
			    return ForceAbortDistributedCopyDirectWriteRun(fs, base_path, run_id);
		    });
		    if (abort_res.is_err()) {
			    throw py::value_error(abort_res.error().what());
		    }
		    auto cleanup = std::move(abort_res).value();
		    py::dict out;
		    out["state"] = "ABORTED";
		    out["safe_to_retry"] = true;
		    out["copy_output_base_path"] = base_path;
		    out["copy_output_run_id"] = run_id;
		    out["data_run_dir_existed"] = cleanup.data_run_dir_existed;
		    out["data_run_dir_removed"] = cleanup.data_run_dir_removed;
		    out["commit_dir_existed"] = cleanup.commit_dir_existed;
		    out["commit_dir_removed"] = cleanup.commit_dir_removed;
		    return out;
	    },
	    py::arg("base_path"), py::arg("run_id"), py::arg("conn") = py::none(),
	    "Explicitly discard a shared-storage direct-write COPY run and verify that it is safe to retry.");

	m.def(
	    "register_copy_direct_write_run_lifecycle",
	    [](const std::string &base_path, const std::string &run_id, idx_t created_epoch_ms) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    auto canonical_res = CanonicalDistributedCopyBasePath(fs, base_path);
		    if (canonical_res.is_err()) {
			    throw py::value_error(canonical_res.error().what());
		    }
		    auto canonical_base_path = std::move(canonical_res).value();
		    auto register_res =
		        WriteDistributedCopyDirectWriteLifecycle(fs, canonical_base_path, run_id, created_epoch_ms);
		    if (register_res.is_err()) {
			    throw py::value_error(register_res.error().what());
		    }
		    auto paths = BuildDistributedCopyFinalizeCommitPaths(fs, canonical_base_path, run_id);
		    py::dict out;
		    out["copy_output_base_path"] = canonical_base_path;
		    out["copy_output_run_id"] = run_id;
		    out["copy_output_commit_dir"] = paths.commit_dir;
		    out["copy_output_lifecycle_path"] = paths.lifecycle_path;
		    out["copy_output_committed_marker_path"] = paths.committed_marker_path;
		    return out;
	    },
	    py::arg("base_path"), py::arg("run_id"), py::arg("created_epoch_ms") = 0,
	    "Register lifecycle metadata for a direct-write distributed COPY run.");

	m.def(
	    "cleanup_expired_copy_direct_write_runs",
	    [](const std::string &base_path, idx_t min_age_ms, idx_t now_epoch_ms, py::object conn_obj) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    auto cleanup_res = WithCopyRecoveryContext(std::move(conn_obj), [&](ClientContext &context) {
			    auto &fs = FileSystem::GetFileSystem(context);
			    return CleanupExpiredDistributedCopyDirectWriteRuns(fs, base_path, min_age_ms, now_epoch_ms);
		    });
		    if (cleanup_res.is_err()) {
			    throw py::value_error(cleanup_res.error().what());
		    }
		    auto cleanup = std::move(cleanup_res).value();
		    py::dict out;
		    out["scanned_runs"] = cleanup.scanned_runs;
		    out["cleaned_runs"] = cleanup.cleaned_runs;
		    out["committed_runs"] = cleanup.committed_runs;
		    out["active_runs"] = cleanup.active_runs;
		    out["skipped_unregistered_runs"] = cleanup.skipped_unregistered_runs;
		    out["errors"] = cleanup.errors;
		    py::list cleaned_run_ids;
		    for (const auto &run_id : cleanup.cleaned_run_ids) {
			    cleaned_run_ids.append(run_id);
		    }
		    out["cleaned_run_ids"] = cleaned_run_ids;
		    py::list error_messages;
		    for (const auto &message : cleanup.error_messages) {
			    error_messages.append(message);
		    }
		    out["error_messages"] = error_messages;
		    return out;
	    },
	    py::arg("base_path"), py::arg("min_age_ms"), py::arg("now_epoch_ms") = 0, py::arg("conn") = py::none(),
	    "TTL cleanup scan for uncommitted direct-write distributed COPY runs.");

	m.def(
	    "make_exchange_source_task_descriptor_for_test",
	    [](py::list handles, py::list partition_indices, idx_t source_partition_count, idx_t source_task_count,
	       bool replicated, bool mark_join_build_summary_valid, bool mark_join_build_has_rows,
	       bool mark_join_build_has_null) {
		    using namespace duckdb::distributed;
		    ExchangeSourceTaskDescriptor desc;
		    for (auto item : partition_indices) {
			    desc.partition_indices.push_back(py::cast<idx_t>(item));
		    }
		    desc.source_partition_count = source_partition_count;
		    desc.source_task_count = source_task_count;
		    desc.replicated = replicated;
		    if (mark_join_build_summary_valid) {
			    desc.mark_join_build_summary =
			        MarkJoinBuildSummary::Create(mark_join_build_has_rows, mark_join_build_has_null);
			    if (!desc.mark_join_build_summary.IsValid()) {
				    throw py::value_error("invalid MARK join build summary");
			    }
		    } else if (mark_join_build_has_rows || mark_join_build_has_null) {
			    throw py::value_error("MARK join build summary values require a valid summary");
		    }

		    for (auto item : handles) {
			    py::dict d = item.cast<py::dict>();
			    ExchangeSourceHandle handle;
			    handle.partition_id = py::int_(d["partition_id"]).cast<idx_t>();
			    if (d.contains("source_task_partition_id")) {
				    handle.source_task_partition_id = py::int_(d["source_task_partition_id"]).cast<idx_t>();
			    }
			    if (d.contains("attempt_id")) {
				    handle.attempt_id = py::int_(d["attempt_id"]).cast<idx_t>();
			    }
			    if (d.contains("node_id")) {
				    handle.node_id = py::str(d["node_id"]).cast<string>();
			    }
			    if (d.contains("flight_host")) {
				    handle.flight_host = py::str(d["flight_host"]).cast<string>();
			    }
			    if (d.contains("flight_port")) {
				    handle.flight_port = py::int_(d["flight_port"]).cast<int>();
			    }
			    if (d.contains("flight_server_epoch")) {
				    handle.flight_server_epoch = py::str(d["flight_server_epoch"]).cast<string>();
			    }
			    if (d.contains("files")) {
				    auto files = py::reinterpret_borrow<py::iterable>(d["files"]);
				    for (auto file_item : files) {
					    py::dict fd = file_item.cast<py::dict>();
					    ExchangeSourceFile file;
					    file.path = py::str(fd["path"]).cast<string>();
					    if (fd.contains("rows")) {
						    file.rows = py::int_(fd["rows"]).cast<idx_t>();
					    }
					    if (fd.contains("file_size")) {
						    file.file_size = py::int_(fd["file_size"]).cast<size_t>();
					    }
					    handle.files.push_back(std::move(file));
				    }
			    }
			    desc.source_handles.push_back(std::move(handle));
		    }

		    auto bytes = desc.SerializeToBytes();
		    return py::bytes(bytes);
	    },
	    py::arg("handles"), py::arg("partition_indices"), py::arg("source_partition_count"),
	    py::arg("source_task_count"), py::arg("replicated") = false, py::arg("mark_join_build_summary_valid") = false,
	    py::arg("mark_join_build_has_rows") = false, py::arg("mark_join_build_has_null") = false,
	    "Build a raw ExchangeSourceTaskDescriptor for Python fast tests.");

	m.def(
	    "exchange_source_task_source_handles_for_test",
	    [](py::bytes bytes_obj) {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto desc = ExchangeSourceTaskDescriptor::DeserializeFromBytes(raw);
		    py::list out;
		    for (const auto &handle : desc.source_handles) {
			    py::dict d;
			    d["partition_id"] = handle.partition_id;
			    if (handle.source_task_partition_id != DConstants::INVALID_INDEX) {
				    d["source_task_partition_id"] = handle.source_task_partition_id;
			    }
			    d["attempt_id"] = handle.attempt_id;
			    d["node_id"] = handle.node_id;
			    if (!handle.flight_host.empty()) {
				    d["flight_host"] = handle.flight_host;
			    }
			    d["flight_port"] = handle.flight_port;
			    if (!handle.flight_server_epoch.empty()) {
				    d["flight_server_epoch"] = handle.flight_server_epoch;
			    }
			    py::list files;
			    for (const auto &file : handle.files) {
				    py::dict fd;
				    fd["path"] = file.path;
				    if (file.rows > 0) {
					    fd["rows"] = file.rows;
				    }
				    fd["file_size"] = file.file_size;
				    files.append(fd);
			    }
			    d["files"] = files;
			    out.append(d);
		    }
		    return out;
	    },
	    py::arg("bytes"), "Return source handle metadata from a raw ExchangeSourceTaskDescriptor.");

	m.def(
	    "flight_exchange_selected_attempt_handles_for_test",
	    []() {
		    using namespace duckdb::distributed;
		    FlightExchangeConfig config;
		    config.node_id = "coordinator";
		    config.local_dirs = {"."};
		    FlightExchangeManager mgr(config, nullptr);

		    ExchangeContext ctx;
		    ctx.query_id = "q";
		    ctx.exchange_id = "flight_exchange_selected_attempt_test";
		    auto exchange = mgr.CreateExchange(ctx, 2);
		    auto sink0 = exchange->AddSink(0);
		    auto sink1 = exchange->AddSink(1);
		    auto sink0_attempt0 = exchange->InstantiateSink(sink0, 0);
		    auto sink0_attempt1 = exchange->InstantiateSink(sink0, 1);
		    auto sink1_attempt0 = exchange->InstantiateSink(sink1, 0);
		    sink0_attempt1.flight_host = "flight-retry.internal";
		    sink0_attempt0.flight_host = "flight-late.internal";
		    sink1_attempt0.flight_host = "flight-first.internal";
		    sink0_attempt1.flight_server_epoch = "worker-retry-epoch";
		    sink0_attempt0.flight_server_epoch = "worker-late-epoch";
		    sink1_attempt0.flight_server_epoch = "worker-first-epoch";
		    exchange->SinkFinished(sink0_attempt1, "worker-retry", 5010);
		    exchange->SinkFinished(sink0_attempt0, "worker-late", 5011);
		    exchange->SinkFinished(sink1_attempt0, "worker-first", 5012);
		    exchange->AllRequiredSinksFinished();
		    auto handles = exchange->GetSourceHandles();
		    exchange->Close();

		    py::list out;
		    for (const auto &handle : handles) {
			    py::dict d;
			    d["partition_id"] = handle.partition_id;
			    d["attempt_id"] = handle.attempt_id;
			    d["node_id"] = handle.node_id;
			    d["flight_host"] = handle.flight_host;
			    d["flight_port"] = handle.flight_port;
			    d["flight_server_epoch"] = handle.flight_server_epoch;
			    d["path"] = handle.files.empty() ? string() : handle.files[0].path;
			    out.append(d);
		    }
		    return out;
	    },
	    "Exercise FlightExchange selected-attempt source handle generation for fast tests.");

	m.def(
	    "flight_exchange_materialized_output_attempt_metadata_for_test",
	    []() {
		    using namespace duckdb::distributed;
		    FlightExchangeConfig config;
		    config.node_id = "coordinator";
		    config.local_dirs = {"."};
		    FlightExchangeManager mgr(config, nullptr);

		    ExchangeContext ctx;
		    ctx.query_id = "q";
		    ctx.exchange_id = "flight_exchange_output_metadata_test";
		    auto exchange = mgr.CreateExchange(ctx, 2);
		    auto sink0 = exchange->AddSink(0);
		    auto sink1 = exchange->AddSink(1);
		    exchange->InstantiateSink(sink0, 0);
		    auto sink0_attempt1 = exchange->InstantiateSink(sink0, 1);
		    auto sink1_attempt0 = exchange->InstantiateSink(sink1, 0);
		    sink0_attempt1.flight_host = "flight-retry.internal";
		    sink1_attempt0.flight_host = "flight-first.internal";
		    sink0_attempt1.flight_server_epoch = "worker-retry-epoch";
		    sink1_attempt0.flight_server_epoch = "worker-first-epoch";

		    std::vector<MaterializedOutput> outputs;
		    auto retry_output = MaterializedOutput({}, make_worker_id("worker-retry"));
		    retry_output.set_flight_port(5010);
		    retry_output.set_exchange_sink_instance(sink0_attempt1);
		    outputs.push_back(std::move(retry_output));

		    auto first_output = MaterializedOutput({}, make_worker_id("worker-first"));
		    first_output.set_flight_port(5012);
		    first_output.set_exchange_sink_instance(sink1_attempt0);
		    outputs.push_back(std::move(first_output));

		    RecordRemoteExchangeFinishedSinks(*exchange, outputs, "test");
		    exchange->AllRequiredSinksFinished();
		    auto handles = exchange->GetSourceHandles();
		    exchange->Close();

		    py::list out;
		    for (const auto &handle : handles) {
			    py::dict d;
			    d["partition_id"] = handle.partition_id;
			    d["attempt_id"] = handle.attempt_id;
			    d["node_id"] = handle.node_id;
			    d["flight_host"] = handle.flight_host;
			    d["flight_port"] = handle.flight_port;
			    d["flight_server_epoch"] = handle.flight_server_epoch;
			    d["path"] = handle.files.empty() ? string() : handle.files[0].path;
			    out.append(d);
		    }
		    return out;
	    },
	    "Verify MaterializedOutput sink attempt metadata drives remote exchange completion.");

	m.def(
	    "ray_task_result_handle_refreshed_worker_id_for_test",
	    [](py::object handle) {
		    using namespace duckdb::distributed;
		    using namespace duckdb::distributed::python::ray;

		    RayTaskResultHandle task_handle(TaskContext::from_node_context(0, 1, 0), handle,
		                                    make_worker_id("worker-original"));

		    std::string worker_id;
		    std::string exchange_sink_query_id;
		    bool has_output = false;
		    bool has_exchange_sink_instance = false;
		    int flight_port = 0;
		    {
			    pybind11::gil_scoped_release release;
			    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
			    while (std::chrono::steady_clock::now() < deadline) {
				    auto polled = task_handle.poll();
				    if (!polled.first) {
					    std::this_thread::sleep_for(std::chrono::milliseconds(5));
					    continue;
				    }
				    if (polled.second.is_err()) {
					    throw polled.second.error();
				    }
				    auto payload = std::move(polled.second).value();
				    has_output = payload.first;
				    const auto &output = payload.second;
				    if (output.worker_id()) {
					    worker_id = *output.worker_id();
				    }
				    flight_port = output.flight_port();
				    has_exchange_sink_instance = output.has_exchange_sink_instance();
				    if (has_exchange_sink_instance) {
					    exchange_sink_query_id = output.exchange_sink_instance().query_id;
				    }
				    task_handle.AckPollResult();
				    task_handle.ReleasePollResult();
				    break;
			    }
		    }
		    if (worker_id.empty()) {
			    throw std::runtime_error("RayTaskResultHandle did not produce output before timeout");
		    }
		    pybind11::dict result;
		    result["worker_id"] = worker_id;
		    result["has_output"] = has_output;
		    result["flight_port"] = flight_port;
		    result["has_exchange_sink_instance"] = has_exchange_sink_instance;
		    result["exchange_sink_query_id"] = exchange_sink_query_id;
		    return result;
	    },
	    py::arg("handle"),
	    "Verify RayTaskResultHandle records the current Python handle worker_id at completion time.");

	m.def("_prepare_ray_task_result_poller_shutdown_race_for_test",
	      &duckdb::distributed::python::ray::PrepareRayTaskResultPollerShutdownRaceForTest, py::arg("handle"));
	m.def("_shutdown_ray_task_result_poller_for_test", &duckdb::distributed::python::ray::ShutdownRayTaskResultPoller);

	m.def(
	    "_ray_task_result_poller_batch_for_test",
	    [](py::list handles, int timeout_ms) {
		    using namespace duckdb::distributed;
		    using namespace duckdb::distributed::python::ray;

		    if (timeout_ms <= 0) {
			    throw pybind11::value_error("timeout_ms must be positive");
		    }

		    struct PollOutcome {
			    bool terminal = false;
			    bool is_error = false;
			    string error;
			    bool has_output = false;
			    string worker_id;
		    };

		    std::vector<std::unique_ptr<RayTaskResultHandle>> task_handles;
		    task_handles.reserve(handles.size());
		    for (size_t index = 0; index < handles.size(); ++index) {
			    auto handle = pybind11::reinterpret_borrow<pybind11::object>(handles[index]);
			    task_handles.push_back(make_uniq<RayTaskResultHandle>(
			        TaskContext::from_node_context(0, index + 1, index), std::move(handle),
			        make_worker_id("worker-original-" + std::to_string(index)),
			        "poller-test." + std::to_string(index)));
		    }

		    std::vector<PollOutcome> outcomes(task_handles.size());
		    {
			    pybind11::gil_scoped_release release;
			    size_t remaining = task_handles.size();
			    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
			    while (remaining > 0 && std::chrono::steady_clock::now() < deadline) {
				    bool had_progress = false;
				    for (size_t index = 0; index < task_handles.size(); ++index) {
					    auto &outcome = outcomes[index];
					    if (outcome.terminal) {
						    continue;
					    }
					    auto polled = task_handles[index]->poll();
					    if (!polled.first) {
						    continue;
					    }
					    outcome.terminal = true;
					    remaining--;
					    had_progress = true;
					    if (polled.second.is_err()) {
						    outcome.is_error = true;
						    outcome.error = polled.second.error().what();
					    } else {
						    auto payload = std::move(polled.second).value();
						    outcome.has_output = payload.first;
						    if (payload.second.worker_id()) {
							    outcome.worker_id = *payload.second.worker_id();
						    }
					    }
					    task_handles[index]->AckPollResult();
				    }
				    if (!had_progress && remaining > 0) {
					    std::this_thread::sleep_for(std::chrono::milliseconds(1));
				    }
			    }
		    }

		    pybind11::list result;
		    for (const auto &outcome : outcomes) {
			    pybind11::dict item;
			    item["terminal"] = outcome.terminal;
			    item["is_error"] = outcome.is_error;
			    item["error"] = outcome.error;
			    item["has_output"] = outcome.has_output;
			    item["worker_id"] = outcome.worker_id;
			    result.append(std::move(item));
		    }
		    return result;
	    },
	    py::arg("handles"), py::arg("timeout_ms") = 2000,
	    "Poll multiple RayTaskResultHandle instances through the shared native poller.");

	m.def(
	    "python_task_result_handle_for_test",
	    [](py::object handle) {
		    using namespace duckdb::distributed;
		    using namespace duckdb::distributed::python::ray;

		    auto task_handle = MakePythonTaskResultHandle(std::move(handle));
		    std::string worker_id;
		    bool has_output = false;
		    int flight_port = 0;
		    {
			    pybind11::gil_scoped_release release;
			    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
			    while (std::chrono::steady_clock::now() < deadline) {
				    auto polled = task_handle.poll();
				    if (!polled.first) {
					    std::this_thread::sleep_for(std::chrono::milliseconds(5));
					    continue;
				    }
				    if (polled.second.is_err()) {
					    throw polled.second.error();
				    }
				    auto payload = std::move(polled.second).value();
				    has_output = payload.first;
				    const auto &output = payload.second;
				    if (output.worker_id()) {
					    worker_id = *output.worker_id();
				    }
				    flight_port = output.flight_port();
				    task_handle.AckPollResult();
				    task_handle.ReleasePollResult();
				    break;
			    }
		    }
		    if (worker_id.empty()) {
			    throw std::runtime_error("PythonTaskResultHandle did not produce output before timeout");
		    }
		    pybind11::dict result;
		    result["worker_id"] = worker_id;
		    result["has_output"] = has_output;
		    result["flight_port"] = flight_port;
		    return result;
	    },
	    py::arg("handle"), "Verify PythonTaskResultHandle can poll a backend-neutral Python handle.");

	m.def(
	    "flight_exchange_selected_attempt_dataplane_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};

		    FlightExchangeConfig config;
		    config.node_id = "node-a";
		    config.local_dirs = {local_dir};
		    config.expected_types = types;
		    FlightExchangeManager mgr(config, &context);

		    ExchangeContext exchange_context;
		    exchange_context.query_id = "q";
		    exchange_context.exchange_id =
		        "flight_exchange_selected_attempt_dataplane_" + UUID::ToString(UUID::GenerateRandomUUID());
		    auto exchange = mgr.CreateExchange(exchange_context, 1);
		    auto sink_handle = exchange->AddSink(0);
		    auto lost_instance = exchange->InstantiateSink(sink_handle, 0);
		    auto selected_instance = exchange->InstantiateSink(sink_handle, 1);

		    auto write_attempt = [&](const ExchangeSinkInstanceHandle &instance, std::vector<int32_t> values) {
			    auto sink = mgr.CreateSink(instance);
			    DataChunk chunk;
			    chunk.Initialize(Allocator::DefaultAllocator(), types);
			    chunk.SetCardinality(values.size());
			    for (idx_t row = 0; row < values.size(); row++) {
				    chunk.SetValue(0, row, Value::INTEGER(values[row]));
			    }
			    auto add_res = sink->AddChunk(0, chunk);
			    if (add_res.is_err()) {
				    throw std::runtime_error(add_res.error().what());
			    }
			    auto finish_res = sink->Finish();
			    if (finish_res.is_err()) {
				    throw std::runtime_error(finish_res.error().what());
			    }
			    return mgr.config().node_id;
		    };

		    auto lost_node_id = write_attempt(lost_instance, {101});
		    lost_instance.flight_host = FlightExchangeManager::GetLocalFlightServerHost();
		    lost_instance.flight_server_epoch = FlightExchangeManager::GetLocalFlightServerEpoch();
		    auto selected_node_id = write_attempt(selected_instance, {201, 202});
		    selected_instance.flight_host = FlightExchangeManager::GetLocalFlightServerHost();
		    selected_instance.flight_server_epoch = FlightExchangeManager::GetLocalFlightServerEpoch();
		    const auto flight_port = FlightExchangeManager::GetLocalFlightServerPort();
		    exchange->SinkFinished(selected_instance, selected_node_id, flight_port);
		    exchange->AllRequiredSinksFinished();
		    auto handles = exchange->GetSourceHandles();

		    auto read_values = [&](const std::vector<ExchangeSourceHandle> &source_handles) {
			    auto source = mgr.CreateSource();
			    auto copied_handles = source_handles;
			    source->AddSourceHandles(std::move(copied_handles));
			    DataChunk output;
			    output.Initialize(Allocator::DefaultAllocator(), types);
			    py::list values;
			    while (source->ReadChunk(output)) {
				    for (idx_t row = 0; row < output.size(); row++) {
					    values.append(output.GetValue(0, row).GetValue<int32_t>());
				    }
			    }
			    return values;
		    };

		    auto selected_values_before_late_loser = read_values(handles);
		    exchange->SinkFinished(lost_instance, lost_node_id, flight_port);
		    auto selected_values_after_late_loser = read_values(handles);

		    auto manifest_exists = [&](const ExchangeSinkInstanceHandle &instance, const std::string &node_id) {
			    ShuffleCacheConfig cache_config;
			    cache_config.exchange_id = instance.output_location;
			    cache_config.node_id = node_id;
			    cache_config.num_partitions = 1;
			    cache_config.local_dirs = {local_dir};
			    ShuffleCache cache(cache_config);
			    return cache.HasCommittedManifest();
		    };

		    py::list handle_attempts;
		    py::list handle_paths;
		    py::list handle_node_ids;
		    for (const auto &handle : handles) {
			    handle_attempts.append(handle.attempt_id);
			    handle_paths.append(handle.files.empty() ? string() : handle.files[0].path);
			    handle_node_ids.append(handle.node_id);
		    }

		    py::dict out;
		    out["lost_output_location"] = lost_instance.output_location;
		    out["selected_output_location"] = selected_instance.output_location;
		    out["lost_node_id"] = lost_node_id;
		    out["selected_node_id"] = selected_node_id;
		    out["handle_attempts"] = handle_attempts;
		    out["handle_paths"] = handle_paths;
		    out["handle_node_ids"] = handle_node_ids;
		    out["selected_values_before_late_loser"] = selected_values_before_late_loser;
		    out["selected_values_after_late_loser"] = selected_values_after_late_loser;
		    out["lost_manifest_exists_after_late_loser"] = manifest_exists(lost_instance, lost_node_id);
		    out["selected_manifest_exists_after_late_loser"] = manifest_exists(selected_instance, selected_node_id);
		    exchange->Close();
		    auto cleanup = ShuffleCacheRegistry::Instance().RemoveAndCleanupByQuery(exchange_context.query_id);
		    if (cleanup.cleanup_errors != 0 || cleanup.cleanup_pending != 0) {
			    throw std::runtime_error("failed to clean selected-attempt dataplane test query");
		    }
		    auto retire = ShuffleCacheRegistry::Instance().RetireQuery(exchange_context.query_id);
		    if (retire.is_err()) {
			    throw std::runtime_error(retire.error().what());
		    }
		    return out;
	    },
	    py::arg("local_dir"), "Verify FlightExchangeSource reads only the selected retry attempt data.");

	m.def(
	    "flight_exchange_local_dirs_from_env_for_test",
	    []() {
		    using namespace duckdb::distributed;
		    py::list out;
		    for (const auto &dir : ResolveFlightExchangeLocalDirsFromEnv()) {
			    out.append(dir);
		    }
		    return out;
	    },
	    "Return parsed DUCKDB_SHUFFLE_DIRS values for regression tests.");

	m.def(
	    "flight_exchange_node_id_from_env_for_test",
	    []() { return duckdb::distributed::ResolveFlightExchangeNodeIdFromEnv(); },
	    "Return the resolved FlightExchange node id for regression tests.");

	m.def(
	    "flight_exchange_unselected_attempt_cleanup_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    const std::string node_id = "node-cleanup";
		    vector<LogicalType> types = {LogicalType::INTEGER};

		    FlightExchangeConfig config;
		    config.node_id = node_id;
		    config.local_dirs = {local_dir};
		    FlightExchangeManager mgr(config, &context);

		    ExchangeContext ctx;
		    ctx.query_id = "q";
		    ctx.exchange_id = "flight_exchange_cleanup_test";
		    auto exchange = mgr.CreateExchange(ctx, 2);
		    auto sink_handle = exchange->AddSink(0);
		    auto selected_instance = exchange->InstantiateSink(sink_handle, 0);
		    auto loser_instance = exchange->InstantiateSink(sink_handle, 1);

		    auto write_attempt = [&](const ExchangeSinkInstanceHandle &instance, int32_t value) {
			    auto sink = mgr.CreateSink(instance);
			    DataChunk chunk;
			    chunk.Initialize(Allocator::DefaultAllocator(), types);
			    chunk.SetCardinality(1);
			    chunk.SetValue(0, 0, Value::INTEGER(value));
			    auto add_res = sink->AddChunk(0, chunk);
			    if (add_res.is_err()) {
				    throw std::runtime_error(add_res.error().what());
			    }
			    auto finish_res = sink->Finish();
			    if (finish_res.is_err()) {
				    throw std::runtime_error(finish_res.error().what());
			    }
			    return mgr.config().node_id;
		    };

		    auto selected_node_id = write_attempt(selected_instance, 101);
		    selected_instance.flight_host = FlightExchangeManager::GetLocalFlightServerHost();
		    selected_instance.flight_server_epoch = FlightExchangeManager::GetLocalFlightServerEpoch();
		    const auto flight_port = FlightExchangeManager::GetLocalFlightServerPort();
		    exchange->SinkFinished(selected_instance, selected_node_id, flight_port);
		    auto loser_node_id = write_attempt(loser_instance, 202);
		    loser_instance.flight_host = FlightExchangeManager::GetLocalFlightServerHost();
		    loser_instance.flight_server_epoch = FlightExchangeManager::GetLocalFlightServerEpoch();

		    auto make_cache = [&](const std::string &output_location, const std::string &cache_node_id) {
			    ShuffleCacheConfig cache_config;
			    cache_config.exchange_id = output_location;
			    cache_config.node_id = cache_node_id;
			    cache_config.num_partitions = 2;
			    cache_config.local_dirs = {local_dir};
			    return make_uniq<ShuffleCache>(std::move(cache_config));
		    };

		    auto selected_cache_before = make_cache(selected_instance.output_location, selected_node_id);
		    auto loser_cache_before = make_cache(loser_instance.output_location, loser_node_id);
		    auto selected_manifest_before = selected_cache_before->HasCommittedManifest();
		    auto loser_manifest_before = loser_cache_before->HasCommittedManifest();
		    auto selected_registry_before =
		        ShuffleCacheRegistry::Instance().Get(selected_instance.output_location) != nullptr;
		    auto loser_registry_before =
		        ShuffleCacheRegistry::Instance().Get(loser_instance.output_location) != nullptr;

		    exchange->SinkFinished(loser_instance, loser_node_id, flight_port);
		    exchange->AllRequiredSinksFinished();
		    auto handles = exchange->GetSourceHandles();

		    auto selected_cache_after = make_cache(selected_instance.output_location, selected_node_id);
		    auto loser_cache_after = make_cache(loser_instance.output_location, loser_node_id);
		    auto selected_manifest_after_cleanup = selected_cache_after->HasCommittedManifest();
		    auto loser_manifest_after_cleanup = loser_cache_after->HasCommittedManifest();
		    auto selected_registry_after_cleanup =
		        ShuffleCacheRegistry::Instance().Get(selected_instance.output_location) != nullptr;
		    auto loser_registry_after_cleanup =
		        ShuffleCacheRegistry::Instance().Get(loser_instance.output_location) != nullptr;

		    exchange->Close();
		    auto selected_cache_after_close = make_cache(selected_instance.output_location, selected_node_id);
		    auto selected_manifest_after_close = selected_cache_after_close->HasCommittedManifest();
		    auto selected_registry_after_close =
		        ShuffleCacheRegistry::Instance().Get(selected_instance.output_location) != nullptr;

		    py::list handle_paths;
		    py::list handle_attempts;
		    for (const auto &handle : handles) {
			    handle_attempts.append(handle.attempt_id);
			    handle_paths.append(handle.files.empty() ? string() : handle.files[0].path);
		    }

		    py::dict out;
		    out["selected_manifest_before"] = selected_manifest_before;
		    out["loser_manifest_before"] = loser_manifest_before;
		    out["selected_registry_before"] = selected_registry_before;
		    out["loser_registry_before"] = loser_registry_before;
		    out["selected_manifest_after_cleanup"] = selected_manifest_after_cleanup;
		    out["loser_manifest_after_cleanup"] = loser_manifest_after_cleanup;
		    out["selected_registry_after_cleanup"] = selected_registry_after_cleanup;
		    out["loser_registry_after_cleanup"] = loser_registry_after_cleanup;
		    out["selected_manifest_after_close"] = selected_manifest_after_close;
		    out["selected_registry_after_close"] = selected_registry_after_close;
		    out["selected_output_location"] = selected_instance.output_location;
		    out["loser_output_location"] = loser_instance.output_location;
		    out["handle_paths"] = handle_paths;
		    out["handle_attempts"] = handle_attempts;
		    auto cleanup = ShuffleCacheRegistry::Instance().RemoveAndCleanupByQuery(ctx.query_id);
		    if (cleanup.cleanup_errors != 0 || cleanup.cleanup_pending != 0) {
			    throw std::runtime_error("failed to clean unselected-attempt test query");
		    }
		    auto retire = ShuffleCacheRegistry::Instance().RetireQuery(ctx.query_id);
		    if (retire.is_err()) {
			    throw std::runtime_error(retire.error().what());
		    }
		    return out;
	    },
	    py::arg("local_dir"), "Exercise cleanup of successful unselected FlightExchange attempts.");

	m.def(
	    "shuffle_cache_registry_query_cleanup_for_test",
	    [](const string &local_dir) {
		    using namespace duckdb::distributed;
		    const string query_id = "query-cleanup";
		    const string keep_query_id = "query-keep";
		    const string node_id = "node-a";
		    const string cleanup_exchange = query_id + "_shuffle_1__sink_0__attempt_0";
		    const string keep_exchange = keep_query_id + "_shuffle_1__sink_0__attempt_0";

		    auto make_cache = [&](const string &exchange_id) {
			    ShuffleCacheConfig config;
			    config.exchange_id = exchange_id;
			    config.node_id = node_id;
			    config.num_partitions = 1;
			    config.local_dirs = {local_dir};
			    return std::make_shared<ShuffleCache>(std::move(config));
		    };

		    ShuffleCacheRegistry::Instance().RemoveAndCleanupByPrefix(query_id + "_");
		    ShuffleCacheRegistry::Instance().RemoveAndCleanupByPrefix(keep_query_id + "_");

		    duckdb::LocalFileSystem fs;
		    auto cleanup_node_dir = local_dir + "/shuffle_" + cleanup_exchange + "/node_" + node_id;
		    auto cleanup_partition_dir = cleanup_node_dir + "/partition_0";
		    fs.CreateDirectoriesRecursive(cleanup_partition_dir);
		    {
			    std::ofstream file(cleanup_partition_dir + "/batch.arrow", std::ios::out | std::ios::binary);
			    file << "data";
		    }
		    auto keep_node_dir = local_dir + "/shuffle_" + keep_exchange + "/node_" + node_id;
		    auto keep_partition_dir = keep_node_dir + "/partition_0";
		    fs.CreateDirectoriesRecursive(keep_partition_dir);
		    {
			    std::ofstream file(keep_partition_dir + "/batch.arrow", std::ios::out | std::ios::binary);
			    file << "keep";
		    }

		    auto cleanup_cache = make_cache(cleanup_exchange);
		    auto keep_cache = make_cache(keep_exchange);
		    auto cleanup_register =
		        ShuffleCacheRegistry::Instance().Register(cleanup_exchange, cleanup_cache, query_id);
		    auto keep_register = ShuffleCacheRegistry::Instance().Register(keep_exchange, keep_cache, keep_query_id);
		    if (cleanup_register.is_err() || keep_register.is_err()) {
			    throw std::runtime_error("failed to register query cleanup test caches");
		    }
		    ShuffleCacheRegistry::Instance().RemoveForDeferredCleanup(cleanup_exchange);
		    auto cleanup_registry_after_defer = ShuffleCacheRegistry::Instance().Get(cleanup_exchange) != nullptr;

		    auto cleanup_result = ShuffleCacheRegistry::Instance().RemoveAndCleanupByPrefix(query_id + "_");
		    py::dict out;
		    out["registry_entries_removed"] = cleanup_result.registry_entries_removed;
		    out["storage_entries_removed"] = cleanup_result.storage_entries_removed;
		    out["cleanup_errors"] = cleanup_result.cleanup_errors;
		    out["cleanup_registry_after_defer"] = cleanup_registry_after_defer;
		    out["cleanup_registry_after"] = ShuffleCacheRegistry::Instance().Get(cleanup_exchange) != nullptr;
		    out["keep_registry_after"] = ShuffleCacheRegistry::Instance().Get(keep_exchange) != nullptr;
		    out["cleanup_node_dir_exists_after"] = fs.DirectoryExists(cleanup_node_dir);
		    out["keep_node_dir_exists_after"] = fs.DirectoryExists(keep_node_dir);

		    ShuffleCacheRegistry::Instance().RemoveAndCleanupByPrefix(keep_query_id + "_");
		    return out;
	    },
	    py::arg("local_dir"));

	m.def(
	    "shuffle_cache_attempt_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb::distributed;
		    ShuffleCacheConfig config;
		    config.exchange_id = "shuffle_cache_manifest_test__sink_3__attempt_2";
		    config.node_id = "node-a";
		    config.num_partitions = 2;
		    config.local_dirs = {local_dir};
		    ShuffleCache cache(std::move(config));
		    ShufflePartitionFile file;
		    file.path = local_dir + "/partition_0/batch.arrow";
		    file.rows = 4;
		    file.bytes = 11;
		    auto reg_res = cache.RegisterPartitionFile(0, std::move(file));
		    if (reg_res.is_err()) {
			    throw std::runtime_error(reg_res.error().what());
		    }
		    auto manifest_res = cache.WriteAttemptManifest(3, 2);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }

		    std::ifstream manifest(cache.ManifestFilePath());
		    if (!manifest.good()) {
			    throw std::runtime_error("failed to read shuffle attempt manifest");
		    }
		    std::string contents((std::istreambuf_iterator<char>(manifest)), std::istreambuf_iterator<char>());

		    py::dict out;
		    out["manifest_path"] = cache.ManifestFilePath();
		    out["committed_path"] = cache.CommittedMarkerPath();
		    out["manifest"] = contents;
		    return out;
	    },
	    py::arg("local_dir"), "Exercise ShuffleCache attempt manifest writing for fast tests.");

	m.def(
	    "shuffle_cache_manifest_recovery_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = "shuffle_cache_manifest_recovery_test__sink_0__attempt_1";
		    write_config.node_id = "node-a";
		    write_config.num_partitions = 2;
		    write_config.local_dirs = {local_dir};

		    ShuffleCache writer(write_config);
		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(3);
		    input.SetValue(0, 0, Value::INTEGER(11));
		    input.SetValue(0, 1, Value::INTEGER(12));
		    input.SetValue(0, 2, Value::INTEGER(13));
		    auto write_res = writer.WriteChunk(context, input, 1, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer.FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto manifest_res = writer.WriteAttemptManifest(0, 1);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }

		    auto reader = std::make_shared<ShuffleCache>(write_config);
		    auto memory_files_res = reader->GetPartitionFiles(1);
		    if (memory_files_res.is_err()) {
			    throw std::runtime_error(memory_files_res.error().what());
		    }
		    auto manifest_files_res = reader->GetPartitionFilesFromManifest(1);
		    if (manifest_files_res.is_err()) {
			    throw std::runtime_error(manifest_files_res.error().what());
		    }
		    auto register_res = ShuffleCacheRegistry::Instance().Register(
		        write_config.exchange_id, reader, "shuffle-cache-manifest-recovery-query", "", 1);
		    if (register_res.is_err()) {
			    throw std::runtime_error(register_res.error().what());
		    }
		    ScopedShuffleCacheRegistrationForTest registration(write_config.exchange_id);
		    FlightExchangeConfig source_config;
		    source_config.node_id = write_config.node_id;
		    ExchangeSourceHandle handle;
		    handle.partition_id = 1;
		    handle.attempt_id = 1;
		    handle.node_id = write_config.node_id;
		    handle.files.push_back(ExchangeSourceFile(write_config.exchange_id, 0));
		    auto values = ReadIntegerExchangeSourceForTest(context, source_config, std::move(handle));

		    py::dict out;
		    out["memory_file_count"] = memory_files_res.value().files.size();
		    out["manifest_file_count"] = manifest_files_res.value().files.size();
		    out["row_count"] = values.size();
		    out["values"] = values;
		    out["manifest_path"] = reader->ManifestFilePath();
		    return out;
	    },
	    py::arg("local_dir"), "Exercise ShuffleCache manifest-backed recovery after in-memory registry loss.");

	m.def(
	    "shuffle_cache_uncommitted_files_invisible_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = "shuffle_cache_uncommitted_invisible_test__sink_0__attempt_1";
		    write_config.node_id = "node-a";
		    write_config.num_partitions = 1;
		    write_config.local_dirs = {local_dir};

		    ShuffleCache writer(write_config);
		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(2);
		    input.SetValue(0, 0, Value::INTEGER(51));
		    input.SetValue(0, 1, Value::INTEGER(52));
		    auto write_res = writer.WriteChunk(context, input, 0, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer.FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto partial_files_res = writer.GetPartitionFiles(0);
		    if (partial_files_res.is_err()) {
			    throw std::runtime_error(partial_files_res.error().what());
		    }

		    ShuffleCache reader(write_config);
		    auto manifest_files_res = reader.GetPartitionFilesFromManifest(0);
		    if (manifest_files_res.is_err()) {
			    throw std::runtime_error(manifest_files_res.error().what());
		    }
		    if (!manifest_files_res.value().files.empty()) {
			    throw std::runtime_error("uncommitted shuffle files unexpectedly became visible");
		    }

		    py::dict out;
		    out["partial_file_count"] = partial_files_res.value().files.size();
		    out["committed_manifest"] = reader.HasCommittedManifest();
		    out["recovered_row_count"] = manifest_files_res.value().total_rows;
		    return out;
	    },
	    py::arg("local_dir"), "Exercise that uncommitted shuffle files are invisible without committed manifest.");

	m.def(
	    "shuffle_cache_duckdb_filesystem_storage_roundtrip_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);
		    auto storage = MakeDuckDBFileSystemShuffleStorage(fs);

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    ShuffleCacheConfig config;
		    config.exchange_id = "shuffle_cache_duckdb_fs_storage_test__sink_0__attempt_1";
		    config.node_id = "node-a";
		    config.num_partitions = 1;
		    config.local_dirs = {local_dir};

		    ShuffleCache writer(config, storage);
		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(4);
		    input.SetValue(0, 0, Value::INTEGER(71));
		    input.SetValue(0, 1, Value::INTEGER(72));
		    input.SetValue(0, 2, Value::INTEGER(73));
		    input.SetValue(0, 3, Value::INTEGER(74));
		    auto write_res = writer.WriteChunk(context, input, 0, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer.FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto manifest_res = writer.WriteAttemptManifest(0, 1);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }

		    auto reader = std::make_shared<ShuffleCache>(config, storage);
		    auto manifest_files_res = reader->GetPartitionFilesFromManifest(0);
		    if (manifest_files_res.is_err()) {
			    throw std::runtime_error(manifest_files_res.error().what());
		    }
		    auto register_res = ShuffleCacheRegistry::Instance().Register(
		        config.exchange_id, reader, "shuffle-cache-duckdb-fs-storage-query", "", 1);
		    if (register_res.is_err()) {
			    throw std::runtime_error(register_res.error().what());
		    }
		    ScopedShuffleCacheRegistrationForTest registration(config.exchange_id);
		    FlightExchangeConfig source_config;
		    source_config.node_id = config.node_id;
		    ExchangeSourceHandle handle;
		    handle.partition_id = 0;
		    handle.attempt_id = 1;
		    handle.node_id = config.node_id;
		    handle.files.push_back(ExchangeSourceFile(config.exchange_id, 0));
		    auto values = ReadIntegerExchangeSourceForTest(context, source_config, std::move(handle));

		    py::dict out;
		    out["committed_manifest"] = reader->HasCommittedManifest();
		    out["manifest_file_count"] = manifest_files_res.value().files.size();
		    out["manifest_total_rows"] = manifest_files_res.value().total_rows;
		    out["row_count"] = values.size();
		    out["values"] = values;
		    out["manifest_tmp_exists"] = fs.FileExists(reader->ManifestFilePath() + ".tmp");
		    out["marker_tmp_exists"] = fs.FileExists(reader->CommittedMarkerPath() + ".tmp");
		    return out;
	    },
	    py::arg("local_dir"), "Exercise ShuffleCache roundtrip through the DuckDB FileSystem backed storage backend.");

	m.def(
	    "shuffle_cache_duckdb_filesystem_storage_minio_roundtrip_for_test",
	    [](const std::string &base_uri, const std::string &endpoint_url, const std::string &access_key,
	       const std::string &secret_key, const std::string &region) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    ConfigureConnectionForS3Endpoint(conn, endpoint_url, access_key, secret_key, region);
		    auto &context = *conn.context;
		    DuckDBPyTransactionGuard tx_guard(context);
		    auto &fs = FileSystem::GetFileSystem(context);
		    auto storage = MakeDuckDBFileSystemShuffleStorage(fs);

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const auto run_id = UUID::ToString(UUID::GenerateRandomUUID());

		    ShuffleCacheConfig config;
		    config.exchange_id = "shuffle_cache_duckdb_fs_minio_storage_test_" + run_id + "__sink_0__attempt_1";
		    config.node_id = "node-a";
		    config.num_partitions = 1;
		    config.local_dirs = {base_uri};

		    ShuffleCache writer(config, storage);
		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(4);
		    input.SetValue(0, 0, Value::INTEGER(171));
		    input.SetValue(0, 1, Value::INTEGER(172));
		    input.SetValue(0, 2, Value::INTEGER(173));
		    input.SetValue(0, 3, Value::INTEGER(174));
		    auto write_res = writer.WriteChunk(context, input, 0, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer.FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto manifest_res = writer.WriteAttemptManifest(0, 1);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }

		    const auto manifest_path = writer.ManifestFilePath();
		    const auto marker_path = writer.CommittedMarkerPath();
		    const auto manifest_exists_before_cleanup = fs.FileExists(manifest_path);
		    const auto marker_exists_before_cleanup = fs.FileExists(marker_path);
		    const auto manifest_tmp_exists = fs.FileExists(manifest_path + ".tmp");
		    const auto marker_tmp_exists = fs.FileExists(marker_path + ".tmp");

		    ShuffleCache reader(config, storage);
		    const auto committed_manifest_before_cleanup = reader.HasCommittedManifest();
		    auto manifest_files_res = reader.GetPartitionFilesFromManifest(0);
		    if (manifest_files_res.is_err()) {
			    throw std::runtime_error(manifest_files_res.error().what());
		    }
		    FlightExchangeConfig source_config;
		    source_config.local_dirs = {base_uri};
		    source_config.node_id = "reader-node";
		    ExchangeSourceHandle handle;
		    handle.partition_id = 0;
		    handle.attempt_id = 1;
		    handle.node_id = config.node_id;
		    handle.files.push_back(ExchangeSourceFile(config.exchange_id, 0));
		    auto values = ReadIntegerExchangeSourceForTest(context, source_config, std::move(handle));

		    auto cleanup_res = reader.RemoveAttemptStorage();
		    if (cleanup_res.is_err()) {
			    throw std::runtime_error(cleanup_res.error().what());
		    }

		    py::dict out;
		    out["base_uri"] = base_uri;
		    out["exchange_id"] = config.exchange_id;
		    out["manifest_path"] = manifest_path;
		    out["marker_path"] = marker_path;
		    out["committed_manifest"] = committed_manifest_before_cleanup;
		    out["manifest_exists_before_cleanup"] = manifest_exists_before_cleanup;
		    out["marker_exists_before_cleanup"] = marker_exists_before_cleanup;
		    out["manifest_file_count"] = manifest_files_res.value().files.size();
		    out["manifest_total_rows"] = manifest_files_res.value().total_rows;
		    out["manifest_total_bytes"] = manifest_files_res.value().total_bytes;
		    out["row_count"] = values.size();
		    out["values"] = values;
		    out["manifest_tmp_exists"] = manifest_tmp_exists;
		    out["marker_tmp_exists"] = marker_tmp_exists;
		    out["cleanup_removed"] = cleanup_res.value();
		    out["manifest_exists_after_cleanup"] = fs.FileExists(manifest_path);
		    out["marker_exists_after_cleanup"] = fs.FileExists(marker_path);
		    out["committed_manifest_after_cleanup"] = reader.HasCommittedManifest();
		    tx_guard.Commit();
		    return out;
	    },
	    py::arg("base_uri"), py::arg("endpoint_url"), py::arg("access_key"), py::arg("secret_key"),
	    py::arg("region") = "us-east-1",
	    "Exercise ShuffleCache DuckDB FileSystem storage against a real MinIO/S3-compatible endpoint.");

	m.def(
	    "shuffle_cache_duckdb_filesystem_storage_minio_fault_matrix_for_test",
	    [](const std::string &base_uri, const std::string &endpoint_url, const std::string &access_key,
	       const std::string &secret_key, const std::string &region) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    ConfigureConnectionForS3Endpoint(conn, endpoint_url, access_key, secret_key, region);
		    auto &context = *conn.context;
		    DuckDBPyTransactionGuard tx_guard(context);
		    auto &fs = FileSystem::GetFileSystem(context);
		    auto storage = MakeDuckDBFileSystemShuffleStorage(fs);

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const auto run_id = UUID::ToString(UUID::GenerateRandomUUID());
		    const std::string node_id = "node-a";

		    auto make_config = [&](const std::string &scenario) {
			    ShuffleCacheConfig config;
			    config.exchange_id =
			        "shuffle_cache_duckdb_fs_minio_fault_matrix_" + run_id + "_" + scenario + "__sink_0__attempt_1";
			    config.node_id = node_id;
			    config.num_partitions = 1;
			    config.local_dirs = {base_uri};
			    return config;
		    };

		    auto write_attempt = [&](const ShuffleCacheConfig &config, std::vector<int32_t> values) {
			    ShuffleCache writer(config, storage);
			    DataChunk input;
			    input.Initialize(Allocator::DefaultAllocator(), types);
			    input.SetCardinality(values.size());
			    for (idx_t row = 0; row < values.size(); row++) {
				    input.SetValue(0, row, Value::INTEGER(values[row]));
			    }
			    auto write_res = writer.WriteChunk(context, input, 0, names);
			    if (write_res.is_err()) {
				    throw std::runtime_error(write_res.error().what());
			    }
			    auto flush_res = writer.FlushAll(context, names);
			    if (flush_res.is_err()) {
				    throw std::runtime_error(flush_res.error().what());
			    }
			    auto manifest_res = writer.WriteAttemptManifest(0, 1);
			    if (manifest_res.is_err()) {
				    throw std::runtime_error(manifest_res.error().what());
			    }
		    };

		    auto first_manifest_file_path = [&](const ShuffleCacheConfig &config) {
			    ShuffleCache cache(config, storage);
			    auto manifest_res =
			        ShuffleCache::ReadAttemptManifest(*storage, cache.ManifestFilePath(), cache.CommittedMarkerPath());
			    if (manifest_res.is_err()) {
				    throw std::runtime_error(manifest_res.error().what());
			    }
			    auto files_res = ShuffleCache::GetPartitionFilesFromManifest(manifest_res.value(), 0);
			    if (files_res.is_err()) {
				    throw std::runtime_error(files_res.error().what());
			    }
			    if (files_res.value().files.empty()) {
				    throw std::runtime_error("expected at least one shuffle file in manifest");
			    }
			    return files_res.value().files[0].path;
		    };

		    auto read_manifest_error_fresh = [&](const ShuffleCacheConfig &config) {
			    DuckDB verify_db(nullptr);
			    Connection verify_conn(verify_db);
			    ConfigureConnectionForS3Endpoint(verify_conn, endpoint_url, access_key, secret_key, region);
			    auto &verify_context = *verify_conn.context;
			    DuckDBPyTransactionGuard verify_guard(verify_context);
			    auto &verify_fs = FileSystem::GetFileSystem(verify_context);
			    auto verify_storage = MakeDuckDBFileSystemShuffleStorage(verify_fs);
			    ShuffleCache cache(config, verify_storage);
			    auto manifest_res = ShuffleCache::ReadAttemptManifest(*verify_storage, cache.ManifestFilePath(),
			                                                          cache.CommittedMarkerPath());
			    std::string error;
			    if (manifest_res.is_err()) {
				    error = manifest_res.error().what();
			    }
			    verify_guard.Commit();
			    return error;
		    };

		    auto read_source_error_fresh = [&](const ShuffleCacheConfig &config) {
			    DuckDB verify_db(nullptr);
			    Connection verify_conn(verify_db);
			    ConfigureConnectionForS3Endpoint(verify_conn, endpoint_url, access_key, secret_key, region);
			    auto &verify_context = *verify_conn.context;
			    DuckDBPyTransactionGuard verify_guard(verify_context);

			    FlightExchangeConfig source_config;
			    source_config.node_id = "reader-node";
			    source_config.local_dirs = {base_uri};
			    source_config.expected_types = types;
			    FlightExchangeSource source(source_config, &verify_context);

			    ExchangeSourceHandle handle;
			    handle.partition_id = 0;
			    handle.attempt_id = 1;
			    handle.node_id = config.node_id;
			    ExchangeSourceFile file;
			    file.path = config.exchange_id;
			    file.file_size = 0;
			    handle.files.push_back(std::move(file));
			    std::vector<ExchangeSourceHandle> handles;
			    handles.push_back(std::move(handle));
			    source.AddSourceHandles(std::move(handles));

			    DataChunk output;
			    output.Initialize(Allocator::DefaultAllocator(), types);
			    std::string error;
			    try {
				    while (source.ReadChunk(output)) {
				    }
			    } catch (const std::exception &ex) {
				    error = ex.what();
			    }
			    verify_guard.Commit();
			    return error;
		    };

		    auto cleanup_attempt = [&](const ShuffleCacheConfig &config) {
			    ShuffleCache cache(config, storage);
			    auto cleanup_res = cache.RemoveAttemptStorage();
			    if (cleanup_res.is_err()) {
				    throw std::runtime_error(cleanup_res.error().what());
			    }
			    return cleanup_res.value();
		    };

		    auto marker_missing_config = make_config("marker_missing");
		    write_attempt(marker_missing_config, {301, 302});
		    {
			    ShuffleCache cache(marker_missing_config, storage);
			    auto remove_res = storage->RemoveAll(cache.CommittedMarkerPath());
			    if (remove_res.is_err()) {
				    throw std::runtime_error(remove_res.error().what());
			    }
		    }

		    auto data_missing_config = make_config("data_missing");
		    write_attempt(data_missing_config, {401, 402});
		    const auto removed_data_file = first_manifest_file_path(data_missing_config);
		    {
			    auto remove_res = storage->RemoveAll(removed_data_file);
			    if (remove_res.is_err()) {
				    throw std::runtime_error(remove_res.error().what());
			    }
		    }

		    auto size_mismatch_config = make_config("size_mismatch");
		    write_attempt(size_mismatch_config, {501, 502});
		    {
			    ShuffleCache cache(size_mismatch_config, storage);
			    auto manifest_text_res = storage->ReadTextFile(cache.ManifestFilePath());
			    if (manifest_text_res.is_err()) {
				    throw std::runtime_error(manifest_text_res.error().what());
			    }
			    auto manifest_text = manifest_text_res.value();
			    auto file_line = manifest_text.find("file=");
			    if (file_line == std::string::npos) {
				    throw std::runtime_error("expected file line in shuffle attempt manifest");
			    }
			    auto first_tab = manifest_text.find('\t', file_line + 5);
			    auto second_tab = manifest_text.find('\t', first_tab + 1);
			    if (first_tab == std::string::npos || second_tab == std::string::npos) {
				    throw std::runtime_error("expected bytes field in shuffle attempt manifest file line");
			    }
			    const auto bytes_text = manifest_text.substr(first_tab + 1, second_tab - first_tab - 1);
			    const auto corrupted_bytes = std::stoull(bytes_text) + 17;
			    auto corrupted_manifest = manifest_text.substr(0, first_tab + 1) + std::to_string(corrupted_bytes) +
			                              manifest_text.substr(second_tab);
			    auto write_res = storage->WriteTextFileAtomically(cache.ManifestFilePath(), corrupted_manifest);
			    if (write_res.is_err()) {
				    throw std::runtime_error(write_res.error().what());
			    }
		    }

		    py::dict out;
		    out["marker_missing_manifest_error"] = read_manifest_error_fresh(marker_missing_config);
		    out["marker_missing_source_error"] = read_source_error_fresh(marker_missing_config);
		    out["marker_missing_cleanup_removed"] = cleanup_attempt(marker_missing_config);
		    out["data_missing_removed_file"] = removed_data_file;
		    out["data_missing_manifest_error"] = read_manifest_error_fresh(data_missing_config);
		    out["data_missing_source_error"] = read_source_error_fresh(data_missing_config);
		    out["data_missing_cleanup_removed"] = cleanup_attempt(data_missing_config);
		    out["size_mismatch_manifest_error"] = read_manifest_error_fresh(size_mismatch_config);
		    out["size_mismatch_source_error"] = read_source_error_fresh(size_mismatch_config);
		    out["size_mismatch_cleanup_removed"] = cleanup_attempt(size_mismatch_config);
		    tx_guard.Commit();
		    return out;
	    },
	    py::arg("base_uri"), py::arg("endpoint_url"), py::arg("access_key"), py::arg("secret_key"),
	    py::arg("region") = "us-east-1",
	    "Exercise MinIO/S3 durable shuffle fault matrix against committed manifest replay.");

	m.def(
	    "flight_exchange_minio_selected_attempt_replay_for_test",
	    [](const std::string &base_uri, const std::string &endpoint_url, const std::string &access_key,
	       const std::string &secret_key, const std::string &region) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    ConfigureConnectionForS3Endpoint(conn, endpoint_url, access_key, secret_key, region);
		    auto &context = *conn.context;
		    DuckDBPyTransactionGuard tx_guard(context);

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    const auto run_id = UUID::ToString(UUID::GenerateRandomUUID());

		    FlightExchangeConfig config;
		    config.node_id = "node-a";
		    config.local_dirs = {base_uri};
		    config.expected_types = types;
		    FlightExchangeManager mgr(config, &context);

		    ExchangeContext exchange_context;
		    exchange_context.query_id = "q";
		    exchange_context.exchange_id = "flight_exchange_minio_selected_attempt_" + run_id;
		    auto exchange = mgr.CreateExchange(exchange_context, 1);
		    auto sink_handle = exchange->AddSink(0);
		    auto lost_instance = exchange->InstantiateSink(sink_handle, 0);
		    auto selected_instance = exchange->InstantiateSink(sink_handle, 1);

		    auto write_attempt = [&](const ExchangeSinkInstanceHandle &instance, std::vector<int32_t> values) {
			    auto sink = mgr.CreateSink(instance);
			    DataChunk chunk;
			    chunk.Initialize(Allocator::DefaultAllocator(), types);
			    chunk.SetCardinality(values.size());
			    for (idx_t row = 0; row < values.size(); row++) {
				    chunk.SetValue(0, row, Value::INTEGER(values[row]));
			    }
			    auto add_res = sink->AddChunk(0, chunk);
			    if (add_res.is_err()) {
				    throw std::runtime_error(add_res.error().what());
			    }
			    auto finish_res = sink->Finish();
			    if (finish_res.is_err()) {
				    throw std::runtime_error(finish_res.error().what());
			    }
		    };

		    write_attempt(lost_instance, {701});
		    const auto lost_node_id = mgr.config().node_id;
		    write_attempt(selected_instance, {801, 802});
		    const auto selected_node_id = mgr.config().node_id;
		    exchange->SinkFinished(sink_handle, 1, selected_node_id, 0);
		    exchange->AllRequiredSinksFinished();
		    auto handles = exchange->GetSourceHandles();

		    auto read_source_values = [&](const std::vector<ExchangeSourceHandle> &source_handles) {
			    auto source = mgr.CreateSource();
			    auto copied_handles = source_handles;
			    source->AddSourceHandles(std::move(copied_handles));
			    DataChunk output;
			    output.Initialize(Allocator::DefaultAllocator(), types);
			    py::list values;
			    while (source->ReadChunk(output)) {
				    for (idx_t row = 0; row < output.size(); row++) {
					    values.append(output.GetValue(0, row).GetValue<int32_t>());
				    }
			    }
			    return values;
		    };

		    auto selected_values_before_cleanup = read_source_values(handles);

		    // The lost attempt committed data, but its status was not selected. If it
		    // is reported later, it must be treated as a loser and cleaned up.
		    exchange->SinkFinished(sink_handle, 0, lost_node_id, 0);

		    auto read_selected_values_fresh = [&]() {
			    DuckDB verify_db(nullptr);
			    Connection verify_conn(verify_db);
			    ConfigureConnectionForS3Endpoint(verify_conn, endpoint_url, access_key, secret_key, region);
			    auto &verify_context = *verify_conn.context;
			    DuckDBPyTransactionGuard verify_guard(verify_context);

			    FlightExchangeConfig verify_config;
			    verify_config.node_id = "reader-node";
			    verify_config.local_dirs = {base_uri};
			    verify_config.expected_types = types;
			    FlightExchangeSource source(verify_config, &verify_context);
			    auto copied_handles = handles;
			    source.AddSourceHandles(std::move(copied_handles));
			    DataChunk output;
			    output.Initialize(Allocator::DefaultAllocator(), types);
			    py::list values;
			    while (source.ReadChunk(output)) {
				    for (idx_t row = 0; row < output.size(); row++) {
					    values.append(output.GetValue(0, row).GetValue<int32_t>());
				    }
			    }
			    verify_guard.Commit();
			    return values;
		    };

		    auto lost_manifest_error_fresh = [&]() {
			    DuckDB verify_db(nullptr);
			    Connection verify_conn(verify_db);
			    ConfigureConnectionForS3Endpoint(verify_conn, endpoint_url, access_key, secret_key, region);
			    auto &verify_context = *verify_conn.context;
			    DuckDBPyTransactionGuard verify_guard(verify_context);
			    auto &verify_fs = FileSystem::GetFileSystem(verify_context);
			    auto verify_storage = MakeDuckDBFileSystemShuffleStorage(verify_fs);
			    ShuffleCacheConfig lost_config;
			    lost_config.exchange_id = lost_instance.output_location;
			    lost_config.node_id = lost_node_id;
			    lost_config.num_partitions = 1;
			    lost_config.local_dirs = {base_uri};
			    ShuffleCache lost_cache(lost_config, verify_storage);
			    auto manifest_res = ShuffleCache::ReadAttemptManifest(*verify_storage, lost_cache.ManifestFilePath(),
			                                                          lost_cache.CommittedMarkerPath());
			    std::string error;
			    if (manifest_res.is_err()) {
				    error = manifest_res.error().what();
			    }
			    verify_guard.Commit();
			    return error;
		    };

		    auto selected_values_after_cleanup = read_selected_values_fresh();
		    auto lost_manifest_after_cleanup_error = lost_manifest_error_fresh();

		    ShuffleCacheConfig selected_config;
		    selected_config.exchange_id = selected_instance.output_location;
		    selected_config.node_id = selected_node_id;
		    selected_config.num_partitions = 1;
		    selected_config.local_dirs = {base_uri};
		    auto &fs = FileSystem::GetFileSystem(context);
		    auto *opener = ClientData::Get(context).file_opener.get();
		    auto storage = MakeDuckDBFileSystemShuffleStorage(fs, opener);
		    ShuffleCache selected_cache(selected_config, storage);
		    auto selected_cleanup_res = selected_cache.RemoveAttemptStorage();
		    if (selected_cleanup_res.is_err()) {
			    throw std::runtime_error(selected_cleanup_res.error().what());
		    }

		    exchange->Close();

		    py::list handle_attempts;
		    py::list handle_paths;
		    py::list handle_node_ids;
		    for (const auto &handle : handles) {
			    handle_attempts.append(handle.attempt_id);
			    handle_node_ids.append(handle.node_id);
			    handle_paths.append(handle.files.empty() ? string() : handle.files[0].path);
		    }

		    py::dict out;
		    out["lost_output_location"] = lost_instance.output_location;
		    out["selected_output_location"] = selected_instance.output_location;
		    out["handle_attempts"] = handle_attempts;
		    out["handle_paths"] = handle_paths;
		    out["handle_node_ids"] = handle_node_ids;
		    out["selected_values_before_cleanup"] = selected_values_before_cleanup;
		    out["selected_values_after_loser_cleanup"] = selected_values_after_cleanup;
		    out["lost_manifest_after_cleanup_error"] = lost_manifest_after_cleanup_error;
		    out["selected_cleanup_removed"] = selected_cleanup_res.value();
		    auto cleanup = ShuffleCacheRegistry::Instance().RemoveAndCleanupByQuery(exchange_context.query_id, storage);
		    if (cleanup.cleanup_errors != 0 || cleanup.cleanup_pending != 0) {
			    throw std::runtime_error("failed to clean MinIO selected-attempt test query");
		    }
		    auto retire = ShuffleCacheRegistry::Instance().RetireQuery(exchange_context.query_id);
		    if (retire.is_err()) {
			    throw std::runtime_error(retire.error().what());
		    }
		    tx_guard.Commit();
		    return out;
	    },
	    py::arg("base_uri"), py::arg("endpoint_url"), py::arg("access_key"), py::arg("secret_key"),
	    py::arg("region") = "us-east-1",
	    "Exercise FTE-style selected-attempt replay and loser cleanup on real MinIO/S3 shuffle storage.");

	m.def(
	    "shuffle_cache_rejects_object_storage_local_dir_for_test",
	    []() {
		    using namespace duckdb::distributed;

		    py::dict out;
		    try {
			    ShuffleCacheConfig config;
			    config.exchange_id = "shuffle_cache_object_storage_reject_test__sink_0__attempt_1";
			    config.node_id = "node-a";
			    config.num_partitions = 1;
			    config.local_dirs = {"s3://bucket/shuffle"};
			    ShuffleCache cache(std::move(config));
			    out["rejected"] = false;
			    out["error"] = "";
		    } catch (const std::exception &ex) {
			    out["rejected"] = true;
			    out["error"] = ex.what();
		    }
		    return out;
	    },
	    "Exercise that object-storage shuffle dirs fail fast until the object backend exists.");

	m.def(
	    "shuffle_cache_duckdb_filesystem_storage_accepts_object_dir_for_test",
	    []() {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    py::dict out;
		    try {
			    DuckDB db(nullptr);
			    Connection conn(db);
			    auto &fs = FileSystem::GetFileSystem(*conn.context);
			    auto storage = MakeDuckDBFileSystemShuffleStorage(fs);

			    ShuffleCacheConfig config;
			    config.exchange_id = "shuffle_cache_duckdb_fs_object_accept_test__sink_0__attempt_1";
			    config.node_id = "node-a";
			    config.num_partitions = 1;
			    config.local_dirs = {"s3://bucket/shuffle"};
			    ShuffleCache cache(std::move(config), storage);
			    out["accepted"] = true;
			    out["error"] = "";
		    } catch (const std::exception &ex) {
			    out["accepted"] = false;
			    out["error"] = ex.what();
		    }
		    return out;
	    },
	    "Exercise that object-capable DuckDB FS storage passes object local_dirs constructor validation.");

	m.def(
	    "shuffle_cache_fake_object_no_rename_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    class FakeObjectShuffleStorage final : public ShuffleStorage {
			public:
			    explicit FakeObjectShuffleStorage(FileSystem &fs) : fs_(fs) {
			    }

			    DuckDBResult<void> CreateDirectories(const std::string &path) const override {
				    directories.insert(path);
				    return DuckDBResult<void>::ok();
			    }

			    bool IsRegularFile(const std::string &path) const override {
				    if (text_files.find(path) != text_files.end()) {
					    return true;
				    }
				    try {
					    return fs_.FileExists(path);
				    } catch (...) {
					    return false;
				    }
			    }

			    DuckDBResult<idx_t> FileSize(const std::string &path) const override {
				    auto text_entry = text_files.find(path);
				    if (text_entry != text_files.end()) {
					    return DuckDBResult<idx_t>::ok(text_entry->second.size());
				    }
				    try {
					    auto handle = fs_.OpenFile(path, FileFlags::FILE_FLAGS_READ);
					    auto size = handle->GetFileSize();
					    if (size > std::numeric_limits<idx_t>::max()) {
						    return DuckDBResult<idx_t>::err(
						        DuckDBError::io_error("fake object storage file size exceeds idx_t max: " + path));
					    }
					    return DuckDBResult<idx_t>::ok(static_cast<idx_t>(size));
				    } catch (const std::exception &ex) {
					    return DuckDBResult<idx_t>::err(
					        DuckDBError::io_error("fake object storage file stat failed: " + std::string(ex.what())));
				    } catch (...) {
					    return DuckDBResult<idx_t>::err(
					        DuckDBError::io_error("fake object storage file stat failed: " + path));
				    }
			    }

			    DuckDBResult<void> WriteTextFileAtomically(const std::string &path,
			                                               const std::string &contents) const override {
				    text_files[path] = contents;
				    text_puts++;
				    return DuckDBResult<void>::ok();
			    }

			    DuckDBResult<std::string> ReadTextFile(const std::string &path) const override {
				    auto entry = text_files.find(path);
				    if (entry == text_files.end()) {
					    return DuckDBResult<std::string>::err(
					        DuckDBError::io_error("fake object storage text file missing: " + path));
				    }
				    return DuckDBResult<std::string>::ok(entry->second);
			    }

			    DuckDBResult<idx_t> RemoveAll(const std::string &path) const override {
				    idx_t removed = 0;
				    auto prefix = path;
				    if (!prefix.empty() && prefix.back() != '/') {
					    prefix += "/";
				    }
				    for (auto it = text_files.begin(); it != text_files.end();) {
					    if (it->first == path || it->first.rfind(prefix, 0) == 0) {
						    it = text_files.erase(it);
						    removed++;
					    } else {
						    ++it;
					    }
				    }
				    RemoveDistributedCopyDirectoryTree(fs_, path);
				    return DuckDBResult<idx_t>::ok(removed);
			    }

			    DuckDBResult<std::shared_ptr<arrow::io::OutputStream>>
			    OpenArrowOutput(const std::string &path) const override {
				    (void)path;
				    return DuckDBResult<std::shared_ptr<arrow::io::OutputStream>>::err(
				        DuckDBError::invalid_state_error("fake object storage arrow output is not implemented"));
			    }

			    DuckDBResult<std::shared_ptr<arrow::io::InputStream>>
			    OpenArrowInput(const std::string &path) const override {
				    (void)path;
				    return DuckDBResult<std::shared_ptr<arrow::io::InputStream>>::err(
				        DuckDBError::invalid_state_error("fake object storage arrow input is not implemented"));
			    }

			    bool HasTextFile(const std::string &path) const {
				    return text_files.find(path) != text_files.end();
			    }

			    std::string TextFile(const std::string &path) const {
				    auto entry = text_files.find(path);
				    return entry == text_files.end() ? std::string() : entry->second;
			    }

			    mutable std::unordered_map<std::string, std::string> text_files;
			    mutable std::unordered_set<std::string> directories;
			    mutable idx_t text_puts = 0;
			    mutable idx_t arrow_output_opens = 0;
			    mutable idx_t arrow_input_opens = 0;

			private:
			    FileSystem &fs_;
		    };

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &fs = FileSystem::GetFileSystem(*conn.context);

		    auto storage = std::make_shared<FakeObjectShuffleStorage>(fs);
		    auto data_dir = local_dir + "/partition_0";
		    if (!fs.DirectoryExists(data_dir)) {
			    fs.CreateDirectoriesRecursive(data_dir);
		    }
		    auto data_file = data_dir + "/batch.arrow";
		    const std::string payload = "payload";
		    {
			    std::ofstream output(data_file, std::ios::out | std::ios::binary);
			    output << payload;
			    output.close();
			    if (!output) {
				    throw std::runtime_error("failed to write fake object data file");
			    }
		    }

		    ShuffleCacheConfig config;
		    config.exchange_id = "shuffle_cache_fake_object_manifest_test__sink_0__attempt_7";
		    config.node_id = "node-a";
		    config.num_partitions = 1;
		    config.local_dirs = {local_dir};
		    ShuffleCache writer(config, storage);
		    ShufflePartitionFile file;
		    file.path = data_file;
		    file.rows = 1;
		    file.bytes = payload.size();
		    auto reg_res = writer.RegisterPartitionFile(0, std::move(file));
		    if (reg_res.is_err()) {
			    throw std::runtime_error(reg_res.error().what());
		    }
		    auto manifest_res = writer.WriteAttemptManifest(0, 7);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }

		    auto manifest_path = writer.ManifestFilePath();
		    auto marker_path = writer.CommittedMarkerPath();
		    ShuffleCache reader(config, storage);
		    auto manifest_files_res = reader.GetPartitionFilesFromManifest(0);
		    if (manifest_files_res.is_err()) {
			    throw std::runtime_error(manifest_files_res.error().what());
		    }

		    py::dict out;
		    out["committed_manifest"] = reader.HasCommittedManifest();
		    out["manifest_exists"] = storage->HasTextFile(manifest_path);
		    out["marker_exists"] = storage->HasTextFile(marker_path);
		    out["manifest_tmp_exists"] = storage->HasTextFile(manifest_path + ".tmp");
		    out["marker_tmp_exists"] = storage->HasTextFile(marker_path + ".tmp");
		    out["manifest"] = storage->TextFile(manifest_path);
		    out["manifest_file_count"] = manifest_files_res.value().files.size();
		    out["manifest_total_rows"] = manifest_files_res.value().total_rows;
		    out["manifest_total_bytes"] = manifest_files_res.value().total_bytes;
		    out["text_puts"] = storage->text_puts;
		    out["arrow_output_opens"] = storage->arrow_output_opens;
		    out["arrow_input_opens"] = storage->arrow_input_opens;
		    return out;
	    },
	    py::arg("local_dir"), "Exercise ShuffleCache manifest commit on a fake no-rename object storage backend.");

	m.def(
	    "flight_exchange_source_rejects_local_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const std::string output_location = "flight_exchange_source_manifest_recovery_test__sink_0__attempt_1";
		    const std::string node_id = "127.0.0.1";
		    const std::string server_epoch = "unpublished-local-manifest-epoch";

		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = output_location;
		    write_config.node_id = node_id;
		    write_config.num_partitions = 2;
		    write_config.local_dirs = {local_dir};
		    ShuffleCache writer(write_config);

		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(4);
		    input.SetValue(0, 0, Value::INTEGER(21));
		    input.SetValue(0, 1, Value::INTEGER(22));
		    input.SetValue(0, 2, Value::INTEGER(23));
		    input.SetValue(0, 3, Value::INTEGER(24));
		    auto write_res = writer.WriteChunk(context, input, 1, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer.FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto manifest_res = writer.WriteAttemptManifest(0, 1);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }
		    ShuffleCacheRegistry::Instance().Remove(output_location);

		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = server_epoch;
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }

		    FlightExchangeConfig source_config;
		    source_config.node_id = node_id;
		    source_config.local_dirs = {local_dir};
		    source_config.expected_types = types;
		    FlightExchangeSource source(source_config, &context);

		    ExchangeSourceHandle handle;
		    handle.partition_id = 1;
		    handle.attempt_id = 1;
		    handle.node_id = node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = server_epoch;
		    ExchangeSourceFile file;
		    file.path = output_location;
		    file.file_size = 0;
		    handle.files.push_back(std::move(file));
		    std::vector<ExchangeSourceHandle> handles;
		    handles.push_back(std::move(handle));
		    source.AddSourceHandles(std::move(handles));

		    DataChunk output;
		    output.Initialize(Allocator::DefaultAllocator(), types);
		    py::list values;
		    while (source.ReadChunk(output)) {
			    for (idx_t row = 0; row < output.size(); row++) {
				    values.append(output.GetValue(0, row).GetValue<int32_t>());
			    }
		    }

		    py::dict out;
		    out["values"] = values;
		    out["finished"] = source.IsFinished();
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    return out;
	    },
	    py::arg("local_dir"),
	    "Verify that FlightExchangeSource rejects a local manifest missing published server identity.");

	m.def(
	    "flight_exchange_source_rejects_shared_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const std::string output_location =
		        "flight_exchange_source_shared_manifest_recovery_test__sink_0__attempt_1";
		    const std::string writer_node_id = "127.0.0.1";
		    const std::string reader_node_id = "reader-node";
		    const std::string server_epoch = "unpublished-shared-manifest-epoch";

		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = output_location;
		    write_config.node_id = writer_node_id;
		    write_config.num_partitions = 2;
		    write_config.local_dirs = {local_dir};
		    ShuffleCache writer(write_config);

		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(3);
		    input.SetValue(0, 0, Value::INTEGER(71));
		    input.SetValue(0, 1, Value::INTEGER(72));
		    input.SetValue(0, 2, Value::INTEGER(73));
		    auto write_res = writer.WriteChunk(context, input, 1, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer.FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto manifest_res = writer.WriteAttemptManifest(0, 1);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }
		    ShuffleCacheRegistry::Instance().Remove(output_location);

		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = server_epoch;
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }

		    FlightExchangeConfig source_config;
		    source_config.node_id = reader_node_id;
		    source_config.local_dirs = {local_dir};
		    source_config.expected_types = types;
		    FlightExchangeSource source(source_config, &context);

		    ExchangeSourceHandle handle;
		    handle.partition_id = 1;
		    handle.attempt_id = 1;
		    handle.node_id = writer_node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = server_epoch;
		    ExchangeSourceFile file;
		    file.path = output_location;
		    file.file_size = 0;
		    handle.files.push_back(std::move(file));
		    std::vector<ExchangeSourceHandle> handles;
		    handles.push_back(std::move(handle));
		    source.AddSourceHandles(std::move(handles));

		    DataChunk output;
		    output.Initialize(Allocator::DefaultAllocator(), types);
		    py::list values;
		    while (source.ReadChunk(output)) {
			    for (idx_t row = 0; row < output.size(); row++) {
				    values.append(output.GetValue(0, row).GetValue<int32_t>());
			    }
		    }

		    py::dict out;
		    out["values"] = values;
		    out["finished"] = source.IsFinished();
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    out["writer_node_id"] = writer_node_id;
		    out["reader_node_id"] = reader_node_id;
		    return out;
	    },
	    py::arg("local_dir"),
	    "Verify that FlightExchangeSource rejects a shared manifest missing published server identity.");

	m.def(
	    "flight_exchange_source_write_unpublished_shared_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const std::string output_location = "flight_exchange_source_cross_process_manifest_test__sink_0__attempt_1";
		    const std::string writer_node_id = "127.0.0.1";
		    const idx_t sink_partition_id = 0;
		    const idx_t attempt_id = 1;
		    const idx_t output_partition_id = 1;

		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = output_location;
		    write_config.node_id = writer_node_id;
		    write_config.num_partitions = 2;
		    write_config.local_dirs = {local_dir};
		    ShuffleCache writer(write_config);

		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(3);
		    input.SetValue(0, 0, Value::INTEGER(81));
		    input.SetValue(0, 1, Value::INTEGER(82));
		    input.SetValue(0, 2, Value::INTEGER(83));
		    auto write_res = writer.WriteChunk(context, input, output_partition_id, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer.FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto manifest_res = writer.WriteAttemptManifest(sink_partition_id, attempt_id);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }

		    py::dict out;
		    out["output_location"] = output_location;
		    out["writer_node_id"] = writer_node_id;
		    out["partition_id"] = output_partition_id;
		    out["attempt_id"] = attempt_id;
		    out["manifest_path"] = writer.ManifestFilePath();
		    out["committed_path"] = writer.CommittedMarkerPath();
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    return out;
	    },
	    py::arg("local_dir"), "Write a committed but unpublished shared-manifest exchange attempt.");

	m.def(
	    "flight_exchange_source_rejects_unpublished_shared_manifest_for_test",
	    [](const std::string &local_dir, const std::string &output_location, const std::string &writer_node_id,
	       const std::string &reader_node_id, int64_t partition_id, int64_t attempt_id) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    if (partition_id < 0 || attempt_id < 0) {
			    throw std::runtime_error("partition_id and attempt_id must be non-negative");
		    }

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    const std::string server_epoch = "unpublished-cross-process-manifest-epoch";
		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = server_epoch;
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }
		    FlightExchangeConfig source_config;
		    source_config.node_id = reader_node_id;
		    source_config.local_dirs = {local_dir};
		    source_config.expected_types = types;
		    FlightExchangeSource source(source_config, &context);

		    ExchangeSourceHandle handle;
		    handle.partition_id = static_cast<idx_t>(partition_id);
		    handle.attempt_id = static_cast<idx_t>(attempt_id);
		    handle.node_id = writer_node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = server_epoch;
		    ExchangeSourceFile file;
		    file.path = output_location;
		    file.file_size = 0;
		    handle.files.push_back(std::move(file));
		    std::vector<ExchangeSourceHandle> handles;
		    handles.push_back(std::move(handle));
		    source.AddSourceHandles(std::move(handles));

		    DataChunk output;
		    output.Initialize(Allocator::DefaultAllocator(), types);
		    py::list values;
		    while (source.ReadChunk(output)) {
			    for (idx_t row = 0; row < output.size(); row++) {
				    values.append(output.GetValue(0, row).GetValue<int32_t>());
			    }
		    }

		    py::dict out;
		    out["values"] = values;
		    out["finished"] = source.IsFinished();
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    out["writer_node_id"] = writer_node_id;
		    out["reader_node_id"] = reader_node_id;
		    return out;
	    },
	    py::arg("local_dir"), py::arg("output_location"), py::arg("writer_node_id"), py::arg("reader_node_id"),
	    py::arg("partition_id"), py::arg("attempt_id"),
	    "Verify that a fresh process rejects a committed but unpublished shared-manifest exchange attempt.");

	m.def(
	    "flight_server_rejects_unpublished_shared_manifest_for_test",
	    [](const std::string &local_dir, const std::string &output_location, const std::string &writer_node_id,
	       int64_t partition_id) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    if (partition_id < 0) {
			    throw std::runtime_error("partition_id must be non-negative");
		    }

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = "shared-manifest-reader-epoch";
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }

		    FlightExchangeConfig source_config;
		    source_config.local_dirs = {local_dir};
		    source_config.node_id = "reader-node";
		    ExchangeSourceHandle handle;
		    handle.partition_id = static_cast<idx_t>(partition_id);
		    handle.attempt_id = 1;
		    handle.node_id = writer_node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = "shared-manifest-reader-epoch";
		    handle.files.push_back(ExchangeSourceFile(output_location, 0));
		    string read_error;
		    try {
			    ReadIntegerExchangeSourceForTest(context, source_config, std::move(handle));
		    } catch (const std::exception &ex) {
			    read_error = ex.what();
		    }
		    auto stop_res = server.Stop();
		    if (stop_res.is_err()) {
			    throw std::runtime_error(stop_res.error().what());
		    }

		    py::dict out;
		    out["fetch_error"] = !read_error.empty();
		    out["error"] = read_error;
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    return out;
	    },
	    py::arg("local_dir"), py::arg("output_location"), py::arg("writer_node_id"), py::arg("partition_id"),
	    "Verify that an unpublished manifest is not served by a fresh FlightServer process.");

	m.def(
	    "remote_exchange_source_rejects_local_dirs_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    const std::string current_node_id = "127.0.0.1";
		    const std::string server_epoch = "unpublished-serialized-manifest-epoch";

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const std::string output_location = "remote_exchange_source_local_dirs_roundtrip_test__sink_0__attempt_1";

		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = output_location;
		    write_config.node_id = current_node_id;
		    write_config.num_partitions = 2;
		    write_config.local_dirs = {local_dir};
		    ShuffleCache writer(write_config);

		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(2);
		    input.SetValue(0, 0, Value::INTEGER(41));
		    input.SetValue(0, 1, Value::INTEGER(42));
		    auto write_res = writer.WriteChunk(context, input, 1, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer.FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto manifest_res = writer.WriteAttemptManifest(0, 1);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }
		    ShuffleCacheRegistry::Instance().Remove(output_location);

		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = server_epoch;
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }

		    Allocator allocator;
		    PhysicalPlan plan(allocator);
		    vector<idx_t> partition_indices = {1};
		    vector<string> source_nodes = {current_node_id};
		    std::vector<ExchangeSourceHandle> source_handles;
		    ExchangeSourceHandle handle;
		    handle.partition_id = 1;
		    handle.attempt_id = 1;
		    handle.node_id = current_node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = server_epoch;
		    ExchangeSourceFile file;
		    file.path = output_location;
		    file.file_size = 0;
		    handle.files.push_back(std::move(file));
		    source_handles.push_back(std::move(handle));

		    FlightExchangeConfig flight_config;
		    flight_config.node_id = current_node_id;
		    flight_config.local_dirs = {local_dir};
		    auto exchange_mgr = std::make_shared<FlightExchangeManager>(std::move(flight_config));

		    auto &source =
		        plan.Make<PhysicalRemoteExchangeSource>(types, 2, "remote_exchange_source_local_dirs_roundtrip_test",
		                                                partition_indices, source_handles, exchange_mgr, source_nodes);

		    MemoryStream stream(allocator);
		    SerializationOptions options;
		    BinarySerializer serializer(stream, options);
		    serializer.Begin();
		    source.Serialize(serializer);
		    serializer.End();

		    stream.Rewind();
		    BinaryDeserializer deserializer(stream);
		    deserializer.Begin();
		    auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
		    deserializer.End();
		    auto *source_ptr = dynamic_cast<PhysicalRemoteExchangeSource *>(deserialized_op.get());
		    if (!source_ptr) {
			    throw std::runtime_error("deserialized operator is not PhysicalRemoteExchangeSource");
		    }

		    ThreadContext thread_context(context);
		    ExecutionContext execution_context(context, thread_context, nullptr);
		    InterruptState interrupt_state;
		    auto global_state = source_ptr->GetGlobalSourceState(context);
		    auto local_state = source_ptr->GetLocalSourceState(execution_context, *global_state);
		    OperatorSourceInput source_input {*global_state, *local_state, interrupt_state};

		    DataChunk output;
		    output.Initialize(Allocator::DefaultAllocator(), types);
		    py::list values;
		    while (true) {
			    auto result = source_ptr->GetData(execution_context, output, source_input);
			    if (output.size() > 0) {
				    for (idx_t row = 0; row < output.size(); row++) {
					    values.append(output.GetValue(0, row).GetValue<int32_t>());
				    }
			    }
			    if (result == SourceResultType::FINISHED) {
				    break;
			    }
			    if (result == SourceResultType::BLOCKED) {
				    throw std::runtime_error("remote exchange source unexpectedly blocked");
			    }
		    }

		    py::dict out;
		    out["values"] = values;
		    out["node_id"] = current_node_id;
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    return out;
	    },
	    py::arg("local_dir"), "Verify that serialized EXCHANGE_SOURCE cannot bypass catalog identity via local_dirs.");

	m.def(
	    "flight_server_rejects_unpublished_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const std::string output_location = "flight_server_manifest_recovery_test__sink_0__attempt_1";
		    const std::string node_id = "node-a";

		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = output_location;
		    write_config.node_id = node_id;
		    write_config.num_partitions = 2;
		    write_config.local_dirs = {local_dir};
		    ShuffleCache writer(write_config);

		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(2);
		    input.SetValue(0, 0, Value::INTEGER(31));
		    input.SetValue(0, 1, Value::INTEGER(32));
		    auto write_res = writer.WriteChunk(context, input, 1, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer.FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto manifest_res = writer.WriteAttemptManifest(0, 1);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }
		    ShuffleCacheRegistry::Instance().Remove(output_location);

		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = "unpublished-manifest-epoch";
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }

		    FlightExchangeConfig source_config;
		    source_config.local_dirs = {local_dir};
		    source_config.node_id = "reader-node";
		    ExchangeSourceHandle handle;
		    handle.partition_id = 1;
		    handle.attempt_id = 1;
		    handle.node_id = node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = "unpublished-manifest-epoch";
		    handle.files.push_back(ExchangeSourceFile(output_location, 0));
		    string read_error;
		    try {
			    ReadIntegerExchangeSourceForTest(context, source_config, std::move(handle));
		    } catch (const std::exception &ex) {
			    read_error = ex.what();
		    }
		    auto stop_res = server.Stop();
		    if (stop_res.is_err()) {
			    throw std::runtime_error(stop_res.error().what());
		    }

		    py::dict out;
		    out["fetch_error"] = !read_error.empty();
		    out["error"] = read_error;
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    return out;
	    },
	    py::arg("local_dir"), "Verify FlightServer rejects a manifest with no published catalog entry.");

	m.def(
	    "flight_server_uncommitted_attempt_rejected_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const std::string output_location = "flight_server_uncommitted_rejected_test__sink_0__attempt_1";
		    const std::string node_id = "node-a";

		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = output_location;
		    write_config.node_id = node_id;
		    write_config.num_partitions = 1;
		    write_config.local_dirs = {local_dir};
		    auto writer = std::make_shared<ShuffleCache>(write_config);

		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(1);
		    input.SetValue(0, 0, Value::INTEGER(61));
		    auto write_res = writer->WriteChunk(context, input, 0, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer->FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto partial_files_res = writer->GetPartitionFiles(0);
		    if (partial_files_res.is_err()) {
			    throw std::runtime_error(partial_files_res.error().what());
		    }
		    const std::string server_epoch = "uncommitted-attempt-epoch";
		    auto register_res = ShuffleCacheRegistry::Instance().Register(
		        output_location, writer, "flight-server-uncommitted-rejected-query", server_epoch, 1);
		    if (register_res.is_err()) {
			    throw std::runtime_error(register_res.error().what());
		    }

		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = server_epoch;
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }

		    FlightExchangeConfig source_config;
		    source_config.local_dirs = {local_dir};
		    source_config.node_id = "reader-node";
		    ExchangeSourceHandle handle;
		    handle.partition_id = 0;
		    handle.attempt_id = 1;
		    handle.node_id = node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = server_epoch;
		    handle.files.push_back(ExchangeSourceFile(output_location, 0));
		    string read_error;
		    try {
			    ReadIntegerExchangeSourceForTest(context, source_config, std::move(handle));
		    } catch (const std::exception &ex) {
			    read_error = ex.what();
		    }
		    auto stop_res = server.Stop();
		    ShuffleCacheRegistry::Instance().Remove(output_location);
		    if (stop_res.is_err()) {
			    throw std::runtime_error(stop_res.error().what());
		    }

		    py::dict out;
		    out["partial_file_count"] = partial_files_res.value().files.size();
		    out["committed_manifest"] = writer->HasCommittedManifest();
		    out["fetch_error"] = !read_error.empty();
		    out["error"] = read_error;
		    return out;
	    },
	    py::arg("local_dir"), "Exercise that FlightServer rejects uncommitted attempts without committed manifest.");

	m.def(
	    "distributed_copy_finalize_missing_staging_preflight_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    auto staging_root = local_dir + "/copy_staging/run-1";
		    auto first_staged = staging_root + "/w_0/part.parquet";
		    auto missing_staged = staging_root + "/w_1/part.parquet";
		    auto final_root = local_dir + "/copy_final";
		    auto &fs = FileSystem::GetFileSystem(context);
		    fs.CreateDirectoriesRecursive(staging_root + "/w_0");
		    {
			    std::ofstream output(first_staged, std::ios::out | std::ios::binary);
			    output << "first";
		    }

		    DistributedCopySpec spec;
		    spec.file_path = final_root;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;

		    vector<DistributedCopyFileInfo> files;
		    DistributedCopyFileInfo first;
		    first.staging_path = first_staged;
		    first.row_count = 1;
		    files.push_back(std::move(first));
		    DistributedCopyFileInfo second;
		    second.staging_path = missing_staged;
		    second.row_count = 1;
		    files.push_back(std::move(second));

		    auto finalize_res = FinalizeCopyFiles(spec, staging_root, std::move(files), context);

		    std::vector<std::string> final_files;
		    bool final_root_exists = false;
		    try {
			    final_root_exists = fs.DirectoryExists(final_root);
			    if (final_root_exists) {
				    ListDistributedCopyFilesRecursive(fs, final_root, final_files);
			    }
		    } catch (...) {
			    final_root_exists = false;
		    }

		    py::dict out;
		    out["finalize_error"] = finalize_res.is_err();
		    out["error"] = finalize_res.is_err() ? finalize_res.error().what() : "";
		    out["first_staging_exists"] = fs.FileExists(first_staged);
		    out["missing_staging_exists"] = fs.FileExists(missing_staged);
		    out["final_root_exists"] = final_root_exists;
		    out["final_file_count"] = final_files.size();
		    return out;
	    },
	    py::arg("local_dir"),
	    "Exercise COPY finalize preflight: missing staging files fail before any final output is moved.");

	m.def(
	    "distributed_copy_finalize_commit_manifest_idempotent_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    auto staging_root = local_dir + "/copy_staging/run-commit";
		    auto first_staged = staging_root + "/w_0/part.parquet";
		    auto second_staged = staging_root + "/w_1/part.parquet";
		    auto final_root = local_dir + "/copy_final_commit";
		    fs.CreateDirectoriesRecursive(staging_root + "/w_0");
		    fs.CreateDirectoriesRecursive(staging_root + "/w_1");

		    const std::string first_body = "first";
		    const std::string second_body = "second";
		    {
			    std::ofstream output(first_staged, std::ios::out | std::ios::binary);
			    output << first_body;
		    }
		    {
			    std::ofstream output(second_staged, std::ios::out | std::ios::binary);
			    output << second_body;
		    }

		    DistributedCopySpec spec;
		    spec.file_path = final_root;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;

		    auto make_files = [&]() {
			    vector<DistributedCopyFileInfo> files;
			    DistributedCopyFileInfo first;
			    first.staging_path = first_staged;
			    first.row_count = 1;
			    first.file_size_bytes = first_body.size();
			    files.push_back(std::move(first));
			    DistributedCopyFileInfo second;
			    second.staging_path = second_staged;
			    second.row_count = 2;
			    second.file_size_bytes = second_body.size();
			    files.push_back(std::move(second));
			    return files;
		    };

		    auto commit_paths = BuildDistributedCopyFinalizeCommitPathsFromStagingRoot(fs, final_root, staging_root);
		    auto first_res = FinalizeCopyFiles(spec, staging_root, make_files(), context);
		    auto second_res = FinalizeCopyFiles(spec, staging_root, make_files(), context);

		    std::vector<std::string> final_files;
		    bool final_root_exists = false;
		    try {
			    final_root_exists = fs.DirectoryExists(final_root);
			    if (final_root_exists) {
				    ListDistributedCopyFilesRecursive(fs, final_root, final_files);
			    }
		    } catch (...) {
			    final_root_exists = false;
		    }

		    py::dict out;
		    out["first_finalize_error"] = first_res.is_err();
		    out["first_error"] = first_res.is_err() ? first_res.error().what() : "";
		    out["second_finalize_error"] = second_res.is_err();
		    out["second_error"] = second_res.is_err() ? second_res.error().what() : "";
		    out["first_rows_copied"] = first_res.is_ok() ? first_res.value().rows_copied : 0;
		    out["second_rows_copied"] = second_res.is_ok() ? second_res.value().rows_copied : 0;
		    out["manifest_exists"] = fs.FileExists(commit_paths.manifest_path);
		    out["committed_exists"] = fs.FileExists(commit_paths.committed_marker_path);
		    out["staging_root_exists"] = fs.DirectoryExists(staging_root);
		    out["final_root_exists"] = final_root_exists;
		    out["final_file_count"] = final_files.size();
		    return out;
	    },
	    py::arg("local_dir"), "Exercise COPY finalize durable commit manifest and committed-marker idempotence.");

	m.def(
	    "distributed_copy_finalize_replays_inprogress_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    auto staging_root = local_dir + "/copy_staging/run-replay";
		    auto first_staged = staging_root + "/w_0/part.parquet";
		    auto second_staged = staging_root + "/w_1/part.parquet";
		    auto final_root = local_dir + "/copy_final_replay";
		    auto first_final = final_root + "/part-0.parquet";
		    auto second_final = final_root + "/part-1.parquet";
		    fs.CreateDirectoriesRecursive(staging_root + "/w_0");
		    fs.CreateDirectoriesRecursive(staging_root + "/w_1");
		    fs.CreateDirectoriesRecursive(final_root);

		    const std::string first_body = "first";
		    const std::string second_body = "second";
		    {
			    std::ofstream output(first_staged, std::ios::out | std::ios::binary);
			    output << first_body;
		    }
		    {
			    std::ofstream output(second_staged, std::ios::out | std::ios::binary);
			    output << second_body;
		    }

		    DistributedCopySpec spec;
		    spec.file_path = final_root;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;

		    vector<DistributedCopyFileInfo> manifest_files;
		    DistributedCopyFileInfo first_manifest;
		    first_manifest.staging_path = first_staged;
		    first_manifest.final_path = first_final;
		    first_manifest.row_count = 1;
		    first_manifest.file_size_bytes = first_body.size();
		    manifest_files.push_back(std::move(first_manifest));
		    DistributedCopyFileInfo second_manifest;
		    second_manifest.staging_path = second_staged;
		    second_manifest.final_path = second_final;
		    second_manifest.row_count = 2;
		    second_manifest.file_size_bytes = second_body.size();
		    manifest_files.push_back(std::move(second_manifest));

		    auto commit_paths = BuildDistributedCopyFinalizeCommitPathsFromStagingRoot(fs, final_root, staging_root);
		    auto manifest_res =
		        WriteDistributedCopyFinalizeManifest(fs, commit_paths, final_root, staging_root, manifest_files);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }
		    fs.MoveFile(first_staged, first_final);

		    auto make_input_files = [&]() {
			    vector<DistributedCopyFileInfo> files;
			    DistributedCopyFileInfo first;
			    first.staging_path = first_staged;
			    first.row_count = 1;
			    first.file_size_bytes = first_body.size();
			    files.push_back(std::move(first));
			    DistributedCopyFileInfo second;
			    second.staging_path = second_staged;
			    second.row_count = 2;
			    second.file_size_bytes = second_body.size();
			    files.push_back(std::move(second));
			    return files;
		    };

		    const bool committed_before = fs.FileExists(commit_paths.committed_marker_path);
		    const bool first_final_before = fs.FileExists(first_final);
		    const bool second_final_before = fs.FileExists(second_final);
		    const bool first_staging_before = fs.FileExists(first_staged);
		    const bool second_staging_before = fs.FileExists(second_staged);
		    auto finalize_res = FinalizeCopyFiles(spec, staging_root, make_input_files(), context);
		    auto idempotent_res = FinalizeCopyFiles(spec, staging_root, make_input_files(), context);

		    std::vector<std::string> final_files;
		    bool final_root_exists = false;
		    try {
			    final_root_exists = fs.DirectoryExists(final_root);
			    if (final_root_exists) {
				    ListDistributedCopyFilesRecursive(fs, final_root, final_files);
			    }
		    } catch (...) {
			    final_root_exists = false;
		    }

		    py::dict out;
		    out["committed_before"] = committed_before;
		    out["first_final_before"] = first_final_before;
		    out["second_final_before"] = second_final_before;
		    out["first_staging_before"] = first_staging_before;
		    out["second_staging_before"] = second_staging_before;
		    out["finalize_error"] = finalize_res.is_err();
		    out["error"] = finalize_res.is_err() ? finalize_res.error().what() : "";
		    out["idempotent_error"] = idempotent_res.is_err();
		    out["idempotent_error_message"] = idempotent_res.is_err() ? idempotent_res.error().what() : "";
		    out["rows_copied"] = finalize_res.is_ok() ? finalize_res.value().rows_copied : 0;
		    out["idempotent_rows_copied"] = idempotent_res.is_ok() ? idempotent_res.value().rows_copied : 0;
		    out["manifest_exists"] = fs.FileExists(commit_paths.manifest_path);
		    out["committed_after"] = fs.FileExists(commit_paths.committed_marker_path);
		    out["staging_root_exists"] = fs.DirectoryExists(staging_root);
		    out["final_root_exists"] = final_root_exists;
		    out["final_file_count"] = final_files.size();
		    return out;
	    },
	    py::arg("local_dir"),
	    "Exercise COPY finalize replay after manifest was written and only a prefix of files was moved.");

	m.def(
	    "distributed_copy_direct_write_commit_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    const std::string run_id = "run-direct";
		    auto final_root = local_dir + "/copy_direct_final";
		    auto first_dir = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_0");
		    auto second_dir = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_1");
		    auto loser_dir = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_loser");
		    auto first_file = first_dir + "/part.parquet";
		    auto second_file = second_dir + "/part.parquet";
		    auto loser_file = loser_dir + "/part.parquet";
		    fs.CreateDirectoriesRecursive(first_dir);
		    fs.CreateDirectoriesRecursive(second_dir);
		    fs.CreateDirectoriesRecursive(loser_dir);

		    const std::string first_body = "first";
		    const std::string second_body = "second";
		    const std::string loser_body = "loser";
		    {
			    std::ofstream output(first_file, std::ios::out | std::ios::binary);
			    output << first_body;
		    }
		    {
			    std::ofstream output(second_file, std::ios::out | std::ios::binary);
			    output << second_body;
		    }
		    {
			    std::ofstream output(loser_file, std::ios::out | std::ios::binary);
			    output << loser_body;
		    }

		    DistributedCopySpec spec;
		    spec.file_path = final_root;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;

		    auto make_files = [&]() {
			    vector<DistributedCopyFileInfo> files;
			    DistributedCopyFileInfo first;
			    first.staging_path = first_file;
			    first.row_count = 1;
			    first.file_size_bytes = first_body.size();
			    files.push_back(std::move(first));
			    DistributedCopyFileInfo second;
			    second.staging_path = second_file;
			    second.row_count = 2;
			    second.file_size_bytes = second_body.size();
			    files.push_back(std::move(second));
			    return files;
		    };

		    auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, final_root, run_id);
		    auto lifecycle_res = WriteDistributedCopyDirectWriteLifecycle(fs, final_root, run_id);
		    if (lifecycle_res.is_err()) {
			    throw std::runtime_error(lifecycle_res.error().what());
		    }
		    auto first_res = FinalizeCopyFiles(spec, "", make_files(), context, run_id);
		    auto replay_loser_dir = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_replay_loser");
		    auto replay_loser_file = replay_loser_dir + "/part.parquet";
		    fs.CreateDirectoriesRecursive(replay_loser_dir);
		    {
			    std::ofstream output(replay_loser_file, std::ios::out | std::ios::binary);
			    output << "replay_loser";
		    }
		    auto second_res = FinalizeCopyFiles(spec, "", make_files(), context, run_id);

		    py::dict out;
		    out["first_finalize_error"] = first_res.is_err();
		    out["first_error"] = first_res.is_err() ? first_res.error().what() : "";
		    out["second_finalize_error"] = second_res.is_err();
		    out["second_error"] = second_res.is_err() ? second_res.error().what() : "";
		    out["first_rows_copied"] = first_res.is_ok() ? first_res.value().rows_copied : 0;
		    out["second_rows_copied"] = second_res.is_ok() ? second_res.value().rows_copied : 0;
		    out["first_final_path"] = first_res.is_ok() ? first_res.value().files[0].final_path : "";
		    out["second_final_path"] = first_res.is_ok() ? first_res.value().files[1].final_path : "";
		    out["first_output_run_id"] = first_res.is_ok() ? first_res.value().output_run_id : "";
		    out["first_output_manifest_path"] = first_res.is_ok() ? first_res.value().output_manifest_path : "";
		    out["first_output_committed_marker_path"] =
		        first_res.is_ok() ? first_res.value().output_committed_marker_path : "";
		    out["first_output_direct_write"] = first_res.is_ok() ? first_res.value().output_direct_write : false;
		    out["first_output_committed"] = first_res.is_ok() ? first_res.value().output_committed : false;
		    out["manifest_exists"] = fs.FileExists(commit_paths.manifest_path);
		    out["committed_exists"] = fs.FileExists(commit_paths.committed_marker_path);
		    out["direct_prefix_exists"] = fs.DirectoryExists(final_root + "/_vane_direct_write_" + run_id);
		    out["first_file_exists"] = fs.FileExists(first_file);
		    out["second_file_exists"] = fs.FileExists(second_file);
		    out["loser_file_exists"] = fs.FileExists(loser_file);
		    out["replay_loser_file_exists"] = fs.FileExists(replay_loser_file);
		    return out;
	    },
	    py::arg("local_dir"), "Exercise COPY direct-write run isolation plus selected-output commit manifest.");

	m.def(
	    "distributed_copy_direct_target_visible_commit_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    const std::string run_id = "run-visible";
		    const std::string other_run_id = "other-run";
		    auto final_root = local_dir + "/copy_direct_target_visible";
		    auto first_file = BuildCopyDirectTargetFilePath(final_root, run_id, "w_0", "part0.parquet");
		    auto second_file = BuildCopyDirectTargetFilePath(final_root, run_id, "w_1", "part1.parquet");
		    auto loser_file = BuildCopyDirectTargetFilePath(final_root, run_id, "w_loser", "part.parquet");
		    auto replay_loser_file =
		        BuildCopyDirectTargetFilePath(final_root, run_id, "w_replay_loser", "part.parquet");
		    auto other_run_file = BuildCopyDirectTargetFilePath(final_root, other_run_id, "w_other", "part.parquet");

		    auto write_file = [&](const std::string &path, const std::string &body) {
			    auto parent = StringUtil::GetFilePath(path);
			    if (!parent.empty() && !fs.DirectoryExists(parent)) {
				    fs.CreateDirectoriesRecursive(parent);
			    }
			    std::ofstream output(path, std::ios::out | std::ios::binary);
			    output << body;
			    output.close();
			    if (!output) {
				    throw std::runtime_error("failed to write test file: " + path);
			    }
		    };

		    const std::string first_body = "first";
		    const std::string second_body = "second";
		    write_file(first_file, first_body);
		    write_file(second_file, second_body);
		    write_file(loser_file, "loser");
		    write_file(other_run_file, "other");

		    DistributedCopySpec spec;
		    spec.file_path = final_root;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;

		    auto make_files = [&]() {
			    vector<DistributedCopyFileInfo> files;
			    DistributedCopyFileInfo first;
			    first.staging_path = first_file;
			    first.row_count = 1;
			    first.file_size_bytes = first_body.size();
			    files.push_back(std::move(first));
			    DistributedCopyFileInfo second;
			    second.staging_path = second_file;
			    second.row_count = 2;
			    second.file_size_bytes = second_body.size();
			    files.push_back(std::move(second));
			    return files;
		    };

		    auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, final_root, run_id);
		    auto lifecycle_res = WriteDistributedCopyDirectWriteLifecycle(fs, final_root, run_id);
		    if (lifecycle_res.is_err()) {
			    throw std::runtime_error(lifecycle_res.error().what());
		    }
		    auto first_res = FinalizeCopyFiles(spec, "", make_files(), context, run_id);
		    write_file(replay_loser_file, "replay_loser");
		    auto second_res = FinalizeCopyFiles(spec, "", make_files(), context, run_id);
		    auto read_res = ReadCommittedDistributedCopyDirectWriteResult(fs, final_root, run_id);

		    py::dict out;
		    out["first_finalize_error"] = first_res.is_err();
		    out["first_error"] = first_res.is_err() ? first_res.error().what() : "";
		    out["second_finalize_error"] = second_res.is_err();
		    out["second_error"] = second_res.is_err() ? second_res.error().what() : "";
		    out["read_committed_error"] = read_res.is_err();
		    out["read_committed_error_message"] = read_res.is_err() ? read_res.error().what() : "";
		    out["read_committed_rows_copied"] = read_res.is_ok() ? read_res.value().rows_copied : 0;
		    out["first_rows_copied"] = first_res.is_ok() ? first_res.value().rows_copied : 0;
		    out["second_rows_copied"] = second_res.is_ok() ? second_res.value().rows_copied : 0;
		    out["first_final_path"] = first_res.is_ok() ? first_res.value().files[0].final_path : "";
		    out["first_output_run_id"] = first_res.is_ok() ? first_res.value().output_run_id : "";
		    out["first_output_direct_write"] = first_res.is_ok() ? first_res.value().output_direct_write : false;
		    out["first_output_committed"] = first_res.is_ok() ? first_res.value().output_committed : false;
		    out["manifest_exists"] = fs.FileExists(commit_paths.manifest_path);
		    out["committed_exists"] = fs.FileExists(commit_paths.committed_marker_path);
		    out["direct_prefix_exists"] = fs.DirectoryExists(final_root + "/_vane_direct_write_" + run_id);
		    out["first_file_exists"] = fs.FileExists(first_file);
		    out["second_file_exists"] = fs.FileExists(second_file);
		    out["loser_file_exists"] = fs.FileExists(loser_file);
		    out["replay_loser_file_exists"] = fs.FileExists(replay_loser_file);
		    out["other_run_file_exists"] = fs.FileExists(other_run_file);
		    return out;
	    },
	    py::arg("local_dir"), "Exercise visible direct-target COPY commit cleanup.");

	m.def(
	    "distributed_copy_direct_target_remote_path_for_test",
	    [](const std::string &base_path, const std::string &run_id, const std::string &worker_dir_name,
	       const std::string &file_name) {
		    using namespace duckdb::distributed;

		    py::dict out;
		    out["direct_target_file"] = BuildCopyDirectTargetFilePath(base_path, run_id, worker_dir_name, file_name);
		    out["legacy_task_directory"] = BuildCopyDirectWriteTaskDirectory(base_path, run_id, worker_dir_name);
		    out["filename_pattern"] = BuildCopyDirectTargetFilenamePattern(run_id, worker_dir_name);
		    return out;
	    },
	    py::arg("base_path"), py::arg("run_id"), py::arg("worker_dir_name"), py::arg("file_name"),
	    "Exercise visible direct-target COPY path helpers.");

	m.def(
	    "distributed_copy_sink_mode_for_test",
	    [](const std::string &output_path, bool use_tmp_file) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    py::dict out;
		    DistributedCopySpec spec;
		    spec.file_path = output_path;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.use_tmp_file = use_tmp_file;
		    try {
			    auto sink = std::make_shared<CopySinkNode>(1, PipelineNodeRef(), std::move(spec));
			    out["construct_error"] = false;
			    out["error"] = "";
			    out["staging_root_base"] = sink->staging_root_base();
			    out["staging_run_id"] = sink->staging_run_id();
			    out["uses_direct_write"] = sink->staging_root_base().empty();
			    out["uses_visible_direct_target"] = sink->staging_root_base().empty();
		    } catch (const std::exception &ex) {
			    out["construct_error"] = true;
			    out["error"] = ex.what();
			    out["staging_root_base"] = "";
			    out["staging_run_id"] = "";
			    out["uses_direct_write"] = false;
			    out["uses_visible_direct_target"] = false;
		    }
		    return out;
	    },
	    py::arg("output_path"), py::arg("use_tmp_file") = false,
	    "Return distributed COPY sink mode for an output path.");

	m.def(
	    "distributed_copy_direct_write_local_invisible_file_commit_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    const std::string run_id = "run-local-invisible";
		    auto final_root = local_dir + "/copy_direct_invisible";
		    auto invisible_file = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_remote") + "/part.parquet";

		    DistributedCopySpec spec;
		    spec.file_path = final_root;
		    spec.file_extension = "parquet";
		    spec.per_thread_output = true;
		    spec.overwrite_mode = CopyOverwriteMode::COPY_ERROR_ON_CONFLICT;

		    vector<DistributedCopyFileInfo> files;
		    DistributedCopyFileInfo info;
		    info.staging_path = invisible_file;
		    info.row_count = 4;
		    info.file_size_bytes = 123;
		    files.push_back(std::move(info));

		    auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, final_root, run_id);
		    auto lifecycle_res = WriteDistributedCopyDirectWriteLifecycle(fs, final_root, run_id);
		    if (lifecycle_res.is_err()) {
			    throw std::runtime_error(lifecycle_res.error().what());
		    }
		    auto finalize_res = FinalizeCopyFiles(spec, "", std::move(files), context, run_id);

		    py::dict out;
		    out["finalize_error"] = finalize_res.is_err();
		    out["error"] = finalize_res.is_err() ? finalize_res.error().what() : "";
		    out["rows_copied"] = finalize_res.is_ok() ? finalize_res.value().rows_copied : 0;
		    out["output_direct_write"] = finalize_res.is_ok() ? finalize_res.value().output_direct_write : false;
		    out["output_committed"] = finalize_res.is_ok() ? finalize_res.value().output_committed : false;
		    out["manifest_exists"] = fs.FileExists(commit_paths.manifest_path);
		    out["committed_exists"] = fs.FileExists(commit_paths.committed_marker_path);
		    out["invisible_file_exists"] = DistributedCopyFileExistsNoThrow(fs, invisible_file);
		    return out;
	    },
	    py::arg("local_dir"),
	    "Exercise local direct-write commit when coordinator cannot see a worker node-local output file.");

	m.def(
	    "distributed_copy_direct_write_committed_reader_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    const std::string run_id = "run-reader";
		    auto final_root = local_dir + "/copy_direct_reader";
		    auto selected_dir = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_selected");
		    auto loser_dir = BuildCopyDirectWriteTaskDirectory(final_root, run_id, "w_loser");
		    auto selected_file = selected_dir + "/part.parquet";
		    auto loser_file = loser_dir + "/part.parquet";

		    auto write_file = [&](const std::string &path, const std::string &body) {
			    auto parent = StringUtil::GetFilePath(path);
			    if (!parent.empty() && !fs.DirectoryExists(parent)) {
				    fs.CreateDirectoriesRecursive(parent);
			    }
			    std::ofstream output(path, std::ios::out | std::ios::binary);
			    output << body;
			    output.close();
			    if (!output) {
				    throw std::runtime_error("failed to write test file: " + path);
			    }
		    };

		    const std::string selected_body = "selected";
		    const std::string loser_body = "loser";
		    write_file(selected_file, selected_body);
		    write_file(loser_file, loser_body);

		    DistributedCopyFileInfo selected_info;
		    selected_info.staging_path = selected_file;
		    selected_info.final_path = selected_file;
		    selected_info.row_count = 7;
		    selected_info.file_size_bytes = selected_body.size();
		    vector<DistributedCopyFileInfo> selected_files;
		    selected_files.push_back(std::move(selected_info));

		    auto commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, final_root, run_id);
		    auto lifecycle_res = WriteDistributedCopyDirectWriteLifecycle(fs, final_root, run_id);
		    if (lifecycle_res.is_err()) {
			    throw std::runtime_error(lifecycle_res.error().what());
		    }
		    auto manifest_res =
		        WriteDistributedCopyFinalizeManifest(fs, commit_paths, final_root, "direct:" + run_id, selected_files);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }

		    auto uncommitted_read = ReadCommittedDistributedCopyDirectWriteResult(fs, final_root, run_id);
		    auto marker_res = WriteDistributedCopyFinalizeCommittedMarker(fs, commit_paths);
		    if (marker_res.is_err()) {
			    throw std::runtime_error(marker_res.error().what());
		    }
		    auto committed_read = ReadCommittedDistributedCopyDirectWriteResult(fs, final_root, run_id);

		    bool committed_contains_loser = false;
		    std::string committed_file_path;
		    idx_t committed_rows = 0;
		    idx_t committed_file_count = 0;
		    if (committed_read.is_ok()) {
			    const auto &result = committed_read.value();
			    committed_rows = result.rows_copied;
			    committed_file_count = result.files.size();
			    for (const auto &info : result.files) {
				    if (info.final_path == loser_file) {
					    committed_contains_loser = true;
				    }
			    }
			    if (!result.files.empty()) {
				    committed_file_path = result.files[0].final_path;
			    }
		    }

		    py::dict out;
		    out["base_path"] = final_root;
		    out["run_id"] = run_id;
		    out["manifest_exists"] = fs.FileExists(commit_paths.manifest_path);
		    out["marker_exists"] = fs.FileExists(commit_paths.committed_marker_path);
		    out["selected_file_exists"] = fs.FileExists(selected_file);
		    out["loser_file_exists"] = fs.FileExists(loser_file);
		    out["uncommitted_error"] = uncommitted_read.is_err();
		    out["uncommitted_error_message"] = uncommitted_read.is_err() ? uncommitted_read.error().what() : "";
		    out["committed_error"] = committed_read.is_err();
		    out["committed_error_message"] = committed_read.is_err() ? committed_read.error().what() : "";
		    out["committed_rows"] = committed_rows;
		    out["committed_file_count"] = committed_file_count;
		    out["committed_file_path"] = committed_file_path;
		    out["committed_contains_loser"] = committed_contains_loser;
		    return out;
	    },
	    py::arg("local_dir"),
	    "Exercise COPY direct-write committed-reader visibility through committed manifest only.");

	m.def(
	    "distributed_copy_direct_write_uncommitted_stale_cleanup_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    auto final_root = local_dir + "/copy_direct_stale_cleanup";
		    const std::string stale_run_id = "run-stale";
		    const std::string committed_run_id = "run-committed";

		    auto write_file = [&](const std::string &path, const std::string &body) {
			    auto parent = StringUtil::GetFilePath(path);
			    if (!parent.empty() && !fs.DirectoryExists(parent)) {
				    fs.CreateDirectoriesRecursive(parent);
			    }
			    std::ofstream output(path, std::ios::out | std::ios::binary);
			    output << body;
			    output.close();
			    if (!output) {
				    throw std::runtime_error("failed to write test file: " + path);
			    }
		    };

		    auto make_file_info = [](const std::string &path, idx_t rows, idx_t bytes) {
			    DistributedCopyFileInfo info;
			    info.staging_path = path;
			    info.final_path = path;
			    info.row_count = rows;
			    info.file_size_bytes = bytes;
			    return info;
		    };

		    auto stale_dir = BuildCopyDirectWriteTaskDirectory(final_root, stale_run_id, "w_stale");
		    auto stale_file = stale_dir + "/part.parquet";
		    const std::string stale_body = "stale";
		    write_file(stale_file, stale_body);

		    auto stale_commit_paths = BuildDistributedCopyFinalizeCommitPaths(fs, final_root, stale_run_id);
		    auto stale_lifecycle_res = WriteDistributedCopyDirectWriteLifecycle(fs, final_root, stale_run_id);
		    if (stale_lifecycle_res.is_err()) {
			    throw std::runtime_error(stale_lifecycle_res.error().what());
		    }
		    vector<DistributedCopyFileInfo> stale_files;
		    stale_files.push_back(make_file_info(stale_file, 1, stale_body.size()));
		    auto stale_manifest_res = WriteDistributedCopyFinalizeManifest(fs, stale_commit_paths, final_root,
		                                                                   "direct:" + stale_run_id, stale_files);
		    if (stale_manifest_res.is_err()) {
			    throw std::runtime_error(stale_manifest_res.error().what());
		    }

		    auto committed_dir = BuildCopyDirectWriteTaskDirectory(final_root, committed_run_id, "w_committed");
		    auto committed_file = committed_dir + "/part.parquet";
		    const std::string committed_body = "committed";
		    write_file(committed_file, committed_body);

		    auto committed_paths = BuildDistributedCopyFinalizeCommitPaths(fs, final_root, committed_run_id);
		    auto committed_lifecycle_res = WriteDistributedCopyDirectWriteLifecycle(fs, final_root, committed_run_id);
		    if (committed_lifecycle_res.is_err()) {
			    throw std::runtime_error(committed_lifecycle_res.error().what());
		    }
		    vector<DistributedCopyFileInfo> committed_files;
		    committed_files.push_back(make_file_info(committed_file, 1, committed_body.size()));
		    auto committed_manifest_res = WriteDistributedCopyFinalizeManifest(
		        fs, committed_paths, final_root, "direct:" + committed_run_id, committed_files);
		    if (committed_manifest_res.is_err()) {
			    throw std::runtime_error(committed_manifest_res.error().what());
		    }
		    auto committed_marker_res = WriteDistributedCopyFinalizeCommittedMarker(fs, committed_paths);
		    if (committed_marker_res.is_err()) {
			    throw std::runtime_error(committed_marker_res.error().what());
		    }

		    auto stale_cleanup_res = CleanupDistributedCopyUncommittedDirectWriteRun(fs, final_root, stale_run_id);
		    if (stale_cleanup_res.is_err()) {
			    throw std::runtime_error(stale_cleanup_res.error().what());
		    }
		    auto committed_cleanup_res =
		        CleanupDistributedCopyUncommittedDirectWriteRun(fs, final_root, committed_run_id);
		    if (committed_cleanup_res.is_err()) {
			    throw std::runtime_error(committed_cleanup_res.error().what());
		    }
		    auto stale_cleanup = std::move(stale_cleanup_res.value());
		    auto committed_cleanup = std::move(committed_cleanup_res.value());

		    py::dict out;
		    out["stale_skipped_committed"] = stale_cleanup.skipped_committed;
		    out["stale_data_run_dir_existed"] = stale_cleanup.data_run_dir_existed;
		    out["stale_data_run_dir_removed"] = stale_cleanup.data_run_dir_removed;
		    out["stale_commit_dir_existed"] = stale_cleanup.commit_dir_existed;
		    out["stale_commit_dir_removed"] = stale_cleanup.commit_dir_removed;
		    out["stale_run_dir_exists"] = DistributedCopyDirectoryExistsNoThrow(fs, stale_dir);
		    out["stale_file_exists"] = DistributedCopyFileExistsNoThrow(fs, stale_file);
		    out["stale_manifest_exists"] = DistributedCopyFileExistsNoThrow(fs, stale_commit_paths.manifest_path);
		    out["stale_commit_dir_exists"] = DistributedCopyDirectoryExistsNoThrow(fs, stale_commit_paths.commit_dir);
		    out["committed_skipped_committed"] = committed_cleanup.skipped_committed;
		    out["committed_data_run_dir_removed"] = committed_cleanup.data_run_dir_removed;
		    out["committed_commit_dir_removed"] = committed_cleanup.commit_dir_removed;
		    out["committed_run_dir_exists"] = DistributedCopyDirectoryExistsNoThrow(fs, committed_dir);
		    out["committed_file_exists"] = DistributedCopyFileExistsNoThrow(fs, committed_file);
		    out["committed_manifest_exists"] = DistributedCopyFileExistsNoThrow(fs, committed_paths.manifest_path);
		    out["committed_marker_exists"] =
		        DistributedCopyFileExistsNoThrow(fs, committed_paths.committed_marker_path);
		    out["committed_commit_dir_exists"] = DistributedCopyDirectoryExistsNoThrow(fs, committed_paths.commit_dir);
		    return out;
	    },
	    py::arg("local_dir"), "Exercise explicit cleanup of uncommitted stale COPY direct-write runs.");

	m.def(
	    "split_exchange_source_task_by_partition",
	    [](py::bytes bytes_obj) {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto desc = ExchangeSourceTaskDescriptor::DeserializeFromBytes(raw);
		    idx_t partition_count = desc.source_partition_count;
		    if (partition_count == 0 && !desc.partition_indices.empty()) {
			    for (auto partition_id : desc.partition_indices) {
				    partition_count = std::max(partition_count, partition_id + 1);
			    }
		    }

		    py::list out;
		    for (auto partition_id : desc.partition_indices) {
			    ExchangeSourceTaskDescriptor single;
			    single.partition_indices = vector<idx_t> {partition_id};
			    single.source_partition_count = partition_count;
			    single.source_task_count = desc.source_task_count;
			    single.replicated = desc.replicated;
			    single.mark_join_build_summary = desc.mark_join_build_summary;
			    for (const auto &handle : desc.source_handles) {
				    if (handle.partition_id == partition_id) {
					    single.source_handles.push_back(handle);
				    }
			    }
			    auto bytes = single.SerializeToBytes();
			    out.append(py::make_tuple(partition_id, py::bytes(bytes), partition_count, desc.source_task_count,
			                              desc.replicated));
		    }
		    return out;
	    },
	    py::arg("bytes"), "Split a raw ExchangeSourceTaskDescriptor into one descriptor per source partition.");

	// Stop the poller before CPython enters late finalization. The callback is
	// registered after the other Ray cleanup hook, so it runs first at exit.
	py::module_::import("atexit").attr("register")(
	    py::cpp_function([]() { duckdb::distributed::python::ray::ShutdownRayTaskResultPoller(); }));
}

} // namespace duckdb
