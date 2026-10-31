// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "ray_binding_internal.hpp"

namespace duckdb {
namespace ray_bindings {
namespace py = ::duckdb::py;

void RegisterRayCoreBindings(py::module_ &m) {
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
	// Windows terminates other threads before DLL static destructors run. Stop
	// Flight while its gRPC threads can still complete shutdown and join.
	py::module_::import("atexit").attr("register")(m.attr("shutdown_local_flight_service"));

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

	m.def("_native_error_diagnostic_for_test", [](const std::string &message) {
		const std::runtime_error error(message);
		return ::vane::CaptureError(error).WithContext("native failure").AppendTo();
	});

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
	           int64_t throw_after, bool preserve_task_order) {
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
			        return self.wait_fte_query_streaming(query_id, timeout_s, std::move(on_output),
			                                             preserve_task_order);
		        }();
		        if (res.is_err()) {
			        throw duckdb::InternalException(res.error().what());
		        }
		        return output_count;
	        },
	        py::arg("query_id"), py::arg("timeout_s") = 0.0, py::arg("fail_after") = -1, py::arg("throw_after") = -1,
	        py::arg("preserve_task_order") = true)
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
	       py::object udf_registrations_obj, py::object udf_actor_handles_obj, py::object memory_source_refs_obj,
	       py::object connection_snapshot_obj) {
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
		    result.memory_source_refs_ = memory_source_refs_obj;
		    result.connection_snapshot_ = connection_snapshot_obj;
		    auto coordinator_connection = LookupQueryCoordinatorConnection(resource_query_id);
		    if (!coordinator_connection.is_none()) {
			    auto &coordinator_wrapper = ExtractPyConnectionWrapper(coordinator_connection);
			    if (coordinator_wrapper.con.ConnectionIsClosed()) {
				    throw py::value_error("DistributedPhysicalPlan coordinator connection is closed");
			    }
			    result.worker_connection_ = py::cast(coordinator_wrapper.Cursor());
			    result.client_context_ = coordinator_wrapper.con.GetConnection().context;
		    }
		    (void)VaneSessionIdFromSnapshot(result.connection_snapshot_);
		    (void)VaneSessionConfigFromSnapshot(result.connection_snapshot_);
		    return result;
	    },
	    py::arg("capsule"), py::arg("query_id") = py::none(), py::arg("resource_query_id") = py::none(),
	    py::arg("udf_registrations") = py::none(), py::arg("udf_actor_handles") = py::none(),
	    py::arg("memory_source_refs") = py::none(), py::arg("connection_snapshot") = py::none(),
	    "Internal helper to create PyPhysicalPlanWrapper from C++ capsule");

	m.def(
	    "_lookup_query_udf_registrations", [](const string &query_id) { return LookupQueryUDFRegistrations(query_id); },
	    py::arg("query_id"));

	m.def(
	    "_lookup_query_udf_actor_handles", [](const string &query_id) { return LookupQueryUDFActorHandles(query_id); },
	    py::arg("query_id"));

	m.def(
	    "_lookup_query_memory_source_refs",
	    [](const string &query_id) { return LookupQueryMemorySourceRefs(query_id); }, py::arg("query_id"));

	m.def("_lookup_memory_source_ref", &LookupMemorySourceRef, py::arg("source_id"), py::arg("partition_index"));

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
		    auto connection = CreateConnectionFromBootstrapSnapshot(
		        LookupBootstrapSnapshot(snapshot), RunnerTypeFromSnapshot(snapshot), false, true, true);
		    PrepareConnectionSnapshotExtensions(connection, snapshot, LookupQueryVaneFSPins(query_id));
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
		    if (snapshot.cast<py::dict>().contains("vane_fs_workspaces")) {
			    ApplyVaneFSWorkspaces(ExtractPyConnectionWrapper(connection),
			                          snapshot.cast<py::dict>()["vane_fs_workspaces"], LookupQueryVaneFSPins(query_id));
		    }
	    },
	    py::arg("connection"), py::arg("query_id"));

	m.def(
	    "_register_query_python_replay_state",
	    [](const string &query_id, PyPhysicalPlanWrapper &plan) {
		    if (query_id.empty()) {
			    throw duckdb::InternalException("Query Python replay registration requires a non-empty query_id");
		    }
		    // Worker resource settings are owned by the Ray actor allocation, and
		    // extension locations are owned by the worker installation. Remove both
		    // from every worker-side replay before the snapshot is registered or used
		    // to resolve a database.
		    plan.connection_snapshot_ = PrepareWorkerConnectionSnapshot(plan.connection_snapshot_);
		    // The resource query owns this lifecycle. A retried FTE task can carry a
		    // physical plan created under a different source plan identifier.
		    return RegisterQueryPythonReplayState(query_id, plan.udf_registrations_, plan.udf_actor_handles_,
		                                          plan.memory_source_refs_, plan.connection_snapshot_,
		                                          plan.worker_connection_, plan.vane_fs_query_pins_);
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
}
} // namespace ray_bindings
} // namespace duckdb
