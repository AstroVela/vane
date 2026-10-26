// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "ray_binding_internal.hpp"

namespace duckdb {
namespace ray_bindings {
namespace py = ::duckdb::py;

void RegisterRayPlansBindings(py::module_ &m) {
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
	    .def("_release_vane_fs_query_pins",
	         [](PyPhysicalPlanWrapper &plan) { plan.vane_fs_query_pins_ = CreateVaneFSQueryPins(); })
	    .def("has_root", &PyPhysicalPlanWrapper::has_root)
	    .def("_memory_source_ref_count_for_test",
	         [](const PyPhysicalPlanWrapper &p) { return PythonMemorySourceObjectRefCount(p.memory_source_refs_); })
	    .def("clone", &PyPhysicalPlanWrapper::clone, py::arg("conn") = py::none())
	    .def("num_partitions", &PyPhysicalPlanWrapper::num_partitions)
	    .def("repr_ascii", &PyPhysicalPlanWrapper::repr_ascii)
	    .def("repr_mermaid", &PyPhysicalPlanWrapper::repr_mermaid)
	    .def("_datasource_scan_cardinalities_for_test", &PyPhysicalPlanWrapper::datasource_scan_cardinalities_for_test)
	    .def("scan_split_batch_map", &PyPhysicalPlanWrapper::scan_split_batch_map)
	    .def("collect_query_resource_graph_metadata", &PyPhysicalPlanWrapper::collect_query_resource_graph_metadata,
	         py::arg("conn") = py::none())
	    .def("collect_resource_graph_metadata", &PyPhysicalPlanWrapper::collect_resource_graph_metadata,
	         py::arg("conn") = py::none())
	    .def("collect_local_resource_graph_metadata", &PyPhysicalPlanWrapper::collect_local_resource_graph_metadata,
	         py::arg("conn") = py::none())
	    .def("collect_udf_nodes", &PyPhysicalPlanWrapper::collect_udf_nodes, py::arg("conn") = py::none())
	    .def("collect_vllm_nodes", &PyPhysicalPlanWrapper::collect_vllm_nodes, py::arg("conn") = py::none())
	    .def("set_udf_actor_handles", &PyPhysicalPlanWrapper::set_udf_actor_handles, py::arg("handles_map"),
	         py::arg("conn") = py::none())
	    .def("set_local_udf_executor_options", &PyPhysicalPlanWrapper::set_local_udf_executor_options,
	         py::arg("options_map"), py::arg("conn") = py::none())
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
				                              p.udf_registrations_, p.udf_actor_handles_, p.connection_snapshot_,
				                              p.memory_source_refs_);
			        }
			        return py::make_tuple(false, py::bytes(""), p.query_id_, p.resource_query_id_, p.udf_registrations_,
			                              p.udf_actor_handles_, p.connection_snapshot_, p.memory_source_refs_);
		        }

		        auto serialized_root = p.serialize_root_for_clone();
		        return py::make_tuple(true, py::bytes(serialized_root), p.query_id_, p.resource_query_id_,
		                              p.udf_registrations_, p.udf_actor_handles_, p.connection_snapshot_,
		                              p.memory_source_refs_);
	        },
	        // __setstate__: store deferred bytes for later deserialization
	        [](py::tuple t) {
		        if (t.size() != 8) {
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
		        result.memory_source_refs_ = t[7];
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
	    .def("_release_vane_fs_query_pins",
	         [](PyLogicalPlan &plan) { plan.vane_fs_query_pins_ = CreateVaneFSQueryPins(); })
	    .def("_memory_source_ref_count_for_test",
	         [](const PyLogicalPlan &p) { return PythonMemorySourceObjectRefCount(p.memory_source_refs_); })
	    .def("to_physical_plan", &PyLogicalPlan::to_physical_plan, py::arg("conn") = py::none(),
	         py::arg("effective_session_config") = py::none())
	    .def(py::pickle(
	        [](const PyLogicalPlan &p) {
		        if (p.serialized_logical_plan_.empty()) {
			        throw duckdb::InternalException("PyLogicalPlan missing serialized logical plan");
		        }
		        (void)DecodeLogicalPlanEnvelope(p.serialized_logical_plan_);
		        return py::make_tuple(p.query_id_, py::bytes(p.serialized_logical_plan_), p.udf_registrations_,
		                              p.connection_snapshot_, p.memory_source_refs_);
	        },
	        [](py::tuple t) {
		        if (t.size() != 5)
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
		        plan.memory_source_refs_ = t[4];
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
}
} // namespace ray_bindings
} // namespace duckdb
