// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "ray_binding_internal.hpp"

namespace duckdb {
namespace ray_bindings {
namespace py = ::duckdb::py;

void RegisterRayRunnerBindings(py::module_ &m) {
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
	    .def(
	        "_wait_fte_query_diagnostic_for_test",
	        [](PyPhysicalPlanWrapperRunner &self, const string &query_id, double timeout_s, bool exhaust_input) {
		        py::gil_scoped_release release;
		        if (exhaust_input) {
			        self.worker_manager_->task_input_stream_exhausted_for_query(query_id, {}).value();
		        }
		        auto result = self.worker_manager_->wait_fte_query(query_id, timeout_s);
		        return result.is_err() ? result.error().Diagnostics().AppendTo() : string();
	        },
	        py::arg("query_id"), py::arg("timeout_s") = 1.0, py::arg("exhaust_input") = false)
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
	           py::object effective_session_config_obj, py::object native_execution_started_obj) {
		        if (!native_execution_started_obj.is_none() && !PyCallable_Check(native_execution_started_obj.ptr())) {
			        throw py::type_error("native_execution_started must be callable");
		        }
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
				        if (d.contains("source_task_order")) {
					        exchange_sink_instance_task.sink_instance.source_task_order =
					            py::int_(d["source_task_order"]).cast<idx_t>();
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
					        deferred_exec_plan.memory_source_refs_ = plan.memory_source_refs_;
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
				            dynamic_filter_domains_obj, native_progress_callback_obj, runtime_context_obj,
				            native_execution_started_obj);
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
	        py::arg("effective_session_config") = py::none(), py::arg("native_execution_started") = py::none(),
	        "Execute physical plan using DuckDB's native Executor");
}
} // namespace ray_bindings
} // namespace duckdb
