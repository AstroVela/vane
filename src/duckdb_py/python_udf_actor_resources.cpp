// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "duckdb_python/python_udf_actor_resources.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb_python/pybind11/gil_wrapper.hpp"
#include "duckdb_python/python_objects.hpp"

#include <pybind11/pybind11.h>
#include <sstream>
#include <unordered_set>

namespace duckdb {

namespace {

namespace py = pybind11;

static shared_ptr<void> WrapPyObjectForUDFActorHandles(const py::object &obj) {
	if (obj.is_none()) {
		return nullptr;
	}
	auto *boxed = new py::object(py::reinterpret_borrow<py::object>(obj));
	return shared_ptr<void>(boxed, [](void *ptr) {
		if (!ptr) {
			return;
		}
		auto *boxed_obj = static_cast<py::object *>(ptr);
		if (!Py_IsInitialized() || PythonIsFinalizing()) {
			boxed_obj->release();
			delete boxed_obj;
			return;
		}
		PythonGILWrapper gil;
		delete boxed_obj;
	});
}

static const UDFFunctionData *TryGetUDFBindData(const FunctionData *bind_data) {
	return dynamic_cast<const UDFFunctionData *>(bind_data);
}

static UDFFunctionData *TryGetMutableUDFBindData(PhysicalOperator &op) {
	const UDFFunctionData *bind_data = nullptr;
	if (op.type == PhysicalOperatorType::INOUT_FUNCTION) {
		bind_data = TryGetUDFBindData(op.Cast<PhysicalTableInOutFunction>().GetBindData());
	} else if (op.type == PhysicalOperatorType::STREAMING_UDF) {
		bind_data = TryGetUDFBindData(op.Cast<PhysicalStreamingUDF>().GetBindData());
	}
	return const_cast<UDFFunctionData *>(bind_data);
}

static void CollectMutableUDFBindDataRecursive(PhysicalOperator &op, vector<UDFFunctionData *> &out) {
	if (auto *bind_data = TryGetMutableUDFBindData(op)) {
		out.push_back(bind_data);
	}
	for (auto &child : op.children) {
		CollectMutableUDFBindDataRecursive(child.get(), out);
	}
}

static bool PayloadStringField(const Value &payload, const string &name, string &result) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return false;
	}
	auto &children = StructValue::GetChildren(payload);
	auto &payload_type = payload.type();
	auto child_count = StructType::GetChildCount(payload_type);
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload_type, i) != name) {
			continue;
		}
		if (children[i].IsNull() || children[i].type().id() != LogicalTypeId::VARCHAR) {
			return false;
		}
		result = children[i].GetValue<string>();
		return true;
	}
	return false;
}

static py::dict BuildUDFNode(idx_t node_id, UDFFunctionData &bind_data, ClientContext &context) {
	auto payload_obj =
	    PythonObject::FromValue(bind_data.payload, bind_data.payload.type(), context.GetClientProperties());
	py::dict meta;
	meta[py::str("node_id")] = py::int_(node_id);
	meta[py::str("payload")] = payload_obj;
	if (py::isinstance<py::dict>(payload_obj)) {
		auto payload_dict = py::reinterpret_borrow<py::dict>(payload_obj);
		auto get = payload_dict.attr("get");
		auto execution_backend = get("execution_backend");
		auto actor_pool_size = get("actor_pool_size");
		auto cpus = get("cpus");
		auto gpus = get("gpus");
		meta[py::str("execution_backend")] = execution_backend;
		meta[py::str("actor_pool_size")] = actor_pool_size;
		meta[py::str("cpus")] = cpus.is_none() ? py::float_(1.0) : cpus;
		meta[py::str("gpus")] = gpus.is_none() ? py::float_(0.0) : gpus;
	}
	return meta;
}

static void AppendCreatedResources(vector<py::object> &resources, const py::object &created_obj) {
	if (created_obj.is_none()) {
		return;
	}
	for (auto item : py::reinterpret_borrow<py::iterable>(created_obj)) {
		resources.push_back(py::reinterpret_borrow<py::object>(item));
	}
}

static void ApplyHandlesMap(const vector<UDFFunctionData *> &bind_nodes, const py::object &handles_obj) {
	if (handles_obj.is_none()) {
		return;
	}
	if (!py::isinstance<py::dict>(handles_obj)) {
		throw InvalidInputException("UDF actor resource helper returned a non-dict handles map");
	}
	auto handles_map = py::reinterpret_borrow<py::dict>(handles_obj);
	for (auto item : handles_map) {
		auto key = py::reinterpret_borrow<py::object>(item.first);
		auto value = py::reinterpret_borrow<py::object>(item.second);
		auto node_id = static_cast<idx_t>(py::cast<int64_t>(py::int_(key)));
		if (node_id >= bind_nodes.size()) {
			throw InvalidInputException("UDF actor resource helper returned unknown node_id %llu",
			                            static_cast<unsigned long long>(node_id));
		}
		bind_nodes[node_id]->actor_handles = WrapPyObjectForUDFActorHandles(value);
	}
}

static string DirectPlanIdentity(PreparedStatementData &prepared) {
	std::ostringstream ss;
	ss << "direct-" << static_cast<const void *>(&prepared);
	return ss.str();
}

} // namespace

class PythonUDFActorResourceState : public ClientContextState {
public:
	void BeginScope() {
		scope_depth++;
	}

	void EndScope() {
		if (scope_depth > 0) {
			scope_depth--;
		}
	}

	bool CanRequestRebind() override {
		return Enabled();
	}

	RebindQueryInfo OnFinalizePrepare(ClientContext &context, PreparedStatementData &prepared,
	                                  PreparedStatementMode) override {
		if (!Enabled()) {
			return RebindQueryInfo::DO_NOT_REBIND;
		}
		PrepareOnce(context, prepared);
		return RebindQueryInfo::DO_NOT_REBIND;
	}

	RebindQueryInfo OnExecutePrepared(ClientContext &context, PreparedStatementCallbackInfo &info,
	                                  RebindQueryInfo current_rebind) override {
		if (!Enabled() || current_rebind == RebindQueryInfo::ATTEMPT_TO_REBIND) {
			return RebindQueryInfo::DO_NOT_REBIND;
		}
		PrepareOnce(context, info.prepared_statement);
		return RebindQueryInfo::DO_NOT_REBIND;
	}

	void QueryEnd(ClientContext &, optional_ptr<ErrorData>) override {
		prepared_statements.clear();
		if (resources.empty()) {
			return;
		}
		if (!Py_IsInitialized() || PythonIsFinalizing()) {
			ReleaseResourcesWithoutPython();
			return;
		}
		PythonGILWrapper gil;
		ShutdownResources();
	}

	~PythonUDFActorResourceState() override {
		if (resources.empty()) {
			return;
		}
		if (!Py_IsInitialized() || PythonIsFinalizing()) {
			ReleaseResourcesWithoutPython();
			return;
		}
		PythonGILWrapper gil;
		ShutdownResources();
	}

private:
	bool Enabled() const {
		return scope_depth > 0;
	}

	void PrepareOnce(ClientContext &context, PreparedStatementData &prepared) {
		if (prepared_statements.find(&prepared) != prepared_statements.end()) {
			return;
		}
		Prepare(context, prepared);
		prepared_statements.insert(&prepared);
	}

	void Prepare(ClientContext &context, PreparedStatementData &prepared) {
		if (!prepared.physical_plan || !prepared.physical_plan->HasRoot()) {
			return;
		}

		vector<UDFFunctionData *> bind_nodes;
		CollectMutableUDFBindDataRecursive(prepared.physical_plan->Root(), bind_nodes);
		if (bind_nodes.empty()) {
			return;
		}

		PythonGILWrapper gil;
		pybind11::list subprocess_nodes;
		for (idx_t node_id = 0; node_id < bind_nodes.size(); node_id++) {
			auto *bind_data = bind_nodes[node_id];
			if (!bind_data || bind_data->actor_handles) {
				continue;
			}
			string backend;
			if (!PayloadStringField(bind_data->payload, "execution_backend", backend)) {
				continue;
			}
			if (backend == "subprocess_actor") {
				subprocess_nodes.append(BuildUDFNode(node_id, *bind_data, context));
			} else if (backend == "ray_actor") {
				throw InvalidInputException("ray_actor UDF execution requires driver-precreated actor handles from a "
				                            "registered query allocation; "
				                            "execute the relation through RayRunner");
			}
		}
		if (pybind11::len(subprocess_nodes) == 0) {
			return;
		}

		try {
			if (pybind11::len(subprocess_nodes) > 0) {
				auto subprocess_module = pybind11::module_::import("vane.execution.udf_subprocess");
				auto result = pybind11::reinterpret_borrow<pybind11::tuple>(
				    subprocess_module.attr("ensure_local_subprocess_actor_pools_for_nodes")(
				        subprocess_nodes, pybind11::arg("plan_identity") = DirectPlanIdentity(prepared)));
				auto created = result[0];
				auto handles_map = result[1];
				AppendCreatedResources(resources, created);
				ApplyHandlesMap(bind_nodes, handles_map);
			}
		} catch (...) {
			ShutdownResources();
			throw;
		}
	}

	void ShutdownResources() {
		if (resources.empty()) {
			return;
		}
		for (auto it = resources.rbegin(); it != resources.rend(); ++it) {
			try {
				if (pybind11::hasattr(*it, "shutdown")) {
					it->attr("shutdown")(pybind11::arg("kill") = true);
				}
			} catch (const pybind11::error_already_set &) {
				PyErr_Clear();
			}
		}
		resources.clear();
	}

	void ReleaseResourcesWithoutPython() {
		for (auto &resource : resources) {
			resource.release();
		}
		resources.clear();
	}

	idx_t scope_depth = 0;
	unordered_set<PreparedStatementData *> prepared_statements;
	vector<pybind11::object> resources;
};

ScopedPythonUDFActorResourcePreparation::ScopedPythonUDFActorResourcePreparation(ClientContext &context) {
	state = context.registered_state->GetOrCreate<PythonUDFActorResourceState>("python_udf_actor_resources");
	state->BeginScope();
}

ScopedPythonUDFActorResourcePreparation::~ScopedPythonUDFActorResourcePreparation() {
	if (state) {
		state->EndScope();
	}
}

} // namespace duckdb
