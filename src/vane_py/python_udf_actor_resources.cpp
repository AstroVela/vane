// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/python_udf_actor_resources.hpp"

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
#include "vane_python/pybind11/gil_wrapper.hpp"
#include "vane_python/python_objects.hpp"

#include <algorithm>
#include <exception>
#include <pybind11/pybind11.h>
#include <sstream>
#include <unordered_set>

namespace duckdb {

namespace {

namespace py = pybind11;

static constexpr idx_t UDF_ACTOR_CLEANUP_WARNING_LIMIT = 16;
static constexpr idx_t UDF_ACTOR_CLEANUP_WARNING_MAX_BYTES = 4 * 1024;
static constexpr const char *UDF_ACTOR_CLEANUP_WARNINGS_OMITTED = "additional UDF actor cleanup warnings omitted";

static bool IsUTF8ContinuationByte(char byte) {
	return (static_cast<unsigned char>(byte) & 0xC0U) == 0x80U;
}

static string BoundCleanupWarning(string warning) {
	StringUtil::Trim(warning);
	if (warning.size() <= UDF_ACTOR_CLEANUP_WARNING_MAX_BYTES) {
		return warning;
	}
	static constexpr const char *OMISSION = "...";
	static constexpr idx_t OMISSION_BYTES = 3;
	const auto remaining = UDF_ACTOR_CLEANUP_WARNING_MAX_BYTES - OMISSION_BYTES;
	idx_t prefix_end = remaining / 2;
	while (prefix_end > 0 && IsUTF8ContinuationByte(warning[prefix_end])) {
		prefix_end--;
	}
	idx_t suffix_start = warning.size() - (remaining - remaining / 2);
	while (suffix_start < warning.size() && IsUTF8ContinuationByte(warning[suffix_start])) {
		suffix_start++;
	}
	return warning.substr(0, prefix_end) + OMISSION + warning.substr(suffix_start);
}

static string BoundedPythonErrorDetail(const py::handle &value) {
#if PY_VERSION_HEX >= 0x030C0000
	auto *args_ptr = PyException_GetArgs(value.ptr());
#else
	// PyException_GetArgs was added in Python 3.12. The project also builds
	// against 3.10 and 3.11, where BaseException's public C layout is the only
	// way to read args without dispatching a provider-defined __getattribute__.
	auto *args_ptr = reinterpret_cast<PyBaseExceptionObject *>(value.ptr())->args;
	Py_XINCREF(args_ptr);
#endif
	if (!args_ptr) {
		if (PyErr_Occurred()) {
			throw py::error_already_set();
		}
		return "<cleanup error message unavailable>";
	}
	auto args_obj = py::reinterpret_steal<py::object>(args_ptr);
	if (!PyTuple_CheckExact(args_obj.ptr())) {
		return "<cleanup error message unavailable>";
	}
	auto args = py::reinterpret_borrow<py::tuple>(args_obj);
	if (args.size() != 1) {
		return "<cleanup error message unavailable>";
	}
	auto text = py::reinterpret_borrow<py::object>(args[0]);
	if (!PyUnicode_CheckExact(text.ptr())) {
		return "<cleanup error message unavailable>";
	}
	const auto character_count = PyUnicode_GetLength(text.ptr());
	if (character_count < 0) {
		throw py::error_already_set();
	}
	if (character_count > static_cast<Py_ssize_t>(UDF_ACTOR_CLEANUP_WARNING_MAX_BYTES)) {
		return "cleanup error message exceeds 4096 characters and was omitted";
	}
	auto detail = text.cast<string>();
	if (detail.size() > UDF_ACTOR_CLEANUP_WARNING_MAX_BYTES) {
		return "cleanup error message exceeds 4096 bytes and was omitted";
	}
	return detail;
}

static string CleanupWarningFromPythonError(const py::error_already_set &error) {
	string detail;
	try {
		detail = BoundedPythonErrorDetail(error.value());
	} catch (...) {
		PyErr_Clear();
		detail = "<cleanup error message unavailable>";
	}
	PyErr_Clear();
	return BoundCleanupWarning("Python UDF actor resource cleanup failed: " + detail);
}

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
	if (bind_data.actor_handles) {
		auto *boxed_options = static_cast<py::object *>(bind_data.actor_handles.get());
		if (!py::isinstance<py::dict>(*boxed_options)) {
			throw InvalidInputException("udf executor options must be a dict");
		}
		meta[py::str("executor_options")] = py::reinterpret_borrow<py::dict>(*boxed_options);
	}
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

static void AppendOwnedResourcesFromError(vector<py::object> &resources, const py::error_already_set &error) {
	try {
		auto value = error.value();
		if (!py::hasattr(value, "owned_actor_pools")) {
			return;
		}
		auto owned_obj = value.attr("owned_actor_pools");
		for (auto item : py::reinterpret_borrow<py::iterable>(owned_obj)) {
			auto resource = py::reinterpret_borrow<py::object>(item);
			const auto duplicate = std::any_of(resources.begin(), resources.end(), [&](const py::object &existing) {
				return existing.ptr() == resource.ptr();
			});
			if (!duplicate) {
				resources.push_back(std::move(resource));
			}
		}
	} catch (...) {
		// Ownership extraction is best effort only for malformed third-party
		// exceptions. Preserve the original preparation failure.
		PyErr_Clear();
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
		if (scope_depth == 0) {
			cleanup_warnings.clear();
			capture_cleanup_warnings = false;
		}
		scope_depth++;
	}

	void EndScope() {
		if (scope_depth > 0) {
			scope_depth--;
		}
		if (scope_depth == 0) {
			cleanup_warnings.clear();
			capture_cleanup_warnings = false;
		}
	}

	vector<string> TakeCleanupWarnings() {
		vector<string> result;
		result.swap(cleanup_warnings);
		return result;
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
		const auto capture_errors = capture_cleanup_warnings;
		capture_cleanup_warnings = false;
		if (resources.empty()) {
			return;
		}
		if (!Py_IsInitialized() || PythonIsFinalizing()) {
			ReleaseResourcesWithoutPython();
			return;
		}
		PythonGILWrapper gil;
		// ClientContext drains every executor task before invoking QueryEnd. The
		// actor callables can therefore release provider state even when the
		// query itself failed; forced termination is reserved for setup rollback
		// and context destruction where quiescence is not established here.
		ShutdownResources(false, capture_errors);
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
		ShutdownResources(true, false);
		if (!resources.empty()) {
			// ShutdownResources retains every owner whose forced cleanup raised.
			// Give transient failures one bounded retry before this state loses its
			// last opportunity to release them during ClientContext destruction.
			ShutdownResources(true, false);
		}
		if (!resources.empty()) {
			// The GIL guard is a local and is destroyed before this state's members.
			// Never let a still-retained py::object reach member destruction after
			// the GIL has been released. Cleanup has already exhausted its bounded
			// retries, so perform the final DECREF while Python is still usable;
			// the Python owners' destructors retain their own last-chance cleanup.
			resources.clear();
		}
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
		// PendingQuery may first close an older streaming query on this
		// connection. Arm warning capture only after preparing the plan that
		// belongs to the active execution scope.
		capture_cleanup_warnings = true;
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
			if (!bind_data) {
				continue;
			}
			string backend;
			if (!PayloadStringField(bind_data->payload, "execution_backend", backend)) {
				continue;
			}
			if (backend == "subprocess_actor") {
				subprocess_nodes.append(BuildUDFNode(node_id, *bind_data, context));
			} else if (backend == "ray_actor" && !bind_data->actor_handles) {
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
		} catch (const pybind11::error_already_set &error) {
			// The Python helper carries pools whose rollback is incomplete on its
			// exception. Retain them in the ClientContext state before retrying;
			// otherwise unwinding the exception would discard the last explicit
			// process/shared-memory owner.
			AppendOwnedResourcesFromError(resources, error);
			ShutdownResources(true, false);
			throw;
		} catch (...) {
			ShutdownResources(true, false);
			throw;
		}
	}

	void ShutdownResources(bool kill, bool capture_errors) {
		if (resources.empty()) {
			return;
		}
		vector<pybind11::object> retry_resources;
		for (auto it = resources.rbegin(); it != resources.rend(); ++it) {
			try {
				if (pybind11::hasattr(*it, "shutdown")) {
					it->attr("shutdown")(pybind11::arg("kill") = kill);
				}
			} catch (const pybind11::error_already_set &error) {
				retry_resources.push_back(std::move(*it));
				if (!capture_errors) {
					PyErr_Clear();
					continue;
				}
				if (cleanup_warnings.size() < UDF_ACTOR_CLEANUP_WARNING_LIMIT) {
					cleanup_warnings.push_back(CleanupWarningFromPythonError(error));
				} else {
					PyErr_Clear();
					cleanup_warnings.back() = UDF_ACTOR_CLEANUP_WARNINGS_OMITTED;
				}
			} catch (const std::exception &) {
				retry_resources.push_back(std::move(*it));
				PyErr_Clear();
				if (capture_errors) {
					AppendNonPythonCleanupWarning();
				}
			} catch (...) {
				retry_resources.push_back(std::move(*it));
				PyErr_Clear();
				if (capture_errors) {
					AppendNonPythonCleanupWarning();
				}
			}
		}
		// LocalSubprocessActorPool keeps failed process owners so shutdown can
		// be retried. Preserve those Python objects in creation order; otherwise
		// clearing this vector would defeat that ownership contract and leak the
		// retained worker after a transient close/kill failure.
		std::reverse(retry_resources.begin(), retry_resources.end());
		resources = std::move(retry_resources);
	}

	void AppendNonPythonCleanupWarning() {
		if (cleanup_warnings.size() < UDF_ACTOR_CLEANUP_WARNING_LIMIT) {
			cleanup_warnings.push_back("Python UDF actor resource cleanup failed: non-Python cleanup exception");
		} else {
			cleanup_warnings.back() = UDF_ACTOR_CLEANUP_WARNINGS_OMITTED;
		}
	}

	void ReleaseResourcesWithoutPython() {
		for (auto &resource : resources) {
			resource.release();
		}
		resources.clear();
	}

	idx_t scope_depth = 0;
	bool capture_cleanup_warnings = false;
	unordered_set<PreparedStatementData *> prepared_statements;
	vector<pybind11::object> resources;
	vector<string> cleanup_warnings;
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

vector<string> ScopedPythonUDFActorResourcePreparation::TakeCleanupWarnings() {
	if (!state) {
		return {};
	}
	return state->TakeCleanupWarnings();
}

} // namespace duckdb
