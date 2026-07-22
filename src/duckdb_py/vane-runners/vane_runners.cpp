// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// C++ translation of src/duckdb_py/duckdb-runners (Rust -> C++)
#include "duckdb_python/pybind11/gil_wrapper.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
// This file mirrors the structure and public behavior of the original
// Rust implementation (`runners.rs` + `python.rs` + parts of `lib.rs`) using
// pybind11 for Python interoperability.

#include <pybind11/pybind11.h>
#include <Python.h>
#include <mutex>
#include <memory>

#include <cstdlib>

namespace py = pybind11;

static bool VaneRunnerCanDecRefPython() {
	if (!Py_IsInitialized()) {
		return false;
	}
	if (duckdb::PythonIsFinalizing()) {
		return false;
	}
	return true;
}

static void VaneRunnerReleasePyObject(py::object &obj) {
	if (!obj.ptr()) {
		return;
	}
	if (!VaneRunnerCanDecRefPython()) {
		obj.release();
		return;
	}
	duckdb::PythonGILWrapper gil;
	PyObject *ptr = obj.release().ptr();
	Py_DECREF(ptr);
}

static void VaneRunnerClosePyObject(py::object &obj) {
	if (!obj.ptr() || !VaneRunnerCanDecRefPython()) {
		return;
	}
	duckdb::PythonGILWrapper gil;
	try {
		if (py::hasattr(obj, "close")) {
			obj.attr("close")();
		}
	} catch (const py::error_already_set &) {
		PyErr_Clear();
	} catch (...) {
	}
}

// ------------------ Runner wrappers ------------------ //
class RunnerBase {
public:
	virtual ~RunnerBase() = default;
	virtual py::object to_pyobj() const = 0;
};

class RayRunner : public RunnerBase {
public:
	static constexpr const char *NAME = "ray";
	explicit RayRunner(py::object pyobj_) : pyobj(pyobj_) {
	}

	~RayRunner() {
		VaneRunnerClosePyObject(pyobj);
		VaneRunnerReleasePyObject(pyobj);
	}

	// factory: import vane.runners.ray_runner and instantiate
	static std::unique_ptr<RayRunner> try_new(const std::pair<bool, std::string> &address,
	                                          const std::pair<bool, size_t> &max_task_backlog,
	                                          const std::pair<bool, bool> &force_client_mode) {
		duckdb::PythonGILWrapper gil;
		py::module_ mod = py::module_::import("vane.runners.ray.runner");
		py::object RayRunnerClass = mod.attr("RayRunner");
		py::object py_address = address.first ? py::cast(address.second) : py::none();
		py::object py_max_task_backlog = max_task_backlog.first ? py::cast(max_task_backlog.second) : py::none();
		py::object instance;
		if (force_client_mode.first) {
			instance = RayRunnerClass(py_address, py_max_task_backlog, force_client_mode.second);
		} else {
			instance = RayRunnerClass(py_address, py_max_task_backlog);
		}
		return std::unique_ptr<RayRunner>(new RayRunner(instance));
	}

	py::object to_pyobj() const override {
		return pyobj;
	}

private:
	py::object pyobj;
};

class LocalRunner : public RunnerBase {
public:
	static constexpr const char *NAME = "local";
	explicit LocalRunner(py::object pyobj_) : pyobj(pyobj_) {
	}

	~LocalRunner() {
		VaneRunnerReleasePyObject(pyobj);
	}

	static std::unique_ptr<LocalRunner> try_new() {
		duckdb::PythonGILWrapper gil;
		py::module_ mod = py::module_::import("vane.runners.local.runner");
		py::object LocalRunnerClass = mod.attr("LocalRunner");
		py::object instance = LocalRunnerClass();
		return std::unique_ptr<LocalRunner>(new LocalRunner(instance));
	}

	static std::unique_ptr<LocalRunner> try_new(py::object num_workers, py::object max_running_tasks,
	                                            py::object execution_mode) {
		duckdb::PythonGILWrapper gil;
		py::module_ mod = py::module_::import("vane.runners.local.runner");
		py::object LocalRunnerClass = mod.attr("LocalRunner");
		py::dict kwargs;
		if (!num_workers.is_none()) {
			kwargs["num_workers"] = num_workers;
		}
		if (!max_running_tasks.is_none()) {
			kwargs["max_running_tasks"] = max_running_tasks;
		}
		if (!execution_mode.is_none()) {
			kwargs["execution_mode"] = execution_mode;
		}
		py::object instance = LocalRunnerClass(**kwargs);
		return std::unique_ptr<LocalRunner>(new LocalRunner(instance));
	}

	py::object to_pyobj() const override {
		return pyobj;
	}

private:
	py::object pyobj;
};

// Variant wrapper
class Runner {
public:
	enum class Type { Ray, Local };

	explicit Runner(std::unique_ptr<RayRunner> r) : type(Type::Ray), ray(std::move(r)) {
	}
	explicit Runner(std::unique_ptr<LocalRunner> l) : type(Type::Local), local(std::move(l)) {
	}

	Type get_type() const {
		return type;
	}

	// call run_iter_tables on the underlying Python runner object
	// lp_builder is expected to be a Python LogicalPlanBuilder-compatible object
	py::iterator run_iter_tables(py::object py_builder, const std::pair<bool, size_t> &results_buffer_size) const {
		duckdb::PythonGILWrapper gil;
		py::object runner_obj = get_pyobj();
		py::object py_buf_size = results_buffer_size.first ? py::cast(results_buffer_size.second) : py::none();
		py::object result = runner_obj.attr("run_iter_tables")(py_builder, py_buf_size);
		return py::iterator(result);
	}

	py::object get_pyobj() const {
		duckdb::PythonGILWrapper gil;
		if (type == Type::Ray) {
			return ray->to_pyobj();
		}
		return local->to_pyobj();
	}

private:
	Type type;
	std::unique_ptr<RayRunner> ray;
	std::unique_ptr<LocalRunner> local;
};

// ---------------- RunnerConfig ---------------- //
struct RunnerConfig {
	enum class Which { Local, Ray } which;
	// Ray config
	std::pair<bool, std::string> address;
	std::pair<bool, size_t> max_task_backlog;
	std::pair<bool, bool> force_client_mode;

	RunnerConfig()
	    : which(Which::Local), address(std::make_pair(false, std::string())),
	      max_task_backlog(std::make_pair(false, size_t(0))), force_client_mode(std::make_pair(false, false)) {
	}

	static RunnerConfig RayCfg(std::pair<bool, std::string> addr, std::pair<bool, size_t> backlog,
	                           std::pair<bool, bool> force) {
		RunnerConfig rc;
		rc.which = Which::Ray;
		rc.address = std::move(addr);
		rc.max_task_backlog = backlog;
		rc.force_client_mode = force;
		return rc;
	}
	static RunnerConfig LocalCfg() {
		RunnerConfig rc;
		rc.which = Which::Local;
		return rc;
	}

	std::unique_ptr<Runner> create_runner() const {
		if (which == Which::Local) {
			return std::unique_ptr<Runner>(new Runner(LocalRunner::try_new()));
		}
		return std::unique_ptr<Runner>(new Runner(RayRunner::try_new(address, max_task_backlog, force_client_mode)));
	}
};

// ---------------- Env helpers ---------------- //
static std::pair<bool, size_t> parse_usize_env_var(const char *var_name) {
	const char *v = std::getenv(var_name);
	if (!v)
		return std::make_pair(false, size_t(0));
	try {
		size_t val = static_cast<size_t>(std::stoull(std::string(v)));
		return std::make_pair(true, val);
	} catch (...) {
		return std::make_pair(false, size_t(0));
	}
}

static RunnerConfig get_ray_runner_config_from_env() {
	const char *RAY_ADDRESS = "RAY_ADDRESS";
	const char *VANE_RAY_MAX_TASK_BACKLOG = "VANE_RAY_MAX_TASK_BACKLOG";

	std::pair<bool, std::string> address = std::make_pair(false, std::string());
	const char *addr = std::getenv(RAY_ADDRESS);
	if (addr)
		address = std::make_pair(true, std::string(addr));

	auto max_task_backlog = parse_usize_env_var(VANE_RAY_MAX_TASK_BACKLOG);
	return RunnerConfig::RayCfg(address, max_task_backlog, std::make_pair(false, false));
}

static RunnerConfig get_runner_config_from_env_or_default() {
	auto rt = duckdb::ResolveRunnerTypeFromEnvironment();
	if (rt == "local-fast") {
		throw std::runtime_error(
		    "The configured internal direct execution mode does not create a runner. "
		    "Use relation APIs directly, set VANE_RUNNER=local for the local runner, or leave it empty for Ray.");
	}
	if (rt == LocalRunner::NAME)
		return RunnerConfig::LocalCfg();
	if (rt == RayRunner::NAME)
		return get_ray_runner_config_from_env();
	throw duckdb::InternalException("normalized runner type '%s' has no runner implementation", rt);
}

// ----------------- Singleton management ----------------- //
static std::shared_ptr<Runner> VANE_RUNNER_PTR = nullptr;
static std::mutex VANE_RUNNER_MUTEX;

// get_or_create_runner will initialize once if not set
static std::shared_ptr<Runner> get_or_create_runner_cpp() {
	std::lock_guard<std::mutex> guard(VANE_RUNNER_MUTEX);
	if (VANE_RUNNER_PTR) {
		return VANE_RUNNER_PTR;
	}
	RunnerConfig cfg = get_runner_config_from_env_or_default();
	std::unique_ptr<Runner> rptr = cfg.create_runner();
	VANE_RUNNER_PTR = std::shared_ptr<Runner>(std::move(rptr));
	return VANE_RUNNER_PTR;
}

// Helper to set runner once, following the Rust OnceLock semantics
static py::object set_runner_ray_py(py::object address_py = py::none(), bool noop_if_initialized = false,
                                    std::pair<bool, size_t> max_task_backlog = std::make_pair(false, size_t(0)),
                                    std::pair<bool, bool> force_client_mode = std::make_pair(false, false)) {
	std::lock_guard<std::mutex> guard(VANE_RUNNER_MUTEX);
	if (VANE_RUNNER_PTR) {
		if (noop_if_initialized && VANE_RUNNER_PTR->get_type() == Runner::Type::Ray) {
			return VANE_RUNNER_PTR->get_pyobj();
		}
		throw std::runtime_error("Cannot set runner more than once");
	}
	std::pair<bool, std::string> address = std::make_pair(false, std::string());
	if (!address_py.is_none()) {
		address = std::make_pair(true, address_py.cast<std::string>());
	}
	auto r = RayRunner::try_new(address, max_task_backlog, force_client_mode);
	VANE_RUNNER_PTR = std::make_shared<Runner>(std::move(r));
	return VANE_RUNNER_PTR->get_pyobj();
}

static py::object set_runner_local_py(py::object num_workers = py::none(), py::object max_running_tasks = py::none(),
                                      py::object execution_mode = py::none()) {
	std::lock_guard<std::mutex> guard(VANE_RUNNER_MUTEX);
	if (VANE_RUNNER_PTR) {
		if (VANE_RUNNER_PTR->get_type() == Runner::Type::Local) {
			return VANE_RUNNER_PTR->get_pyobj();
		}
		throw std::runtime_error("Cannot set runner more than once");
	}
	auto r = LocalRunner::try_new(num_workers, max_running_tasks, execution_mode);
	VANE_RUNNER_PTR = std::make_shared<Runner>(std::move(r));
	return VANE_RUNNER_PTR->get_pyobj();
}

// Helper to teardown runner explicitly during Python exit
static void teardown_runner_cpp() {
	std::shared_ptr<Runner> runner;
	{
		std::lock_guard<std::mutex> guard(VANE_RUNNER_MUTEX);
		if (VANE_RUNNER_PTR) {
			runner = VANE_RUNNER_PTR;
			VANE_RUNNER_PTR.reset();
		} else {
			return;
		}
	}
	try {
		duckdb::PythonGILWrapper gil;
		py::object runner_obj = runner->get_pyobj();
		VaneRunnerClosePyObject(runner_obj);
	} catch (const py::error_already_set &) {
		PyErr_Clear();
	} catch (...) {
	}
}

// Get runner Python object or None
static py::object get_runner_py() {
	std::lock_guard<std::mutex> guard(VANE_RUNNER_MUTEX);
	if (!VANE_RUNNER_PTR)
		return py::none();
	return VANE_RUNNER_PTR->get_pyobj();
}

static py::object get_or_create_runner_py() {
	auto r = get_or_create_runner_cpp();
	return r->get_pyobj();
}

static py::object get_or_infer_runner_type_py() {
	std::lock_guard<std::mutex> guard(VANE_RUNNER_MUTEX);
	if (VANE_RUNNER_PTR) {
		if (VANE_RUNNER_PTR->get_type() == Runner::Type::Ray)
			return py::str(RayRunner::NAME);
		return py::str(LocalRunner::NAME);
	}
	auto rt = duckdb::ResolveRunnerTypeFromEnvironment();
	if (rt == "local-fast") {
		return py::str("local-fast");
	}
	if (rt == LocalRunner::NAME) {
		return py::str(LocalRunner::NAME);
	}
	if (rt == RayRunner::NAME) {
		return py::str(RayRunner::NAME);
	}
	throw duckdb::InternalException("normalized runner type '%s' has no public runner name", rt);
}

// ------------------ Python binding ------------------ //
namespace duckdb {

void register_vane_runners(py::module_ &m) {
	m.doc() = "C++ translation of duckdb-runners (wrapped via pybind11)";

	m.def("get_runner", []() -> py::object { return get_runner_py(); }, "Return the current runner or None");

	m.def(
	    "get_or_create_runner", []() -> py::object { return get_or_create_runner_py(); },
	    "Get or create the global runner");

	m.def(
	    "get_or_infer_runner_type", []() -> py::object { return get_or_infer_runner_type_py(); },
	    "Infer/get runner type");

	m.def(
	    "set_runner_ray",
	    [](py::object address, bool noop_if_initialized, py::object max_task_backlog_py,
	       bool force_client_mode) -> py::object {
		    std::pair<bool, size_t> max_task_backlog = std::make_pair(false, size_t(0));
		    if (!max_task_backlog_py.is_none())
			    max_task_backlog = std::make_pair(true, max_task_backlog_py.cast<size_t>());
		    return set_runner_ray_py(address, noop_if_initialized, max_task_backlog,
		                             std::make_pair(true, force_client_mode));
	    },
	    py::arg("address") = py::none(), py::arg("noop_if_initialized") = false,
	    py::arg("max_task_backlog") = py::none(), py::arg("force_client_mode") = false);

	m.def(
	    "set_runner_local",
	    [](py::object num_workers, py::object max_running_tasks, py::object execution_mode) -> py::object {
		    return set_runner_local_py(num_workers, max_running_tasks, execution_mode);
	    },
	    py::arg("num_workers") = py::none(), py::arg("max_running_tasks") = py::none(),
	    py::arg("execution_mode") = py::none());

	// Expose teardown helper and register it with atexit so teardown runs
	// while Python is still fully initialized (avoids destructor races).
	m.def("teardown_runner", []() { teardown_runner_cpp(); });

	try {
		py::module_ atexit = py::module_::import("atexit");
		atexit.attr("register")(m.attr("teardown_runner"));
	} catch (...) {
	}

	// The public `vane` package installs its runner aliases after this native
	// module has finished initializing. Importing that package from here would
	// expose a partially initialized extension when `_vane_duckdb` is imported
	// directly.
}

} // namespace duckdb

// Provide a global wrapper so both pybind11 init functions and
// `duckdb::register_vane_runners` callers (in other translation units)
// can find the symbol regardless of namespace.
void register_vane_runners(py::module_ &m) {
	duckdb::register_vane_runners(m);
}
