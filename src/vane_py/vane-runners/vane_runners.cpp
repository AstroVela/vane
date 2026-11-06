// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Native bindings for Vane's runner implementations.
#include "duckdb/common/environment.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"

#include <pybind11/pybind11.h>
#include <Python.h>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <thread>

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
class RayRunner {
public:
	static constexpr const char *NAME = "ray";
	explicit RayRunner(py::object pyobj_) : pyobj(pyobj_) {
	}

	~RayRunner() {
		VaneRunnerClosePyObject(pyobj);
		VaneRunnerReleasePyObject(pyobj);
	}

	// Factory: import Vane's Ray runner and instantiate it.
	static std::unique_ptr<RayRunner> try_new(const std::pair<bool, std::string> &address,
	                                          const std::pair<bool, size_t> &max_task_backlog) {
		duckdb::PythonGILWrapper gil;
		py::module_ mod = py::module_::import("vane.runners.ray.runner");
		py::object RayRunnerClass = mod.attr("RayRunner");
		py::object py_address = address.first ? py::cast(address.second) : py::none();
		py::object py_max_task_backlog = max_task_backlog.first ? py::cast(max_task_backlog.second) : py::none();
		py::object instance = RayRunnerClass(py_address, py_max_task_backlog);
		return std::unique_ptr<RayRunner>(new RayRunner(instance));
	}

	py::object to_pyobj() const {
		return pyobj;
	}

private:
	py::object pyobj;
};

// ---------------- RunnerConfig ---------------- //
struct RunnerConfig {
	// Ray config
	std::pair<bool, std::string> address;
	std::pair<bool, size_t> max_task_backlog;

	RunnerConfig() : address(std::make_pair(false, std::string())), max_task_backlog(std::make_pair(false, size_t(0))) {
	}

	static RunnerConfig RayCfg(std::pair<bool, std::string> addr, std::pair<bool, size_t> backlog) {
		RunnerConfig rc;
		rc.address = std::move(addr);
		rc.max_task_backlog = backlog;
		return rc;
	}
	std::unique_ptr<RayRunner> create_runner() const {
		return RayRunner::try_new(address, max_task_backlog);
	}
};

// ---------------- Env helpers ---------------- //
static std::pair<bool, size_t> parse_usize_env_var(const char *var_name) {
	const duckdb::EnvironmentVariable v_env(var_name);
	const char *v = v_env.GetValue();
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
	const duckdb::EnvironmentVariable addr_env(RAY_ADDRESS);
	const char *addr = addr_env.GetValue();
	if (addr)
		address = std::make_pair(true, std::string(addr));

	auto max_task_backlog = parse_usize_env_var(VANE_RAY_MAX_TASK_BACKLOG);
	return RunnerConfig::RayCfg(address, max_task_backlog);
}

static RunnerConfig get_runner_config_from_env_or_default() {
	auto rt = duckdb::ResolveRunnerTypeFromEnvironment();
	if (rt == "local") {
		throw std::runtime_error("Local execution does not create a runner. Use connection or relation APIs directly.");
	}
	if (rt == RayRunner::NAME)
		return get_ray_runner_config_from_env();
	throw duckdb::InternalException("normalized runner type '%s' has no runner implementation", rt);
}

// ----------------- Runner registry ----------------- //
enum class VaneRunnerState { UNINITIALIZED, INITIALIZING, INITIALIZED };

struct RunnerSlot {
	std::shared_ptr<RayRunner> runner;
	VaneRunnerState state = VaneRunnerState::UNINITIALIZED;
	std::thread::id initializer_thread;
};

static RunnerSlot VANE_RUNNER_SLOT;
static std::mutex VANE_RUNNER_MUTEX;
static std::condition_variable VANE_RUNNER_CONDITION;

// Called with the registry mutex held.
static bool current_thread_initializes_runner() {
	return VANE_RUNNER_SLOT.state == VaneRunnerState::INITIALIZING &&
	       VANE_RUNNER_SLOT.initializer_thread == std::this_thread::get_id();
}

struct RunnerInitializationResult {
	std::shared_ptr<RayRunner> runner;
	bool created;
};

// All callers enter from Python with the GIL held. Wait without the GIL so the
// initializer can return from Python, then unlock the runner mutex before the
// gil_scoped_release destructor reacquires the GIL.
static void wait_for_runner_initialization(RunnerSlot &slot) {
	py::gil_scoped_release release;
	std::unique_lock<std::mutex> lock(VANE_RUNNER_MUTEX);
	VANE_RUNNER_CONDITION.wait(lock, [&] { return slot.state != VaneRunnerState::INITIALIZING; });
	lock.unlock();
}

template <class FACTORY>
static RunnerInitializationResult initialize_runner(FACTORY &&factory) {
	auto &slot = VANE_RUNNER_SLOT;
	for (;;) {
		bool should_initialize = false;
		{
			std::lock_guard<std::mutex> guard(VANE_RUNNER_MUTEX);
			if (slot.state == VaneRunnerState::INITIALIZED) {
				return {slot.runner, false};
			}
			if (current_thread_initializes_runner()) {
				throw std::runtime_error("Recursive runner initialization is not supported");
			}
			if (slot.state == VaneRunnerState::UNINITIALIZED) {
				slot.state = VaneRunnerState::INITIALIZING;
				slot.initializer_thread = std::this_thread::get_id();
				should_initialize = true;
			}
		}

		if (should_initialize) {
			break;
		}
		wait_for_runner_initialization(slot);
	}

	try {
		auto candidate = factory();
		{
			std::lock_guard<std::mutex> guard(VANE_RUNNER_MUTEX);
			slot.runner = candidate;
			slot.state = VaneRunnerState::INITIALIZED;
			slot.initializer_thread = std::thread::id();
		}
		VANE_RUNNER_CONDITION.notify_all();
		return {std::move(candidate), true};
	} catch (...) {
		{
			std::lock_guard<std::mutex> guard(VANE_RUNNER_MUTEX);
			slot.state = VaneRunnerState::UNINITIALIZED;
			slot.initializer_thread = std::thread::id();
		}
		VANE_RUNNER_CONDITION.notify_all();
		throw;
	}
}

// Initialize the Ray runner once, preserving any explicitly configured instance.
static std::shared_ptr<RayRunner> get_or_create_runner_cpp() {
	RunnerConfig cfg = get_runner_config_from_env_or_default();
	auto result = initialize_runner([&] { return std::shared_ptr<RayRunner>(cfg.create_runner()); });
	return std::move(result.runner);
}

// Configure the shared Ray runner.
static py::object set_runner_ray_py(py::object address_py = py::none(), bool noop_if_initialized = false,
                                    std::pair<bool, size_t> max_task_backlog = std::make_pair(false, size_t(0))) {
	auto result = initialize_runner([&] {
		std::pair<bool, std::string> address = std::make_pair(false, std::string());
		if (!address_py.is_none()) {
			address = std::make_pair(true, address_py.cast<std::string>());
		}
		auto runner = RayRunner::try_new(address, max_task_backlog);
		return std::shared_ptr<RayRunner>(std::move(runner));
	});
	if (!result.created && !noop_if_initialized) {
		throw std::runtime_error("Cannot set Ray runner more than once");
	}
	return result.runner->to_pyobj();
}

// Close the Ray runner, including when the current environment selects local.
static void teardown_runner_cpp() {
	std::shared_ptr<RayRunner> runner;
	for (;;) {
		{
			std::lock_guard<std::mutex> guard(VANE_RUNNER_MUTEX);
			if (current_thread_initializes_runner()) {
				throw std::runtime_error("Cannot tear down a runner while initializing it");
			}
			if (VANE_RUNNER_SLOT.state != VaneRunnerState::INITIALIZING) {
				runner = std::move(VANE_RUNNER_SLOT.runner);
				VANE_RUNNER_SLOT.state = VaneRunnerState::UNINITIALIZED;
				break;
			}
		}
		wait_for_runner_initialization(VANE_RUNNER_SLOT);
	}
	if (runner) {
		duckdb::PythonGILWrapper gil;
		py::object runner_obj = runner->to_pyobj();
		VaneRunnerClosePyObject(runner_obj);
	}
}

// Global convenience functions use the current configuration; connection
// dispatch initializes Ray independently of that configuration.
static py::object get_runner_py() {
	auto type = duckdb::ResolveRunnerTypeFromEnvironment();
	if (type == "local") {
		return py::none();
	}
	auto &slot = VANE_RUNNER_SLOT;
	for (;;) {
		std::shared_ptr<RayRunner> runner;
		bool uninitialized = false;
		{
			std::lock_guard<std::mutex> guard(VANE_RUNNER_MUTEX);
			if (slot.state == VaneRunnerState::UNINITIALIZED) {
				uninitialized = true;
			}
			if (slot.state == VaneRunnerState::INITIALIZED) {
				runner = slot.runner;
			}
			if (slot.state == VaneRunnerState::INITIALIZING && current_thread_initializes_runner()) {
				throw std::runtime_error("Cannot get a runner while initializing it");
			}
		}
		if (uninitialized) {
			return py::none();
		}
		if (runner) {
			return runner->to_pyobj();
		}
		wait_for_runner_initialization(slot);
	}
}

static py::object get_or_create_runner_py() {
	auto r = get_or_create_runner_cpp();
	return r->to_pyobj();
}

static py::object get_or_infer_runner_type_py() {
	return py::str(duckdb::ResolveRunnerTypeFromEnvironment());
}

// ------------------ Python binding ------------------ //
namespace duckdb {

void register_vane_runners(py::module_ &m) {
	m.doc() = "Native runner lifecycle functions for Vane";

	m.def(
	    "get_runner", []() -> py::object { return get_runner_py(); },
	    "Return the runner selected by VANE_RUNNER, or None if it is not initialized");

	m.def(
	    "get_or_create_runner", []() -> py::object { return get_or_create_runner_py(); },
	    "Get or create the runner selected by VANE_RUNNER");

	m.def(
	    "get_or_infer_runner_type", []() -> py::object { return get_or_infer_runner_type_py(); },
	    "Return the normalized runner type selected by VANE_RUNNER");

	m.def(
	    "set_runner_ray",
	    [](py::object address, bool noop_if_initialized, py::object max_task_backlog_py) -> py::object {
		    std::pair<bool, size_t> max_task_backlog = std::make_pair(false, size_t(0));
		    if (!max_task_backlog_py.is_none())
			    max_task_backlog = std::make_pair(true, max_task_backlog_py.cast<size_t>());
		    return set_runner_ray_py(address, noop_if_initialized, max_task_backlog);
	    },
	    py::arg("address") = py::none(), py::arg("noop_if_initialized") = false,
	    py::arg("max_task_backlog") = py::none());

	// Expose teardown helper and register it with atexit so teardown runs
	// while Python is still fully initialized (avoids destructor races).
	m.def("teardown_runner", []() { teardown_runner_cpp(); });

	try {
		py::module_ atexit = py::module_::import("atexit");
		atexit.attr("register")(m.attr("teardown_runner"));
	} catch (...) {
	}
}

} // namespace duckdb

// Provide a global wrapper so both pybind11 init functions and
// `duckdb::register_vane_runners` callers (in other translation units)
// can find the symbol regardless of namespace.
void register_vane_runners(py::module_ &m) {
	duckdb::register_vane_runners(m);
}
