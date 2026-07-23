// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "duckdb_python/pybind11/gil_wrapper.hpp"

#include <pybind11/pybind11.h>

#include <mutex>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace duckdb {
namespace distributed {
namespace python {
namespace ray {

static inline bool SafePyObjectCanDecRef() {
	if (!Py_IsInitialized()) {
		return false;
	}
	if (PythonIsFinalizing()) {
		return false;
	}
	return true;
}

struct SafePyObject {
	bool has_value_;
	py::object obj_;

	SafePyObject() : has_value_(false), obj_() {
	}
	explicit SafePyObject(py::object o) : has_value_(true), obj_(std::move(o)) {
	}

	// Copy: must acquire GIL to safely increment Python refcounts
	SafePyObject(const SafePyObject &other) : has_value_(false), obj_() {
		if (other.has_value_ && other.obj_.ptr() && SafePyObjectCanDecRef()) {
			PythonGILWrapper acquire;
			obj_ = other.obj_;
			has_value_ = true;
		}
	}
	SafePyObject &operator=(const SafePyObject &other) {
		if (&other == this)
			return *this;
		reset_with_gil();
		if (other.has_value_ && other.obj_.ptr() && SafePyObjectCanDecRef()) {
			PythonGILWrapper acquire;
			obj_ = other.obj_;
			has_value_ = true;
		}
		return *this;
	}

	SafePyObject(SafePyObject &&other) noexcept : has_value_(other.has_value_), obj_(std::move(other.obj_)) {
		other.has_value_ = false;
	}
	SafePyObject &operator=(SafePyObject &&other) noexcept {
		reset_with_gil();
		has_value_ = other.has_value_;
		obj_ = std::move(other.obj_);
		other.has_value_ = false;
		return *this;
	}

	~SafePyObject() {
		if (!obj_.ptr()) {
			has_value_ = false;
			return;
		}
		if (!SafePyObjectCanDecRef()) {
			// During interpreter finalization CPython state can already be gone.
			// Leak the ref instead of letting py::object's destructor DECREF.
			obj_.release();
			has_value_ = false;
			return;
		}
		PythonGILWrapper acquire;
		PyObject *ptr = obj_.release().ptr();
		Py_DECREF(ptr);
		has_value_ = false;
	}

	void reset_with_gil() {
		if (!obj_.ptr()) {
			has_value_ = false;
			return;
		}
		if (!SafePyObjectCanDecRef()) {
			obj_.release();
			has_value_ = false;
			return;
		}
		PythonGILWrapper acquire;
		PyObject *ptr = obj_.release().ptr();
		Py_DECREF(ptr);
		has_value_ = false;
	}

	py::object get() const {
		return (has_value_ && obj_.ptr()) ? obj_ : py::none();
	}
	bool empty() const {
		return !has_value_ || !obj_.ptr();
	}
	bool has_value() const {
		return has_value_ && obj_.ptr();
	}
};

struct SafePythonException {
	SafePyObject type;
	SafePyObject value;
	SafePyObject traceback;

	SafePythonException() = default;
	explicit SafePythonException(const py::error_already_set &error)
	    : type(py::object(error.type())), value(py::object(error.value())), traceback(py::object(error.trace())) {
	}

	bool has_value() const {
		return type.has_value() && value.has_value();
	}

	void Restore() const {
		if (!has_value()) {
			return;
		}
		auto type_obj = type.get();
		auto value_obj = value.get();
		PyObject *traceback_ptr = nullptr;
		if (traceback.has_value()) {
			auto traceback_obj = traceback.get();
			traceback_ptr = traceback_obj.release().ptr();
		}
		PyErr_Restore(type_obj.release().ptr(), value_obj.release().ptr(), traceback_ptr);
	}
};

// Retains normalized Python exception triples while a background DuckDB task
// reports its string-only DuckDBError through PlanExecutionStatus.
class PythonExceptionStore {
public:
	void Store(const std::string &query_id, const py::error_already_set &error) {
		if (query_id.empty()) {
			return;
		}
		PythonGILWrapper gil;
		SafePythonException stored(error);
		std::lock_guard<std::mutex> guard(mutex_);
		errors_.try_emplace(query_id, std::move(stored));
	}

	void RethrowAsCause(const std::string &query_id, const std::string &message) {
		auto stored = Take(query_id);
		if (!stored.has_value()) {
			return;
		}
		PythonGILWrapper gil;
		stored.Restore();
		py::raise_from(PyExc_RuntimeError, message.c_str());
		throw py::error_already_set();
	}

	void Discard(const std::string &query_id) {
		(void)Take(query_id);
	}

private:
	SafePythonException Take(const std::string &query_id) {
		SafePythonException stored;
		if (query_id.empty()) {
			return stored;
		}
		{
			std::lock_guard<std::mutex> guard(mutex_);
			auto entry = errors_.find(query_id);
			if (entry == errors_.end()) {
				return stored;
			}
			stored = std::move(entry->second);
			errors_.erase(entry);
		}
		return stored;
	}

	std::mutex mutex_;
	std::unordered_map<std::string, SafePythonException> errors_;
};

} // namespace ray
} // namespace python
} // namespace distributed
} // namespace duckdb
