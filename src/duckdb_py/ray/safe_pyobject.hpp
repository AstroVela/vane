// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "duckdb_python/pybind11/gil_wrapper.hpp"

#include <pybind11/pybind11.h>

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

} // namespace ray
} // namespace python
} // namespace distributed
} // namespace duckdb
