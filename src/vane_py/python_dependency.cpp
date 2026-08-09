// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "vane_python/python_dependency.hpp"
#include "duckdb/common/helper.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"

namespace duckdb {

PythonDependencyItem::PythonDependencyItem(unique_ptr<RegisteredObject> &&object) : object(std::move(object)) {
}

PythonDependencyItem::~PythonDependencyItem() { // NOLINT - cannot throw in exception
	PythonGILWrapper gil;
	object.reset();
}

shared_ptr<DependencyItem> PythonDependencyItem::Create(py::object object) {
	auto registered_object = make_uniq<RegisteredObject>(std::move(object));
	return make_shared_ptr<PythonDependencyItem>(std::move(registered_object));
}

shared_ptr<DependencyItem> PythonDependencyItem::Create(unique_ptr<RegisteredObject> &&object) {
	return make_shared_ptr<PythonDependencyItem>(std::move(object));
}

} // namespace duckdb
