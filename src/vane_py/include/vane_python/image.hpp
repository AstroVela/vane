// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"

#include "duckdb/common/types/value.hpp"

namespace duckdb {

//! The immutable Python representation of a decoded engine IMAGE value.
class PythonImage final {
public:
	PythonImage(string data, uint32_t width, uint32_t height, string mode);

	static void Initialize(py::handle &m);
	static PythonImage FromPython(const py::handle &data, const py::handle &width, const py::handle &height,
	                              const py::handle &mode);
	static py::object FromValue(const Value &value);

	Value ToValue() const;
	string Repr() const;
	bool Equals(const PythonImage &other) const;
	bool NotEquals(const PythonImage &other) const;
	Py_hash_t Hash() const;
	py::tuple State() const;

	py::bytes Data() const;
	uint32_t Width() const;
	uint32_t Height() const;
	uint8_t Channels() const;
	const string &Mode() const;
	string DType() const;

private:
	string data;
	uint32_t width;
	uint32_t height;
	uint8_t channels;
	string mode;
};

} // namespace duckdb
