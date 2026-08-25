// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"

#include "duckdb/common/types/value.hpp"

#include <optional>

namespace duckdb {

class PythonFile final {
public:
	PythonFile(string url, std::optional<string> content_type, std::optional<int64_t> position,
	           std::optional<int64_t> size, std::optional<string> checksum);

	static void Initialize(py::handle &m);
	static PythonFile FromPython(const py::handle &url, const py::handle &content_type, const py::handle &position,
	                             const py::handle &size, const py::handle &checksum);
	static PythonFile FromValue(const Value &value, const LogicalType &type);

	Value ToValue() const;
	string ToString() const;
	string Repr() const;
	bool Equals(const PythonFile &other) const;
	bool NotEquals(const PythonFile &other) const;
	Py_hash_t Hash() const;
	py::tuple State() const;

	const string &Url() const;
	const std::optional<string> &ContentType() const;
	const std::optional<int64_t> &Position() const;
	const std::optional<int64_t> &Size() const;
	const std::optional<string> &Checksum() const;

private:
	static void Validate(const std::optional<int64_t> &position, const std::optional<int64_t> &size,
	                     const std::optional<string> &checksum);

private:
	string url;
	std::optional<string> content_type;
	std::optional<int64_t> position;
	std::optional<int64_t> size;
	std::optional<string> checksum;
};

} // namespace duckdb
