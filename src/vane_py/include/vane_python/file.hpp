// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"

#include "duckdb/common/types/value.hpp"

#include <optional>

namespace duckdb {

class DuckDBPyConnection;

//! Python-facing schema specialization selector for the FILE logical family.
class PythonFileMediaType final {
public:
	explicit PythonFileMediaType(FileMediaType media_type);

	static PythonFileMediaType Unknown();
	static PythonFileMediaType Image();
	static PythonFileMediaType Audio();
	static PythonFileMediaType Video();

	FileMediaType Type() const;
	string Repr() const;
	bool Equals(const PythonFileMediaType &other) const;
	Py_hash_t Hash() const;

private:
	FileMediaType media_type;
};

//! The immutable Python representation of an engine FILE value.
class PythonFile {
public:
	PythonFile(string url, std::optional<string> content_type, std::optional<int64_t> position,
	           std::optional<int64_t> size, std::optional<string> checksum,
	           FileMediaType media_type = FileMediaType::UNKNOWN);
	virtual ~PythonFile() = default;

	static void Initialize(py::handle &m);
	static PythonFile FromPython(const py::handle &url, const py::handle &content_type, const py::handle &position,
	                             const py::handle &size, const py::handle &checksum);
	static py::object FromValue(const Value &value);

	Value ToValue() const;
	string ToString() const;
	string Repr() const;
	bool Equals(const PythonFile &other) const;
	bool NotEquals(const PythonFile &other) const;
	Py_hash_t Hash() const;
	py::tuple State() const;
	bool Exists(shared_ptr<DuckDBPyConnection> connection) const;
	py::object Stat(shared_ptr<DuckDBPyConnection> connection) const;
	py::object MimeType(const string &detect, shared_ptr<DuckDBPyConnection> connection) const;
	bool IsMediaType(FileMediaType target) const;
	py::object AsMediaType(FileMediaType target) const;

	const string &Url() const;
	const std::optional<string> &ContentType() const;
	const std::optional<int64_t> &Position() const;
	const std::optional<int64_t> &Size() const;
	const std::optional<string> &Checksum() const;
	FileMediaType MediaType() const;

private:
	FileMediaType media_type;
	string url;
	std::optional<string> content_type;
	std::optional<int64_t> position;
	std::optional<int64_t> size;
	std::optional<string> checksum;
};

class PythonImageFile final : public PythonFile {
public:
	PythonImageFile(string url, std::optional<string> content_type, std::optional<int64_t> position,
	                std::optional<int64_t> size, std::optional<string> checksum);
};

class PythonAudioFile final : public PythonFile {
public:
	PythonAudioFile(string url, std::optional<string> content_type, std::optional<int64_t> position,
	                std::optional<int64_t> size, std::optional<string> checksum);
};

class PythonVideoFile final : public PythonFile {
public:
	PythonVideoFile(string url, std::optional<string> content_type, std::optional<int64_t> position,
	                std::optional<int64_t> size, std::optional<string> checksum);
};

} // namespace duckdb
