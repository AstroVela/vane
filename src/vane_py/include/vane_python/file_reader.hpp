// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"

#include "duckdb/common/common.hpp"

#include <atomic>
#include <mutex>

namespace duckdb {

class ClientContext;
struct DuckDBPyConnection;
class PythonFile;
class ResolvedFile;

//! Native, range-aware cursor used by the public Python VaneFileReader.
class PythonFileReaderHandle final {
public:
	~PythonFileReaderHandle();

	static void Initialize(py::module_ &m);
	static shared_ptr<PythonFileReaderHandle> Open(const PythonFile &file, idx_t buffer_size,
	                                               shared_ptr<DuckDBPyConnection> connection);

	py::bytes Read(int64_t size);
	int64_t Seek(int64_t offset, int whence);
	int64_t Tell();
	int64_t Size();
	py::object GuessMimeType();
	void Close();
	bool Closed() const;
	string ToString() const;
	string Repr() const;

private:
	PythonFileReaderHandle(string url, idx_t buffer_size, shared_ptr<DuckDBPyConnection> connection,
	                       shared_ptr<ClientContext> context, unique_ptr<ResolvedFile> resolved);

	void RequireOpen() const;
	idx_t ReadLocked(data_ptr_t target, idx_t requested_size);
	void FillBufferLocked();

	const string url;
	const idx_t buffer_size;
	shared_ptr<DuckDBPyConnection> connection;
	shared_ptr<ClientContext> context;
	unique_ptr<ResolvedFile> resolved;
	vector<data_t> buffer;
	uint64_t buffer_start = 0;
	uint64_t position = 0;
	std::atomic<bool> closed {false};
	mutable std::mutex lock;
};

} // namespace duckdb
