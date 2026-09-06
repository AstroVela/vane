// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"

#include "duckdb/common/common.hpp"

#include <atomic>
#include <mutex>

namespace duckdb {

class ClientContext;
class DuckDBPyConnection;
class PythonFile;
class PythonDataSourceExecutionContext;
class ResolvedFile;

//! Native, range-aware cursor used by the public Python VaneFileReader.
class PythonFileReaderHandle final {
public:
	~PythonFileReaderHandle();

	static void Initialize(py::module_ &m);
	static shared_ptr<PythonFileReaderHandle> Open(const PythonFile &file, idx_t buffer_size,
	                                               shared_ptr<DuckDBPyConnection> connection);
	static shared_ptr<PythonFileReaderHandle>
	OpenInDataSourceContext(const PythonFile &file, idx_t buffer_size,
	                        const shared_ptr<PythonDataSourceExecutionContext> &execution_context);

	py::bytes Read(int64_t size);
	py::bytes ReadAndCheckInterrupted(int64_t size);
	int64_t Seek(int64_t offset, int whence);
	int64_t Tell();
	int64_t Size();
	void CheckInterrupted();
	py::object GuessMimeType();
	void Close();
	void CloseAndCheckInterrupted();
	bool Closed() const;
	string ToString() const;
	string Repr() const;

private:
	PythonFileReaderHandle(string url, idx_t buffer_size, shared_ptr<DuckDBPyConnection> connection,
	                       shared_ptr<PythonDataSourceExecutionContext> execution_context,
	                       shared_ptr<ClientContext> context, unique_ptr<ResolvedFile> resolved,
	                       uint64_t interrupt_generation);

	void RequireOpen() const;
	std::unique_lock<std::mutex> LockDataSourceContext() const;
	void CloseInternal(bool check_interrupted);
	py::bytes ReadInternal(int64_t size, bool check_retained_interrupt);
	idx_t ReadLocked(data_ptr_t target, idx_t requested_size);
	void FillBufferLocked();

	const string url;
	const idx_t buffer_size;
	shared_ptr<DuckDBPyConnection> connection;
	shared_ptr<PythonDataSourceExecutionContext> execution_context;
	shared_ptr<ClientContext> context;
	unique_ptr<ResolvedFile> resolved;
	const uint64_t interrupt_generation;
	vector<data_t> buffer;
	uint64_t buffer_start = 0;
	uint64_t position = 0;
	std::atomic<bool> closed {false};
	mutable std::mutex close_lock;
	mutable std::mutex lock;
};

} // namespace duckdb
