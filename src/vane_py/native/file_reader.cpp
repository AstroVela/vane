// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/file_reader.hpp"

#include "file_resolver.hpp"
#include "file_value.hpp"
#include "vane_python/file.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/main/client_context.hpp"

#include <cstring>
#include <exception>
#include <limits>
#include <utility>

namespace duckdb {

namespace {

class ReaderContextScope {
public:
	ReaderContextScope(ClientContext &context_p, const DuckDBPyConnection &connection_p,
	                   uint64_t interrupt_generation_p)
	    : context(context_p), connection(connection_p), interrupt_generation(interrupt_generation_p) {
	}

	void CheckInterrupted() const {
		if (connection.InterruptInProgress() || connection.InterruptGeneration() != interrupt_generation ||
		    context.IsInterrupted()) {
			throw InterruptException();
		}
	}

private:
	ClientContext &context;
	const DuckDBPyConnection &connection;
	const uint64_t interrupt_generation;
};

template <class FUNC>
void RunReaderContextOperation(ClientContext &context, DuckDBPyConnection &connection, uint64_t interrupt_generation,
                               FUNC &&operation) {
	// The caller captures the generation when the reader operation becomes active.
	// It preserves an interrupt that races with this stale-state reset or with
	// RunFunctionInTransaction's auto-transaction startup reset.
	context.ClearInterrupt();
	std::exception_ptr operation_error;
	context.RunFunctionInTransaction(
	    [&]() {
		    ReaderContextScope context_scope(context, connection, interrupt_generation);
		    try {
			    context_scope.CheckInterrupted();
			    operation(context_scope);
			    context_scope.CheckInterrupted();
		    } catch (...) {
			    // FILE I/O is not a database statement. Preserve the exception until
			    // after DuckDB releases its public context/transaction boundary so an
			    // I/O failure cannot invalidate an explicit database transaction.
			    operation_error = std::current_exception();
		    }
	    },
	    false);
	if (operation_error) {
		std::rethrow_exception(operation_error);
	}
}

} // namespace

PythonFileReaderHandle::PythonFileReaderHandle(string url_p, idx_t buffer_size_p,
                                               shared_ptr<DuckDBPyConnection> connection_p,
                                               shared_ptr<ClientContext> context_p, unique_ptr<ResolvedFile> resolved_p,
                                               uint64_t interrupt_generation_p)
    : url(std::move(url_p)), buffer_size(buffer_size_p), connection(std::move(connection_p)),
      context(std::move(context_p)), resolved(std::move(resolved_p)), interrupt_generation(interrupt_generation_p) {
}

PythonFileReaderHandle::~PythonFileReaderHandle() = default;

void PythonFileReaderHandle::Initialize(py::module_ &m) {
	py::class_<PythonFileReaderHandle, shared_ptr<PythonFileReaderHandle>>(m, "_VaneFileReaderHandle",
	                                                                       py::module_local(), py::is_final())
	    .def("_read", &PythonFileReaderHandle::Read, py::arg("size") = -1)
	    .def("_read_and_check_interrupted", &PythonFileReaderHandle::ReadAndCheckInterrupted, py::arg("size") = -1)
	    .def("_seek", &PythonFileReaderHandle::Seek, py::arg("offset"), py::arg("whence") = 0)
	    .def("_tell", &PythonFileReaderHandle::Tell)
	    .def("_size", &PythonFileReaderHandle::Size)
	    .def("_check_interrupted", &PythonFileReaderHandle::CheckInterrupted)
	    .def("_guess_mime_type", &PythonFileReaderHandle::GuessMimeType)
	    .def("_close", &PythonFileReaderHandle::Close)
	    .def("_close_and_check_interrupted", &PythonFileReaderHandle::CloseAndCheckInterrupted)
	    .def_property_readonly("_closed", &PythonFileReaderHandle::Closed)
	    .def("__str__", &PythonFileReaderHandle::ToString)
	    .def("__repr__", &PythonFileReaderHandle::Repr);

	m.def("_open_file_reader", &PythonFileReaderHandle::Open, py::arg("file"), py::kw_only(), py::arg("buffer_size"),
	      py::arg("connection") = py::none());
}

shared_ptr<PythonFileReaderHandle> PythonFileReaderHandle::Open(const PythonFile &file, idx_t buffer_size,
                                                                shared_ptr<DuckDBPyConnection> connection) {
	if (buffer_size == 0) {
		throw InvalidInputException("FILE reader buffer_size must be greater than zero");
	}
	if (!connection) {
		connection = DuckDBPyConnection::DefaultConnection();
	}
	auto reference = FileReference::FromValue(file.ToValue(), "File.open");
	auto context = connection->con.GetConnection().context;
	auto interrupt_generation = connection->InterruptGeneration();
	unique_ptr<ResolvedFile> resolved;
	{
		D_ASSERT(py::gil_check());
		py::gil_scoped_release release;
		unique_lock<mutex> connection_guard(connection->py_connection_lock);
		RunReaderContextOperation(*context, *connection, interrupt_generation,
		                          [&](ReaderContextScope &) { resolved = ResolvedFile::Open(*context, reference); });
	}
	return shared_ptr<PythonFileReaderHandle>(new PythonFileReaderHandle(std::move(reference.url), buffer_size,
	                                                                     std::move(connection), std::move(context),
	                                                                     std::move(resolved), interrupt_generation));
}

void PythonFileReaderHandle::RequireOpen() const {
	if (closed.load() || !resolved) {
		throw py::value_error("I/O operation on closed VaneFileReader");
	}
}

void PythonFileReaderHandle::FillBufferLocked() {
	auto logical_size = resolved->LogicalSize();
	if (position >= logical_size) {
		buffer.clear();
		buffer_start = logical_size;
		return;
	}
	auto read_size = MinValue<uint64_t>(logical_size - position, buffer_size);
	vector<data_t> next_buffer(NumericCast<idx_t>(read_size));
	resolved->ReadExact(next_buffer.data(), read_size, position);
	buffer = std::move(next_buffer);
	buffer_start = position;
}

idx_t PythonFileReaderHandle::ReadLocked(data_ptr_t target, idx_t requested_size) {
	if (requested_size == 0) {
		return 0;
	}
	if (requested_size >= buffer_size) {
		resolved->ReadExact(target, requested_size, position);
		position += requested_size;
		return requested_size;
	}

	idx_t total_read = 0;
	while (total_read < requested_size) {
		auto in_buffer = position >= buffer_start && position - buffer_start < buffer.size();
		if (!in_buffer) {
			FillBufferLocked();
			if (buffer.empty()) {
				break;
			}
		}
		auto buffer_offset = NumericCast<idx_t>(position - buffer_start);
		auto copy_size = MinValue<idx_t>(requested_size - total_read, buffer.size() - buffer_offset);
		memcpy(target + total_read, buffer.data() + buffer_offset, copy_size);
		total_read += copy_size;
		position += copy_size;
	}
	return total_read;
}

py::bytes PythonFileReaderHandle::Read(int64_t size) {
	return ReadInternal(size, false);
}

py::bytes PythonFileReaderHandle::ReadAndCheckInterrupted(int64_t size) {
	return ReadInternal(size, true);
}

py::bytes PythonFileReaderHandle::ReadInternal(int64_t size, bool check_retained_interrupt) {
	string result;
	D_ASSERT(py::gil_check());
	// Generic reads establish an independent operation generation so a reader can
	// be reused after an interrupted read. Video decoding instead retains the
	// generation captured when the reader opened: checking it inside this native
	// operation closes the gap between a Python callback's pre-read check and the
	// connector read itself.
	auto operation_generation =
	    check_retained_interrupt ? interrupt_generation : (connection ? connection->InterruptGeneration() : 0);
	{
		py::gil_scoped_release release;
		unique_lock<mutex> reader_guard(lock);
		RequireOpen();
		auto logical_size = resolved->LogicalSize();
		auto remaining = position < logical_size ? logical_size - position : 0;
		auto requested_size = size < 0 ? remaining : MinValue<uint64_t>(remaining, NumericCast<uint64_t>(size));
		if (requested_size > 0) {
			auto initial_position = position;
			try {
				unique_lock<mutex> connection_guard(connection->py_connection_lock);
				RunReaderContextOperation(
				    *context, *connection, operation_generation, [&](ReaderContextScope &context_scope) {
					    result.resize(NumericCast<idx_t>(requested_size));
					    context_scope.CheckInterrupted();
					    auto read_size =
					        ReadLocked(reinterpret_cast<data_ptr_t>(result.data()), NumericCast<idx_t>(requested_size));
					    if (read_size != requested_size) {
						    throw InternalException(
						        "FILE reader produced fewer bytes than its bounded logical request");
					    }
				    });
			} catch (...) {
				position = initial_position;
				throw;
			}
		}
	}
	return py::bytes(result);
}

int64_t PythonFileReaderHandle::Seek(int64_t offset, int whence) {
	uint64_t result;
	{
		D_ASSERT(py::gil_check());
		py::gil_scoped_release release;
		unique_lock<mutex> reader_guard(lock);
		RequireOpen();
		uint64_t base;
		switch (whence) {
		case 0:
			base = 0;
			break;
		case 1:
			base = position;
			break;
		case 2:
			base = resolved->LogicalSize();
			break;
		default:
			throw py::value_error("invalid whence for VaneFileReader.seek");
		}
		if (base > NumericCast<uint64_t>(std::numeric_limits<int64_t>::max())) {
			throw py::value_error("seek base exceeds signed 64-bit range");
		}
		if (offset < 0) {
			auto magnitude = NumericCast<uint64_t>(-(offset + 1)) + 1;
			if (magnitude > base) {
				throw py::value_error("negative seek position");
			}
			position = base - magnitude;
		} else {
			auto positive_offset = NumericCast<uint64_t>(offset);
			auto maximum_position = NumericCast<uint64_t>(std::numeric_limits<int64_t>::max());
			if (positive_offset > maximum_position - base) {
				throw py::value_error("seek position exceeds signed 64-bit range");
			}
			position = base + positive_offset;
		}
		result = position;
	}
	return NumericCast<int64_t>(result);
}

int64_t PythonFileReaderHandle::Tell() {
	uint64_t result;
	{
		D_ASSERT(py::gil_check());
		py::gil_scoped_release release;
		unique_lock<mutex> reader_guard(lock);
		RequireOpen();
		result = position;
	}
	return NumericCast<int64_t>(result);
}

int64_t PythonFileReaderHandle::Size() {
	uint64_t result;
	{
		D_ASSERT(py::gil_check());
		py::gil_scoped_release release;
		unique_lock<mutex> reader_guard(lock);
		RequireOpen();
		result = resolved->LogicalSize();
	}
	return NumericCast<int64_t>(result);
}

void PythonFileReaderHandle::CheckInterrupted() {
	D_ASSERT(py::gil_check());
	py::gil_scoped_release release;
	unique_lock<mutex> reader_guard(lock);
	RequireOpen();
	unique_lock<mutex> connection_guard(connection->py_connection_lock);
	ReaderContextScope(*context, *connection, interrupt_generation).CheckInterrupted();
}

py::object PythonFileReaderHandle::GuessMimeType() {
	string result;
	bool found;
	D_ASSERT(py::gil_check());
	// Establish the pending-operation boundary before another Python thread can
	// interrupt this connection or this call can wait for the reader mutex.
	auto interrupt_generation = connection ? connection->InterruptGeneration() : 0;
	{
		py::gil_scoped_release release;
		unique_lock<mutex> reader_guard(lock);
		RequireOpen();
		unique_lock<mutex> connection_guard(connection->py_connection_lock);
		RunReaderContextOperation(*context, *connection, interrupt_generation,
		                          [&](ReaderContextScope &) { found = resolved->GuessMimeType(result); });
	}
	return found ? py::cast(std::move(result)) : py::none();
}

void PythonFileReaderHandle::Close() {
	CloseInternal(false);
}

void PythonFileReaderHandle::CloseAndCheckInterrupted() {
	CloseInternal(true);
}

void PythonFileReaderHandle::CloseInternal(bool check_interrupted) {
	// Only one caller owns teardown. Other close calls wait without the GIL for
	// that owner to finish, then observe the completed close. In particular, a
	// later closer must not clear the context needed by a checked close.
	unique_lock<mutex> close_guard(close_lock, std::defer_lock);
	{
		D_ASSERT(py::gil_check());
		py::gil_scoped_release release;
		close_guard.lock();
	}
	const auto first_close = !closed.exchange(true);
	if (!first_close) {
		return;
	}
	unique_lock<mutex> reader_guard(lock, std::defer_lock);
	{
		D_ASSERT(py::gil_check());
		py::gil_scoped_release release;
		reader_guard.lock();
		resolved.reset();
		buffer.clear();
	}
	// Keep the reader lock across GIL reacquisition and release of the retained
	// context/connection. On a checked close, compare against the reader's
	// retained open generation after cleanup. This covers both the gap
	// after the caller's final check and interrupts that race with native handle
	// destruction. The GIL prevents another Python thread from advancing the
	// generation between this check and releasing the retained connection.
	std::exception_ptr close_error;
	if (check_interrupted && context && connection) {
		try {
			ReaderContextScope(*context, *connection, interrupt_generation).CheckInterrupted();
		} catch (...) {
			close_error = std::current_exception();
		}
	}
	context.reset();
	connection.reset();
	if (close_error) {
		std::rethrow_exception(close_error);
	}
}

bool PythonFileReaderHandle::Closed() const {
	return closed.load();
}

string PythonFileReaderHandle::ToString() const {
	return url;
}

string PythonFileReaderHandle::Repr() const {
	return "_VaneFileReaderHandle(url=" + url + ", closed=" + (Closed() ? "True" : "False") + ")";
}

} // namespace duckdb
