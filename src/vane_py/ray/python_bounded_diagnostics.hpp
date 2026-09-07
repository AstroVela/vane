// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/execution/distributed/error_diagnostics.hpp"
#include "duckdb/execution/distributed/common_types.hpp"

#include <pybind11/pybind11.h>

#include <exception>
#include <string>
#include <frameobject.h>

namespace vane {

// Read canonical CPython exception storage, not provider-defined __str__,
// __repr__, attribute access, or traceback formatting. All Unicode conversion
// happens after slicing and every traceback/cause traversal has a fixed cap.
inline std::string PythonDiagnosticPrefix(PyObject *value, size_t max_bytes) {
	const auto size = PyUnicode_GetLength(value);
	if (size < 0) {
		throw pybind11::error_already_set();
	}
	auto prefix = pybind11::reinterpret_steal<pybind11::object>(
	    PyUnicode_Substring(value, 0, std::min(size, static_cast<Py_ssize_t>(max_bytes + 1))));
	if (!prefix) {
		throw pybind11::error_already_set();
	}
	auto encoded = pybind11::reinterpret_steal<pybind11::object>(
	    PyUnicode_AsEncodedString(prefix.ptr(), "utf-8", "backslashreplace"));
	if (!encoded) {
		throw pybind11::error_already_set();
	}
	return duckdb::distributed::BoundDiagnosticText(
	    std::string_view(PyBytes_AS_STRING(encoded.ptr()), static_cast<size_t>(PyBytes_GET_SIZE(encoded.ptr()))),
	    max_bytes);
}

inline std::string PythonExceptionMessage(PyObject *value, size_t max_bytes) {
	auto *args = reinterpret_cast<PyBaseExceptionObject *>(value)->args;
	if (!args || !PyTuple_CheckExact(args)) {
		return "[no message]";
	}
	std::string message;
	const auto count = PyTuple_GET_SIZE(args);
	const auto retained = std::min(count, static_cast<Py_ssize_t>(4));
	for (Py_ssize_t i = 0; i < retained && message.size() < max_bytes; i++) {
		if (i) {
			message += ", ";
		}
		auto *arg = PyTuple_GET_ITEM(args, i);
		if (PyUnicode_Check(arg)) {
			message += PythonDiagnosticPrefix(arg, max_bytes);
		} else if (PyLong_CheckExact(arg)) {
			int overflow = 0;
			const auto number = PyLong_AsLongLongAndOverflow(arg, &overflow);
			message += overflow ? "<int>" : std::to_string(number);
		} else {
			message += "<" + duckdb::distributed::BoundDiagnosticCString(Py_TYPE(arg)->tp_name, 64) + ">";
		}
	}
	if (retained < count) {
		message += " [additional arguments omitted]";
	}
	return duckdb::distributed::BoundDiagnosticText(message, max_bytes);
}

inline pybind11::object PythonTransportedExceptionCause(PyObject *value) {
	auto *dict = reinterpret_cast<PyBaseExceptionObject *>(value)->dict;
	if (!dict || !PyDict_CheckExact(dict)) {
		return {};
	}
	// RayTaskError stores its transported exception in its instance dictionary.
	// A hash lookup can invoke a custom key's __eq__; inspect a bounded number
	// of canonical string keys directly instead.
	constexpr size_t MAX_TRANSPORT_FIELDS = 32;
	Py_ssize_t position = 0;
	PyObject *key = nullptr;
	PyObject *field = nullptr;
	for (size_t i = 0; i < MAX_TRANSPORT_FIELDS && PyDict_Next(dict, &position, &key, &field); i++) {
		if (PyUnicode_CheckExact(key) && PyUnicode_GET_LENGTH(key) == 5 &&
		    PyUnicode_CompareWithASCIIString(key, "cause") == 0 && PyExceptionInstance_Check(field)) {
			return pybind11::reinterpret_borrow<pybind11::object>(field);
		}
	}
	return {};
}

inline duckdb::distributed::ErrorDiagnostics CapturePythonError(const pybind11::error_already_set &error) {
	using duckdb::distributed::BoundDiagnosticCString;
	using duckdb::distributed::BoundDiagnosticText;
	using duckdb::distributed::ErrorDiagnostic;
	using duckdb::distributed::ErrorDiagnostics;
	pybind11::gil_scoped_acquire gil;
	auto *value = error.value().ptr();
	auto type = BoundDiagnosticCString(Py_TYPE(value)->tp_name, ErrorDiagnostic::MAX_TYPE_BYTES);
	auto message = PythonExceptionMessage(value, ErrorDiagnostic::MAX_MESSAGE_BYTES);

	std::string traceback;
	auto trace = error.trace();
	auto *tb = trace && !trace.is_none() ? reinterpret_cast<PyTracebackObject *>(trace.ptr()) : nullptr;
	size_t frames = 0;
	while (tb && frames < ErrorDiagnostic::MAX_TRACEBACK_FRAMES &&
	       traceback.size() < ErrorDiagnostic::MAX_TRACEBACK_BYTES) {
		auto code =
		    pybind11::reinterpret_steal<pybind11::object>(reinterpret_cast<PyObject *>(PyFrame_GetCode(tb->tb_frame)));
		if (!code) {
			throw pybind11::error_already_set();
		}
		auto *frame_code = reinterpret_cast<PyCodeObject *>(code.ptr());
		traceback += PythonDiagnosticPrefix(frame_code->co_filename, 128) + ":" +
		             std::to_string(PyFrame_GetLineNumber(tb->tb_frame)) + " in " +
		             PythonDiagnosticPrefix(frame_code->co_name, 64) + "\n";
		tb = tb->tb_next;
		frames++;
	}
	if (tb) {
		traceback =
		    BoundDiagnosticText(traceback, ErrorDiagnostic::MAX_TRACEBACK_BYTES - 24) + "\n[traceback truncated]";
	}

	std::string causes;
	std::vector<pybind11::object> seen {error.value()};
	auto current = pybind11::reinterpret_borrow<pybind11::object>(value);
	for (size_t i = 0; i <= ErrorDiagnostic::MAX_CAUSES; i++) {
		auto cause = pybind11::reinterpret_steal<pybind11::object>(PyException_GetCause(current.ptr()));
		const char *relationship = "caused by ";
		if (!cause) {
			cause = PythonTransportedExceptionCause(current.ptr());
		}
		if (!cause && !reinterpret_cast<PyBaseExceptionObject *>(current.ptr())->suppress_context) {
			cause = pybind11::reinterpret_steal<pybind11::object>(PyException_GetContext(current.ptr()));
			relationship = "during handling of ";
		}
		if (!cause) {
			break;
		}
		if (std::any_of(seen.begin(), seen.end(),
		                [&](const pybind11::object &previous) { return previous.ptr() == cause.ptr(); })) {
			causes += " [exception chain cycle]";
			break;
		}
		if (i == ErrorDiagnostic::MAX_CAUSES) {
			causes += " [exception chain limit]";
			break;
		}
		seen.push_back(cause);
		if (!causes.empty()) {
			causes += "; ";
		}
		causes += relationship + BoundDiagnosticCString(Py_TYPE(cause.ptr())->tp_name, 32) + ": " +
		          PythonExceptionMessage(cause.ptr(), 64);
		current = std::move(cause);
	}
	return ErrorDiagnostics::FromDiagnostic(ErrorDiagnostic(type, message, traceback, causes));
}

inline duckdb::distributed::ErrorDiagnostics CaptureError(const duckdb::distributed::ErrorDiagnostics &error) {
	return error;
}

inline duckdb::distributed::ErrorDiagnostics CaptureError(std::string_view message) {
	return duckdb::distributed::ErrorDiagnostics::FromText(message);
}

inline duckdb::distributed::ErrorDiagnostics CaptureError(const std::exception &error) {
	if (const auto *python_error = dynamic_cast<const pybind11::error_already_set *>(&error)) {
		return CapturePythonError(*python_error);
	}
	if (const auto *native_error = dynamic_cast<const duckdb::distributed::DuckDBError *>(&error)) {
		return native_error->Diagnostics();
	}
	return duckdb::distributed::ErrorDiagnostics::FromText(duckdb::distributed::BoundDiagnosticCString(
	    error.what(), duckdb::distributed::ErrorDiagnostic::MAX_MESSAGE_BYTES));
}

inline duckdb::distributed::ErrorDiagnostics CaptureError(const std::exception_ptr &error) {
	try {
		std::rethrow_exception(error);
	} catch (const std::exception &caught) {
		return CaptureError(caught);
	} catch (...) {
		return duckdb::distributed::ErrorDiagnostics::FromText("unknown exception");
	}
}

inline std::string BoundedPythonDiagnosticText(const pybind11::object &value) {
	const auto character_count = PyUnicode_GetLength(value.ptr());
	if (character_count < 0) {
		throw pybind11::error_already_set();
	}
	const auto max_bytes = duckdb::distributed::ErrorDiagnostics::MAX_DETAIL_BYTES;
	if (static_cast<size_t>(character_count) <= max_bytes) {
		return duckdb::distributed::ErrorDiagnostics::BoundDetailText(value.cast<std::string>());
	}

	// Avoid copying an untrusted Python string in full. Each retained Unicode
	// slice is capped by a constant number of code points (and therefore at
	// most four times that many UTF-8 bytes), after which the shared byte
	// limiter preserves useful context from both ends.
	const auto edge_characters = static_cast<Py_ssize_t>(max_bytes / 2);
	auto prefix = pybind11::reinterpret_steal<pybind11::object>(PyUnicode_Substring(value.ptr(), 0, edge_characters));
	if (!prefix) {
		throw pybind11::error_already_set();
	}
	auto suffix = pybind11::reinterpret_steal<pybind11::object>(
	    PyUnicode_Substring(value.ptr(), character_count - edge_characters, character_count));
	if (!suffix) {
		throw pybind11::error_already_set();
	}
	auto retained = prefix.cast<std::string>() + "..." + suffix.cast<std::string>();
	return duckdb::distributed::ErrorDiagnostics::BoundDetailText(retained);
}

} // namespace vane
