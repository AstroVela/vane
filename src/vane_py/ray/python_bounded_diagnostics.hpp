// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/execution/distributed/error_diagnostics.hpp"
#include "duckdb/execution/distributed/common_types.hpp"

#include <pybind11/pybind11.h>

#include <exception>
#include <cstring>
#include <string>
#include <frameobject.h>

namespace vane {

// Read canonical CPython exception storage, not provider-defined __str__,
// __repr__, attribute access, or traceback formatting. All Unicode conversion
// happens after slicing and every traceback/cause traversal has a fixed cap.
inline std::string PythonDiagnosticText(PyObject *value, size_t max_bytes, bool retain_tail) {
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
	if (retain_tail) {
		std::string retained(PyBytes_AS_STRING(encoded.ptr()), static_cast<size_t>(PyBytes_GET_SIZE(encoded.ptr())));
		if (size > static_cast<Py_ssize_t>(max_bytes + 1)) {
			auto suffix = pybind11::reinterpret_steal<pybind11::object>(
			    PyUnicode_Substring(value, size - static_cast<Py_ssize_t>(max_bytes + 1), size));
			if (!suffix) {
				throw pybind11::error_already_set();
			}
			auto tail = pybind11::reinterpret_steal<pybind11::object>(
			    PyUnicode_AsEncodedString(suffix.ptr(), "utf-8", "backslashreplace"));
			if (!tail) {
				throw pybind11::error_already_set();
			}
			retained += "...";
			retained.append(PyBytes_AS_STRING(tail.ptr()), static_cast<size_t>(PyBytes_GET_SIZE(tail.ptr())));
		}
		return duckdb::distributed::ErrorDiagnostics::BoundDetailText(retained, max_bytes);
	}
	return duckdb::distributed::BoundDiagnosticText(
	    std::string_view(PyBytes_AS_STRING(encoded.ptr()), static_cast<size_t>(PyBytes_GET_SIZE(encoded.ptr()))),
	    max_bytes);
}

// Inspect canonical dictionaries without hashing or comparing provider keys.
// The caller holds the GIL and consumes the borrowed result without executing
// Python code. Both instance-field and type-metadata reads have a fixed cap.
inline PyObject *PythonDiagnosticDictField(PyObject *dict, const char *name) {
	if (!dict || !PyDict_CheckExact(dict)) {
		return nullptr;
	}
	constexpr size_t MAX_FIELDS = 32;
	const auto name_length = static_cast<Py_ssize_t>(std::strlen(name));
	Py_ssize_t position = 0;
	PyObject *key = nullptr;
	PyObject *field = nullptr;
	for (size_t i = 0; i < MAX_FIELDS && PyDict_Next(dict, &position, &key, &field); i++) {
		if (PyUnicode_CheckExact(key) && PyUnicode_GET_LENGTH(key) == name_length &&
		    PyUnicode_CompareWithASCIIString(key, name) == 0) {
			return field;
		}
	}
	return nullptr;
}

struct PythonRayExceptionMessageSchema {
	const char *type;
	const char *field;
	const char *empty_message = "[no message]";
};

inline const PythonRayExceptionMessageSchema *PythonRayExceptionMessageFields(PyObject *value) {
	// Ray's constructors do not consistently populate BaseException.args.
	// Select the canonical message schema through the stored MRO and module
	// metadata, including inherited Ray actor errors, without attribute lookup.
	static constexpr PythonRayExceptionMessageSchema schemas[] = {
	    // Dynamic RayTaskError subclasses also inherit their cause's type, but
	    // their own canonical storage remains args plus the transported cause.
	    {"RayTaskError", nullptr},
	    {"RayActorError", "error_msg"},
	    {"TaskCancelledError", "error_message", "Task was cancelled."},
	    {"RuntimeEnvSetupError", "error_message", "Failed to set up runtime environment."},
	    {"TaskUnschedulableError", "error_message"},
	    {"ActorUnschedulableError", "error_message"},
	    {"OutOfMemoryError", "message"},
	    {"NodeDiedError", "message"},
	    {"RpcError", "message"}};
	auto *mro = Py_TYPE(value)->tp_mro;
	if (!mro || !PyTuple_CheckExact(mro)) {
		return nullptr;
	}
	constexpr Py_ssize_t MAX_BASES = 32;
	const auto count = std::min(PyTuple_GET_SIZE(mro), MAX_BASES);
	for (Py_ssize_t i = 0; i < count; i++) {
		auto *type = reinterpret_cast<PyTypeObject *>(PyTuple_GET_ITEM(mro, i));
		for (const auto &schema : schemas) {
			if (std::strcmp(type->tp_name, schema.type) != 0) {
				continue;
			}
			auto *module = PythonDiagnosticDictField(type->tp_dict, "__module__");
			if (module && PyUnicode_CheckExact(module) && PyUnicode_GET_LENGTH(module) == 14 &&
			    PyUnicode_CompareWithASCIIString(module, "ray.exceptions") == 0) {
				return schema.field ? &schema : nullptr;
			}
		}
	}
	return nullptr;
}

inline std::string PythonExceptionMessage(PyObject *value, size_t max_bytes) {
	if (const auto *schema = PythonRayExceptionMessageFields(value)) {
		auto *message =
		    PythonDiagnosticDictField(reinterpret_cast<PyBaseExceptionObject *>(value)->dict, schema->field);
		// Actor death text can put the reason after actor metadata or a remote
		// traceback. Retain both ends within the primary-message budget.
		return message && PyUnicode_Check(message) && PyUnicode_GET_LENGTH(message)
		           ? PythonDiagnosticText(message, max_bytes, true)
		           : duckdb::distributed::BoundDiagnosticText(schema->empty_message, max_bytes);
	}
	auto *args = reinterpret_cast<PyBaseExceptionObject *>(value)->args;
	if (!args || !PyTuple_CheckExact(args)) {
		return "[no message]";
	}
	std::string message;
	const auto count = PyTuple_GET_SIZE(args);
	const auto retained = std::min(count, static_cast<Py_ssize_t>(4));
	// Sample every retained argument before bounding the combined message. A
	// long earlier argument must not hide the reason in a later one. The fixed
	// argument count and bounded conversions also bound this temporary buffer.
	for (Py_ssize_t i = 0; i < retained; i++) {
		if (i) {
			message += ", ";
		}
		auto *arg = PyTuple_GET_ITEM(args, i);
		if (PyUnicode_Check(arg)) {
			message += PythonDiagnosticText(arg, max_bytes, true);
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
	return duckdb::distributed::ErrorDiagnostics::BoundDetailText(message, max_bytes);
}

inline pybind11::object PythonTransportedExceptionCause(PyObject *value) {
	auto *cause = PythonDiagnosticDictField(reinterpret_cast<PyBaseExceptionObject *>(value)->dict, "cause");
	if (cause && PyExceptionInstance_Check(cause)) {
		return pybind11::reinterpret_borrow<pybind11::object>(cause);
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
		traceback += PythonDiagnosticText(frame_code->co_filename, 128, false) + ":" +
		             std::to_string(PyFrame_GetLineNumber(tb->tb_frame)) + " in " +
		             PythonDiagnosticText(frame_code->co_name, 64, false) + "\n";
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
	// Borrow the native what() text. Finding its end requires a scan of the
	// NUL-terminated buffer, but only bounded normalized edges are copied.
	// A prefix limiter here would discard the reason before aggregation.
	const auto *message = error.what();
	return duckdb::distributed::ErrorDiagnostics::FromText(message ? std::string_view(message) : "unknown error");
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
	return PythonDiagnosticText(value.ptr(), duckdb::distributed::ErrorDiagnostics::MAX_DETAIL_BYTES, true);
}

} // namespace vane
