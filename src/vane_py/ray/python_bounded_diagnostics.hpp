// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bounded_diagnostics.hpp"

#include <pybind11/pybind11.h>

#include <string>

namespace vane {

inline std::string BoundedPythonDiagnosticText(const pybind11::object &value) {
	const auto character_count = PyUnicode_GetLength(value.ptr());
	if (character_count < 0) {
		throw pybind11::error_already_set();
	}
	const auto max_bytes = BoundedErrorDetails::MAX_DETAIL_BYTES;
	if (static_cast<size_t>(character_count) <= max_bytes) {
		return BoundedErrorDetails::BoundDetailText(value.cast<std::string>());
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
	return BoundedErrorDetails::BoundDetailText(retained);
}

} // namespace vane
