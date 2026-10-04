// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#pragma once

#include "vane_python/pybind11/pybind_wrapper.hpp"

#if PY_VERSION_HEX < 0x030D0000 && !defined(Py_LIMITED_API)
extern "C" int _Py_IsFinalizing(void);
#endif

namespace duckdb {

inline bool PythonIsFinalizing() {
#if PY_VERSION_HEX >= 0x030D0000
#if !defined(Py_LIMITED_API) || Py_LIMITED_API + 0 >= 0x030D0000
	return Py_IsFinalizing();
#else
	return false;
#endif
#elif !defined(Py_LIMITED_API)
	return _Py_IsFinalizing();
#else
	return false;
#endif
}

//! Reuse a Python thread state across GIL acquisitions on a native thread.
//! The extra GIL-state reference also lets pybind11 reuse that state, but must
//! be released before the OS thread exits. Otherwise faulthandler can follow
//! a retained state to an invalid pthread handle when reading its thread name.
//! Python-created threads already own their state; do not retain or delete it.
struct PythonGILWrapper {
	PythonGILWrapper() : state(Acquire()) {
	}

	~PythonGILWrapper() {
		PyGILState_Release(state);
	}

	PythonGILWrapper(const PythonGILWrapper &) = delete;
	PythonGILWrapper &operator=(const PythonGILWrapper &) = delete;

private:
	PyGILState_STATE state;

	struct NativeThreadState {
		PyThreadState *owned = nullptr;

		~NativeThreadState() {
			// Interpreter shutdown owns any remaining states. In particular,
			// PyGILState_Ensure cannot be called once finalization has started.
			if (!owned || !Py_IsInitialized() || PythonIsFinalizing() || PyGILState_GetThisThreadState() != owned) {
				return;
			}
			auto gil = PyGILState_Ensure();
			owned->gilstate_counter--;
			owned = nullptr;
			// Release the temporary acquisition and the retained reference
			// together, allowing CPython to clear and delete its native state.
			PyGILState_Release(gil);
		}
	};

	static PyGILState_STATE Acquire() {
		static thread_local NativeThreadState thread_state;
		auto existing = PyGILState_GetThisThreadState();
		auto gil = PyGILState_Ensure();
		if (!existing) {
			thread_state.owned = PyGILState_GetThisThreadState();
			thread_state.owned->gilstate_counter++;
		}
		return gil;
	}
};

} // namespace duckdb
