// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#pragma once

#include "duckdb_python/pybind11/pybind_wrapper.hpp"

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

//! Thread-safe GIL wrapper that keeps a PERSISTENT PyThreadState per thread.
//!
//! Problem: Both pybind11's gil_scoped_acquire AND Python's PyGILState_Ensure
//! create a new PyThreadState on C++ threads and DESTROY it on release when
//! gilstate_counter reaches 0. With many DuckDB pipeline threads cycling the
//! GIL rapidly, the concurrent PyThreadState_New/DeleteCurrent calls corrupt
//! CPython's tstate linked list → SIGSEGV in _PyThreadState_Attach.
//!
//! This also affects third-party libs (e.g. libtorch_python.so) that use
//! pybind11's gil_scoped_acquire on threads we control: pybind11 falls back
//! to PyGILState_GetThisThreadState() when its own TLS is empty (gil.h:79),
//! so if OUR tstate persists, pybind11 will reuse it and never create/delete.
//!
//! Fix: On first GIL acquisition per thread, bump gilstate_counter by +1.
//! This extra +1 ensures the counter NEVER reaches 0, so the tstate is never
//! deleted. The tstate lives for the thread's lifetime (bounded leak, ~few KB
//! per pipeline thread).
struct PythonGILWrapper {
	PythonGILWrapper() : state(PyGILState_Ensure()) {
		// Once per thread: boost gilstate_counter so it never reaches 0.
		// This prevents PyGILState_Release and pybind11's dec_ref from
		// deleting the tstate, eliminating the rapid create/delete race.
		static thread_local bool boosted = BoostCounter();
		(void)boosted;
	}

	~PythonGILWrapper() {
		PyGILState_Release(state);
	}

	PythonGILWrapper(const PythonGILWrapper &) = delete;
	PythonGILWrapper &operator=(const PythonGILWrapper &) = delete;

private:
	PyGILState_STATE state;

	static bool BoostCounter() {
		PyThreadState *tstate = PyGILState_GetThisThreadState();
		if (tstate) {
			tstate->gilstate_counter++;
		}
		return true;
	}
};

} // namespace duckdb
