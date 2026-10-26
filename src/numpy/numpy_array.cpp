//===----------------------------------------------------------------------===//
//                         DuckDB
//
// numpy_array.cpp
//
// Out-of-line definitions for the NumpyArray facade (numpy_array.hpp). This is the
// ONLY translation unit that uses the numpy C API, so it does not need
// PY_ARRAY_UNIQUE_SYMBOL / NO_IMPORT_ARRAY (those coordinate the C-API function
// pointer table across multiple TUs).
//===----------------------------------------------------------------------===//

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "duckdb_python/numpy/numpy_array.hpp"

#include <stdexcept>
#include <unordered_map>

namespace duckdb {
namespace numpy_internal {

namespace {

//! Lazy, guarded one-time init of the numpy C-API function pointer table. numpy is always
//! already imported by the time we allocate a result array, so import_array should succeed;
//! if it does not, the returned value is false and the caller raises. Runs exactly once
//! (function-local static initializer, GIL held on the result path).
bool EnsureNumpyCApi() {
	static bool ok = []() -> bool {
		// import_array1(ret) expands to `return ret;` on failure, so wrap it in a lambda that
		// returns int and surface success via the return value.
		auto do_import = []() -> int {
			import_array1(-1);
			return 0;
		};
		return do_import() == 0;
	}();
	return ok;
}

} // namespace

nb::object NumpyEmpty(idx_t count, const string &dtype) {
	// Process-lifetime cache of parsed np.dtype objects, keyed by dtype string. The parse is
	// otherwise repeated per call; a LIST/ARRAY column allocates one array per row. Leaked on
	// purpose (numpy is never unloaded; no Python destructor runs after finalization). Only ever
	// touched on the single-threaded, GIL-held result path.
	static auto &dtype_cache = *new std::unordered_map<string, PyObject *>();
	PyObject *&descr = dtype_cache[dtype];
	if (!descr) {
		nb::object d = nb::module_::import_("numpy").attr("dtype")(dtype);
		descr = d.release().ptr();
	}

	if (!EnsureNumpyCApi()) {
		throw std::runtime_error("Failed to initialize the numpy C API (import_array failed)");
	}

	npy_intp dims[1] = {static_cast<npy_intp>(count)};
	// PyArray_NewFromDescr STEALS a reference to descr UNCONDITIONALLY for a non-NULL descr, including on
	// failure: numpy releases the reference either explicitly on an early-validation failure or via the
	// array's dealloc on its `fail:` path (see numpy _core/src/multiarray/ctors.c; the only non-stealing
	// path is descr == NULL, which never happens here). descr is a single cached np.dtype reused across
	// every allocation, so hand the call its own reference to consume.
	//
	// We use PyArray_NewFromDescr rather than PyArray_Empty: PyArray_Empty fills object-dtype arrays with
	// incref'd Py_None (PyArray_FillObjectArray), which the array_wrapper store path then overwrites
	// without a decref, leaking one Py_None ref per cell. NewFromDescr zero-fills object arrays instead
	// (object dtype is NPY_NEEDS_INIT, so numpy memsets the buffer to NULL), which numpy reads back as
	// None and array_wrapper overwrites cleanly. Non-object dtypes are left uninitialized either way
	// (callers fill immediately), and skipping the Py_None fill is if anything cheaper on the hot
	// LIST/ARRAY result path.
	Py_INCREF(descr);
	PyObject *arr = PyArray_NewFromDescr(&PyArray_Type, reinterpret_cast<PyArray_Descr *>(descr), 1, dims,
	                                     nullptr /* strides: C-contiguous */, nullptr /* data: numpy allocates */,
	                                     0 /* flags: C order */, nullptr /* obj */);
	if (!arr) {
		// The steal has already balanced the Py_INCREF above (it happens even on failure), so we must NOT
		// decref again: an extra decref would drop the cache's own reference and, once freed, leave
		// dtype_cache holding a dangling pointer -> use-after-free on the next allocation of this dtype.
		throw nb::python_error();
	}
	// PyArray_NewFromDescr returns a NEW reference; hand ownership to nanobind via steal.
	return nb::steal<nb::object>(arr);
}

} // namespace numpy_internal
} // namespace duckdb
