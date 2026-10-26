//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/numpy/numpy_array.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb.hpp"

#include <type_traits>

namespace duckdb {

namespace numpy_internal {

//! Mirror of the leading fields of numpy's `PyArrayObject` (stable ABI across numpy 1.x and 2.x).
//! Reading `data` is a plain struct field access (no Python call, allocation, or GIL). Obtaining
//! the pointer this way, instead of via a `ctypes.data` attribute chain, keeps the numpy columnar
//! path fast for LIST/ARRAY columns, whose per-element converter allocates a fresh array per row.
struct NumpyArrayProxy {
	PyObject_HEAD char *data;
};

//! Borrowed handle to the `numpy.ndarray` type, fetched once under the GIL and intentionally leaked
//! for process lifetime (numpy is never unloaded). Used to gate the data-pointer read: the façade
//! may also wrap non-ndarray objects (e.g. a pandas Index) whose buffer pointer is never read; for
//! those the read must be skipped so a foreign object is never reinterpreted as a numpy array.
inline PyTypeObject *NumpyNdarrayType() {
	static PyTypeObject *cached = []() -> PyTypeObject * {
		nb::object ndarray = nb::module_::import_("numpy").attr("ndarray");
		return reinterpret_cast<PyTypeObject *>(ndarray.release().ptr());
	}();
	return cached;
}

//! Allocate a 1-D numpy array of `count` elements with the given numpy dtype string (e.g. "int64",
//! "float32", "object", "datetime64[us]") via the numpy C API (PyArray_NewFromDescr). Primitive dtypes
//! are left uninitialized (callers fill immediately); object dtype is zero-filled (NULL, read as None). The
//! parsed np.dtype objects are cached to avoid a dtype-string parse on every call. This is hot: a
//! LIST/ARRAY column allocates one array per row. Defined in numpy_array.cpp (the single TU that
//! pulls in the numpy C API). Only ever called on the single-threaded, GIL-held result path.
nb::object NumpyEmpty(idx_t count, const string &dtype);

} // namespace numpy_internal

//! Thin façade over the numpy array representation.
//!
//! This class is the SINGLE place in the codebase that owns the underlying numpy-array
//! object. Under nanobind there is no `nb::array` (and no `nb::dtype`); the array is held
//! as a plain `nb::object` and the few buffer operations go through numpy directly.
//!
//! Performance note: `Data()`/`MutableData()` are on the HOT path. The numpy scan calls `Data()`
//! once per column per 2048-row chunk (see numpy_scan.cpp), and DuckDB drives that scan from
//! multiple threads WITHOUT holding the GIL. It is also on the LIST/ARRAY result path, where a
//! fresh array (and buffer pointer) is materialized per row. The pointer is read directly from the
//! numpy array's C struct (see `numpy_internal::NumpyArrayProxy`): a plain field access, no Python
//! call, allocation, or GIL. We compute it ONCE, eagerly, in the constructor (single-threaded with
//! the GIL held at bind/result time) and cache it; the cache is invalidated (and recomputed) by
//! `Resize()`, the only operation that reallocates the buffer. The struct read is dtype-agnostic
//! (works for the `object` dtype that DLPack/`nb::ndarray` cannot represent).
//!
//! Ownership is move-only: the ctor takes by value and moves, GetArray() hands back a reference, and
//! no method copies the array buffer. Copy is deleted on purpose: two copies would share one numpy
//! object but cache the buffer pointer independently, so a `Resize()` on one (which reallocates and
//! refreshes only its own `cached_data_`) would leave the other's cached pointer dangling. Move
//! transfers array + pointer together and is safe.
class NumpyArray {
public:
	NumpyArray() = default;
	//! Wrap an existing numpy array object (no copy; the object is moved in). The buffer pointer is
	//! computed eagerly here (GIL held) so the hot scan path never makes a Python call.
	explicit NumpyArray(nb::object arr) : array(std::move(arr)) {
		EnsurePointer();
	}

	NumpyArray(NumpyArray &&) = default;
	NumpyArray &operator=(NumpyArray &&) = default;
	NumpyArray(const NumpyArray &) = delete;
	NumpyArray &operator=(const NumpyArray &) = delete;

public:
	//! Allocate a fresh, contiguous 1-D numpy array of `count` elements with the given numpy
	//! dtype string (e.g. "int64", "float32", "object", "datetime64[us]"). Uninitialized; callers
	//! fill it immediately.
	static NumpyArray Allocate(const string &dtype, idx_t count) {
		NumpyArray result(numpy_internal::NumpyEmpty(count, dtype));
		result.length_ = count;
		return result;
	}

	//! Produce a numpy array from an arbitrary Python object (np.asarray semantics: no copy
	//! when `obj` already is an ndarray). The object is moved into the call.
	static NumpyArray FromObject(nb::object obj) {
		auto numpy = nb::module_::import_("numpy");
		return NumpyArray(numpy.attr("asarray")(std::move(obj)));
	}

	//! Read-only pointer to the underlying data buffer (hot path: plain cached read, no GIL).
	const void *Data() const {
		return cached_data_;
	}

	//! Mutable pointer to the underlying data buffer (hot path: plain cached read, no GIL).
	void *MutableData() {
		return cached_data_;
	}

	//! Resize the underlying numpy buffer in place. This REALLOCATES the buffer, so the cached
	//! pointer is invalidated and recomputed (GIL held; only runs on the single-threaded result
	//! path). Resizing to the current length is a genuine no-op in numpy, so we skip the Python
	//! `resize` call entirely in that case. The LIST/ARRAY per-element path allocates each array at
	//! its exact final size, so its `ToArray()` shrink-to-count is always such a no-op: hot, worth
	//! skipping.
	void Resize(idx_t count) {
		if (length_ != DConstants::INVALID_INDEX && count == length_) {
			return;
		}
		array.attr("resize")(count, nb::arg("refcheck") = false);
		length_ = count;
		cached_data_ = nullptr;
		EnsurePointer();
	}

	//! Access the underlying array, e.g. for `.attr(...)` calls, iteration, or to hand it
	//! back to Python. Returned by reference, never copied.
	nb::object &GetArray() {
		return array;
	}
	const nb::object &GetArray() const {
		return array;
	}

private:
	//! Compute and cache the buffer start address of the underlying numpy array, if not already
	//! cached and a numpy ndarray is held. The pointer is read directly from the array's C struct
	//! (dtype-agnostic, works for the `object` dtype too). Only ever called with the GIL held
	//! (construction / Resize).
	void EnsurePointer() {
		// Some NumpyArray wrappers hold non-ndarray objects (e.g. a pandas Index) whose buffer pointer is never read.
		// Gate the read on an actual numpy ndarray so we never reinterpret a foreign object's memory as an array.
		if (!cached_data_ && array.ptr() != nullptr &&
		    PyObject_TypeCheck(array.ptr(), numpy_internal::NumpyNdarrayType())) {
			cached_data_ = reinterpret_cast<numpy_internal::NumpyArrayProxy *>(array.ptr())->data;
		}
	}

	//! The owned numpy array (formerly `nb::array`).
	nb::object array;
	//! Cached buffer start address; see the class-level performance note.
	void *cached_data_ = nullptr;
	//! Known current element count, tracked so `Resize()` can skip a no-op. Set by `Allocate()` and
	//! updated by `Resize()`; `INVALID_INDEX` means "unknown" (arrays wrapped from arbitrary objects),
	//! in which case `Resize()` never skips. The array is only ever resized through `Resize()`, so
	//! this never goes stale.
	idx_t length_ = DConstants::INVALID_INDEX;
};

//! NumpyArray must stay move-only: copying would duplicate the cached raw buffer pointer while sharing
//! one numpy object, so a Resize() on one copy would dangle the other's pointer.
static_assert(!std::is_copy_constructible<NumpyArray>::value && !std::is_copy_assignable<NumpyArray>::value,
              "NumpyArray must remain move-only (see cached_data_ note)");
static_assert(std::is_move_constructible<NumpyArray>::value && std::is_move_assignable<NumpyArray>::value,
              "NumpyArray must remain movable");

} // namespace duckdb
