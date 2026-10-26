//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/nb/casters.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/detail/nb_list.h>
#include <nanobind/operators.h>
#include <cassert>

// Custom type_caster specializations must be visible in every TU that converts the type (otherwise it is
// UB); keep ALL of them here, in this universally-included umbrella, never in scattered per-feature headers.
#include "duckdb_python/nb/conversions/identifier.hpp"
#include "duckdb_python/nb/conversions/python_udf_type_enum.hpp"
#include "duckdb_python/nb/conversions/null_handling_enum.hpp"
#include "duckdb_python/nb/conversions/exception_handling_enum.hpp"
#include "duckdb_python/nb/conversions/explain_enum.hpp"
#include "duckdb_python/nb/conversions/render_mode_enum.hpp"
#include "duckdb_python/nb/conversions/python_csv_line_terminator_enum.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/assert.hpp"
#include "duckdb/common/helper.hpp"
#include <memory>
#include <type_traits>

// nanobind has no holder-type declaration macros; std::shared_ptr / std::unique_ptr support is
// provided by the <nanobind/stl/shared_ptr.h> / <nanobind/stl/unique_ptr.h> includes above.

// Python interop helpers (raw CPython accessors, guarded isinstance, string coercion, tuple builder, GIL/collection).
#include "duckdb_python/pyutil.hpp"

// Canonical short alias for nanobind, used throughout the bindings.
namespace nb = nanobind;

namespace nanobind {

namespace detail {

// duckdb::vector behaves like a Python list on the boundary; reuse nanobind's list_caster.
template <typename Type, bool SAFE>
struct type_caster<duckdb::vector<Type, SAFE>> : list_caster<duckdb::vector<Type, SAFE>, Type> {};
} // namespace detail
} // namespace nanobind

namespace duckdb {

template <class T, typename... ARGS>
void DefineMethod(std::vector<const char *> aliases, T &mod, ARGS &&...args) {
	for (auto &alias : aliases) {
		mod.def(alias, args...);
	}
}

} // namespace duckdb
