#pragma once
#include "duckdb_python/nb/casters.hpp"
#include "duckdb/common/identifier.hpp"

namespace nanobind {
namespace detail {
template <>
struct type_caster<duckdb::Identifier> {
	NB_TYPE_CASTER(duckdb::Identifier, const_name("str"))

	// Python str -> Identifier
	bool from_python(handle src, uint8_t, cleanup_list *) noexcept {
		if (!PyUnicode_Check(src.ptr())) {
			return false;
		}
		try {
			value = duckdb::Identifier(nanobind::cast<std::string>(src));
		} catch (...) {
			return false;
		}
		return true;
	}

	// Identifier -> Python str
	static handle from_cpp(const duckdb::Identifier &id, rv_policy, cleanup_list *) noexcept {
		auto &str_value = id.GetIdentifierName();
		return PyUnicode_FromStringAndSize(str_value.data(), (Py_ssize_t)str_value.size());
	}
};
} // namespace detail
} // namespace nanobind
