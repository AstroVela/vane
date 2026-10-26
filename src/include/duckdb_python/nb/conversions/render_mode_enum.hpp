#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/box_renderer.hpp"
#include "duckdb/common/enum_util.hpp"
#include "duckdb_python/nb/conversions/enum_string_caster.hpp"

namespace duckdb {

inline RenderMode RenderModeFromString(const string &value) {
	return EnumUtil::FromString<RenderMode>(value.empty() ? "ROWS" : value);
}

inline RenderMode RenderModeFromInteger(int64_t value) {
	if (value == 0) {
		return RenderMode::ROWS;
	}
	if (value == 1) {
		return RenderMode::COLUMNS;
	}
	throw InvalidInputException("Unrecognized type for 'render_mode'");
}

} // namespace duckdb

//! See enum_string_caster.hpp for the rationale (tri-modal str/int/enum input, umbrella visibility).
DUCKDB_PY_ENUM_STRING_INT_CASTER(duckdb::RenderMode, duckdb::RenderModeFromString, duckdb::RenderModeFromInteger,
                                 "RenderMode")
