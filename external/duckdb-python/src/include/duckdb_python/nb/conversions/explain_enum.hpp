#pragma once

#include "duckdb/parser/statement/explain_statement.hpp"
#include "duckdb/common/common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb_python/nb/conversions/enum_string_caster.hpp"

namespace duckdb {

inline ExplainType ExplainTypeFromString(const string &type) {
	auto ltype = StringUtil::Lower(type);
	if (ltype.empty() || ltype == "standard") {
		return ExplainType::EXPLAIN_STANDARD;
	}
	if (ltype == "analyze") {
		return ExplainType::EXPLAIN_ANALYZE;
	}
	throw InvalidInputException("Unrecognized type for 'explain'");
}

inline ExplainType ExplainTypeFromInteger(int64_t value) {
	if (value == 0) {
		return ExplainType::EXPLAIN_STANDARD;
	}
	if (value == 1) {
		return ExplainType::EXPLAIN_ANALYZE;
	}
	throw InvalidInputException("Unrecognized type for 'explain'");
}

} // namespace duckdb

//! See enum_string_caster.hpp for the rationale (tri-modal str/int/enum input, umbrella visibility).
DUCKDB_PY_ENUM_STRING_INT_CASTER(duckdb::ExplainType, duckdb::ExplainTypeFromString, duckdb::ExplainTypeFromInteger,
                                 "ExplainType")
