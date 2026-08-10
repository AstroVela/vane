// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/common/types/type_manager.hpp"
#include "duckdb/function/cast/cast_function_set.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"

namespace duckdb {

CastFunctionSet &TypeManager::GetCastFunctions() {
	return *cast_functions;
}

static string RewriteCanonicalImageTypes(const string &type) {
	string rewritten;
	rewritten.reserve(type.size() + 2);
	bool in_single_quotes = false;
	bool in_double_quotes = false;
	for (idx_t index = 0; index < type.size(); ++index) {
		const auto character = type[index];
		if (in_single_quotes) {
			rewritten.push_back(character);
			if (character == '\'' && index + 1 < type.size() && type[index + 1] == '\'') {
				rewritten.push_back(type[index + 1]);
				++index;
			} else if (character == '\'') {
				in_single_quotes = false;
			}
			continue;
		}
		if (in_double_quotes) {
			rewritten.push_back(character);
			if (character == '"' && index + 1 < type.size() && type[index + 1] == '"') {
				rewritten.push_back(type[index + 1]);
				++index;
			} else if (character == '"') {
				in_double_quotes = false;
			}
			continue;
		}
		if (character == '\'') {
			in_single_quotes = true;
			rewritten.push_back(character);
			continue;
		}
		if (character == '"') {
			in_double_quotes = true;
			rewritten.push_back(character);
			continue;
		}
		const auto token_boundary =
		    index == 0 || (!StringUtil::CharacterIsAlphaNumeric(type[index - 1]) && type[index - 1] != '_');
		if (token_boundary && index + 5 < type.size() && type[index + 5] == '(' &&
		    StringUtil::CIEquals(type.substr(index, 5), "image")) {
			const auto close = type.find(')', index + 6);
			if (close != string::npos) {
				auto mode = type.substr(index + 6, close - index - 6);
				StringUtil::Trim(mode);
				const auto quoted = mode.size() >= 2 && ((mode.front() == '\'' && mode.back() == '\'') ||
				                                         (mode.front() == '"' && mode.back() == '"'));
				auto mode_value = quoted ? mode.substr(1, mode.size() - 2) : mode;
				bool identifier_mode = !mode_value.empty();
				for (const auto mode_character : mode_value) {
					if (!StringUtil::CharacterIsAlphaNumeric(mode_character) && mode_character != '_') {
						identifier_mode = false;
						break;
					}
				}
				if (identifier_mode) {
					ImageType::ParseMode(mode_value);
					if (!quoted) {
						rewritten.append(type, index, 6);
						rewritten.push_back('\'');
						rewritten += mode_value;
						rewritten.append("')");
						index = close;
						continue;
					}
				}
			}
		}
		rewritten.push_back(character);
	}
	return rewritten;
}

static LogicalType TransformStringToUnboundType(const string &str) {
	auto normalized = str;
	StringUtil::Trim(normalized);
	auto normalized_lower = StringUtil::Lower(normalized);
	if (normalized_lower == "null") {
		return LogicalType::SQLNULL;
	}
	auto parser_type = RewriteCanonicalImageTypes(normalized);
	ColumnList column_list;
	try {
		column_list = Parser::ParseColumnList("dummy " + parser_type);
	} catch (const std::runtime_error &e) {
		const vector<string> suggested_types {"BIGINT",
		                                      "INT8",
		                                      "LONG",
		                                      "BIT",
		                                      "BITSTRING",
		                                      "BLOB",
		                                      "BYTEA",
		                                      "BINARY,",
		                                      "VARBINARY",
		                                      "BOOLEAN",
		                                      "BOOL",
		                                      "LOGICAL",
		                                      "DATE",
		                                      "DECIMAL(prec, scale)",
		                                      "DOUBLE",
		                                      "FLOAT8",
		                                      "FLOAT",
		                                      "FLOAT4",
		                                      "REAL",
		                                      "HUGEINT",
		                                      "INTEGER",
		                                      "INT4",
		                                      "INT",
		                                      "SIGNED",
		                                      "INTERVAL",
		                                      "SMALLINT",
		                                      "INT2",
		                                      "SHORT",
		                                      "TIME",
		                                      "TIMESTAMPTZ",
		                                      "TIMESTAMP",
		                                      "DATETIME",
		                                      "TINYINT",
		                                      "INT1",
		                                      "UBIGINT",
		                                      "UHUGEINT",
		                                      "UINTEGER",
		                                      "USMALLINT",
		                                      "UTINYINT",
		                                      "UUID",
		                                      "VARCHAR",
		                                      "CHAR",
		                                      "BPCHAR",
		                                      "TEXT",
		                                      "STRING",
		                                      "MAP(INTEGER, VARCHAR)",
		                                      "UNION(num INTEGER, text VARCHAR)"};
		std::ostringstream error;
		error << "Value \"" << str << "\" can not be converted to a DuckDB Type." << '\n';
		error << "Possible examples as suggestions: " << '\n';
		auto suggestions = StringUtil::TopNJaroWinkler(suggested_types, str);
		for (auto &suggestion : suggestions) {
			error << "* " << suggestion << '\n';
		}
		throw InvalidInputException(error.str());
	}
	return column_list.GetColumn(LogicalIndex(0)).Type();
}

// This has to be called with a level of indirection (through "parse_function") in order to avoid being included in
// extensions that statically link the core DuckDB library.
static LogicalType ParseLogicalTypeInternal(const string &type_str, ClientContext &context) {
	auto type = TransformStringToUnboundType(type_str);
	if (type.IsUnbound()) {
		if (!context.transaction.HasActiveTransaction()) {
			throw InternalException(
			    "Context does not have a transaction active, try running ClientContext::BindLogicalType instead");
		}
		auto binder = Binder::CreateBinder(context, nullptr);
		binder->BindLogicalType(type);
	}
	return type;
}

LogicalType TypeManager::ParseLogicalType(const string &type_str, ClientContext &context) const {
	return parse_function(type_str, context);
}

TypeManager &TypeManager::Get(DatabaseInstance &db) {
	return DBConfig::GetConfig(db).GetTypeManager();
}

TypeManager &TypeManager::Get(ClientContext &context) {
	return DBConfig::GetConfig(context).GetTypeManager();
}

TypeManager::TypeManager(DBConfig &config_p) {
	cast_functions = make_uniq<CastFunctionSet>(config_p);
	parse_function = ParseLogicalTypeInternal;
}

TypeManager::~TypeManager() {
}

} // namespace duckdb
