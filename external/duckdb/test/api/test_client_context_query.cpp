// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

using namespace duckdb;

namespace {

void ExtensionSchemaFunction(DataChunk &, ExpressionState &, Vector &result) {
	result.Reference(Value("extension"));
}

unique_ptr<FunctionData> ExtensionSchemaBind(ClientContext &, ScalarFunction &, vector<unique_ptr<Expression>> &) {
	throw InvalidInputException("extension bind callback invoked");
}

template <bool REPLACE>
class ClientContextOverloadExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		vector<LogicalType> arguments;
		if (!REPLACE) {
			arguments.push_back(LogicalType::INTEGER);
		}
		ScalarFunction function("current_schema", std::move(arguments), LogicalType::VARCHAR, ExtensionSchemaFunction,
		                        ExtensionSchemaBind);
		// Depending on client state does not declare a safe native read.
		function.SetRequiresClientContext();
		loader.RegisterFunction(std::move(function));
	}

	string Name() override {
		return REPLACE ? "replace_client_context_overload" : "add_client_context_overload";
	}
};

} // namespace

TEST_CASE("Native client reads do not inherit eligibility across extension overloads", "[client_context_query]") {
	DuckDB db(nullptr);
	Connection connection(db, "ray");
	REQUIRE_NOTHROW(connection.RelationFromQuery("SELECT current_schema()"));

	string query;
	SECTION("added overload") {
		db.LoadStaticExtension<ClientContextOverloadExtension<false>>();
		query = "SELECT current_schema(1)";
	}
	SECTION("replaced overload") {
		db.LoadStaticExtension<ClientContextOverloadExtension<true>>();
		query = "SELECT current_schema()";
	}

	// Exercise actual lazy relation binding, including the transaction precheck.
	REQUIRE_THROWS_WITH(connection.RelationFromQuery(query), Catch::Matchers::Contains("client-context function"));
	connection.BeginTransaction();
	REQUIRE_THROWS_WITH(connection.RelationFromQuery(query), Catch::Matchers::Contains("auto-commit"));
	connection.Rollback();

	// The overload was really registered: native binding reaches its callback.
	Connection native_connection(db, "local-fast");
	REQUIRE_THROWS_WITH(native_connection.RelationFromQuery(query),
	                    Catch::Matchers::Contains("extension bind callback invoked"));
}
