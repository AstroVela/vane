// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

using namespace duckdb;

namespace {

void ExtensionSchemaFunction(DataChunk &, ExpressionState &, Vector &result) {
	result.Reference(Value("extension"));
}

unique_ptr<FunctionData> ExtensionSchemaBind(ClientContext &, ScalarFunction &, vector<unique_ptr<Expression>> &) {
	throw InvalidInputException("extension bind callback invoked");
}

unique_ptr<FunctionData> ExtensionSchemasBind(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &,
                                              vector<string> &) {
	throw InvalidInputException("extension bind callback invoked");
}

template <bool REPLACE, bool VARARGS = false>
class ClientContextOverloadExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		vector<LogicalType> arguments;
		if (!REPLACE) {
			arguments.push_back(LogicalType::INTEGER);
		}
		ScalarFunction function("current_schema", std::move(arguments), LogicalType::VARCHAR, ExtensionSchemaFunction,
		                        ExtensionSchemaBind);
		TableFunction table_function("duckdb_schemas", function.arguments, nullptr, ExtensionSchemasBind);
		if (VARARGS) {
			function.varargs = LogicalType::INTEGER;
			table_function.varargs = LogicalType::INTEGER;
		}
		// Depending on client state does not declare a safe native read.
		function.SetRequiresClientContext();
		table_function.SetRequiresClientContext();
		loader.RegisterFunction(std::move(function));
		if (!REPLACE) {
			// Native table-function registration adds overloads but cannot replace them.
			loader.RegisterFunction(std::move(table_function));
		}
	}

	string Name() override {
		return REPLACE ? "replace_client_context_overload" : "add_client_context_overload";
	}
};

class ClientReadTableExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		TableFunction function("client_read_probe", {}, nullptr, ExtensionSchemasBind);
		// Declaring a native client read must also protect its derived uses.
		function.SetClientContextRead();
		loader.RegisterFunction(std::move(function));
	}

	string Name() override {
		return "client_read_table";
	}
};

} // namespace

TEST_CASE("Native client reads do not inherit eligibility across extension overloads", "[client_context_query]") {
	DuckDB db(nullptr);
	Connection connection(db, "ray");
	REQUIRE_NOTHROW(connection.RelationFromQuery("SELECT current_schema()"));

	string arguments;
	bool replaced = false;
	SECTION("added overload") {
		db.LoadStaticExtension<ClientContextOverloadExtension<false>>();
		arguments = "(1)";
	}
	SECTION("added variadic overload") {
		db.LoadStaticExtension<ClientContextOverloadExtension<false, true>>();
		arguments = "(1, 2)";
	}
	SECTION("replaced overload") {
		db.LoadStaticExtension<ClientContextOverloadExtension<true>>();
		arguments = "()";
		replaced = true;
	}
	for (bool metadata : {false, true}) {
		if (replaced && metadata) {
			continue;
		}
		string query_prefix = metadata ? "SELECT * FROM duckdb_schemas" : "SELECT current_schema";
		auto native_query = query_prefix + "()";
		auto query = query_prefix + arguments;
		INFO(query);
		if (!replaced) {
			REQUIRE_NOTHROW(connection.RelationFromQuery(native_query));
			REQUIRE_FALSE(connection.Query(native_query)->HasError());
		}

		// Exercise actual lazy relation binding, including the transaction precheck.
		REQUIRE_THROWS_WITH(connection.RelationFromQuery(query), Catch::Matchers::Contains("client-context"));
		connection.BeginTransaction();
		if (!replaced) {
			REQUIRE_NOTHROW(connection.RelationFromQuery(native_query));
			REQUIRE_FALSE(connection.Query(native_query)->HasError());
		}
		REQUIRE_THROWS_WITH(connection.RelationFromQuery(query), Catch::Matchers::Contains("auto-commit"));
		connection.Rollback();

		// The overload was really registered: native binding reaches its callback.
		Connection native_connection(db, "local-fast");
		REQUIRE_THROWS_WITH(native_connection.RelationFromQuery(query),
		                    Catch::Matchers::Contains("extension bind callback invoked"));
	}
}

TEST_CASE("Declaring a native table read protects derived runner queries", "[client_context_query]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<ClientReadTableExtension>();
	Connection connection(db, "ray");
	// The explicit native-read capability admits the direct call to its binder.
	REQUIRE_THROWS_WITH(connection.RelationFromQuery("SELECT * FROM client_read_probe()"),
	                    Catch::Matchers::Contains("extension bind callback invoked"));
	for (auto query : {"SELECT * FROM client_read_probe() WHERE true", "SELECT * FROM client_read_probe(), range(3)"}) {
		INFO(query);
		REQUIRE_THROWS_WITH(connection.RelationFromQuery(query), Catch::Matchers::Contains("client-context"));
	}
	Connection native_connection(db, "local-fast");
	REQUIRE_THROWS_WITH(native_connection.RelationFromQuery("SELECT * FROM client_read_probe() WHERE true"),
	                    Catch::Matchers::Contains("extension bind callback invoked"));
}
