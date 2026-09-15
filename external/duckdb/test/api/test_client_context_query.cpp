// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parser/parsed_data/create_collation_info.hpp"
#include "duckdb/parser/tableref.hpp"

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

class MetadataComputationReplacement : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		ScalarFunction function("lower", {LogicalType::VARCHAR}, LogicalType::VARCHAR, ExtensionSchemaFunction,
		                        ExtensionSchemaBind);
		loader.RegisterFunction(std::move(function));
	}
	string Name() override {
		return "metadata_computation_replacement";
	}
};

class MetadataComputationOverload : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		ScalarFunction function("lower", {LogicalType::INTEGER}, LogicalType::VARCHAR, ExtensionSchemaFunction,
		                        ExtensionSchemaBind);
		loader.RegisterFunction(std::move(function));
	}
	string Name() override {
		return "metadata_computation_overload";
	}
};

class MetadataCollationExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		ScalarFunction function("metadata_collation_probe", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
		                        ExtensionSchemaFunction, ExtensionSchemaBind);
		CreateCollationInfo info("metadata_collation_probe", std::move(function), true, false);
		loader.RegisterCollation(info);
	}
	string Name() override {
		return "metadata_collation_extension";
	}
};

struct MetadataReplacementScanData : public ReplacementScanData {
	idx_t calls = 0;
};

unique_ptr<TableRef> MetadataReplacementScan(ClientContext &, ReplacementScanInput &,
                                             optional_ptr<ReplacementScanData> data) {
	data->Cast<MetadataReplacementScanData>().calls++;
	throw InvalidInputException("replacement scan callback invoked");
}

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
	for (auto query : {"SELECT * FROM range(3), client_read_probe()"}) {
		INFO(query);
		REQUIRE_THROWS_WITH(connection.RelationFromQuery(query), Catch::Matchers::Contains("client-context"));
	}
	Connection native_connection(db, "local-fast");
	REQUIRE_THROWS_WITH(native_connection.RelationFromQuery("SELECT * FROM client_read_probe() WHERE true"),
	                    Catch::Matchers::Contains("extension bind callback invoked"));
}

TEST_CASE("Metadata computations do not trust replacement builtins", "[client_context_query]") {
	DuckDB db(nullptr);
	Connection connection(db, "ray");
	auto sql = "SELECT lower(schema_name) FROM duckdb_schemas() WHERE schema_name = 'main'";
	REQUIRE_NOTHROW(connection.RelationFromQuery(sql));
	db.LoadStaticExtension<MetadataComputationReplacement>();
	REQUIRE_THROWS_WITH(connection.RelationFromQuery(sql), Catch::Matchers::Contains("client metadata"));
	connection.BeginTransaction();
	REQUIRE_THROWS_WITH(connection.RelationFromQuery(sql), Catch::Matchers::Contains("client metadata"));
	connection.Rollback();
	Connection native_connection(db, "local-fast");
	REQUIRE_THROWS_WITH(native_connection.RelationFromQuery(sql),
	                    Catch::Matchers::Contains("extension bind callback invoked"));
}

TEST_CASE("Metadata computation admission checks the selected overload", "[client_context_query]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<MetadataComputationOverload>();
	Connection connection(db, "ray");
	REQUIRE_NOTHROW(connection.RelationFromQuery("SELECT lower(schema_name) FROM duckdb_schemas()"));
	for (auto sql : {"SELECT lower(1) FROM duckdb_schemas()", "SELECT lower(1) FROM duckdb_schemas() WHERE false"}) {
		REQUIRE_THROWS_WITH(connection.RelationFromQuery(sql),
		                    Catch::Matchers::Contains("Native client metadata does not support function lower"));
	}
	connection.BeginTransaction();
	REQUIRE_THROWS_WITH(connection.RelationFromQuery("SELECT lower(1) FROM duckdb_schemas()"),
	                    Catch::Matchers::Contains("Native client metadata does not support function lower"));
	connection.Rollback();
	Connection native_connection(db, "local-fast");
	REQUIRE_THROWS_WITH(native_connection.RelationFromQuery("SELECT lower(1) FROM duckdb_schemas()"),
	                    Catch::Matchers::Contains("extension bind callback invoked"));
}

TEST_CASE("Metadata collations are admitted before extension bind callbacks", "[client_context_query]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<MetadataCollationExtension>();
	Connection connection(db, "ray");
	for (auto sql :
	     {"SELECT schema_name COLLATE metadata_collation_probe FROM duckdb_schemas()",
	      "SELECT schema_name FROM duckdb_schemas() ORDER BY schema_name COLLATE metadata_collation_probe"}) {
		REQUIRE_THROWS_WITH(connection.RelationFromQuery(sql),
		                    Catch::Matchers::Contains("does not support function metadata_collation_probe"));
		REQUIRE_FALSE(connection.context->GetClientMetadataBinder());
	}
	// Install the default without invoking this deliberately throwing callback.
	connection.context->config.user_settings.SetUserSetting(DefaultCollationSetting::SettingIndex,
	                                                        Value("metadata_collation_probe"));
	connection.BeginTransaction();
	for (auto sql : {"SELECT schema_name FROM duckdb_schemas() WHERE schema_name = 'main'",
	                 "SELECT min(schema_name) FROM duckdb_schemas()",
	                 "SELECT schema_name FROM duckdb_schemas() GROUP BY schema_name",
	                 "SELECT row_number() OVER (ORDER BY schema_name) FROM duckdb_schemas()"}) {
		REQUIRE_THROWS_WITH(connection.RelationFromQuery(sql),
		                    Catch::Matchers::Contains("does not support function metadata_collation_probe"));
		REQUIRE_FALSE(connection.context->GetClientMetadataBinder());
		REQUIRE_NOTHROW(connection.RelationFromQuery("SELECT current_schema()"));
		REQUIRE_FALSE(connection.context->GetClientMetadataBinder());
	}
	connection.Commit();
	Connection native_connection(db, "local-fast");
	REQUIRE_THROWS_WITH(native_connection.RelationFromQuery(
	                        "SELECT schema_name COLLATE metadata_collation_probe FROM duckdb_schemas()"),
	                    Catch::Matchers::Contains("extension bind callback invoked"));
}

TEST_CASE("Replacement scans are admitted before callbacks in explicit transactions", "[client_context_query]") {
	DBConfig config;
	auto data = make_uniq<MetadataReplacementScanData>();
	auto &probe = *data;
	config.replacement_scans.emplace_back(MetadataReplacementScan, std::move(data));
	DuckDB db(nullptr, &config);
	Connection connection(db, "ray");
	connection.BeginTransaction();
	for (auto sql :
	     {"SELECT * FROM replacement_probe", "SELECT * FROM 'replacement_probe.json'",
	      "SELECT * FROM duckdb_schemas(), replacement_probe", "SELECT * FROM replacement_probe, duckdb_schemas()"}) {
		REQUIRE_THROWS_WITH(connection.RelationFromQuery(sql), Catch::Matchers::Contains("auto-commit"));
		REQUIRE(probe.calls == 0);
		REQUIRE_NOTHROW(connection.RelationFromQuery("SELECT current_schema()"));
	}
	connection.Commit();
	REQUIRE_THROWS_WITH(connection.RelationFromQuery("SELECT * FROM duckdb_schemas(), replacement_probe"),
	                    Catch::Matchers::Contains("Client metadata queries cannot mix"));
	REQUIRE(probe.calls == 0);
	// Ordinary auto-commit queries still reach registered replacement scans.
	REQUIRE_THROWS_WITH(connection.RelationFromQuery("SELECT * FROM replacement_probe"),
	                    Catch::Matchers::Contains("replacement scan callback invoked"));
	REQUIRE(probe.calls == 1);
}
