// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/parser/parsed_data/create_collation_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/tableref/column_data_ref.hpp"
#include "duckdb/parser/tableref.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/function/cast/cast_function_set.hpp"

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
	// Once both source kinds are known, reject before this source's callback.
	REQUIRE_THROWS_WITH(connection.RelationFromQuery("SELECT * FROM range(3), client_read_probe()"),
	                    Catch::Matchers::Contains("Client metadata queries cannot mix"));
	Connection native_connection(db, "local-fast");
	REQUIRE_THROWS_WITH(native_connection.RelationFromQuery("SELECT * FROM client_read_probe() WHERE true"),
	                    Catch::Matchers::Contains("extension bind callback invoked"));
}

TEST_CASE("Metadata expressions use native replacement builtins", "[client_context_query]") {
	DuckDB db(nullptr);
	Connection connection(db, "ray");
	auto sql = "SELECT lower(schema_name) FROM duckdb_schemas() WHERE schema_name = 'main'";
	REQUIRE_NOTHROW(connection.RelationFromQuery(sql));
	db.LoadStaticExtension<MetadataComputationReplacement>();
	REQUIRE_THROWS_WITH(connection.RelationFromQuery(sql),
	                    Catch::Matchers::Contains("extension bind callback invoked"));
	connection.BeginTransaction();
	REQUIRE_THROWS_WITH(connection.RelationFromQuery(sql),
	                    Catch::Matchers::Contains("extension bind callback invoked"));
	connection.Rollback();
	Connection native_connection(db, "local-fast");
	REQUIRE_THROWS_WITH(native_connection.RelationFromQuery(sql),
	                    Catch::Matchers::Contains("extension bind callback invoked"));
}

TEST_CASE("Metadata expressions use native overload selection", "[client_context_query]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<MetadataComputationOverload>();
	Connection connection(db, "ray");
	REQUIRE_NOTHROW(connection.RelationFromQuery("SELECT lower(schema_name) FROM duckdb_schemas()"));
	for (auto sql : {"SELECT lower(1) FROM duckdb_schemas()", "SELECT lower(1) FROM duckdb_schemas() WHERE false"}) {
		REQUIRE_THROWS_WITH(connection.RelationFromQuery(sql),
		                    Catch::Matchers::Contains("extension bind callback invoked"));
	}
	connection.BeginTransaction();
	REQUIRE_THROWS_WITH(connection.RelationFromQuery("SELECT lower(1) FROM duckdb_schemas()"),
	                    Catch::Matchers::Contains("extension bind callback invoked"));
	connection.Rollback();
	Connection native_connection(db, "local-fast");
	REQUIRE_THROWS_WITH(native_connection.RelationFromQuery("SELECT lower(1) FROM duckdb_schemas()"),
	                    Catch::Matchers::Contains("extension bind callback invoked"));
}

TEST_CASE("Metadata expressions use native collation and cast callbacks", "[client_context_query]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<MetadataCollationExtension>();
	Connection connection(db, "ray");
	for (auto sql :
	     {"SELECT schema_name COLLATE metadata_collation_probe FROM duckdb_schemas()",
	      "SELECT schema_name FROM duckdb_schemas() ORDER BY schema_name COLLATE metadata_collation_probe"}) {
		REQUIRE_THROWS_WITH(connection.RelationFromQuery(sql),
		                    Catch::Matchers::Contains("extension bind callback invoked"));
	}
	CastFunctionSet::Get(*connection.context)
	    .RegisterCastFunction(LogicalType::VARCHAR, LogicalType::INTEGER,
	                          [](BindCastInput &, const LogicalType &, const LogicalType &) -> BoundCastInfo {
		                          throw InvalidInputException("extension cast callback invoked");
	                          });
	REQUIRE_THROWS_WITH(connection.RelationFromQuery("SELECT CAST(schema_name AS INTEGER) FROM duckdb_schemas()"),
	                    Catch::Matchers::Contains("extension cast callback invoked"));
}

TEST_CASE("Metadata expressions execute native extension functions", "[client_context_query]") {
	DuckDB db(nullptr);
	Connection connection(db, "ray");
	ScalarFunction function("metadata_scalar_probe", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                        ExtensionSchemaFunction);
	CreateScalarFunctionInfo info(std::move(function));
	connection.context->RegisterFunction(info);
	PendingQueryParameters parameters;
	parameters.bound_plan_handler = [](Planner &, unique_ptr<LogicalOperator> &, PreparedStatementData &prepared) {
		REQUIRE(prepared.native_client_query);
		return false;
	};
	auto pending =
	    connection.PendingQuery("SELECT metadata_scalar_probe(schema_name) FROM duckdb_schemas() LIMIT 1", parameters);
	REQUIRE(pending);
	REQUIRE_FALSE(pending->HasError());
	auto result = pending->Execute();
	REQUIRE_FALSE(result->HasError());
	auto chunk = result->Fetch();
	REQUIRE(chunk);
	REQUIRE(chunk->size() == 1);
	REQUIRE(chunk->GetValue(0, 0).ToString() == "extension");
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
	// Ordinary auto-commit queries still reach registered replacement scans.
	REQUIRE_THROWS_WITH(connection.RelationFromQuery("SELECT * FROM replacement_probe"),
	                    Catch::Matchers::Contains("replacement scan callback invoked"));
	REQUIRE(probe.calls == 1);
}

TEST_CASE("Query replacements retain materialized data dependencies", "[client_context_query]") {
	DuckDB db(nullptr);
	Connection connection(db, "ray");
	TableFunction function("materialized_query_probe", {}, nullptr, nullptr);
	function.bind_replace = [](ClientContext &context, TableFunctionBindInput &) -> unique_ptr<TableRef> {
		auto collection = make_uniq<ColumnDataCollection>(context, vector<LogicalType> {LogicalType::INTEGER});
		auto ref = make_uniq<ColumnDataRef>(std::move(collection), vector<string> {"value"});
		ref->alias = "materialized";
		return std::move(ref);
	};
	CreateTableFunctionInfo info(std::move(function));
	connection.context->RegisterFunction(info);
	for (auto sql : {"SELECT * FROM materialized_query_probe(), duckdb_schemas()",
	                 "SELECT * FROM duckdb_schemas(), materialized_query_probe()"}) {
		REQUIRE_THROWS_WITH(connection.RelationFromQuery(sql),
		                    Catch::Matchers::Contains("Client metadata queries cannot mix"));
	}
	connection.BeginTransaction();
	REQUIRE_THROWS_WITH(connection.RelationFromQuery("SELECT * FROM materialized_query_probe()"),
	                    Catch::Matchers::Contains("auto-commit"));
	connection.Commit();
	Connection native_connection(db, "local-fast");
	REQUIRE_NOTHROW(native_connection.RelationFromQuery("SELECT * FROM materialized_query_probe(), duckdb_schemas()"));
}

namespace {

bool HasMetadataScan(LogicalOperator &plan) {
	if (plan.type == LogicalOperatorType::LOGICAL_GET &&
	    plan.Cast<LogicalGet>().function.GetSourceKind() == TableFunctionSourceKind::CLIENT_METADATA) {
		return true;
	}
	for (auto &child : plan.children) {
		if (HasMetadataScan(*child)) {
			return true;
		}
	}
	return false;
}

void InjectOrdinaryScan(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	if (!HasMetadataScan(*plan)) {
		return;
	}
	Parser parser;
	parser.ParseQuery("SELECT * FROM range(1)");
	Planner replacement(input.context);
	replacement.CreatePlan(std::move(parser.statements[0]));
	plan = std::move(replacement.plan);
}

} // namespace

TEST_CASE("Native metadata routing rechecks optimizer scan replacements", "[client_context_query]") {
	DBConfig config;
	OptimizerExtension extension;
	extension.optimize_function = InjectOrdinaryScan;
	OptimizerExtension::Register(config, std::move(extension));
	DuckDB db(nullptr, &config);
	Connection connection(db, "ray");
	const bool explicit_transaction = GENERATE(false, true);
	if (explicit_transaction) {
		connection.BeginTransaction();
	}
	REQUIRE_FALSE(connection.Query("CREATE TABLE transaction_marker(value INTEGER)")->HasError());
	PendingQueryParameters parameters;
	parameters.bound_plan_handler = [](Planner &, unique_ptr<LogicalOperator> &, PreparedStatementData &) {
		return false;
	};
	auto pending = connection.PendingQuery("SELECT schema_name FROM duckdb_schemas()", parameters);
	REQUIRE(pending);
	REQUIRE(pending->HasError());
	REQUIRE(pending->GetErrorType() == ExceptionType::BINDER);
	REQUIRE(StringUtil::Contains(pending->GetError(), "ordinary data source"));
	REQUIRE_FALSE(connection.Query("SELECT * FROM transaction_marker")->HasError());
	if (explicit_transaction) {
		connection.Commit();
	}
}
