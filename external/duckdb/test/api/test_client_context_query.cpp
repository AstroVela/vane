// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/client_context_state.hpp"
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
#include "duckdb/planner/operator_extension.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_column_data_get.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
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

struct PhysicalParameterBindState : public ClientContextState {
	bool admitted = false;
	idx_t calls = 0;
};

class ClientReadCollationExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		ScalarFunction function("client_read_collation", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
		                        ScalarFunction::NopFunction);
		function.SetBindCallback(
		    [](ClientContext &context, ScalarFunction &, vector<unique_ptr<Expression>> &) -> unique_ptr<FunctionData> {
			    auto state = context.registered_state->Get<PhysicalParameterBindState>("physical_parameters");
			    if (state && state->admitted) {
				    auto binder = context.GetQueryBinder();
				    REQUIRE(binder);
				    auto parameters = binder->GetParameters();
				    REQUIRE(parameters);
				    auto &values = parameters->GetParameterData();
				    auto entry = values.find("excluded");
				    REQUIRE(entry != values.end());
				    REQUIRE(entry->second.GetValue() == Value("x"));
				    state->calls++;
			    }
			    return nullptr;
		    });
		function.SetClientContextRead();
		CreateCollationInfo info("client_read_collation", std::move(function), true, false);
		loader.RegisterCollation(info);

		ScalarFunction unsupported("unmarked_client_collation", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
		                           ScalarFunction::NopFunction);
		unsupported.SetRequiresClientContext();
		CreateCollationInfo unsupported_info("unmarked_client_collation", std::move(unsupported), true, false);
		loader.RegisterCollation(unsupported_info);
	}
	string Name() override {
		return "client_read_collation_extension";
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

struct AdmissionRetryInfo : public OperatorExtensionInfo {
	idx_t calls = 0;
};

class AdmissionRebindState : public ClientContextState {
public:
	idx_t calls = 0;
	bool CanRequestRebind() override {
		return true;
	}
	RebindQueryInfo OnPlanningError(ClientContext &, SQLStatement &, ErrorData &) override {
		calls++;
		return RebindQueryInfo::ATTEMPT_TO_REBIND;
	}
};

class AdmissionRetryExtension : public OperatorExtension {
public:
	explicit AdmissionRetryExtension(shared_ptr<AdmissionRetryInfo> info) {
		operator_info = std::move(info);
		Bind = [](ClientContext &, Binder &, OperatorExtensionInfo *info, SQLStatement &) -> BoundStatement {
			static_cast<AdmissionRetryInfo &>(*info).calls++;
			throw InvalidInputException("operator extension callback invoked");
		};
	}
	std::string GetName() override {
		return "admission_retry_probe";
	}
	unique_ptr<LogicalExtensionOperator> Deserialize(Deserializer &) override {
		throw NotImplementedException("admission_retry_probe has no operator");
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

TEST_CASE("Context-only collation binding retains the owning query dependency", "[client_context_query]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<ClientReadCollationExtension>();
	Connection connection(db, "ray");
	const bool explicit_transaction = GENERATE(false, true);
	if (explicit_transaction) {
		connection.BeginTransaction();
	}
	PendingQueryParameters parameters;
	parameters.bound_plan_handler = [](Planner &, unique_ptr<LogicalOperator> &, PreparedStatementData &prepared) {
		REQUIRE(prepared.native_client_query);
		return false;
	};
	for (auto sql : {"SELECT 'a' COLLATE client_read_collation", "SELECT 'a' COLLATE client_read_collation = 'a'",
	                 "SELECT min('a' COLLATE client_read_collation)",
	                 "SELECT 'a' AS value ORDER BY value COLLATE client_read_collation",
	                 "SELECT lag('a') OVER (ORDER BY 'a' COLLATE client_read_collation)"}) {
		INFO(sql);
		REQUIRE_NOTHROW(connection.RelationFromQuery(sql));
		auto pending = connection.PendingQuery(sql, parameters);
		REQUIRE_FALSE(pending->HasError());
		REQUIRE_FALSE(pending->Execute()->HasError());
		REQUIRE_FALSE(connection.context->GetQueryBinder());
	}
	// Default collations have no explicit COLLATE expression to register a read.
	REQUIRE_FALSE(connection.Query("SET default_collation = client_read_collation")->HasError());
	auto pending = connection.PendingQuery("SELECT 'a' = 'b'", parameters);
	REQUIRE_FALSE(pending->HasError());
	auto result = pending->Execute();
	REQUIRE_FALSE(result->HasError());
	REQUIRE(result->Fetch()->GetValue(0, 0) == Value(false));
	REQUIRE_FALSE(connection.Query("SET default_collation = ''")->HasError());
	REQUIRE_THROWS_WITH(connection.RelationFromQuery("SELECT 'a' COLLATE unmarked_client_collation"),
	                    Catch::Matchers::Contains(explicit_transaction ? "auto-commit" : "client-context"));
	REQUIRE_FALSE(connection.context->GetQueryBinder());
	if (explicit_transaction) {
		connection.Commit();
	}
	REQUIRE_THROWS_WITH(connection.RelationFromQuery("SELECT 'a' COLLATE client_read_collation FROM range(1)"),
	                    Catch::Matchers::Contains("Client metadata queries cannot mix"));
	parameters.bound_plan_handler = [](Planner &, unique_ptr<LogicalOperator> &, PreparedStatementData &prepared) {
		REQUIRE_FALSE(prepared.native_client_query);
		return false;
	};
	pending = connection.PendingQuery("SELECT 42", parameters);
	REQUIRE_FALSE(pending->HasError());
	REQUIRE_FALSE(pending->Execute()->HasError());
}

TEST_CASE("Physical planning retains query dependency admission", "[client_context_query]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<ClientReadCollationExtension>();
	Connection connection(db, "ray");
	connection.context->config.enable_optimizer = GENERATE(false, true);
	auto rebind_probe = make_shared_ptr<AdmissionRebindState>();
	connection.context->registered_state->Insert("admission_rebind_probe", rebind_probe);
	const bool explicit_transaction = GENERATE(false, true);
	auto operation = GENERATE("EXCEPT", "INTERSECT", "EXCEPT ALL", "INTERSECT ALL");
	if (explicit_transaction) {
		connection.BeginTransaction();
	}
	PendingQueryParameters parameters;
	parameters.bound_plan_handler = [](Planner &, unique_ptr<LogicalOperator> &, PreparedStatementData &prepared) {
		REQUIRE(prepared.native_client_query);
		return false;
	};
	auto sql = string("SELECT schema_name FROM duckdb_schemas() ") + operation + " SELECT 'x'";
	INFO(sql);
	REQUIRE_FALSE(connection.Query("SET default_collation = unmarked_client_collation")->HasError());
	auto pending = connection.PendingQuery(sql, parameters);
	REQUIRE(pending->HasError());
	REQUIRE(StringUtil::Contains(pending->GetError(), explicit_transaction ? "auto-commit" : "client-context"));
	REQUIRE(Binder::IsQueryAdmissionError(pending->GetErrorObject()));
	REQUIRE(rebind_probe->calls == 0);
	REQUIRE_FALSE(connection.context->GetQueryBinder());
	// A rejected physical plan must not poison the transaction or its next query.
	REQUIRE_FALSE(connection.Query("SET default_collation = client_read_collation")->HasError());
	pending = connection.PendingQuery(sql, parameters);
	REQUIRE_FALSE(pending->HasError());
	REQUIRE_FALSE(pending->Execute()->HasError());
	REQUIRE_FALSE(connection.context->GetQueryBinder());
	if (explicit_transaction) {
		connection.Commit();
	}
}

TEST_CASE("Physical binding callbacks retain the owning query parameters", "[client_context_query]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<ClientReadCollationExtension>();
	Connection connection(db, "ray");
	connection.context->config.enable_optimizer = false;
	auto state = make_shared_ptr<PhysicalParameterBindState>();
	connection.context->registered_state->Insert("physical_parameters", state);
	REQUIRE_FALSE(connection.Query("SET default_collation = client_read_collation")->HasError());
	const bool explicit_transaction = GENERATE(false, true);
	if (explicit_transaction) {
		connection.BeginTransaction();
	}
	case_insensitive_map_t<BoundParameterData> values;
	values.emplace("excluded", BoundParameterData(Value("x")));
	PendingQueryParameters parameters;
	parameters.parameters = values;
	parameters.bound_plan_handler = [&](Planner &, unique_ptr<LogicalOperator> &, PreparedStatementData &prepared) {
		REQUIRE(prepared.native_client_query);
		state->admitted = true;
		return false;
	};
	auto pending =
	    connection.PendingQuery("SELECT schema_name FROM duckdb_schemas() EXCEPT SELECT $excluded", parameters);
	REQUIRE_FALSE(pending->HasError());
	REQUIRE(state->calls > 0);
	REQUIRE_FALSE(connection.context->GetQueryBinder());
	REQUIRE_FALSE(pending->Execute()->HasError());
	if (explicit_transaction) {
		connection.Commit();
	}
}

TEST_CASE("Query admission rejection cannot enter operator-extension retries", "[client_context_query]") {
	DBConfig config;
	auto probe = make_shared_ptr<AdmissionRetryInfo>();
	OperatorExtension::Register(config, make_shared_ptr<AdmissionRetryExtension>(probe));
	DuckDB db(nullptr, &config);
	db.LoadStaticExtension<ClientContextOverloadExtension<false>>();
	Connection connection(db, "ray");
	auto rebind_probe = make_shared_ptr<AdmissionRebindState>();
	connection.context->registered_state->Insert("admission_rebind_probe", rebind_probe);
	const bool explicit_transaction = GENERATE(false, true);
	if (explicit_transaction) {
		connection.BeginTransaction();
	}
	PendingQueryParameters parameters;
	parameters.bound_plan_handler = [](Planner &, unique_ptr<LogicalOperator> &, PreparedStatementData &) {
		FAIL("Rejected query reached the bound plan handler");
		return false;
	};
	for (auto sql : {"SELECT * FROM duckdb_schemas(), range(1)", "SELECT * FROM range(1), duckdb_schemas()",
	                 "WITH rejected AS (SELECT * FROM duckdb_schemas(), range(1)) SELECT * FROM rejected",
	                 "SELECT current_schema(1)", "SELECT * FROM duckdb_schemas(1)",
	                 "SELECT nextval('absent_sequence') FROM duckdb_schemas()"}) {
		INFO(sql);
		auto pending = connection.PendingQuery(sql, parameters);
		REQUIRE(pending->HasError());
		REQUIRE(Binder::IsQueryAdmissionError(pending->GetErrorObject()));
		REQUIRE(probe->calls == 0);
		REQUIRE(rebind_probe->calls == 0);
		REQUIRE_FALSE(connection.context->GetQueryBinder());
	}
	REQUIRE_FALSE(connection.Query("SELECT current_schema()")->HasError());
	if (explicit_transaction) {
		// Source-free runner queries reach the completed-query transaction check.
		auto pending = connection.PendingQuery("SELECT 42", parameters);
		REQUIRE(pending->HasError());
		REQUIRE(Binder::IsQueryAdmissionError(pending->GetErrorObject()));
		REQUIRE(probe->calls == 0);
		REQUIRE(rebind_probe->calls == 0);
		connection.Commit();
	}
	// The installed extension still handles ordinary native binding failures.
	connection.context->registered_state->Remove("admission_rebind_probe");
	auto result = connection.Query("SELECT absent_column");
	REQUIRE(result->HasError());
	REQUIRE(StringUtil::Contains(result->GetError(), "operator extension callback invoked"));
	REQUIRE(probe->calls == 1);
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
	if (plan.GetSourceKind() == QuerySourceKind::CLIENT_METADATA) {
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

void InjectColumnDataScan(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	if (HasMetadataScan(*plan)) {
		vector<LogicalType> types {LogicalType::VARCHAR};
		auto data = make_uniq<ColumnDataCollection>(input.context, types);
		plan = make_uniq<LogicalColumnDataGet>(0, types, std::move(data));
	}
}

class UnclassifiedScan : public LogicalExtensionOperator {
public:
	PhysicalOperator &CreatePlan(ClientContext &, PhysicalPlanGenerator &) override {
		throw InternalException("Unclassified scan reached physical planning");
	}
	string GetExtensionName() const override {
		return "unclassified_scan";
	}
	void ResolveTypes() override {
		types = {LogicalType::VARCHAR};
	}
};

class DerivedExtensionOperator : public LogicalExtensionOperator {
public:
	explicit DerivedExtensionOperator(unique_ptr<LogicalOperator> child) {
		children.push_back(std::move(child));
	}
	QuerySourceKind GetSourceKind() const override {
		return QuerySourceKind::NONE;
	}
	vector<ColumnBinding> GetColumnBindings() override {
		return children[0]->GetColumnBindings();
	}
	PhysicalOperator &CreatePlan(ClientContext &, PhysicalPlanGenerator &planner) override {
		return planner.CreatePlan(*children[0]);
	}
	string GetExtensionName() const override {
		return "derived_extension";
	}
	void ResolveTypes() override {
		types = children[0]->types;
	}
};

void InjectDerivedOperator(OptimizerExtensionInput &, unique_ptr<LogicalOperator> &plan) {
	if (HasMetadataScan(*plan)) {
		plan = make_uniq<DerivedExtensionOperator>(std::move(plan));
	}
}

struct ExpressionDependencyProbe : public FunctionData {
	explicit ExpressionDependencyProbe(unique_ptr<Expression> expression_p) : expression(std::move(expression_p)) {
	}
	unique_ptr<Expression> expression;
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<ExpressionDependencyProbe>(expression->Copy());
	}
	bool Equals(const FunctionData &other) const override {
		return expression->Equals(*other.Cast<ExpressionDependencyProbe>().expression);
	}
	void VisitExpressionDependencies(const std::function<void(const Expression &)> &callback) const override {
		callback(*expression);
	}
};

unique_ptr<FunctionData> BindExpressionDependencyProbe(ClientContext &context, ScalarFunction &,
                                                       vector<unique_ptr<Expression>> &) {
	vector<unique_ptr<Expression>> children;
	children.push_back(make_uniq<BoundConstantExpression>(Value("metadata_sequence")));
	FunctionBinder binder(context);
	ErrorData error;
	auto expression = binder.BindScalarFunction(DEFAULT_SCHEMA, "nextval", std::move(children), error);
	if (!expression) {
		error.Throw();
	}
	return make_uniq<ExpressionDependencyProbe>(std::move(expression));
}

void ExecuteExpressionDependencyProbe(DataChunk &input, ExpressionState &state, Vector &result) {
	auto &function = state.expr.Cast<BoundFunctionExpression>();
	auto &data = function.bind_info->Cast<ExpressionDependencyProbe>();
	ExpressionExecutor executor(state.GetContext(), *data.expression);
	executor.ExecuteExpression(input, result);
}

void InjectUnclassifiedScan(OptimizerExtensionInput &, unique_ptr<LogicalOperator> &plan) {
	if (HasMetadataScan(*plan)) {
		plan = make_uniq<UnclassifiedScan>();
	}
}

struct ReplacementQueryInfo : public OptimizerExtensionInfo {
	explicit ReplacementQueryInfo(string sql_p) : sql(std::move(sql_p)) {
	}
	string sql;
};

void InjectReplacementQuery(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	if (!HasMetadataScan(*plan)) {
		return;
	}
	Parser parser;
	parser.ParseQuery(static_cast<ReplacementQueryInfo &>(*input.info).sql);
	Planner replacement(input.context);
	replacement.CreatePlan(std::move(parser.statements[0]));
	plan = std::move(replacement.plan);
}

} // namespace

TEST_CASE("Native metadata routing rechecks optimizer scan replacements", "[client_context_query]") {
	DBConfig config;
	OptimizerExtension extension;
	extension.optimize_function = GENERATE(&InjectOrdinaryScan, &InjectColumnDataScan, &InjectUnclassifiedScan);
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
	REQUIRE(StringUtil::Contains(pending->GetError(),
	                             explicit_transaction ? "auto-commit" : "Client metadata queries cannot mix"));
	REQUIRE_FALSE(connection.Query("SELECT * FROM transaction_marker")->HasError());
	if (explicit_transaction) {
		connection.Commit();
	}
}

TEST_CASE("Native metadata routing rechecks optimized expression effects", "[client_context_query]") {
	auto replacement_sql =
	    GENERATE("SELECT nextval('metadata_sequence') FROM duckdb_schemas()",
	             "SELECT schema_name FROM duckdb_schemas() WHERE nextval('metadata_sequence') > 0",
	             "SELECT schema_name FROM duckdb_schemas() ORDER BY nextval('metadata_sequence')",
	             "SELECT sum(nextval('metadata_sequence')) OVER () FROM duckdb_schemas()",
	             "SELECT list_transform([1], x -> nextval('metadata_sequence')) FROM duckdb_schemas()",
	             "SELECT optimizer_client_probe(schema_name) FROM duckdb_schemas()",
	             "SELECT optimizer_expression_probe(schema_name) FROM duckdb_schemas()");
	INFO(replacement_sql);
	DBConfig config;
	OptimizerExtension extension;
	extension.optimize_function = InjectReplacementQuery;
	extension.optimizer_info = make_shared_ptr<ReplacementQueryInfo>(replacement_sql);
	OptimizerExtension::Register(config, std::move(extension));
	DuckDB db(nullptr, &config);
	Connection connection(db, "ray");
	ScalarFunction function("optimizer_client_probe", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
	                        ScalarFunction::NopFunction);
	function.SetRequiresClientContext();
	CreateScalarFunctionInfo info(std::move(function));
	connection.context->RegisterFunction(info);
	ScalarFunction dependency_function("optimizer_expression_probe", {LogicalType::VARCHAR}, LogicalType::BIGINT,
	                                   ExecuteExpressionDependencyProbe, BindExpressionDependencyProbe);
	dependency_function.SetStability(FunctionStability::VOLATILE);
	CreateScalarFunctionInfo dependency_info(std::move(dependency_function));
	connection.context->RegisterFunction(dependency_info);
	const bool explicit_transaction = GENERATE(false, true);
	if (explicit_transaction) {
		connection.BeginTransaction();
	}
	REQUIRE_FALSE(connection.Query("CREATE SEQUENCE metadata_sequence")->HasError());
	REQUIRE_FALSE(connection.Query("CREATE TABLE transaction_marker(value INTEGER)")->HasError());
	PendingQueryParameters parameters;
	parameters.bound_plan_handler = [](Planner &, unique_ptr<LogicalOperator> &, PreparedStatementData &prepared) {
		REQUIRE(prepared.native_client_query);
		return false;
	};
	auto pending = connection.PendingQuery("SELECT schema_name FROM duckdb_schemas()", parameters);
	REQUIRE(pending->HasError());
	REQUIRE(pending->GetErrorType() == ExceptionType::BINDER);
	REQUIRE(StringUtil::Contains(pending->GetError(),
	                             explicit_transaction ? "auto-commit" : "Runner execution does not support"));
	// Rejection happens before execution and leaves the caller's transaction usable.
	auto sequence = connection.Query("SELECT nextval('metadata_sequence')");
	REQUIRE_FALSE(sequence->HasError());
	REQUIRE(sequence->Fetch()->GetValue(0, 0) == Value::BIGINT(1));
	auto native_probe = connection.Query("SELECT optimizer_expression_probe('value')");
	REQUIRE_FALSE(native_probe->HasError());
	REQUIRE(native_probe->Fetch()->GetValue(0, 0) == Value::BIGINT(2));
	REQUIRE_FALSE(connection.Query("SELECT * FROM transaction_marker")->HasError());
	if (explicit_transaction) {
		connection.Commit();
	}
}

TEST_CASE("Declared expression-derived extension operators retain metadata routing", "[client_context_query]") {
	DBConfig config;
	OptimizerExtension extension;
	extension.optimize_function = InjectDerivedOperator;
	OptimizerExtension::Register(config, std::move(extension));
	DuckDB db(nullptr, &config);
	Connection connection(db, "ray");
	connection.BeginTransaction();
	REQUIRE_FALSE(connection.Query("CREATE TABLE alpha(value INTEGER)")->HasError());
	PendingQueryParameters parameters;
	parameters.bound_plan_handler = [](Planner &, unique_ptr<LogicalOperator> &, PreparedStatementData &prepared) {
		REQUIRE(prepared.native_client_query);
		return false;
	};
	auto pending =
	    connection.PendingQuery("SELECT table_name FROM duckdb_tables() WHERE table_name = 'alpha'", parameters);
	REQUIRE_FALSE(pending->HasError());
	auto result = pending->Execute();
	REQUIRE_FALSE(result->HasError());
	REQUIRE(result->Fetch()->GetValue(0, 0) == Value("alpha"));
	connection.Commit();
}
