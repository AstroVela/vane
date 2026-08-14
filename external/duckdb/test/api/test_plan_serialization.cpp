// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "catch.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/enums/database_modification_type.hpp"
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/main/relation.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_local_exchange.hpp"
#include "duckdb/planner/operator/logical_repartition.hpp"
#include "duckdb/planner/logical_write_target.hpp"
#include "duckdb/planner/parsed_data/bound_create_table_info.hpp"
#include "duckdb/transaction/meta_transaction.hpp"
#include "test_helpers.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"

#include <map>
#include <set>
#include <tuple>

using namespace duckdb;
using namespace std;

static const LogicalLocalExchange *FindLocalExchange(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_LOCAL_EXCHANGE) {
		return &op.Cast<LogicalLocalExchange>();
	}
	for (auto &child : op.children) {
		if (!child) {
			continue;
		}
		auto result = FindLocalExchange(*child);
		if (result) {
			return result;
		}
	}
	return nullptr;
}

static const LogicalRepartition *FindRepartition(const LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_REPARTITION) {
		return &op.Cast<LogicalRepartition>();
	}
	for (auto &child : op.children) {
		if (!child) {
			continue;
		}
		auto result = FindRepartition(*child);
		if (result) {
			return result;
		}
	}
	return nullptr;
}

static string SerializeUnoptimizedPlan(Connection &con, const string &sql) {
	string serialized_plan;
	con.context->RunFunctionInTransaction([&]() {
		Parser parser;
		parser.ParseQuery(sql);
		if (parser.statements.size() != 1) {
			throw InternalException("Expected exactly one statement in serialization test");
		}
		Planner planner(*con.context);
		planner.CreatePlan(std::move(parser.statements[0]));

		MemoryStream stream(Allocator::Get(*con.context));
		SerializationOptions options;
		options.serialization_compatibility = SerializationCompatibility::Latest();
		options.serialize_default_values = true;
		BinarySerializer::Serialize(*planner.plan, stream, options);
		serialized_plan.assign(reinterpret_cast<const char *>(stream.GetData()), stream.GetPosition());
	});
	return serialized_plan;
}

static duckdb::unique_ptr<LogicalOperator> DeserializePlan(Connection &con, const string &serialized_plan) {
	duckdb::unique_ptr<LogicalOperator> result;
	con.context->RunFunctionInTransaction([&]() {
		MemoryStream stream(Allocator::Get(*con.context));
		stream.WriteData(const_data_ptr_cast(serialized_plan.data()), serialized_plan.size());
		stream.Rewind();
		bound_parameter_map_t parameters;
		result = BinaryDeserializer::Deserialize<LogicalOperator>(stream, *con.context, parameters);
	});
	return result;
}

static string GetTableWriteIdentity(Connection &con, const string &table_name,
                                    const string &catalog_name = INVALID_CATALOG) {
	string result;
	con.context->RunFunctionInTransaction([&]() {
		auto &table = Catalog::GetEntry<TableCatalogEntry>(*con.context, catalog_name, DEFAULT_SCHEMA, table_name);
		result = table.GetLogicalWriteTargetIdentity();
	});
	return result;
}

static string GetTableWriteDefinition(Connection &con, const string &table_name,
                                      const string &catalog_name = INVALID_CATALOG) {
	string result;
	con.context->RunFunctionInTransaction([&]() {
		auto &table = Catalog::GetEntry<TableCatalogEntry>(*con.context, catalog_name, DEFAULT_SCHEMA, table_name);
		result = table.GetLogicalWriteTargetDefinition(*con.context);
	});
	return result;
}

static void test_helper(string sql, duckdb::vector<string> fixtures = duckdb::vector<string>()) {
	DuckDB db;
	Connection con(db);

	for (const auto &fixture : fixtures) {
		con.SendQuery(fixture);
	}

	Parser p;
	p.ParseQuery(sql);

	for (auto &statement : p.statements) {
		con.context->transaction.BeginTransaction();
		// Should that be the default "ToString"?
		string statement_sql(statement->query.c_str() + statement->stmt_location, statement->stmt_length);
		Planner planner(*con.context);
		planner.CreatePlan(std::move(statement));
		auto plan = std::move(planner.plan);

		Optimizer optimizer(*planner.binder, *con.context);

		plan = optimizer.Optimize(std::move(plan));

		// LogicalOperator's copy utilizes its serialize and deserialize methods
		auto new_plan = plan->Copy(*con.context);

		auto optimized_plan = optimizer.Optimize(std::move(new_plan));
		con.context->transaction.Commit();
	}
}

static void test_helper_multi_db(string sql, duckdb::vector<string> fixtures = duckdb::vector<string>()) {
	DuckDB db;
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("ATTACH DATABASE ':memory:' AS new_db;"));

	for (const auto &fixture : fixtures) {
		con.SendQuery(fixture);
	}

	Parser p;
	p.ParseQuery(sql);

	for (auto &statement : p.statements) {
		con.context->transaction.BeginTransaction();
		// Should that be the default "ToString"?
		string statement_sql(statement->query.c_str() + statement->stmt_location, statement->stmt_length);
		Planner planner(*con.context);
		planner.CreatePlan(std::move(statement));
		auto plan = std::move(planner.plan);

		Optimizer optimizer(*planner.binder, *con.context);

		plan = optimizer.Optimize(std::move(plan));

		// LogicalOperator's copy utilizes its serialize and deserialize methods
		auto new_plan = plan->Copy(*con.context);

		auto optimized_plan = optimizer.Optimize(std::move(new_plan));
		con.context->transaction.Commit();
	}
}

TEST_CASE("Test logical_set", "[serialization]") {
	test_helper("SET memory_limit='10GB'");
}

TEST_CASE("Test logical_show", "[serialization]") {
	test_helper("SHOW SELECT 42");
}

TEST_CASE("Test logical_explain", "[serialization]") {
	test_helper("EXPLAIN SELECT 42");
}

TEST_CASE("Test logical_empty_result", "[serialization]") {
	test_helper("SELECT * FROM (SELECT 42) WHERE 1>2");
}

TEST_CASE("Test create_table", "[serialization]") {
	test_helper("CREATE TABLE tbl (foo INTEGER)");
}

TEST_CASE("Test insert_into", "[serialization]") {
	test_helper("INSERT INTO tbl VALUES(1)", {"CREATE TABLE tbl (foo INTEGER)"});
}

TEST_CASE("Test logical_delete", "[serialization]") {
	test_helper("DELETE FROM tbl", {"CREATE TABLE tbl (foo INTEGER)"});
}

// TODO: only select for now
// TEST_CASE("Test logical_create_index", "[serialization]") {
//	test_helper("CREATE INDEX idx ON tbl (foo)", {"CREATE TABLE tbl (foo INTEGER)"});
//}
// TODO: only select for now
// TEST_CASE("Test logical_create_schema", "[serialization]") {
//	test_helper("CREATE SCHEMA test");
//}
// TODO: only select for now
// TEST_CASE("Test logical_create_view", "[serialization]") {
//	test_helper("CREATE VIEW test_view AS (SELECT 42)");
//}

TEST_CASE("Test logical_update", "[serialization]") {
	test_helper("UPDATE tbl SET foo=42", {"CREATE TABLE tbl (foo INTEGER)"});
}

TEST_CASE("Test logical_merge_into", "[serialization]") {
	test_helper("MERGE INTO tbl USING (VALUES (1)) src(foo) USING (foo) WHEN MATCHED THEN UPDATE",
	            {"CREATE TABLE tbl (foo INTEGER)"});
}

TEST_CASE("Serialized DML rejects a replacement at the same catalog path", "[serialization][dml]") {
	DuckDB db;
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE target(i INTEGER)"));

	for (auto &sql : duckdb::vector<string> {
	         "INSERT INTO target VALUES (1)",
	         "UPDATE target SET i = 2",
	         "DELETE FROM target",
	         "MERGE INTO target USING (VALUES (1)) src(i) USING (i) WHEN MATCHED THEN UPDATE",
	     }) {
		auto serialized_plan = SerializeUnoptimizedPlan(con, sql);
		REQUIRE_NO_FAIL(con.Query("DROP TABLE target"));
		REQUIRE_NO_FAIL(con.Query("CREATE TABLE target(i INTEGER)"));
		REQUIRE_THROWS_WITH(DeserializePlan(con, serialized_plan), Catch::Matchers::Contains("was replaced"));
	}
}

TEST_CASE("Duck table write target state survives persistence and detects definition changes", "[serialization][dml]") {
	auto db_path = TestCreatePath("logical_write_target_identity.db");
	string original_identity;
	string altered_definition;
	string altered_plan;
	string unindexed_definition;
	string unindexed_plan;
	string renamed_definition;
	string renamed_plan;
	{
		DuckDB db(db_path);
		Connection con(db);
		REQUIRE_NO_FAIL(con.Query("CREATE TABLE target(i INTEGER)"));
		original_identity = GetTableWriteIdentity(con, "target");
		REQUIRE(!original_identity.empty());
		auto original_definition = GetTableWriteDefinition(con, "target");
		REQUIRE(!original_definition.empty());

		auto serialized_plan = SerializeUnoptimizedPlan(con, "UPDATE target SET i = 2");
		REQUIRE_NO_FAIL(con.Query("ALTER TABLE target ADD COLUMN j INTEGER"));
		REQUIRE_NO_FAIL(con.Query("CREATE INDEX target_i_idx ON target(i)"));
		REQUIRE(GetTableWriteIdentity(con, "target") == original_identity);
		altered_definition = GetTableWriteDefinition(con, "target");
		REQUIRE(altered_definition != original_definition);
		REQUIRE_THROWS_WITH(DeserializePlan(con, serialized_plan), Catch::Matchers::Contains("definition changed"));
		altered_plan = SerializeUnoptimizedPlan(con, "UPDATE target SET i = 3");
		REQUIRE(DeserializePlan(con, altered_plan) != nullptr);
		REQUIRE_NO_FAIL(con.Query("PRAGMA force_checkpoint"));
	}
	{
		DuckDB db(db_path);
		Connection con(db);
		REQUIRE(GetTableWriteIdentity(con, "target") == original_identity);
		REQUIRE(GetTableWriteDefinition(con, "target") == altered_definition);
		REQUIRE(DeserializePlan(con, altered_plan) != nullptr);
		REQUIRE_NO_FAIL(con.Query("SET wal_autocheckpoint = '1TB'"));
		REQUIRE_NO_FAIL(con.Query("PRAGMA disable_checkpoint_on_shutdown"));
		REQUIRE_NO_FAIL(con.Query("DROP INDEX target_i_idx"));
		unindexed_definition = GetTableWriteDefinition(con, "target");
		REQUIRE(unindexed_definition != altered_definition);
		REQUIRE_THROWS_WITH(DeserializePlan(con, altered_plan), Catch::Matchers::Contains("definition changed"));
		unindexed_plan = SerializeUnoptimizedPlan(con, "UPDATE target SET i = 4");
		REQUIRE(DeserializePlan(con, unindexed_plan) != nullptr);
		REQUIRE_NO_FAIL(con.Query("ALTER TABLE target RENAME COLUMN j TO k"));
		REQUIRE(GetTableWriteIdentity(con, "target") == original_identity);
		renamed_definition = GetTableWriteDefinition(con, "target");
		REQUIRE(renamed_definition != unindexed_definition);
		REQUIRE_THROWS_WITH(DeserializePlan(con, unindexed_plan), Catch::Matchers::Contains("definition changed"));
		renamed_plan = SerializeUnoptimizedPlan(con, "UPDATE target SET i = 5");
	}
	{
		DuckDB db(db_path);
		Connection con(db);
		REQUIRE(GetTableWriteIdentity(con, "target") == original_identity);
		REQUIRE(GetTableWriteDefinition(con, "target") == renamed_definition);
		REQUIRE(DeserializePlan(con, renamed_plan) != nullptr);
		REQUIRE_NO_FAIL(con.Query("DROP TABLE target"));
		REQUIRE_NO_FAIL(con.Query("CREATE TABLE target(i INTEGER)"));
		REQUIRE(GetTableWriteIdentity(con, "target") != original_identity);
	}
}

TEST_CASE("Serialized DML rejects a remapped physical column", "[serialization][dml]") {
	DuckDB db;
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE target(a INTEGER, b INTEGER)"));

	auto original_identity = GetTableWriteIdentity(con, "target");
	auto original_definition = GetTableWriteDefinition(con, "target");
	auto serialized_plan = SerializeUnoptimizedPlan(con, "UPDATE target SET a = 99");
	REQUIRE_NO_FAIL(con.Query("ALTER TABLE target DROP COLUMN a"));
	REQUIRE(GetTableWriteIdentity(con, "target") != original_identity);
	REQUIRE(GetTableWriteDefinition(con, "target") != original_definition);
	REQUIRE_THROWS_WITH(DeserializePlan(con, serialized_plan), Catch::Matchers::Contains("was replaced"));
}

TEST_CASE("Serialized DML rejects changed index state", "[serialization][dml]") {
	DuckDB db;
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE target(a INTEGER, b INTEGER)"));

	auto without_index_definition = GetTableWriteDefinition(con, "target");
	auto without_index_plan = SerializeUnoptimizedPlan(con, "UPDATE target SET a = 99");
	REQUIRE_NO_FAIL(con.Query("CREATE INDEX target_a_idx ON target(a)"));
	auto with_index_definition = GetTableWriteDefinition(con, "target");
	REQUIRE(with_index_definition != without_index_definition);
	REQUIRE_THROWS_WITH(DeserializePlan(con, without_index_plan), Catch::Matchers::Contains("definition changed"));

	auto with_index_plan = SerializeUnoptimizedPlan(con, "UPDATE target SET a = 100");
	REQUIRE_NO_FAIL(con.Query("DROP INDEX target_a_idx"));
	REQUIRE(GetTableWriteDefinition(con, "target") == without_index_definition);
	REQUIRE_THROWS_WITH(DeserializePlan(con, with_index_plan), Catch::Matchers::Contains("definition changed"));
}

TEST_CASE("Serialized DML rejects changed index expressions", "[serialization][dml]") {
	DuckDB db;
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE target(a INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("CREATE UNIQUE INDEX target_expr_idx ON target((a % 2))"));

	auto modulo_two_definition = GetTableWriteDefinition(con, "target");
	auto modulo_two_plan = SerializeUnoptimizedPlan(con, "UPDATE target SET a = 99");
	REQUIRE_NO_FAIL(con.Query("DROP INDEX target_expr_idx"));
	REQUIRE_NO_FAIL(con.Query("CREATE UNIQUE INDEX target_expr_idx ON target((a % 3))"));

	REQUIRE(GetTableWriteDefinition(con, "target") != modulo_two_definition);
	REQUIRE_THROWS_WITH(DeserializePlan(con, modulo_two_plan), Catch::Matchers::Contains("definition changed"));
}

TEST_CASE("Copy database assigns a new table write identity", "[serialization][dml]") {
	DuckDB db;
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("ATTACH ':memory:' AS source"));
	REQUIRE_NO_FAIL(con.Query("ATTACH ':memory:' AS target"));
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE source.main.target(i INTEGER)"));

	auto source_identity = GetTableWriteIdentity(con, "target", "source");
	REQUIRE(!source_identity.empty());
	REQUIRE_NO_FAIL(con.Query("COPY FROM DATABASE source TO target (SCHEMA)"));
	auto copied_identity = GetTableWriteIdentity(con, "target", "target");
	REQUIRE(!copied_identity.empty());
	REQUIRE(copied_identity != source_identity);
}

TEST_CASE("Logical write targets require a non-empty identity", "[serialization][dml]") {
	MemoryStream stream;
	BinarySerializer serializer(stream);
	serializer.Begin();
	serializer.WriteProperty<string>(100, "catalog_name", "memory");
	serializer.WriteProperty<string>(101, "schema_name", "main");
	serializer.WriteProperty<string>(102, "table_name", "target");
	serializer.WriteProperty<string>(103, "identity", "");
	serializer.WriteProperty<string>(104, "definition", "CREATE TABLE memory.main.target(i INTEGER)");
	serializer.End();
	stream.Rewind();

	BinaryDeserializer deserializer(stream);
	deserializer.Begin();
	REQUIRE_THROWS_WITH(LogicalWriteTarget::Deserialize(deserializer),
	                    Catch::Matchers::Contains("does not provide a logical write target identity"));
}

TEST_CASE("Checkpoint-loaded tables without an identity remain unsupported", "[serialization][dml]") {
	auto db_path = TestCreatePath("logical_write_target_legacy.db");
	{
		DuckDB db(db_path);
		Connection con(db);
		con.context->RunFunctionInTransaction([&]() {
			auto &catalog = Catalog::GetCatalog(*con.context, "");
			auto &transaction = MetaTransaction::Get(*con.context);
			transaction.ModifyDatabase(catalog.GetAttached(), DatabaseModificationType::CREATE_CATALOG_ENTRY);
			auto &schema = catalog.GetSchema(*con.context, DEFAULT_SCHEMA);
			auto info = make_uniq<CreateTableInfo>(schema, "legacy_target");
			info->columns.AddColumn(ColumnDefinition("i", LogicalType::INTEGER));
			auto bound_info = Binder::BindCreateTableCheckpoint(std::move(info), schema);
			REQUIRE(schema.ParentCatalog().CreateTable(*con.context, *bound_info) != nullptr);
		});
		REQUIRE(GetTableWriteIdentity(con, "legacy_target").empty());
		REQUIRE_NO_FAIL(con.Query("PRAGMA force_checkpoint"));
	}
	{
		DuckDB db(db_path);
		Connection con(db);
		REQUIRE(GetTableWriteIdentity(con, "legacy_target").empty());
		REQUIRE_THROWS_WITH(SerializeUnoptimizedPlan(con, "UPDATE legacy_target SET i = 2"),
		                    Catch::Matchers::Contains("does not provide a logical write target identity"));
	}
}

TEST_CASE("Logical repartition round-trips hash random and into-partitions modes", "[serialization][repartition]") {
	DuckDB db;
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE integers(i INTEGER)"));

	for (auto &test_case : duckdb::vector<std::tuple<idx_t, duckdb::vector<string>, RepartitionSpec::Type>> {
	         {3, {"i"}, RepartitionSpec::Type::Hash},
	         {0, {"i"}, RepartitionSpec::Type::Hash},
	         {3, {}, RepartitionSpec::Type::Random},
	         {0, {}, RepartitionSpec::Type::Random},
	     }) {
		auto num_partitions = std::get<0>(test_case);
		auto partition_by = std::get<1>(test_case);
		auto expected_type = std::get<2>(test_case);
		auto relation = con.Table("integers")->Repartition(num_partitions, partition_by);
		con.context->RunFunctionInTransaction([&]() {
			auto binder = Binder::CreateBinder(*con.context);
			auto bound = relation->Bind(*binder);
			auto copied_plan = bound.plan->Copy(*con.context);
			auto repartition = FindRepartition(*copied_plan);
			REQUIRE(repartition != nullptr);
			REQUIRE(repartition->repartition_spec->type() == expected_type);
			if (expected_type == RepartitionSpec::Type::Hash) {
				auto *hash_spec = dynamic_cast<HashRepartitionSpec *>(repartition->repartition_spec.get());
				REQUIRE(hash_spec != nullptr);
				REQUIRE(hash_spec->config()->num_partitions == num_partitions);
				REQUIRE(hash_spec->config()->by.size() == 1);
				REQUIRE(repartition->expressions.size() == 1);
			} else {
				auto *random_spec = dynamic_cast<RandomRepartitionSpec *>(repartition->repartition_spec.get());
				REQUIRE(random_spec != nullptr);
				REQUIRE(random_spec->config()->num_partitions == num_partitions);
				REQUIRE(repartition->expressions.empty());
			}
			PhysicalPlanGenerator physical_planner(*con.context);
			REQUIRE(physical_planner.Plan(std::move(copied_plan)) != nullptr);
		});
	}

	auto relation = con.Table("integers");
	con.context->RunFunctionInTransaction([&]() {
		auto binder = Binder::CreateBinder(*con.context);
		auto bound = relation->Bind(*binder);
		auto into = make_uniq<LogicalRepartition>(RepartitionSpec::create_into_partitions(5));
		into->children.push_back(std::move(bound.plan));
		auto copied_into = into->Copy(*con.context);
		auto repartition = FindRepartition(*copied_into);
		REQUIRE(repartition != nullptr);
		REQUIRE(repartition->repartition_spec->type() == RepartitionSpec::Type::IntoPartitions);
		auto *into_spec = dynamic_cast<IntoPartitionsRepartitionSpec *>(repartition->repartition_spec.get());
		REQUIRE(into_spec != nullptr);
		REQUIRE(into_spec->config()->num_partitions == 5);
		REQUIRE(repartition->expressions.empty());
		PhysicalPlanGenerator physical_planner(*con.context);
		REQUIRE(physical_planner.Plan(std::move(copied_into)) != nullptr);
	});
}

TEST_CASE("Logical repartition serializes rewritten operator expressions", "[serialization][repartition]") {
	DuckDB db;
	Connection con(db);
	duckdb::vector<ExprRef> bind_expressions;
	bind_expressions.push_back(std::make_shared<BoundReferenceExpression>(LogicalType::INTEGER, 0));
	LogicalRepartition repartition(RepartitionSpec::create_hash(2, std::move(bind_expressions)));

	// Optimizers rewrite LogicalOperator::expressions without updating the bind-time
	// HashRepartitionConfig copy. Serialization must follow the rewritten expression.
	repartition.expressions.push_back(make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 1));
	auto copied_plan = repartition.Copy(*con.context);
	auto &copied_repartition = copied_plan->Cast<LogicalRepartition>();
	auto *hash_spec = dynamic_cast<HashRepartitionSpec *>(copied_repartition.repartition_spec.get());
	REQUIRE(hash_spec != nullptr);
	REQUIRE(copied_repartition.expressions.size() == 1);
	REQUIRE(hash_spec->config()->by.size() == 1);
	REQUIRE(copied_repartition.expressions[0]->Cast<BoundReferenceExpression>().index == 1);
	REQUIRE(hash_spec->config()->by[0]->Cast<BoundReferenceExpression>().index == 1);
}

TEST_CASE("Logical repartition rejects unsupported and malformed wire states", "[serialization][repartition]") {
	{
		LogicalRepartition missing_spec(nullptr);
		MemoryStream stream;
		REQUIRE_THROWS_WITH(BinarySerializer::Serialize(missing_spec, stream),
		                    Catch::Matchers::Contains("missing its repartition specification"));
	}
	{
		duckdb::vector<ExprRef> config_expressions;
		config_expressions.push_back(std::make_shared<BoundReferenceExpression>(LogicalType::INTEGER, 0));
		LogicalRepartition empty_hash(RepartitionSpec::create_hash(2, std::move(config_expressions)));
		MemoryStream stream;
		REQUIRE_THROWS_WITH(BinarySerializer::Serialize(empty_hash, stream),
		                    Catch::Matchers::Contains("requires at least one partition expression"));
	}
	{
		duckdb::vector<ExprRef> config_expressions;
		config_expressions.push_back(std::make_shared<BoundReferenceExpression>(LogicalType::INTEGER, 0));
		LogicalRepartition null_hash(RepartitionSpec::create_hash(2, std::move(config_expressions)));
		null_hash.expressions.push_back(nullptr);
		MemoryStream stream;
		REQUIRE_THROWS_WITH(BinarySerializer::Serialize(null_hash, stream),
		                    Catch::Matchers::Contains("null partition expression"));
	}
	{
		duckdb::vector<BoundOrderByNode> orders;
		orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_LAST,
		                    make_uniq<BoundReferenceExpression>(LogicalType::INTEGER, 0));
		LogicalRepartition range(RepartitionSpec::create_range(2, std::move(orders), {}));
		MemoryStream stream;
		REQUIRE_THROWS_WITH(BinarySerializer::Serialize(range, stream),
		                    Catch::Matchers::Contains("Range repartition cannot be serialized"));
	}

	auto deserialize_wire = [](uint8_t wire_type, uint64_t num_partitions) {
		MemoryStream stream;
		BinarySerializer serializer(stream);
		serializer.Begin();
		serializer.WriteProperty<uint8_t>(200, "repartition_type", wire_type);
		serializer.WriteProperty<uint64_t>(201, "num_partitions", num_partitions);
		serializer.WriteProperty<duckdb::vector<duckdb::unique_ptr<Expression>>>(202, "partition_by", {});
		serializer.End();
		stream.Rewind();
		BinaryDeserializer deserializer(stream);
		deserializer.Begin();
		return LogicalRepartition::Deserialize(deserializer);
	};

	REQUIRE_THROWS_WITH(deserialize_wire(1, 3), Catch::Matchers::Contains("requires at least one"));
	REQUIRE_THROWS_WITH(deserialize_wire(3, 0), Catch::Matchers::Contains("requires a positive partition count"));
	REQUIRE_THROWS_WITH(deserialize_wire(4, 3), Catch::Matchers::Contains("Range repartition cannot be deserialized"));
	REQUIRE_THROWS_WITH(deserialize_wire(99, 3), Catch::Matchers::Contains("unknown wire repartition type"));
}

// TODO(stephwang): revisit this later since it doesn't work yet
// TEST_CASE("Test logical_copy_to_file", "[serialization]") {
//	test_helper("COPY tbl TO 'test_table.csv' ( DELIMITER '|', HEADER )", {"CREATE TABLE tbl (foo INTEGER)"});
//}

// TODO(stephwang): revisit this later since it doesn't work yet
// TEST_CASE("Test logical_prepare", "[serialization]") {
//	test_helper("PREPARE v1 AS SELECT 42");
//}

TEST_CASE("Test logical_simple with DROP", "[serialization]") {
	test_helper("DROP TABLE tbl", {"CREATE TABLE tbl (foo INTEGER)"});
}

TEST_CASE("Test logical_simple with ALTER", "[serialization]") {
	test_helper("ALTER TABLE tbl ADD COLUMN bar INTEGER", {"CREATE TABLE tbl (foo INTEGER)"});
}

TEST_CASE("Test logical_simple with LOAD", "[serialization]") {
	test_helper("LOAD foo");
}

TEST_CASE("Test logical_local_exchange preserves explicit partitions across serialization", "[serialization]") {
	DuckDB db;
	Connection con(db);

	REQUIRE_NO_FAIL(con.Query("CREATE TABLE integers(i INTEGER)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO integers VALUES (1), (2), (3), (4)"));

	con.context->transaction.BeginTransaction();
	auto rel = con.Table("integers")->LocalExchange(1)->Limit(2);
	auto binder = Binder::CreateBinder(*con.context);
	auto bound = rel->Bind(*binder);

	auto copied_plan = bound.plan->Copy(*con.context);
	auto local_exchange = FindLocalExchange(*copied_plan);
	REQUIRE(local_exchange != nullptr);
	REQUIRE(local_exchange->repartition_spec != nullptr);
	REQUIRE(local_exchange->repartition_spec->type() == RepartitionSpec::Type::Random);

	auto *random_spec = dynamic_cast<RandomRepartitionSpec *>(local_exchange->repartition_spec.get());
	REQUIRE(random_spec != nullptr);
	REQUIRE(random_spec->config()->num_partitions == 1);
	con.context->transaction.Commit();
}

// below test cases are oriented towards multi-databases
TEST_CASE("Test create_table with catalog", "[serialization]") {
	test_helper_multi_db("CREATE TABLE new_db.main.tbl(i INTEGER);");
}

TEST_CASE("Test logical_insert with catalog", "[serialization]") {
	test_helper_multi_db("INSERT INTO new_db.main.tbl VALUES(1)", {"CREATE TABLE new_db.main.tbl (foo INTEGER)"});
}

TEST_CASE("Test logical_update with catalog", "[serialization]") {
	test_helper_multi_db("UPDATE new_db.main.tbl SET foo=42", {"CREATE TABLE new_db.main.tbl (foo INTEGER)"});
}
