#include "catch.hpp"
#include "test_helpers.hpp"

using namespace duckdb;
using namespace std;

static void CreateSimpleTable(Connection &con) {
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE a (i TINYINT)"));
	REQUIRE_NO_FAIL(con.Query("INSERT INTO a VALUES (11), (12), (13)"));
}

static void ModifySimpleTable(Connection &con) {
	REQUIRE_NO_FAIL(con.Query("INSERT INTO a VALUES (14)"));
	REQUIRE_NO_FAIL(con.Query("DELETE FROM a where i=12"));
}

static void CheckSimpleQuery(Connection &con) {
	auto statements = con.ExtractStatements("SELECT COUNT(*) FROM a WHERE i=12");
	REQUIRE(statements.size() == 1);
	duckdb::vector<duckdb::Value> values = {Value(12)};
	auto pending_result = con.PendingQuery("SELECT COUNT(*) FROM a WHERE i=?", values, true);

	if (pending_result->HasError()) {
		printf("%s\n", pending_result->GetError().c_str());
	}

	REQUIRE(!pending_result->HasError());

	auto result = pending_result->Execute();
	REQUIRE(CHECK_COLUMN(result, 0, {1}));
}

static void CheckCatalogErrorQuery(Connection &con) {
	duckdb::vector<Value> values = {Value(12)};
	auto pending_result = con.PendingQuery("SELECT COUNT(*) FROM b WHERE i=?", values, true);
	REQUIRE((pending_result->HasError() && pending_result->GetErrorType() == ExceptionType::CATALOG));
}

static void CheckConversionErrorQuery(Connection &con) {
	// Check query with invalid prepared value
	duckdb::vector<Value> values = {Value("fawakaaniffoo")};
	auto pending_result = con.PendingQuery("SELECT COUNT(*) FROM a WHERE i=?", values, true);
	REQUIRE(!pending_result->HasError());
	auto result = pending_result->Execute();
	REQUIRE((result->HasError() && result->GetErrorType() == ExceptionType::CONVERSION));
}

static void CheckSimpleQueryAfterModification(Connection &con) {
	duckdb::vector<Value> values = {Value(14)};
	auto pending_result = con.PendingQuery("SELECT COUNT(*) FROM a WHERE i=?", values, true);
	REQUIRE(!pending_result->HasError());
	auto result = pending_result->Execute();
	REQUIRE(CHECK_COLUMN(result, 0, {1}));
}

TEST_CASE("Pending Query with Parameters", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	con.EnableQueryVerification();

	CreateSimpleTable(con);
	CheckSimpleQuery(con);
	CheckSimpleQuery(con);
}

TEST_CASE("Pending Query with Parameters Catalog Error", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	con.EnableQueryVerification();

	CreateSimpleTable(con);

	CheckCatalogErrorQuery(con);

	// Verify things are still sane
	CheckSimpleQuery(con);
}

TEST_CASE("Pending Query with Parameters Type Conversion Error", "[api]") {
	DuckDB db(nullptr);
	Connection con(db);
	con.EnableQueryVerification();

	CreateSimpleTable(con);

	CheckConversionErrorQuery(con);

	// Verify things are still sane
	CheckSimpleQuery(con);
}

TEST_CASE("Pending Query with Parameters with transactions", "[api]") {
	DuckDB db(nullptr);
	Connection con1(db);
	Connection con2(db);
	duckdb::vector<Value> empty_values = {};
	con1.EnableQueryVerification();

	CreateSimpleTable(con1);

	// CheckConversionErrorQuery(con1);

	// Begin a transaction in the PrepareAndExecute
	auto pending_result1 = con1.PendingQuery("BEGIN TRANSACTION", empty_values, true);
	if (pending_result1->HasError()) {
		printf("%s\n", pending_result1->GetError().c_str());
	}
	REQUIRE(!pending_result1->HasError());

	auto result1 = pending_result1->Execute();
	REQUIRE(!result1->HasError());
	CheckSimpleQuery(con1);

	// Modify table on other connection, leaving transaction open
	con2.BeginTransaction();
	ModifySimpleTable(con2);
	CheckSimpleQueryAfterModification(con2);

	// con1 sees nothing: both transactions are open
	CheckSimpleQuery(con1);

	con2.Commit();

	// con1 still sees nothing: its transaction was started before con2's
	CheckSimpleQuery(con1);

	// con 1 commits
	auto pending_result2 = con1.PendingQuery("COMMIT", empty_values, true);
	auto result2 = pending_result2->Execute();
	REQUIRE(!result2->HasError());

	// now con1 should see changes from con2
	CheckSimpleQueryAfterModification(con1);
	CheckSimpleQueryAfterModification(con2);
}

TEST_CASE("FILE query parameters are validated before storage", "[api][file]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("CREATE TABLE files(value FILE)"));

	SECTION("top-level FILE values are validated") {
		auto invalid_file = Value::STRUCT(FileLogicalType::Create(), {Value(), Value(), Value(), Value(), Value()});
		auto result = con.Query("INSERT INTO files VALUES (?)", invalid_file);
		REQUIRE(result->HasError());
		REQUIRE(StringUtil::Contains(result->GetError(), "Query parameter FILE url cannot be NULL"));

		auto count = con.Query("SELECT count(*) FROM files");
		REQUIRE(CHECK_COLUMN(count, 0, {0}));
	}

	SECTION("nested FILE values are validated recursively") {
		REQUIRE_NO_FAIL(con.Query("CREATE TABLE nested_files(value STRUCT(payload FILE))"));
		auto file_type = FileLogicalType::Create();
		auto invalid_file = Value::STRUCT(file_type, {Value("object"), Value(), Value(), Value(), Value("invalid")});
		auto struct_type = LogicalType::STRUCT({{"payload", file_type}});
		auto nested_file = Value::STRUCT(struct_type, {std::move(invalid_file)});

		auto result = con.Query("INSERT INTO nested_files VALUES (?)", nested_file);
		REQUIRE(result->HasError());
		REQUIRE(StringUtil::Contains(result->GetError(),
		                             "Query parameter FILE checksum must have the form <algorithm>:<digest>"));

		auto count = con.Query("SELECT count(*) FROM nested_files");
		REQUIRE(CHECK_COLUMN(count, 0, {0}));
	}

	SECTION("malformed FILE field counts are rejected safely") {
		auto file_type = FileLogicalType::Create();
		REQUIRE_THROWS_WITH(Value::STRUCT(file_type, {Value("object"), Value(), Value(), Value()}),
		                    Catch::Matchers::Contains("STRUCT value requires 5 fields, but 4 values were provided"));
		REQUIRE_THROWS_WITH(Value::STRUCT(file_type, {Value("object"), Value(), Value(), Value(), Value(), Value()}),
		                    Catch::Matchers::Contains("STRUCT value requires 5 fields, but 6 values were provided"));

		auto four_field_type = LogicalType::STRUCT({{"url", LogicalType::VARCHAR},
		                                            {"content_type", LogicalType::VARCHAR},
		                                            {"position", LogicalType::BIGINT},
		                                            {"size", LogicalType::BIGINT}});
		auto malformed_file = Value::STRUCT(four_field_type, {Value("object"), Value(), Value(), Value()});
		malformed_file.Reinterpret(std::move(file_type));
		auto result = con.Query("SELECT ?", malformed_file);
		REQUIRE(result->HasError());
		REQUIRE(StringUtil::Contains(result->GetError(), "Query parameter FILE must contain exactly 5 fields"));
	}

	SECTION("valid FILE values remain accepted") {
		auto valid_file =
		    Value::STRUCT(FileLogicalType::Create(), {Value("object"), Value("application/octet-stream"),
		                                              Value::BIGINT(0), Value::BIGINT(1), Value("sha256:digest")});
		REQUIRE_NO_FAIL(con.Query("INSERT INTO files VALUES (?)", valid_file));

		auto result = con.Query("SELECT value.url, value.position, value.size FROM files");
		REQUIRE(CHECK_COLUMN(result, 0, {"object"}));
		REQUIRE(CHECK_COLUMN(result, 1, {0}));
		REQUIRE(CHECK_COLUMN(result, 2, {1}));
	}
}
