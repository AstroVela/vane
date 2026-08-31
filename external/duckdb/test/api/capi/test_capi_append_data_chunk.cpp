#include "capi_tester.hpp"

using namespace duckdb;
using namespace std;

namespace {

static duckdb_value CreateCAPIFileValue(duckdb_logical_type type, const char *url, bool has_position, int64_t position,
                                        bool has_size, int64_t size, const char *checksum) {
	duckdb_value fields[5] = {url ? duckdb_create_varchar(url) : duckdb_create_null_value(), duckdb_create_null_value(),
	                          has_position ? duckdb_create_int64(position) : duckdb_create_null_value(),
	                          has_size ? duckdb_create_int64(size) : duckdb_create_null_value(),
	                          checksum ? duckdb_create_varchar(checksum) : duckdb_create_null_value()};
	auto result = duckdb_create_struct_value(type, fields);
	for (auto &field : fields) {
		duckdb_destroy_value(&field);
	}
	return result;
}

} // namespace

TEST_CASE("Test casting columns in AppendDataChunk in C API", "[capi]") {
	duckdb::vector<string> tables;
	tables.push_back("CREATE TABLE test(i BIGINT, j VARCHAR);");
	tables.push_back("CREATE TABLE test(i BIGINT, j BOOLEAN);");

	for (idx_t i = 0; i < tables.size(); i++) {
		CAPITester tester;
		REQUIRE(tester.OpenDatabase(nullptr));
		REQUIRE(duckdb_vector_size() == STANDARD_VECTOR_SIZE);

		tester.Query(tables[i]);

		duckdb_logical_type types[2];
		types[0] = duckdb_create_logical_type(DUCKDB_TYPE_SMALLINT);
		types[1] = duckdb_create_logical_type(DUCKDB_TYPE_BOOLEAN);

		auto data_chunk = duckdb_create_data_chunk(types, 2);
		REQUIRE(data_chunk);

		auto smallint_col = duckdb_data_chunk_get_vector(data_chunk, 0);
		auto boolean_col = duckdb_data_chunk_get_vector(data_chunk, 1);

		auto smallint_data = reinterpret_cast<int16_t *>(duckdb_vector_get_data(smallint_col));
		smallint_data[0] = 15;
		smallint_data[1] = -15;

		auto boolean_data = reinterpret_cast<bool *>(duckdb_vector_get_data(boolean_col));
		boolean_data[0] = false;
		boolean_data[1] = true;

		duckdb_data_chunk_set_size(data_chunk, 2);

		duckdb_appender appender;
		auto status = duckdb_appender_create(tester.connection, nullptr, "test", &appender);
		REQUIRE(status == DuckDBSuccess);

		REQUIRE(duckdb_append_data_chunk(appender, data_chunk) == DuckDBSuccess);
		duckdb_appender_close(appender);

		auto result = tester.Query("SELECT i, j FROM test;");
		REQUIRE(result->Fetch<int64_t>(0, 0) == 15);
		REQUIRE(result->Fetch<int64_t>(0, 1) == -15);
		auto str = result->Fetch<string>(1, 0);
		REQUIRE(str.compare("false") == 0);
		str = result->Fetch<string>(1, 1);
		REQUIRE(str.compare("true") == 0);

		duckdb_appender_destroy(&appender);
		duckdb_destroy_data_chunk(&data_chunk);
		duckdb_destroy_logical_type(&types[0]);
		duckdb_destroy_logical_type(&types[1]);
	}
}

TEST_CASE("FILE values are governed at C value and appender boundaries", "[capi][file]") {
	CAPITester tester;
	REQUIRE(tester.OpenDatabase(nullptr));
	REQUIRE_NO_FAIL(tester.Query("LOAD file"));
	REQUIRE_NO_FAIL(tester.Query("CREATE TABLE file_values(value IMAGEFILE)"));

	duckdb_appender appender;
	REQUIRE(duckdb_appender_create(tester.connection, nullptr, "file_values", &appender) == DuckDBSuccess);
	auto image_file_type = duckdb_appender_column_type(appender, 0);
	REQUIRE(image_file_type);

	// Valid exact-typed values remain constructible and use the documented C
	// string-representation API without exposing a SQL cast.
	auto valid_file = CreateCAPIFileValue(image_file_type, "image.png", true, 1, true, 2, "sha256:abc");
	REQUIRE(valid_file);
	auto rendered = duckdb_get_varchar(valid_file);
	REQUIRE(rendered);
	REQUIRE(StringUtil::Contains(rendered, "image.png"));
	duckdb_free(rendered);
	REQUIRE(duckdb_get_int64(valid_file) == NumericLimits<int64_t>::Minimum());
	auto invalid_bignum = duckdb_get_bignum(valid_file);
	REQUIRE(!invalid_bignum.data);
	REQUIRE(invalid_bignum.size == 0);
	auto invalid_blob = duckdb_get_blob(valid_file);
	REQUIRE(!invalid_blob.data);
	REQUIRE(invalid_blob.size == 0);
	auto invalid_bit = duckdb_get_bit(valid_file);
	REQUIRE(!invalid_bit.data);
	REQUIRE(invalid_bit.size == 0);

	auto sql_string = duckdb_value_to_string(valid_file);
	REQUIRE(sql_string);
	REQUIRE(StringUtil::Contains(sql_string, "image_file(file("));
	auto roundtrip_result = tester.Query("SELECT typeof(" + string(sql_string) + "), (" + string(sql_string) + ").url");
	REQUIRE_NO_FAIL(*roundtrip_result);
	REQUIRE(roundtrip_result->Fetch<string>(0, 0) == "IMAGEFILE");
	REQUIRE(roundtrip_result->Fetch<string>(1, 0) == "image.png");
	duckdb_free(sql_string);

	// C value construction rejects every governed field invariant before the
	// invalid logical value can reach an appender.
	auto invalid_file = CreateCAPIFileValue(image_file_type, nullptr, false, 0, false, 0, nullptr);
	REQUIRE(!invalid_file);
	invalid_file = CreateCAPIFileValue(image_file_type, "image.png", true, 1, false, 0, nullptr);
	REQUIRE(!invalid_file);
	invalid_file = CreateCAPIFileValue(image_file_type, "image.png", true, -1, true, 2, nullptr);
	REQUIRE(!invalid_file);
	invalid_file = CreateCAPIFileValue(image_file_type, "image.png", false, 0, false, 0, "invalid");
	REQUIRE(!invalid_file);

	REQUIRE(duckdb_appender_begin_row(appender) == DuckDBSuccess);
	REQUIRE(duckdb_append_value(appender, valid_file) == DuckDBSuccess);
	REQUIRE(duckdb_appender_end_row(appender) == DuckDBSuccess);
	REQUIRE(duckdb_appender_flush(appender) == DuckDBSuccess);
	REQUIRE(duckdb_appender_destroy(&appender) == DuckDBSuccess);
	duckdb_destroy_value(&valid_file);

	// Direct vector writes can bypass value constructors. The data-chunk
	// appender validates exact-typed FILE values before accepting the chunk.
	auto data_chunk = duckdb_create_data_chunk(&image_file_type, 1);
	REQUIRE(data_chunk);
	auto file_vector = duckdb_data_chunk_get_vector(data_chunk, 0);
	auto url_vector = duckdb_struct_vector_get_child(file_vector, 0);
	auto content_type_vector = duckdb_struct_vector_get_child(file_vector, 1);
	auto position_vector = duckdb_struct_vector_get_child(file_vector, 2);
	auto size_vector = duckdb_struct_vector_get_child(file_vector, 3);
	auto checksum_vector = duckdb_struct_vector_get_child(file_vector, 4);
	duckdb_vector_assign_string_element(url_vector, 0, "invalid-range.png");
	reinterpret_cast<int64_t *>(duckdb_vector_get_data(position_vector))[0] = 1;
	for (auto vector : {content_type_vector, size_vector, checksum_vector}) {
		duckdb_vector_ensure_validity_writable(vector);
		duckdb_validity_set_row_invalid(duckdb_vector_get_validity(vector), 0);
	}
	duckdb_data_chunk_set_size(data_chunk, 1);

	REQUIRE(duckdb_appender_create(tester.connection, nullptr, "file_values", &appender) == DuckDBSuccess);
	REQUIRE(duckdb_append_data_chunk(appender, data_chunk) == DuckDBError);
	auto appender_error = duckdb_appender_error(appender);
	REQUIRE(appender_error);
	REQUIRE(StringUtil::Contains(appender_error, "position and size"));
	REQUIRE(duckdb_appender_destroy(&appender) == DuckDBSuccess);

	auto result = tester.Query("SELECT count(*) FROM file_values");
	REQUIRE_NO_FAIL(*result);
	REQUIRE(result->Fetch<int64_t>(0, 0) == 1);

	duckdb_destroy_data_chunk(&data_chunk);
	duckdb_destroy_logical_type(&image_file_type);
}

TEST_CASE("Test casting error in AppendDataChunk in C API", "[capi]") {
	CAPITester tester;
	REQUIRE(tester.OpenDatabase(nullptr));
	REQUIRE(duckdb_vector_size() == STANDARD_VECTOR_SIZE);

	tester.Query("CREATE TABLE test(i BIGINT, j BOOLEAN[]);");

	duckdb_logical_type types[2];
	types[0] = duckdb_create_logical_type(DUCKDB_TYPE_SMALLINT);
	types[1] = duckdb_create_logical_type(DUCKDB_TYPE_BOOLEAN);

	auto data_chunk = duckdb_create_data_chunk(types, 2);
	REQUIRE(data_chunk);

	auto smallint_col = duckdb_data_chunk_get_vector(data_chunk, 0);
	auto boolean_col = duckdb_data_chunk_get_vector(data_chunk, 1);

	auto smallint_data = reinterpret_cast<int16_t *>(duckdb_vector_get_data(smallint_col));
	smallint_data[0] = 15;
	smallint_data[1] = -15;

	auto boolean_data = reinterpret_cast<bool *>(duckdb_vector_get_data(boolean_col));
	boolean_data[0] = false;
	boolean_data[1] = true;

	duckdb_data_chunk_set_size(data_chunk, 2);

	duckdb_appender appender;
	auto status = duckdb_appender_create(tester.connection, nullptr, "test", &appender);
	REQUIRE(status == DuckDBSuccess);

	REQUIRE(duckdb_append_data_chunk(appender, data_chunk) == DuckDBError);
	auto error_msg = duckdb_appender_error(appender);
	REQUIRE(string(error_msg) == "type mismatch in AppendDataChunk, expected BOOLEAN[], got BOOLEAN for column 1");

	duckdb_appender_close(appender);
	duckdb_appender_destroy(&appender);
	duckdb_destroy_data_chunk(&data_chunk);
	duckdb_destroy_logical_type(&types[0]);
	duckdb_destroy_logical_type(&types[1]);
}

TEST_CASE("Test casting timestamps in AppendDataChunk in C API", "[capi]") {
	CAPITester tester;
	REQUIRE(tester.OpenDatabase(nullptr));
	REQUIRE(duckdb_vector_size() == STANDARD_VECTOR_SIZE);

	tester.Query("CREATE TABLE test(i TIMESTAMP, j DATE);");

	duckdb_logical_type types[2];
	types[0] = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
	types[1] = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);

	auto data_chunk = duckdb_create_data_chunk(types, 2);
	REQUIRE(data_chunk);

	auto ts_column = duckdb_data_chunk_get_vector(data_chunk, 0);
	auto date_column = duckdb_data_chunk_get_vector(data_chunk, 1);

	duckdb_vector_assign_string_element(ts_column, 0, "2017-07-23 13:10:11");
	duckdb_vector_assign_string_element(date_column, 0, "1993-08-14");
	duckdb_data_chunk_set_size(data_chunk, 1);

	duckdb_appender appender;
	auto status = duckdb_appender_create(tester.connection, nullptr, "test", &appender);
	REQUIRE(status == DuckDBSuccess);

	REQUIRE(duckdb_append_data_chunk(appender, data_chunk) == DuckDBSuccess);
	duckdb_appender_close(appender);

	auto result = tester.Query("SELECT i::VARCHAR, j::VARCHAR FROM test;");
	auto str = result->Fetch<string>(0, 0);
	REQUIRE(str.compare("2017-07-23 13:10:11") == 0);
	str = result->Fetch<string>(1, 0);
	REQUIRE(str.compare("1993-08-14") == 0);

	duckdb_appender_destroy(&appender);
	duckdb_destroy_data_chunk(&data_chunk);
	duckdb_destroy_logical_type(&types[0]);
	duckdb_destroy_logical_type(&types[1]);
}
