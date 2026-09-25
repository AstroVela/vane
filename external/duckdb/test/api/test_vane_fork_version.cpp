#include "catch.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb.h"

using namespace duckdb;

TEST_CASE("Vane fork versions use the SourceID extension identity", "[api][version]") {
	const string version = DuckDB::LibraryVersion();
	const string expected_prefix =
	    StringUtil::Format("v%d.%d.%d-vane.", DUCKDB_MAJOR_VERSION, DUCKDB_MINOR_VERSION, DUCKDB_PATCH_VERSION);

	REQUIRE(StringUtil::StartsWith(version, expected_prefix));
	REQUIRE(string(duckdb_library_version()) == version);
	REQUIRE_FALSE(ExtensionHelper::IsRelease(version));
	REQUIRE(ExtensionHelper::GetVersionDirectoryName() == DuckDB::SourceID());
	REQUIRE(string(Extension::DefaultVersion()) == DuckDB::SourceID());
}
