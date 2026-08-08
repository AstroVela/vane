#include "catch.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension.hpp"
#include "duckdb/main/extension_helper.hpp"
#include "duckdb.h"

using namespace duckdb;

TEST_CASE("Vane fork versions use the SourceID extension identity", "[api][version]") {
	const string version = DuckDB::LibraryVersion();

	REQUIRE(StringUtil::StartsWith(version, "v1.5.0-vane."));
	REQUIRE(string(duckdb_library_version()) == version);
	REQUIRE_FALSE(ExtensionHelper::IsRelease(version));
	REQUIRE(ExtensionHelper::GetVersionDirectoryName() == DuckDB::SourceID());
	REQUIRE(string(Extension::DefaultVersion()) == DuckDB::SourceID());
}
