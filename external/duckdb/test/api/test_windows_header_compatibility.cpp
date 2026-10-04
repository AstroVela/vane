#ifdef _WIN32
#include <windows.h>
#endif
#include "catch.hpp"

// Keep the Windows macro regression active on non-Windows CI as well. These
// object-style macros are expanded even when they appear after a scoped enum's
// qualifier.
#ifndef _WIN32
#define CALLBACK
#define OPTIONAL
#define REQUIRED
#define STRICT 1
#endif

#include "test_helpers.hpp"
#include "duckdb.hpp"

#ifndef _WIN32
#undef CALLBACK
#undef OPTIONAL
#undef REQUIRED
#undef STRICT
#endif

using namespace duckdb;
using namespace std;

TEST_CASE("Test compatibility with windows.h", "[windows]") {
	DuckDB db(nullptr);
	Connection con(db);

	// This test solely exists to check if compilation is hindered by including windows.h
	// before including duckdb.hpp
	con.BeginTransaction();
	con.Query("select 42;");
	con.Commit();
}
