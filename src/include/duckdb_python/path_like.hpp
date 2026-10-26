#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb_python/nb/casters.hpp"
#include "duckdb/main/external_dependencies.hpp"
#include "duckdb/common/types/value.hpp"

namespace duckdb {

struct DuckDBPyConnection;

struct PathLike {
	static PathLike Create(const nb::object &object, DuckDBPyConnection &connection);
	// The file(s) extracted from object
	vector<string> files;
	shared_ptr<ExternalDependency> dependency;
};

} // namespace duckdb
