#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb_python/pytype.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"

namespace duckdb {

class DuckDBPyFunctional {
public:
	DuckDBPyFunctional() = delete;

public:
	static void Initialize(nb::module_ &m);
};

} // namespace duckdb
