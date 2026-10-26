//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb_python/pystatement.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb_python/nb/casters.hpp"
#include "duckdb.hpp"

namespace duckdb {

struct DuckDBPyStatement {
public:
	explicit DuckDBPyStatement(unique_ptr<SQLStatement> statement);

public:
	//! Create a copy of the wrapped statement
	unique_ptr<SQLStatement> GetStatement();
	string Query() const;
	nb::set NamedParameters() const;
	StatementType Type() const;
	nb::list ExpectedResultType() const;

public:
	static void Initialize(nb::handle &m);

private:
	unique_ptr<SQLStatement> statement;
};

} // namespace duckdb
