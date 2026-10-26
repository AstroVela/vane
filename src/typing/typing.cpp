#include "duckdb_python/typing.hpp"
#include "duckdb_python/pytype.hpp"

namespace duckdb {

//! Heap-allocate an owned DuckDBPyType. Spelled std::unique_ptr (not duckdb::unique_ptr) so the `m.attr(...) =`
//! assignment finds nanobind's type_caster<std::unique_ptr<T>> and transfers ownership to Python.
static std::unique_ptr<DuckDBPyType> MakeType(LogicalType type) {
	return make_uniq<DuckDBPyType>(std::move(type));
}

static void DefineBaseTypes(nb::handle &m) {
	m.attr("SQLNULL") = MakeType(LogicalType::SQLNULL);
	m.attr("BOOLEAN") = MakeType(LogicalType::BOOLEAN);
	m.attr("TINYINT") = MakeType(LogicalType::TINYINT);
	m.attr("UTINYINT") = MakeType(LogicalType::UTINYINT);
	m.attr("SMALLINT") = MakeType(LogicalType::SMALLINT);
	m.attr("USMALLINT") = MakeType(LogicalType::USMALLINT);
	m.attr("INTEGER") = MakeType(LogicalType::INTEGER);
	m.attr("UINTEGER") = MakeType(LogicalType::UINTEGER);
	m.attr("BIGINT") = MakeType(LogicalType::BIGINT);
	m.attr("UBIGINT") = MakeType(LogicalType::UBIGINT);
	m.attr("HUGEINT") = MakeType(LogicalType::HUGEINT);
	m.attr("UHUGEINT") = MakeType(LogicalType::UHUGEINT);
	m.attr("UUID") = MakeType(LogicalType::UUID);
	m.attr("FLOAT") = MakeType(LogicalType::FLOAT);
	m.attr("DOUBLE") = MakeType(LogicalType::DOUBLE);
	m.attr("DATE") = MakeType(LogicalType::DATE);

	m.attr("TIMESTAMP") = MakeType(LogicalType::TIMESTAMP);
	m.attr("TIMESTAMP_MS") = MakeType(LogicalType::TIMESTAMP_MS);
	m.attr("TIMESTAMP_NS") = MakeType(LogicalType::TIMESTAMP_NS);
	m.attr("TIMESTAMP_S") = MakeType(LogicalType::TIMESTAMP_S);

	m.attr("TIME") = MakeType(LogicalType::TIME);
	m.attr("TIME_NS") = MakeType(LogicalType::TIME_NS);

	m.attr("TIME_TZ") = MakeType(LogicalType::TIME_TZ);
	m.attr("TIMESTAMP_TZ") = MakeType(LogicalType::TIMESTAMP_TZ);

	m.attr("VARCHAR") = MakeType(LogicalType::VARCHAR);

	m.attr("BLOB") = MakeType(LogicalType::BLOB);
	m.attr("BIT") = MakeType(LogicalType::BIT);
	m.attr("INTERVAL") = MakeType(LogicalType::INTERVAL);
	m.attr("VARIANT") = MakeType(LogicalType::VARIANT());
}

void DuckDBPyTyping::Initialize(nb::module_ &parent) {
	auto m = parent.def_submodule("_sqltypes", "This module contains classes and methods related to typing");
	DuckDBPyType::Initialize(m);

	DefineBaseTypes(m);
}

} // namespace duckdb
