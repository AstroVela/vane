// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/string.hpp"

namespace duckdb {

class ClientContext;
class Deserializer;
class Serializer;
class TableCatalogEntry;

//! A write target captured when a logical DML operator is bound.
//!
//! The catalog path resolves the target on the receiving process. The identity
//! distinguishes a particular table incarnation from a different table later
//! created at the same path.
class LogicalWriteTarget {
public:
	explicit LogicalWriteTarget(const TableCatalogEntry &table);

	TableCatalogEntry &Resolve(ClientContext &context) const;

	void Serialize(Serializer &serializer) const;
	static LogicalWriteTarget Deserialize(Deserializer &deserializer);

	const string &CatalogName() const {
		return catalog_name;
	}
	const string &SchemaName() const {
		return schema_name;
	}
	const string &TableName() const {
		return table_name;
	}
	const string &Identity() const {
		return identity;
	}

private:
	LogicalWriteTarget(string catalog_name, string schema_name, string table_name, string identity);
	void Validate() const;

private:
	string catalog_name;
	string schema_name;
	string table_name;
	string identity;
};

} // namespace duckdb
