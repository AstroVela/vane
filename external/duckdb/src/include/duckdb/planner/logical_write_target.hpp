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
//! created at the same path, while the definition prevents a plan bound against
//! an earlier physical column mapping or write-relevant schema from being reused.
class LogicalWriteTarget {
public:
	LogicalWriteTarget(ClientContext &context, TableCatalogEntry &table);

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
	const string &Definition() const {
		return definition;
	}

private:
	LogicalWriteTarget(string catalog_name, string schema_name, string table_name, string identity, string definition);
	void Validate() const;

private:
	string catalog_name;
	string schema_name;
	string table_name;
	string identity;
	string definition;
};

} // namespace duckdb
