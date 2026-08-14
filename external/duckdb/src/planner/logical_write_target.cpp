// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/planner/logical_write_target.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

LogicalWriteTarget::LogicalWriteTarget(ClientContext &context, TableCatalogEntry &table)
    : catalog_name(table.catalog.GetName()), schema_name(table.schema.name), table_name(table.name),
      identity(table.GetLogicalWriteTargetIdentity()) {
	if (!identity.empty()) {
		definition = table.GetLogicalWriteTargetDefinition(context);
	}
}

LogicalWriteTarget::LogicalWriteTarget(string catalog_name_p, string schema_name_p, string table_name_p,
                                       string identity_p, string definition_p)
    : catalog_name(std::move(catalog_name_p)), schema_name(std::move(schema_name_p)),
      table_name(std::move(table_name_p)), identity(std::move(identity_p)), definition(std::move(definition_p)) {
}

void LogicalWriteTarget::Validate() const {
	if (catalog_name.empty() || schema_name.empty() || table_name.empty()) {
		throw SerializationException("Cannot serialize a logical write target with an incomplete catalog path");
	}
	if (identity.empty()) {
		throw SerializationException("Table \"%s.%s.%s\" does not provide a logical write target identity",
		                             catalog_name, schema_name, table_name);
	}
	if (definition.empty()) {
		throw SerializationException("Table \"%s.%s.%s\" does not provide a logical write target definition",
		                             catalog_name, schema_name, table_name);
	}
}

TableCatalogEntry &LogicalWriteTarget::Resolve(ClientContext &context) const {
	Validate();
	auto &table = Catalog::GetEntry<TableCatalogEntry>(context, catalog_name, schema_name, table_name);
	auto current_identity = table.GetLogicalWriteTargetIdentity();
	if (current_identity.empty()) {
		throw CatalogException("Table \"%s.%s.%s\" does not provide a logical write target identity", catalog_name,
		                       schema_name, table_name);
	}
	if (current_identity != identity) {
		throw CatalogException("Logical write target \"%s.%s.%s\" was replaced after the plan was bound", catalog_name,
		                       schema_name, table_name);
	}
	auto current_definition = table.GetLogicalWriteTargetDefinition(context);
	if (current_definition.empty()) {
		throw CatalogException("Table \"%s.%s.%s\" does not provide a logical write target definition", catalog_name,
		                       schema_name, table_name);
	}
	if (current_definition != definition) {
		throw CatalogException("Logical write target \"%s.%s.%s\" definition changed after the plan was bound",
		                       catalog_name, schema_name, table_name);
	}
	return table;
}

void LogicalWriteTarget::Serialize(Serializer &serializer) const {
	Validate();
	serializer.WriteProperty<string>(100, "catalog_name", catalog_name);
	serializer.WriteProperty<string>(101, "schema_name", schema_name);
	serializer.WriteProperty<string>(102, "table_name", table_name);
	serializer.WriteProperty<string>(103, "identity", identity);
	serializer.WriteProperty<string>(104, "definition", definition);
}

LogicalWriteTarget LogicalWriteTarget::Deserialize(Deserializer &deserializer) {
	auto catalog_name = deserializer.ReadProperty<string>(100, "catalog_name");
	auto schema_name = deserializer.ReadProperty<string>(101, "schema_name");
	auto table_name = deserializer.ReadProperty<string>(102, "table_name");
	auto identity = deserializer.ReadProperty<string>(103, "identity");
	auto definition = deserializer.ReadProperty<string>(104, "definition");
	LogicalWriteTarget result(std::move(catalog_name), std::move(schema_name), std::move(table_name),
	                          std::move(identity), std::move(definition));
	result.Validate();
	return result;
}

} // namespace duckdb
