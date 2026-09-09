// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/create_table_relation.hpp"
#include "duckdb/main/relation/query_relation.hpp"
#include "duckdb/parser/statement/create_statement.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/bound_parameter_map.hpp"

namespace duckdb {

CreateTableRelation::CreateTableRelation(shared_ptr<Relation> child_p, string schema_name, string table_name,
                                         bool temporary_p, OnCreateConflict on_conflict,
                                         case_insensitive_map_t<unique_ptr<ParsedExpression>> table_options_p,
                                         vector<unique_ptr<ParsedExpression>> partition_keys_p)
    : CreateTableRelation(std::move(child_p), INVALID_CATALOG, std::move(schema_name), std::move(table_name),
                          temporary_p, on_conflict, std::move(table_options_p), std::move(partition_keys_p)) {
}

CreateTableRelation::CreateTableRelation(shared_ptr<Relation> child_p, string catalog_name, string schema_name,
                                         string table_name, bool temporary_p, OnCreateConflict on_conflict,
                                         case_insensitive_map_t<unique_ptr<ParsedExpression>> table_options_p,
                                         vector<unique_ptr<ParsedExpression>> partition_keys_p)
    : Relation(child_p->context, RelationType::CREATE_TABLE_RELATION), child(std::move(child_p)),
      catalog_name(std::move(catalog_name)), schema_name(std::move(schema_name)), table_name(std::move(table_name)),
      temporary(temporary_p), on_conflict(on_conflict), table_options(std::move(table_options_p)),
      partition_keys(std::move(partition_keys_p)) {
	TryBindRelation(columns);
}

CreateTableRelation::CreateTableRelation(const shared_ptr<ClientContext> &context,
                                         unique_ptr<CreateStatement> statement_p,
                                         case_insensitive_map_t<BoundParameterData> parameters_p)
    : Relation(context, RelationType::CREATE_TABLE_RELATION), temporary(false),
      on_conflict(OnCreateConflict::ERROR_ON_CONFLICT), statement(std::move(statement_p)),
      parameters(std::move(parameters_p)) {
	if (!statement || !statement->info || statement->info->type != CatalogType::TABLE_ENTRY) {
		throw InvalidInputException("CreateTableRelation requires a CREATE TABLE AS statement");
	}
	auto &info = statement->info->Cast<CreateTableInfo>();
	if (!info.query) {
		throw InvalidInputException("CreateTableRelation requires a CREATE TABLE AS query");
	}
	// Reuse SQL relation binding to retain typed parameters, output names and
	// Python replacement scans before entering the write runner.
	child = make_shared_ptr<QueryRelation>(context, std::move(info.query), "ctas_source", "", parameters);
	TryBindRelation(columns);
}

CreateTableRelation::~CreateTableRelation() = default;

BoundStatement CreateTableRelation::Bind(Binder &binder) {
	if (statement) {
		BoundParameterMap parameter_map(parameters);
		struct ParameterScope {
			Binder &binder;
			optional_ptr<BoundParameterMap> previous;
			~ParameterScope() {
				binder.SetParameters(previous);
			}
		} scope {binder, binder.GetParameters()};
		binder.SetParameters(parameter_map);
		auto copy = statement->Copy();
		auto &info = copy->Cast<CreateStatement>().info->Cast<CreateTableInfo>();
		info.query = make_uniq<SelectStatement>();
		info.query->node = child->GetQueryNode();
		return binder.Bind(*copy);
	}
	auto query_node = TryGetSerializableChildQueryNode(*child, binder);
	if (!query_node) {
		throw NotImplementedException(
		    "Cannot create a table from a relation that cannot be faithfully represented as a "
		    "SQL query node; conversion would discard the exchange or lose relation bindings");
	}
	auto select = make_uniq<SelectStatement>();
	select->node = std::move(query_node);

	CreateStatement stmt;
	auto info = make_uniq<CreateTableInfo>();
	info->catalog = catalog_name;
	info->schema = schema_name;
	info->table = table_name;
	info->query = std::move(select);
	info->on_conflict = on_conflict;
	info->temporary = temporary;
	for (auto &entry : table_options) {
		info->options.emplace(entry.first, entry.second->Copy());
	}
	for (auto &partition_key : partition_keys) {
		info->partition_keys.push_back(partition_key->Copy());
	}
	stmt.info = std::move(info);
	return binder.Bind(stmt.Cast<SQLStatement>());
}

unique_ptr<QueryNode> CreateTableRelation::GetQueryNode() {
	throw InternalException("Cannot create a query node from a create table relation");
}

string CreateTableRelation::GetQuery() {
	return string();
}

const vector<ColumnDefinition> &CreateTableRelation::Columns() {
	return columns;
}

string CreateTableRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Create Table\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
