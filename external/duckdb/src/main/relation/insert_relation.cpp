// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/insert_relation.hpp"
#include "duckdb/parser/statement/insert_statement.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/bound_parameter_map.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

InsertRelation::InsertRelation(shared_ptr<Relation> child_p, string schema_name, string table_name)
    : Relation(child_p->context, RelationType::INSERT_RELATION), child(std::move(child_p)),
      schema_name(std::move(schema_name)), table_name(std::move(table_name)) {
	TryBindRelation(columns);
}

InsertRelation::InsertRelation(shared_ptr<Relation> child_p, string catalog_name, string schema_name, string table_name)
    : Relation(child_p->context, RelationType::INSERT_RELATION), child(std::move(child_p)),
      catalog_name(std::move(catalog_name)), schema_name(std::move(schema_name)), table_name(std::move(table_name)) {
	TryBindRelation(columns);
}

InsertRelation::InsertRelation(const shared_ptr<ClientContext> &context, unique_ptr<InsertStatement> statement_p,
                               case_insensitive_map_t<BoundParameterData> parameters_p)
    : Relation(context, RelationType::INSERT_RELATION), statement(std::move(statement_p)),
      parameters(std::move(parameters_p)) {
	if (!statement) {
		throw InvalidInputException("InsertRelation requires an INSERT statement");
	}
	TryBindRelation(columns);
}

InsertRelation::~InsertRelation() = default;

BoundStatement InsertRelation::Bind(Binder &binder) {
	if (statement) {
		// Bind the complete INSERT so target-column inference, DEFAULT values,
		// BY NAME and CTE scopes retain DuckDB's statement semantics.
		BoundParameterMap parameter_map(parameters);
		struct ParameterScope {
			Binder &binder;
			optional_ptr<BoundParameterMap> previous;
			BindingMode previous_mode;
			optional_ptr<const case_insensitive_map_t<unique_ptr<TableRef>>> previous_scans;
			~ParameterScope() {
				binder.SetParameters(previous);
				binder.SetBindingMode(previous_mode);
				binder.SetReplacementScanBindings(previous_scans);
			}
		} scope {binder, binder.GetParameters(), binder.GetBindingMode(), binder.GetReplacementScanBindings()};
		binder.SetParameters(parameter_map);
		binder.SetReplacementScanBindings(replacement_scans);
		binder.SetBindingMode(BindingMode::EXTRACT_REPLACEMENT_SCANS);
		auto copy = statement->Copy();
		auto result = binder.Bind(*copy);
		if (columns.empty()) {
			// Preserve Python sources from the caller's frame. Keep the complete
			// INSERT AST: wrapping VALUES in a SELECT would lose DEFAULT inference.
			for (auto &entry : binder.GetReplacementScans()) {
				if (entry.second->external_dependency) {
					replacement_scans.emplace(entry.first, entry.second->Copy());
				}
			}
		}
		return result;
	}
	auto query_node = TryGetSerializableChildQueryNode(*child, binder);
	if (!query_node) {
		throw NotImplementedException("Cannot insert from a relation that cannot be faithfully represented as a SQL "
		                              "query node; conversion would discard the exchange or lose relation bindings");
	}
	InsertStatement stmt;
	auto select = make_uniq<SelectStatement>();
	select->node = std::move(query_node);

	stmt.catalog = catalog_name;
	stmt.schema = schema_name;
	stmt.table = table_name;
	stmt.select_statement = std::move(select);
	return binder.Bind(stmt.Cast<SQLStatement>());
}

unique_ptr<QueryNode> InsertRelation::GetQueryNode() {
	throw InternalException("Cannot create a query node from an insert relation");
}

string InsertRelation::GetQuery() {
	return string();
}

const vector<ColumnDefinition> &InsertRelation::Columns() {
	return columns;
}

string InsertRelation::ToString(idx_t depth) {
	if (statement) {
		return RenderWhitespace(depth) + "Insert From SQL\n";
	}
	string str = RenderWhitespace(depth) + "Insert\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
