// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/insert_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"
#include "duckdb/planner/expression/bound_parameter_data.hpp"

namespace duckdb {

class InsertStatement;

class InsertRelation : public Relation {
public:
	InsertRelation(shared_ptr<Relation> child, string schema_name, string table_name);
	InsertRelation(shared_ptr<Relation> child, string catalog_name, string schema_name, string table_name);
	InsertRelation(const shared_ptr<ClientContext> &context, unique_ptr<InsertStatement> statement,
	               case_insensitive_map_t<BoundParameterData> parameters);
	~InsertRelation() override;

	shared_ptr<Relation> child;
	string catalog_name;
	string schema_name;
	string table_name;
	vector<ColumnDefinition> columns;
	unique_ptr<InsertStatement> statement;
	case_insensitive_map_t<BoundParameterData> parameters;
	case_insensitive_map_t<unique_ptr<TableRef>> replacement_scans;

public:
	BoundStatement Bind(Binder &binder) override;
	unique_ptr<QueryNode> GetQueryNode() override;
	string GetQuery() override;
	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;
	bool IsReadOnly() override {
		return false;
	}

protected:
	bool CanSerializeToQueryNodeInternal(Binder &) override {
		return false;
	}
};

} // namespace duckdb
