// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/create_table_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"

namespace duckdb {

class CreateTableRelation : public Relation {
public:
	CreateTableRelation(shared_ptr<Relation> child, string schema_name, string table_name, bool temporary,
	                    OnCreateConflict on_conflict,
	                    case_insensitive_map_t<unique_ptr<ParsedExpression>> table_options = {},
	                    vector<unique_ptr<ParsedExpression>> partition_keys = {});
	CreateTableRelation(shared_ptr<Relation> child, string catalog_name, string schema_name, string table_name,
	                    bool temporary, OnCreateConflict on_conflict,
	                    case_insensitive_map_t<unique_ptr<ParsedExpression>> table_options = {},
	                    vector<unique_ptr<ParsedExpression>> partition_keys = {});

	shared_ptr<Relation> child;
	string catalog_name;
	string schema_name;
	string table_name;
	vector<ColumnDefinition> columns;
	bool temporary;
	OnCreateConflict on_conflict;
	case_insensitive_map_t<unique_ptr<ParsedExpression>> table_options;
	vector<unique_ptr<ParsedExpression>> partition_keys;

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
