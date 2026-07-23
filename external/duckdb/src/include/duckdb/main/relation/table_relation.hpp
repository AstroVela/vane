// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/table_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"
#include "duckdb/main/table_description.hpp"

namespace duckdb {

class TableRelation : public Relation {
public:
	TableRelation(const shared_ptr<ClientContext> &context, unique_ptr<TableDescription> description);
	TableRelation(const shared_ptr<RelationContextWrapper> &context, unique_ptr<TableDescription> description);

	unique_ptr<TableDescription> description;

public:
	unique_ptr<QueryNode> GetQueryNode() override;

	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;
	string GetAlias() override;

	void Insert(const vector<vector<Value>> &values) override;
	void Insert(vector<vector<unique_ptr<ParsedExpression>>> &&expressions) override;
	void Update(const string &update, const string &condition = string()) override;
	void Update(vector<string> column_names, vector<unique_ptr<ParsedExpression>> &&update,
	            unique_ptr<ParsedExpression> condition = nullptr) override;
	void Delete(const string &condition = string()) override;

protected:
	unique_ptr<TableRef> GetTableRefInternal() override;
	BoundStatement BindAsInput(Binder &binder) override;
};

} // namespace duckdb
