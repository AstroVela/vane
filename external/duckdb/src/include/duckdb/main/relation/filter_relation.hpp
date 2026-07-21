// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/filter_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"
#include "duckdb/parser/parsed_expression.hpp"

namespace duckdb {

class FilterRelation : public Relation {
public:
	DUCKDB_API FilterRelation(shared_ptr<Relation> child, unique_ptr<ParsedExpression> condition);

	unique_ptr<ParsedExpression> condition;
	shared_ptr<Relation> child;

public:
	unique_ptr<QueryNode> GetQueryNode() override;
	BoundStatement Bind(Binder &binder) override;

	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;
	string GetAlias() override;

public:
	bool InheritsColumnBindings() override {
		return true;
	}
	Relation *ChildRelation() override {
		return child.get();
	}
};

} // namespace duckdb
