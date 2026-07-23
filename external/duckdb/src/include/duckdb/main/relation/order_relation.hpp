// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/order_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"
#include "duckdb/parser/parsed_expression.hpp"
#include "duckdb/parser/result_modifier.hpp"

namespace duckdb {

class OrderRelation : public Relation {
public:
	DUCKDB_API OrderRelation(shared_ptr<Relation> child, vector<OrderByNode> orders);

	vector<OrderByNode> orders;
	shared_ptr<Relation> child;
	vector<ColumnDefinition> columns;

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
	bool CanSerializeToQueryNodeInternal(Binder &binder) override;
	bool CanBindAsInputInternal(Binder &binder) override {
		return child->CanBindAsInputInternal(binder);
	}

protected:
	BoundStatement BindAsInput(Binder &binder) override;
};

} // namespace duckdb
