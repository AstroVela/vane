// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/subquery_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"

namespace duckdb {

class SubqueryRelation : public Relation {
public:
	SubqueryRelation(shared_ptr<Relation> child, const string &alias);
	shared_ptr<Relation> child;

public:
	unique_ptr<QueryNode> GetQueryNode() override;
	BoundStatement Bind(Binder &binder) override;

	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;

public:
	bool InheritsColumnBindings() override {
		return false;
	}
	Relation *ChildRelation() override {
		return child.get();
	}
	bool CanBindAsInputInternal(Binder &binder) override {
		return child->CanBindAsInputInternal(binder);
	}
};

} // namespace duckdb
