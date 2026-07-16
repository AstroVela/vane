// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/limit_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"

namespace duckdb {

class LimitRelation : public Relation {
public:
	DUCKDB_API LimitRelation(shared_ptr<Relation> child, int64_t limit, int64_t offset);

	int64_t limit;
	int64_t offset;
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
