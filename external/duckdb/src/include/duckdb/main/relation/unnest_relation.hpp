// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/unnest_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"

namespace duckdb {

class UnnestRelation : public Relation {
public:
	DUCKDB_API UnnestRelation(shared_ptr<Relation> child, string column_name);

	string column_name;
	shared_ptr<Relation> child;

public:
	unique_ptr<QueryNode> GetQueryNode() override;
	BoundStatement Bind(Binder &binder) override;

	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;
	string GetAlias() override;

public:
	bool InheritsColumnBindings() override {
		return false;
	}
	Relation *ChildRelation() override {
		return child.get();
	}

private:
	vector<ColumnDefinition> columns;
};

} // namespace duckdb
