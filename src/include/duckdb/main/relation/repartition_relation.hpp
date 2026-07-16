// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/repartition_relation.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"

namespace duckdb {

class RepartitionRelation : public Relation {
public:
	DUCKDB_API RepartitionRelation(shared_ptr<Relation> child, idx_t num_partitions,
	                               vector<unique_ptr<ParsedExpression>> partition_by);

	idx_t num_partitions; // 0 = auto
	vector<unique_ptr<ParsedExpression>> partition_by;
	shared_ptr<Relation> child;
	vector<ColumnDefinition> columns;

public:
	unique_ptr<QueryNode> GetQueryNode() override;
	BoundStatement Bind(Binder &binder) override;
	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;
	string GetQuery() override;
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
