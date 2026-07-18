// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/main/relation.hpp"

namespace duckdb {

class LocalExchangeRelation : public Relation {
public:
	DUCKDB_API LocalExchangeRelation(shared_ptr<Relation> child, idx_t num_partitions);

	idx_t num_partitions; // 0 = auto
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
