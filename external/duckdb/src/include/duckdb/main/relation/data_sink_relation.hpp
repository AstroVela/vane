// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/main/relation.hpp"

namespace duckdb {

class DataSinkRelation : public Relation {
public:
	DUCKDB_API DataSinkRelation(shared_ptr<Relation> child, string operation_id);

	shared_ptr<Relation> child;
	string operation_id;
	vector<ColumnDefinition> columns;

	unique_ptr<QueryNode> GetQueryNode() override;
	BoundStatement Bind(Binder &binder) override;
	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;
	string GetQuery() override;
	string GetAlias() override;

	bool InheritsColumnBindings() override {
		return true;
	}
	Relation *ChildRelation() override {
		return child.get();
	}

protected:
	bool ContainsNonSQLRelation() override {
		return true;
	}
	bool CanBindAsInputInternal(Binder &binder) override {
		return ChildCanBindAsInput(*child, binder);
	}

	BoundStatement BindAsInput(Binder &binder) override;
};

} // namespace duckdb
