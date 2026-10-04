// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/main/relation.hpp"

namespace duckdb {

class MergeIntoStatement;

class MergeRelation : public Relation {
public:
	MergeRelation(shared_ptr<Relation> source, unique_ptr<MergeIntoStatement> statement);
	~MergeRelation() override;

	shared_ptr<Relation> source;
	unique_ptr<MergeIntoStatement> statement;
	vector<ColumnDefinition> columns;

public:
	BoundStatement Bind(Binder &binder) override;
	unique_ptr<QueryNode> GetQueryNode() override;
	string GetQuery() override;
	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;
	bool IsReadOnly() override {
		return false;
	}
	Relation *ChildRelation() override {
		return source.get();
	}

protected:
	bool CanSerializeToQueryNodeInternal(Binder &) override {
		return false;
	}
};

} // namespace duckdb
