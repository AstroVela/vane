// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/explain_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"

namespace duckdb {

class ExplainRelation : public Relation {
public:
	explicit ExplainRelation(shared_ptr<Relation> child, ExplainType type = ExplainType::EXPLAIN_STANDARD,
	                         ExplainFormat format = ExplainFormat::DEFAULT);

	shared_ptr<Relation> child;
	vector<ColumnDefinition> columns;
	ExplainType type;
	ExplainFormat format;

public:
	BoundStatement Bind(Binder &binder) override;
	unique_ptr<QueryNode> GetQueryNode() override;
	string GetQuery() override;
	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;
	bool IsReadOnly() override {
		return false;
	}
	bool CanSerializeToQueryNodeInternal(Binder &) override {
		return false;
	}
};

} // namespace duckdb
