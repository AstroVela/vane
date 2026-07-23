// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/setop_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"
#include "duckdb/common/enums/set_operation_type.hpp"

namespace duckdb {

class SetOpRelation : public Relation {
public:
	SetOpRelation(shared_ptr<Relation> left, shared_ptr<Relation> right, SetOperationType setop_type,
	              bool setop_all = false);

	shared_ptr<Relation> left;
	shared_ptr<Relation> right;
	SetOperationType setop_type;
	vector<ColumnDefinition> columns;
	bool setop_all;

public:
	unique_ptr<QueryNode> GetQueryNode() override;

	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;
	string GetAlias() override;
	bool ContainsNonSQLRelation() override {
		return left->ContainsNonSQLRelation() || right->ContainsNonSQLRelation();
	}
	bool CanSerializeToQueryNodeInternal(Binder &binder) override {
		return left->CanSerializeToQueryNodeInternal(binder) && right->CanSerializeToQueryNodeInternal(binder);
	}
};

} // namespace duckdb
