// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/cross_product_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"
#include "duckdb/common/enums/joinref_type.hpp"

namespace duckdb {

class CrossProductRelation : public Relation {
public:
	DUCKDB_API CrossProductRelation(shared_ptr<Relation> left, shared_ptr<Relation> right,
	                                JoinRefType join_ref_type = JoinRefType::CROSS);

	shared_ptr<Relation> left;
	shared_ptr<Relation> right;
	JoinRefType ref_type;
	vector<ColumnDefinition> columns;

public:
	unique_ptr<QueryNode> GetQueryNode() override;

	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;

	bool ContainsNonSQLRelation() override {
		return left->ContainsNonSQLRelation() || right->ContainsNonSQLRelation();
	}
	bool CanSerializeToQueryNodeInternal(Binder &binder) override {
		return left->CanSerializeToQueryNodeInternal(binder) && right->CanSerializeToQueryNodeInternal(binder);
	}

protected:
	unique_ptr<TableRef> GetTableRefInternal() override;
	BoundStatement BindAsInput(Binder &binder) override;
};

} // namespace duckdb
