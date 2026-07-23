// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/create_view_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"

namespace duckdb {

class CreateViewRelation : public Relation {
public:
	CreateViewRelation(shared_ptr<Relation> child, string view_name, bool replace, bool temporary);
	CreateViewRelation(shared_ptr<Relation> child, string schema_name, string view_name, bool replace, bool temporary);

	shared_ptr<Relation> child;
	string schema_name;
	string view_name;
	bool replace;
	bool temporary;
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
	bool CanSerializeToQueryNodeInternal(Binder &) override {
		return false;
	}
};

} // namespace duckdb
