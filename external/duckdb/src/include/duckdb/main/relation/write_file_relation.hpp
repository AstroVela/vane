// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/write_file_relation.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"

namespace duckdb {

//! A format-neutral COPY TO relation. The named CopyFunction owns format
//! validation, options, bind state, and execution.
class WriteFileRelation : public Relation {
public:
	WriteFileRelation(shared_ptr<Relation> child, string file_path, string format,
	                  case_insensitive_map_t<vector<Value>> options);

	shared_ptr<Relation> child;
	string file_path;
	string format;
	vector<ColumnDefinition> columns;
	case_insensitive_map_t<vector<Value>> options;

public:
	BoundStatement Bind(Binder &binder) override;
	unique_ptr<QueryNode> GetQueryNode() override;
	string GetQuery() override;
	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;
	bool IsReadOnly() override {
		return false;
	}

protected:
	bool CanSerializeToQueryNodeInternal(Binder &) override {
		return false;
	}
};

} // namespace duckdb
