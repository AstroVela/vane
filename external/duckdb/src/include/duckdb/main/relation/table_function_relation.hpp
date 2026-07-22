// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/relation/table_function_relation.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/relation.hpp"

namespace duckdb {

class TableFunctionRelation : public Relation {
public:
	TableFunctionRelation(const shared_ptr<ClientContext> &context, string name, vector<Value> parameters,
	                      named_parameter_map_t named_parameters, shared_ptr<Relation> input_relation_p = nullptr,
	                      bool auto_init = true);
	TableFunctionRelation(const shared_ptr<RelationContextWrapper> &context, string name, vector<Value> parameters,
	                      named_parameter_map_t named_parameters, shared_ptr<Relation> input_relation_p = nullptr,
	                      bool auto_init = true);
	TableFunctionRelation(const shared_ptr<ClientContext> &context, string name, vector<Value> parameters,
	                      shared_ptr<Relation> input_relation_p = nullptr, bool auto_init = true);
	~TableFunctionRelation() override {
	}

	string name;
	vector<Value> parameters;
	named_parameter_map_t named_parameters;
	vector<ColumnDefinition> columns;
	shared_ptr<Relation> input_relation;

public:
	unique_ptr<QueryNode> GetQueryNode() override;
	unique_ptr<TableRef> GetTableRef() override;
	BoundStatement Bind(Binder &binder) override;

	const vector<ColumnDefinition> &Columns() override;
	string ToString(idx_t depth) override;
	string GetAlias() override;
	void AddNamedParameter(const string &name, Value argument);
	void RemoveNamedParameterIfExists(const string &name);
	void SetNamedParameters(named_parameter_map_t &&named_parameters);
	const vector<string> &GetVirtualColumnNames() const {
		return virtual_column_names;
	}
	bool ContainsNonSQLRelation() override {
		return input_relation && input_relation->ContainsNonSQLRelation();
	}
	bool CanSerializeToQueryNode() override {
		return !input_relation || input_relation->CanSerializeToQueryNode();
	}
	bool CanBindAsInput() override {
		return !input_relation || input_relation->CanBindAsInput();
	}

private:
	void InitializeColumns();
	void CaptureVirtualColumnNames(const LogicalOperator &plan);
	BoundStatement BindAsInput(Binder &binder) override;

private:
	vector<string> virtual_column_names;
	//! Whether or not to auto initialize the columns on construction
	bool auto_initialize;
};

} // namespace duckdb
