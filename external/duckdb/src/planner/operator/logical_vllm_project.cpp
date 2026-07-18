// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/planner/operator/logical_vllm_project.hpp"

namespace duckdb {

LogicalVLLMProject::LogicalVLLMProject(idx_t table_index_p, unique_ptr<Expression> vllm_expr_p,
                                       string output_column_name_p)
    : LogicalOperator(LogicalOperatorType::LOGICAL_VLLM_PROJECT), table_index(table_index_p),
      vllm_expr(std::move(vllm_expr_p)), output_column_name(std::move(output_column_name_p)) {
}

vector<ColumnBinding> LogicalVLLMProject::GetColumnBindings() {
	D_ASSERT(!children.empty());
	auto bindings = children[0]->GetColumnBindings();
	bindings.emplace_back(table_index, 0);
	return bindings;
}

void LogicalVLLMProject::ResolveTypes() {
	D_ASSERT(!children.empty());
	types = children[0]->types;
	types.push_back(vllm_expr->return_type);
}

vector<idx_t> LogicalVLLMProject::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

string LogicalVLLMProject::GetName() const {
	return LogicalOperator::GetName();
}

} // namespace duckdb
