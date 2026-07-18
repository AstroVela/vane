// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/optimizer/vllm_project_rewriter.hpp"

#include "duckdb/common/constants.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar/vllm_functions.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_vllm_project.hpp"

namespace duckdb {

VLLMProjectRewriter::VLLMProjectRewriter(Binder &binder_p) : binder(binder_p) {
}

static bool IsVLLMFunction(const Expression &expr) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &func = expr.Cast<BoundFunctionExpression>().function;
	return StringUtil::CIEquals(func.name, "vllm");
}

static bool ContainsVLLM(const Expression &expr) {
	if (IsVLLMFunction(expr)) {
		return true;
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!found && ContainsVLLM(child)) {
			found = true;
		}
	});
	return found;
}

unique_ptr<LogicalOperator> VLLMProjectRewriter::Optimize(unique_ptr<LogicalOperator> op) {
	return Rewrite(std::move(op));
}

unique_ptr<LogicalOperator> VLLMProjectRewriter::Rewrite(unique_ptr<LogicalOperator> op) {
	if (!op) {
		return op;
	}
	for (auto &child : op->children) {
		child = Rewrite(std::move(child));
	}

	if (op->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return op;
	}

	auto &proj = op->Cast<LogicalProjection>();
	idx_t vllm_index = DConstants::INVALID_INDEX;

	for (idx_t i = 0; i < proj.expressions.size(); i++) {
		auto &expr = proj.expressions[i];
		if (!ContainsVLLM(*expr)) {
			continue;
		}
		if (!IsVLLMFunction(*expr)) {
			throw NotImplementedException("vllm must be used as a top-level projection expression");
		}
		if (vllm_index != DConstants::INVALID_INDEX) {
			throw NotImplementedException("Only one vllm expression per projection is supported");
		}
		vllm_index = i;
	}

	if (vllm_index == DConstants::INVALID_INDEX) {
		return op;
	}

	D_ASSERT(proj.children.size() == 1);

	auto vllm_expr = std::move(proj.expressions[vllm_index]);
	auto output_name = vllm_expr->GetName();
	auto output_type = vllm_expr->return_type;
	auto table_index = binder.GenerateTableIndex();

	auto output_binding = ColumnBinding(table_index, 0);

	auto vllm_project = make_uniq<LogicalVLLMProject>(table_index, std::move(vllm_expr), output_name);
	vllm_project->children.push_back(std::move(proj.children[0]));
	proj.children[0] = std::move(vllm_project);

	proj.expressions[vllm_index] = make_uniq<BoundColumnRefExpression>(output_name, output_type, output_binding);

	return op;
}

} // namespace duckdb
