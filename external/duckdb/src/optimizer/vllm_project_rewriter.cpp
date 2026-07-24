// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/optimizer/vllm_project_rewriter.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
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

static bool IsVLLMShortCircuitBoundary(const Expression &expr) {
	return expr.type == ExpressionType::CASE_EXPR || expr.type == ExpressionType::CONJUNCTION_AND ||
	       expr.type == ExpressionType::CONJUNCTION_OR || expr.type == ExpressionType::OPERATOR_COALESCE;
}

struct VLLMExtractionState {
	VLLMExtractionState(Binder &binder_p, LogicalProjection &projection_p)
	    : binder(binder_p), projection(projection_p) {
	}

	Binder &binder;
	LogicalProjection &projection;
	unique_ptr<LogicalOperator> current_child;
};

static void ExtractVLLMExpressions(unique_ptr<Expression> &expr, VLLMExtractionState &state,
                                   bool inside_short_circuit = false) {
	if (!expr) {
		return;
	}
	if (inside_short_circuit && ContainsVLLM(*expr)) {
		throw NotImplementedException(
		    "vllm expressions are not supported inside CASE, AND/OR, or COALESCE short-circuit expressions");
	}

	// Extract children first so nested vllm calls are available as bound columns
	// to the operators that depend on them.
	const bool child_inside_short_circuit = inside_short_circuit || IsVLLMShortCircuitBoundary(*expr);
	ExpressionIterator::EnumerateChildren(*expr, [&](unique_ptr<Expression> &child) {
		ExtractVLLMExpressions(child, state, child_inside_short_circuit);
	});

	if (!IsVLLMFunction(*expr)) {
		return;
	}
	if (!state.current_child) {
		D_ASSERT(state.projection.children.size() == 1);
		state.current_child = std::move(state.projection.children[0]);
	}

	auto vllm_expr = std::move(expr);
	auto replacement_name = vllm_expr->GetName();
	auto output_type = vllm_expr->return_type;
	auto table_index = state.binder.GenerateTableIndex();
	auto output_name = StringUtil::Format("__vane_vllm_%llu", static_cast<unsigned long long>(table_index));
	auto output_binding = ColumnBinding(table_index, 0);

	auto vllm_project = make_uniq<LogicalVLLMProject>(table_index, std::move(vllm_expr), output_name);
	vllm_project->children.push_back(std::move(state.current_child));
	state.current_child = std::move(vllm_project);

	expr = make_uniq<BoundColumnRefExpression>(replacement_name, output_type, output_binding);
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
	VLLMExtractionState state(binder, proj);
	for (auto &expr : proj.expressions) {
		ExtractVLLMExpressions(expr, state);
	}
	if (state.current_child) {
		proj.children[0] = std::move(state.current_child);
	}

	return op;
}

} // namespace duckdb
