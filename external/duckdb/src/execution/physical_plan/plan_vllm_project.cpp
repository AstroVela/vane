// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/projection/physical_vllm.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/scalar/vllm_functions.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_vllm_project.hpp"

namespace duckdb {

namespace {

static BoundFunctionExpression &GetVLLMFunction(Expression &expr) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		throw InternalException("vllm operator expected a bound function expression");
	}
	auto &bound = expr.Cast<BoundFunctionExpression>();
	if (!StringUtil::CIEquals(bound.function.name, "vllm")) {
		throw InternalException("vllm operator expected a vllm function expression");
	}
	return bound;
}

} // namespace

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalVLLMProject &op) {
	D_ASSERT(op.children.size() == 1);
	auto &plan = CreatePlan(*op.children[0]);
	auto vllm_expr = std::move(op.vllm_expr);
	if (!vllm_expr) {
		throw InternalException("vllm operator is missing expression");
	}
	auto &bound = GetVLLMFunction(*vllm_expr);
	if (!bound.bind_info) {
		throw InternalException("vllm function is missing bind data");
	}
	if (bound.children.size() != 1) {
		throw InternalException("vllm function expected a single prompt argument");
	}

	auto &bind_data = bound.bind_info->Cast<VLLMFunctionData>();
	auto model = std::move(bind_data.model);
	auto options = std::move(bind_data.options);
	auto prompt_expr = std::move(bound.children[0]);

	auto &vllm = Make<PhysicalVLLM>(op.types, std::move(prompt_expr), std::move(model), std::move(options),
	                                op.output_column_name, op.estimated_cardinality);
	vllm.children.push_back(plan);
	return vllm;
}

} // namespace duckdb
