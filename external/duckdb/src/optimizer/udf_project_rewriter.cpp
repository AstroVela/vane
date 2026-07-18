// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/udf_project_rewriter.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/optimizer/udf_project_rewriter.hpp"

#include "duckdb/common/constants.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_udf_project.hpp"

namespace duckdb {

UDFProjectRewriter::UDFProjectRewriter(Binder &binder_p) : binder(binder_p) {
}

static bool IsUDFFunction(const Expression &expr) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &func = expr.Cast<BoundFunctionExpression>().function;
	if (StringUtil::CIEquals(func.name, "udf")) {
		return true;
	}
	if (!func.function_info) {
		return false;
	}
	return dynamic_cast<RegisteredUDFFunctionInfo *>(func.function_info.get()) != nullptr;
}

static bool ContainsUDF(const Expression &expr) {
	if (IsUDFFunction(expr)) {
		return true;
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!found && ContainsUDF(child)) {
			found = true;
		}
	});
	return found;
}

static bool PayloadHasField(const Value &payload, const string &name) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return false;
	}
	auto &children = StructValue::GetChildren(payload);
	auto child_count = StructType::GetChildCount(payload.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) == name && i < children.size() && !children[i].IsNull()) {
			return true;
		}
	}
	return false;
}

static vector<string> PayloadOutputSchemaNames(const Value &payload) {
	vector<string> names;
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return names;
	}
	auto &children = StructValue::GetChildren(payload);
	auto child_count = StructType::GetChildCount(payload.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) != "output_schema" || i >= children.size() ||
		    children[i].IsNull()) {
			continue;
		}
		auto &entries = ListValue::GetChildren(children[i]);
		names.reserve(entries.size());
		for (auto &entry : entries) {
			if (entry.IsNull() || entry.type().id() != LogicalTypeId::STRUCT) {
				continue;
			}
			auto &entry_children = StructValue::GetChildren(entry);
			auto entry_child_count = StructType::GetChildCount(entry.type());
			for (idx_t entry_idx = 0; entry_idx < entry_child_count; entry_idx++) {
				if (StructType::GetChildName(entry.type(), entry_idx) == "name" && entry_idx < entry_children.size() &&
				    !entry_children[entry_idx].IsNull()) {
					names.push_back(StringValue::Get(entry_children[entry_idx]));
					break;
				}
			}
		}
		return names;
	}
	return names;
}

static UDFMode GetUDFProjectMode(const Expression &expr) {
	D_ASSERT(IsUDFFunction(expr));
	auto &bound_func = expr.Cast<BoundFunctionExpression>();
	if (!bound_func.bind_info) {
		throw InvalidInputException("udf expression is missing bind data");
	}
	auto &bind_data = bound_func.bind_info->Cast<UDFFunctionData>();
	return ClassifyUDFMode(bind_data.payload);
}

static bool ContainsResultOnlyBatchUDF(const Expression &expr) {
	if (IsUDFFunction(expr)) {
		return GetUDFProjectMode(expr) == UDFMode::RESULT_ONLY_BATCH;
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!found && ContainsResultOnlyBatchUDF(child)) {
			found = true;
		}
	});
	return found;
}

static bool ChildrenContainUDF(const Expression &expr) {
	bool found = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!found && ContainsUDF(child)) {
			found = true;
		}
	});
	return found;
}

static bool IsShortCircuitBoundary(const Expression &expr) {
	return expr.type == ExpressionType::CASE_EXPR || expr.type == ExpressionType::CONJUNCTION_AND ||
	       expr.type == ExpressionType::CONJUNCTION_OR || expr.type == ExpressionType::OPERATOR_COALESCE;
}

static void PopulateFlatMapOutputTypes(LogicalUDFProject &project, const LogicalType &return_type) {
	if (return_type.id() != LogicalTypeId::STRUCT) {
		project.flat_map_output_types.push_back(return_type);
		if (project.flat_map_output_names.empty()) {
			project.flat_map_output_names.push_back(project.output_column_name);
		}
		return;
	}
	auto &children = StructType::GetChildTypes(return_type);
	project.flat_map_output_types.reserve(children.size());
	if (project.flat_map_output_names.empty()) {
		project.flat_map_output_names.reserve(children.size());
	}
	for (auto &child : children) {
		project.flat_map_output_types.push_back(child.second);
		if (project.flat_map_output_names.size() < children.size()) {
			project.flat_map_output_names.push_back(child.first);
		}
	}
}

static constexpr const char *BATCH_V1_RESTRICTION =
    "batch expression UDFs must be a top-level projection and the only output in v1; "
    "use row_preserving=True for passthrough, multiple, or nested batch UDF composition";

struct ScalarExtractionState {
	ScalarExtractionState(Binder &binder_p, LogicalProjection &projection_p)
	    : binder(binder_p), projection(projection_p) {
	}

	Binder &binder;
	LogicalProjection &projection;
	unique_ptr<LogicalOperator> current_child;
};

static void ExtractScalarUDF(unique_ptr<Expression> &expr, ScalarExtractionState &state, const string &replacement_name,
                             UDFMode mode) {
	if (!state.current_child) {
		D_ASSERT(state.projection.children.size() == 1);
		state.current_child = std::move(state.projection.children[0]);
	}

	auto udf_expr = std::move(expr);
	auto output_type = udf_expr->return_type;
	auto table_index = state.binder.GenerateTableIndex();
	auto output_name = StringUtil::Format("__vane_udf_%llu", static_cast<unsigned long long>(table_index));
	auto output_binding = ColumnBinding(table_index, 0);

	auto udf_project = make_uniq<LogicalUDFProject>(table_index, std::move(udf_expr), output_name);
	udf_project->is_scalar_map = mode == UDFMode::SCALAR_MAP;
	udf_project->is_row_preserving_batch = mode == UDFMode::ROW_PRESERVING_BATCH;
	udf_project->children.push_back(std::move(state.current_child));
	state.current_child = std::move(udf_project);

	expr = make_uniq<BoundColumnRefExpression>(replacement_name, output_type, output_binding);
}

static void ExtractScalarUDFs(unique_ptr<Expression> &expr, ScalarExtractionState &state,
                              bool inside_short_circuit = false) {
	if (!expr) {
		return;
	}
	if (inside_short_circuit && ContainsUDF(*expr)) {
		throw NotImplementedException(
		    "udf expressions are not supported inside CASE, AND/OR, or COALESCE short-circuit expressions");
	}

	const bool child_inside_short_circuit = inside_short_circuit || IsShortCircuitBoundary(*expr);
	ExpressionIterator::EnumerateChildren(
	    *expr, [&](unique_ptr<Expression> &child) { ExtractScalarUDFs(child, state, child_inside_short_circuit); });

	if (!IsUDFFunction(*expr)) {
		return;
	}
	auto mode = GetUDFProjectMode(*expr);
	if (!UDFModePreservesRows(mode)) {
		throw InvalidInputException(BATCH_V1_RESTRICTION);
	}
	auto replacement_name = expr->GetName();
	ExtractScalarUDF(expr, state, replacement_name, mode);
}

unique_ptr<LogicalOperator> UDFProjectRewriter::Optimize(unique_ptr<LogicalOperator> op) {
	return Rewrite(std::move(op));
}

unique_ptr<LogicalOperator> UDFProjectRewriter::Rewrite(unique_ptr<LogicalOperator> op) {
	if (!op) {
		return op;
	}
	for (auto &child : op->children) {
		child = Rewrite(std::move(child));
	}

	if (op->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		for (auto &expr : op->expressions) {
			if (expr && ContainsUDF(*expr)) {
				throw InvalidInputException(
				    "udf can only be used in a projection and must be planned as a UDF operator");
			}
		}
		return op;
	}

	auto &proj = op->Cast<LogicalProjection>();
	idx_t batch_index = DConstants::INVALID_INDEX;

	for (idx_t i = 0; i < proj.expressions.size(); i++) {
		auto &expr = proj.expressions[i];
		if (expr && ContainsResultOnlyBatchUDF(*expr)) {
			if (batch_index != DConstants::INVALID_INDEX || proj.expressions.size() != 1 || !IsUDFFunction(*expr) ||
			    GetUDFProjectMode(*expr) != UDFMode::RESULT_ONLY_BATCH || ChildrenContainUDF(*expr)) {
				throw InvalidInputException(BATCH_V1_RESTRICTION);
			}
			batch_index = i;
		}
	}

	D_ASSERT(proj.children.size() == 1);

	if (batch_index != DConstants::INVALID_INDEX) {
		auto udf_expr = std::move(proj.expressions[batch_index]);
		auto output_name = udf_expr->GetName();
		auto output_type = udf_expr->return_type;
		auto table_index = binder.GenerateTableIndex();

		auto output_binding = ColumnBinding(table_index, 0);

		auto udf_project = make_uniq<LogicalUDFProject>(table_index, std::move(udf_expr), output_name);

		auto &bound_func = udf_project->udf_expr->Cast<BoundFunctionExpression>();
		if (bound_func.bind_info) {
			auto &bind_data = bound_func.bind_info->Cast<UDFFunctionData>();
			auto &payload = bind_data.payload;
			if (payload.type().id() == LogicalTypeId::STRUCT && PayloadHasField(payload, "output_schema")) {
				udf_project->is_flat_map = true;
				udf_project->flat_map_output_names = PayloadOutputSchemaNames(payload);
				PopulateFlatMapOutputTypes(*udf_project, bind_data.return_type);
			} else {
				throw InvalidInputException("batch expression udf payload requires output_schema");
			}
		}

		udf_project->children.push_back(std::move(proj.children[0]));
		proj.children[0] = std::move(udf_project);

		proj.expressions.clear();
		proj.expressions.push_back(make_uniq<BoundColumnRefExpression>(output_name, output_type, output_binding));
		return op;
	}

	ScalarExtractionState state(binder, proj);
	for (auto &expr : proj.expressions) {
		ExtractScalarUDFs(expr, state);
	}
	if (state.current_child) {
		proj.children[0] = std::move(state.current_child);
	}

	return op;
}

} // namespace duckdb
