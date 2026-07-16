// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/external_block.hpp"
#include "duckdb/parallel/thread_context.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/planner/expression.hpp"

namespace duckdb {

class ProjectionState : public OperatorState {
public:
	explicit ProjectionState(ExecutionContext &context, const vector<unique_ptr<Expression>> &expressions)
	    : executor(context.client, expressions) {
	}

	ExpressionExecutor executor;

public:
	void Finalize(const PhysicalOperator &op, ExecutionContext &context) override {
		context.thread.profiler.Flush(op);
	}
};

PhysicalProjection::PhysicalProjection(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                       vector<unique_ptr<Expression>> select_list, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::PROJECTION, std::move(types), estimated_cardinality),
      select_list(std::move(select_list)) {
}

OperatorResultType PhysicalProjection::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                               GlobalOperatorState &gstate, OperatorState &state_p) const {
	auto &state = state_p.Cast<ProjectionState>();
	state.executor.Execute(input, chunk);
	return OperatorResultType::NEED_MORE_INPUT;
}

namespace {

void StoreLazyProjectionBatch(ExecutionBatch &output, unique_ptr<LazyDataChunk> lazy) {
	output = ExecutionBatch();
	output.kind = ExecutionBatchKind::LAZY_DATA_CHUNK;
	if (lazy) {
		lazy->RecomputeCardinality();
		output.rows = lazy->cardinality;
		output.estimated_bytes = lazy->EstimatedBytes();
	}
	output.lazy = std::move(lazy);
}

bool TryGetBoundReference(const Expression &expr, idx_t &index) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_REF) {
		return false;
	}
	auto &ref = expr.Cast<BoundReferenceExpression>();
	if (ref.index == DConstants::INVALID_INDEX) {
		return false;
	}
	index = ref.index;
	return true;
}

bool TryGetStructExtractField(const Expression &expr, string &field_name) {
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &func = expr.Cast<BoundFunctionExpression>();
	if (!StringUtil::CIEquals(func.function.name, "struct_extract") || func.children.size() != 2) {
		return false;
	}
	idx_t child_ref;
	if (!TryGetBoundReference(*func.children[0], child_ref) || child_ref != 0) {
		return false;
	}
	if (func.children[1]->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
		return false;
	}
	auto &constant = func.children[1]->Cast<BoundConstantExpression>();
	if (constant.value.IsNull()) {
		return false;
	}
	field_name = StringValue::Get(constant.value);
	return true;
}

unique_ptr<LazyDataChunk> MakeStructExtractProjection(const LazyDataChunk &input,
                                                      const vector<unique_ptr<Expression>> &select_list,
                                                      vector<idx_t> &column_ids, vector<string> &names) {
	if (!input.wrap_columns_as_struct || input.logical_types.size() != 1 ||
	    input.logical_types[0].id() != LogicalTypeId::STRUCT) {
		return nullptr;
	}
	auto &struct_type = input.logical_types[0];
	vector<LogicalType> raw_types;
	vector<string> raw_names;
	auto child_count = StructType::GetChildCount(struct_type);
	raw_types.reserve(child_count);
	raw_names.reserve(child_count);
	for (idx_t child_idx = 0; child_idx < child_count; child_idx++) {
		raw_types.push_back(StructType::GetChildType(struct_type, child_idx));
		raw_names.push_back(StructType::GetChildName(struct_type, child_idx));
	}

	column_ids.clear();
	names.clear();
	column_ids.reserve(select_list.size());
	names.reserve(select_list.size());
	for (auto &expr : select_list) {
		string field_name;
		if (!TryGetStructExtractField(*expr, field_name)) {
			return nullptr;
		}
		auto child_idx = StructType::GetChildIndexUnsafe(struct_type, field_name);
		column_ids.push_back(child_idx);
		names.push_back(expr->GetName());
	}

	auto raw_input = input;
	raw_input.logical_types = std::move(raw_types);
	raw_input.names = std::move(raw_names);
	raw_input.wrap_columns_as_struct = false;
	return ProjectLazyDataChunk(raw_input, column_ids, names);
}

unique_ptr<LazyDataChunk> TryBuildLazyProjection(const LazyDataChunk &input,
                                                 const vector<unique_ptr<Expression>> &select_list) {
	vector<idx_t> column_ids;
	vector<string> names;
	column_ids.reserve(select_list.size());
	names.reserve(select_list.size());

	bool direct_refs = true;
	for (auto &expr : select_list) {
		idx_t index;
		if (!TryGetBoundReference(*expr, index)) {
			direct_refs = false;
			break;
		}
		column_ids.push_back(index);
		names.push_back(expr->GetName());
	}
	if (direct_refs) {
		return ProjectLazyDataChunk(input, column_ids, names);
	}

	return MakeStructExtractProjection(input, select_list, column_ids, names);
}

} // namespace

OperatorResultType PhysicalProjection::ExecuteBatch(ExecutionContext &context, ExecutionBatch &input,
                                                    ExecutionBatch &output, GlobalOperatorState &gstate,
                                                    OperatorState &state) const {
	if (input.kind == ExecutionBatchKind::LAZY_DATA_CHUNK && input.lazy) {
		auto projected = TryBuildLazyProjection(*input.lazy, select_list);
		if (projected) {
			StoreLazyProjectionBatch(output, std::move(projected));
			return OperatorResultType::NEED_MORE_INPUT;
		}
	}
	return PhysicalOperator::ExecuteBatch(context, input, output, gstate, state);
}

unique_ptr<OperatorState> PhysicalProjection::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<ProjectionState>(context, select_list);
}

InsertionOrderPreservingMap<string> PhysicalProjection::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	string projections;
	for (idx_t i = 0; i < select_list.size(); i++) {
		if (i > 0) {
			projections += "\n";
		}
		auto &expr = select_list[i];
		projections += expr->GetName();
	}
	result["__projections__"] = projections;
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

void PhysicalProjection::SerializeOperatorData(Serializer &serializer) const {
	// Serialize projection-specific field: select list (expressions)
	serializer.WriteProperty(103, "select_list", select_list);
}

} // namespace duckdb
