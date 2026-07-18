// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"

#include "duckdb/execution/distributed/pipeline_node/expression_scan.hpp"
#include "duckdb/execution/distributed/pipeline_node/pivot.hpp"
#include "duckdb/execution/distributed/pipeline_node/streaming_udf_passthrough.hpp"
#include "duckdb/execution/distributed/pipeline_node/table_inout.hpp"
#include "duckdb/execution/distributed/pipeline_node/unnest.hpp"
#include "duckdb/execution/distributed/pipeline_node/window.hpp"
#include "duckdb/execution/operator/aggregate/physical_streaming_window.hpp"
#include "duckdb/execution/operator/aggregate/physical_window.hpp"
#include "duckdb/execution/operator/projection/physical_pivot.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/execution/operator/projection/physical_unnest.hpp"
#include "duckdb/execution/operator/scan/physical_expression_scan.hpp"

namespace duckdb {
namespace distributed {

namespace {

PipelineNodeRef FirstMiscUnaryChildImpl(const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	if (children.empty()) {
		return nullptr;
	}
	return children[0]->inner();
}

BoundPivotInfo CopyPivotInfoForMiscTranslator(const BoundPivotInfo &info) {
	BoundPivotInfo copy;
	copy.group_count = info.group_count;
	copy.types = info.types;
	copy.pivot_values = info.pivot_values;
	copy.aggregates.reserve(info.aggregates.size());
	for (const auto &expr : info.aggregates) {
		if (expr) {
			copy.aggregates.push_back(expr->Copy());
		} else {
			copy.aggregates.push_back(nullptr);
		}
	}
	return copy;
}

template <class ExpressionList>
std::vector<ExpressionRef> CopyExpressionListForMiscTranslator(const ExpressionList &source) {
	std::vector<ExpressionRef> expressions;
	expressions.reserve(source.size());
	for (auto &expr : source) {
		if (!expr) {
			continue;
		}
		auto copy = expr->Copy();
		expressions.emplace_back(ExpressionRef(copy.release()));
	}
	return expressions;
}

template <class ExpressionMatrix>
std::vector<std::vector<ExpressionRef>> CopyExpressionMatrixForMiscTranslator(const ExpressionMatrix &source) {
	std::vector<std::vector<ExpressionRef>> expressions;
	expressions.reserve(source.size());
	for (auto &expr_list : source) {
		expressions.push_back(CopyExpressionListForMiscTranslator(expr_list));
	}
	return expressions;
}

} // namespace

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslatePivot(
    const PhysicalPivot &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstMiscUnaryChildImpl(children);
	auto bound_pivot = CopyPivotInfoForMiscTranslator(op.bound_pivot);
	std::vector<LogicalType> output_types = op.GetTypes();
	return std::make_shared<PivotNode>(get_next_pipeline_node_id(), child_impl, std::move(bound_pivot),
	                                   std::move(output_types));
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateUnnest(
    const PhysicalUnnest &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstMiscUnaryChildImpl(children);
	auto select_list = CopyExpressionListForMiscTranslator(op.select_list);
	std::vector<LogicalType> output_types = op.GetTypes();
	return std::make_shared<UnnestNode>(get_next_pipeline_node_id(), child_impl, std::move(select_list),
	                                    std::move(output_types));
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateTableInOut(
    const PhysicalTableInOutFunction &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstMiscUnaryChildImpl(children);
	auto bind_data = op.GetBindData() ? op.GetBindData()->Copy() : nullptr;
	auto column_ids = op.GetColumnIds();
	auto projected_input = op.GetProjectedInput();
	auto ordinality_idx = op.ordinality_idx;
	std::vector<LogicalType> output_types = op.GetTypes();
	return std::make_shared<TableInOutNode>(get_next_pipeline_node_id(), child_impl, op.GetFunction(),
	                                        std::move(bind_data), std::move(column_ids), std::move(projected_input),
	                                        ordinality_idx, std::move(output_types), op.estimated_cardinality);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateStreamingUDF(
    const PhysicalStreamingUDF &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstMiscUnaryChildImpl(children);
	auto bind_data = op.GetBindData() ? op.GetBindData()->Copy() : nullptr;
	auto column_ids = op.GetColumnIds();
	auto projected_input = op.GetProjectedInput();
	auto ordinality_idx = op.ordinality_idx;
	std::vector<LogicalType> output_types = op.GetTypes();
	return std::make_shared<StreamingUDFPassthroughNode>(
	    get_next_pipeline_node_id(), child_impl, op.GetFunction(), std::move(bind_data), std::move(column_ids),
	    std::move(projected_input), ordinality_idx, std::move(output_types), op.estimated_cardinality);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateWindow(
    const PhysicalWindow &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstMiscUnaryChildImpl(children);
	auto select_list = CopyExpressionListForMiscTranslator(op.select_list);
	std::vector<LogicalType> output_types = op.GetTypes();
	return std::make_shared<WindowNode>(get_next_pipeline_node_id(), child_impl, std::move(select_list),
	                                    std::move(output_types));
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateStreamingWindow(
    const PhysicalStreamingWindow &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstMiscUnaryChildImpl(children);
	auto select_list = CopyExpressionListForMiscTranslator(op.select_list);
	std::vector<LogicalType> output_types = op.GetTypes();
	return std::make_shared<StreamingWindowNode>(get_next_pipeline_node_id(), child_impl, std::move(select_list),
	                                             std::move(output_types));
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateExpressionScan(
    const PhysicalExpressionScan &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstMiscUnaryChildImpl(children);
	auto expressions = CopyExpressionMatrixForMiscTranslator(op.expressions);
	std::vector<LogicalType> output_types = op.GetTypes();
	return std::make_shared<ExpressionScanNode>(get_next_pipeline_node_id(), child_impl, std::move(expressions),
	                                            std::move(output_types));
}

} // namespace distributed
} // namespace duckdb
