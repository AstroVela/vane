// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"

#include <algorithm>

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/pipeline_node/expression_scan.hpp"
#include "duckdb/execution/distributed/pipeline_node/pivot.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/distributed/pipeline_node/streaming_udf_passthrough.hpp"
#include "duckdb/execution/distributed/pipeline_node/table_inout.hpp"
#include "duckdb/execution/distributed/pipeline_node/unnest.hpp"
#include "duckdb/execution/distributed/pipeline_node/window.hpp"
#include "duckdb/execution/operator/aggregate/physical_streaming_window.hpp"
#include "duckdb/execution/operator/aggregate/physical_window.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/execution/operator/projection/physical_pivot.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/execution/operator/projection/physical_unnest.hpp"
#include "duckdb/execution/operator/scan/physical_expression_scan.hpp"
#include "duckdb/planner/expression/bound_window_expression.hpp"

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

template <class ExpressionList>
const BoundWindowExpression *GetCommonWindowDefinition(const ExpressionList &select_list, const char *node_name) {
	const BoundWindowExpression *first = nullptr;
	for (const auto &expr : select_list) {
		if (!expr || expr->GetExpressionClass() != ExpressionClass::BOUND_WINDOW) {
			throw InvalidInputException("%s requires bound window expressions", node_name);
		}
		const auto &window = expr->template Cast<BoundWindowExpression>();
		if (!first) {
			first = &window;
			continue;
		}
		if (!first->PartitionsAreEquivalent(window)) {
			throw InvalidInputException("%s received incompatible partition definitions", node_name);
		}
	}
	return first;
}

bool WindowPartitionKeysMatch(const std::shared_ptr<DistributedPipelineNode> &input_node,
                              const vector<unique_ptr<Expression>> &required_partitions, size_t target_partitions) {
	if (!input_node) {
		return false;
	}
	auto input = input_node->inner();
	// Some schema-changing nodes preserve stale clustering expressions. Reuse
	// hash metadata only when this node established or validated the mapping.
	if (!std::dynamic_pointer_cast<RepartitionNode>(input) && !std::dynamic_pointer_cast<WindowNode>(input)) {
		return false;
	}
	auto clustering = input_node->config().clustering_spec();
	if (!clustering || clustering->type() != ClusteringSpec::Type::Hash ||
	    clustering->num_partitions() != target_partitions) {
		return false;
	}

	auto actual_partitions = clustering->partition_by();
	if (actual_partitions.size() != required_partitions.size()) {
		return false;
	}
	std::vector<bool> matched(actual_partitions.size(), false);
	for (const auto &required : required_partitions) {
		bool found = false;
		for (idx_t index = 0; index < actual_partitions.size(); index++) {
			if (!matched[index] && required && actual_partitions[index] &&
			    Expression::Equals(*required, *actual_partitions[index])) {
				matched[index] = true;
				found = true;
				break;
			}
		}
		if (!found) {
			return false;
		}
	}
	return true;
}

std::vector<ExprRef> CopyWindowPartitionKeys(const BoundWindowExpression &window) {
	std::vector<ExprRef> result;
	result.reserve(window.partitions.size());
	for (const auto &partition : window.partitions) {
		if (!partition) {
			throw InvalidInputException("Window contains a null partition expression");
		}
		auto copy = partition->Copy();
		result.emplace_back(ExpressionRef(copy.release()));
	}
	return result;
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
	if (children.empty() || !children[0]) {
		throw InvalidInputException("Window requires a child node");
	}
	auto input_node = children[0];
	auto window_definition = GetCommonWindowDefinition(op.select_list, "Window");
	if (window_definition) {
		auto clustering = input_node->config().clustering_spec();
		if (!clustering) {
			throw InvalidInputException("Window child is missing clustering metadata");
		}
		const auto input_partitions = clustering->num_partitions();
		if (window_definition->partitions.empty()) {
			if (input_partitions > 1) {
				input_node = gen_gather_node(input_node);
			}
		} else {
			const auto target_partitions = std::max<size_t>(1, std::max(input_partitions, plan_config_.num_partitions));
			if (target_partitions > 1 &&
			    !WindowPartitionKeysMatch(input_node, window_definition->partitions, target_partitions)) {
				auto spec =
				    RepartitionSpec::create_hash(target_partitions, CopyWindowPartitionKeys(*window_definition));
				auto shuffle = gen_shuffle_node(std::move(spec), input_node->config().schema(), input_node);
				if (!shuffle) {
					throw InvalidInputException("Failed to repartition Window input: %s", shuffle.error().what());
				}
				input_node = shuffle.value();
			}
		}
	}
	auto select_list = CopyExpressionListForMiscTranslator(op.select_list);
	std::vector<LogicalType> output_types = op.GetTypes();
	return std::make_shared<WindowNode>(get_next_pipeline_node_id(), input_node->inner(), std::move(select_list),
	                                    std::move(output_types));
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateStreamingWindow(
    const PhysicalStreamingWindow &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	if (children.empty() || !children[0]) {
		throw InvalidInputException("StreamingWindow requires a child node");
	}
	auto input_node = children[0];
	auto window_definition = GetCommonWindowDefinition(op.select_list, "StreamingWindow");
	if (window_definition &&
	    (!window_definition->partitions.empty() || !window_definition->orders.empty() ||
	     !window_definition->arg_orders.empty() || window_definition->exclude_clause != WindowExcludeMode::NO_OTHER)) {
		throw InvalidInputException("StreamingWindow requires an empty OVER clause");
	}
	if (window_definition) {
		auto clustering = input_node->config().clustering_spec();
		if (!clustering) {
			throw InvalidInputException("StreamingWindow child is missing clustering metadata");
		}
		if (clustering->num_partitions() > 1) {
			input_node = gen_gather_node(input_node);
		}
	}
	auto select_list = CopyExpressionListForMiscTranslator(op.select_list);
	std::vector<LogicalType> output_types = op.GetTypes();
	return std::make_shared<StreamingWindowNode>(get_next_pipeline_node_id(), input_node->inner(),
	                                             std::move(select_list), std::move(output_types));
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
