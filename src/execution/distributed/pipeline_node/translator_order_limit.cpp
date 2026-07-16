// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"

#include "duckdb/execution/distributed/pipeline_node/limit.hpp"
#include "duckdb/execution/distributed/pipeline_node/sample.hpp"
#include "duckdb/execution/distributed/pipeline_node/sort.hpp"
#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/helper/physical_limit_percent.hpp"
#include "duckdb/execution/operator/helper/physical_reservoir_sample.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_sample.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/operator/order/physical_top_n.hpp"

namespace duckdb {
namespace distributed {

namespace {

PipelineNodeRef FirstOrderLimitChildImpl(const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	if (children.empty()) {
		return nullptr;
	}
	return children[0]->inner();
}

} // namespace

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateLimit(
    const PhysicalLimit &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstOrderLimitChildImpl(children);
	auto limit_val = CopyBoundLimitNode(op.limit_val);
	auto offset_val = CopyBoundLimitNode(op.offset_val);
	return std::make_shared<LimitNode>(get_next_pipeline_node_id(), child_impl, std::move(limit_val),
	                                   std::move(offset_val), exchange_mgr_);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateStreamingLimit(
    const PhysicalStreamingLimit &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstOrderLimitChildImpl(children);
	auto limit_val = CopyBoundLimitNode(op.limit_val);
	auto offset_val = CopyBoundLimitNode(op.offset_val);
	return std::make_shared<StreamingLimitNode>(get_next_pipeline_node_id(), child_impl, std::move(limit_val),
	                                            std::move(offset_val), op.parallel, exchange_mgr_);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateLimitPercent(
    const PhysicalLimitPercent &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstOrderLimitChildImpl(children);
	auto limit_val = CopyBoundLimitNode(op.limit_val);
	auto offset_val = CopyBoundLimitNode(op.offset_val);
	return std::make_shared<LimitPercentNode>(get_next_pipeline_node_id(), child_impl, std::move(limit_val),
	                                          std::move(offset_val), exchange_mgr_);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateOrderBy(
    const PhysicalOrder &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstOrderLimitChildImpl(children);
	vector<BoundOrderByNode> orders;
	orders.reserve(op.orders.size());
	for (const auto &order : op.orders) {
		orders.push_back(order.Copy());
	}
	vector<idx_t> projections = op.projections;
	vector<LogicalType> output_types = op.GetTypes();
	return std::make_shared<OrderByNode>(get_next_pipeline_node_id(), child_impl, std::move(orders),
	                                     std::move(projections), std::move(output_types), op.is_index_sort,
	                                     exchange_mgr_);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateTopN(
    const PhysicalTopN &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstOrderLimitChildImpl(children);
	vector<BoundOrderByNode> orders;
	orders.reserve(op.orders.size());
	for (const auto &order : op.orders) {
		orders.push_back(order.Copy());
	}
	return std::make_shared<TopNNode>(get_next_pipeline_node_id(), child_impl, std::move(orders), op.limit, op.offset,
	                                  exchange_mgr_);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateReservoirSample(
    const PhysicalReservoirSample &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstOrderLimitChildImpl(children);
	unique_ptr<SampleOptions> options;
	if (op.options) {
		options = op.options->Copy();
	}
	vector<LogicalType> output_types = op.GetTypes();
	return std::make_shared<ReservoirSampleNode>(get_next_pipeline_node_id(), child_impl, std::move(options),
	                                             std::move(output_types));
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateStreamingSample(
    const PhysicalStreamingSample &op, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	auto child_impl = FirstOrderLimitChildImpl(children);
	unique_ptr<SampleOptions> options;
	if (op.sample_options) {
		options = op.sample_options->Copy();
	}
	vector<LogicalType> output_types = op.GetTypes();
	return std::make_shared<StreamingSampleNode>(get_next_pipeline_node_id(), child_impl, std::move(options),
	                                             std::move(output_types));
}

} // namespace distributed
} // namespace duckdb
