// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"

#include <algorithm>
#include <stdexcept>

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/pipeline_node/aggregate.hpp"
#include "duckdb/execution/distributed/pipeline_node/projection.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_partitioned_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_perfecthash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"

namespace duckdb {
namespace distributed {

std::shared_ptr<DistributedPipelineNode> PhysicalPlanToPipelineNodeTranslator::gen_without_pre_agg(
    std::shared_ptr<DistributedPipelineNode> input_node, const std::vector<BoundExpr> &group_by,
    const std::vector<BoundAggExpr> &aggregations, SchemaRef output_schema,
    const std::vector<BoundExpr> &partition_by) {
	std::shared_ptr<DistributedPipelineNode> agg_input;
	if (partition_by.empty()) {
		agg_input = gen_gather_node(input_node);
	} else {
		std::vector<ExprRef> by;
		by.reserve(partition_by.size());
		for (auto &e : partition_by) {
			by.push_back(e);
		}
		auto target_partitions = plan_config_.num_partitions > 0
		                             ? plan_config_.num_partitions
		                             : input_node->config().clustering_spec()->num_partitions();
		auto spec = RepartitionSpec::create_hash(target_partitions, std::move(by));
		auto res = gen_shuffle_node(spec, input_node->config().schema(), input_node);
		if (!res) {
			throw std::runtime_error(std::string("gen_shuffle_node failed in aggregate translation: ") +
			                         res.error().what());
		}
		agg_input = res.value();
	}
	if (!agg_input) {
		throw std::runtime_error("aggregate translation produced null shuffle/gather input");
	}
	return std::make_shared<AggregateNode>(get_next_pipeline_node_id(), plan_config_, group_by, aggregations,
	                                       output_schema, agg_input)
	    ->into_node();
}

std::shared_ptr<DistributedPipelineNode>
PhysicalPlanToPipelineNodeTranslator::gen_with_pre_agg(std::shared_ptr<DistributedPipelineNode> input_node,
                                                       const GroupByAggSplit &split_details, SchemaRef output_schema) {
	auto initial_agg =
	    std::make_shared<AggregateNode>(get_next_pipeline_node_id(), plan_config_, split_details.first_stage_group_by,
	                                    split_details.first_stage_aggs, split_details.first_stage_schema, input_node)
	        ->into_node();

	size_t raw_partitions = initial_agg->config().clustering_spec()->num_partitions();
	size_t num_partitions = plan_config_.num_partitions > 0 ? std::min(raw_partitions, plan_config_.num_partitions) : 1;

	std::shared_ptr<DistributedPipelineNode> shuffle;
	if (split_details.partition_by.empty()) {
		shuffle = gen_gather_node(initial_agg);
	} else {
		std::vector<ExprRef> by;
		by.reserve(split_details.partition_by.size());
		for (auto &e : split_details.partition_by) {
			by.push_back(e);
		}
		auto spec = RepartitionSpec::create_hash(num_partitions, std::move(by));
		auto res = gen_shuffle_node(spec, split_details.second_stage_schema, initial_agg);
		if (!res) {
			throw std::runtime_error(std::string("gen_shuffle_node failed in aggregate translation: ") +
			                         res.error().what());
		}
		shuffle = res.value();
	}

	auto final_aggregation =
	    std::make_shared<AggregateNode>(get_next_pipeline_node_id(), plan_config_, split_details.second_stage_group_by,
	                                    split_details.second_stage_aggs, split_details.second_stage_schema, shuffle)
	        ->into_node();

	std::vector<std::string> projection_names;
	projection_names.reserve(split_details.final_exprs.size());
	for (auto &expr : split_details.final_exprs) {
		projection_names.push_back(expr ? expr->GetName() : std::string());
	}
	auto proj = std::make_shared<ProjectionNode>(get_next_pipeline_node_id(), final_aggregation->inner(),
	                                             split_details.final_exprs, std::move(projection_names), output_schema);
	return std::make_shared<DistributedPipelineNode>(proj);
}

std::shared_ptr<DistributedPipelineNode> PhysicalPlanToPipelineNodeTranslator::gen_agg_nodes(
    std::shared_ptr<DistributedPipelineNode> input_node, const std::vector<BoundExpr> &group_by,
    const std::vector<BoundAggExpr> &aggregations, SchemaRef output_schema,
    const std::vector<BoundExpr> &partition_by) {
	if (!input_node) {
		return std::make_shared<AggregateNode>(get_next_pipeline_node_id(), plan_config_, group_by, aggregations,
		                                       output_schema, input_node)
		    ->into_node();
	}

	const size_t input_partitions = input_node->config().clustering_spec()->num_partitions();
	if (input_partitions <= 1) {
		return std::make_shared<AggregateNode>(get_next_pipeline_node_id(), plan_config_, group_by, aggregations,
		                                       output_schema, input_node)
		    ->into_node();
	}

	auto split_res = split_groupby_aggs(group_by, aggregations, partition_by, input_node->config().schema());
	if (split_res.is_err()) {
		throw std::runtime_error(std::string("split_groupby_aggs failed: ") + split_res.error().what());
	}

	auto split = split_res.value();
	if (split.first_stage_aggs.empty()) {
		return gen_without_pre_agg(input_node, group_by, aggregations, output_schema, partition_by);
	}
	return gen_with_pre_agg(input_node, split, output_schema);
}

std::shared_ptr<DistributedPipelineNode> PhysicalPlanToPipelineNodeTranslator::TranslateHashGroupBy(
    const PhysicalHashAggregate &ha, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	DistributedPipelineNodeRef child_node = nullptr;
	if (!children.empty()) {
		child_node = children[0];
	}

	std::vector<BoundExpr> group_by;
	for (auto &g : ha.grouped_aggregate_data.groups) {
		if (!g) {
			continue;
		}
		auto copy = g->Copy();
		group_by.emplace_back(ExpressionRef(copy.release()));
	}

	std::vector<BoundAggExpr> aggs;
	for (auto &a : ha.grouped_aggregate_data.aggregates) {
		if (!a) {
			continue;
		}
		auto copy = a->Copy();
		aggs.emplace_back(ExpressionRef(copy.release()));
	}

	SchemaRef schema = nullptr;
	if (!ha.GetTypes().empty()) {
		schema = MakeSchemaRef(ha.GetTypes());
	}

	auto agg_node = gen_agg_nodes(child_node, group_by, aggs, schema, group_by);
	if (!agg_node) {
		throw InternalException("distributed hash aggregate translation produced null node");
	}
	return agg_node;
}

std::shared_ptr<DistributedPipelineNode> PhysicalPlanToPipelineNodeTranslator::TranslatePerfectHashGroupBy(
    const PhysicalPerfectHashAggregate &pha, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	DistributedPipelineNodeRef child_node = nullptr;
	if (!children.empty()) {
		child_node = children[0];
	}

	std::vector<BoundExpr> group_by;
	group_by.reserve(pha.groups.size());
	for (auto &g : pha.groups) {
		if (!g) {
			continue;
		}
		auto copy = g->Copy();
		group_by.emplace_back(ExpressionRef(copy.release()));
	}

	std::vector<BoundAggExpr> aggs;
	aggs.reserve(pha.aggregates.size());
	for (auto &a : pha.aggregates) {
		if (!a) {
			continue;
		}
		auto copy = a->Copy();
		aggs.emplace_back(ExpressionRef(copy.release()));
	}

	std::vector<Value> group_minima;
	group_minima.reserve(pha.group_minima.size());
	for (auto &val : pha.group_minima) {
		group_minima.push_back(val);
	}

	std::vector<idx_t> required_bits;
	required_bits.reserve(pha.required_bits.size());
	for (auto bits : pha.required_bits) {
		required_bits.push_back(bits);
	}

	std::vector<LogicalType> output_types;
	output_types.reserve(pha.GetTypes().size());
	for (auto &type : pha.GetTypes()) {
		output_types.push_back(type);
	}

	SchemaRef schema = nullptr;
	if (!pha.GetTypes().empty()) {
		schema = MakeSchemaRef(pha.GetTypes());
	}

	size_t child_parts = child_node ? child_node->config().clustering_spec()->num_partitions() : 1;
	if (plan_config_.num_partitions > 1 || child_parts > 1) {
		auto agg_node = gen_agg_nodes(child_node, group_by, aggs, schema, group_by);
		if (!agg_node) {
			throw InternalException("distributed perfect hash aggregate translation produced null node");
		}
		return agg_node;
	}

	auto node_impl = std::make_shared<PerfectHashAggregateNode>(
	    get_next_pipeline_node_id(), std::move(group_by), std::move(aggs), std::move(group_minima),
	    std::move(required_bits), std::move(output_types), child_node);
	return std::make_shared<DistributedPipelineNode>(node_impl);
}

std::shared_ptr<DistributedPipelineNode> PhysicalPlanToPipelineNodeTranslator::TranslatePartitionedAggregate(
    const PhysicalPartitionedAggregate &pa, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	DistributedPipelineNodeRef child_node = nullptr;
	if (!children.empty()) {
		child_node = children[0];
	}

	std::vector<BoundExpr> group_by;
	group_by.reserve(pa.groups.size());
	for (auto &g : pa.groups) {
		if (!g) {
			continue;
		}
		auto copy = g->Copy();
		group_by.emplace_back(ExpressionRef(copy.release()));
	}

	std::vector<BoundAggExpr> aggs;
	aggs.reserve(pa.aggregates.size());
	for (auto &a : pa.aggregates) {
		if (!a) {
			continue;
		}
		auto copy = a->Copy();
		aggs.emplace_back(ExpressionRef(copy.release()));
	}

	std::vector<column_t> partitions;
	partitions.reserve(pa.partitions.size());
	for (auto partition : pa.partitions) {
		partitions.push_back(partition);
	}

	std::vector<LogicalType> output_types;
	output_types.reserve(pa.GetTypes().size());
	for (auto &type : pa.GetTypes()) {
		output_types.push_back(type);
	}

	SchemaRef schema = nullptr;
	if (!pa.GetTypes().empty()) {
		schema = MakeSchemaRef(pa.GetTypes());
	}

	size_t child_parts = child_node ? child_node->config().clustering_spec()->num_partitions() : 1;
	if (plan_config_.num_partitions > 1 || child_parts > 1) {
		auto agg_node = gen_agg_nodes(child_node, group_by, aggs, schema, group_by);
		if (!agg_node) {
			throw InternalException("distributed partitioned aggregate translation produced null node");
		}
		return agg_node;
	}

	auto node_impl =
	    std::make_shared<PartitionedAggregateNode>(get_next_pipeline_node_id(), std::move(group_by), std::move(aggs),
	                                               std::move(partitions), std::move(output_types), child_node);
	return std::make_shared<DistributedPipelineNode>(node_impl);
}

std::shared_ptr<DistributedPipelineNode> PhysicalPlanToPipelineNodeTranslator::TranslateUngroupedAggregate(
    const PhysicalUngroupedAggregate &ua, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	DistributedPipelineNodeRef child_node = nullptr;
	if (!children.empty()) {
		child_node = children[0];
	}

	std::vector<BoundExpr> group_by;
	std::vector<BoundAggExpr> aggs;
	for (auto &a : ua.aggregates) {
		if (!a) {
			continue;
		}
		auto copy = a->Copy();
		aggs.emplace_back(ExpressionRef(copy.release()));
	}

	SchemaRef schema = nullptr;
	if (!ua.GetTypes().empty()) {
		schema = MakeSchemaRef(ua.GetTypes());
	}

	auto agg_node = gen_agg_nodes(child_node, group_by, aggs, schema, group_by);
	if (!agg_node) {
		throw InternalException("distributed ungrouped aggregate translation produced null node");
	}
	return agg_node;
}

} // namespace distributed
} // namespace duckdb
