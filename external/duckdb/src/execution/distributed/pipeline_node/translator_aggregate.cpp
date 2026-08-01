// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"

#include <algorithm>
#include <stdexcept>

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/pipeline_node/aggregate.hpp"
#include "duckdb/execution/distributed/pipeline_node/grouping_set_expand.hpp"
#include "duckdb/execution/distributed/pipeline_node/projection.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_partitioned_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_perfecthash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/function/function_binder.hpp"

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
		agg_input = gen_shuffle_node(std::move(spec), input_node->config().schema(), input_node);
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
		shuffle = gen_shuffle_node(std::move(spec), split_details.second_stage_schema, initial_agg);
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

std::shared_ptr<DistributedPipelineNode> PhysicalPlanToPipelineNodeTranslator::gen_grouping_sets_agg(
    std::shared_ptr<DistributedPipelineNode> input_node, const std::vector<BoundExpr> &group_by,
    const std::vector<BoundAggExpr> &aggregations, const std::vector<GroupingSet> &grouping_sets,
    const std::vector<std::vector<idx_t>> &grouping_functions, const std::vector<LogicalType> &input_types,
    SchemaRef output_schema) {
	auto make_single_stage = [&](std::shared_ptr<DistributedPipelineNode> child) {
		return std::make_shared<AggregateNode>(get_next_pipeline_node_id(), plan_config_, group_by, aggregations,
		                                       output_schema, std::move(child), grouping_sets, grouping_functions)
		    ->into_node();
	};

	if (!input_node) {
		return make_single_stage(nullptr);
	}
	const auto input_partitions = input_node->config().clustering_spec()->num_partitions();
	if (input_partitions <= 1) {
		return make_single_stage(input_node);
	}

	std::vector<idx_t> aggregate_filter_indexes;
	aggregate_filter_indexes.reserve(aggregations.size());
	for (const auto &aggregation : aggregations) {
		auto &aggregate = aggregation->Cast<BoundAggregateExpression>();
		if (!aggregate.filter) {
			aggregate_filter_indexes.push_back(DConstants::INVALID_INDEX);
			continue;
		}
		if (aggregate.filter->GetExpressionType() != ExpressionType::BOUND_REF) {
			throw InternalException("Grouping-set aggregate filter is not a bound reference");
		}
		aggregate_filter_indexes.push_back(aggregate.filter->Cast<BoundReferenceExpression>().index);
	}

	std::vector<LogicalType> expanded_types(input_types.begin(), input_types.end());
	expanded_types.reserve(input_types.size() + group_by.size() + grouping_functions.size() + 1 + aggregations.size());
	for (const auto &group : group_by) {
		expanded_types.push_back(group->return_type);
	}
	expanded_types.resize(expanded_types.size() + grouping_functions.size(), LogicalType::BIGINT);
	expanded_types.push_back(LogicalType::UBIGINT);
	// Every aggregate gets an effective filter. Real rows use TRUE or the
	// original filter; synthetic empty-set rows use FALSE for every aggregate.
	expanded_types.resize(expanded_types.size() + aggregations.size(), LogicalType::BOOLEAN);
	auto expanded_schema = MakeSchemaRef(expanded_types);

	auto expand_impl = std::make_shared<GroupingSetExpandNode>(get_next_pipeline_node_id(), input_node->inner(),
	                                                           group_by, grouping_sets, grouping_functions,
	                                                           aggregate_filter_indexes, expanded_types);
	auto expanded = std::make_shared<DistributedPipelineNode>(std::move(expand_impl));

	const idx_t expanded_group_offset = input_types.size();
	const idx_t expanded_grouping_function_offset = expanded_group_offset + group_by.size();
	const idx_t grouping_set_id_offset = expanded_grouping_function_offset + grouping_functions.size();
	const idx_t effective_filter_offset = grouping_set_id_offset + 1;

	std::vector<ExprRef> partition_by;
	partition_by.reserve(group_by.size() + 1);
	for (idx_t group_idx = 0; group_idx < group_by.size(); group_idx++) {
		partition_by.emplace_back(std::make_shared<BoundReferenceExpression>(group_by[group_idx]->return_type,
		                                                                     expanded_group_offset + group_idx));
	}
	partition_by.emplace_back(std::make_shared<BoundReferenceExpression>(LogicalType::UBIGINT, grouping_set_id_offset));

	const auto raw_partitions = expanded->config().clustering_spec()->num_partitions();
	const auto configured_partitions =
	    plan_config_.num_partitions > 0 ? std::min(raw_partitions, plan_config_.num_partitions) : raw_partitions;
	const auto target_partitions = std::max<size_t>(1, configured_partitions);
	auto repartition_spec = RepartitionSpec::create_hash(target_partitions, std::move(partition_by));
	auto shuffle = gen_shuffle_node(std::move(repartition_spec), std::move(expanded_schema), expanded);

	std::vector<BoundExpr> final_group_by;
	final_group_by.reserve(group_by.size() + grouping_functions.size() + 1);
	for (idx_t group_idx = 0; group_idx < group_by.size(); group_idx++) {
		auto ref =
		    make_uniq<BoundReferenceExpression>(group_by[group_idx]->return_type, expanded_group_offset + group_idx);
		final_group_by.emplace_back(ExpressionRef(ref.release()));
	}
	for (idx_t function_idx = 0; function_idx < grouping_functions.size(); function_idx++) {
		auto ref =
		    make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, expanded_grouping_function_offset + function_idx);
		final_group_by.emplace_back(ExpressionRef(ref.release()));
	}
	auto grouping_set_ref = make_uniq<BoundReferenceExpression>(LogicalType::UBIGINT, grouping_set_id_offset);
	final_group_by.emplace_back(ExpressionRef(grouping_set_ref.release()));

	std::vector<BoundAggExpr> final_aggregations;
	final_aggregations.reserve(aggregations.size());
	for (idx_t aggregate_idx = 0; aggregate_idx < aggregations.size(); aggregate_idx++) {
		auto copy =
		    FunctionBinder::UnbindSortedAggregate(aggregations[aggregate_idx]->Cast<BoundAggregateExpression>());
		auto &aggregate = *copy;
		aggregate.filter =
		    make_uniq<BoundReferenceExpression>(LogicalType::BOOLEAN, effective_filter_offset + aggregate_idx);
		final_aggregations.emplace_back(ExpressionRef(copy.release()));
	}

	std::vector<LogicalType> final_aggregate_types;
	final_aggregate_types.reserve(final_group_by.size() + final_aggregations.size());
	for (const auto &group : final_group_by) {
		final_aggregate_types.push_back(group->return_type);
	}
	for (const auto &aggregate : final_aggregations) {
		final_aggregate_types.push_back(aggregate->return_type);
	}
	auto final_aggregate_schema = MakeSchemaRef(final_aggregate_types);
	auto final_aggregation = std::make_shared<AggregateNode>(get_next_pipeline_node_id(), plan_config_, final_group_by,
	                                                         final_aggregations, final_aggregate_schema, shuffle)
	                             ->into_node();

	std::vector<BoundExpr> final_expressions;
	final_expressions.reserve(group_by.size() + aggregations.size() + grouping_functions.size());
	for (idx_t group_idx = 0; group_idx < group_by.size(); group_idx++) {
		auto ref = make_uniq<BoundReferenceExpression>(group_by[group_idx]->return_type, group_idx);
		final_expressions.emplace_back(ExpressionRef(ref.release()));
	}
	const idx_t final_aggregate_offset = final_group_by.size();
	for (idx_t aggregate_idx = 0; aggregate_idx < aggregations.size(); aggregate_idx++) {
		auto ref = make_uniq<BoundReferenceExpression>(aggregations[aggregate_idx]->return_type,
		                                               final_aggregate_offset + aggregate_idx);
		final_expressions.emplace_back(ExpressionRef(ref.release()));
	}
	const idx_t final_grouping_function_offset = group_by.size();
	for (idx_t function_idx = 0; function_idx < grouping_functions.size(); function_idx++) {
		auto ref =
		    make_uniq<BoundReferenceExpression>(LogicalType::BIGINT, final_grouping_function_offset + function_idx);
		final_expressions.emplace_back(ExpressionRef(ref.release()));
	}

	std::vector<std::string> projection_names;
	projection_names.reserve(final_expressions.size());
	for (const auto &expression : final_expressions) {
		projection_names.push_back(expression->GetName());
	}
	auto projection = std::make_shared<ProjectionNode>(
	    get_next_pipeline_node_id(), final_aggregation->inner(), std::move(final_expressions),
	    std::move(projection_names), output_schema, ClusteringSpec::unknown_with_num_partitions(target_partitions));
	return std::make_shared<DistributedPipelineNode>(std::move(projection));
}

static bool RequiresGroupingSetsPlan(const PhysicalHashAggregate &aggregate) {
	if (!aggregate.grouped_aggregate_data.grouping_functions.empty() || aggregate.grouping_sets.size() != 1) {
		return true;
	}
	if (aggregate.grouping_sets.empty()) {
		return false;
	}
	const auto &grouping_set = aggregate.grouping_sets[0];
	if (grouping_set.size() != aggregate.grouped_aggregate_data.groups.size()) {
		return true;
	}
	for (idx_t group_idx = 0; group_idx < aggregate.grouped_aggregate_data.groups.size(); group_idx++) {
		if (grouping_set.find(group_idx) == grouping_set.end()) {
			return true;
		}
	}
	return false;
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
		auto &aggregate = a->Cast<BoundAggregateExpression>();
		auto copy = FunctionBinder::UnbindSortedAggregate(aggregate);
		if (aggregate.filter) {
			auto filter_index = ha.filter_indexes.find(aggregate.filter.get());
			if (filter_index == ha.filter_indexes.end()) {
				throw InternalException("hash aggregate filter is missing its input index");
			}
			auto &copied_filter = copy->filter->Cast<BoundReferenceExpression>();
			copied_filter.index = filter_index->second;
		}
		aggs.emplace_back(ExpressionRef(copy.release()));
	}

	SchemaRef schema = nullptr;
	if (!ha.GetTypes().empty()) {
		schema = MakeSchemaRef(ha.GetTypes());
	}

	if (RequiresGroupingSetsPlan(ha)) {
		std::vector<GroupingSet> grouping_sets;
		grouping_sets.reserve(ha.grouping_sets.size());
		for (const auto &grouping_set : ha.grouping_sets) {
			grouping_sets.push_back(grouping_set);
		}
		std::vector<std::vector<idx_t>> grouping_functions;
		grouping_functions.reserve(ha.grouped_aggregate_data.grouping_functions.size());
		for (const auto &grouping_function : ha.grouped_aggregate_data.grouping_functions) {
			grouping_functions.emplace_back(grouping_function.begin(), grouping_function.end());
		}
		std::vector<LogicalType> input_types;
		if (!ha.children.empty()) {
			const auto &child_types = ha.children[0].get().GetTypes();
			input_types.assign(child_types.begin(), child_types.end());
		}
		auto agg_node =
		    gen_grouping_sets_agg(child_node, group_by, aggs, grouping_sets, grouping_functions, input_types, schema);
		if (!agg_node) {
			throw InternalException("distributed grouping sets translation produced null node");
		}
		return agg_node;
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
