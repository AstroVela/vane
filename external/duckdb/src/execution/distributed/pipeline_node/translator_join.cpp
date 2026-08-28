// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/broadcast_join.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/cross_product.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/delim_join.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/hash_join.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/join_output_types.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/nested_loop_join.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/distributed/utils/optional.hpp"
#include "duckdb/execution/operator/join/physical_blockwise_nl_join.hpp"
#include "duckdb/execution/operator/join/physical_delim_join.hpp"
#include "duckdb/execution/operator/join/physical_cross_product.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/execution/operator/join/physical_range_join.hpp"

namespace duckdb {
namespace distributed {

namespace {

duckdb::vector<std::string> BuildJoinOutputNames(const PhysicalHashJoin &hj, const SchemaRef &left_schema,
                                                 const SchemaRef &right_schema, idx_t output_count) {
	duckdb::vector<std::string> output_names;
	auto left_names = duckdb::distributed::GetSchemaNames(left_schema);
	auto right_names = duckdb::distributed::GetSchemaNames(right_schema);

	auto append_by_index = [&](const duckdb::vector<std::string> &names, const duckdb::vector<idx_t> &indices) {
		for (auto idx : indices) {
			if (idx < names.size() && !names[idx].empty()) {
				output_names.push_back(names[idx]);
			} else {
				output_names.push_back("c" + std::to_string(output_names.size()));
			}
		}
	};

	if (JoinOutputsLeft(hj.join_type)) {
		if (!hj.lhs_output_columns.col_idxs.empty()) {
			append_by_index(left_names, hj.lhs_output_columns.col_idxs);
		} else if (!left_names.empty()) {
			output_names.insert(output_names.end(), left_names.begin(), left_names.end());
		}
	}

	if (JoinOutputsRight(hj.join_type)) {
		if (!hj.payload_columns.col_idxs.empty()) {
			append_by_index(right_names, hj.payload_columns.col_idxs);
		} else if (!right_names.empty()) {
			output_names.insert(output_names.end(), right_names.begin(), right_names.end());
		}
	}

	if (hj.join_type == JoinType::MARK) {
		output_names.push_back("mark");
	}

	if (output_count != 0 && output_names.size() != output_count) {
		output_names.clear();
	}
	return output_names;
}

duckdb::vector<std::string> BuildCrossProductOutputNames(const SchemaRef &left_schema, const SchemaRef &right_schema,
                                                         idx_t output_count) {
	if (!left_schema || !right_schema) {
		return {};
	}
	auto output_names = duckdb::distributed::GetSchemaNames(left_schema);
	auto right_names = duckdb::distributed::GetSchemaNames(right_schema);
	output_names.insert(output_names.end(), right_names.begin(), right_names.end());
	if (output_names.size() != output_count) {
		return {};
	}
	return output_names;
}

duckdb::vector<std::string> BuildComparisonJoinOutputNames(JoinType join_type, const SchemaRef &left_schema,
                                                           const SchemaRef &right_schema, idx_t output_count,
                                                           const duckdb::vector<idx_t> &left_projection_map = {},
                                                           const duckdb::vector<idx_t> &right_projection_map = {}) {
	if (!left_schema || !right_schema) {
		return {};
	}

	duckdb::vector<std::string> output_names;
	bool names_valid = true;
	auto append_names = [&](const SchemaRef &schema, const duckdb::vector<idx_t> &projection_map) {
		auto names = duckdb::distributed::GetSchemaNames(schema);
		if (projection_map.empty()) {
			output_names.insert(output_names.end(), names.begin(), names.end());
			return;
		}
		for (auto index : projection_map) {
			if (index >= names.size()) {
				names_valid = false;
				return;
			}
			output_names.push_back(names[index]);
		}
	};
	if (JoinOutputsLeft(join_type)) {
		append_names(left_schema, left_projection_map);
	}
	if (JoinOutputsRight(join_type)) {
		append_names(right_schema, right_projection_map);
	}
	if (join_type == JoinType::MARK) {
		output_names.push_back("mark");
	}
	if (!names_valid || output_names.size() != output_count) {
		return {};
	}
	return output_names;
}

duckdb::vector<JoinCondition> CopyJoinConditions(const duckdb::vector<JoinCondition> &conditions) {
	duckdb::vector<JoinCondition> copy;
	copy.reserve(conditions.size());
	for (const auto &cond : conditions) {
		JoinCondition new_cond;
		new_cond.comparison = cond.comparison;
		if (cond.left) {
			new_cond.left = cond.left->Copy();
		}
		if (cond.right) {
			new_cond.right = cond.right->Copy();
		}
		copy.push_back(std::move(new_cond));
	}
	return copy;
}

duckdb::vector<unique_ptr<BaseStatistics>> CopyJoinStats(const duckdb::vector<unique_ptr<BaseStatistics>> &stats) {
	duckdb::vector<unique_ptr<BaseStatistics>> copy;
	copy.reserve(stats.size());
	for (const auto &entry : stats) {
		if (entry) {
			copy.push_back(entry->ToUnique());
		} else {
			copy.push_back(nullptr);
		}
	}
	return copy;
}

unique_ptr<JoinFilterPushdownInfo> CopyJoinFilterPushdownInfo(const unique_ptr<JoinFilterPushdownInfo> &info) {
	if (!info) {
		return nullptr;
	}
	auto copy = make_uniq<JoinFilterPushdownInfo>();
	copy->join_condition = info->join_condition;
	copy->probe_info.reserve(info->probe_info.size());
	for (const auto &probe : info->probe_info) {
		JoinFilterPushdownFilter new_probe;
		new_probe.dynamic_filters = probe.dynamic_filters;
		new_probe.columns = probe.columns;
		copy->probe_info.push_back(std::move(new_probe));
	}
	copy->min_max_aggregates.reserve(info->min_max_aggregates.size());
	for (const auto &expr : info->min_max_aggregates) {
		if (expr) {
			copy->min_max_aggregates.push_back(expr->Copy());
		} else {
			copy->min_max_aggregates.push_back(nullptr);
		}
	}
	return copy;
}

idx_t EstimateRowWidthBytes(const duckdb::vector<LogicalType> &types) {
	constexpr idx_t VARIABLE_TYPE_AVG_BYTES = 32;
	idx_t width = 0;
	for (auto &type : types) {
		auto physical = type.InternalType();
		if (physical == PhysicalType::VARCHAR || physical == PhysicalType::LIST || physical == PhysicalType::STRUCT ||
		    physical == PhysicalType::ARRAY) {
			width += VARIABLE_TYPE_AVG_BYTES;
		} else {
			width += GetTypeIdSize(physical);
		}
	}
	return MaxValue<idx_t>(width, 1);
}

idx_t EstimateDataSizeBytes(idx_t cardinality, const duckdb::vector<LogicalType> &types) {
	return cardinality * EstimateRowWidthBytes(types);
}

idx_t GetAutoBroadcastThresholdBytes() {
	constexpr idx_t DEFAULT_THRESHOLD_BYTES = 10ULL * 1024 * 1024;
	const char *env = std::getenv("VANE_DISTRIBUTED_AUTO_BROADCAST_THRESHOLD_BYTES");
	if (!env || !*env) {
		return DEFAULT_THRESHOLD_BYTES;
	}
	std::string value(env);
	std::transform(value.begin(), value.end(), value.begin(),
	               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	if (value == "0" || value == "false" || value == "no" || value == "off") {
		return 0;
	}
	errno = 0;
	char *end = nullptr;
	unsigned long long parsed = std::strtoull(env, &end, 10);
	if (errno != 0 || end == env || *end != '\0') {
		return DEFAULT_THRESHOLD_BYTES;
	}
	return static_cast<idx_t>(parsed);
}

enum class DistributedJoinStrategyOverride { kDefault, kHash, kBroadcast, kBroadcastLeft, kBroadcastRight };

DistributedJoinStrategyOverride GetJoinStrategyOverride() {
	const char *env = std::getenv("VANE_DISTRIBUTED_JOIN_STRATEGY");
	if (!env || !*env) {
		return DistributedJoinStrategyOverride::kDefault;
	}
	std::string value(env);
	std::transform(value.begin(), value.end(), value.begin(),
	               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	if (value == "hash") {
		return DistributedJoinStrategyOverride::kHash;
	}
	if (value == "broadcast") {
		return DistributedJoinStrategyOverride::kBroadcast;
	}
	if (value == "broadcast_left" || value == "broadcast-left" || value == "left") {
		return DistributedJoinStrategyOverride::kBroadcastLeft;
	}
	if (value == "broadcast_right" || value == "broadcast-right" || value == "right") {
		return DistributedJoinStrategyOverride::kBroadcastRight;
	}
	return DistributedJoinStrategyOverride::kDefault;
}

Optional<bool> BroadcastReceiverRepartitionOverride() {
	const char *env = std::getenv("VANE_DISTRIBUTED_BROADCAST_JOIN_RECEIVER_REPARTITION");
	if (!env || !*env) {
		return Optional<bool> {};
	}
	std::string value(env);
	std::transform(value.begin(), value.end(), value.begin(),
	               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	if (value == "1" || value == "true" || value == "yes" || value == "on") {
		return true;
	}
	if (value == "0" || value == "false" || value == "no" || value == "off") {
		return false;
	}
	return Optional<bool> {};
}

} // namespace

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateCrossProduct(
    const PhysicalCrossProduct &cross_product, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	if (children.size() != 2 || !children[0] || !children[1]) {
		throw InvalidInputException("Distributed cross product requires exactly two input nodes");
	}

	SchemaRef schema = nullptr;
	if (!cross_product.GetTypes().empty()) {
		auto output_names = BuildCrossProductOutputNames(children[0]->config().schema(), children[1]->config().schema(),
		                                                 cross_product.GetTypes().size());
		if (!output_names.empty()) {
			schema = MakeSchemaRef(cross_product.GetTypes(), output_names);
		} else {
			schema = MakeSchemaRef(cross_product.GetTypes());
		}
	}

	// Correctness baseline: every row on each side must meet every row on the
	// other side. The gather policy stays in translation so a future broadcast
	// or partition-pair strategy does not change task fan-in or physical serde.
	auto left = gen_gather_node(children[0]);
	auto right = gen_gather_node(children[1]);
	return std::make_shared<CrossProductNode>(get_next_pipeline_node_id(), plan_config_, cross_product.GetTypes(),
	                                          cross_product.estimated_cardinality, std::move(left), std::move(right),
	                                          std::move(schema));
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateHashJoin(
    const PhysicalHashJoin &hj, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	DistributedPipelineNodeRef left_node = nullptr;
	DistributedPipelineNodeRef right_node = nullptr;
	if (children.size() > 0) {
		left_node = children[0];
	}
	if (children.size() > 1) {
		right_node = children[1];
	}

	auto conditions = CopyJoinConditions(hj.conditions);
	auto join_stats = CopyJoinStats(hj.join_stats);
	auto filter_pushdown = CopyJoinFilterPushdownInfo(hj.filter_pushdown);

	SchemaRef schema = nullptr;
	if (!hj.GetTypes().empty()) {
		SchemaRef left_schema = left_node ? left_node->config().schema() : nullptr;
		SchemaRef right_schema = right_node ? right_node->config().schema() : nullptr;
		auto output_names =
		    BuildJoinOutputNames(hj, left_schema, right_schema, static_cast<idx_t>(hj.GetTypes().size()));
		if (!output_names.empty() && output_names.size() == hj.GetTypes().size()) {
			schema = MakeSchemaRef(hj.GetTypes(), output_names);
		} else {
			schema = MakeSchemaRef(hj.GetTypes());
		}
	}

	std::vector<duckdb::ExprRef> left_partition_by;
	std::vector<duckdb::ExprRef> right_partition_by;
	duckdb::vector<unique_ptr<Expression>> mark_join_build_summary_expressions;
	const bool correlated_mark_counts =
	    hj.join_type == JoinType::MARK && !hj.delim_types.empty() && hj.delim_types.size() + 1 == conditions.size();
	const auto partition_condition_count = correlated_mark_counts ? hj.delim_types.size() : conditions.size();
	left_partition_by.reserve(conditions.size());
	right_partition_by.reserve(conditions.size());
	for (idx_t condition_idx = 0; condition_idx < conditions.size(); condition_idx++) {
		const auto &cond = conditions[condition_idx];
		if (hj.join_type == JoinType::MARK && !correlated_mark_counts && cond.right &&
		    cond.comparison != ExpressionType::COMPARE_DISTINCT_FROM &&
		    cond.comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			mark_join_build_summary_expressions.push_back(cond.right->Copy());
		}
		if (condition_idx >= partition_condition_count) {
			continue;
		}
		if (cond.comparison != ExpressionType::COMPARE_EQUAL &&
		    cond.comparison != ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
			continue;
		}
		if (cond.left && cond.right) {
			left_partition_by.emplace_back(duckdb::ExprRef(cond.left->Copy().release()));
			right_partition_by.emplace_back(duckdb::ExprRef(cond.right->Copy().release()));
		}
	}

	auto join_override = GetJoinStrategyOverride();
	Optional<BroadcastJoinSide> broadcast_side;

	if (join_override == DistributedJoinStrategyOverride::kBroadcastLeft ||
	    join_override == DistributedJoinStrategyOverride::kBroadcastRight) {
		auto requested_side = join_override == DistributedJoinStrategyOverride::kBroadcastLeft
		                          ? BroadcastJoinSide::LEFT
		                          : BroadcastJoinSide::RIGHT;
		ValidateBroadcastJoinSide(hj.join_type, requested_side);
		broadcast_side = requested_side;
	} else if (join_override == DistributedJoinStrategyOverride::kBroadcast && left_node && right_node) {
		bool left_safe = IsBroadcastJoinSideSemanticallySafe(hj.join_type, BroadcastJoinSide::LEFT);
		bool right_safe = IsBroadcastJoinSideSemanticallySafe(hj.join_type, BroadcastJoinSide::RIGHT);
		if (!left_safe && !right_safe) {
			throw InvalidInputException("Cannot use broadcast strategy for a %s join because neither side can be "
			                            "broadcast without changing its semantics",
			                            EnumUtil::ToString(hj.join_type));
		}
		if (left_safe && right_safe) {
			size_t left_parts = left_node->config().clustering_spec()->num_partitions();
			size_t right_parts = right_node->config().clustering_spec()->num_partitions();
			broadcast_side = left_parts > right_parts ? BroadcastJoinSide::RIGHT : BroadcastJoinSide::LEFT;
		} else {
			broadcast_side = left_safe ? BroadcastJoinSide::LEFT : BroadcastJoinSide::RIGHT;
		}
	} else if (join_override == DistributedJoinStrategyOverride::kDefault && left_node && right_node) {
		auto threshold_bytes = GetAutoBroadcastThresholdBytes();
		if (threshold_bytes > 0) {
			idx_t left_card = hj.children.size() > 0 ? hj.children[0].get().estimated_cardinality : 0;
			idx_t right_card = hj.children.size() > 1 ? hj.children[1].get().estimated_cardinality : 0;
			auto &left_types = hj.children.size() > 0 ? hj.children[0].get().GetTypes() : hj.GetTypes();
			auto &right_types = hj.children.size() > 1 ? hj.children[1].get().GetTypes() : hj.GetTypes();
			idx_t left_bytes = EstimateDataSizeBytes(left_card, left_types);
			idx_t right_bytes = EstimateDataSizeBytes(right_card, right_types);
			bool left_small = left_card > 0 && left_bytes <= threshold_bytes &&
			                  IsBroadcastJoinSideSemanticallySafe(hj.join_type, BroadcastJoinSide::LEFT);
			bool right_small = right_card > 0 && right_bytes <= threshold_bytes &&
			                   IsBroadcastJoinSideSemanticallySafe(hj.join_type, BroadcastJoinSide::RIGHT);
			if (left_small || right_small) {
				if (left_small && right_small) {
					broadcast_side = left_bytes <= right_bytes ? BroadcastJoinSide::LEFT : BroadcastJoinSide::RIGHT;
				} else if (left_small) {
					broadcast_side = BroadcastJoinSide::LEFT;
				} else {
					broadcast_side = BroadcastJoinSide::RIGHT;
				}
			}
		}
	}

	if (broadcast_side && left_node && right_node) {
		bool broadcast_right = *broadcast_side == BroadcastJoinSide::RIGHT;
		auto broadcaster = broadcast_right ? right_node : left_node;
		auto receiver = broadcast_right ? left_node : right_node;

		bool repartition_receiver = BroadcastReceiverRepartitionOverride().value_or(false);
		if (repartition_receiver && plan_config_.num_partitions > 1) {
			const auto &receiver_keys = broadcast_right ? left_partition_by : right_partition_by;
			if (!receiver_keys.empty()) {
				size_t target_partitions = plan_config_.num_partitions;
				auto receiver_spec = RepartitionSpec::create_hash(target_partitions, receiver_keys);
				receiver = gen_shuffle_node(std::move(receiver_spec), receiver->config().schema(), receiver);
			}
		}

		return std::make_shared<BroadcastJoinNode>(
		    get_next_pipeline_node_id(), plan_config_, std::move(conditions), hj.join_type, hj.GetTypes(),
		    hj.delim_types, hj.condition_types, hj.payload_columns, hj.lhs_output_columns, hj.rhs_output_columns,
		    std::move(join_stats), std::move(filter_pushdown), hj.estimated_cardinality, *broadcast_side, broadcaster,
		    receiver, schema, exchange_mgr_);
	}

	DistributedPipelineNodeRef join_left = left_node;
	DistributedPipelineNodeRef join_right = right_node;
	optional_idx mark_build_summary_source_node_id;
	bool needs_join_repartition = (plan_config_.num_partitions > 1);
	if (!needs_join_repartition && left_node && right_node) {
		size_t left_parts = left_node->config().clustering_spec()->num_partitions();
		size_t right_parts = right_node->config().clustering_spec()->num_partitions();
		if (left_parts > 1 || right_parts > 1) {
			needs_join_repartition = true;
		}
	}
	if (needs_join_repartition && left_node && right_node) {
		if (left_partition_by.empty() || right_partition_by.empty()) {
			join_left = gen_gather_node(left_node);
			join_right = gen_gather_node(right_node);
		} else {
			size_t target_partitions =
			    std::max(static_cast<size_t>(plan_config_.num_partitions), static_cast<size_t>(1));
			auto left_spec = RepartitionSpec::create_hash(target_partitions, std::move(left_partition_by));
			join_left = gen_shuffle_node(std::move(left_spec), left_node->config().schema(), left_node);
			auto right_spec = RepartitionSpec::create_hash(target_partitions, std::move(right_partition_by));
			join_right = gen_shuffle_node(std::move(right_spec), right_node->config().schema(), right_node);
			if (hj.join_type == JoinType::MARK && !correlated_mark_counts && target_partitions > 1) {
				auto right_repartition = std::dynamic_pointer_cast<RepartitionNode>(join_right->inner());
				if (!right_repartition) {
					throw InternalException("MARK join build shuffle is not a RepartitionNode");
				}
				right_repartition->EnableMarkJoinBuildSummary(std::move(mark_join_build_summary_expressions));
				mark_build_summary_source_node_id = optional_idx(right_repartition->node_id());
			}
		}
	}

	return std::make_shared<HashJoinNode>(get_next_pipeline_node_id(), plan_config_, std::move(conditions),
	                                      hj.join_type, hj.GetTypes(), hj.delim_types, hj.condition_types,
	                                      hj.payload_columns, hj.lhs_output_columns, hj.rhs_output_columns,
	                                      std::move(join_stats), std::move(filter_pushdown), hj.estimated_cardinality,
	                                      join_left, join_right, schema, mark_build_summary_source_node_id);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateDelimJoin(
    const PhysicalDelimJoin &dj, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	DistributedPipelineNodeRef child_node = nullptr;
	if (!children.empty()) {
		child_node = children[0];
	}
	SchemaRef schema = nullptr;
	if (!dj.GetTypes().empty()) {
		schema = MakeSchemaRef(dj.GetTypes());
	}
	return std::make_shared<DelimJoinNode>(get_next_pipeline_node_id(), dj, child_node, schema, plan_config_.db);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateNestedLoopJoin(
    const PhysicalNestedLoopJoin &nlj, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	return TranslateComparisonNestedLoopJoin(nlj, nlj.predicate.get(), children);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateRangeJoin(
    const PhysicalRangeJoin &range_join, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	return TranslateComparisonNestedLoopJoin(range_join, nullptr, children, range_join.left_projection_map,
	                                         range_join.right_projection_map);
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateComparisonNestedLoopJoin(
    const PhysicalComparisonJoin &join, const Expression *predicate,
    const std::vector<std::shared_ptr<DistributedPipelineNode>> &children, duckdb::vector<idx_t> left_projection_map,
    duckdb::vector<idx_t> right_projection_map) {
	if (children.size() != 2 || !children[0] || !children[1]) {
		throw InvalidInputException("Distributed %s requires exactly two input nodes", EnumUtil::ToString(join.type));
	}

	SchemaRef schema = nullptr;
	if (!join.GetTypes().empty()) {
		auto output_names = BuildComparisonJoinOutputNames(join.join_type, children[0]->config().schema(),
		                                                   children[1]->config().schema(), join.GetTypes().size(),
		                                                   left_projection_map, right_projection_map);
		schema = output_names.empty() ? MakeSchemaRef(join.GetTypes()) : MakeSchemaRef(join.GetTypes(), output_names);
	}

	// Correctness baseline for non-equality joins: every potentially matching
	// row pair must reach the same worker. Keeping this policy in translation
	// leaves the task fan-in reusable for a future range-partition strategy.
	auto left = gen_gather_node(children[0]);
	auto right = gen_gather_node(children[1]);
	return std::make_shared<NestedLoopJoinNode>(
	    get_next_pipeline_node_id(), plan_config_, join.type, CopyJoinConditions(join.conditions),
	    predicate ? predicate->Copy() : nullptr, join.join_type, join.GetTypes(), join.estimated_cardinality,
	    std::move(left), std::move(right), std::move(schema), std::move(left_projection_map),
	    std::move(right_projection_map));
}

std::shared_ptr<PipelineNodeImpl> PhysicalPlanToPipelineNodeTranslator::TranslateBlockwiseNLJoin(
    const PhysicalBlockwiseNLJoin &join, const std::vector<std::shared_ptr<DistributedPipelineNode>> &children) {
	if (children.size() != 2 || !children[0] || !children[1]) {
		throw InvalidInputException("Distributed blockwise nested-loop join requires exactly two input nodes");
	}
	if (!join.condition) {
		throw InvalidInputException("Distributed blockwise nested-loop join requires a condition");
	}

	SchemaRef schema = nullptr;
	if (!join.GetTypes().empty()) {
		auto output_names = BuildComparisonJoinOutputNames(join.join_type, children[0]->config().schema(),
		                                                   children[1]->config().schema(), join.GetTypes().size());
		schema = output_names.empty() ? MakeSchemaRef(join.GetTypes()) : MakeSchemaRef(join.GetTypes(), output_names);
	}

	auto left = gen_gather_node(children[0]);
	auto right = gen_gather_node(children[1]);
	return std::make_shared<NestedLoopJoinNode>(get_next_pipeline_node_id(), plan_config_, join.condition->Copy(),
	                                            join.join_type, join.GetTypes(), join.estimated_cardinality,
	                                            std::move(left), std::move(right), std::move(schema));
}

} // namespace distributed
} // namespace duckdb
