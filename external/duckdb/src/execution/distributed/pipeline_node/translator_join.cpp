// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdlib>

#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/broadcast_join.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/delim_join.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/hash_join.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/distributed/utils/optional.hpp"
#include "duckdb/execution/operator/join/physical_delim_join.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

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

	if (!hj.lhs_output_columns.col_idxs.empty()) {
		append_by_index(left_names, hj.lhs_output_columns.col_idxs);
	} else if (!left_names.empty()) {
		output_names.insert(output_names.end(), left_names.begin(), left_names.end());
	}

	if (hj.join_type != JoinType::ANTI && hj.join_type != JoinType::SEMI && hj.join_type != JoinType::MARK) {
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
		return 0;
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
	return 0;
}

} // namespace

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
	left_partition_by.reserve(conditions.size());
	right_partition_by.reserve(conditions.size());
	for (const auto &cond : conditions) {
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
	bool force_hash_join = (join_override == DistributedJoinStrategyOverride::kHash);
	bool force_broadcast_join = (join_override == DistributedJoinStrategyOverride::kBroadcast ||
	                             join_override == DistributedJoinStrategyOverride::kBroadcastLeft ||
	                             join_override == DistributedJoinStrategyOverride::kBroadcastRight);

	if (force_broadcast_join && hj.join_type == JoinType::OUTER) {
		force_broadcast_join = false;
	}

	if (!force_hash_join && !force_broadcast_join && left_node && right_node && hj.join_type != JoinType::OUTER) {
		auto threshold_bytes = GetAutoBroadcastThresholdBytes();
		if (threshold_bytes > 0) {
			idx_t left_card = hj.children.size() > 0 ? hj.children[0].get().estimated_cardinality : 0;
			idx_t right_card = hj.children.size() > 1 ? hj.children[1].get().estimated_cardinality : 0;
			auto &left_types = hj.children.size() > 0 ? hj.children[0].get().GetTypes() : hj.GetTypes();
			auto &right_types = hj.children.size() > 1 ? hj.children[1].get().GetTypes() : hj.GetTypes();
			idx_t left_bytes = EstimateDataSizeBytes(left_card, left_types);
			idx_t right_bytes = EstimateDataSizeBytes(right_card, right_types);
			bool left_small = left_card > 0 && left_bytes <= threshold_bytes;
			bool right_small = right_card > 0 && right_bytes <= threshold_bytes;
			if (left_small || right_small) {
				force_broadcast_join = true;
				if (left_small && right_small) {
					join_override = (left_bytes <= right_bytes) ? DistributedJoinStrategyOverride::kBroadcastLeft
					                                            : DistributedJoinStrategyOverride::kBroadcastRight;
				} else if (left_small) {
					join_override = DistributedJoinStrategyOverride::kBroadcastLeft;
				} else {
					join_override = DistributedJoinStrategyOverride::kBroadcastRight;
				}
			}
		}
	}

	if (force_broadcast_join && left_node && right_node) {
		bool is_swapped = false;
		if (join_override == DistributedJoinStrategyOverride::kBroadcastLeft) {
			is_swapped = false;
		} else if (join_override == DistributedJoinStrategyOverride::kBroadcastRight) {
			is_swapped = true;
		} else {
			size_t left_parts = left_node->config().clustering_spec()->num_partitions();
			size_t right_parts = right_node->config().clustering_spec()->num_partitions();
			bool left_is_larger = left_parts > right_parts;
			switch (hj.join_type) {
			case JoinType::INNER:
				is_swapped = left_is_larger;
				break;
			case JoinType::LEFT:
			case JoinType::SEMI:
			case JoinType::ANTI:
				is_swapped = true;
				break;
			case JoinType::RIGHT:
				is_swapped = false;
				break;
			default:
				is_swapped = left_is_larger;
				break;
			}
		}
		auto broadcaster = is_swapped ? right_node : left_node;
		auto receiver = is_swapped ? left_node : right_node;

		bool repartition_receiver = BroadcastReceiverRepartitionOverride().value_or(false);
		if (repartition_receiver && plan_config_.num_partitions > 1) {
			const auto &receiver_keys = is_swapped ? left_partition_by : right_partition_by;
			if (!receiver_keys.empty()) {
				size_t target_partitions = plan_config_.num_partitions;
				auto receiver_spec = RepartitionSpec::create_hash(target_partitions, receiver_keys);
				auto shuffle_res = gen_shuffle_node(receiver_spec, receiver->config().schema(), receiver);
				if (shuffle_res) {
					receiver = shuffle_res.value();
				}
			}
		}

		return std::make_shared<BroadcastJoinNode>(
		    get_next_pipeline_node_id(), plan_config_, std::move(conditions), hj.join_type, hj.GetTypes(),
		    hj.delim_types, hj.condition_types, hj.payload_columns, hj.lhs_output_columns, hj.rhs_output_columns,
		    std::move(join_stats), std::move(filter_pushdown), hj.estimated_cardinality, is_swapped, broadcaster,
		    receiver, schema, exchange_mgr_);
	}

	DistributedPipelineNodeRef join_left = left_node;
	DistributedPipelineNodeRef join_right = right_node;
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
			auto left_shuffle_res = gen_shuffle_node(left_spec, left_node->config().schema(), left_node);
			if (left_shuffle_res) {
				join_left = left_shuffle_res.value();
			}
			auto right_spec = RepartitionSpec::create_hash(target_partitions, std::move(right_partition_by));
			auto right_shuffle_res = gen_shuffle_node(right_spec, right_node->config().schema(), right_node);
			if (right_shuffle_res) {
				join_right = right_shuffle_res.value();
			}
		}
	}

	return std::make_shared<HashJoinNode>(
	    get_next_pipeline_node_id(), plan_config_, std::move(conditions), hj.join_type, hj.GetTypes(), hj.delim_types,
	    hj.condition_types, hj.payload_columns, hj.lhs_output_columns, hj.rhs_output_columns, std::move(join_stats),
	    std::move(filter_pushdown), hj.estimated_cardinality, join_left, join_right, schema);
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
	DistributedPipelineNodeRef left_node = nullptr;
	DistributedPipelineNodeRef right_node = nullptr;
	if (children.size() > 0) {
		left_node = children[0];
	}
	if (children.size() > 1) {
		right_node = children[1];
	}

	auto conditions = CopyJoinConditions(nlj.conditions);

	SchemaRef schema = nullptr;
	if (!nlj.GetTypes().empty()) {
		schema = MakeSchemaRef(nlj.GetTypes());
	}

	auto &lhs_input_types = nlj.children[0].get().GetTypes();
	auto &rhs_input_types = nlj.children[1].get().GetTypes();

	duckdb::vector<LogicalType> condition_types;
	unordered_map<idx_t, idx_t> build_columns_in_conditions;
	for (idx_t cond_idx = 0; cond_idx < conditions.size(); cond_idx++) {
		auto &condition = conditions[cond_idx];
		condition_types.push_back(condition.left->return_type);
		if (condition.right->GetExpressionClass() == ExpressionClass::BOUND_REF) {
			build_columns_in_conditions.emplace(condition.right->Cast<BoundReferenceExpression>().index, cond_idx);
		}
	}

	PhysicalHashJoin::JoinProjectionColumns lhs_output_columns;
	lhs_output_columns.col_idxs.reserve(lhs_input_types.size());
	for (idx_t i = 0; i < lhs_input_types.size(); i++) {
		lhs_output_columns.col_idxs.push_back(i);
		lhs_output_columns.col_types.push_back(lhs_input_types[i]);
	}

	PhysicalHashJoin::JoinProjectionColumns payload_columns;
	PhysicalHashJoin::JoinProjectionColumns rhs_output_columns;

	if (nlj.join_type != JoinType::ANTI && nlj.join_type != JoinType::SEMI && nlj.join_type != JoinType::MARK) {
		for (idx_t rhs_col = 0; rhs_col < rhs_input_types.size(); rhs_col++) {
			auto &rhs_col_type = rhs_input_types[rhs_col];
			auto it = build_columns_in_conditions.find(rhs_col);
			if (it == build_columns_in_conditions.end()) {
				payload_columns.col_idxs.push_back(rhs_col);
				payload_columns.col_types.push_back(rhs_col_type);
				rhs_output_columns.col_idxs.push_back(condition_types.size() + payload_columns.col_types.size() - 1);
			} else {
				rhs_output_columns.col_idxs.push_back(it->second);
			}
			rhs_output_columns.col_types.push_back(rhs_col_type);
		}
	}

	DistributedPipelineNodeRef join_left = left_node;
	DistributedPipelineNodeRef join_right = right_node;
	bool needs_nlj_gather = (plan_config_.num_partitions > 1);
	if (!needs_nlj_gather && left_node && right_node) {
		size_t left_parts = left_node->config().clustering_spec()->num_partitions();
		size_t right_parts = right_node->config().clustering_spec()->num_partitions();
		if (left_parts > 1 || right_parts > 1) {
			needs_nlj_gather = true;
		}
	}
	if (needs_nlj_gather && left_node && right_node) {
		join_left = gen_gather_node(left_node);
		join_right = gen_gather_node(right_node);
	}

	return std::make_shared<HashJoinNode>(get_next_pipeline_node_id(), plan_config_, std::move(conditions),
	                                      nlj.join_type, nlj.GetTypes(), duckdb::vector<LogicalType> {},
	                                      condition_types, payload_columns, lhs_output_columns, rhs_output_columns,
	                                      duckdb::vector<unique_ptr<BaseStatistics>> {}, nullptr,
	                                      nlj.estimated_cardinality, join_left, join_right, schema);
}

} // namespace distributed
} // namespace duckdb
