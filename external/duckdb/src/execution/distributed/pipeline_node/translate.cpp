// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <functional>

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"
#include "duckdb/execution/physical_operator_visitor.hpp"
#include "duckdb/common/exception.hpp"

#include "duckdb/execution/distributed/pipeline_node/translator_scan.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/distributed/exchange/flight_exchange_manager.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/projection/physical_vllm.hpp"

#include "duckdb/execution/operator/helper/physical_limit.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_limit.hpp"
#include "duckdb/execution/operator/helper/physical_limit_percent.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/operator/exchange/physical_local_exchange.hpp"
#include "duckdb/execution/operator/exchange/physical_repartition.hpp"
#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_perfecthash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_partitioned_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_streaming_window.hpp"
#include "duckdb/execution/operator/aggregate/physical_window.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/execution/operator/scan/physical_dummy_scan.hpp"
#include "duckdb/execution/operator/scan/physical_expression_scan.hpp"
#include "duckdb/execution/operator/projection/physical_unnest.hpp"
#include "duckdb/execution/operator/projection/physical_pivot.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/execution/operator/helper/physical_reservoir_sample.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_sample.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/execution/operator/join/physical_delim_join.hpp"
#include "duckdb/execution/operator/persistent/physical_copy_to_file.hpp"
#include "duckdb/execution/operator/persistent/physical_batch_copy_to_file.hpp"
#include "duckdb/execution/distributed/plan/scan_task.hpp"

namespace duckdb {
namespace distributed {

PhysicalPlanToPipelineNodeTranslator::PhysicalPlanToPipelineNodeTranslator(PlanConfig plan_config,
                                                                           DuckPhysicalPlanRef plan,
                                                                           ClientContext *client_context)
    : plan_config_(std::move(plan_config)), plan_(std::move(plan)),
      exchange_mgr_(std::make_shared<FlightExchangeManager>(ResolveFlightExchangeConfigFromEnv(), client_context)) {
}

DuckDBResult<std::shared_ptr<DistributedPipelineNode>>
PhysicalPlanToPipelineNodeTranslator::physical_plan_to_pipeline_node(PlanConfig plan_config, DuckPhysicalPlanRef plan,
                                                                     ClientContext *client_context) {
	PhysicalPlanToPipelineNodeTranslator translator(std::move(plan_config), plan, client_context);
	if (!plan) {
		return DuckDBResult<std::shared_ptr<DistributedPipelineNode>>::err(
		    DuckDBError::invalid_state_error("physical plan is null"));
	}
	if (!plan->HasRoot()) {
		return DuckDBResult<std::shared_ptr<DistributedPipelineNode>>::err(
		    DuckDBError::invalid_state_error("physical plan has no root"));
	}
	try {
		translator.VisitOperator(plan->Root());
	} catch (const std::exception &ex) {
		return DuckDBResult<std::shared_ptr<DistributedPipelineNode>>::err(
		    DuckDBError::invalid_state_error(std::string("failed to translate physical plan: ") + ex.what()));
	}
	if (translator.node_stack_.empty()) {
		return DuckDBResult<std::shared_ptr<DistributedPipelineNode>>::err(
		    DuckDBError::invalid_state_error("translation produced empty node stack"));
	}
	if (!translator.node_stack_.back()) {
		return DuckDBResult<std::shared_ptr<DistributedPipelineNode>>::err(
		    DuckDBError::invalid_state_error("translation produced null root node"));
	}
	return DuckDBResult<std::shared_ptr<DistributedPipelineNode>>::ok(translator.node_stack_.back());
}

DuckDBResult<std::shared_ptr<DistributedPipelineNode>>
PhysicalPlanToPipelineNodeTranslator::gen_shuffle_node(std::shared_ptr<RepartitionSpec> repartition_spec,
                                                       SchemaRef schema,
                                                       std::shared_ptr<DistributedPipelineNode> child) {
	size_t upstream_num = child->config().clustering_spec()->num_partitions();
	auto clustering = repartition_spec->to_clustering_spec(upstream_num);
	size_t num_partitions = clustering->num_partitions();

	auto plan_cfg_ptr = std::make_shared<PlanConfig>(plan_config_);
	return DuckDBResult<std::shared_ptr<DistributedPipelineNode>>::ok(
	    RepartitionNode::create(get_next_pipeline_node_id(), plan_cfg_ptr, repartition_spec, num_partitions, schema,
	                            child, exchange_mgr_)
	        ->into_node());
}

std::shared_ptr<DistributedPipelineNode>
PhysicalPlanToPipelineNodeTranslator::gen_gather_node(std::shared_ptr<DistributedPipelineNode> input_node) {
	if (input_node->config().clustering_spec()->num_partitions() == 1) {
		return input_node;
	}

	auto spec = RepartitionSpec::create_into_partitions(1);
	auto res = gen_shuffle_node(spec, input_node->config().schema(), input_node);
	if (!res) {
		throw std::runtime_error(std::string("gen_gather_node failed: ") + res.error().what());
	}
	return res.value();
}

void PhysicalPlanToPipelineNodeTranslator::VisitOperator(::duckdb::PhysicalOperator &op) {
	// First recurse into children using the base helper
	PhysicalOperatorVisitor::VisitOperatorChildren(op);

	// collect child distributed nodes (if any)
	size_t n_children = op.children.size();
	std::vector<std::shared_ptr<DistributedPipelineNode>> children;
	children.reserve(n_children);
	for (size_t i = 0; i < n_children; ++i) {
		if (node_stack_.empty()) {
			throw InternalException("Translator node stack underflow: missing child result");
		}
		children.push_back(node_stack_.back());
		node_stack_.pop_back();
	}
	// children were pushed left-to-right; reverse to restore original order
	std::reverse(children.begin(), children.end());

	// Create a pipeline node implementation depending on operator type
	std::shared_ptr<PipelineNodeImpl> node_impl;
	switch (op.type) {
	case PhysicalOperatorType::FILTER: {
		auto &pf = static_cast<PhysicalFilter &>(op);
		node_impl = TranslateFilter(pf, children);
		break;
	}
	case PhysicalOperatorType::PROJECTION: {
		auto &pp = static_cast<PhysicalProjection &>(op);
		node_impl = TranslateProjection(pp, children);
		break;
	}
	case PhysicalOperatorType::LOCAL_EXCHANGE: {
		auto &pr = static_cast<PhysicalLocalExchange &>(op);
		node_impl = TranslateLocalExchange(pr, children);
		break;
	}
	case PhysicalOperatorType::REPARTITION: {
		auto &pre = static_cast<PhysicalRepartition &>(op);
		node_stack_.push_back(TranslateRepartition(pre, children));
		return;
	}
	case PhysicalOperatorType::VLLM_PROJECT: {
		auto &pv = static_cast<PhysicalVLLM &>(op);
		node_impl = TranslateVLLMProject(pv, children);
		break;
	}
	case PhysicalOperatorType::LIMIT: {
		auto &pl = static_cast<PhysicalLimit &>(op);
		node_impl = TranslateLimit(pl, children);
		break;
	}
	case PhysicalOperatorType::STREAMING_LIMIT: {
		auto &pl = static_cast<PhysicalStreamingLimit &>(op);
		node_impl = TranslateStreamingLimit(pl, children);
		break;
	}
	case PhysicalOperatorType::LIMIT_PERCENT: {
		auto &pl = static_cast<PhysicalLimitPercent &>(op);
		node_impl = TranslateLimitPercent(pl, children);
		break;
	}
	case PhysicalOperatorType::ORDER_BY: {
		auto &po = static_cast<PhysicalOrder &>(op);
		node_impl = TranslateOrderBy(po, children);
		break;
	}
	case PhysicalOperatorType::TOP_N: {
		auto &topn = static_cast<PhysicalTopN &>(op);
		node_impl = TranslateTopN(topn, children);
		break;
	}
	case PhysicalOperatorType::RESERVOIR_SAMPLE: {
		auto &rs = static_cast<PhysicalReservoirSample &>(op);
		node_impl = TranslateReservoirSample(rs, children);
		break;
	}
	case PhysicalOperatorType::STREAMING_SAMPLE: {
		auto &ss = static_cast<PhysicalStreamingSample &>(op);
		node_impl = TranslateStreamingSample(ss, children);
		break;
	}
	case PhysicalOperatorType::PIVOT: {
		auto &pp = static_cast<PhysicalPivot &>(op);
		node_impl = TranslatePivot(pp, children);
		break;
	}
	case PhysicalOperatorType::UNNEST: {
		auto &pu = static_cast<PhysicalUnnest &>(op);
		node_impl = TranslateUnnest(pu, children);
		break;
	}
	case PhysicalOperatorType::INOUT_FUNCTION: {
		auto &pio = static_cast<PhysicalTableInOutFunction &>(op);
		node_impl = TranslateTableInOut(pio, children);
		break;
	}
	case PhysicalOperatorType::STREAMING_UDF: {
		auto &pio = static_cast<PhysicalStreamingUDF &>(op);
		node_impl = TranslateStreamingUDF(pio, children);
		break;
	}
	case PhysicalOperatorType::WINDOW:
	case PhysicalOperatorType::STREAMING_WINDOW: {
		if (op.type == PhysicalOperatorType::WINDOW) {
			auto &pw = static_cast<PhysicalWindow &>(op);
			node_impl = TranslateWindow(pw, children);
		} else {
			auto &psw = static_cast<PhysicalStreamingWindow &>(op);
			node_impl = TranslateStreamingWindow(psw, children);
		}
		break;
	}
	case PhysicalOperatorType::EXPRESSION_SCAN: {
		auto &es = static_cast<PhysicalExpressionScan &>(op);
		node_impl = TranslateExpressionScan(es, children);
		break;
	}
	case PhysicalOperatorType::HASH_GROUP_BY: {
		auto &ha = static_cast<PhysicalHashAggregate &>(op);
		auto agg_node = TranslateHashGroupBy(ha, children);
		node_stack_.push_back(agg_node);
		return;
	}
	case PhysicalOperatorType::PERFECT_HASH_GROUP_BY: {
		auto &pha = static_cast<PhysicalPerfectHashAggregate &>(op);
		auto agg_node = TranslatePerfectHashGroupBy(pha, children);
		node_stack_.push_back(agg_node);
		return;
	}
	case PhysicalOperatorType::PARTITIONED_AGGREGATE: {
		auto &pa = static_cast<PhysicalPartitionedAggregate &>(op);
		auto agg_node = TranslatePartitionedAggregate(pa, children);
		node_stack_.push_back(agg_node);
		return;
	}
	case PhysicalOperatorType::UNGROUPED_AGGREGATE: {
		auto &ua = static_cast<PhysicalUngroupedAggregate &>(op);
		auto agg_node = TranslateUngroupedAggregate(ua, children);
		node_stack_.push_back(agg_node);
		return;
	}
	case PhysicalOperatorType::HASH_JOIN: {
		auto &hj = static_cast<PhysicalHashJoin &>(op);
		node_impl = TranslateHashJoin(hj, children);
		break;
	}
	case PhysicalOperatorType::LEFT_DELIM_JOIN:
	case PhysicalOperatorType::RIGHT_DELIM_JOIN: {
		auto &dj = static_cast<PhysicalDelimJoin &>(op);
		node_impl = TranslateDelimJoin(dj, children);
		break;
	}
	case PhysicalOperatorType::CTE: {
		node_stack_.push_back(TranslateCTESource(op));
		return;
	}
	case PhysicalOperatorType::DUMMY_SCAN: {
		auto &dummy_scan = static_cast<PhysicalDummyScan &>(op);
		node_stack_.push_back(TranslateDummyScanSource(dummy_scan));
		return;
	}
	case PhysicalOperatorType::COLUMN_DATA_SCAN:
	case PhysicalOperatorType::CHUNK_SCAN:
	case PhysicalOperatorType::CTE_SCAN:
	case PhysicalOperatorType::DELIM_SCAN: {
		auto &col_scan = static_cast<PhysicalColumnDataScan &>(op);
		node_stack_.push_back(TranslateColumnDataScanSource(col_scan));
		return;
	}
	case PhysicalOperatorType::RECURSIVE_CTE_SCAN:
	case PhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN: {
		throw NotImplementedException("Distributed pipeline does not support recursive CTE scans");
	}
	case PhysicalOperatorType::COPY_TO_FILE:
	case PhysicalOperatorType::BATCH_COPY_TO_FILE: {
		if (op.type == PhysicalOperatorType::COPY_TO_FILE) {
			auto &copy_op = static_cast<PhysicalCopyToFile &>(op);
			node_impl = TranslateCopyToFile(copy_op, children);
		} else {
			auto &batch_op = static_cast<PhysicalBatchCopyToFile &>(op);
			node_impl = TranslateBatchCopyToFile(batch_op, children);
		}
		break;
	}
	case PhysicalOperatorType::TABLE_SCAN: {
		auto &table_scan = static_cast<PhysicalTableScan &>(op);
		node_stack_.push_back(TranslateTableScanSource(table_scan));
		return;
	}
	case PhysicalOperatorType::NESTED_LOOP_JOIN: {
		auto &nlj = static_cast<PhysicalNestedLoopJoin &>(op);
		node_impl = TranslateNestedLoopJoin(nlj, children);
		break;
	}
	default: {
		throw NotImplementedException("Distributed pipeline does not support operator type: %s", op.GetName());
	}
	}

	std::shared_ptr<DistributedPipelineNode> dist_node;
	// Avoid wrapping a DistributedPipelineNode inside another
	if (auto existing = std::dynamic_pointer_cast<DistributedPipelineNode>(node_impl)) {
		dist_node = existing;
	} else {
		dist_node = std::make_shared<DistributedPipelineNode>(node_impl);
	}

	// If we have children, wire them into the distributed node.
	// Prefer the node_impl-defined children when present, since some nodes
	// (e.g., distributed hash join) wrap/insert additional pipeline nodes.
	const auto impl_children = node_impl ? node_impl->children() : std::vector<PipelineNodeRef> {};
	if (!children.empty() && impl_children.empty()) {
		auto r = dist_node->with_new_children(std::move(children));
		if (!r.is_ok()) {
			throw InternalException("Failed to set children on DistributedPipelineNode");
		}
		dist_node = r.value();
	}

	// Push the constructed node onto the stack
	node_stack_.push_back(dist_node);
}

// Wrapper implementation for translator API declared in `translator_api.hpp`.
// This keeps the heavy translator implementation out of widely-included
// headers while providing a single externally-linkable symbol.
DuckDBResult<std::shared_ptr<DistributedPipelineNode>>
physical_plan_to_pipeline_node_wrapper(PlanConfig plan_config, DuckPhysicalPlanRef plan,
                                       ClientContext *client_context) {
	auto result = PhysicalPlanToPipelineNodeTranslator::physical_plan_to_pipeline_node(std::move(plan_config), plan,
	                                                                                   client_context);
	return result;
}

std::unordered_map<idx_t, std::vector<ScanTaskDescriptor>>
physical_plan_scan_task_map_wrapper(DuckPhysicalPlanRef plan, DuckDBExecutionConfigRef config,
                                    shared_ptr<DatabaseInstance> db) {
	std::unordered_map<idx_t, std::vector<ScanTaskDescriptor>> out;
	if (!plan || !plan->HasRoot()) {
		return out;
	}

	DuckDBExecutionConfigRef exec_cfg = std::move(config);
	if (!exec_cfg) {
		exec_cfg = std::make_shared<DuckDBExecutionConfig>(DuckDBExecutionConfig::from_env());
	}

	idx_t max_id = 0;
	std::function<void(PhysicalOperator &)> update_max;
	update_max = [&](PhysicalOperator &op) -> void {
		if (op.type == PhysicalOperatorType::TABLE_SCAN) {
			auto &scan = op.Cast<PhysicalTableScan>();
			if (scan.extra_info.scan_node_id.IsValid()) {
				max_id = std::max(max_id, scan.extra_info.scan_node_id.GetIndex());
			}
			if (scan.extra_info.scan_group_id.IsValid()) {
				max_id = std::max(max_id, scan.extra_info.scan_group_id.GetIndex());
			}
		}
		for (auto &child : op.children) {
			update_max(child.get());
		}
	};
	update_max(plan->Root());

	idx_t next_id = max_id + 1;
	std::function<void(PhysicalOperator &)> collect;
	collect = [&](PhysicalOperator &op) -> void {
		if (op.type == PhysicalOperatorType::TABLE_SCAN) {
			auto &scan = op.Cast<PhysicalTableScan>();
			if (!scan.extra_info.scan_group_id.IsValid()) {
				if (scan.extra_info.scan_node_id.IsValid()) {
					scan.extra_info.scan_group_id = scan.extra_info.scan_node_id;
				} else {
					scan.extra_info.scan_group_id = optional_idx(next_id++);
				}
			}
			if (!scan.extra_info.scan_node_id.IsValid()) {
				scan.extra_info.scan_node_id = optional_idx(next_id++);
			}

			auto tasks = MakeTableScanTasks(scan, *exec_cfg, db);
			if (!tasks.empty()) {
				out.emplace(scan.extra_info.scan_node_id.GetIndex(), std::move(tasks));
			}
		}
		for (auto &child : op.children) {
			collect(child.get());
		}
	};
	collect(plan->Root());
	return out;
}

} // namespace distributed
} // namespace duckdb
