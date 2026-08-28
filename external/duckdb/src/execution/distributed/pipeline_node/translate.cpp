// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <functional>

#include "duckdb/execution/distributed/pipeline_node/translator.hpp"
#include "duckdb/execution/physical_operator_visitor.hpp"
#include "duckdb/common/error_data.hpp"
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
#include "duckdb/execution/operator/scan/physical_empty_result.hpp"
#include "duckdb/execution/operator/scan/physical_expression_scan.hpp"
#include "duckdb/execution/operator/projection/physical_unnest.hpp"
#include "duckdb/execution/operator/projection/physical_pivot.hpp"
#include "duckdb/execution/operator/projection/physical_tableinout_function.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/execution/operator/helper/physical_reservoir_sample.hpp"
#include "duckdb/execution/operator/helper/physical_streaming_sample.hpp"
#include "duckdb/execution/operator/helper/physical_data_sink.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/join/physical_asof_join.hpp"
#include "duckdb/execution/operator/join/physical_blockwise_nl_join.hpp"
#include "duckdb/execution/operator/join/physical_cross_product.hpp"
#include "duckdb/execution/operator/join/physical_nested_loop_join.hpp"
#include "duckdb/execution/operator/join/physical_range_join.hpp"
#include "duckdb/execution/operator/join/physical_delim_join.hpp"
#include "duckdb/execution/operator/persistent/physical_copy_to_file.hpp"
#include "duckdb/execution/operator/persistent/physical_batch_copy_to_file.hpp"
#include "duckdb/execution/operator/set/physical_union.hpp"
#include "duckdb/execution/distributed/plan/scan_split.hpp"
#include "duckdb/execution/distributed/extension_write_task_provider.hpp"
#include "duckdb/execution/distributed/pipeline_node/copy_finish.hpp"
#include "duckdb/execution/distributed/pipeline_node/extension_write_sink.hpp"
#include "duckdb/main/settings.hpp"

namespace duckdb {
namespace distributed {

namespace {

void ValidateResolvedExtensionWriteInfo(ClientContext &context, const DistributedExtensionWriteInfo &info,
                                        const DistributedExtensionWritePlan &plan) {
	info.Validate();
	plan.Validate();
	if (info.capability.extension_name != plan.extension_name ||
	    info.capability.capability.kind != DistributedExtensionCapabilityKind::WRITE_OPERATOR ||
	    info.capability.capability.name != plan.operator_name || info.worker_bind_data != plan.worker_bind_data) {
		throw InvalidInputException(
		    "Pre-resolved distributed extension write protocol does not match physical operator "
		    "plan '%s.%s'",
		    plan.extension_name, plan.operator_name);
	}
	auto authoritative = ResolveDistributedExtensionWriteInfo(context, plan);
	if (info.capability != authoritative.capability || info.mode != authoritative.mode ||
	    info.fragment_codec != authoritative.fragment_codec ||
	    info.worker_bind_data != authoritative.worker_bind_data) {
		throw InvalidInputException(
		    "Pre-resolved distributed extension write protocol does not match the database-local registration for "
		    "'%s.%s'",
		    plan.extension_name, plan.operator_name);
	}
}

} // namespace

PhysicalPlanToPipelineNodeTranslator::PhysicalPlanToPipelineNodeTranslator(
    PlanConfig plan_config, DuckPhysicalPlanRef plan, ClientContext *client_context,
    optional_ptr<const DistributedExtensionWriteInfo> resolved_extension_write_info)
    : plan_config_(std::move(plan_config)), plan_(std::move(plan)), client_context_(client_context),
      resolved_extension_write_info_(resolved_extension_write_info),
      exchange_mgr_(std::make_shared<FlightExchangeManager>(ResolveFlightExchangeConfigFromEnv(), client_context)) {
}

void PhysicalPlanToPipelineNodeTranslator::CollectUnionOrderRequirements(const PhysicalOperator &op,
                                                                         bool output_order_required) {
	bool child_order_required;
	if (op.type == PhysicalOperatorType::UNION) {
		auto &union_op = op.Cast<PhysicalUnion>();
		child_order_required = output_order_required || !union_op.allow_out_of_order;
		if (child_order_required) {
			ordered_unions_.insert(&union_op);
		}
	} else if (op.IsSink()) {
		// A sink starts a new input pipeline. Its input ordering contract replaces
		// any ordering required by consumers of the sink's materialized output.
		const auto partition_info = op.RequiredPartitionInfo();
		child_order_required = op.SinkOrderDependent() || partition_info.batch_index || !op.ParallelSink();
	} else {
		switch (op.OperatorOrder()) {
		case OrderPreservationType::NO_ORDER:
			child_order_required = false;
			break;
		case OrderPreservationType::INSERTION_ORDER:
			child_order_required = output_order_required;
			break;
		case OrderPreservationType::FIXED_ORDER:
			// Operators such as streaming LIMIT and streaming WINDOW select or
			// compute rows according to their input sequence.
			child_order_required = true;
			break;
		default:
			throw InternalException("Unknown physical operator order-preservation type");
		}
	}

	for (const auto &child : op.children) {
		CollectUnionOrderRequirements(child.get(), child_order_required);
	}
}

bool PhysicalPlanToPipelineNodeTranslator::UnionAllowsOutOfOrder(const PhysicalUnion &op) const {
	return op.allow_out_of_order && ordered_unions_.find(&op) == ordered_unions_.end();
}

DuckDBResult<std::shared_ptr<DistributedPipelineNode>>
PhysicalPlanToPipelineNodeTranslator::physical_plan_to_pipeline_node(
    PlanConfig plan_config, DuckPhysicalPlanRef plan, ClientContext *client_context,
    optional_ptr<const DistributedExtensionWriteInfo> resolved_extension_write_info) {
	PhysicalPlanToPipelineNodeTranslator translator(std::move(plan_config), plan, client_context,
	                                                resolved_extension_write_info);
	if (!plan) {
		return DuckDBResult<std::shared_ptr<DistributedPipelineNode>>::err(
		    DuckDBError::invalid_state_error("physical plan is null"));
	}
	if (!plan->HasRoot()) {
		return DuckDBResult<std::shared_ptr<DistributedPipelineNode>>::err(
		    DuckDBError::invalid_state_error("physical plan has no root"));
	}
	if (resolved_extension_write_info &&
	    (plan->Root().type != PhysicalOperatorType::EXTENSION || !plan->Root().GetExtensionWriteTaskProvider())) {
		return DuckDBResult<std::shared_ptr<DistributedPipelineNode>>::err(DuckDBError::invalid_state_error(
		    "pre-resolved distributed extension write protocol requires an extension write root"));
	}
	try {
		bool root_order_required = true;
		if (client_context) {
			root_order_required = Settings::Get<PreserveInsertionOrderSetting>(*client_context);
		}
		translator.CollectUnionOrderRequirements(plan->Root(), root_order_required);
		translator.VisitOperator(plan->Root());
	} catch (const NotImplementedException &ex) {
		ErrorData error(ex);
		return DuckDBResult<std::shared_ptr<DistributedPipelineNode>>::err(
		    DuckDBError::value_error(std::string("failed to translate physical plan: ") + error.RawMessage()));
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

std::shared_ptr<DistributedPipelineNode>
PhysicalPlanToPipelineNodeTranslator::gen_shuffle_node(std::shared_ptr<RepartitionSpec> repartition_spec,
                                                       SchemaRef schema,
                                                       std::shared_ptr<DistributedPipelineNode> child) {
	if (!repartition_spec) {
		throw InternalException("Cannot build shuffle node without a repartition specification");
	}
	if (!child) {
		throw InternalException("Cannot build shuffle node without a child");
	}
	auto child_clustering = child->config().clustering_spec();
	if (!child_clustering) {
		throw InternalException("Cannot build shuffle node without child clustering metadata");
	}

	size_t upstream_num = child_clustering->num_partitions();
	auto clustering = repartition_spec->to_clustering_spec(upstream_num);
	if (!clustering) {
		throw InternalException("Cannot build shuffle node without output clustering metadata");
	}
	size_t num_partitions = clustering->num_partitions();

	auto plan_cfg_ptr = std::make_shared<PlanConfig>(plan_config_);
	auto repartition_node =
	    RepartitionNode::create(get_next_pipeline_node_id(), plan_cfg_ptr, std::move(repartition_spec), num_partitions,
	                            std::move(schema), std::move(child), exchange_mgr_);
	if (!repartition_node) {
		throw InternalException("Failed to build shuffle node");
	}
	auto node = repartition_node->into_node();
	if (!node) {
		throw InternalException("Failed to wrap shuffle node");
	}
	return node;
}

std::shared_ptr<DistributedPipelineNode>
PhysicalPlanToPipelineNodeTranslator::gen_gather_node(std::shared_ptr<DistributedPipelineNode> input_node) {
	if (input_node->config().clustering_spec()->num_partitions() == 1) {
		return input_node;
	}

	auto spec = RepartitionSpec::create_into_partitions(1);
	return gen_shuffle_node(std::move(spec), input_node->config().schema(), input_node);
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
	case PhysicalOperatorType::UNION: {
		auto &union_op = static_cast<PhysicalUnion &>(op);
		node_impl = TranslateUnion(union_op, children);
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
	case PhysicalOperatorType::CROSS_PRODUCT: {
		auto &cross_product = static_cast<PhysicalCrossProduct &>(op);
		node_impl = TranslateCrossProduct(cross_product, children);
		break;
	}
	case PhysicalOperatorType::ASOF_JOIN: {
		auto &asof_join = static_cast<PhysicalAsOfJoin &>(op);
		node_impl = TranslateAsOfJoin(asof_join, children);
		break;
	}
	case PhysicalOperatorType::PIECEWISE_MERGE_JOIN:
	case PhysicalOperatorType::IE_JOIN: {
		auto &range_join = static_cast<PhysicalRangeJoin &>(op);
		node_impl = TranslateRangeJoin(range_join, children);
		break;
	}
	case PhysicalOperatorType::BLOCKWISE_NL_JOIN: {
		auto &blockwise_join = static_cast<PhysicalBlockwiseNLJoin &>(op);
		node_impl = TranslateBlockwiseNLJoin(blockwise_join, children);
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
	case PhysicalOperatorType::EMPTY_RESULT: {
		auto &empty_result = static_cast<PhysicalEmptyResult &>(op);
		node_stack_.push_back(TranslateEmptyResultSource(empty_result));
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
	case PhysicalOperatorType::DATA_SINK: {
		auto &data_sink = static_cast<PhysicalDataSink &>(op);
		node_impl = TranslateDataSink(data_sink, children);
		break;
	}
	case PhysicalOperatorType::EXTENSION: {
		auto provider = op.GetExtensionWriteTaskProvider();
		if (!provider) {
			throw NotImplementedException(
			    "Distributed pipeline does not support extension operator without an extension write provider: %s",
			    op.GetName());
		}
		if (!client_context_) {
			throw InvalidInputException("Distributed extension write requires a ClientContext");
		}
		DistributedExtensionWriteInfo owned_write_info;
		optional_ptr<const DistributedExtensionWriteInfo> write_info = resolved_extension_write_info_;
		if (write_info) {
			ValidateResolvedExtensionWriteInfo(*client_context_, *write_info, provider->WritePlan());
		} else {
			owned_write_info = ResolveDistributedExtensionWriteInfo(*client_context_, provider->WritePlan());
			write_info = &owned_write_info;
		}
		if (&op != &plan_->Root()) {
			throw InvalidInputException("Distributed extension write %s must be the physical plan root",
			                            write_info->Name());
		}
		if (children.size() != 1 || !children[0]) {
			throw InvalidInputException("Distributed extension write %s requires exactly one child",
			                            write_info->Name());
		}
		if (write_info->mode == DistributedWriteMode::FILE_ARTIFACT) {
			auto copy_finish = std::dynamic_pointer_cast<CopyFinishNode>(children[0]->inner());
			if (!copy_finish) {
				throw InvalidInputException(
				    "Distributed file-artifact write %s requires a COPY child returning written-file statistics",
				    write_info->Name());
			}
			// The extension root is coordinator-only. Workers execute the translated
			// COPY child and return the fixed file-artifact envelope.
			node_stack_.push_back(children[0]);
			return;
		}
		if (write_info->mode != DistributedWriteMode::CALLBACK) {
			throw InvalidInputException("Distributed extension write %s has an unknown mode", write_info->Name());
		}
		auto write_sink =
		    std::make_shared<ExtensionWriteSinkNode>(get_next_pipeline_node_id(), children[0], *write_info);
		node_stack_.push_back(std::make_shared<DistributedPipelineNode>(std::move(write_sink)));
		return;
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
DuckDBResult<std::shared_ptr<DistributedPipelineNode>> physical_plan_to_pipeline_node_wrapper(
    PlanConfig plan_config, DuckPhysicalPlanRef plan, ClientContext *client_context,
    optional_ptr<const DistributedExtensionWriteInfo> resolved_extension_write_info) {
	auto result = PhysicalPlanToPipelineNodeTranslator::physical_plan_to_pipeline_node(
	    std::move(plan_config), plan, client_context, resolved_extension_write_info);
	return result;
}

std::unordered_map<idx_t, std::vector<ScanSplit>>
physical_plan_scan_split_map_wrapper(DuckPhysicalPlanRef plan, DuckDBExecutionConfigRef config,
                                     shared_ptr<DatabaseInstance> db) {
	std::unordered_map<idx_t, std::vector<ScanSplit>> out;
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

	auto allocate_scan_node_id = [&]() {
		if (max_id == NumericLimits<idx_t>::Maximum()) {
			throw InvalidInputException("Cannot allocate a distributed scan node ID: the ID space is exhausted");
		}
		return ++max_id;
	};
	std::function<void(PhysicalOperator &)> collect;
	collect = [&](PhysicalOperator &op) -> void {
		if (op.type == PhysicalOperatorType::TABLE_SCAN) {
			auto &scan = op.Cast<PhysicalTableScan>();
			if (!scan.extra_info.scan_group_id.IsValid()) {
				if (scan.extra_info.scan_node_id.IsValid()) {
					scan.extra_info.scan_group_id = scan.extra_info.scan_node_id;
				} else {
					scan.extra_info.scan_group_id = optional_idx(allocate_scan_node_id());
				}
			}
			if (!scan.extra_info.scan_node_id.IsValid()) {
				scan.extra_info.scan_node_id = optional_idx(allocate_scan_node_id());
			}

			auto splits = MakeTableScanSplits(scan, *exec_cfg, db);
			out.emplace(scan.extra_info.scan_node_id.GetIndex(), std::move(splits));
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
