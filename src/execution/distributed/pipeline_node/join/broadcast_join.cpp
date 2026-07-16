// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/join/broadcast_join.hpp"

#include <algorithm>
#include <functional>
#include <unordered_set>

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/plan/exchange_source_task.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/distributed/pipeline_node/join/hash_join_metadata.hpp"
#include "duckdb/execution/distributed/pipeline_node/pipeline_node.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/distributed/exchange/exchange_manager.hpp"
#include "duckdb/execution/operator/exchange/physical_remote_exchange_source.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/execution/physical_plan.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"

namespace duckdb {
namespace distributed {

namespace {
std::string MakeBroadcastShuffleStageId(const PipelineNodeContext &context) {
	std::string prefix = context.query_id().empty() ? std::string("query") : context.query_id();
	return prefix + "_broadcast_shuffle_" + std::to_string(context.node_id());
}

vector<std::string> CollectSourceNodes(const std::vector<ExchangeSourceHandle> &handles) {
	std::unordered_set<std::string> node_set;
	for (const auto &handle : handles) {
		if (!handle.node_id.empty()) {
			node_set.insert(handle.node_id);
		}
	}
	vector<std::string> source_nodes(node_set.begin(), node_set.end());
	std::sort(source_nodes.begin(), source_nodes.end());
	return source_nodes;
}
} // namespace

BroadcastJoinNode::BroadcastJoinNode(
    NodeID node_id, const PlanConfig &plan_config, duckdb::vector<JoinCondition> conditions, JoinType join_type,
    duckdb::vector<LogicalType> output_types, duckdb::vector<LogicalType> delim_types,
    duckdb::vector<LogicalType> condition_types, PhysicalHashJoin::JoinProjectionColumns payload_columns,
    PhysicalHashJoin::JoinProjectionColumns lhs_output_columns,
    PhysicalHashJoin::JoinProjectionColumns rhs_output_columns, duckdb::vector<unique_ptr<BaseStatistics>> join_stats,
    unique_ptr<JoinFilterPushdownInfo> filter_pushdown, idx_t estimated_cardinality, bool is_swapped,
    std::shared_ptr<DistributedPipelineNode> broadcaster, std::shared_ptr<DistributedPipelineNode> receiver,
    SchemaRef schema, std::shared_ptr<ExchangeManager> exchange_mgr)
    : context_(plan_config.query_idx, plan_config.query_id, node_id, "BroadcastJoin"),
      broadcaster_(std::move(broadcaster)), receiver_(std::move(receiver)), is_swapped_(is_swapped),
      conditions_(std::move(conditions)), join_type_(join_type), output_types_(std::move(output_types)),
      delim_types_(std::move(delim_types)), condition_types_(std::move(condition_types)),
      payload_columns_(std::move(payload_columns)), lhs_output_columns_(std::move(lhs_output_columns)),
      rhs_output_columns_(std::move(rhs_output_columns)), join_stats_(std::move(join_stats)),
      filter_pushdown_(std::move(filter_pushdown)), estimated_cardinality_(estimated_cardinality),
      exchange_mgr_(std::move(exchange_mgr)) {
	ClusteringSpecRef clustering = ClusteringSpec::unknown_with_num_partitions(1);
	if (receiver_) {
		clustering = receiver_->config().clustering_spec();
	}
	config_ = PipelineNodeConfig(std::move(schema), plan_config.config, std::move(clustering));
}

std::shared_ptr<DistributedPipelineNode> BroadcastJoinNode::into_node() {
	return std::make_shared<DistributedPipelineNode>(shared_from_this());
}

std::vector<PipelineNodeRef> BroadcastJoinNode::children() const {
	std::vector<PipelineNodeRef> result;
	if (broadcaster_)
		result.push_back(broadcaster_->inner());
	if (receiver_)
		result.push_back(receiver_->inner());
	return result;
}

std::vector<std::string> BroadcastJoinNode::multiline_display(bool /*verbose*/) const {
	std::vector<std::string> res;
	res.push_back("Broadcast Join");
	res.push_back("Join type: " + std::to_string(static_cast<int>(join_type_)));
	res.push_back("Conditions: " + std::to_string(conditions_.size()));
	res.push_back("Swapped: " + std::string(is_swapped_ ? "true" : "false"));
	return res;
}

duckdb::vector<JoinCondition> BroadcastJoinNode::CopyConditions(const duckdb::vector<JoinCondition> &conditions) {
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

unique_ptr<JoinFilterPushdownInfo> BroadcastJoinNode::CopyFilterPushdownInfo(const JoinFilterPushdownInfo &info) {
	auto copy = make_uniq<JoinFilterPushdownInfo>();
	copy->join_condition = info.join_condition;
	copy->probe_info.reserve(info.probe_info.size());
	for (const auto &probe : info.probe_info) {
		JoinFilterPushdownFilter new_probe;
		new_probe.dynamic_filters = probe.dynamic_filters;
		new_probe.columns = probe.columns;
		copy->probe_info.push_back(std::move(new_probe));
	}
	copy->min_max_aggregates.reserve(info.min_max_aggregates.size());
	for (const auto &expr : info.min_max_aggregates) {
		if (expr) {
			copy->min_max_aggregates.push_back(expr->Copy());
		} else {
			copy->min_max_aggregates.push_back(nullptr);
		}
	}
	return copy;
}

duckdb::vector<unique_ptr<BaseStatistics>>
BroadcastJoinNode::CopyJoinStats(const duckdb::vector<unique_ptr<BaseStatistics>> &stats) {
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

std::pair<bool, SubmittableTask<WorkerTask>>
BroadcastJoinNode::PollNextWithWait(SubmittableTaskStream<WorkerTask> &stream) {
	// poll_next() is blocking (ChannelStream uses recv()), so no retry needed.
	return stream.poll_next();
}

SubmittableTask<WorkerTask> BroadcastJoinNode::BuildBroadcastHashJoinTask(SubmittableTask<WorkerTask> receiver_task,
                                                                          const DuckPhysicalPlanRef &broadcast_plan,
                                                                          ClientContext *client_context) {
	auto input_plan = receiver_task.task()->plan();
	if (!input_plan || !input_plan->HasRoot() || !broadcast_plan || !broadcast_plan->HasRoot()) {
		throw InvalidInputException("BroadcastJoinNode cannot build join task from task without a physical plan root");
	}

	// Clone the receiver's plan to avoid mutating a shared plan template.
	// Multiple receiver tasks may share the same PhysicalPlan; modifying
	// it in-place for one task would corrupt the plan for others.
	input_plan = ClonePhysicalPlanOrThrow(input_plan, "BuildBroadcastHashJoinTask", client_context);

	auto &input_root = input_plan->Root();
	auto &broadcast_root_ref = broadcast_plan->Root();

	// Determine the broadcast root to attach to the hash join.
	// If the broadcast plan root is a RemoteExchangeSource (exchange-based path),
	// create a copy in input_plan's arena to avoid cross-plan references.
	// Otherwise, clone the entire broadcast plan and use
	// its root directly.
	PhysicalOperator *broadcast_root_ptr = nullptr;

	if (broadcast_root_ref.type == PhysicalOperatorType::EXCHANGE_SOURCE) {
		auto &broadcast_source = broadcast_root_ref.Cast<PhysicalRemoteExchangeSource>();
		auto broadcast_types = broadcast_source.types; // copy types
		broadcast_root_ptr = &input_plan->Make<PhysicalRemoteExchangeSource>(
		    std::move(broadcast_types), broadcast_source.estimated_cardinality, broadcast_source.ExchangeId(),
		    broadcast_source.PartitionIndices(), broadcast_source.SourceHandles(), exchange_mgr_,
		    broadcast_source.SourceNodes(), broadcast_source.RuntimeSourceNodeId());
	} else {
		// In-memory scan path: create a PhysicalColumnDataScan directly inside
		// input_plan's arena.  We must NOT clone the broadcast plan into a
		// separate DuckPhysicalPlanRef because that temporary plan is destroyed
		// at function exit, leaving input_plan's hash join with a dangling
		// reference (use-after-free / corrupted LogicalTypeId).
		//
		// The original broadcast_plan is kept alive by BroadcastJoinNode for
		// the entire pipeline execution, so a non-owning pointer to its
		// ColumnDataCollection is safe.
		auto &broadcast_scan = broadcast_root_ref.Cast<PhysicalColumnDataScan>();
		auto broadcast_types = broadcast_scan.GetTypes();
		broadcast_root_ptr = &input_plan->Make<PhysicalColumnDataScan>(
		    std::move(broadcast_types), broadcast_scan.type, broadcast_scan.estimated_cardinality,
		    optionally_owned_ptr<ColumnDataCollection>(broadcast_scan.collection.get()));
	}
	auto &broadcast_root = *broadcast_root_ptr;

	auto conditions = CopyConditions(conditions_);
	auto &left_child = is_swapped_ ? input_root : broadcast_root;
	auto &right_child = is_swapped_ ? broadcast_root : input_root;
	FixHashJoinConditionTypes(conditions, left_child.GetTypes(), right_child.GetTypes());

	LogicalComparisonJoin dummy_join(join_type_);
	dummy_join.types = output_types_;

	auto delim_types = delim_types_;
	auto &hash_join = input_plan
	                      ->Make<PhysicalHashJoin>(dummy_join, std::move(conditions), join_type_,
	                                               std::move(delim_types), estimated_cardinality_, true)
	                      .Cast<PhysicalHashJoin>();

	hash_join.condition_types = condition_types_;
	hash_join.payload_columns = payload_columns_;
	hash_join.lhs_output_columns = lhs_output_columns_;
	hash_join.rhs_output_columns = rhs_output_columns_;
	hash_join.join_stats = CopyJoinStats(join_stats_);
	if (filter_pushdown_) {
		hash_join.filter_pushdown = CopyFilterPushdownInfo(*filter_pushdown_);
	}
	hash_join.children.push_back(left_child);
	hash_join.children.push_back(right_child);
	input_plan->SetRoot(hash_join);
	RepairHashJoinMetadataAfterChildAttach(hash_join, left_child.GetTypes(), right_child.GetTypes());

	TaskContext task_context = receiver_task.task()->task_context();
	task_context.add_node_id(node_id());
	// No longer need broadcast_plan as a dependent since exchange source is now in input_plan
	auto merged_ctx = MergeTaskContext(receiver_task.task()->context(), context_.to_hashmap());
	WorkerTask new_task(task_context, input_plan, receiver_task.task()->config(), std::move(merged_ctx), "WorkerTask");
	new_task.mutable_inputs() = receiver_task.task()->inputs();
	return std::move(receiver_task).with_new_task(std::move(new_task));
}

SubmittableTaskStream<WorkerTask> BroadcastJoinNode::produce_tasks(PlanExecutionContext &plan_context) {
	if (!broadcaster_ || !receiver_) {
		return SubmittableTaskStream<WorkerTask>::from_receiver(Receiver<SubmittableTask<WorkerTask>>());
	}

	auto broadcaster_input = broadcaster_->produce_tasks(plan_context);
	auto receiver_input = receiver_->produce_tasks(plan_context);

	auto channel_pair_ = create_channel<SubmittableTask<WorkerTask>>(1);
	auto result_tx = std::move(channel_pair_.first);
	auto result_rx = std::move(channel_pair_.second);
	auto result_tx_ptr = std::make_shared<Sender<SubmittableTask<WorkerTask>>>(std::move(result_tx));

	auto fte_task_submitter = plan_context.fte_task_submitter_ref();
	auto self_shared = shared_from_this();
	auto *client_context = plan_context.client_context();

	auto broadcaster_ptr = std::make_shared<SubmittableTaskStream<WorkerTask>>(std::move(broadcaster_input));
	auto receiver_ptr = std::make_shared<SubmittableTaskStream<WorkerTask>>(std::move(receiver_input));

	auto exec_cfg = config_.execution_config();
	if (!exec_cfg || exec_cfg->shuffle_algorithm() != "flight_shuffle") {
		throw NotImplementedException(
		    "BroadcastJoinNode now requires flight_shuffle exchange algorithm. Legacy in-memory path is removed.");
	}

	auto shuffle_stage_id = MakeBroadcastShuffleStageId(context_);
	auto num_partitions = static_cast<idx_t>(1);

	static auto broadcast_exchange_impl = [](std::shared_ptr<PipelineNodeImpl> self_shared_impl,
	                                         std::shared_ptr<SubmittableTaskStream<WorkerTask>> broadcaster_ptr,
	                                         std::shared_ptr<SubmittableTaskStream<WorkerTask>> receiver_ptr,
	                                         std::shared_ptr<Sender<SubmittableTask<WorkerTask>>> result_tx_ptr,
	                                         std::shared_ptr<FteTaskSubmitter> fte_task_submitter,
	                                         std::string shuffle_stage_id, idx_t num_partitions,
	                                         ::duckdb::ClientContext *client_context,
	                                         std::shared_ptr<ExchangeManager> exchange_mgr) -> DuckDBResult<void> {
		auto self_shared = std::static_pointer_cast<BroadcastJoinNode>(self_shared_impl);
		// poll_next() is blocking (ChannelStream uses recv()), so a
		// nullopt return means the stream is truly exhausted.
		auto first_task_opt = broadcaster_ptr->poll_next();
		if (!first_task_opt.first) {
			result_tx_ptr->close();
			return DuckDBResult<void>::ok();
		}

		auto first_task = std::move(first_task_opt.second);
		duckdb::vector<LogicalType> output_types;
		idx_t estimated_cardinality = 0;
		auto plan = first_task.task()->plan();
		if (plan && plan->HasRoot()) {
			output_types = plan->Root().GetTypes();
			estimated_cardinality = plan->Root().estimated_cardinality;
		}
		if (output_types.empty()) {
			result_tx_ptr->close();
			return DuckDBResult<void>::err(DuckDBError::internal_error("broadcast exchange missing output types"));
		}

		// Create Exchange coordinator via ExchangeManager SPI (1 partition = broadcast)
		ExchangeContext exchange_ctx;
		exchange_ctx.query_id = self_shared->context().query_id();
		exchange_ctx.exchange_id = shuffle_stage_id;
		exchange_mgr->SetContext(client_context);
		auto exchange = std::shared_ptr<Exchange>(exchange_mgr->CreateExchange(exchange_ctx, num_partitions).release());

		// Atomic counter for generating unique task partition IDs for sink handles
		auto sink_task_counter = std::make_shared<std::atomic<idx_t>>(0);

		// Build sink plan builder using AddRemoteExchangeSinkPlan (shared with repartition)
		auto plan_builder = [shuffle_stage_id, num_partitions, exchange, exchange_mgr,
		                     sink_task_counter](DuckPhysicalPlanRef plan) {
			auto repartition_spec = RepartitionSpec::create_into_partitions(num_partitions);
			auto task_partition_id = sink_task_counter->fetch_add(1);
			auto sink_handle_obj = exchange->AddSink(task_partition_id);
			auto sink_instance = exchange->InstantiateSink(sink_handle_obj, /*attempt_id=*/0);
			return AddRemoteExchangeSinkPlan(std::move(plan), repartition_spec, num_partitions, shuffle_stage_id,
			                                 sink_instance, exchange_mgr);
		};
		auto node_ref = std::static_pointer_cast<PipelineNodeImpl>(self_shared);
		auto first_with_sink =
		    append_plan_to_existing_task(std::move(first_task), node_ref, plan_builder, client_context);

		struct ExchangeSinkStream {
			std::shared_ptr<SubmittableTaskStream<WorkerTask>> input;
			std::pair<bool, SubmittableTask<WorkerTask>> first;
			PipelineNodeRef node;
			std::function<DuckPhysicalPlanRef(DuckPhysicalPlanRef)> plan_builder;
			::duckdb::ClientContext *client_context = nullptr;

			std::pair<bool, SubmittableTask<WorkerTask>> poll_next() {
				if (first.first) {
					auto out = std::make_pair(true, std::move(first.second));
					first.first = false;
					return out;
				}
				if (!input) {
					return std::make_pair(false, SubmittableTask<WorkerTask>());
				}
				auto next = input->poll_next();
				if (!next.first) {
					return std::make_pair(false, SubmittableTask<WorkerTask>());
				}
				return std::make_pair(
				    true, append_plan_to_existing_task(std::move(next.second), node, plan_builder, client_context));
			}

			bool is_exhausted() const {
				if (first.first) {
					return false;
				}
				if (!input) {
					return true;
				}
				return input->is_exhausted();
			}

			struct Iterator {
				ExchangeSinkStream *parent = nullptr;
				std::pair<bool, SubmittableTask<WorkerTask>> cur;
				Iterator() = default;
				explicit Iterator(ExchangeSinkStream *p) : parent(p) {
					++(*this);
				}
				SubmittableTask<WorkerTask> operator*() {
					return std::move(cur.second);
				}
				Iterator &operator++() {
					cur = parent ? parent->poll_next() : std::make_pair(false, SubmittableTask<WorkerTask>());
					return *this;
				}
				bool equals(const Iterator &other) const {
					return !cur.first && !other.cur.first;
				}
				bool operator==(const Iterator &other) const {
					return equals(other);
				}
				bool operator!=(const Iterator &other) const {
					return !equals(other);
				}
			};

			Iterator begin() {
				return Iterator(this);
			}
			Iterator end() {
				return Iterator();
			}
		};

		ExchangeSinkStream stream;
		stream.input = broadcaster_ptr;
		stream.first = std::make_pair(true, std::move(first_with_sink));
		stream.node = node_ref;
		stream.plan_builder = plan_builder;
		stream.client_context = client_context;
		auto sink_stream = SubmittableTaskStream<WorkerTask>(boxed<SubmittableTask<WorkerTask>>(std::move(stream)));
		auto mat_res = sink_stream.materialize(fte_task_submitter.get());
		if (!mat_res.success) {
			result_tx_ptr->close();
			exchange->Close();
			return DuckDBResult<void>::err(DuckDBError::internal_error(mat_res.error));
		}

		RecordRemoteExchangeFinishedSinks(*exchange, mat_res.outputs, "broadcast");
		exchange->AllRequiredSinksFinished();
		auto broadcast_handles = exchange->GetSourceHandles();
		auto source_nodes = CollectSourceNodes(broadcast_handles);

		while (true) {
			auto receiver_task = BroadcastJoinNode::PollNextWithWait(*receiver_ptr);
			if (!receiver_task.first) {
				result_tx_ptr->close();
				break;
			}
			ExchangeSourceTaskDescriptor source_task;
			source_task.partition_indices = vector<idx_t> {0};
			source_task.source_handles = broadcast_handles;
			source_task.source_partition_count = 1;
			source_task.source_task_count = 1;
			source_task.replicated = true;
			auto broadcast_plan =
			    MakeRemoteExchangeSourcePlan(output_types, estimated_cardinality, shuffle_stage_id, {}, {},
			                                 exchange_mgr, source_nodes, optional_idx(self_shared->node_id()));
			if (!broadcast_plan || !broadcast_plan->HasRoot()) {
				result_tx_ptr->close();
				exchange->Close();
				return DuckDBResult<void>::err(
				    DuckDBError::internal_error("broadcast exchange failed to create exchange source plan"));
			}
			auto joined_task = self_shared->BuildBroadcastHashJoinTask(std::move(receiver_task.second), broadcast_plan,
			                                                           client_context);
			joined_task.task()->mutable_inputs()[static_cast<SourceNodeId>(self_shared->node_id())] =
			    TaskInput::make_exchange_source_task(source_task.SerializeToBytes());
			auto send_res = result_tx_ptr->send(std::move(joined_task));
			// Receiver drop is a normal cancellation signal (e.g. downstream
			// finished processing before upstream exhausted).
			if (send_res.is_err()) {
				result_tx_ptr->close();
				exchange->Close();
				return DuckDBResult<void>::ok();
			}
		}
		exchange->Close();
		return DuckDBResult<void>::ok();
	};

	auto exchange_mgr_copy = exchange_mgr_;
	plan_context.spawn([self_shared, broadcaster_ptr, receiver_ptr, result_tx_ptr, fte_task_submitter, shuffle_stage_id,
	                    num_partitions, client_context, exchange_mgr_copy]() mutable -> DuckDBResult<void> {
		return broadcast_exchange_impl(self_shared, broadcaster_ptr, receiver_ptr, result_tx_ptr, fte_task_submitter,
		                               shuffle_stage_id, num_partitions, client_context, exchange_mgr_copy);
	});
	return SubmittableTaskStream<WorkerTask>::from_receiver(std::move(result_rx));
}

} // namespace distributed
} // namespace duckdb
