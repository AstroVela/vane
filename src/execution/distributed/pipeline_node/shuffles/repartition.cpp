// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/distributed/plan/exchange_source_task.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/operator/exchange/physical_remote_exchange_sink.hpp"
#include "duckdb/execution/operator/exchange/physical_remote_exchange_source.hpp"
#include "duckdb/common/allocator.hpp"
#include "duckdb/execution/distributed/exchange/exchange_manager.hpp"

#include <algorithm>
#include <chrono>
#include <functional>
#include "duckdb/execution/distributed/utils/optional.hpp"
#include <sstream>
#include <unordered_set>

namespace duckdb {
namespace distributed {

namespace {

std::string MakeShuffleStageId(const PipelineNodeContext &context) {
	std::string prefix = context.query_id().empty() ? std::string("query") : context.query_id();
	return prefix + "_shuffle_" + std::to_string(context.node_id());
}

vector<std::string> CollectShuffleSourceNodes(const std::vector<ExchangeSourceHandle> &handles) {
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

std::string ExchangeSourceHandleKey(const ExchangeSourceHandle &handle) {
	std::ostringstream ss;
	ss << handle.partition_id << '|' << handle.attempt_id << '|' << handle.node_id << '|' << handle.flight_port;
	for (const auto &file : handle.files) {
		ss << '|' << file.path << ':' << file.rows << ':' << file.file_size;
	}
	return ss.str();
}

vector<unique_ptr<Expression>> CopyPartitionByExpressions(const std::shared_ptr<RepartitionSpec> &spec) {
	vector<unique_ptr<Expression>> result;
	if (!spec) {
		return result;
	}
	auto exprs = spec->repartition_by();
	result.reserve(exprs.size());
	for (auto &expr_ref : exprs) {
		if (expr_ref) {
			result.push_back(expr_ref->Copy());
		} else {
			result.push_back(nullptr);
		}
	}
	return result;
}

RepartitionSpec::Type ResolveRepartitionType(const std::shared_ptr<RepartitionSpec> &spec) {
	if (!spec) {
		return RepartitionSpec::Type::Random;
	}
	return spec->type();
}

idx_t ResolveExchangeSourceTaskCount(idx_t num_partitions, const DuckDBExecutionConfigRef &exec_cfg) {
	if (num_partitions == 0) {
		return 1;
	}

	// Default to one source task per exchange partition. If runtime capacity
	// information is available, only cap by total worker slots, not by node
	// count, so multi-partition plans can still fan out within a node.
	idx_t target_tasks = num_partitions;
	if (exec_cfg) {
		auto worker_slots = static_cast<idx_t>(exec_cfg->distributed_worker_slots());
		if (worker_slots > 0) {
			target_tasks = std::min(num_partitions, worker_slots);
		}
	}

	const char *env_val = std::getenv("VANE_EXCHANGE_SOURCE_TASKS");
	if (env_val && env_val[0]) {
		try {
			auto parsed = static_cast<idx_t>(std::stoull(env_val));
			if (parsed > 0 && parsed <= num_partitions) {
				target_tasks = parsed;
			}
		} catch (...) {
		}
	}

	return std::max<idx_t>(1, target_tasks);
}

} // namespace

// ─── RemoteExchange (SPI-based) plan builders ────────────────────────

DuckPhysicalPlanRef AddRemoteExchangeSinkPlan(DuckPhysicalPlanRef plan, const std::shared_ptr<RepartitionSpec> &spec,
                                              idx_t num_partitions, const std::string &exchange_id,
                                              const distributed::ExchangeSinkInstanceHandle &sink_handle,
                                              std::shared_ptr<distributed::ExchangeManager> exchange_mgr) {
	if (!plan || !plan->HasRoot()) {
		return plan;
	}
	auto partition_exprs = CopyPartitionByExpressions(spec);
	auto repartition_type = ResolveRepartitionType(spec);
	auto &old_root = plan->Root();
	auto estimated = old_root.estimated_cardinality;
	auto &sink = plan->Make<PhysicalRemoteExchangeSink>(old_root.GetTypes(), estimated, exchange_id, num_partitions,
	                                                    repartition_type, std::move(partition_exprs), sink_handle,
	                                                    std::move(exchange_mgr));
	sink.children.push_back(old_root);
	plan->SetRoot(sink);
	return plan;
}

DuckPhysicalPlanRef AddRemoteRangeExchangeSinkPlan(DuckPhysicalPlanRef plan, const vector<BoundOrderByNode> &orders,
                                                   idx_t num_partitions, const std::string &exchange_id,
                                                   const distributed::ExchangeSinkInstanceHandle &sink_handle,
                                                   std::shared_ptr<distributed::ExchangeManager> exchange_mgr,
                                                   vector<string> boundary_keys) {
	if (!plan || !plan->HasRoot()) {
		return plan;
	}
	vector<unique_ptr<Expression>> partition_exprs;
	vector<string> order_modifiers;
	partition_exprs.reserve(orders.size());
	order_modifiers.reserve(orders.size());
	for (auto &order : orders) {
		partition_exprs.push_back(order.expression->Copy());
		order_modifiers.push_back(order.GetOrderModifier());
	}
	auto &old_root = plan->Root();
	auto estimated = old_root.estimated_cardinality;
	auto &sink = plan->Make<PhysicalRemoteExchangeSink>(old_root.GetTypes(), estimated, exchange_id, num_partitions,
	                                                    RepartitionSpec::Type::Range, std::move(partition_exprs),
	                                                    sink_handle, std::move(exchange_mgr), std::move(boundary_keys),
	                                                    std::move(order_modifiers));
	sink.children.push_back(old_root);
	plan->SetRoot(sink);
	return plan;
}

DuckPhysicalPlanRef MakeRemoteExchangeSourcePlan(const vector<LogicalType> &types, idx_t estimated_cardinality,
                                                 const std::string &exchange_id, vector<idx_t> partition_indices,
                                                 std::vector<distributed::ExchangeSourceHandle> source_handles,
                                                 std::shared_ptr<distributed::ExchangeManager> exchange_mgr,
                                                 const vector<std::string> &source_nodes,
                                                 optional_idx runtime_source_node_id) {
	Allocator &alloc = Allocator::DefaultAllocator();
	auto plan = std::make_shared<duckdb::PhysicalPlan>(alloc);
	auto types_copy = types;
	auto &source = plan->Make<PhysicalRemoteExchangeSource>(
	    std::move(types_copy), estimated_cardinality, exchange_id, std::move(partition_indices),
	    std::move(source_handles), std::move(exchange_mgr), source_nodes, runtime_source_node_id);
	plan->SetRoot(source);
	return plan;
}

// Constructor implementation.
RepartitionNode::RepartitionNode(PipelineNodeConfig config, PipelineNodeContext context,
                                 std::shared_ptr<::duckdb::RepartitionSpec> repartition_spec, size_t num_partitions,
                                 std::shared_ptr<DistributedPipelineNode> child,
                                 std::shared_ptr<ExchangeManager> exchange_mgr)
    : config_(std::move(config)), context_(std::move(context)), repartition_spec_(std::move(repartition_spec)),
      num_partitions_(num_partitions), child_(std::move(child)), exchange_mgr_(std::move(exchange_mgr)) {
}

// Static factory method.
std::shared_ptr<RepartitionNode> RepartitionNode::create(NodeID node_id, const std::shared_ptr<PlanConfig> &plan_config,
                                                         std::shared_ptr<RepartitionSpec> repartition_spec,
                                                         size_t num_partitions, SchemaRef schema,
                                                         std::shared_ptr<DistributedPipelineNode> child,
                                                         std::shared_ptr<ExchangeManager> exchange_mgr) {
	PipelineNodeContext context(plan_config->query_idx, plan_config->query_id, node_id, NODE_NAME);
	auto upstream_num = child ? child->config().clustering_spec()->num_partitions() : 1;
	// Use num_partitions (not the node-count-capped remote_partitions) for the
	// clustering spec so downstream knows the actual output partition count.
	ClusteringSpecRef clustering_spec;
	if (repartition_spec) {
		clustering_spec = repartition_spec->to_clustering_spec(upstream_num);
	} else {
		clustering_spec = ClusteringSpec::unknown_with_num_partitions(num_partitions);
	}
	auto config = PipelineNodeConfig(schema, plan_config->config, std::move(clustering_spec));

	// std::make_shared cannot access the private constructor in some
	// standard library implementations when used in this static method
	// context; construct with `new` inside class scope instead.
	return std::shared_ptr<RepartitionNode>(new RepartitionNode(std::move(config), std::move(context),
	                                                            std::move(repartition_spec), num_partitions,
	                                                            std::move(child), std::move(exchange_mgr)));
}

// Convert to node.
std::shared_ptr<DistributedPipelineNode> RepartitionNode::into_node() {
	return std::make_shared<DistributedPipelineNode>(shared_from_this());
}

// PipelineNodeImpl interface.
const PipelineNodeContext &RepartitionNode::context() const {
	return context_;
}

const PipelineNodeConfig &RepartitionNode::config() const {
	return config_;
}

std::vector<PipelineNodeRef> RepartitionNode::children() const {
	return {child_->inner()};
}

std::vector<std::string> RepartitionNode::multiline_display(bool verbose) const {
	std::vector<std::string> result;
	// For now, avoid depending on the concrete RepartitionSpec implementation
	// here and provide a simplified display string. A full implementation
	// would call into the RepartitionSpec for richer details.
	result.push_back("Repartition");

	return result;
}

// Produce task stream.
SubmittableTaskStream<WorkerTask> RepartitionNode::produce_tasks(PlanExecutionContext &plan_context) {
	auto input_node = child_->produce_tasks(plan_context);
	auto self_shared = shared_from_this();
	auto *client_context = plan_context.client_context();

	auto exec_cfg = config_.execution_config();
	auto shuffle_stage_id = MakeShuffleStageId(context_);
	if (!exec_cfg || exec_cfg->shuffle_algorithm() != "flight_shuffle") {
		throw NotImplementedException("RepartitionNode requires shuffle_algorithm=flight_shuffle. "
		                              "The legacy map_reduce materialize+transpose path has been removed.");
	}

	auto channel_pair_ = create_channel<SubmittableTask<WorkerTask>>(num_partitions_);
	auto result_tx = std::move(channel_pair_.first);
	auto result_rx = std::move(channel_pair_.second);
	auto result_tx_ptr = std::make_shared<Sender<SubmittableTask<WorkerTask>>>(std::move(result_tx));

	auto task_id_counter = std::make_shared<TaskIDCounter>(plan_context.task_id_counter());
	auto fte_task_submitter = plan_context.fte_task_submitter_ref();
	auto input_ptr = std::make_shared<SubmittableTaskStream<WorkerTask>>(std::move(input_node));
	// Use num_partitions_ (not the node-count-capped remote_partitions)
	// so the exchange source reads ALL hash partitions produced by sinks.
	auto num_partitions = num_partitions_;
	auto exchange_mgr = exchange_mgr_;

	plan_context.spawn([self_shared, input_ptr, task_id_counter, result_tx_ptr, shuffle_stage_id, num_partitions,
	                    client_context, exchange_mgr, fte_task_submitter]() mutable -> DuckDBResult<void> {
		auto repartition_profile_start = std::chrono::steady_clock::now();
		// poll_next() is blocking (ChannelStream::recv() waits for data
		// or channel close).  A nullopt return means the child pipeline's
		// channel is closed — no tasks will ever arrive.
		auto first_poll_profile_start = std::chrono::steady_clock::now();
		auto first_task_opt = input_ptr->poll_next();
		auto first_poll_profile_end = std::chrono::steady_clock::now();
		auto first_poll_profile_ms =
		    std::chrono::duration_cast<std::chrono::milliseconds>(first_poll_profile_end - first_poll_profile_start)
		        .count();
		if (!first_task_opt.first) {
			result_tx_ptr->close();
			return DuckDBResult<void>::ok();
		}
		auto first_task = std::move(first_task_opt.second);

		duckdb::vector<LogicalType> output_types;
		idx_t estimated_cardinality = 0;
		{
			auto plan = first_task.task()->plan();
			if (plan && plan->HasRoot()) {
				output_types = plan->Root().GetTypes();
				estimated_cardinality = plan->Root().estimated_cardinality;
			}
		}
		if (output_types.empty()) {
			result_tx_ptr->close();
			return DuckDBResult<void>::err(DuckDBError::internal_error("exchange shuffle missing output types"));
		}

		// Create the Exchange coordinator via ExchangeManager SPI
		distributed::ExchangeContext exchange_ctx;
		exchange_ctx.query_id = self_shared->context().query_id();
		exchange_ctx.exchange_id = shuffle_stage_id;
		exchange_mgr->SetContext(client_context);
		auto exchange = std::shared_ptr<distributed::Exchange>(
		    exchange_mgr->CreateExchange(exchange_ctx, num_partitions).release());

		// Atomic counter for generating unique task partition IDs for sink handles
		auto sink_task_counter = std::make_shared<std::atomic<idx_t>>(0);

		// Build sink plan builder: RemoteExchangeSink
		auto repartition_spec = self_shared->repartition_spec_;
		auto plan_builder = [repartition_spec, num_partitions, shuffle_stage_id, exchange, exchange_mgr,
		                     sink_task_counter](DuckPhysicalPlanRef plan) {
			// Each task gets its own sink handle via the Exchange SPI
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
		stream.input = input_ptr;
		stream.first = std::make_pair(true, std::move(first_with_sink));
		stream.node = node_ref;
		stream.plan_builder = plan_builder;
		stream.client_context = client_context;
		auto sink_stream = SubmittableTaskStream<WorkerTask>(boxed<SubmittableTask<WorkerTask>>(std::move(stream)));
		idx_t target_tasks = ResolveExchangeSourceTaskCount(num_partitions, self_shared->config_.execution_config());
		auto sent_source_handle_keys = std::make_shared<std::unordered_set<std::string>>();
		auto make_source_update_tasks = [self_shared, task_id_counter, result_tx_ptr, exchange_mgr, shuffle_stage_id,
		                                 num_partitions, output_types, estimated_cardinality,
		                                 target_tasks](const std::vector<distributed::ExchangeSourceHandle> &handles,
		                                               const char *reason) -> DuckDBResult<void> {
			if (handles.empty()) {
				return DuckDBResult<void>::ok();
			}
			auto source_nodes = CollectShuffleSourceNodes(handles);
			for (idx_t task_idx = 0; task_idx < target_tasks; task_idx++) {
				idx_t part_start = task_idx * num_partitions / target_tasks;
				idx_t part_end = (task_idx + 1) * num_partitions / target_tasks;

				vector<idx_t> partition_indices;
				std::vector<distributed::ExchangeSourceHandle> task_handles;
				for (idx_t partition_idx = part_start; partition_idx < part_end; partition_idx++) {
					bool has_partition_handle = false;
					for (const auto &handle : handles) {
						if (handle.partition_id != partition_idx) {
							continue;
						}
						task_handles.push_back(handle);
						has_partition_handle = true;
					}
					if (has_partition_handle) {
						partition_indices.push_back(partition_idx);
					}
				}
				if (task_handles.empty()) {
					continue;
				}

				ExchangeSourceTaskDescriptor source_task;
				source_task.partition_indices = std::move(partition_indices);
				source_task.source_handles = std::move(task_handles);
				source_task.source_partition_count = num_partitions;
				source_task.source_task_count = target_tasks;
				auto plan =
				    MakeRemoteExchangeSourcePlan(output_types, estimated_cardinality, shuffle_stage_id, {}, {},
				                                 exchange_mgr, source_nodes, optional_idx(self_shared->node_id()));
				TaskContext task_context = TaskContext::from_node_context(
				    self_shared->context().query_idx(), self_shared->node_id(), task_id_counter->next());
				WorkerTask task(task_context, plan, self_shared->config_.execution_config(),
				                self_shared->context().to_hashmap());
				task.mutable_inputs()[static_cast<SourceNodeId>(self_shared->node_id())] =
				    TaskInput::make_exchange_source_task(source_task.SerializeToBytes());
				auto send_res = result_tx_ptr->send(SubmittableTask<WorkerTask>(std::move(task)));
				if (send_res.is_err()) {
					return DuckDBResult<void>::err(send_res.error());
				}
			}
			return DuckDBResult<void>::ok();
		};
		auto send_new_source_handles = [exchange, sent_source_handle_keys,
		                                make_source_update_tasks](const char *reason) -> DuckDBResult<void> {
			auto selected_source_handles = exchange->GetSourceHandles();
			std::vector<distributed::ExchangeSourceHandle> new_handles;
			for (const auto &handle : selected_source_handles) {
				auto key = ExchangeSourceHandleKey(handle);
				if (!sent_source_handle_keys->insert(key).second) {
					continue;
				}
				new_handles.push_back(handle);
			}
			return make_source_update_tasks(new_handles, reason);
		};
		auto on_sink_output = [exchange,
		                       send_new_source_handles](const MaterializedOutput &output) -> DuckDBResult<void> {
			if (!output.has_exchange_sink_instance()) {
				return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
				    "streaming remote exchange sink output is missing exchange sink instance metadata"));
			}
			std::string node_id;
			auto worker_id = output.worker_id();
			if (worker_id) {
				node_id = *worker_id;
			}
			const auto &sink_instance = output.exchange_sink_instance();
			exchange->SinkFinished(sink_instance.sink_handle, sink_instance.attempt_id, node_id, output.flight_port());
			return send_new_source_handles("sink_output");
		};
		auto materialize_profile_start = std::chrono::steady_clock::now();
		auto mat_res = sink_stream.materialize(fte_task_submitter.get(),
		                                       fte_task_submitter ? on_sink_output : MaterializedOutputCallback {});
		auto materialize_profile_end = std::chrono::steady_clock::now();
		if (!mat_res.success) {
			// Explicitly close the output channel on error. Relying on sender
			// destruction can deadlock if a pending background task keeps the
			// sender object alive while the upstream task_generator is blocked
			// in poll_next().
			result_tx_ptr->close();
			exchange->Close();
			return DuckDBResult<void>::err(DuckDBError::internal_error(mat_res.error));
		}

		auto finish_profile_start = std::chrono::steady_clock::now();
		exchange->AllRequiredSinksFinished();
		auto send_res = send_new_source_handles("final");
		if (send_res.is_err()) {
			result_tx_ptr->close();
			exchange->Close();
			return DuckDBResult<void>::err(send_res.error());
		}
		auto finish_profile_end = std::chrono::steady_clock::now();
		auto repartition_profile_end = std::chrono::steady_clock::now();

		exchange->Close();
		result_tx_ptr->close();
		return DuckDBResult<void>::ok();
	});

	return SubmittableTaskStream<WorkerTask>::from_receiver(std::move(result_rx));
}

} // namespace distributed
} // namespace duckdb
