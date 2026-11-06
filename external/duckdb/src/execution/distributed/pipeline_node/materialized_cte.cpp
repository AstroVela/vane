// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/materialized_cte.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/exchange/flight_exchange_manager.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/distributed/plan/exchange_source_task.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"
#include "duckdb/execution/operator/scan/physical_empty_result.hpp"

namespace duckdb {
namespace distributed {

struct MaterializedCTEOutput {
	std::string exchange_id;
	std::shared_ptr<ExchangeManager> manager;
	bool preserve_order;
	std::vector<std::vector<ExchangeSourceHandle>> partitions;
};

// Every active writer owns per-bucket shuffle metadata. Bound the bucket count independently
// of the number of input tasks; a scan with thousands of files still schedules all of them.
static constexpr size_t MAX_CTE_OUTPUT_PARTITIONS = 64;

// The scope drives the definition's complete task stream through this barrier before it opens its body.
// Both the scope and all its CTE scans reference this same node in the resource graph.
class MaterializedCTEInput : public PipelineNodeImpl, public std::enable_shared_from_this<MaterializedCTEInput> {
public:
	MaterializedCTEInput(PipelineNodeContext context, DistributedPipelineNodeRef definition, bool preserve_order)
	    : context_(std::move(context)), definition_(std::move(definition)),
	      flight_config_(ResolveFlightExchangeConfigFromEnv()), preserve_order_(preserve_order),
	      config_(definition_->config().schema(), definition_->config().execution_config(),
	              ClusteringSpec::unknown_with_num_partitions(
	                  preserve_order_ ? 1 : std::min(MAX_CTE_OUTPUT_PARTITIONS, definition_->num_partitions()))) {
	}

	const PipelineNodeContext &context() const override {
		return context_;
	}
	const PipelineNodeConfig &config() const override {
		return config_;
	}
	std::vector<PipelineNodeRef> children() const override {
		return {definition_};
	}
	std::vector<std::string> multiline_display(bool) const override {
		return {"CTEInput"};
	}
	bool is_materialization_barrier() const override {
		return true;
	}
	std::vector<NodeID> materialized_input_node_ids() const override {
		return {definition_->node_id()};
	}

	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &ctx) override {
		return definition_->produce_tasks(ctx);
	}

	std::shared_ptr<const MaterializedCTEOutput> Publish(PlanExecutionContext &ctx) {
		const auto partition_count = config_.clustering_spec()->num_partitions();
		auto output = std::make_shared<MaterializedCTEOutput>();
		output->exchange_id = context_.query_id() + "_cte_" + std::to_string(node_id());
		output->manager = std::make_shared<FlightExchangeManager>(flight_config_, ctx.client_context());
		output->preserve_order = preserve_order_;
		output->partitions.resize(partition_count);
		auto submitter = ctx.fte_task_submitter_ref();
		if (!submitter) {
			throw InternalException("Materialized CTE requires an FTE task submitter");
		}
		auto exchange =
		    output->manager->CreateExchange({context_.query_id(), output->exchange_id, true}, partition_count);
		auto writes = produce_tasks(ctx).pipeline_instruction(
		    shared_from_this(),
		    [&](DuckPhysicalPlanRef plan) {
			    return AddRemoteExchangeSinkPlan(std::move(plan), nullptr, *exchange, output->manager, false, {},
			                                     preserve_order_);
		    },
		    ctx.client_context());
		if (preserve_order_) {
			writes = writes.map_tasks([order = idx_t(0)](SubmittableTask<WorkerTask> task) mutable {
				return TagOrderedExchangeTask(std::move(task), order++);
			});
		}
		auto completed = writes.materialize(submitter.get());
		if (!completed.success) {
			throw IOException("Materializing CTE %s failed: %s", output->exchange_id, completed.error);
		}
		RecordRemoteExchangeFinishedSinks(*exchange, completed.outputs, "materialized CTE");
		exchange->AllRequiredSinksFinished();
		for (auto &handle : exchange->GetSourceHandles()) {
			if (handle.partition_id >= partition_count) {
				throw InternalException("Materialized CTE returned an out-of-range output partition");
			}
			output->partitions[handle.partition_id].push_back(std::move(handle));
		}
		// Closing the coordinator exchange leaves selected attempts under the query's shuffle-cache ownership.
		exchange->Close();
		if (FteRunnerDebugEnabled()) {
			std::unordered_set<idx_t> writers;
			for (const auto &task : completed.outputs) {
				writers.insert(task.exchange_sink_instance().sink_handle.task_partition_id);
			}
			FteRunnerDebugLog("event=cte_published query_id=" + context_.query_id() + " producer_node_id=" +
			                  std::to_string(node_id()) + " writer_tasks=" + std::to_string(writers.size()) +
			                  " output_partitions=" + std::to_string(partition_count));
		}
		auto ready = submitter->materialization_barrier_completed(context_.query_id(), node_id());
		if (ready.is_err()) {
			throw IOException("Publishing materialized CTE failed: %s", ready.error().what());
		}
		return output;
	}

private:
	PipelineNodeContext context_;
	DistributedPipelineNodeRef definition_;
	FlightExchangeConfig flight_config_;
	bool preserve_order_;
	PipelineNodeConfig config_;
};

namespace {

class MaterializedCTEScan : public PipelineNodeImpl {
public:
	MaterializedCTEScan(PipelineNodeContext context, DistributedPipelineNodeRef input)
	    : context_(std::move(context)), input_(std::move(input)) {
	}

	const PipelineNodeContext &context() const override {
		return context_;
	}
	const PipelineNodeConfig &config() const override {
		return input_->config();
	}
	std::vector<PipelineNodeRef> children() const override {
		return {input_};
	}
	std::vector<std::string> multiline_display(bool) const override {
		return {"CTEScan", "Producer: " + std::to_string(input_->node_id())};
	}
	bool has_single_task_output() const override {
		return config().clustering_spec()->num_partitions() == 1;
	}

	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &ctx) override {
		auto output = ctx.ReadCTE(input_->node_id());
		auto channel = create_channel<SubmittableTask<WorkerTask>>(1);
		auto counter = ctx.task_id_counter();
		ctx.spawn([sender = std::move(channel.first), output, context = context_, input = input_,
		           counter]() mutable -> DuckDBResult<void> {
			// Closing the sender also wakes a blocked consumer when building a descriptor throws.
			auto tx = std::move(sender);
			auto types = GetSchemaTypes(input->config().schema());
			const auto width = output->partitions.size();
			bool has_rows = false;
			for (idx_t partition = 0; partition < width; ++partition) {
				auto &handles = output->partitions[partition];
				if (handles.empty()) {
					continue;
				}
				has_rows = true;
				ExchangeSourceTaskDescriptor descriptor;
				descriptor.partition_indices = {partition};
				descriptor.source_handles = handles;
				descriptor.source_partition_count = width;
				descriptor.source_task_count = width;
				auto plan = MakeRemoteExchangeSourcePlan(types, 0, output->exchange_id, {}, {}, output->manager, {},
				                                         optional_idx(context.node_id()), output->preserve_order);
				auto id = TaskContext::from_node_context(context.query_idx(), context.node_id(), counter.next());
				WorkerTask task(id, std::move(plan), input->config().execution_config(), context.to_hashmap());
				task.mutable_inputs()[context.node_id()] =
				    TaskInput::make_exchange_source_task(descriptor.SerializeToBytes());
				if (tx.send(SubmittableTask<WorkerTask>(std::move(task))).is_err()) {
					return DuckDBResult<void>::ok();
				}
			}
			if (!has_rows) {
				// Downstream aggregates still need one typed, zero-row input.
				auto plan = std::make_shared<PhysicalPlan>(Allocator::DefaultAllocator());
				plan->SetRoot(plan->Make<PhysicalEmptyResult>(std::move(types), 0));
				auto id = TaskContext::from_node_context(context.query_idx(), context.node_id(), counter.next());
				auto metadata = context.to_hashmap();
				metadata["stable_task_partition_id"] = "0";
				WorkerTask task(id, std::move(plan), input->config().execution_config(), std::move(metadata));
				tx.send(SubmittableTask<WorkerTask>(std::move(task)));
			}
			tx.close();
			return DuckDBResult<void>::ok();
		});
		return SubmittableTaskStream<WorkerTask>::from_receiver(std::move(channel.second));
	}

private:
	PipelineNodeContext context_;
	DistributedPipelineNodeRef input_;
};

class MaterializedCTEScope : public PipelineNodeImpl {
public:
	MaterializedCTEScope(PipelineNodeContext context, DistributedPipelineNodeRef input, DistributedPipelineNodeRef body)
	    : context_(std::move(context)), input_(std::move(input)), body_(std::move(body)) {
	}

	const PipelineNodeContext &context() const override {
		return context_;
	}
	const PipelineNodeConfig &config() const override {
		return body_->config();
	}
	std::vector<PipelineNodeRef> children() const override {
		return {body_, input_};
	}
	bool has_single_task_output() const override {
		return body_->has_single_task_output();
	}
	std::vector<std::string> multiline_display(bool) const override {
		return {"CTE"};
	}

	SubmittableTaskStream<WorkerTask> produce_tasks(PlanExecutionContext &ctx) override {
		auto input = std::static_pointer_cast<MaterializedCTEInput>(input_->implementation());
		auto body_context = ctx.WithCTE(input->node_id(), input->Publish(ctx));
		// The scope contributes a dependency and a task identity, but no physical operator.
		return body_->produce_tasks(body_context).map_tasks([scope = context_](SubmittableTask<WorkerTask> submitted) {
			auto *task = submitted.task();
			auto id = task->task_context();
			id.add_node_id(scope.node_id());
			auto metadata = MergeTaskContext(task->context(), scope.to_hashmap());
			WorkerTask tagged(id, task->plan(), task->config(), std::move(metadata), task->name(),
			                  std::move(task->mutable_inputs()));
			return std::move(submitted).with_new_task(std::move(tagged));
		});
	}

private:
	PipelineNodeContext context_;
	DistributedPipelineNodeRef input_;
	DistributedPipelineNodeRef body_;
};

} // namespace

DistributedPipelineNodeRef MakeMaterializedCTEInput(PipelineNodeContext context, DistributedPipelineNodeRef definition,
                                                    bool preserve_order) {
	return std::make_shared<DistributedPipelineNode>(
	    std::make_shared<MaterializedCTEInput>(std::move(context), std::move(definition), preserve_order));
}

DistributedPipelineNodeRef MakeMaterializedCTEScan(PipelineNodeContext context, DistributedPipelineNodeRef input,
                                                   const vector<LogicalType> &scan_types) {
	if (GetSchemaTypes(input->config().schema()) != scan_types) {
		throw InvalidInputException("Distributed CTE scan schema differs from its producer");
	}
	return std::make_shared<DistributedPipelineNode>(
	    std::make_shared<MaterializedCTEScan>(std::move(context), std::move(input)));
}

DistributedPipelineNodeRef MakeMaterializedCTEScope(PipelineNodeContext context, DistributedPipelineNodeRef input,
                                                    DistributedPipelineNodeRef body) {
	return std::make_shared<DistributedPipelineNode>(
	    std::make_shared<MaterializedCTEScope>(std::move(context), std::move(input), std::move(body)));
}

} // namespace distributed
} // namespace duckdb
