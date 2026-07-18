// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/distributed/pipeline_node/sort.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/column/column_data_scan_states.hpp"
#include "duckdb/execution/distributed/exchange/exchange.hpp"
#include "duckdb/execution/distributed/exchange/exchange_manager.hpp"
#include "duckdb/execution/distributed/pipeline_node/materialized_coordinator.hpp"
#include "duckdb/execution/distributed/pipeline_node/shuffles/repartition.hpp"
#include "duckdb/execution/distributed/plan/exchange_source_task.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/operator/helper/physical_reservoir_sample.hpp"
#include "duckdb/execution/operator/order/physical_order.hpp"
#include "duckdb/execution/operator/order/physical_top_n.hpp"
#include "duckdb/function/create_sort_key.hpp"
#include "duckdb/parser/parsed_data/sample_options.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/execution/distributed/plan/runner.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_set>

namespace duckdb {
namespace distributed {

static vector<BoundOrderByNode> CopyOrderBys(const vector<BoundOrderByNode> &orders) {
	vector<BoundOrderByNode> copies;
	copies.reserve(orders.size());
	for (const auto &order : orders) {
		copies.push_back(order.Copy());
	}
	return copies;
}

static void FixOrderByReferenceTypes(Expression &expr, const vector<LogicalType> &input_types) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		auto &ref = expr.Cast<BoundReferenceExpression>();
		if (ref.index < input_types.size() && ref.return_type != input_types[ref.index]) {
			ref.return_type = input_types[ref.index];
		}
	}
	ExpressionIterator::EnumerateChildren(expr,
	                                      [&](Expression &child) { FixOrderByReferenceTypes(child, input_types); });
}

static void FixOrderByTypes(vector<BoundOrderByNode> &orders, const vector<LogicalType> &input_types) {
	for (auto &order : orders) {
		if (!order.expression) {
			continue;
		}
		FixOrderByReferenceTypes(*order.expression, input_types);
	}
}

static PipelineNodeConfig BuildOrderByConfig(const PipelineNodeRef &child, const vector<idx_t> &projections,
                                             const vector<LogicalType> &output_types) {
	auto exec_config = child ? child->config().execution_config() : DuckDBExecutionConfigRef();
	auto clustering = child ? child->config().clustering_spec() : ClusteringSpec::unknown_with_num_partitions(1);
	if (output_types.empty()) {
		return PipelineNodeConfig(nullptr, std::move(exec_config), std::move(clustering));
	}

	duckdb::vector<std::string> input_names;
	if (child) {
		input_names = duckdb::distributed::GetSchemaNames(child->config().schema());
	}

	duckdb::vector<std::string> output_names;
	if (!projections.empty() && !input_names.empty()) {
		output_names.reserve(projections.size());
		for (auto idx : projections) {
			if (idx < input_names.size() && !input_names[idx].empty()) {
				output_names.push_back(input_names[idx]);
			} else {
				output_names.push_back("c" + std::to_string(output_names.size()));
			}
		}
	} else if (projections.empty() && !input_names.empty() && input_names.size() == output_types.size()) {
		output_names = input_names;
	}

	SchemaRef schema;
	if (!output_names.empty() && output_names.size() == output_types.size()) {
		schema = duckdb::distributed::MakeSchemaRef(output_types, output_names);
	} else {
		schema = duckdb::distributed::MakeSchemaRef(output_types);
	}
	return PipelineNodeConfig(std::move(schema), std::move(exec_config), std::move(clustering));
}

static DuckPhysicalPlanRef AddPhysicalOrderPlan(DuckPhysicalPlanRef input_plan,
                                                const vector<BoundOrderByNode> &order_specs,
                                                const vector<idx_t> &projection_specs, bool is_index_sort) {
	auto orders = CopyOrderBys(order_specs);
	auto input_types = input_plan->Root().GetTypes();
	FixOrderByTypes(orders, input_types);
	auto projections = projection_specs;
	if (projections.empty()) {
		projections.reserve(input_types.size());
		for (idx_t i = 0; i < input_types.size(); ++i) {
			projections.push_back(i);
		}
		auto types = input_types;
		idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		auto &old_root = input_plan->Root();
		auto &order_op = input_plan->Make<duckdb::PhysicalOrder>(
		    std::move(types), std::move(orders), std::move(projections), estimated_cardinality, is_index_sort);
		order_op.children.push_back(old_root);
		input_plan->SetRoot(order_op);
		return input_plan;
	}

	duckdb::vector<idx_t> filtered;
	filtered.reserve(projections.size());
	duckdb::vector<LogicalType> types;
	types.reserve(projections.size());
	for (auto idx : projections) {
		if (idx < input_types.size()) {
			filtered.push_back(idx);
			types.push_back(input_types[idx]);
		}
	}
	if (filtered.empty()) {
		filtered.reserve(input_types.size());
		for (idx_t i = 0; i < input_types.size(); ++i) {
			filtered.push_back(i);
		}
		types = input_types;
	}
	idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

	auto &old_root = input_plan->Root();
	auto &order_op = input_plan->Make<duckdb::PhysicalOrder>(std::move(types), std::move(orders), std::move(filtered),
	                                                         estimated_cardinality, is_index_sort);
	order_op.children.push_back(old_root);
	input_plan->SetRoot(order_op);
	return input_plan;
}

static DuckPhysicalPlanRef AddReservoirSamplePlan(DuckPhysicalPlanRef input_plan, idx_t sample_rows) {
	if (sample_rows == 0) {
		return input_plan;
	}
	auto options = make_uniq<SampleOptions>();
	auto clamped = std::min<idx_t>(sample_rows, NumericLimits<int64_t>::Maximum());
	options->sample_size = Value::BIGINT(static_cast<int64_t>(clamped));
	options->is_percentage = false;
	options->method = SampleMethod::RESERVOIR_SAMPLE;
	options->SetSeed(1);
	options->repeatable = true;

	auto types = input_plan->Root().GetTypes();
	idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;
	auto &old_root = input_plan->Root();
	auto &sample_op =
	    input_plan->Make<duckdb::PhysicalReservoirSample>(std::move(types), std::move(options), estimated_cardinality);
	sample_op.children.push_back(old_root);
	input_plan->SetRoot(sample_op);
	return input_plan;
}

static std::string MakeOrderByExchangeId(const PipelineNodeContext &context, const char *suffix) {
	std::string prefix = context.query_id().empty() ? std::string("query") : context.query_id();
	return prefix + "_orderby_" + std::to_string(context.node_id()) + "_" + suffix;
}

static std::string MakeOrderByInternalQueryId(const PipelineNodeContext &context, const char *suffix) {
	std::string prefix = context.query_id().empty() ? std::string("query") : context.query_id();
	return prefix + "_orderby_" + std::to_string(context.node_id()) + "_" + suffix;
}

static std::shared_ptr<Exchange> CreateOrderByExchange(const PipelineNodeContext &context,
                                                       const std::string &exchange_id, idx_t num_partitions,
                                                       const std::shared_ptr<ExchangeManager> &exchange_mgr,
                                                       ClientContext *client_context) {
	ExchangeContext exchange_ctx;
	exchange_ctx.query_id = context.query_id();
	exchange_ctx.exchange_id = exchange_id;
	exchange_mgr->SetContext(client_context);
	auto exchange_unique = exchange_mgr->CreateExchange(exchange_ctx, num_partitions);
	if (!exchange_unique) {
		throw InvalidInputException("ORDER BY failed to create exchange " + exchange_id);
	}
	return std::shared_ptr<Exchange>(exchange_unique.release());
}

static idx_t ResolveOrderBySourceTaskCount(idx_t num_partitions, const DuckDBExecutionConfigRef &exec_cfg) {
	if (num_partitions == 0) {
		return 1;
	}
	idx_t target_tasks = num_partitions;
	if (exec_cfg) {
		auto worker_slots = static_cast<idx_t>(exec_cfg->distributed_worker_slots());
		if (worker_slots > 0) {
			target_tasks = std::min(num_partitions, worker_slots);
		}
	}
	const char *env_val = std::getenv("VANE_ORDER_BY_SOURCE_TASKS");
	if (!env_val || !env_val[0]) {
		env_val = std::getenv("VANE_EXCHANGE_SOURCE_TASKS");
	}
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

static idx_t ResolveOrderBySampleRows(idx_t num_partitions) {
	idx_t sample_rows = std::max<idx_t>(8192, num_partitions * 256);
	const char *env_val = std::getenv("VANE_ORDER_BY_RANGE_SAMPLE_ROWS");
	if (env_val && env_val[0]) {
		try {
			auto parsed = static_cast<idx_t>(std::stoull(env_val));
			if (parsed > 0) {
				sample_rows = parsed;
			}
		} catch (...) {
		}
	}
	return sample_rows;
}

static vector<string> CollectOrderBySourceNodes(const std::vector<ExchangeSourceHandle> &handles) {
	std::unordered_set<std::string> node_set;
	for (const auto &handle : handles) {
		if (!handle.node_id.empty()) {
			node_set.insert(handle.node_id);
		}
	}
	vector<string> source_nodes(node_set.begin(), node_set.end());
	std::sort(source_nodes.begin(), source_nodes.end());
	return source_nodes;
}

static idx_t EstimateRowsFromHandles(const std::vector<ExchangeSourceHandle> &handles, idx_t fallback) {
	idx_t total = 0;
	for (const auto &handle : handles) {
		for (const auto &file : handle.files) {
			total += file.rows;
		}
	}
	return total == 0 ? fallback : total;
}

static bool OrderByDebugEnabled() {
	const char *value = std::getenv("DUCKDB_DISTRIBUTED_DEBUG");
	return value && value[0] != '\0' && value[0] != '0';
}

static void DebugOrderByHandles(const char *stage, const std::vector<ExchangeSourceHandle> &handles) {
	if (!OrderByDebugEnabled()) {
		return;
	}
	idx_t rows = 0;
	size_t bytes = 0;
	std::unordered_map<idx_t, idx_t> rows_by_partition;
	for (const auto &handle : handles) {
		idx_t handle_rows = 0;
		for (const auto &file : handle.files) {
			handle_rows += file.rows;
			bytes += file.file_size;
		}
		rows += handle_rows;
		rows_by_partition[handle.partition_id] += handle_rows;
	}
	std::cerr << "[distributed-orderby] " << stage << " handles=" << handles.size() << " rows=" << rows
	          << " bytes=" << bytes << " partitions=" << rows_by_partition.size() << std::endl;
	for (const auto &entry : rows_by_partition) {
		std::cerr << "[distributed-orderby] " << stage << " partition=" << entry.first << " rows=" << entry.second
		          << std::endl;
	}
}

static void DebugOrderByOutputs(const char *stage, const std::vector<MaterializedOutput> &outputs) {
	if (!OrderByDebugEnabled()) {
		return;
	}
	idx_t rows = 0;
	for (const auto &output : outputs) {
		for (const auto &fragment : output.fragments()) {
			if (!fragment) {
				continue;
			}
			auto collection = fragment->to_column_data();
			if (collection) {
				rows += collection->Count();
			}
		}
	}
	std::cerr << "[distributed-orderby] " << stage << " outputs=" << outputs.size() << " rows=" << rows << std::endl;
}

static DuckDBResult<void> RecordOrderByExchangeSinkOutput(Exchange &exchange, const MaterializedOutput &output,
                                                          const char *context) {
	if (!output.has_exchange_sink_instance()) {
		return DuckDBResult<void>::err(
		    DuckDBError::invalid_state_error(std::string(context ? context : "orderby") +
		                                     " exchange sink output is missing exchange sink instance metadata"));
	}
	std::string node_id;
	auto worker_id = output.worker_id();
	if (worker_id) {
		node_id = *worker_id;
	}
	const auto &sink_instance = output.exchange_sink_instance();
	exchange.SinkFinished(sink_instance.sink_handle, sink_instance.attempt_id, node_id, output.flight_port());
	return DuckDBResult<void>::ok();
}

static std::vector<ExchangeSourceHandle>
SelectHandlesForPartitionRange(const std::vector<ExchangeSourceHandle> &handles, idx_t part_start, idx_t part_end) {
	std::vector<ExchangeSourceHandle> selected;
	for (idx_t partition_idx = part_start; partition_idx < part_end; partition_idx++) {
		for (const auto &handle : handles) {
			if (handle.partition_id == partition_idx) {
				selected.push_back(handle);
			}
		}
	}
	return selected;
}

static vector<idx_t> MakePartitionRange(idx_t part_start, idx_t part_end) {
	vector<idx_t> partition_indices;
	partition_indices.reserve(part_end - part_start);
	for (idx_t partition_idx = part_start; partition_idx < part_end; partition_idx++) {
		partition_indices.push_back(partition_idx);
	}
	return partition_indices;
}

static bool SortKeyStringLess(const std::string &left, const std::string &right) {
	const auto min_size = std::min(left.size(), right.size());
	auto cmp = std::memcmp(left.data(), right.data(), min_size);
	if (cmp != 0) {
		return cmp < 0;
	}
	return left.size() < right.size();
}

static vector<string> CollectSortKeyBoundaries(const std::vector<MaterializedOutput> &outputs,
                                               const vector<BoundOrderByNode> &order_specs,
                                               const vector<LogicalType> &input_types, ClientContext &client_context,
                                               idx_t num_partitions) {
	if (order_specs.empty()) {
		return {};
	}
	vector<BoundOrderByNode> orders = CopyOrderBys(order_specs);
	FixOrderByTypes(orders, input_types);

	ExpressionExecutor executor(client_context);
	vector<LogicalType> key_types;
	key_types.reserve(orders.size());
	vector<OrderModifiers> modifiers;
	modifiers.reserve(orders.size());
	for (auto &order : orders) {
		executor.AddExpression(*order.expression);
		key_types.push_back(order.expression->return_type);
		modifiers.push_back(OrderModifiers::Parse(order.GetOrderModifier()));
	}

	vector<string> sort_keys;
	DataChunk key_chunk;
	key_chunk.InitializeEmpty(key_types);
	Vector sort_key(LogicalType::BLOB);

	for (const auto &output : outputs) {
		for (const auto &fragment : output.fragments()) {
			if (!fragment) {
				continue;
			}
			auto collection = fragment->to_column_data();
			if (!collection || collection->Count() == 0) {
				continue;
			}
			ColumnDataScanState scan_state;
			DataChunk scan_chunk;
			collection->InitializeScan(scan_state);
			collection->InitializeScanChunk(scan_chunk);
			while (collection->Scan(scan_state, scan_chunk)) {
				if (scan_chunk.size() == 0) {
					continue;
				}
				key_chunk.Reset();
				executor.Execute(scan_chunk, key_chunk);
				CreateSortKeyHelpers::CreateSortKey(key_chunk, modifiers, sort_key);
				sort_key.Flatten(scan_chunk.size());
				auto sort_key_data = FlatVector::GetData<string_t>(sort_key);
				for (idx_t row_idx = 0; row_idx < scan_chunk.size(); row_idx++) {
					sort_keys.emplace_back(sort_key_data[row_idx].GetData(), sort_key_data[row_idx].GetSize());
				}
				scan_chunk.Reset();
			}
		}
	}

	if (sort_keys.empty() || num_partitions <= 1) {
		return {};
	}
	std::sort(sort_keys.begin(), sort_keys.end(), SortKeyStringLess);

	vector<string> boundaries;
	boundaries.reserve(num_partitions - 1);
	for (idx_t partition_idx = 1; partition_idx < num_partitions; partition_idx++) {
		auto key_idx = (partition_idx * sort_keys.size()) / num_partitions;
		if (key_idx >= sort_keys.size()) {
			key_idx = sort_keys.size() - 1;
		}
		boundaries.push_back(sort_keys[key_idx]);
	}
	return boundaries;
}

static SubmittableTaskStream<WorkerTask> MakeTaskStreamFromVector(std::vector<WorkerTask> tasks) {
	struct VectorTaskStream {
		std::vector<WorkerTask> tasks;
		idx_t index = 0;

		std::pair<bool, SubmittableTask<WorkerTask>> poll_next() {
			if (index >= tasks.size()) {
				return std::make_pair(false, SubmittableTask<WorkerTask>());
			}
			return std::make_pair(true, SubmittableTask<WorkerTask>(std::move(tasks[index++])));
		}

		std::pair<bool, SubmittableTask<WorkerTask>> try_poll_next() {
			return poll_next();
		}

		bool is_exhausted() const {
			return index >= tasks.size();
		}

		struct Iterator {
			VectorTaskStream *parent = nullptr;
			std::pair<bool, SubmittableTask<WorkerTask>> cur;
			Iterator() = default;
			explicit Iterator(VectorTaskStream *p) : parent(p) {
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

	VectorTaskStream stream;
	stream.tasks = std::move(tasks);
	auto boxed_stream = boxed<SubmittableTask<WorkerTask>>(std::move(stream));
	return SubmittableTaskStream<WorkerTask>(std::move(boxed_stream));
}

static std::vector<WorkerTask>
BuildExchangeSourceTasks(const PipelineNodeContext &context, const PipelineNodeConfig &config,
                         TaskIDCounter &task_id_counter, const std::shared_ptr<ExchangeManager> &exchange_mgr,
                         const std::string &exchange_id, const std::vector<ExchangeSourceHandle> &source_handles,
                         idx_t source_partition_count, idx_t source_task_count, const vector<LogicalType> &types,
                         idx_t estimated_cardinality,
                         const std::function<DuckPhysicalPlanRef(DuckPhysicalPlanRef)> &plan_builder,
                         const std::string &task_name, const std::string &query_id_override = std::string()) {
	std::vector<WorkerTask> tasks;
	if (source_handles.empty()) {
		return tasks;
	}
	auto source_nodes = CollectOrderBySourceNodes(source_handles);
	for (idx_t task_idx = 0; task_idx < source_task_count; task_idx++) {
		idx_t part_start = task_idx * source_partition_count / source_task_count;
		idx_t part_end = (task_idx + 1) * source_partition_count / source_task_count;
		auto task_handles = SelectHandlesForPartitionRange(source_handles, part_start, part_end);
		if (task_handles.empty()) {
			continue;
		}
		auto partition_indices = MakePartitionRange(part_start, part_end);
		auto plan =
		    MakeRemoteExchangeSourcePlan(types, estimated_cardinality, exchange_id, std::move(partition_indices),
		                                 std::move(task_handles), exchange_mgr, source_nodes, optional_idx());
		if (plan_builder) {
			plan = plan_builder(std::move(plan));
		}
		TaskContext task_context =
		    TaskContext::from_node_context(context.query_idx(), context.node_id(), task_id_counter.next());
		auto task_context_map = context.to_hashmap();
		auto effective_query_id = context.query_id().empty() ? std::string("query") : context.query_id();
		if (!query_id_override.empty()) {
			effective_query_id = query_id_override;
			task_context_map["query_id"] = effective_query_id;
		}
		task_context_map["fragment_id"] = effective_query_id + ":orderby:" + std::to_string(context.node_id()) + ":" +
		                                  task_name + ":" + std::to_string(task_idx);
		task_context_map["preserve_plan_exchange_sink_instance"] = "1";
		WorkerTask task(task_context, plan, config.execution_config(), std::move(task_context_map), task_name);
		tasks.push_back(std::move(task));
	}
	return tasks;
}

static SubmittableTask<WorkerTask>
AppendPlanToExistingTaskWithQueryId(SubmittableTask<WorkerTask> submittable_task, const PipelineNodeRef &node,
                                    const std::function<DuckPhysicalPlanRef(DuckPhysicalPlanRef)> &plan_builder,
                                    ClientContext *client_context, const std::string &query_id) {
	auto old_task = submittable_task.task();
	auto working_plan = ClonePhysicalPlanOrThrow(old_task->plan(), "append_orderby_internal", client_context);
	auto new_plan = plan_builder(std::move(working_plan));
	TaskContext task_context = old_task->task_context();
	task_context.add_node_id(node->node_id());
	auto merged_context = MergeTaskContext(old_task->context(), node->context().to_hashmap());
	if (!query_id.empty()) {
		merged_context["query_id"] = query_id;
	}
	WorkerTask new_task(task_context, new_plan, old_task->config(), std::move(merged_context));
	new_task.mutable_inputs() = old_task->inputs();
	return SubmittableTask<WorkerTask>(std::move(new_task));
}

OrderByNode::OrderByNode(NodeID node_id, PipelineNodeRef child, vector<BoundOrderByNode> orders,
                         vector<idx_t> projections, vector<LogicalType> output_types, bool is_index_sort,
                         std::shared_ptr<ExchangeManager> exchange_mgr)
    : ctx_(InheritPipelineNodeContext(child, node_id, "OrderBy")),
      config_(BuildOrderByConfig(child, projections, output_types)), child_(std::move(child)),
      orders_(std::move(orders)), projections_(std::move(projections)), output_types_(std::move(output_types)),
      is_index_sort_(is_index_sort), exchange_mgr_(std::move(exchange_mgr)) {
}

std::vector<PipelineNodeRef> OrderByNode::children() const {
	return {child_};
}

SubmittableTaskStream<WorkerTask> OrderByNode::produce_tasks(PlanExecutionContext &plan_context) {
	const auto *orders_ptr = &orders_;
	const auto *projections_ptr = &projections_;
	const bool is_index_sort = is_index_sort_;
	auto order_plan_builder = [orders_ptr, projections_ptr,
	                           is_index_sort](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		return AddPhysicalOrderPlan(std::move(input_plan), *orders_ptr, *projections_ptr, is_index_sort);
	};

	if (!ChildHasMultiplePartitions(child_)) {
		auto input_stream = child_->produce_tasks(plan_context);
		return input_stream.pipeline_instruction(shared_from_this(), order_plan_builder, plan_context.client_context());
	}

	if (!exchange_mgr_) {
		throw InvalidInputException("Distributed ORDER BY requires an ExchangeManager");
	}
	auto exec_cfg = config_.execution_config();
	if (!exec_cfg || exec_cfg->shuffle_algorithm() != "flight_shuffle") {
		throw NotImplementedException("Distributed ORDER BY requires shuffle_algorithm=flight_shuffle");
	}

	auto input_stream = child_->produce_tasks(plan_context);
	auto channel_pair = create_channel<SubmittableTask<WorkerTask>>(1);
	auto result_tx = std::move(channel_pair.first);
	auto result_rx = std::move(channel_pair.second);
	auto result_tx_ptr = std::make_shared<Sender<SubmittableTask<WorkerTask>>>(std::move(result_tx));
	auto input_stream_ptr = std::make_shared<SubmittableTaskStream<WorkerTask>>(std::move(input_stream));
	auto task_id_counter = std::make_shared<TaskIDCounter>(plan_context.task_id_counter());
	auto fte_task_submitter = plan_context.fte_task_submitter_ref();
	auto client_context = plan_context.client_context();
	auto self_shared = shared_from_this();
	auto exchange_mgr = exchange_mgr_;

	plan_context.spawn([self_shared, input_stream_ptr, result_tx_ptr, task_id_counter, fte_task_submitter,
	                    client_context, exchange_mgr, order_plan_builder]() mutable -> DuckDBResult<void> {
		try {
			if (!fte_task_submitter) {
				result_tx_ptr->close();
				return DuckDBResult<void>::err(
				    DuckDBError::invalid_state_error("Distributed ORDER BY requires FTE task submitter"));
			}

			auto first_task_opt = input_stream_ptr->poll_next();
			if (!first_task_opt.first) {
				result_tx_ptr->close();
				return DuckDBResult<void>::ok();
			}
			auto first_task = std::move(first_task_opt.second);
			auto first_plan = first_task.task()->plan();
			if (!first_plan || !first_plan->HasRoot()) {
				result_tx_ptr->close();
				return DuckDBResult<void>::err(
				    DuckDBError::internal_error("Distributed ORDER BY input task has no physical plan"));
			}

			auto input_types = first_plan->Root().GetTypes();
			auto input_estimated_cardinality = first_plan->Root().estimated_cardinality;
			auto num_partitions = self_shared->child_->config().clustering_spec()->num_partitions();
			if (num_partitions == 0) {
				num_partitions = 1;
			}

			const auto staging_exchange_id = MakeOrderByExchangeId(self_shared->context(), "stage");
			const auto stage_query_id = MakeOrderByInternalQueryId(self_shared->context(), "stage_query");
			auto staging_exchange = CreateOrderByExchange(self_shared->context(), staging_exchange_id, num_partitions,
			                                              exchange_mgr, client_context);
			auto staging_sink_counter = std::make_shared<std::atomic<idx_t>>(0);
			auto stage_plan_builder = [staging_exchange, exchange_mgr, staging_exchange_id, staging_sink_counter,
			                           num_partitions](DuckPhysicalPlanRef plan) -> DuckPhysicalPlanRef {
				auto task_partition_id = staging_sink_counter->fetch_add(1);
				auto sink_handle = staging_exchange->AddSink(task_partition_id);
				auto sink_instance = staging_exchange->InstantiateSink(sink_handle, /*attempt_id=*/0);
				return AddRemoteExchangeSinkPlan(std::move(plan), nullptr, num_partitions, staging_exchange_id,
				                                 sink_instance, exchange_mgr);
			};

			auto node_ref = std::static_pointer_cast<PipelineNodeImpl>(self_shared);
			auto first_with_stage_sink = AppendPlanToExistingTaskWithQueryId(
			    std::move(first_task), node_ref, stage_plan_builder, client_context, stage_query_id);

			struct FirstThenMappedStream {
				std::shared_ptr<SubmittableTaskStream<WorkerTask>> input;
				std::pair<bool, SubmittableTask<WorkerTask>> first;
				PipelineNodeRef node;
				std::function<DuckPhysicalPlanRef(DuckPhysicalPlanRef)> plan_builder;
				ClientContext *client_context = nullptr;
				std::string query_id;

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
					return std::make_pair(true,
					                      AppendPlanToExistingTaskWithQueryId(std::move(next.second), node,
					                                                          plan_builder, client_context, query_id));
				}

				std::pair<bool, SubmittableTask<WorkerTask>> try_poll_next() {
					if (first.first) {
						auto out = std::make_pair(true, std::move(first.second));
						first.first = false;
						return out;
					}
					if (!input) {
						return std::make_pair(false, SubmittableTask<WorkerTask>());
					}
					auto next = input->try_poll_next();
					if (!next.first) {
						return std::make_pair(false, SubmittableTask<WorkerTask>());
					}
					return std::make_pair(true,
					                      AppendPlanToExistingTaskWithQueryId(std::move(next.second), node,
					                                                          plan_builder, client_context, query_id));
				}

				bool is_exhausted() const {
					if (first.first) {
						return false;
					}
					return !input || input->is_exhausted();
				}

				struct Iterator {
					FirstThenMappedStream *parent = nullptr;
					std::pair<bool, SubmittableTask<WorkerTask>> cur;
					Iterator() = default;
					explicit Iterator(FirstThenMappedStream *p) : parent(p) {
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

			FirstThenMappedStream stage_stream_impl;
			stage_stream_impl.input = input_stream_ptr;
			stage_stream_impl.first = std::make_pair(true, std::move(first_with_stage_sink));
			stage_stream_impl.node = node_ref;
			stage_stream_impl.plan_builder = stage_plan_builder;
			stage_stream_impl.client_context = client_context;
			stage_stream_impl.query_id = stage_query_id;
			auto stage_stream =
			    SubmittableTaskStream<WorkerTask>(boxed<SubmittableTask<WorkerTask>>(std::move(stage_stream_impl)));
			auto stage_on_output = [staging_exchange](const MaterializedOutput &output) -> DuckDBResult<void> {
				return RecordOrderByExchangeSinkOutput(*staging_exchange, output, "orderby-stage");
			};
			auto stage_mat_res = stage_stream.materialize(fte_task_submitter.get(), stage_on_output);
			if (!stage_mat_res.success) {
				result_tx_ptr->close();
				staging_exchange->Close();
				return DuckDBResult<void>::err(DuckDBError::internal_error(stage_mat_res.error));
			}
			staging_exchange->AllRequiredSinksFinished();
			auto staging_handles = staging_exchange->GetSourceHandles();
			DebugOrderByOutputs("stage-output", stage_mat_res.outputs);
			DebugOrderByHandles("stage-handles", staging_handles);
			auto staging_estimated_cardinality = EstimateRowsFromHandles(staging_handles, input_estimated_cardinality);

			auto source_task_count =
			    ResolveOrderBySourceTaskCount(num_partitions, self_shared->config().execution_config());
			auto sample_rows = ResolveOrderBySampleRows(num_partitions);
			const auto sample_query_id = MakeOrderByInternalQueryId(self_shared->context(), "sample_query");
			auto sample_plan_builder = [sample_rows](DuckPhysicalPlanRef plan) -> DuckPhysicalPlanRef {
				return AddReservoirSamplePlan(std::move(plan), sample_rows);
			};
			auto sample_tasks = BuildExchangeSourceTasks(
			    self_shared->context(), self_shared->config(), *task_id_counter, exchange_mgr, staging_exchange_id,
			    staging_handles, num_partitions, source_task_count, input_types, staging_estimated_cardinality,
			    sample_plan_builder, "OrderByRangeSample", sample_query_id);
			auto sample_stream = MakeTaskStreamFromVector(std::move(sample_tasks));
			auto sample_mat_res = sample_stream.materialize(fte_task_submitter.get());
			if (!sample_mat_res.success) {
				result_tx_ptr->close();
				staging_exchange->Close();
				return DuckDBResult<void>::err(DuckDBError::internal_error(sample_mat_res.error));
			}
			DebugOrderByOutputs("sample-output", sample_mat_res.outputs);
			auto boundaries = CollectSortKeyBoundaries(sample_mat_res.outputs, self_shared->orders_, input_types,
			                                           *client_context, num_partitions);
			if (OrderByDebugEnabled()) {
				std::cerr << "[distributed-orderby] boundaries=" << boundaries.size() << std::endl;
			}

			const auto range_exchange_id = MakeOrderByExchangeId(self_shared->context(), "range");
			auto range_exchange = CreateOrderByExchange(self_shared->context(), range_exchange_id, num_partitions,
			                                            exchange_mgr, client_context);
			const auto range_query_id = MakeOrderByInternalQueryId(self_shared->context(), "range_query");
			auto range_sink_counter = std::make_shared<std::atomic<idx_t>>(0);
			auto range_plan_builder = [self_shared, range_exchange, exchange_mgr, range_exchange_id, range_sink_counter,
			                           num_partitions, boundaries](DuckPhysicalPlanRef plan) -> DuckPhysicalPlanRef {
				auto task_partition_id = range_sink_counter->fetch_add(1);
				auto sink_handle = range_exchange->AddSink(task_partition_id);
				auto sink_instance = range_exchange->InstantiateSink(sink_handle, /*attempt_id=*/0);
				return AddRemoteRangeExchangeSinkPlan(std::move(plan), self_shared->orders_, num_partitions,
				                                      range_exchange_id, sink_instance, exchange_mgr, boundaries);
			};

			auto range_tasks = BuildExchangeSourceTasks(
			    self_shared->context(), self_shared->config(), *task_id_counter, exchange_mgr, staging_exchange_id,
			    staging_handles, num_partitions, source_task_count, input_types, staging_estimated_cardinality,
			    range_plan_builder, "OrderByRangeShuffle", range_query_id);
			auto range_stream = MakeTaskStreamFromVector(std::move(range_tasks));
			auto range_on_output = [range_exchange](const MaterializedOutput &output) -> DuckDBResult<void> {
				return RecordOrderByExchangeSinkOutput(*range_exchange, output, "orderby-range");
			};
			auto range_mat_res = range_stream.materialize(fte_task_submitter.get(), range_on_output);
			if (!range_mat_res.success) {
				result_tx_ptr->close();
				range_exchange->Close();
				staging_exchange->Close();
				return DuckDBResult<void>::err(DuckDBError::internal_error(range_mat_res.error));
			}
			range_exchange->AllRequiredSinksFinished();
			auto range_handles = range_exchange->GetSourceHandles();
			DebugOrderByOutputs("range-output", range_mat_res.outputs);
			DebugOrderByHandles("range-handles", range_handles);
			auto range_estimated_cardinality = EstimateRowsFromHandles(range_handles, staging_estimated_cardinality);

			auto final_tasks =
			    BuildExchangeSourceTasks(self_shared->context(), self_shared->config(), *task_id_counter, exchange_mgr,
			                             range_exchange_id, range_handles, num_partitions, source_task_count,
			                             input_types, range_estimated_cardinality, order_plan_builder, "OrderByFinal");
			if (OrderByDebugEnabled()) {
				std::cerr << "[distributed-orderby] final-tasks=" << final_tasks.size() << std::endl;
			}
			for (auto &task : final_tasks) {
				auto send_res = result_tx_ptr->send(SubmittableTask<WorkerTask>(std::move(task)));
				if (send_res.is_err()) {
					result_tx_ptr->close();
					range_exchange->Close();
					staging_exchange->Close();
					return DuckDBResult<void>::err(send_res.error());
				}
			}

			result_tx_ptr->close();
			range_exchange->Close();
			staging_exchange->Close();
			return DuckDBResult<void>::ok();
		} catch (const std::exception &ex) {
			result_tx_ptr->close();
			return DuckDBResult<void>::err(DuckDBError::internal_error(ex.what()));
		}
	});

	return SubmittableTaskStream<WorkerTask>::from_receiver(std::move(result_rx));
}

std::vector<std::string> OrderByNode::multiline_display(bool /*verbose*/) const {
	std::vector<std::string> lines;
	std::string ord;
	for (size_t i = 0; i < orders_.size(); ++i) {
		if (i > 0)
			ord += ", ";
		ord += orders_[i].ToString();
	}
	lines.push_back(std::string("OrderBy: ") + ord);
	return lines;
}

namespace {

using PlanBuilder = MaterializedPlanBuilder;

} // anonymous namespace

TopNNode::TopNNode(NodeID node_id, PipelineNodeRef child, vector<BoundOrderByNode> orders, idx_t limit, idx_t offset,
                   std::shared_ptr<ExchangeManager> exchange_mgr)
    : ctx_(InheritPipelineNodeContext(child, node_id, "TopN")), config_(child ? child->config() : PipelineNodeConfig()),
      child_(std::move(child)), orders_(std::move(orders)), limit_(limit), offset_(offset),
      exchange_mgr_(std::move(exchange_mgr)) {
}

SubmittableTaskStream<WorkerTask> TopNNode::produce_tasks(PlanExecutionContext &plan_context) {
	const auto *orders_ptr = &orders_;
	const idx_t limit_val = limit_;
	const idx_t offset_val = offset_;

	// Final plan builder: applies TopN(limit, offset) to the merged output.
	auto final_plan_builder = [orders_ptr, limit_val,
	                           offset_val](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
		auto orders = CopyOrderBys(*orders_ptr);
		auto input_types = input_plan->Root().GetTypes();
		FixOrderByTypes(orders, input_types);
		auto types = input_plan->Root().GetTypes();
		idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

		auto &old_root = input_plan->Root();
		auto &topn = input_plan->Make<duckdb::PhysicalTopN>(types, std::move(orders), limit_val, offset_val, nullptr,
		                                                    estimated_cardinality);
		topn.children.push_back(old_root);
		input_plan->SetRoot(topn);
		return input_plan;
	};

	// Single-partition path: just use pipeline_instruction (no coordinator).
	if (!ChildHasMultiplePartitions(child_)) {
		auto input_stream = child_->produce_tasks(plan_context);
		return input_stream.pipeline_instruction(shared_from_this(), final_plan_builder, plan_context.client_context());
	}

	// Multi-partition path: use global TopN coordinator.
	// Phase A per-task builder: TopN(limit=target_rows, offset=0) on each
	// worker to reduce data volume before shipping to the coordinator.
	idx_t target_rows = limit_val + offset_val;
	// Guard against overflow.
	if (target_rows < limit_val || target_rows < offset_val) {
		target_rows = std::numeric_limits<idx_t>::max();
	}

	auto per_task_builder_factory = [orders_ptr, target_rows](idx_t /*remaining_rows*/) -> PlanBuilder {
		return [orders_ptr, target_rows](DuckPhysicalPlanRef input_plan) -> DuckPhysicalPlanRef {
			auto orders = CopyOrderBys(*orders_ptr);
			auto input_types = input_plan->Root().GetTypes();
			FixOrderByTypes(orders, input_types);
			auto types = input_plan->Root().GetTypes();
			idx_t estimated_cardinality = input_plan->Root().estimated_cardinality;

			auto &old_root = input_plan->Root();
			auto &topn = input_plan->Make<duckdb::PhysicalTopN>(types, std::move(orders), target_rows, 0, nullptr,
			                                                    estimated_cardinality);
			topn.children.push_back(old_root);
			input_plan->SetRoot(topn);
			return input_plan;
		};
	};

	return ProduceWithMaterializedCoordinator(
	    plan_context, child_, std::static_pointer_cast<PipelineNodeImpl>(shared_from_this()),
	    std::move(final_plan_builder), std::move(per_task_builder_factory), exchange_mgr_);
}

std::vector<std::string> TopNNode::multiline_display(bool /*verbose*/) const {
	std::vector<std::string> lines;
	std::string ord;
	for (size_t i = 0; i < orders_.size(); ++i) {
		if (i > 0)
			ord += ", ";
		ord += orders_[i].ToString();
	}
	lines.push_back(std::string("TopN: ") + ord + " Limit: " + std::to_string(limit_) +
	                " Offset: " + std::to_string(offset_));
	return lines;
}

} // namespace distributed
} // namespace duckdb
