// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "direct_task.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/prepared_statement_data.hpp"

#include <set>

namespace duckdb {
namespace vane_execution {
namespace {

ExchangeWakeup Wakeup(const InterruptState &interrupt) {
	return [interrupt]() {
		interrupt.Callback();
	};
}

class DirectSourceState : public GlobalSourceState {
public:
	explicit DirectSourceState(idx_t count) : ended(count, false) {
	}
	vector<bool> ended;
	idx_t cursor = 0;
};

class DirectSource : public PhysicalOperator {
public:
	DirectSource(PhysicalPlan &plan, const vector<LogicalType> &types, vector<DirectInput> inputs_p)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, types, 0), inputs(std::move(inputs_p)) {
		if (inputs.empty()) {
			throw InvalidInputException("direct source needs at least one input");
		}
		for (auto &input : inputs) {
			if (!input.channel || input.channel->types != types) {
				throw InvalidInputException("direct source schema mismatch");
			}
			input.channel->ValidateConsumer(input.consumer);
		}
	}
	string GetName() const override {
		return "DIRECT_EXCHANGE_SOURCE";
	}
	bool IsSource() const override {
		return true;
	}
	ExecutionBatchRequirement GetExecutionBatchRequirement(PipelineOperatorRole) const override {
		return ExecutionBatchRequirement::BATCH_REQUIRED;
	}
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &) const override {
		return make_uniq<DirectSourceState>(inputs.size());
	}
	SourceResultType GetDataInternal(ExecutionContext &, DataChunk &chunk, OperatorSourceInput &input) const override {
		auto &state = input.global_state.Cast<DirectSourceState>();
		idx_t ended = 0;
		for (idx_t step = 0; step < inputs.size(); step++) {
			auto index = (state.cursor + step) % inputs.size();
			if (!state.ended[index]) {
				shared_ptr<DirectBatch> batch;
				auto result = inputs[index].channel->Poll(inputs[index].consumer, batch, Wakeup(input.interrupt_state));
				if (result == DirectRead::DATA) {
					batch->Reference(chunk);
					state.cursor = (index + 1) % inputs.size();
					return SourceResultType::HAVE_MORE_OUTPUT;
				}
				state.ended[index] = result == DirectRead::END || result == DirectRead::CLOSED;
			}
			ended += state.ended[index];
		}
		return ended == inputs.size() ? SourceResultType::FINISHED : SourceResultType::BLOCKED;
	}
	const vector<DirectInput> inputs;
};

class DirectSinkLocal : public LocalSinkState {
public:
	bool initialized = false;
	idx_t output = 0;
	idx_t channel = 0;
	idx_t offset = 0;
	vector<vector<vector<idx_t>>> selections;
	void ResetBatchInput() override {
		initialized = false;
		output = channel = offset = 0;
		selections.clear();
	}
};

class DirectSinkGlobal : public GlobalSinkState {
public:
	explicit DirectSinkGlobal(ClientContext &context_p) : context(context_p) {
	}
	ClientContext &context;
};

class DirectCollector : public PhysicalResultCollector {
public:
	DirectCollector(PreparedStatementData &data, vector<DirectOutput> outputs_p)
	    : PhysicalResultCollector(*data.physical_plan, data), outputs(std::move(outputs_p)) {
	}
	ExecutionBatchRequirement GetExecutionBatchRequirement(PipelineOperatorRole) const override {
		return ExecutionBatchRequirement::BATCH_REQUIRED;
	}
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override {
		return make_uniq<DirectSinkGlobal>(context);
	}
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &) const override {
		return make_uniq<DirectSinkLocal>();
	}
	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override {
		auto &state = input.local_state.Cast<DirectSinkLocal>();
		bool live = false;
		for (auto &output : outputs) {
			for (auto &channel : output.channels) {
				live = channel->HasConsumers() || live;
			}
		}
		if (!live) {
			state.ResetBatchInput();
			return SinkResultType::FINISHED;
		}
		if (!state.initialized) {
			state.selections.resize(outputs.size());
			for (idx_t out = 0; out < outputs.size(); out++) {
				auto &output = outputs[out];
				auto &selection = state.selections[out];
				selection.resize(output.channels.size());
				if (!output.partitioning.empty()) {
					auto ids = HashPartitions(context.client, output.partitioning, chunk, output.channels.size());
					for (idx_t row = 0; row < ids.size(); row++) {
						selection[ids[row]].push_back(row);
					}
				} else {
					for (auto &rows : selection) {
						for (idx_t row = 0; row < chunk.size(); row++) {
							rows.push_back(row);
						}
					}
				}
			}
			state.initialized = true;
		}
		for (; state.output < outputs.size(); state.output++, state.channel = 0) {
			auto &output = outputs[state.output];
			for (; state.channel < output.channels.size(); state.channel++, state.offset = 0) {
				auto &channel = *output.channels[state.channel];
				auto &rows = state.selections[state.output][state.channel];
				while (state.offset < rows.size()) {
					auto count = channel.FrameRows(chunk, rows, state.offset);
					auto result = channel.TryWrite(output.producer, channel.LastSequence(output.producer) + 1, chunk,
					                               rows, state.offset, count, Wakeup(input.interrupt_state));
					if (result == DirectWrite::BLOCKED) {
						return SinkResultType::BLOCKED;
					}
					if (result == DirectWrite::CLOSED) {
						break;
					}
					state.offset += count;
				}
			}
		}
		// The executor owns the input batch while BLOCKED; only row offsets are
		// retained here. Resume cannot replay any accepted frame or target.
		state.ResetBatchInput();
		return SinkResultType::NEED_MORE_INPUT;
	}
	SinkFinalizeType Finalize(Pipeline &, Event &, ClientContext &, OperatorSinkFinalizeInput &) const override {
		for (auto &output : outputs) {
			for (auto &channel : output.channels) {
				channel->Finish(output.producer, channel->LastSequence(output.producer));
			}
		}
		// FINISH closes production. Outstanding borrowed frames are drained by
		// TaskService, never by this finalizer or a native worker thread.
		return SinkFinalizeType::READY;
	}
	unique_ptr<QueryResult> GetResult(GlobalSinkState &state) const override {
		auto &context = state.Cast<DirectSinkGlobal>().context;
		return make_uniq<MaterializedQueryResult>(statement_type, properties, names, CreateCollection(context),
		                                          context.GetClientProperties());
	}
	const vector<DirectOutput> outputs;
};

} // namespace

DirectTaskService::DirectTaskService(shared_ptr<DatabaseInstance> database_p) : database(std::move(database_p)) {
}

DirectTaskService::~DirectTaskService() {
	try {
		Cancel("task service destroyed");
		Release();
	} catch (...) {
	}
}

void DirectTaskService::CheckCanceled() const {
	if (canceled.load()) {
		throw InvalidInputException("direct task service is canceled");
	}
}

void DirectTaskService::Prepare(const string &id, const string &payload, const string &connection_snapshot,
                                const string &source_snapshot, const SourceAssignments &assignments,
                                const std::map<string, vector<DirectInput>> &inputs,
                                const vector<DirectOutput> &outputs) {
	lock_guard<mutex> operation(operation_lock);
	CheckCanceled();
	if (started || id.empty()) {
		throw InvalidInputException("prepare requires a task id and must precede start");
	}
	for (auto &task : tasks) {
		if (task->id == id) {
			throw InvalidInputException("duplicate direct task id");
		}
	}
	std::set<std::pair<DirectChannel *, string>> input_ids, output_ids;
	for (auto &task : tasks) {
		for (auto &port : task->inputs) {
			for (auto &input : port.second) {
				input_ids.emplace(input.channel.get(), input.consumer);
			}
		}
		for (auto &output : task->outputs) {
			for (auto &channel : output.channels) {
				output_ids.emplace(channel.get(), output.producer);
			}
		}
	}
	for (auto &port : inputs) {
		for (auto &input : port.second) {
			if (!input_ids.emplace(input.channel.get(), input.consumer).second) {
				throw InvalidInputException("direct consumer is already bound to a task input");
			}
		}
	}
	auto task = make_uniq<Task>();
	task->id = id;
	task->connection = make_uniq<Connection>(*database);
	task->fragment = FragmentSpec::Deserialize(payload);
	task->source_snapshot = source_snapshot;
	task->inputs = inputs;
	task->outputs = outputs;
	auto &context = *task->connection->context;
	ApplyConnection(context, connection_snapshot);
	ValidateSources(context, task->fragment, source_snapshot, false);
	std::set<string> used;
	auto plan = Load(
	    context, task->fragment,
	    [&](PhysicalPlan &owner, const string &port, const vector<LogicalType> &types) -> PhysicalOperator & {
		    auto input = inputs.find(port);
		    if (input == inputs.end()) {
			    throw InvalidInputException("missing direct input port %s", port);
		    }
		    used.insert(port);
		    return owner.Make<DirectSource>(types, input->second);
	    },
	    assignments);
	if (used.size() != inputs.size() || outputs.empty()) {
		throw InvalidInputException("unexpected direct input port or missing output");
	}
	for (auto &output : outputs) {
		if (output.channels.empty() || output.producer.empty()) {
			throw InvalidInputException("missing direct output channel or producer");
		}
		for (auto &channel : output.channels) {
			if (!channel || channel->types != task->fragment.root.types ||
			    !output_ids.emplace(channel.get(), output.producer).second) {
				throw InvalidInputException("direct output schema mismatch or duplicate writer");
			}
			channel->LastSequence(output.producer); // Registration must precede preparation.
		}
		if (!output.partitioning.empty()) {
			DataChunk empty;
			empty.Initialize(Allocator::Get(context), task->fragment.root.types);
			HashPartitions(context, output.partitioning, empty, output.channels.size());
		}
	}
	task->prepared = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
	task->prepared->names = task->fragment.names;
	task->prepared->types = task->fragment.root.types;
	task->prepared->physical_plan = std::move(plan);
	task->prepared->properties.return_type = StatementReturnType::QUERY_RESULT;
	task->prepared->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	task->prepared->memory_type = QueryResultMemoryType::IN_MEMORY;
	{
		lock_guard<mutex> registry(registry_lock);
		CheckCanceled();
		contexts.push_back(task->connection->context);
		for (auto &entry : inputs) {
			for (auto &input : entry.second) {
				channels.push_back(input.channel);
			}
		}
		for (auto &output : outputs) {
			channels.insert(channels.end(), output.channels.begin(), output.channels.end());
		}
		tasks.push_back(std::move(task));
	}
}

void DirectTaskService::Start(const string &id, const string &token) {
	lock_guard<mutex> operation(operation_lock);
	CheckCanceled();
	if (token.empty()) {
		throw InvalidInputException("direct start token must be nonempty");
	}
	for (auto &task : tasks) {
		if (task->id != id) {
			continue;
		}
		if (!task->token.empty()) {
			if (task->token != token) {
				throw InvalidInputException("conflicting direct start token");
			}
			return;
		}
		if (task->released) {
			throw InvalidInputException("cannot start a released direct task");
		}
		started = true;
		task->token = token;
		try {
			auto &context = *task->connection->context;
			ValidateSources(context, task->fragment, task->source_snapshot, false);
			PendingQueryParameters parameters;
			auto outputs = task->outputs;
			parameters.get_result_collector = [outputs](ClientContext &, PreparedStatementData &data) {
				return make_uniq<DirectCollector>(data, outputs);
			};
			task->pending =
			    context.PendingQueryPreparedStatementNoRebind("direct fragment task", task->prepared, parameters);
			if (task->pending->HasError()) {
				task->pending->ThrowError();
			}
			task->state = "RUNNING";
			// Cancellation before PendingQuery initialization may have had its
			// interrupt cleared by query startup. Recheck before publishing start.
			CheckCanceled();
		} catch (std::exception &ex) {
			Fail(*task, ex.what());
			throw;
		}
		return;
	}
	throw InvalidInputException("unknown direct task: %s", id);
}

void DirectTaskService::Refresh(Task &task) {
	if (task.state == "FINISHED" || task.state == "FAILED") {
		return;
	}
	if (canceled.load()) {
		task.state = "CANCELED";
		if (task.error.empty()) {
			lock_guard<mutex> registry(registry_lock);
			task.error = cancel_reason;
		}
		return;
	}
	bool live = false;
	bool drained = true;
	for (auto &output : task.outputs) {
		for (auto &channel : output.channels) {
			auto status = channel->ProducerStatus(output.producer);
			// Abort discards queued frames. That releases credit, but cannot
			// turn failed delivery into success, even if another output is pending.
			if (!status.error.empty()) {
				Fail(task, status.error);
				return;
			}
			live = live || status.has_consumers;
			drained = drained && status.drained;
		}
	}
	if (task.state == "RUNNING" && !live) {
		// A source can be BLOCKED forever and never call Sink again. Stop its
		// executor before closing production; cleanup also closes every input,
		// propagating the loss of demand upstream. Borrowed output stays leased.
		try {
			if (task.pending->CheckPulse() == PendingExecutionResult::EXECUTION_ERROR) {
				task.pending->ThrowError();
			}
			Cleanup(task);
			for (auto &output : task.outputs) {
				for (auto &channel : output.channels) {
					channel->Finish(output.producer, channel->LastSequence(output.producer));
				}
			}
			task.state = "OUTPUT_PENDING";
		} catch (std::exception &ex) {
			Fail(task, ex.what());
			throw;
		}
		Refresh(task);
		return;
	}
	if (task.state == "OUTPUT_PENDING" && drained) {
		task.state = "FINISHED";
	}
}

void DirectTaskService::Cleanup(Task &task) {
	if (task.released) {
		return;
	}
	if (task.connection) {
		task.connection->context->CancelTransaction();
	}
	task.pending.reset();
	task.prepared.reset();
	task.connection.reset();
	for (auto &port : task.inputs) {
		for (auto &input : port.second) {
			input.channel->CloseConsumer(input.consumer);
		}
	}
	task.released = true;
}

void DirectTaskService::Fail(Task &task, const string &message) {
	task.state = canceled.load() ? "CANCELED" : "FAILED";
	task.error = message;
	Cancel(message);
	Cleanup(task);
}

idx_t DirectTaskService::Pump(idx_t steps) {
	lock_guard<mutex> operation(operation_lock);
	idx_t progress = 0;
	for (idx_t step = 0; step < steps && !tasks.empty(); step++) {
		auto &task = *tasks[next_task++ % tasks.size()];
		if (canceled.load()) {
			for (auto &pending : tasks) {
				if (pending->state != "FINISHED" && pending->state != "FAILED") {
					pending->state = "CANCELED";
				}
				Cleanup(*pending);
			}
			break;
		}
		Refresh(task);
		if (task.state != "RUNNING") {
			continue;
		}
		try {
			auto result = task.pending->ExecuteTask();
			CheckCanceled();
			if (result == PendingExecutionResult::EXECUTION_ERROR) {
				task.pending->ThrowError();
			}
			if (PendingQueryResult::IsResultReady(result)) {
				auto control = task.pending->Execute();
				if (control->HasError()) {
					control->ThrowError();
				}
				{
					lock_guard<mutex> registry(registry_lock);
					CheckCanceled();
					task.state = "OUTPUT_PENDING";
				}
				Cleanup(task);
				Refresh(task);
				progress++;
			} else if (result == PendingExecutionResult::RESULT_NOT_READY) {
				progress++;
			}
		} catch (std::exception &ex) {
			Fail(task, ex.what());
			throw;
		}
	}
	return progress;
}

void DirectTaskService::Cancel(const string &reason) {
	Stop(reason, false);
}

bool DirectTaskService::Expire() {
	return Stop("execution deadline exceeded", true);
}

bool DirectTaskService::Stop(const string &reason, bool only_running) {
	vector<shared_ptr<ClientContext>> interrupt;
	vector<shared_ptr<DirectChannel>> abort;
	string message;
	{
		lock_guard<mutex> registry(registry_lock);
		if (only_running) {
			if (canceled.load()) {
				return false;
			}
			bool unfinished = false;
			// Membership and output bindings are immutable after preparation.
			// Read FINISH at its native publication point: background executor
			// threads can complete without any subsequent Pump call. A deadline
			// is accepted only while at least one output is still producing.
			for (auto &task : tasks) {
				for (auto &output : task->outputs) {
					for (auto &channel : output.channels) {
						unfinished = !channel->ProducerStatus(output.producer).finished || unfinished;
					}
				}
			}
			if (!unfinished) {
				return false;
			}
		}
		if (!canceled.exchange(true)) {
			cancel_reason = reason.empty() ? "canceled" : reason;
		}
		message = cancel_reason;
		for (auto &weak_context : contexts) {
			auto context = weak_context.lock();
			if (context) {
				interrupt.push_back(std::move(context));
			}
		}
		abort = channels;
	}
	for (auto &context : interrupt) {
		context->Interrupt();
	}
	for (auto &channel : abort) {
		channel->Abort(message);
	}
	return true;
}

void DirectTaskService::Release() {
	lock_guard<mutex> operation(operation_lock);
	for (auto &task : tasks) {
		Refresh(*task);
		if (task->state == "RUNNING" || task->state == "PREPARED" ||
		    (canceled.load() && task->state == "OUTPUT_PENDING")) {
			if (!canceled.load()) {
				throw InvalidInputException("cancel active tasks before release");
			}
			task->state = "CANCELED";
		}
		Cleanup(*task);
	}
}

vector<DirectTaskStatus> DirectTaskService::Status() {
	lock_guard<mutex> operation(operation_lock);
	vector<DirectTaskStatus> result;
	for (auto &task : tasks) {
		Refresh(*task);
		DirectTaskStatus status;
		status.task_id = task->id;
		status.state = task->state;
		status.error = task->error;
		status.released = task->released;
		result.push_back(std::move(status));
	}
	return result;
}

} // namespace vane_execution
} // namespace duckdb
