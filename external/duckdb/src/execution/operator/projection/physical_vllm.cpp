// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/projection/physical_vllm.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/vllm_executor.hpp"
#include "duckdb/common/serializer/serializer.hpp"

#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <numeric>
#include <queue>
#include <utility>

namespace duckdb {

namespace {

struct VLLMBucket {
	idx_t length;
	idx_t start;
	idx_t end;
};

struct VLLMBucketCompare {
	bool operator()(const VLLMBucket &left, const VLLMBucket &right) const {
		return left.length < right.length;
	}
};

static vector<string> GetPrompts(ExpressionExecutor &executor, DataChunk &input) {
	Vector result(LogicalType::VARCHAR);
	executor.ExecuteExpression(input, result);

	UnifiedVectorFormat format;
	result.ToUnifiedFormat(input.size(), format);
	auto data = reinterpret_cast<string_t *>(format.data);

	vector<string> prompts;
	prompts.reserve(input.size());
	for (idx_t i = 0; i < input.size(); i++) {
		auto idx = format.sel->get_index(i);
		if (!format.validity.RowIsValid(idx)) {
			throw InvalidInputException("vllm prompt cannot be NULL");
		}
		prompts.push_back(data[idx].GetString());
	}
	return prompts;
}

static idx_t CommonPrefixLength(const string &left, const string &right) {
	const auto max_len = MinValue<idx_t>(left.size(), right.size());
	idx_t match_len = 0;
	for (; match_len < max_len; match_len++) {
		if (left[match_len] != right[match_len]) {
			break;
		}
	}
	return match_len;
}

static idx_t CompleteUTF8PrefixLength(const string &value, idx_t prefix_len) {
	if (prefix_len >= value.size()) {
		return value.size();
	}
	while (prefix_len > 0 && (static_cast<unsigned char>(value[prefix_len]) & 0xC0) == 0x80) {
		prefix_len--;
	}
	return prefix_len;
}

static string ComputeBucketPrefix(const vector<string> &prompts, idx_t start, idx_t end) {
	if (end <= start) {
		return string();
	}
	const auto &first = prompts[start];
	if (first.empty()) {
		return string();
	}

	idx_t prefix_len = first.size();
	for (idx_t i = start + 1; i < end; i++) {
		prefix_len = MinValue<idx_t>(prefix_len, CommonPrefixLength(first, prompts[i]));
		if (prefix_len == 0) {
			break;
		}
	}

	return first.substr(0, CompleteUTF8PrefixLength(first, prefix_len));
}

static unique_ptr<DataChunk> CopyChunk(ClientContext &context, DataChunk &input) {
	auto copy = make_uniq<DataChunk>();
	copy->Initialize(context, input.GetTypes(), input.size());
	copy->Append(input, true);
	return copy;
}

static unique_ptr<DataChunk> BuildOutputChunk(ExecutionContext &context, DataChunk &rows, const vector<string> &outputs,
                                              const vector<bool> &output_validity) {
	if (rows.size() != outputs.size()) {
		throw InvalidInputException("vllm output count (%d) does not match input rows (%d)", outputs.size(),
		                            rows.size());
	}
	if (!output_validity.empty() && output_validity.size() != outputs.size()) {
		throw InvalidInputException("vllm output validity count (%d) does not match input rows (%d)",
		                            output_validity.size(), rows.size());
	}

	DataChunk output;
	output.Move(rows);

	DataChunk extra;
	extra.Initialize(context.client, {LogicalType::VARCHAR}, output.size());
	auto &extra_vec = extra.data[0];
	auto extra_data = FlatVector::GetData<string_t>(extra_vec);
	auto &extra_validity = FlatVector::Validity(extra_vec);
	extra_validity.SetAllValid(output.size());
	if (output_validity.empty()) {
		for (idx_t i = 0; i < output.size(); i++) {
			extra_data[i] = StringVector::AddString(extra_vec, outputs[i]);
		}
	} else {
		for (idx_t i = 0; i < output.size(); i++) {
			if (output_validity[i]) {
				extra_data[i] = StringVector::AddString(extra_vec, outputs[i]);
			} else {
				extra_validity.SetInvalid(i);
				extra_data[i] = string_t();
			}
		}
	}
	extra.SetCardinality(output.size());

	output.Fuse(extra);
	auto output_ptr = make_uniq<DataChunk>();
	output_ptr->Move(output);
	return output_ptr;
}

struct VLLMGlobalOperatorState : public GlobalOperatorState {
	VLLMGlobalOperatorState(string model_p, Value options_p)
	    : model(std::move(model_p)), options(std::move(options_p)) {
	}

	~VLLMGlobalOperatorState() override {
		ShutdownExecutorNoThrow();
	}

	string model;
	Value options;
	VLLMConfig config;
	bool config_initialized = false;

	//! Shared executor — created once, used by all pipeline threads.
	//! GIL serializes Python calls; ray.get() releases GIL for real parallelism.
	std::mutex executor_mutex;
	std::mutex executor_call_mutex;
	shared_ptr<VLLMExecutor> executor;

	//! Global backpressure tracking (atomic for lock-free access).
	//! Signed to avoid underflow when concurrent executors share an actor pool
	//! and concurrent executors receive results belonging to another executor.
	std::atomic<int64_t> submitted_prompts {0};
	std::atomic<int64_t> completed_prompts {0};

	//! Thread coordination for FinalExecute.
	std::atomic<idx_t> active_threads {1};
	std::atomic<idx_t> finished_threads {0};
	std::atomic<bool> global_finished_submitting {false};

	void PipelineMaxThreadsResolved(idx_t max_threads) override {
		active_threads.store(MaxValue<idx_t>(1, max_threads));
	}

	int64_t InflightPrompts() const {
		const auto submitted = submitted_prompts.load();
		const auto completed = completed_prompts.load();
		if (completed > submitted) {
			throw InternalException("vllm completed prompt count exceeded submitted prompt count");
		}
		return submitted - completed;
	}

	bool CanSubmitMore() const {
		auto inflight = InflightPrompts();
		if (config.inflight_limit == 0) {
			return true; // No backpressure limit (shared pool mode)
		}
		return static_cast<idx_t>(inflight) < config.inflight_limit;
	}

	void EnsureExecutor(ExecutionContext &context) {
		std::lock_guard<std::mutex> lock(executor_mutex);
		if (executor) {
			return;
		}
		auto factory = GetVLLMExecutorFactory();
		if (!factory) {
			throw InvalidInputException("vllm executor is not available in this build");
		}
		auto exec = factory(context.client, model, options, config);
		if (!exec) {
			throw InvalidInputException("vllm executor factory did not return an executor");
		}
		config.Validate();
		config_initialized = true;
		executor = shared_ptr<VLLMExecutor>(std::move(exec));
	}

	shared_ptr<VLLMExecutor> ExecutorRef() {
		std::lock_guard<std::mutex> lock(executor_mutex);
		return executor;
	}

	bool HasExecutor() {
		return static_cast<bool>(ExecutorRef());
	}

	template <class FUNC>
	auto WithExecutor(FUNC &&func) -> decltype(func(std::declval<VLLMExecutor &>())) {
		std::lock_guard<std::mutex> call_lock(executor_call_mutex);
		auto exec = ExecutorRef();
		if (!exec) {
			throw InvalidInputException("vllm executor is not initialized");
		}
		return func(*exec);
	}

	template <class FUNC>
	bool WithExecutorIfPresent(FUNC &&func) {
		std::lock_guard<std::mutex> call_lock(executor_call_mutex);
		auto exec = ExecutorRef();
		if (!exec) {
			return false;
		}
		func(*exec);
		return true;
	}

	bool WaitForExecutorResult(ClientContext &context) {
		auto exec = ExecutorRef();
		if (!exec) {
			return false;
		}
		// This call may block. Keep it outside executor_call_mutex so another
		// producer can submit the work that eventually wakes this waiter.
		exec->WaitForResult(context);
		return true;
	}

	VLLMWakeupRegistrationResult RegisterWakeup(ExecutionContext &context) {
		if (!context.interrupt_state) {
			return VLLMWakeupRegistrationResult::UNSUPPORTED;
		}
		// Another finalizer can drain the last result and shut the shared
		// executor down between this task's readiness check and wakeup
		// registration. Treat that terminal state as immediately actionable.
		auto result = VLLMWakeupRegistrationResult::READY;
		WithExecutorIfPresent([&](VLLMExecutor &exec) { result = exec.RegisterWakeup(*context.interrupt_state); });
		return result;
	}

	//! Called by each thread when it finishes submitting. The last thread
	//! signals the executor that no more prompts will arrive.
	void ThreadFinishedSubmitting(ClientContext &context) {
		auto finished = finished_threads.fetch_add(1) + 1;
		auto expected = active_threads.load();
		if (finished > expected) {
			throw InternalException("vllm producer finished more than once");
		}
		if (finished == expected) {
			if (!global_finished_submitting.exchange(true)) {
				WithExecutorIfPresent([&](VLLMExecutor &exec) { exec.FinishedSubmitting(context); });
			}
		}
	}

	void ShutdownExecutor() {
		shared_ptr<VLLMExecutor> executor_to_shutdown;
		{
			std::lock_guard<std::mutex> call_lock(executor_call_mutex);
			{
				std::lock_guard<std::mutex> lock(executor_mutex);
				executor_to_shutdown = executor;
			}
			if (executor_to_shutdown) {
				executor_to_shutdown->Shutdown();
				std::lock_guard<std::mutex> lock(executor_mutex);
				if (executor == executor_to_shutdown) {
					executor.reset();
				}
			}
		}
	}

	void ShutdownExecutorNoThrow() {
		try {
			ShutdownExecutor();
		} catch (...) {
		}
	}
};

struct VLLMOperatorState : public OperatorState {
	VLLMOperatorState(ClientContext &context, const Expression &prompt_expr) : prompt_executor(context, prompt_expr) {
	}

	ExpressionExecutor prompt_executor;

	vector<unique_ptr<DataChunk>> buffer;
	idx_t buffer_size = 0;
	bool finished_submitting = false;
	std::deque<unique_ptr<DataChunk>> pending_outputs;
};

static void TakeReadyResultOnce(ExecutionContext &context, VLLMGlobalOperatorState &gstate, VLLMOperatorState &state,
                                idx_t input_column_count);

static void SubmitPrompts(ExecutionContext &context, VLLMGlobalOperatorState &gstate, VLLMOperatorState &state,
                          optional_ptr<const string> prefix, const vector<string> &prompts, DataChunk &rows) {
	if (prompts.empty()) {
		return;
	}

	if (!gstate.config.batch_size.IsValid() || prompts.size() <= gstate.config.batch_size.GetIndex()) {
		gstate.WithExecutor([&](VLLMExecutor &exec) {
			exec.Submit(prefix, prompts, rows, context.client);
			// Publish after Python has installed its reservation and wait state.
			// executor_call_mutex also prevents a result drain from racing ahead
			// of this native accounting update.
			gstate.submitted_prompts.fetch_add(prompts.size());
		});
		return;
	}

	const auto batch_size = gstate.config.batch_size.GetIndex();
	const idx_t input_column_count = rows.ColumnCount();
	for (idx_t offset = 0; offset < prompts.size(); offset += batch_size) {
		// Backpressure between batches: wait for inflight to drop so that
		// large chunks are submitted as a stream, not in one burst.
		if (offset > 0) {
			while (!gstate.CanSubmitMore()) {
				TakeReadyResultOnce(context, gstate, state, input_column_count);
				if (gstate.CanSubmitMore()) {
					break;
				}
				if (!gstate.WaitForExecutorResult(context.client)) {
					throw InternalException("vllm executor disappeared during batch backpressure");
				}
			}
		}
		const idx_t count = MinValue<idx_t>(batch_size, prompts.size() - offset);
		vector<string> batch_prompts(prompts.begin() + offset, prompts.begin() + offset + count);

		DataChunk batch_rows;
		batch_rows.Initialize(context.client, rows.GetTypes(), count);
		SelectionVector sel(count);
		for (idx_t i = 0; i < count; i++) {
			sel.set_index(i, offset + i);
		}
		batch_rows.Slice(rows, sel, count, 0);
		batch_rows.Flatten();

		gstate.WithExecutor([&](VLLMExecutor &exec) {
			exec.Submit(prefix, std::move(batch_prompts), batch_rows, context.client);
			gstate.submitted_prompts.fetch_add(count);
		});
	}
}

static void TakeReadyResultOnce(ExecutionContext &context, VLLMGlobalOperatorState &gstate, VLLMOperatorState &state,
                                idx_t input_column_count) {
	std::lock_guard<std::mutex> call_lock(gstate.executor_call_mutex);
	auto exec = gstate.ExecutorRef();
	if (!exec) {
		return;
	}
	auto result = exec->TakeReadyResult(context.client);
	if (!result.first) {
		return;
	}
	if (!result.second.rows) {
		throw InvalidInputException("vllm executor returned empty rows");
	}
	if (result.second.rows->ColumnCount() != input_column_count) {
		throw InvalidInputException("vllm executor returned %d columns, expected %d", result.second.rows->ColumnCount(),
		                            input_column_count);
	}
	gstate.completed_prompts.fetch_add(result.second.rows->size());
	auto output = BuildOutputChunk(context, *result.second.rows, result.second.outputs, result.second.outputs_validity);
	state.pending_outputs.push_back(std::move(output));
}

static void PopAndSubmitTasks(ExecutionContext &context, VLLMGlobalOperatorState &gstate, VLLMOperatorState &state,
                              idx_t max_buffer_size) {
	if (state.buffer_size <= max_buffer_size) {
		return;
	}
	if (state.buffer.empty()) {
		return;
	}

	auto types = state.buffer[0]->GetTypes();
	DataChunk concatted;
	concatted.Initialize(context.client, types, state.buffer_size);
	for (auto &chunk_ptr : state.buffer) {
		concatted.Append(*chunk_ptr, true);
	}

	auto prompts = GetPrompts(state.prompt_executor, concatted);
	if (prompts.empty()) {
		state.buffer.clear();
		state.buffer_size = 0;
		return;
	}

	vector<idx_t> order(prompts.size());
	std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(), [&](idx_t left, idx_t right) { return prompts[left] < prompts[right]; });

	SelectionVector sel(order.size());
	vector<string> sorted_prompts;
	sorted_prompts.reserve(order.size());
	for (idx_t i = 0; i < order.size(); i++) {
		sel.set_index(i, order[i]);
		sorted_prompts.push_back(prompts[order[i]]);
	}

	DataChunk sorted;
	sorted.Initialize(context.client, types, order.size());
	sorted.Slice(concatted, sel, order.size(), 0);

	std::priority_queue<VLLMBucket, std::vector<VLLMBucket>, VLLMBucketCompare> splits;
	idx_t prev_split_idx = 0;
	for (idx_t i = 0; i + 1 < sorted_prompts.size(); i++) {
		const auto common_prefix_len = CommonPrefixLength(sorted_prompts[i], sorted_prompts[i + 1]);
		const double p1_ratio =
		    sorted_prompts[i].empty() ? 0.0 : static_cast<double>(common_prefix_len) / sorted_prompts[i].size();
		const double p2_ratio =
		    sorted_prompts[i + 1].empty() ? 0.0 : static_cast<double>(common_prefix_len) / sorted_prompts[i + 1].size();

		if (p1_ratio < gstate.config.prefix_match_threshold && p2_ratio < gstate.config.prefix_match_threshold) {
			const idx_t next_split = i + 1;
			splits.push({next_split - prev_split_idx, prev_split_idx, next_split});
			prev_split_idx = next_split;
		}
	}
	const idx_t end_idx = sorted_prompts.size();
	if (end_idx > prev_split_idx) {
		splits.push({end_idx - prev_split_idx, prev_split_idx, end_idx});
	}

	vector<string> curr_prompts;
	unique_ptr<DataChunk> curr_rows;
	const idx_t min_bucket_size = MaxValue<idx_t>(1, gstate.config.min_bucket_size);
	const idx_t input_column_count = types.size();

	while (state.buffer_size > max_buffer_size && !splits.empty()) {
		// Backpressure: wait for inflight to drop before submitting the next bucket.
		while (!gstate.CanSubmitMore()) {
			TakeReadyResultOnce(context, gstate, state, input_column_count);
			if (gstate.CanSubmitMore()) {
				break;
			}
			if (!gstate.WaitForExecutorResult(context.client)) {
				throw InternalException("vllm executor disappeared during bucket backpressure");
			}
		}

		auto bucket = splits.top();
		splits.pop();

		vector<string> bucket_prompts(sorted_prompts.begin() + bucket.start, sorted_prompts.begin() + bucket.end);
		auto prefix = ComputeBucketPrefix(sorted_prompts, bucket.start, bucket.end);

		DataChunk bucket_rows;
		bucket_rows.Initialize(context.client, types, bucket.length);
		SelectionVector bucket_sel(bucket.length);
		for (idx_t i = 0; i < bucket.length; i++) {
			bucket_sel.set_index(i, bucket.start + i);
		}
		bucket_rows.Slice(sorted, bucket_sel, bucket.length, 0);
		bucket_rows.Flatten();

		if (bucket.length >= min_bucket_size) {
			optional_ptr<const string> prefix_ptr;
			if (!prefix.empty()) {
				prefix_ptr = optional_ptr<const string>(prefix);
			}
			SubmitPrompts(context, gstate, state, prefix_ptr, bucket_prompts, bucket_rows);
		} else {
			if (!curr_rows) {
				curr_rows = make_uniq<DataChunk>();
				curr_rows->Initialize(context.client, types, bucket_rows.size());
			}
			curr_rows->Append(bucket_rows, true);
			curr_prompts.insert(curr_prompts.end(), bucket_prompts.begin(), bucket_prompts.end());
			if (curr_prompts.size() >= min_bucket_size) {
				SubmitPrompts(context, gstate, state, nullptr, curr_prompts, *curr_rows);
				curr_prompts.clear();
				curr_rows.reset();
			}
		}

		state.buffer_size -= bucket.length;
	}

	if (curr_rows) {
		SubmitPrompts(context, gstate, state, nullptr, curr_prompts, *curr_rows);
	}

	state.buffer.clear();
	state.buffer_size = 0;
	while (!splits.empty()) {
		auto bucket = splits.top();
		splits.pop();
		DataChunk remaining;
		remaining.Initialize(context.client, types, bucket.length);
		SelectionVector remaining_sel(bucket.length);
		for (idx_t i = 0; i < bucket.length; i++) {
			remaining_sel.set_index(i, bucket.start + i);
		}
		remaining.Slice(sorted, remaining_sel, bucket.length, 0);
		remaining.Flatten();
		state.buffer_size += bucket.length;
		auto remaining_ptr = make_uniq<DataChunk>();
		remaining_ptr->Move(remaining);
		state.buffer.push_back(std::move(remaining_ptr));
	}
}

} // namespace

PhysicalVLLM::PhysicalVLLM(PhysicalPlan &physical_plan, vector<LogicalType> types, unique_ptr<Expression> prompt_expr_p,
                           string model_p, Value options_p, string output_column_name_p, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::VLLM_PROJECT, std::move(types), estimated_cardinality),
      prompt_expr(std::move(prompt_expr_p)), model(std::move(model_p)), options(std::move(options_p)),
      output_column_name(std::move(output_column_name_p)) {
}

unique_ptr<GlobalOperatorState> PhysicalVLLM::GetGlobalOperatorState(ClientContext &) const {
	return make_uniq<VLLMGlobalOperatorState>(model, options);
}

unique_ptr<OperatorState> PhysicalVLLM::GetOperatorState(ExecutionContext &context) const {
	if (!prompt_expr) {
		throw InvalidInputException("vllm operator is missing prompt expression");
	}
	return make_uniq<VLLMOperatorState>(context.client, *prompt_expr);
}

OperatorResultType PhysicalVLLM::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                         GlobalOperatorState &gstate_p, OperatorState &state_p) const {
	auto &gstate = gstate_p.Cast<VLLMGlobalOperatorState>();
	auto &state = state_p.Cast<VLLMOperatorState>();

	if (!state.pending_outputs.empty()) {
		auto output = std::move(state.pending_outputs.front());
		state.pending_outputs.pop_front();
		chunk.Move(*output);
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}

	if (input.size() == 0) {
		chunk.SetCardinality(0);
		return OperatorResultType::NEED_MORE_INPUT;
	}

	gstate.EnsureExecutor(context);

	// Backpressure: if inflight is at limit, drain ready results and wait.
	while (!gstate.CanSubmitMore()) {
		TakeReadyResultOnce(context, gstate, state, input.ColumnCount());
		if (gstate.CanSubmitMore()) {
			break;
		}
		if (!gstate.WaitForExecutorResult(context.client)) {
			throw InternalException("vllm executor disappeared during operator backpressure");
		}
	}

	if (gstate.config.do_prefix_routing) {
		state.buffer_size += input.size();
		state.buffer.push_back(CopyChunk(context.client, input));
		PopAndSubmitTasks(context, gstate, state, gstate.config.max_buffer_size);
	} else {
		auto prompts = GetPrompts(state.prompt_executor, input);
		SubmitPrompts(context, gstate, state, nullptr, prompts, input);
	}

	TakeReadyResultOnce(context, gstate, state, input.ColumnCount());
	if (!state.pending_outputs.empty()) {
		auto output = std::move(state.pending_outputs.front());
		state.pending_outputs.pop_front();
		chunk.Move(*output);
	} else {
		chunk.SetCardinality(0);
	}
	return OperatorResultType::NEED_MORE_INPUT;
}

OperatorFinalizeResultType PhysicalVLLM::FinalExecute(ExecutionContext &context, DataChunk &chunk,
                                                      GlobalOperatorState &gstate_p, OperatorState &state_p) const {
	auto &gstate = gstate_p.Cast<VLLMGlobalOperatorState>();
	auto &state = state_p.Cast<VLLMOperatorState>();

	if (!state.pending_outputs.empty()) {
		auto output = std::move(state.pending_outputs.front());
		state.pending_outputs.pop_front();
		chunk.Move(*output);
		return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
	}

	if (!state.finished_submitting) {
		if (!state.buffer.empty()) {
			gstate.EnsureExecutor(context);
			PopAndSubmitTasks(context, gstate, state, 0);
		}
		state.finished_submitting = true;
		gstate.ThreadFinishedSubmitting(context.client);
	}

	if (!gstate.HasExecutor()) {
		chunk.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}

	TakeReadyResultOnce(context, gstate, state, types.size() - 1);
	if (!state.pending_outputs.empty()) {
		auto output = std::move(state.pending_outputs.front());
		state.pending_outputs.pop_front();
		chunk.Move(*output);
		return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
	}

	bool all_tasks_finished = false;
	if (!gstate.WithExecutorIfPresent(
	        [&](VLLMExecutor &exec) { all_tasks_finished = exec.AllTasksFinished(context.client); })) {
		chunk.SetCardinality(0);
		return OperatorFinalizeResultType::FINISHED;
	}
	if (all_tasks_finished) {
		gstate.ShutdownExecutor();
		return OperatorFinalizeResultType::FINISHED;
	}

	// Arm before yielding so result, error, cancellation, or a later producer
	// cannot race between the readiness check above and task suspension.
	auto wakeup_result = gstate.RegisterWakeup(context);
	chunk.SetCardinality(0);
	if (wakeup_result == VLLMWakeupRegistrationResult::ARMED) {
		return OperatorFinalizeResultType::BLOCKED;
	}
	if (wakeup_result == VLLMWakeupRegistrationResult::UNSUPPORTED && gstate.InflightPrompts() > 0) {
		// Compatibility fallback for legacy executors. Waiting with zero inflight
		// would deadlock while another producer is still able to submit.
		gstate.WaitForExecutorResult(context.client);
	}
	return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
}

InsertionOrderPreservingMap<string> PhysicalVLLM::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["__prompt_expr__"] = prompt_expr ? prompt_expr->GetName() : string("<null>");
	result["__model__"] = model;
	result["__output_column__"] = output_column_name;
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

void PhysicalVLLM::SerializeOperatorData(Serializer &serializer) const {
	serializer.WriteProperty(103, "prompt_expr", prompt_expr);
	serializer.WriteProperty(104, "model", model);
	serializer.WriteProperty(105, "options", options);
	serializer.WriteProperty(106, "output_column_name", output_column_name);
}

} // namespace duckdb
