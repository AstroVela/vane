// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/projection/physical_media_project.hpp"

#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/managed_media_scalar.hpp"
#include "duckdb/execution/media_buffer.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"

#include <limits>

namespace duckdb {
namespace {
idx_t AddBytes(idx_t left, idx_t right) {
	if (right > std::numeric_limits<idx_t>::max() - left) {
		throw OutOfMemoryException("Managed media working-set size overflow");
	}
	return left + right;
}

idx_t FixedBytes(const LogicalType &type) {
	if (type.InternalType() == PhysicalType::ARRAY) {
		auto child_bytes = FixedBytes(ArrayType::GetChildType(type));
		auto count = ArrayType::GetSize(type);
		if (ArrayType::GetChildType(type).IsNested() || count > std::numeric_limits<idx_t>::max() / child_bytes) {
			throw NotImplementedException("Managed media requires fixed arrays of primitive numeric values");
		}
		return count * child_bytes;
	}
	if (!type.IsNumeric() && type.id() != LogicalTypeId::BOOLEAN) {
		throw NotImplementedException("Managed media input requires fixed numeric/boolean columns or primitive arrays");
	}
	return GetTypeIdSize(type.InternalType());
}

idx_t ManagedColumn(const vector<unique_ptr<Expression>> &expressions) {
	idx_t column = DConstants::INVALID_INDEX;
	for (idx_t i = 0; i < expressions.size(); i++) {
		auto &expression = *expressions[i];
		if (GetManagedMediaScalar(expression)) {
			if (column != DConstants::INVALID_INDEX) {
				throw NotImplementedException("Managed media currently supports one scalar call per query");
			}
			column = i;
		} else if (expression.GetExpressionClass() != ExpressionClass::BOUND_REF &&
		           (expression.GetExpressionClass() != ExpressionClass::BOUND_CONSTANT ||
		            expression.return_type.IsNested())) {
			throw NotImplementedException("Managed media projection fields must be column references or primitive constants");
		}
		if (!GetManagedMediaScalar(expression)) {
			FixedBytes(expression.return_type);
		}
	}
	if (column == DConstants::INVALID_INDEX) {
		throw InternalException("MEDIA_PROJECT has no managed scalar capability");
	}
	return column;
}

} // namespace

struct MediaProjectBatch {
	shared_ptr<ClientContext> context; // Storage and requests are destroyed before their host.
	vector<unique_ptr<ColumnDataCollection>> prefixes;
	unique_ptr<MediaAdmissionRequest> request;
	idx_t rows = 0;
};

//! Every detached vector retains this ancestor until its actual allocation dies.
struct MediaProjectBatchLifetime {
	MediaProjectBatchLifetime(MediaResourceLease lease_p, idx_t row_bytes, idx_t output_bytes)
	    : lease(std::move(lease_p)), row_pool({0, row_bytes}), output_pool({0, output_bytes}),
	      row_scope(row_pool.CreateScope()), output_scope(output_pool.CreateScope()) {
	}
	MediaResourceLease lease; // Destroyed last, after both child pools/scopes.
	MediaResourcePool row_pool;
	MediaResourcePool output_pool;
	shared_ptr<MediaResourceScope> row_scope;
	shared_ptr<MediaResourceScope> output_scope;
};

struct MediaProjectRowLifetime {
	MediaProjectRowLifetime(shared_ptr<MediaProjectBatchLifetime> batch_p, MediaResourceLease lease_p)
	    : batch(std::move(batch_p)), lease(std::move(lease_p)) {
	}
	shared_ptr<MediaProjectBatchLifetime> batch;
	MediaResourceLease lease;
};

namespace {
void Wake(vector<InterruptState> tasks) {
	for (auto &task : tasks) {
		task.Callback();
	}
}
} // namespace

struct MediaProjectState : public StateWithBlockableTasks {
	MediaProjectState(ClientContext &context_p, const PhysicalMediaProject &op)
	    : input_types(op.children[0].get().GetTypes()),
	      scalar_column(ManagedColumn(op.select_list)),
	      function(op.select_list[scalar_column]->Cast<BoundFunctionExpression>()),
	      info(*GetManagedMediaScalar(function)) {
		auto &db = DatabaseInstance::GetDatabase(context_p);
		if (!NativeMediaScalarAdmissionEnabled(context_p) || !db.GetMediaOutputPool()) {
			throw InvalidInputException("MEDIA_PROJECT requires native media scalar admission on the executing host");
		}
		if (function.children.empty() || function.children[0]->GetExpressionClass() != ExpressionClass::BOUND_REF) {
			throw NotImplementedException("Managed media image input must be a column reference");
		}
		for (idx_t i = 1; i < function.children.size(); i++) {
			if (function.children[i]->GetExpressionClass() != ExpressionClass::BOUND_CONSTANT) {
				throw NotImplementedException("Managed media dimensions and options must be bound constants");
			}
		}
		for (auto &type : input_types) {
			row_bytes = AddBytes(row_bytes, FixedBytes(type));
		}
		working_set = info.working_set(context_p, function);
		if (!working_set.output_bytes) {
			throw InvalidInputException("Managed media scalar must declare a nonempty output reservation");
		}
		auto rest = AddBytes(AddBytes(row_bytes, working_set.output_bytes), working_set.scratch_bytes);
		auto capacity = db.GetMediaOutputPool()->Snapshot().capacity.bytes;
		if (rest >= capacity || row_bytes > capacity - rest) {
			throw OutOfMemoryException("Managed media minimum working set exceeds VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES");
		}
		auto copy_limit = MinValue(capacity - rest, ClientConfig::GetConfig(context_p).streaming_buffer_size);
		prefix_rows = MinValue<idx_t>(STANDARD_VECTOR_SIZE, copy_limit / row_bytes);
		if (!prefix_rows) {
			throw OutOfMemoryException("Managed media input row exceeds streaming_buffer_size receiving capacity");
		}
		reservation_bytes = AddBytes(prefix_rows * row_bytes, rest);
		scope = db.GetMediaOutputPool()->CreateScope();
	}

	~MediaProjectState() {
		Close();
	}

	void Close() {
		unique_ptr<MediaProjectBatch> discarded;
		shared_ptr<MediaProjectBatchLifetime> active;
		vector<InterruptState> wakeups;
		{
			auto guard = Lock();
			closed = true;
			discarded = std::move(queued);
			active = active_lifetime.lock();
			PreventBlocking(guard);
			wakeups = TakeBlockedTasks(guard);
		}
		// Cancellation and receipt callbacks must never run under the rendezvous lock.
		scope->Close();
		if (active) {
			active->row_scope->Close();
			active->output_scope->Close();
		}
		discarded.reset();
		Wake(std::move(wakeups));
	}

	vector<LogicalType> input_types;
	idx_t scalar_column;
	const BoundFunctionExpression &function;
	const ManagedMediaScalarFunctionInfo &info;
	ManagedMediaWorkingSet working_set {0, 0};
	idx_t row_bytes = 0;
	idx_t prefix_rows = 0;
	idx_t reservation_bytes = 0;
	shared_ptr<MediaResourceScope> scope;
	unique_ptr<MediaProjectBatch> queued;
	weak_ptr<MediaProjectBatchLifetime> active_lifetime;
	bool occupied = false;
	bool input_finished = false;
	bool closed = false;
};

namespace {
struct MediaProjectSinkState : public GlobalSinkState {
	explicit MediaProjectSinkState(shared_ptr<MediaProjectState> state_p) : state(std::move(state_p)) {
	}
	shared_ptr<MediaProjectState> state;
};

struct MediaProjectSourceState : public GlobalSourceState {
	explicit MediaProjectSourceState(shared_ptr<MediaProjectState> state_p) : state(std::move(state_p)) {
	}
	void OnConsumerFinished() override {
		state->Close();
	}
	shared_ptr<MediaProjectState> state;
};

struct MediaProjectLocalState : public LocalSourceState {
	unique_ptr<MediaProjectBatch> batch;
	shared_ptr<MediaProjectBatchLifetime> lifetime;
	unique_ptr<MediaAdmissionRequest> output_request;
	unique_ptr<MediaAdmissionRequest> row_request;
	MediaResourceLease output_lease;
	DataChunk cache;
	idx_t prefix = 0;
	idx_t row = 0;
};

void CopyOwnedRow(MediaProjectState &state, MediaProjectLocalState &local, DataChunk &owned,
	              MediaResourceLease row_lease) {
	auto lifetime = make_shared_ptr<MediaProjectRowLifetime>(local.lifetime, std::move(row_lease));
	MediaResourcePool columns({0, state.row_bytes});
	auto scope = columns.CreateScope();
	owned.InitializeEmpty(state.input_types);
	for (idx_t column = 0; column < state.input_types.size(); column++) {
		auto &type = state.input_types[column];
		auto lease = scope->Request({0, FixedBytes(type)})->TakeGrant();
		if (type.InternalType() == PhysicalType::ARRAY) {
			AllocateMediaArray(owned.data[column], 1, std::move(lease), lifetime);
		} else {
			auto vector = AllocateMediaVector(type, 1, std::move(lease), Allocator::DefaultAllocatorReference(), nullptr,
			                                  lifetime);
			owned.data[column].Reference(*vector);
		}
		VectorOperations::Copy(local.cache.data[column], owned.data[column], local.row + 1, local.row, 0);
	}
	owned.SetCardinality(1);
}
} // namespace

PhysicalMediaProject::PhysicalMediaProject(PhysicalPlan &plan, vector<LogicalType> types,
	                                       vector<unique_ptr<Expression>> expressions, idx_t estimated_cardinality)
	    : PhysicalOperator(plan, TYPE, std::move(types), estimated_cardinality), select_list(std::move(expressions)) {
}

shared_ptr<MediaProjectState> PhysicalMediaProject::GetState(ClientContext &context) const {
	std::lock_guard<std::mutex> guard(state_lock);
	if (!state) {
		state = make_shared_ptr<MediaProjectState>(context, *this);
	}
	return state;
}

void PhysicalMediaProject::ResetExecutionState() {
	shared_ptr<MediaProjectState> previous;
	{
		std::lock_guard<std::mutex> guard(state_lock);
		previous = std::move(state);
	}
	if (previous) {
		previous->Close();
	}
}

void PhysicalMediaProject::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	ResetExecutionState();
	auto &build = meta_pipeline.GetState();
	build.SetPipelineSource(current, *this);
	auto &child = meta_pipeline.CreateChildMetaPipeline(current, *this, MetaPipelineType::REGULAR, false);
	child.Build(children[0].get());
}

unique_ptr<GlobalSinkState> PhysicalMediaProject::GetGlobalSinkState(ClientContext &context) const {
	return make_uniq<MediaProjectSinkState>(GetState(context));
}

unique_ptr<LocalSinkState> PhysicalMediaProject::GetLocalSinkState(ExecutionContext &) const {
	return make_uniq<LocalSinkState>();
}

SinkResultType PhysicalMediaProject::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &state = *input.global_state.Cast<MediaProjectSinkState>().state;
	{
		auto guard = state.Lock();
		state.RemoveBlockedTask(guard, &input.local_state);
		if (state.closed) {
			return SinkResultType::FINISHED;
		}
		if (state.occupied) {
			state.BlockTask(guard, &input.local_state, input.interrupt_state);
			return SinkResultType::BLOCKED; // Rejected: retain caller input and never ACK it.
		}
	}
	if (!chunk.size()) {
		return SinkResultType::NEED_MORE_INPUT;
	}
	auto batch = make_uniq<MediaProjectBatch>();
	batch->context = context.client.shared_from_this();
	batch->rows = chunk.size();
	for (idx_t offset = 0; offset < chunk.size();) {
		if (context.client.IsInterrupted()) {
			throw InterruptException();
		}
		auto rows = MinValue(state.prefix_rows, chunk.size() - offset);
		DataChunk selected;
		selected.InitializeEmpty(state.input_types);
		selected.Reference(chunk);
		selected.Slice(offset, rows);
		auto prefix = make_uniq<ColumnDataCollection>(context.client, state.input_types);
		prefix->Append(selected); // Local append state releases all BufferManager pins.
		batch->prefixes.push_back(std::move(prefix));
		offset += rows;
	}
	// Queue the complete working set before ACK releases upstream credits. A
	// subsequent producer cannot consume the bytes needed to execute this batch.
	batch->request = state.scope->Request({0, state.reservation_bytes});
	vector<InterruptState> wakeups;
	{
		auto guard = state.Lock();
		if (state.closed) {
			return SinkResultType::FINISHED;
		}
		state.queued = std::move(batch);
		state.occupied = true;
		wakeups = state.TakeBlockedTasks(guard);
	}
	chunk.AcknowledgeMediaFlow();
	Wake(std::move(wakeups));
	return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType PhysicalMediaProject::Finalize(Pipeline &, Event &, ClientContext &,
	                                            OperatorSinkFinalizeInput &input) const {
	auto &state = *input.global_state.Cast<MediaProjectSinkState>().state;
	vector<InterruptState> wakeups;
	{
		auto guard = state.Lock();
		state.input_finished = true;
		wakeups = state.TakeBlockedTasks(guard);
	}
	Wake(std::move(wakeups));
	return SinkFinalizeType::READY;
}

unique_ptr<GlobalSourceState> PhysicalMediaProject::GetGlobalSourceState(ClientContext &context) const {
	return make_uniq<MediaProjectSourceState>(GetState(context));
}

unique_ptr<LocalSourceState> PhysicalMediaProject::GetLocalSourceState(ExecutionContext &, GlobalSourceState &) const {
	return make_uniq<MediaProjectLocalState>();
}

SourceResultType PhysicalMediaProject::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                                   OperatorSourceInput &input) const {
	auto &state = *input.global_state.Cast<MediaProjectSourceState>().state;
	auto &local = input.local_state.Cast<MediaProjectLocalState>();
	chunk.Destroy(); // Empty-schema Reset would keep the preceding borrowed buffers alive.
	chunk.InitializeEmpty(types);
	while (true) {
		if (context.client.IsInterrupted()) {
			throw InterruptException();
		}
		{
			auto guard = state.Lock();
			state.RemoveBlockedTask(guard, &local);
			if (state.closed) {
				return SourceResultType::FINISHED;
			}
			if (!local.batch) {
				if (state.queued) {
					local.batch = std::move(state.queued);
				} else if (state.input_finished) {
					return SourceResultType::FINISHED;
				} else {
					state.BlockTask(guard, &local, input.interrupt_state);
					return SourceResultType::BLOCKED;
				}
			}
		}
		if (!local.lifetime) {
			auto status = local.batch->request->PollOrArm(input.interrupt_state);
			if (status == MediaAdmissionStatus::WAITING) {
				return SourceResultType::BLOCKED;
			}
			if (status == MediaAdmissionStatus::CANCELLED) {
				return SourceResultType::FINISHED;
			}
			local.lifetime = make_shared_ptr<MediaProjectBatchLifetime>(local.batch->request->TakeGrant(), state.row_bytes,
			                                               state.working_set.output_bytes);
			local.batch->request.reset();
			bool closed;
			{
				auto guard = state.Lock();
				closed = state.closed;
				state.active_lifetime = local.lifetime;
			}
			if (closed) {
				local.lifetime->row_scope->Close();
				local.lifetime->output_scope->Close();
				return SourceResultType::FINISHED;
			}
		}
		if (local.row == local.cache.size()) {
			local.cache.Destroy();
			if (local.prefix == local.batch->prefixes.size()) {
				local.batch.reset();
				local.lifetime.reset();
				local.prefix = 0;
				local.row = 0;
				vector<InterruptState> wakeups;
				{
					auto guard = state.Lock();
					state.occupied = false;
					wakeups = state.TakeBlockedTasks(guard);
				}
				Wake(std::move(wakeups));
				continue;
			}
			auto &prefix = local.batch->prefixes[local.prefix++];
			local.cache.Initialize(Allocator::DefaultAllocator(), state.input_types, prefix->Count());
			ColumnDataScanState scan;
			prefix->InitializeScan(scan, ColumnDataScanProperties::DISALLOW_ZERO_COPY);
			prefix->Scan(scan, local.cache);
			local.row = 0;
		}
		if (!local.output_lease) {
			if (!local.output_request) {
				local.output_request = local.lifetime->output_scope->Request({0, state.working_set.output_bytes});
			}
			auto status = local.output_request->PollOrArm(input.interrupt_state);
			if (status == MediaAdmissionStatus::WAITING) {
				return SourceResultType::BLOCKED;
			}
			if (status == MediaAdmissionStatus::CANCELLED) {
				return SourceResultType::FINISHED;
			}
			local.output_lease = local.output_request->TakeGrant();
			local.output_request.reset();
		}
		if (!local.row_request) {
			local.row_request = local.lifetime->row_scope->Request({0, state.row_bytes});
		}
		auto status = local.row_request->PollOrArm(input.interrupt_state);
		if (status == MediaAdmissionStatus::WAITING) {
			return SourceResultType::BLOCKED;
		}
		if (status == MediaAdmissionStatus::CANCELLED) {
			return SourceResultType::FINISHED;
		}
		DataChunk owned;
		CopyOwnedRow(state, local, owned, local.row_request->TakeGrant());
		local.row_request.reset();
		DataChunk args;
		args.InitializeEmpty(state.function.function.arguments);
		for (idx_t i = 0; i < state.function.children.size(); i++) {
			auto &child = *state.function.children[i];
			if (child.GetExpressionClass() == ExpressionClass::BOUND_REF) {
				args.data[i].Reference(owned.data[child.Cast<BoundReferenceExpression>().index]);
			} else {
				args.data[i].Reference(child.Cast<BoundConstantExpression>().value);
			}
		}
		args.SetCardinality(1);
		state.info.execute(context.client, args, chunk.data[state.scalar_column], std::move(local.output_lease),
		                   local.lifetime);
		for (idx_t i = 0; i < select_list.size(); i++) {
			if (i == state.scalar_column) {
				continue;
			}
			auto &expression = *select_list[i];
			if (expression.GetExpressionClass() == ExpressionClass::BOUND_REF) {
				chunk.data[i].Reference(owned.data[expression.Cast<BoundReferenceExpression>().index]);
			} else {
				chunk.data[i].Reference(expression.Cast<BoundConstantExpression>().value);
			}
		}
		chunk.SetCardinality(1);
		local.row++; // Commit once. A blocked downstream retries this output, never the scalar.
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
}

InsertionOrderPreservingMap<string> PhysicalMediaProject::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	string expressions;
	for (auto &expression : select_list) {
		if (!expressions.empty()) {
			expressions += "\n";
		}
		expressions += expression->GetName();
	}
	result["__projections__"] = expressions;
	result["media_admission"] = "fixed-shape scalar; one receiving batch; one row working set";
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

void PhysicalMediaProject::SerializeOperatorData(Serializer &serializer) const {
	serializer.WriteProperty(103, "select_list", select_list);
}

void ValidateManagedMediaPlan(ClientContext &context, LogicalOperator &root) {
	if (!NativeMediaScalarAdmissionEnabled(context)) {
		return;
	}
	idx_t calls = 0;
	std::function<void(LogicalOperator &)> visit = [&](LogicalOperator &op) {
		LogicalOperatorVisitor::EnumerateExpressions(op, [&](unique_ptr<Expression> *root_expression) {
			std::function<void(const Expression &, bool)> check = [&](const Expression &expression, bool top_level) {
				if (auto info = GetManagedMediaScalar(expression)) {
					if (!top_level || op.type != LogicalOperatorType::LOGICAL_PROJECTION) {
						throw NotImplementedException("Managed media scalar requires a top-level projection field; "
						                              "nested, conditional and filter contexts are not implemented");
					}
					if (++calls > 1) {
						throw NotImplementedException("Managed media currently supports one scalar call per query");
					}
					info->working_set(context, expression.Cast<BoundFunctionExpression>());
				}
				ExpressionIterator::EnumerateChildren(expression, [&](const Expression &child) { check(child, false); });
			};
			check(**root_expression, true);
		});
		for (auto &child : op.children) {
			visit(*child);
		}
	};
	visit(root);
}

} // namespace duckdb
