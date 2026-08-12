// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/operator/persistent/physical_distributed_extension_write.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

namespace {

class DistributedExtensionWriteGlobalSinkState final : public GlobalSinkState {
public:
	shared_ptr<const DistributedWriteOperatorExtension> write_operator;
	unique_ptr<DistributedWriteGlobalState> extension_state;
	DistributedWriteTaskResult result;
	bool finalized = false;
};

class DistributedExtensionWriteLocalSinkState final : public LocalSinkState {
public:
	explicit DistributedExtensionWriteLocalSinkState(unique_ptr<DistributedWriteLocalState> extension_state_p)
	    : extension_state(std::move(extension_state_p)) {
	}

	unique_ptr<DistributedWriteLocalState> extension_state;
};

class DistributedExtensionWriteGlobalSourceState final : public GlobalSourceState {
public:
	idx_t MaxThreads() override {
		return 1;
	}

	bool emitted = false;
};

static idx_t ApplyTaskContext(PhysicalOperator &op, const DistributedWriteTaskContext &task_context) {
	idx_t result = 0;
	if (op.type == PhysicalOperatorType::DISTRIBUTED_EXTENSION_WRITE) {
		auto *write = dynamic_cast<PhysicalDistributedExtensionWrite *>(&op);
		if (!write) {
			throw InternalException("DISTRIBUTED_EXTENSION_WRITE has an unexpected physical implementation");
		}
		write->ApplyRuntimeTaskContext(task_context);
		result++;
	}
	for (auto &child : op.children) {
		result += ApplyTaskContext(child.get(), task_context);
	}
	return result;
}

static idx_t CountDistributedWriteOperators(const PhysicalOperator &op) {
	idx_t result = op.type == PhysicalOperatorType::DISTRIBUTED_EXTENSION_WRITE ? 1 : 0;
	for (const auto &child : op.children) {
		result += CountDistributedWriteOperators(child.get());
	}
	return result;
}

static void ValidateExistingTaskContext(const PhysicalOperator &op, const DistributedWriteTaskContext &task_context) {
	if (op.type == PhysicalOperatorType::DISTRIBUTED_EXTENSION_WRITE) {
		auto *write = dynamic_cast<const PhysicalDistributedExtensionWrite *>(&op);
		if (!write) {
			throw InternalException("DISTRIBUTED_EXTENSION_WRITE has an unexpected physical implementation");
		}
		const auto &existing = write->task_context;
		if ((!existing.operation_id.empty() || !existing.task_attempt_id.empty()) &&
		    (existing.operation_id != task_context.operation_id ||
		     existing.task_attempt_id != task_context.task_attempt_id)) {
			throw InvalidInputException("distributed extension write '%s' runtime task context cannot change",
			                            write->info.Name());
		}
	}
	for (const auto &child : op.children) {
		ValidateExistingTaskContext(child.get(), task_context);
	}
}

} // namespace

PhysicalDistributedExtensionWrite::PhysicalDistributedExtensionWrite(PhysicalPlan &physical_plan,
                                                                     DistributedExtensionWriteInfo info_p,
                                                                     idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::DISTRIBUTED_EXTENSION_WRITE, {LogicalType::BLOB},
                       estimated_cardinality),
      info(std::move(info_p)) {
	info.Validate();
	if (info.mode != DistributedWriteMode::CALLBACK) {
		throw InternalException("PhysicalDistributedExtensionWrite requires callback mode");
	}
}

unique_ptr<GlobalSinkState> PhysicalDistributedExtensionWrite::GetGlobalSinkState(ClientContext &context) const {
	task_context.Validate();
	auto write_operator = DistributedExtensionManager::Get(context).GetWriteOperator(info.capability);
	write_operator->Validate(info.capability.CanonicalIdentity());
	if (write_operator->mode != DistributedWriteMode::CALLBACK ||
	    write_operator->fragment_codec != info.fragment_codec) {
		throw InvalidInputException("distributed extension write '%s' worker contract does not match the plan",
		                            info.Name());
	}
	auto result = make_uniq<DistributedExtensionWriteGlobalSinkState>();
	result->write_operator = std::move(write_operator);
	result->extension_state = result->write_operator->callbacks.initialize_global(context, info, task_context);
	if (!result->extension_state) {
		throw InvalidInputException("distributed extension write '%s' initialize_global returned null", info.Name());
	}
	return std::move(result);
}

unique_ptr<LocalSinkState> PhysicalDistributedExtensionWrite::GetLocalSinkState(ExecutionContext &context) const {
	if (!sink_state) {
		throw InternalException("distributed extension write '%s' has no global sink state", info.Name());
	}
	auto &global = sink_state->Cast<DistributedExtensionWriteGlobalSinkState>();
	auto local =
	    global.write_operator->callbacks.initialize_local(context, info, task_context, *global.extension_state);
	if (!local) {
		throw InvalidInputException("distributed extension write '%s' initialize_local returned null", info.Name());
	}
	return make_uniq<DistributedExtensionWriteLocalSinkState>(std::move(local));
}

SinkResultType PhysicalDistributedExtensionWrite::Sink(ExecutionContext &context, DataChunk &chunk,
                                                       OperatorSinkInput &input) const {
	auto &global = input.global_state.Cast<DistributedExtensionWriteGlobalSinkState>();
	auto &local = input.local_state.Cast<DistributedExtensionWriteLocalSinkState>();
	global.write_operator->callbacks.sink(context, info, task_context, *global.extension_state, *local.extension_state,
	                                      chunk);
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalDistributedExtensionWrite::Combine(ExecutionContext &context,
                                                                 OperatorSinkCombineInput &input) const {
	auto &global = input.global_state.Cast<DistributedExtensionWriteGlobalSinkState>();
	auto &local = input.local_state.Cast<DistributedExtensionWriteLocalSinkState>();
	global.write_operator->callbacks.combine(context, info, task_context, *global.extension_state,
	                                         *local.extension_state);
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalDistributedExtensionWrite::Finalize(Pipeline &, Event &, ClientContext &context,
                                                             OperatorSinkFinalizeInput &input) const {
	auto &global = input.global_state.Cast<DistributedExtensionWriteGlobalSinkState>();
	if (global.finalized) {
		throw InternalException("distributed extension write '%s' finalized more than once", info.Name());
	}
	global.result.capability = info.capability;
	global.result.fragment_codec = info.fragment_codec;
	global.result.operation_id = task_context.operation_id;
	global.result.task_attempt_id = task_context.task_attempt_id;
	global.result.fragments =
	    global.write_operator->callbacks.finalize(context, info, task_context, *global.extension_state);
	global.result.Validate();
	global.finalized = true;
	return SinkFinalizeType::READY;
}

unique_ptr<GlobalSourceState> PhysicalDistributedExtensionWrite::GetGlobalSourceState(ClientContext &) const {
	return make_uniq<DistributedExtensionWriteGlobalSourceState>();
}

SourceResultType PhysicalDistributedExtensionWrite::GetDataInternal(ExecutionContext &, DataChunk &chunk,
                                                                    OperatorSourceInput &input) const {
	auto &source = input.global_state.Cast<DistributedExtensionWriteGlobalSourceState>();
	if (source.emitted) {
		return SourceResultType::FINISHED;
	}
	if (!sink_state) {
		throw InternalException("distributed extension write '%s' has no finalized sink state", info.Name());
	}
	auto &global = sink_state->Cast<DistributedExtensionWriteGlobalSinkState>();
	if (!global.finalized) {
		throw InternalException("distributed extension write '%s' source ran before sink finalization", info.Name());
	}
	auto bytes = global.result.SerializeToBytes();
	chunk.SetValue(0, 0, Value::BLOB(reinterpret_cast<const_data_ptr_t>(bytes.data()), bytes.size()));
	chunk.SetCardinality(1);
	source.emitted = true;
	return SourceResultType::HAVE_MORE_OUTPUT;
}

void PhysicalDistributedExtensionWrite::ApplyRuntimeTaskContext(DistributedWriteTaskContext context) {
	context.Validate();
	if ((!task_context.operation_id.empty() || !task_context.task_attempt_id.empty()) &&
	    (task_context.operation_id != context.operation_id ||
	     task_context.task_attempt_id != context.task_attempt_id)) {
		throw InvalidInputException("distributed extension write '%s' runtime task context cannot change", info.Name());
	}
	task_context = std::move(context);
}

InsertionOrderPreservingMap<string> PhysicalDistributedExtensionWrite::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Write"] = info.Name();
	result["Capability"] = info.capability.CanonicalIdentity();
	result["Fragment Codec"] = info.fragment_codec.CanonicalIdentity();
	return result;
}

void PhysicalDistributedExtensionWrite::SerializeOperatorData(Serializer &serializer) const {
	serializer.WriteObject(103, "distributed_write_info", [&](Serializer &object) { info.Serialize(object); });
	// Runtime identities are intentionally never transported in a cached plan
	// template. The selected task attempt installs them immediately before
	// worker execution.
	serializer.WriteProperty(104, "operation_id", string());
	serializer.WriteProperty(105, "task_attempt_id", string());
}

idx_t ValidateDistributedWriteTaskContextAssignment(const PhysicalPlan &plan,
                                                    const DistributedWriteTaskContext &task_context) {
	if (!plan.HasRoot()) {
		return 0;
	}
	const auto write_count = CountDistributedWriteOperators(plan.Root());
	if (write_count > 1) {
		throw InvalidInputException("worker plan contains more than one distributed extension write operator");
	}
	if (write_count == 1) {
		task_context.Validate();
		ValidateExistingTaskContext(plan.Root(), task_context);
	}
	return write_count;
}

idx_t ApplyDistributedWriteTaskContext(PhysicalPlan &plan, const DistributedWriteTaskContext &task_context) {
	const auto write_count = ValidateDistributedWriteTaskContextAssignment(plan, task_context);
	if (write_count == 0) {
		return 0;
	}
	return ApplyTaskContext(plan.Root(), task_context);
}

} // namespace duckdb
