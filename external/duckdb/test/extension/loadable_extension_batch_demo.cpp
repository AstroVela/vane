// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/external_block.hpp"
#include "duckdb/execution/operator/persistent/physical_copy_to_file.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

using namespace duckdb;

namespace {

void Observe(DataChunk &chunk, vector<int64_t> &values) {
	for (idx_t row = 0; row < chunk.size(); row++) {
		values.push_back(chunk.GetValue(0, row).GetValue<int64_t>());
	}
}

class BatchDemoSink : public PhysicalOperator {
public:
	BatchDemoSink(PhysicalPlan &plan, vector<int64_t> &values_p)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, {}, 3), values(values_p) {
	}

	bool IsSink() const override {
		return true;
	}

	SinkResultType Sink(ExecutionContext &, DataChunk &, OperatorSinkInput &) const override;

private:
	vector<int64_t> &values;
	mutable idx_t calls = 0;
};

SinkResultType BatchDemoSink::Sink(ExecutionContext &, DataChunk &chunk, OperatorSinkInput &) const {
	if (++calls <= 2) {
		return SinkResultType::BLOCKED;
	}
	Observe(chunk, values);
	return SinkResultType::NEED_MORE_INPUT;
}

class BatchDemoOperator : public PhysicalOperator {
public:
	BatchDemoOperator(PhysicalPlan &plan, vector<int64_t> &values_p)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, {LogicalType::BIGINT}, 3), values(values_p) {
	}

	OperatorResultType Execute(ExecutionContext &, DataChunk &input, DataChunk &output, GlobalOperatorState &,
	                           OperatorState &) const override {
		Observe(input, values);
		output.Reference(input);
		return ++calls <= 2 ? OperatorResultType::HAVE_MORE_OUTPUT : OperatorResultType::NEED_MORE_INPUT;
	}

private:
	vector<int64_t> &values;
	mutable idx_t calls = 0;
};

struct BatchDemoCopyData : public TableFunctionData {
	explicit BatchDemoCopyData(vector<int64_t> &values_p) : values(values_p) {
	}
	vector<int64_t> &values;
};

PhysicalOperator *MakeCopy(PhysicalPlan &plan, vector<int64_t> &values) {
	CopyFunction function("batch_demo");
	function.copy_to_initialize_global = [](ClientContext &, FunctionData &, const string &) {
		return make_uniq<GlobalFunctionData>();
	};
	function.copy_to_initialize_local = [](ExecutionContext &, FunctionData &) {
		return make_uniq<LocalFunctionData>();
	};
	function.copy_to_sink = [](ExecutionContext &, FunctionData &bind, GlobalFunctionData &, LocalFunctionData &,
	                           DataChunk &chunk) {
		Observe(chunk, bind.Cast<BatchDemoCopyData>().values);
	};
	auto result = new PhysicalCopyToFile(plan, {LogicalType::BIGINT}, std::move(function),
	                                     make_uniq<BatchDemoCopyData>(values), 3);
	result->use_tmp_file = false;
	result->per_thread_output = false;
	result->partition_output = false;
	result->write_partition_columns = false;
	result->write_empty_file = false;
	result->hive_file_pattern = false;
	result->rotate = false;
	result->return_type = CopyFunctionReturnType::CHANGED_ROWS;
	result->overwrite_mode = CopyOverwriteMode::COPY_OVERWRITE_OR_IGNORE;
	return result;
}

} // namespace

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(loadable_extension_batch_demo, loader) {
	// A real loadable artifact, deliberately without a private reader registration.
	(void)loader;
}

DUCKDB_EXTENSION_API bool batch_demo_has_private_reader() {
	return GetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE) != nullptr;
}

// Test-only factories put the operator vtable and inherited adapters in this DSO.
DUCKDB_EXTENSION_API PhysicalOperator *batch_demo_make_operator(PhysicalPlan &plan, idx_t kind,
                                                                vector<int64_t> &values) {
	if (kind == 0) {
		return new BatchDemoSink(plan, values);
	}
	if (kind == 1) {
		return new BatchDemoOperator(plan, values);
	}
	return MakeCopy(plan, values);
}
}
