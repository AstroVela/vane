// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/serializer/memory_stream.hpp"
#include "duckdb/execution/external_block.hpp"
#include "duckdb/execution/operator/persistent/physical_copy_to_file.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table/datasource_scan.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/materialized_query_result.hpp"
#include "duckdb/main/prepared_statement.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/parser/statement/copy_statement.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

using namespace duckdb;

namespace {

// Restore source bind data using this DSO's deserializer and scan callbacks.
// Merely planning a catalog function would keep its host-side function pointers
// and would not exercise private engine callback lookup.
void RestoreDataSources(ClientContext &context, LogicalOperator &op) {
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		if (get.function.name == "datasource_scan") {
			MemoryStream stream;
			BinarySerializer serializer(stream);
			serializer.Begin();
			get.function.serialize(serializer, get.bind_data.get(), get.function);
			serializer.End();
			stream.Rewind();
			get.function = DataSourceScanFunction::GetFunction();
			BinaryDeserializer deserializer(stream);
			deserializer.Set<ClientContext &>(context);
			deserializer.Begin();
			get.bind_data = get.function.deserialize(deserializer, get.function);
			deserializer.End();
		}
	}
	for (auto &child : op.children) {
		RestoreDataSources(context, *child);
	}
}

// Model COPY extensions that bind their input and recursively plan it from
// their self-contained DSO. Return the input rows to keep the fixture stateless.
class BatchDemoLogicalCopy : public LogicalExtensionOperator {
public:
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override {
		RestoreDataSources(context, *children[0]);
		return planner.CreatePlan(*children[0]);
	}

	vector<ColumnBinding> GetColumnBindings() override {
		return children[0]->GetColumnBindings();
	}

	void ResolveTypes() override {
		types = children[0]->types;
	}
};

BoundStatement PlanBatchDemoCopy(Binder &binder, CopyStatement &statement) {
	auto input = binder.Bind(*statement.info->select_statement->Copy());
	auto copy = make_uniq<BatchDemoLogicalCopy>();
	copy->children.push_back(std::move(input.plan));
	input.plan = std::move(copy);
	return input;
}

struct BatchDemoQueryData : public TableFunctionData {
	string query;
};

unique_ptr<FunctionData> BindBatchDemoQuery(ClientContext &context, TableFunctionBindInput &input,
                                            vector<LogicalType> &types, vector<string> &names) {
	// Unlike the COPY path, this creates both the connection and its planner in
	// the private engine. Virtual dispatch on a host planner cannot help here.
	Connection inner(*context.db);
	auto data = make_uniq<BatchDemoQueryData>();
	data->query = input.inputs[0].GetValue<string>();
	auto prepared = inner.Prepare(data->query);
	if (prepared->HasError()) {
		throw InvalidInputException(prepared->GetError());
	}
	types = prepared->GetTypes();
	names = prepared->GetNames();
	return std::move(data);
}

struct BatchDemoQueryState : public GlobalTableFunctionState {
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataScanState scan;
};

unique_ptr<GlobalTableFunctionState> InitBatchDemoQuery(ClientContext &context, TableFunctionInitInput &input) {
	// Binding may run more than once (for example, for a parameterized query).
	// Execute only after the outer query has entered its execution scope.
	Connection inner(*context.db);
	auto result = inner.Query(input.bind_data->Cast<BatchDemoQueryData>().query);
	if (result->HasError()) {
		result->ThrowError();
	}
	auto state = make_uniq<BatchDemoQueryState>();
	state->collection = result->TakeCollection();
	state->collection->InitializeScan(state->scan);
	return std::move(state);
}

void ReadBatchDemoQuery(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &state = input.global_state->Cast<BatchDemoQueryState>();
	state.collection->Scan(state.scan, output);
}

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
	CopyFunction copy("batch_demo_replan");
	copy.plan = PlanBatchDemoCopy;
	loader.RegisterFunction(copy);
	loader.RegisterFunction(TableFunction("batch_demo_query", {LogicalType::VARCHAR}, ReadBatchDemoQuery,
	                                      BindBatchDemoQuery, InitBatchDemoQuery));
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
