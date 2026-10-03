// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "fragment_plan.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/pyresult.hpp"
#include "vane_python/python_conversion.hpp"

#include <pybind11/stl.h>
#include <set>

namespace duckdb {
namespace {

using namespace vane_execution;

std::unique_lock<std::recursive_mutex> LockPlanningConnection(DuckDBPyConnection &connection) {
	py::gil_scoped_release release;
	return std::unique_lock<std::recursive_mutex>(*connection.py_connection_lock);
}

py::dict Port(const string &id, const vector<LogicalType> &types) {
	py::dict result;
	result["port_id"] = id;
	result["schema"] = py::bytes(SerializeSchema(types));
	return result;
}

void InputPorts(const PlanNode &node, py::list &result) {
	if (!node.input_port.empty()) {
		result.append(Port(node.input_port, node.types));
	}
	for (auto &child : node.children) {
		InputPorts(child, result);
	}
}

py::list DescribeSources(const vector<SourceSpec> &specifications) {
	py::list sources;
	for (auto &source : specifications) {
		py::dict spec;
		spec["source_id"] = source.source_id;
		spec["function_name"] = source.function_name;
		spec["capability"] = source.capability;
		spec["codec"] = source.codec;
		spec["requires_snapshot"] = source.requires_snapshot;
		py::list splits;
		for (auto &split : source.splits) {
			py::dict item;
			item["split_id"] = split.split_id;
			item["payload"] = py::bytes(split.payload);
			splits.append(item);
		}
		spec["splits"] = splits;
		sources.append(spec);
	}
	return sources;
}

py::dict DescribeFragment(const FragmentSpec &fragment) {
	py::dict result;
	result["fragment_id"] = fragment.fragment_id;
	result["native_plan"] = py::bytes(fragment.Serialize());
	result["partition_count"] = fragment.partition_count;
	result["names"] = fragment.names;
	py::list inputs;
	InputPorts(fragment.root, inputs);
	result["inputs"] = inputs;
	py::list outputs;
	outputs.append(Port("out", fragment.root.types));
	result["outputs"] = outputs;
	result["sources"] = DescribeSources(fragment.sources);
	result["source_dependencies"] = DescribeSources(fragment.source_dependencies);
	return result;
}

py::dict DescribeGraph(const FragmentGraph &graph) {
	py::dict result;
	result["query_id"] = graph.query_id;
	result["engine_identity"] = EngineIdentity();
	py::list fragments;
	for (auto &fragment : graph.fragments) {
		fragments.append(DescribeFragment(fragment));
	}
	result["fragments"] = fragments;
	py::list exchanges;
	for (auto &exchange : graph.exchanges) {
		py::dict spec;
		spec["exchange_id"] = exchange.exchange_id;
		spec["producer_fragment_id"] = exchange.producer;
		spec["producer_port"] = "out";
		spec["consumer_fragment_id"] = exchange.consumer;
		spec["consumer_port"] = "in";
		spec["distribution"] = exchange.distribution;
		spec["partitioning"] =
		    exchange.partitioning.empty() ? py::object(py::none()) : py::object(py::bytes(exchange.partitioning));
		exchanges.append(spec);
	}
	result["exchanges"] = exchanges;
	py::dict root;
	root["fragment_id"] = graph.fragments.back().fragment_id;
	root["output_port"] = "out";
	result["result"] = root;
	return result;
}

py::dict CompileGraph(DuckDBPyConnection &connection, const string &sql, const string &query_id, idx_t partitions,
                      const vector<idx_t> &hash_columns) {
	auto lock = LockPlanningConnection(connection);
	auto &context = *connection.con.GetConnection().context;
	FragmentGraph graph;
	{
		py::gil_scoped_release release;
		graph = Compile(context, sql, query_id, partitions, hash_columns);
	}
	return DescribeGraph(graph);
}

py::dict CompileSubmission(DuckDBPyConnection &connection, const string &sql, const string &query_id, idx_t partitions,
                           const vector<idx_t> &hash_columns, bool require_replay) {
	auto lock = LockPlanningConnection(connection);
	auto &context = *connection.con.GetConnection().context;
	FragmentGraph graph;
	string snapshot;
	vector<string> sources;
	{
		py::gil_scoped_release release;
		snapshot = CaptureConnection(context);
		graph = Compile(context, sql, query_id, partitions, hash_columns);
		for (auto &fragment : graph.fragments) {
			sources.push_back(CaptureSources(context, fragment, require_replay));
		}
		if (CaptureConnection(context) != snapshot) {
			throw InvalidInputException("connection settings changed during submission compilation");
		}
	}
	py::dict result;
	result["graph"] = DescribeGraph(graph);
	result["connection_snapshot"] = py::bytes(snapshot);
	py::dict source_snapshots;
	for (idx_t i = 0; i < graph.fragments.size(); i++) {
		source_snapshots[py::str(graph.fragments[i].fragment_id)] = py::bytes(sources[i]);
	}
	result["source_snapshots"] = source_snapshots;
	result["result_names"] = graph.fragments.back().names;
	return result;
}

py::dict CompilerCapabilities(DuckDBPyConnection &connection) {
	auto lock = LockPlanningConnection(connection);
	vector<std::pair<string, string>> scans;
	{
		py::gil_scoped_release release;
		scans = ScanCapabilities(*connection.con.GetConnection().context);
	}
	py::dict result;
	result["engine_identity"] = EngineIdentity();
	result["protocol_version"] = 1;
	result["type_profile"] = "vane.basic-types:1";
	result["connection_profile"] = "vane.builtin-session:1";
	result["distributions"] = vector<string> {"gather", "hash"};
	result["scans"] = scans;
	return result;
}

// Validation can reconstruct a plan with unbound inputs, but can never execute
// it: every such input fails explicitly if an executor is accidentally started.
class UnboundFragmentInput : public PhysicalOperator {
public:
	UnboundFragmentInput(PhysicalPlan &plan, const vector<LogicalType> &types)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, types, 0) {
	}
	string GetName() const override {
		return "UNBOUND_FRAGMENT_INPUT";
	}
	bool IsSource() const override {
		return true;
	}
	SourceResultType GetDataInternal(ExecutionContext &, DataChunk &, OperatorSourceInput &) const override {
		throw InvalidInputException("fragment input has not been bound to an exchange reader");
	}
};

py::dict InspectFragment(DuckDBPyConnection &connection, const string &payload) {
	auto lock = LockPlanningConnection(connection);
	auto &context = *connection.con.GetConnection().context;
	FragmentSpec fragment;
	{
		py::gil_scoped_release release;
		fragment = FragmentSpec::Deserialize(payload);
		SourceAssignments assignments;
		for (auto &source : fragment.sources) {
			assignments.emplace(source.source_id, vector<string>());
		}
		auto plan = Load(
		    context, fragment,
		    [](PhysicalPlan &owner, const string &, const vector<LogicalType> &types) -> PhysicalOperator & {
			    return owner.Make<UnboundFragmentInput>(types);
		    },
		    assignments);
	}
	return DescribeFragment(fragment);
}

py::dict InspectSubmittedFragment(DuckDBPyConnection &connection, const string &payload, const string &snapshot,
                                  const string &sources, bool require_replay) {
	auto lock = LockPlanningConnection(connection);
	auto &context = *connection.con.GetConnection().context;
	{
		py::gil_scoped_release release;
		auto fragment = FragmentSpec::Deserialize(payload);
		ApplyConnection(context, snapshot);
		ValidateSources(context, fragment, sources, require_replay);
	}
	return InspectFragment(connection, payload);
}

unique_ptr<ColumnDataCollection> RowsToCollection(ClientContext &context, const vector<LogicalType> &types,
                                                  const py::list &rows) {
	auto collection = make_uniq<ColumnDataCollection>(context, types);
	ColumnDataAppendState state;
	collection->InitializeAppend(state);
	DataChunk chunk;
	chunk.Initialize(Allocator::Get(context), types);
	for (auto row_handle : rows) {
		if (!py::isinstance<py::tuple>(row_handle) && !py::isinstance<py::list>(row_handle)) {
			throw InvalidInputException("validation input rows must be lists or tuples");
		}
		auto row = py::reinterpret_borrow<py::sequence>(row_handle);
		if (row.size() != types.size()) {
			throw InvalidInputException("validation input row width does not match fragment schema");
		}
		const auto index = chunk.size();
		for (idx_t column = 0; column < types.size(); column++) {
			chunk.SetValue(column, index, TransformPythonValue(row[column], types[column]));
		}
		chunk.SetCardinality(index + 1);
		if (chunk.size() == STANDARD_VECTOR_SIZE) {
			collection->Append(state, chunk);
			chunk.Reset();
		}
	}
	if (chunk.size()) {
		collection->Append(state, chunk);
	}
	return collection;
}

// Finite, materialized execution for native compiler tests only. Production
// local queries and the future TaskRuntime do not call this function.
py::list ExecuteFragmentForTest(DuckDBPyConnection &connection, const string &payload, const py::dict &input_rows,
                                const py::dict &source_assignments) {
	auto lock = LockPlanningConnection(connection);
	auto &context = *connection.con.GetConnection().context;
	auto fragment = FragmentSpec::Deserialize(payload);
	SourceAssignments assignments;
	for (auto item : source_assignments) {
		assignments.emplace(py::cast<string>(item.first), py::cast<vector<string>>(item.second));
	}
	std::set<string> used_inputs;
	auto plan = Load(
	    context, fragment,
	    [&](PhysicalPlan &owner, const string &port, const vector<LogicalType> &types) -> PhysicalOperator & {
		    if (!input_rows.contains(py::str(port))) {
			    throw InvalidInputException("missing fragment input %s", port);
		    }
		    used_inputs.insert(port);
		    auto collection = RowsToCollection(context, types, input_rows[py::str(port)].cast<py::list>());
		    auto count = collection->Count();
		    return owner.Make<PhysicalColumnDataScan>(types, PhysicalOperatorType::COLUMN_DATA_SCAN, count,
		                                              std::move(collection));
	    },
	    assignments);
	if (used_inputs.size() != input_rows.size()) {
		throw InvalidInputException("unexpected fragment input");
	}
	auto prepared = make_shared_ptr<PreparedStatementData>(StatementType::SELECT_STATEMENT);
	prepared->names = fragment.names;
	prepared->types = fragment.root.types;
	prepared->physical_plan = std::move(plan);
	prepared->properties.return_type = StatementReturnType::QUERY_RESULT;
	prepared->output_type = QueryResultOutputType::FORCE_MATERIALIZED;
	prepared->memory_type = QueryResultMemoryType::IN_MEMORY;
	unique_ptr<QueryResult> result;
	{
		py::gil_scoped_release release;
		PendingQueryParameters parameters;
		result = context.ExecutePreparedStatementNoRebind("native fragment compiler validation", prepared, parameters);
	}
	if (result->HasError()) {
		result->ThrowError();
	}
	DuckDBPyResult rows(std::move(result));
	return rows.Fetchall();
}

vector<idx_t> HashRowsForTest(DuckDBPyConnection &connection, const string &schema, const string &partitioning,
                              const py::list &rows, idx_t partitions) {
	auto lock = LockPlanningConnection(connection);
	auto &context = *connection.con.GetConnection().context;
	auto collection = RowsToCollection(context, DeserializeSchema(schema), rows);
	ColumnDataScanState state;
	collection->InitializeScan(state);
	DataChunk chunk;
	collection->InitializeScanChunk(chunk);
	vector<idx_t> result;
	while (collection->Scan(state, chunk)) {
		auto ids = HashPartitions(context, partitioning, chunk, partitions);
		result.insert(result.end(), ids.begin(), ids.end());
	}
	return result;
}

void ValidateHash(DuckDBPyConnection &connection, const string &schema, const string &partitioning) {
	auto lock = LockPlanningConnection(connection);
	auto &context = *connection.con.GetConnection().context;
	DataChunk chunk;
	chunk.Initialize(Allocator::Get(context), DeserializeSchema(schema));
	HashPartitions(context, partitioning, chunk, 1);
}

} // namespace

void RegisterExecutionPlanBindings(py::module_ &module) {
	auto execution = module.def_submodule("execution_plan", "Native distributed fragment compiler and loader");
	execution.def("engine_identity", &vane_execution::EngineIdentity);
	execution.def("compile", &CompileGraph, py::arg("connection"), py::arg("sql"), py::arg("query_id"),
	              py::arg("partition_count"), py::arg("hash_columns"));
	execution.def("compile_submission", &CompileSubmission, py::arg("connection"), py::arg("sql"), py::arg("query_id"),
	              py::arg("partition_count"), py::arg("hash_columns"), py::arg("require_replay"));
	execution.def("compiler_capabilities", &CompilerCapabilities, py::arg("connection"));
	execution.def("inspect_submitted_fragment", &InspectSubmittedFragment, py::arg("connection"), py::arg("payload"),
	              py::arg("connection_snapshot"), py::arg("source_snapshot"), py::arg("require_replay"));
	execution.def("inspect_fragment", &InspectFragment, py::arg("connection"), py::arg("payload"));
	execution.def("validate_hash", &ValidateHash, py::arg("connection"), py::arg("schema"), py::arg("partitioning"));
	execution.def("_execute_fragment_for_test", &ExecuteFragmentForTest, py::arg("connection"), py::arg("payload"),
	              py::arg("inputs"), py::arg("source_assignments"));
	execution.def("_hash_rows_for_test", &HashRowsForTest, py::arg("connection"), py::arg("schema"),
	              py::arg("partitioning"), py::arg("rows"), py::arg("partition_count"));
}

} // namespace duckdb
