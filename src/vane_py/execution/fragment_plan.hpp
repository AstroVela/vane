// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/distributed_table_function.hpp"

#include <functional>

namespace duckdb {
namespace vane_execution {

// These are compiler/loader contracts. They do not own tasks, connections,
// exchange endpoints, attempt output locations, or a recovery policy.
struct SourceSpec {
	string source_id;
	string function_name;
	string capability;
	string codec;
	vector<DistributedScanSplit> splits;
	bool requires_snapshot = false;
	void Serialize(Serializer &serializer) const;
	static SourceSpec Deserialize(Deserializer &deserializer);
};

struct PlanNode {
	string input_port;
	vector<LogicalType> types;
	string native_operator;
	string source_id;
	vector<PlanNode> children;
	void Serialize(Serializer &serializer) const;
	static PlanNode Deserialize(Deserializer &deserializer);
};

struct FragmentSpec {
	string fragment_id;
	idx_t partition_count = 1;
	vector<string> names;
	PlanNode root;
	vector<SourceSpec> sources;
	// Bound file dependencies survive scan/statistics pruning and need no task
	// assignments. Their source codec still describes the original file set.
	vector<SourceSpec> source_dependencies;
	string Serialize() const;
	static FragmentSpec Deserialize(const string &payload);
};

struct ExchangeSpec {
	string exchange_id;
	string producer;
	string consumer;
	string distribution;
	string partitioning;
};

struct FragmentGraph {
	string query_id;
	vector<FragmentSpec> fragments;
	vector<ExchangeSpec> exchanges;
};

string EngineIdentity();
string SerializeSchema(const vector<LogicalType> &types);
vector<LogicalType> DeserializeSchema(const string &payload);

// Submission snapshots cover the supported built-in SQL profile. They do not
// transport runner configuration, credentials, Python state or attached DBs.
string CaptureConnection(ClientContext &context);
void ApplyConnection(ClientContext &context, const string &snapshot);
string CaptureSources(ClientContext &context, const FragmentSpec &fragment, bool require_replay);
void ValidateSources(ClientContext &context, const FragmentSpec &fragment, const string &snapshot, bool require_replay);
vector<std::pair<string, string>> ScanCapabilities(ClientContext &context);

// hash_columns is an explicit physical output-distribution request, expressed
// as result-column positions. Native code binds and serializes the expressions.
FragmentGraph Compile(ClientContext &context, const string &sql, const string &query_id, idx_t partitions,
                      const vector<idx_t> &hash_columns, const string &snapshot_directory = "", idx_t source_budget = 0,
                      idx_t *source_bytes = nullptr);

using InputFactory = std::function<PhysicalOperator &(PhysicalPlan &, const string &, const vector<LogicalType> &)>;
using SourceAssignments = unordered_map<string, vector<string>>;

// All input ports and source assignments must be bound explicitly. Loading is
// separate from execution and never creates an executor or initializes a scan.
unique_ptr<PhysicalPlan> Load(ClientContext &context, const FragmentSpec &fragment, const InputFactory &inputs,
                              const SourceAssignments &assignments);
vector<idx_t> HashPartitions(ClientContext &context, const string &partitioning, DataChunk &chunk, idx_t partitions);

} // namespace vane_execution
} // namespace duckdb
