// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "direct_exchange.hpp"
#include "fragment_plan.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/pending_query_result.hpp"

namespace duckdb {
namespace vane_execution {

struct DirectInput {
	shared_ptr<DirectChannel> channel;
	string consumer;
};

struct DirectOutput {
	vector<shared_ptr<DirectChannel>> channels;
	string producer;
	// Empty means replicate to each output (one channel for GATHER). HASH is
	// evaluated by the native fragment codec, never by the transport.
	string partitioning;
};

struct DirectTaskStatus {
	string task_id;
	string state;
	string error;
	bool released = false;
};

// One query's in-process TaskService. Control is serialized separately from
// cancellation; cancel can interrupt ExecuteTask without waiting for its lock.
// This is an internal distributed-runtime facility, not a local query backend.
class DirectTaskService {
public:
	explicit DirectTaskService(shared_ptr<DatabaseInstance> database);
	~DirectTaskService();
	void Prepare(const string &task_id, const string &payload, const string &connection_snapshot,
	             const string &source_snapshot, const SourceAssignments &assignments,
	             const std::map<string, vector<DirectInput>> &inputs, const vector<DirectOutput> &outputs);
	void Start(const string &task_id, const string &token);
	idx_t Pump(idx_t steps);
	void Cancel(const string &reason);
	bool Expire();
	void Release();
	vector<DirectTaskStatus> Status();

private:
	struct Task {
		string id;
		string state = "PREPARED";
		string token;
		string error;
		bool released = false;
		unique_ptr<Connection> connection;
		FragmentSpec fragment;
		string source_snapshot;
		std::map<string, vector<DirectInput>> inputs;
		vector<DirectOutput> outputs;
		shared_ptr<PreparedStatementData> prepared;
		unique_ptr<PendingQueryResult> pending;
	};
	void CheckCanceled() const;
	void Refresh(Task &task);
	void Cleanup(Task &task);
	void Fail(Task &task, const string &message);
	bool Stop(const string &reason, bool only_running);
	shared_ptr<DatabaseInstance> database;
	mutex operation_lock;
	mutex registry_lock;
	vector<weak_ptr<ClientContext>> contexts;
	vector<shared_ptr<DirectChannel>> channels;
	vector<unique_ptr<Task>> tasks;
	atomic<bool> canceled {false};
	string cancel_reason;
	idx_t unfinished_production = 0; // Guarded by registry_lock, including deadline vs completion.
	bool started = false;
	idx_t next_task = 0;
};

} // namespace vane_execution
} // namespace duckdb
