// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/*
 * Minimal, self-contained test helpers for distributed execution tests.
 *
 * Provides lightweight mock types that satisfy the interfaces used by the
 * local distributed tests. The goal is not to be feature-complete but to
 * implement the small subset of behavior required by the unit tests in
 * src/execution/distributed/tests/*.cpp.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/scheduling/task.hpp"
#include "duckdb/execution/distributed/scheduling/worker.hpp"

namespace duckdb {
namespace distributed {
namespace testing {

using ::duckdb::distributed::TaskResourceRequest;
using ::duckdb::distributed::WorkerId;
using ::duckdb::distributed::WorkerSnapshot;

//-----------------------------------------------------------------------------
// Mock Worker
//-----------------------------------------------------------------------------

class MockWorker {
public:
	MockWorker(WorkerId id, double cpus, double gpus) : id_(std::move(id)), cpus_(cpus), gpus_(gpus) {
	}
	const WorkerId &id() const {
		return id_;
	}
	double total_num_cpus() const {
		return cpus_;
	}
	double total_num_gpus() const {
		return gpus_;
	}

private:
	WorkerId id_;
	double cpus_ = 0.0;
	double gpus_ = 0.0;
};

// Return an unordered_map of MockWorker given simple (WorkerId, slots) configs
inline std::unordered_map<WorkerId, MockWorker, WorkerIdHash, WorkerIdEqual>
setup_workers(const std::vector<std::pair<WorkerId, size_t>> &configs) {
	std::unordered_map<WorkerId, MockWorker, WorkerIdHash, WorkerIdEqual> workers;
	for (const auto &cfg : configs) {
		workers.emplace(cfg.first, MockWorker(cfg.first, static_cast<double>(cfg.second), 0.0));
	}
	return workers;
}

//-----------------------------------------------------------------------------
// Mock WorkerManager
//-----------------------------------------------------------------------------

class MockWorkerManager : public ::duckdb::distributed::WorkerManager {
public:
	explicit MockWorkerManager(std::unordered_map<WorkerId, MockWorker, WorkerIdHash, WorkerIdEqual> workers)
	    : workers_(std::move(workers)) {
	}

	DuckDBResult<std::vector<WorkerSnapshot>> worker_snapshots() const override {
		std::vector<WorkerSnapshot> snaps;
		for (const auto &kv : workers_) {
			const auto &worker = kv.second;
			snaps.emplace_back(worker.id(), worker.total_num_cpus(), worker.total_num_gpus());
		}
		return DuckDBResult<std::vector<WorkerSnapshot>>::ok(std::move(snaps));
	}

	DuckDBResult<void> try_autoscale(const std::vector<TaskResourceRequest> &reqs) override {
		return DuckDBResult<void>::ok();
	}

	DuckDBResult<void> shutdown() override {
		return DuckDBResult<void>::ok();
	}

private:
	std::unordered_map<WorkerId, MockWorker, WorkerIdHash, WorkerIdEqual> workers_;
};

} // namespace testing
} // namespace distributed
} // namespace duckdb
