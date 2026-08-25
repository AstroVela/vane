// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace duckdb {
namespace distributed {
namespace python {
namespace ray {

class QueryLifecycleCoordinator {
public:
	explicit QueryLifecycleCoordinator(std::string component_name);

	struct LifecycleRef {
		std::string owner_query_id;
		uint64_t generation = 0;

		explicit operator bool() const {
			return !owner_query_id.empty() && generation != 0;
		}
	};

	struct Registration {
		LifecycleRef lifecycle;
		std::string query_id;
		uint64_t token = 0;
		bool publish = false;
	};

	struct Operation {
		LifecycleRef lifecycle;

		explicit operator bool() const {
			return static_cast<bool>(lifecycle);
		}
	};

	struct Abort {
		LifecycleRef lifecycle;
		std::vector<std::string> execution_query_ids;
		uint64_t token = 0;
		bool had_active_operations = false;
	};

	struct Teardown {
		LifecycleRef lifecycle;
		std::vector<std::string> execution_query_ids;
		uint64_t token = 0;
	};

	Registration BeginRegistration(const std::string &query_id, const std::string &owner_query_id);
	void CompleteRegistration(const Registration &registration, bool succeeded);
	void RegisterImmediate(const std::string &query_id, const std::string &owner_query_id);

	std::optional<Operation> BeginOperation(const std::string &query_id,
	                                        const std::string &requested_owner_query_id = {},
	                                        bool create_if_missing = false);
	void EndOperation(const Operation &operation) noexcept;
	void Close(const LifecycleRef &lifecycle);
	void WaitForOperations(const LifecycleRef &lifecycle);
	void WaitForAllOperations();

	std::string OwnerForQuery(const std::string &query_id) const;
	bool IsClosing(const LifecycleRef &lifecycle) const;
	bool IsQueryClosing(const std::string &query_id) const;
	std::vector<std::string> QueryIds(const LifecycleRef &lifecycle) const;
	std::vector<std::string> OwnerQueryIds() const;

	std::optional<Abort> BeginAbort(const std::string &query_id);
	std::optional<Abort> BeginAbort(const Teardown &teardown);
	void CompleteAbort(const Abort &abort, const std::optional<std::string> &error);

	std::optional<Teardown> BeginTeardown(const std::string &query_id);
	void MarkDropping(const Teardown &teardown);
	void CompleteTeardown(const Teardown &teardown, const std::optional<std::string> &error);

	bool BeginShutdown();
	void FinishShutdown(bool succeeded);

private:
	enum class Phase : uint8_t { OPEN, CLOSING, QUIESCING, QUIESCED, DROPPING };

	struct Attempt {
		uint64_t token = 0;
		std::thread::id leader;
		size_t waiters = 0;
		bool complete = false;
		std::optional<std::string> error;
	};

	struct LifecycleState {
		LifecycleRef ref;
		Phase phase = Phase::OPEN;
		std::unordered_set<std::string> query_ids;
		std::unordered_set<std::string> registered_query_ids;
		std::unordered_map<std::string, uint64_t> pending_registrations;
		uint64_t active_operations = 0;
		std::shared_ptr<Attempt> abort_attempt;
		std::shared_ptr<Attempt> teardown_attempt;
	};

	mutable std::mutex mutex_;
	mutable std::condition_variable condition_;
	const std::string component_name_;
	std::unordered_map<std::string, LifecycleRef> query_bindings_;
	std::unordered_map<std::string, std::shared_ptr<LifecycleState>> lifecycles_by_owner_;
	uint64_t next_generation_ = 1;
	uint64_t next_registration_token_ = 1;
	uint64_t next_abort_token_ = 1;
	uint64_t next_teardown_token_ = 1;
	bool shutdown_started_ = false;
	bool shutdown_running_ = false;
	bool shutdown_finished_ = false;

	static uint64_t NextToken(uint64_t &counter);
	static std::vector<std::string> OrderedQueryIds(const LifecycleState &lifecycle);
	static bool SameLifecycle(const LifecycleRef &lhs, const LifecycleRef &rhs);

	std::shared_ptr<LifecycleState> FindLifecycleLocked(const LifecycleRef &lifecycle) const;
	std::shared_ptr<LifecycleState> ResolveLocked(const std::string &query_id) const;
	std::shared_ptr<LifecycleState> ResolveOwnerLocked(const std::string &owner_query_id) const;
	std::shared_ptr<LifecycleState> CreateLifecycleLocked(const std::string &owner_query_id);
	void BindQueryLocked(const std::shared_ptr<LifecycleState> &lifecycle, const std::string &query_id);
	void EnsureTransitionAllowedLocked(const std::string &query_id, const char *operation) const;
	static void EnsureAttemptLeader(const Attempt &attempt, const std::string &operation,
	                                const std::string &owner_query_id);
	void WaitForAttemptLocked(std::unique_lock<std::mutex> &guard, const std::shared_ptr<Attempt> &attempt,
	                          const std::string &operation, const std::string &owner_query_id);
	std::optional<Abort> BeginAbortLocked(std::unique_lock<std::mutex> &guard,
	                                      const std::shared_ptr<LifecycleState> &lifecycle);
};

} // namespace ray
} // namespace python
} // namespace distributed
} // namespace duckdb
