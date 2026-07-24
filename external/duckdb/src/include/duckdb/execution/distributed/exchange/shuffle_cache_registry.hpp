// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file shuffle_cache_registry.hpp
 * @brief Process-local catalog of published ShuffleCache instances.
 *
 * Sinks publish committed attempts after FlushAll. Sources and the Flight
 * service borrow catalog entries by opaque exchange-attempt ID. Query close
 * fences late publishers and keeps cleanup work visible until all readers and
 * native executions have drained.
 */

#pragma once

#include "duckdb/execution/distributed/exchange/shuffle_cache.hpp"

#include <algorithm>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace duckdb {
namespace distributed {

class ShuffleCacheRegistry {
private:
	struct CacheLeaseState {
		explicit CacheLeaseState(std::shared_ptr<ShuffleCache> cache_p) : cache(std::move(cache_p)) {
		}

		bool TryAcquireReader() {
			std::lock_guard<std::mutex> lock(cleanup_mutex);
			if (cleanup_requested || cleanup_complete) {
				return false;
			}
			active_readers++;
			return true;
		}

		void ReleaseReader() {
			std::lock_guard<std::mutex> lock(cleanup_mutex);
			D_ASSERT(active_readers > 0);
			active_readers--;
		}

		void RequestCleanup() {
			std::lock_guard<std::mutex> lock(cleanup_mutex);
			cleanup_requested = true;
		}

		DuckDBResult<idx_t> CleanupIfReady() {
			std::lock_guard<std::mutex> lock(cleanup_mutex);
			if (!cleanup_requested || cleanup_complete || active_readers > 0) {
				return DuckDBResult<idx_t>::ok(0);
			}
			auto result = cache->RemoveAttemptStorage();
			if (result.is_ok()) {
				cleanup_complete = true;
			}
			return result;
		}

		bool CleanupComplete() const {
			std::lock_guard<std::mutex> lock(cleanup_mutex);
			return cleanup_complete;
		}

		std::shared_ptr<ShuffleCache> cache;

	private:
		mutable std::mutex cleanup_mutex;
		idx_t active_readers = 0;
		bool cleanup_requested = false;
		bool cleanup_complete = false;
	};

	struct CacheBorrowLease {
		explicit CacheBorrowLease(std::shared_ptr<CacheLeaseState> state_p) : state(std::move(state_p)) {
		}

		~CacheBorrowLease() {
			state->ReleaseReader();
		}

		std::shared_ptr<CacheLeaseState> state;
	};

	struct Entry {
		std::string exchange_id;
		std::string query_id;
		std::shared_ptr<CacheLeaseState> state;
		std::string server_epoch;
		idx_t attempt_id = 0;
	};

	struct QueryState {
		idx_t active_executions = 0;
		bool closing = false;
	};

	using RegistryMap = std::unordered_map<std::string, Entry>;

public:
	struct CleanupResult {
		idx_t registry_entries_removed = 0;
		idx_t storage_entries_removed = 0;
		idx_t cleanup_errors = 0;
		idx_t cleanup_pending = 0;
		idx_t active_executions = 0;
	};

	static ShuffleCacheRegistry &Instance() {
		static ShuffleCacheRegistry instance;
		return instance;
	}

	/// Mark one native query execution active, unless teardown already started.
	DuckDBResult<void> BeginQueryExecution(const std::string &query_id) {
		if (query_id.empty()) {
			return DuckDBResult<void>::err(
			    DuckDBError::value_error("query execution registration requires a query id"));
		}
		std::lock_guard<std::mutex> lock(mutex_);
		auto &state = query_states_[query_id];
		if (state.closing) {
			return DuckDBResult<void>::err(
			    DuckDBError::invalid_state_error("query is closing and cannot start new native work: " + query_id));
		}
		state.active_executions++;
		return DuckDBResult<void>::ok();
	}

	/// Release one native query execution after the underlying thread exits.
	DuckDBResult<void> EndQueryExecution(const std::string &query_id) {
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = query_states_.find(query_id);
		if (it == query_states_.end() || it->second.active_executions == 0) {
			return DuckDBResult<void>::err(
			    DuckDBError::invalid_state_error("query has no active native execution: " + query_id));
		}
		it->second.active_executions--;
		if (it->second.active_executions == 0 && !it->second.closing) {
			query_states_.erase(it);
		}
		return DuckDBResult<void>::ok();
	}

	/// Fence late publishers and move all currently published attempts to cleanup ownership.
	DuckDBResult<void> CloseQuery(const std::string &query_id) {
		if (query_id.empty()) {
			return DuckDBResult<void>::err(DuckDBError::value_error("query close requires a query id"));
		}
		std::vector<std::shared_ptr<CacheLeaseState>> removed;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			query_states_[query_id].closing = true;
			for (auto it = registry_.begin(); it != registry_.end();) {
				if (it->second.query_id != query_id) {
					++it;
					continue;
				}
				removed.push_back(it->second.state);
				deferred_cleanup_.push_back(std::move(it->second));
				it = registry_.erase(it);
			}
		}
		for (const auto &state : removed) {
			if (state) {
				state->RequestCleanup();
			}
		}
		return DuckDBResult<void>::ok();
	}

	/// Publish a committed ShuffleCache descriptor for an exchange attempt.
	DuckDBResult<void> Register(const std::string &exchange_id, std::shared_ptr<ShuffleCache> cache,
	                            const std::string &query_id, std::string server_epoch = std::string(),
	                            idx_t attempt_id = 0) {
		if (exchange_id.empty() || query_id.empty() || !cache) {
			return DuckDBResult<void>::err(
			    DuckDBError::value_error("shuffle cache registration requires an exchange id, query id, and cache"));
		}

		auto state = std::make_shared<CacheLeaseState>(std::move(cache));
		bool query_closing = false;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			auto query_it = query_states_.find(query_id);
			query_closing = query_it != query_states_.end() && query_it->second.closing;
			if (!query_closing) {
				auto existing = registry_.find(exchange_id);
				if (existing != registry_.end()) {
					if (!DescriptorsMatch(existing->second, state->cache, query_id, server_epoch, attempt_id)) {
						return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
						    "conflicting shuffle cache registration for exchange " + exchange_id));
					}
					return DuckDBResult<void>::ok();
				}
				Entry entry {exchange_id, query_id, std::move(state), std::move(server_epoch), attempt_id};
				registry_.emplace(exchange_id, std::move(entry));
				return DuckDBResult<void>::ok();
			}
			deferred_cleanup_.push_back(Entry {exchange_id, query_id, state, std::move(server_epoch), attempt_id});
		}

		state->RequestCleanup();
		auto cleanup_result = state->CleanupIfReady();
		RemoveCompletedDeferred();
		if (cleanup_result.is_err()) {
			return DuckDBResult<void>::err(cleanup_result.error());
		}
		return DuckDBResult<void>::err(
		    DuckDBError::invalid_state_error("query closed before shuffle cache publication: " + query_id));
	}

	/// Look up a ShuffleCache by exchange_id.
	std::shared_ptr<ShuffleCache> Get(const std::string &exchange_id) const {
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = registry_.find(exchange_id);
		if (it == registry_.end()) {
			return nullptr;
		}
		return Borrow(it->second.state);
	}

	/// Resolve a ticket only while its exact exchange attempt is published.
	DuckDBResult<std::shared_ptr<ShuffleCache>> Resolve(const std::string &exchange_id, const std::string &server_epoch,
	                                                    const std::string &node_id, idx_t attempt_id) const {
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = registry_.find(exchange_id);
		if (it == registry_.end() || !it->second.state || !it->second.state->cache) {
			return DuckDBResult<std::shared_ptr<ShuffleCache>>::err(
			    DuckDBError::invalid_state_error("flight exchange attempt is not published: " + exchange_id));
		}
		const auto &entry = it->second;
		const auto &config = entry.state->cache->config();
		if (entry.server_epoch != server_epoch) {
			return DuckDBResult<std::shared_ptr<ShuffleCache>>::err(
			    DuckDBError::invalid_state_error("flight ticket server epoch is stale"));
		}
		if (entry.attempt_id != attempt_id) {
			return DuckDBResult<std::shared_ptr<ShuffleCache>>::err(
			    DuckDBError::invalid_state_error("flight ticket attempt id does not match the published attempt"));
		}
		if (config.shuffle_stage_id != exchange_id || config.node_id != node_id) {
			return DuckDBResult<std::shared_ptr<ShuffleCache>>::err(
			    DuckDBError::invalid_state_error("flight ticket exchange or node identity does not match"));
		}
		auto borrowed = Borrow(entry.state);
		if (!borrowed) {
			return DuckDBResult<std::shared_ptr<ShuffleCache>>::err(
			    DuckDBError::invalid_state_error("flight exchange attempt is closing: " + exchange_id));
		}
		return DuckDBResult<std::shared_ptr<ShuffleCache>>::ok(std::move(borrowed));
	}

	/// Remove a catalog entry without taking ownership of its storage.
	void Remove(const std::string &exchange_id) {
		std::lock_guard<std::mutex> lock(mutex_);
		registry_.erase(exchange_id);
	}

	/// Move a catalog entry to explicit deferred-cleanup ownership.
	void RemoveForDeferredCleanup(const std::string &exchange_id) {
		std::shared_ptr<CacheLeaseState> removed;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			auto it = registry_.find(exchange_id);
			if (it == registry_.end()) {
				return;
			}
			removed = it->second.state;
			deferred_cleanup_.push_back(std::move(it->second));
			registry_.erase(it);
		}
		if (removed) {
			removed->RequestCleanup();
		}
	}

	/// Retry all cleanup owned by one exact query.
	CleanupResult RemoveAndCleanupByQuery(const std::string &query_id) {
		if (query_id.empty()) {
			return {};
		}
		auto close_result = CloseQuery(query_id);
		if (close_result.is_err()) {
			CleanupResult result;
			result.cleanup_errors = 1;
			return result;
		}
		return CleanupMatching([&](const Entry &entry) { return entry.query_id == query_id; }, &query_id);
	}

	/// Test/maintenance helper for exchange-prefix cleanup.
	CleanupResult RemoveAndCleanupByPrefix(const std::string &exchange_id_prefix) {
		if (exchange_id_prefix.empty()) {
			return {};
		}
		return CleanupMatching([&](const Entry &entry) { return entry.exchange_id.rfind(exchange_id_prefix, 0) == 0; },
		                       nullptr);
	}

private:
	ShuffleCacheRegistry() = default;

	static std::shared_ptr<ShuffleCache> Borrow(const std::shared_ptr<CacheLeaseState> &state) {
		if (!state || !state->cache || !state->TryAcquireReader()) {
			return nullptr;
		}
		auto lease = std::make_shared<CacheBorrowLease>(state);
		return std::shared_ptr<ShuffleCache>(lease, state->cache.get());
	}

	static bool DescriptorsMatch(const Entry &left, const std::shared_ptr<ShuffleCache> &right_cache,
	                             const std::string &right_query_id, const std::string &right_server_epoch,
	                             idx_t right_attempt_id) {
		if (!left.state || !left.state->cache || !right_cache || left.state->cache.get() != right_cache.get() ||
		    left.query_id != right_query_id || left.server_epoch != right_server_epoch ||
		    left.attempt_id != right_attempt_id) {
			return false;
		}
		const auto &left_config = left.state->cache->config();
		const auto &right_config = right_cache->config();
		return left_config.shuffle_stage_id == right_config.shuffle_stage_id &&
		       left_config.node_id == right_config.node_id &&
		       left_config.num_partitions == right_config.num_partitions &&
		       left_config.local_dirs == right_config.local_dirs;
	}

	template <class MATCH>
	CleanupResult CleanupMatching(MATCH &&matches, const std::string *query_id) {
		CleanupResult result;
		std::vector<std::shared_ptr<CacheLeaseState>> candidates;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			for (auto it = registry_.begin(); it != registry_.end();) {
				if (!matches(it->second)) {
					++it;
					continue;
				}
				deferred_cleanup_.push_back(std::move(it->second));
				it = registry_.erase(it);
				result.registry_entries_removed++;
			}
			std::unordered_set<CacheLeaseState *> seen;
			for (const auto &entry : deferred_cleanup_) {
				if (matches(entry) && entry.state && seen.insert(entry.state.get()).second) {
					candidates.push_back(entry.state);
				}
			}
			if (query_id) {
				auto query_it = query_states_.find(*query_id);
				if (query_it != query_states_.end()) {
					result.active_executions = query_it->second.active_executions;
				}
			}
		}

		for (const auto &state : candidates) {
			state->RequestCleanup();
			auto cleanup_result = state->CleanupIfReady();
			if (cleanup_result.is_err()) {
				result.cleanup_errors++;
				continue;
			}
			result.storage_entries_removed += cleanup_result.value();
		}

		{
			std::lock_guard<std::mutex> lock(mutex_);
			deferred_cleanup_.erase(
			    std::remove_if(deferred_cleanup_.begin(), deferred_cleanup_.end(),
			                   [&](const Entry &entry) { return matches(entry) && entry.state->CleanupComplete(); }),
			    deferred_cleanup_.end());
			for (const auto &entry : deferred_cleanup_) {
				if (matches(entry)) {
					result.cleanup_pending++;
				}
			}
			if (query_id) {
				auto query_it = query_states_.find(*query_id);
				result.active_executions = query_it == query_states_.end() ? 0 : query_it->second.active_executions;
			}
		}
		return result;
	}

	void RemoveCompletedDeferred() {
		std::lock_guard<std::mutex> lock(mutex_);
		deferred_cleanup_.erase(
		    std::remove_if(deferred_cleanup_.begin(), deferred_cleanup_.end(),
		                   [](const Entry &entry) { return entry.state && entry.state->CleanupComplete(); }),
		    deferred_cleanup_.end());
	}

	mutable std::mutex mutex_;
	RegistryMap registry_;
	std::vector<Entry> deferred_cleanup_;
	std::unordered_map<std::string, QueryState> query_states_;
};

} // namespace distributed
} // namespace duckdb
