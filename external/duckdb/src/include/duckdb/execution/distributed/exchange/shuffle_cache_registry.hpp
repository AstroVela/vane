// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file shuffle_cache_registry.hpp
 * @brief Process-local catalog of published ShuffleCache instances.
 *
 * Sinks publish committed attempts after FlushAll. Sources and the Flight
 * service borrow catalog entries by opaque exchange-attempt ID. A borrowed
 * entry keeps its storage alive until the reader releases it.
 */

#pragma once

#include "duckdb/execution/distributed/exchange/shuffle_cache.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace duckdb {
namespace distributed {

class ShuffleCacheRegistry {
private:
	struct CacheLeaseState {
		explicit CacheLeaseState(std::shared_ptr<ShuffleCache> cache_p) : cache(std::move(cache_p)) {
		}

		~CacheLeaseState() {
			try {
				std::lock_guard<std::mutex> lock(cleanup_mutex);
				if (cleanup_requested && !cleanup_complete) {
					(void)CleanupLocked();
				}
			} catch (...) {
			}
		}

		void RequestCleanup() {
			std::lock_guard<std::mutex> lock(cleanup_mutex);
			cleanup_requested = true;
		}

		DuckDBResult<idx_t> CleanupIfRequested() {
			std::lock_guard<std::mutex> lock(cleanup_mutex);
			if (!cleanup_requested || cleanup_complete) {
				return DuckDBResult<idx_t>::ok(0);
			}
			return CleanupLocked();
		}

		std::shared_ptr<ShuffleCache> cache;

	private:
		DuckDBResult<idx_t> CleanupLocked() {
			auto result = cache->RemoveAttemptStorage();
			if (result.is_ok()) {
				cleanup_complete = true;
			}
			return result;
		}

		std::mutex cleanup_mutex;
		bool cleanup_requested = false;
		bool cleanup_complete = false;
	};

	struct Entry {
		std::shared_ptr<CacheLeaseState> state;
		std::string server_epoch;
		idx_t attempt_id = 0;
	};

	using RegistryMap = std::unordered_map<std::string, Entry>;

public:
	struct CleanupResult {
		idx_t registry_entries_removed = 0;
		idx_t storage_entries_removed = 0;
		idx_t cleanup_errors = 0;
	};

	static ShuffleCacheRegistry &Instance() {
		static ShuffleCacheRegistry instance;
		return instance;
	}

	/// Publish a committed ShuffleCache descriptor for an exchange attempt.
	DuckDBResult<void> Register(const std::string &exchange_id, std::shared_ptr<ShuffleCache> cache,
	                            std::string server_epoch = std::string(), idx_t attempt_id = 0) {
		if (exchange_id.empty() || !cache) {
			return DuckDBResult<void>::err(
			    DuckDBError::value_error("shuffle cache registration requires an exchange id and cache"));
		}
		std::lock_guard<std::mutex> lock(mutex_);
		auto existing = registry_.find(exchange_id);
		if (existing != registry_.end()) {
			if (!DescriptorsMatch(existing->second, cache, server_epoch, attempt_id)) {
				return DuckDBResult<void>::err(DuckDBError::invalid_state_error(
				    "conflicting shuffle cache registration for exchange " + exchange_id));
			}
			return DuckDBResult<void>::ok();
		}
		Entry entry {std::make_shared<CacheLeaseState>(std::move(cache)), std::move(server_epoch), attempt_id};
		deferred_cleanup_.erase(exchange_id);
		registry_.emplace(exchange_id, std::move(entry));
		return DuckDBResult<void>::ok();
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
		return DuckDBResult<std::shared_ptr<ShuffleCache>>::ok(Borrow(entry.state));
	}

	/// Remove a ShuffleCache (when exchange closes).
	void Remove(const std::string &exchange_id) {
		std::lock_guard<std::mutex> lock(mutex_);
		registry_.erase(exchange_id);
	}

	void RemoveForDeferredCleanup(const std::string &exchange_id) {
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = registry_.find(exchange_id);
		if (it == registry_.end()) {
			return;
		}
		if (it->second.state && it->second.state->cache) {
			deferred_cleanup_[exchange_id] = it->second;
		}
		registry_.erase(it);
	}

	CleanupResult RemoveAndCleanupByPrefix(const std::string &exchange_id_prefix) {
		CleanupResult result;
		if (exchange_id_prefix.empty()) {
			return result;
		}
		std::vector<std::pair<std::string, std::shared_ptr<CacheLeaseState>>> removed;
		{
			std::lock_guard<std::mutex> lock(mutex_);
			CollectByPrefix(registry_, exchange_id_prefix, removed);
			CollectByPrefix(deferred_cleanup_, exchange_id_prefix, removed);
		}
		result.registry_entries_removed = static_cast<idx_t>(removed.size());
		for (auto &entry : removed) {
			if (!entry.second) {
				continue;
			}
			entry.second->RequestCleanup();
			if (entry.second.use_count() > 1) {
				continue;
			}
			auto cleanup_res = entry.second->CleanupIfRequested();
			if (cleanup_res.is_err()) {
				result.cleanup_errors++;
				continue;
			}
			result.storage_entries_removed += cleanup_res.value();
		}
		return result;
	}

private:
	ShuffleCacheRegistry() = default;

	static std::shared_ptr<ShuffleCache> Borrow(const std::shared_ptr<CacheLeaseState> &state) {
		if (!state || !state->cache) {
			return nullptr;
		}
		return std::shared_ptr<ShuffleCache>(state, state->cache.get());
	}

	static bool DescriptorsMatch(const Entry &left, const std::shared_ptr<ShuffleCache> &right_cache,
	                             const std::string &right_server_epoch, idx_t right_attempt_id) {
		if (!left.state || !left.state->cache || !right_cache || left.state->cache.get() != right_cache.get() ||
		    left.server_epoch != right_server_epoch || left.attempt_id != right_attempt_id) {
			return false;
		}
		const auto &left_config = left.state->cache->config();
		const auto &right_config = right_cache->config();
		return left_config.shuffle_stage_id == right_config.shuffle_stage_id &&
		       left_config.node_id == right_config.node_id &&
		       left_config.num_partitions == right_config.num_partitions &&
		       left_config.local_dirs == right_config.local_dirs;
	}

	static void CollectByPrefix(RegistryMap &registry, const std::string &exchange_id_prefix,
	                            std::vector<std::pair<std::string, std::shared_ptr<CacheLeaseState>>> &removed) {
		for (auto it = registry.begin(); it != registry.end();) {
			if (it->first.rfind(exchange_id_prefix, 0) != 0) {
				++it;
				continue;
			}
			removed.emplace_back(it->first, it->second.state);
			it = registry.erase(it);
		}
	}

	mutable std::mutex mutex_;
	RegistryMap registry_;
	RegistryMap deferred_cleanup_;
};

} // namespace distributed
} // namespace duckdb
