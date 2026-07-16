// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file shuffle_cache_registry.hpp
 * @brief Global registry of ShuffleCache instances keyed by exchange_id.
 *
 * Sinks register their ShuffleCache after FlushAll; Sources and the
 * FlightServer look up caches by exchange_id to read partition data.
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
	using RegistryMap = std::unordered_map<std::string, std::shared_ptr<ShuffleCache>>;

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

	/// Register a ShuffleCache for an exchange (called after sink FlushAll).
	void Register(const std::string &exchange_id, std::shared_ptr<ShuffleCache> cache) {
		std::lock_guard<std::mutex> lock(mutex_);
		deferred_cleanup_.erase(exchange_id);
		registry_[exchange_id] = std::move(cache);
	}

	/// Look up a ShuffleCache by exchange_id.
	std::shared_ptr<ShuffleCache> Get(const std::string &exchange_id) const {
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = registry_.find(exchange_id);
		if (it == registry_.end()) {
			return nullptr;
		}
		return it->second;
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
		if (it->second) {
			deferred_cleanup_[exchange_id] = it->second;
		}
		registry_.erase(it);
	}

	CleanupResult RemoveAndCleanupByPrefix(const std::string &exchange_id_prefix) {
		CleanupResult result;
		if (exchange_id_prefix.empty()) {
			return result;
		}
		std::vector<std::pair<std::string, std::shared_ptr<ShuffleCache>>> removed;
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
			auto cleanup_res = entry.second->RemoveAttemptStorage();
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

	static void CollectByPrefix(RegistryMap &registry, const std::string &exchange_id_prefix,
	                            std::vector<std::pair<std::string, std::shared_ptr<ShuffleCache>>> &removed) {
		for (auto it = registry.begin(); it != registry.end();) {
			if (it->first.rfind(exchange_id_prefix, 0) != 0) {
				++it;
				continue;
			}
			removed.emplace_back(it->first, it->second);
			it = registry.erase(it);
		}
	}

	mutable std::mutex mutex_;
	RegistryMap registry_;
	RegistryMap deferred_cleanup_;
};

} // namespace distributed
} // namespace duckdb
