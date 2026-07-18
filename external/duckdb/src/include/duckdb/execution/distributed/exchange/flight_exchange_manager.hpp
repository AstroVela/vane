// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

/**
 * @file flight_exchange_manager.hpp
 * @brief Concrete FlightExchangeManager — disk-first + Arrow Flight transport.
 *
 * Aligned with Vane's shuffle service design:
 *   - Sink writes to ShuffleCache (disk IPC files) instead of memory buffers
 *   - Source reads from ShuffleCacheRegistry (local) or via Arrow Flight (remote)
 */

#pragma once

#include "duckdb/execution/distributed/exchange/exchange_manager.hpp"
#include "duckdb/execution/distributed/exchange/flight_server.hpp"
#include "duckdb/execution/distributed/exchange/shuffle_cache.hpp"
#include "duckdb/execution/distributed/exchange/shuffle_cache_registry.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/types.hpp"

#include <string>
#include <cstdlib>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace duckdb {

class ClientContext;

namespace distributed {

struct FlightExchangeConfig {
	std::string flight_bind_host = "0.0.0.0";
	int flight_port = 0;
	std::vector<std::string> local_dirs; // shuffle directories for IPC files
	std::string node_id;
	std::string flight_location_template;
	double flight_timeout_seconds = 0.0;
	std::vector<LogicalType> expected_types;
};

inline std::string ResolveFlightExchangeEnvString(const char *name) {
	const char *value = std::getenv(name);
	return value ? std::string(value) : std::string();
}

inline int ResolveFlightExchangeEnvInt(const char *name, int fallback) {
	const char *value = std::getenv(name);
	if (!value || !*value) {
		return fallback;
	}
	try {
		return std::stoi(value);
	} catch (...) {
		return fallback;
	}
}

inline std::string FlightExchangeJoinPath(const std::string &base, const std::string &child) {
	if (base.empty()) {
		return child;
	}
	auto last = base[base.size() - 1];
	if (last == '/' || last == '\\') {
		return base + child;
	}
	return base + "/" + child;
}

inline unsigned long long ResolveVaneProcessId() {
#if defined(_WIN32)
	return static_cast<unsigned long long>(_getpid());
#else
	return static_cast<unsigned long long>(getpid());
#endif
}

inline void SetFlightExchangeEnvString(const char *name, const std::string &value) {
#if defined(_WIN32)
	_putenv_s(name, value.c_str());
#else
	setenv(name, value.c_str(), 1);
#endif
}

inline std::string BuildDefaultVaneSessionDir() {
	auto cwd = FileSystem::GetWorkingDirectory();
	if (cwd.empty()) {
		throw std::runtime_error("cannot resolve default VANE_SESSION_DIR: current working directory is empty");
	}
	auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
	               .count();
	auto session_name = std::string("session_") + std::to_string(ResolveVaneProcessId()) + "_" + std::to_string(now);
	return FlightExchangeJoinPath(FlightExchangeJoinPath(cwd, "vane"), session_name);
}

inline std::string ResolveVaneSessionDirFromEnv() {
	auto session_dir = ResolveFlightExchangeEnvString("VANE_SESSION_DIR");
	if (!session_dir.empty()) {
		return session_dir;
	}
	session_dir = BuildDefaultVaneSessionDir();
	SetFlightExchangeEnvString("VANE_SESSION_DIR", session_dir);
	return session_dir;
}

inline vector<string> DefaultFlightExchangeLocalDirs() {
	return {FlightExchangeJoinPath(ResolveVaneSessionDirFromEnv(), "flight_shuffle")};
}

inline vector<string> ResolveFlightExchangeLocalDirsFromEnv() {
	auto dirs = ResolveFlightExchangeEnvString("DUCKDB_SHUFFLE_DIRS");
	if (dirs.empty()) {
		dirs = ResolveFlightExchangeEnvString("VANE_SHUFFLE_LOCAL_DIRS");
	}
	if (dirs.empty()) {
		return DefaultFlightExchangeLocalDirs();
	}
	auto split_dirs = [&](char delimiter) {
		vector<string> result;
		std::string token;
		std::istringstream stream(dirs);
		while (std::getline(stream, token, delimiter)) {
			if (!token.empty()) {
				result.push_back(token);
			}
		}
		return result;
	};
	if (dirs.find(',') != std::string::npos) {
		auto result = split_dirs(',');
		return result.empty() ? DefaultFlightExchangeLocalDirs() : result;
	}
	if (dirs.find(';') != std::string::npos) {
		auto result = split_dirs(';');
		return result.empty() ? DefaultFlightExchangeLocalDirs() : result;
	}
	if (dirs.find("://") != std::string::npos) {
		return {dirs};
	}
	auto result = split_dirs(':');
	return result.empty() ? DefaultFlightExchangeLocalDirs() : result;
}

inline std::string ResolveFlightExchangeNodeIdFromEnv() {
	auto node_id = ResolveFlightExchangeEnvString("VANE_WORKER_ID");
	if (!node_id.empty()) {
		return node_id;
	}
	node_id = ResolveFlightExchangeEnvString("RAY_NODE_IP_ADDRESS");
	if (!node_id.empty()) {
		return node_id;
	}
	node_id = ResolveFlightExchangeEnvString("RAY_NODE_ID");
	if (!node_id.empty()) {
		return node_id;
	}
	node_id = ResolveFlightExchangeEnvString("HOSTNAME");
	if (!node_id.empty()) {
		return node_id;
	}
	return "local";
}

inline FlightExchangeConfig ResolveFlightExchangeConfigFromEnv() {
	FlightExchangeConfig config;
	config.node_id = ResolveFlightExchangeNodeIdFromEnv();
	config.local_dirs = ResolveFlightExchangeLocalDirsFromEnv();
	config.flight_bind_host = "0.0.0.0";
	config.flight_port = ResolveFlightExchangeEnvInt("DUCKDB_FLIGHT_PORT", 0);
	return config;
}

// ─── FlightExchange (coordinator) ────────────────────────

class FlightExchange : public Exchange {
public:
	FlightExchange(const ExchangeContext &ctx, idx_t output_partition_count, const FlightExchangeConfig &config,
	               ClientContext *context = nullptr);
	~FlightExchange() override;

	ExchangeSinkHandle AddSink(idx_t task_partition_id) override;
	ExchangeSinkInstanceHandle InstantiateSink(const ExchangeSinkHandle &handle, idx_t attempt_id) override;
	void SinkFinished(const ExchangeSinkHandle &handle, idx_t attempt_id) override;
	void SinkFinished(const ExchangeSinkHandle &handle, idx_t attempt_id, const std::string &node_id,
	                  int flight_port) override;
	void AllRequiredSinksFinished() override;
	std::vector<ExchangeSourceHandle> GetSourceHandles() override;
	idx_t GetNumPartitions() const override;
	void CleanupUnselectedAttempts();
	void Close() override;

private:
	struct SinkAttemptMetadata {
		idx_t task_partition_id = 0;
		idx_t attempt_id = 0;
		std::string output_location;
		std::string node_id;
		int flight_port = 0;
	};

	ExchangeContext ctx_;
	idx_t output_partition_count_;
	FlightExchangeConfig config_;
	ClientContext *context_;

	std::mutex mutex_;
	std::vector<idx_t> all_sinks_;
	std::unordered_map<idx_t, std::unordered_map<idx_t, SinkAttemptMetadata>> sink_attempts_;
	std::unordered_map<idx_t, idx_t> selected_attempts_; // task_partition_id -> attempt_id
	std::unordered_set<std::string> cleaned_output_locations_;
	bool closed_ = false;

	std::vector<SinkAttemptMetadata> CollectUnselectedAttemptsForCleanupLocked();
	bool CleanupAttemptStorage(const SinkAttemptMetadata &attempt_metadata, const char *reason);
};

// ─── FlightExchangeSink (worker) ─────────────────────────

class FlightExchangeSink : public ExchangeSink {
public:
	FlightExchangeSink(std::shared_ptr<ShuffleCache> shuffle_cache, const ExchangeSinkInstanceHandle &handle,
	                   ClientContext *context);
	~FlightExchangeSink() override;

	DuckDBResult<void> AddChunk(idx_t partition_id, DataChunk &chunk) override;
	bool IsBlocked() const override;
	void WaitUnblocked() override;
	DuckDBResult<void> Finish() override;
	DuckDBResult<void> Abort() override;
	size_t GetMemoryUsage() const override;
	DuckDBResult<void> EnsureSchema(ClientContext &context, const vector<LogicalType> &types,
	                                const vector<string> &names) override;

private:
	std::shared_ptr<ShuffleCache> shuffle_cache_;
	ExchangeSinkInstanceHandle handle_;
	ClientContext *context_;
	bool finished_ = false;
};

// ─── FlightExchangeSource (worker) ──────────────────────

class FlightExchangeSource : public ExchangeSource {
public:
	explicit FlightExchangeSource(const std::string &exchange_id, ClientContext *context);
	explicit FlightExchangeSource(const FlightExchangeConfig &config, ClientContext *context);
	~FlightExchangeSource() override;

	void AddSourceHandles(std::vector<ExchangeSourceHandle> handles) override;
	bool ReadChunk(DataChunk &chunk) override;
	bool IsBlocked() const override;
	void WaitUnblocked() override;
	bool IsFinished() const override;
	size_t GetMemoryUsage() const override;
	void Close() override;

private:
	struct PartitionStreamState;

	FlightExchangeConfig config_;
	std::string exchange_id_;
	ClientContext *context_;
	std::shared_ptr<ShuffleCache> cache_;
	std::string cache_key_;
	std::vector<ExchangeSourceHandle> handles_;
	idx_t current_handle_idx_ = 0;
	bool closed_ = false;

	std::unique_ptr<PartitionStreamState> stream_state_;

	DuckDBResult<std::unique_ptr<PartitionStreamState>> OpenPartitionStream(const ExchangeSourceHandle &handle);
	DuckDBResult<bool> ReadStreamChunk(DataChunk &chunk);
};

// ─── FlightExchangeManager (factory) ────────────────────

class FlightExchangeManager : public ExchangeManager {
public:
	explicit FlightExchangeManager(FlightExchangeConfig config, ClientContext *context = nullptr);
	~FlightExchangeManager() override;

	std::unique_ptr<Exchange> CreateExchange(const ExchangeContext &ctx, idx_t output_partition_count) override;
	std::unique_ptr<ExchangeSink> CreateSink(const ExchangeSinkInstanceHandle &handle) override;
	std::unique_ptr<ExchangeSource> CreateSource() override;

	void SetContext(ClientContext *ctx) override {
		context_ = ctx;
		RefreshRuntimeNodeId();
	}

	static int GetLocalFlightServerPort();

	const FlightExchangeConfig &config() const {
		return config_;
	}

	void Shutdown() override;

private:
	FlightExchangeConfig config_;
	ClientContext *context_;

	void RefreshRuntimeNodeId();
};

} // namespace distributed
} // namespace duckdb
