// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// physical_remote_exchange_source.cpp — Delegates to ExchangeManager SPI
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/operator/exchange/physical_remote_exchange_source.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/exchange/flight_exchange_manager.hpp"
#include "duckdb/execution/distributed/plan/exchange_source_task.hpp"
#include "duckdb/execution/distributed/plan/fte_split_queue.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/algorithm.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/parallel/task_scheduler.hpp"

#include <mutex>

namespace duckdb {

namespace {
struct RemoteSourceGlobalState : public GlobalSourceState {
	RemoteSourceGlobalState(std::shared_ptr<distributed::ExchangeManager> exchange_mgr_p,
	                        std::vector<distributed::ExchangeSourceHandle> source_handles_p,
	                        std::shared_ptr<distributed::FteSplitQueue> runtime_split_queue_p, idx_t max_threads_p)
	    : exchange_mgr(std::move(exchange_mgr_p)), source_handles(std::move(source_handles_p)),
	      runtime_split_queue(std::move(runtime_split_queue_p)), max_threads(MaxValue<idx_t>(idx_t(1), max_threads_p)),
	      next_handle_idx(0) {
	}

	idx_t MaxThreads() override {
		return max_threads;
	}

	std::shared_ptr<distributed::ExchangeManager> exchange_mgr;
	std::vector<distributed::ExchangeSourceHandle> source_handles;
	std::shared_ptr<distributed::FteSplitQueue> runtime_split_queue;
	idx_t max_threads;
	atomic<idx_t> next_handle_idx;
	std::mutex source_lock;
};

struct RemoteSourceLocalState : public LocalSourceState {
	explicit RemoteSourceLocalState(std::unique_ptr<distributed::ExchangeSource> source_p)
	    : source(std::move(source_p)) {
	}

	std::unique_ptr<distributed::ExchangeSource> source;
	bool finished = false;
	bool dynamic_no_more_splits = false;
};

idx_t RemoteExchangeSchedulerThreads(ClientContext &context) {
	auto &scheduler = TaskScheduler::GetScheduler(context);
	auto threads = NumericCast<idx_t>(scheduler.NumberOfThreads());
	return MaxValue<idx_t>(idx_t(1), threads);
}

idx_t ResolveRemoteExchangeMaxThreads(ClientContext &context,
                                      const std::vector<distributed::ExchangeSourceHandle> &source_handles,
                                      const vector<idx_t> &partition_indices, idx_t source_partition_count,
                                      idx_t source_task_count,
                                      const std::shared_ptr<distributed::FteSplitQueue> &runtime_split_queue) {
	idx_t max_threads = 1;
	auto consider = [&](idx_t value) {
		if (value > 0) {
			max_threads = MaxValue<idx_t>(max_threads, value);
		}
	};

	consider(source_task_count);
	if (source_task_count == 0) {
		consider(source_partition_count);
	}
	consider(source_handles.size());
	consider(partition_indices.size());

	if (runtime_split_queue) {
		bool has_dynamic_width = false;
		auto queue_task_count = runtime_split_queue->ExchangeSourceTaskCount();
		if (queue_task_count > 0) {
			consider(queue_task_count);
			has_dynamic_width = true;
		} else {
			auto queue_partition_count = runtime_split_queue->ExchangeSourcePartitionCount();
			if (queue_partition_count > 0) {
				consider(queue_partition_count);
				has_dynamic_width = true;
			}
		}
		auto submitted_splits = runtime_split_queue->SubmittedSplits();
		if (submitted_splits > 0) {
			consider(submitted_splits);
			has_dynamic_width = true;
		} else {
			auto buffered_splits = runtime_split_queue->BufferedSplits();
			if (buffered_splits > 0) {
				consider(buffered_splits);
				has_dynamic_width = true;
			}
		}
		if (!has_dynamic_width && source_task_count == 0 && source_partition_count == 0 && source_handles.empty() &&
		    partition_indices.empty()) {
			// Dynamic split queues can receive splits after the pipeline is scheduled.
			// Keep enough source tasks alive to wait on the queue instead of resolving to 1.
			consider(RemoteExchangeSchedulerThreads(context));
		}
	}

	return MinValue<idx_t>(max_threads, RemoteExchangeSchedulerThreads(context));
}

bool AssignNextStaticHandle(RemoteSourceGlobalState &gstate, RemoteSourceLocalState &lstate) {
	while (true) {
		auto handle_idx = gstate.next_handle_idx.fetch_add(1);
		if (handle_idx >= gstate.source_handles.size()) {
			return false;
		}
		std::vector<distributed::ExchangeSourceHandle> handles;
		handles.push_back(gstate.source_handles[handle_idx]);
		lstate.source->AddSourceHandles(std::move(handles));
		if (!lstate.source->IsFinished()) {
			return true;
		}
	}
}

SourceResultType AssignNextDynamicSplit(RemoteSourceGlobalState &gstate, RemoteSourceLocalState &lstate,
                                        const duckdb::InterruptState &interrupt_state) {
	auto &queue = gstate.runtime_split_queue;
	while (queue && lstate.source->IsFinished() && !lstate.dynamic_no_more_splits) {
		auto split = queue->TryGetNext();
		if (split.state == distributed::FteSplitQueue::GetResult::BLOCKED) {
			if (queue->RegisterBlockedTask(interrupt_state)) {
				return SourceResultType::BLOCKED;
			}
			return SourceResultType::HAVE_MORE_OUTPUT;
		}
		if (split.state == distributed::FteSplitQueue::GetResult::CANCELED) {
			lstate.finished = true;
			return SourceResultType::FINISHED;
		}
		if (split.state == distributed::FteSplitQueue::GetResult::FINISHED) {
			lstate.dynamic_no_more_splits = true;
			return SourceResultType::FINISHED;
		}
		if (split.input.kind != distributed::TaskInput::Kind::ExchangeSourceTask) {
			throw InvalidInputException("remote exchange source dynamic queue received non-exchange split");
		}
		auto descriptor =
		    distributed::ExchangeSourceTaskDescriptor::DeserializeFromBytes(split.input.exchange_source_task_bytes);
		if (!descriptor.source_handles.empty()) {
			lstate.source->AddSourceHandles(std::move(descriptor.source_handles));
		}
	}
	return SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace

PhysicalRemoteExchangeSource::PhysicalRemoteExchangeSource(
    PhysicalPlan &physical_plan, vector<LogicalType> types, idx_t estimated_cardinality, std::string exchange_id,
    vector<idx_t> partition_indices, std::vector<distributed::ExchangeSourceHandle> source_handles,
    std::shared_ptr<distributed::ExchangeManager> exchange_mgr, const vector<std::string> &source_nodes,
    optional_idx runtime_source_node_id)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXCHANGE_SOURCE, std::move(types), estimated_cardinality),
      exchange_id_(std::move(exchange_id)), partition_indices_(std::move(partition_indices)),
      source_handles_(std::move(source_handles)), exchange_mgr_(std::move(exchange_mgr)), source_nodes_(source_nodes),
      runtime_source_node_id_(runtime_source_node_id) {
	if (!exchange_mgr_) {
		throw InvalidInputException("remote exchange source requires a non-null ExchangeManager");
	}
}

void PhysicalRemoteExchangeSource::ApplyRuntimeTaskDescriptor(
    const distributed::ExchangeSourceTaskDescriptor &descriptor) {
	partition_indices_ = descriptor.partition_indices;
	source_handles_ = descriptor.source_handles;
	runtime_source_partition_count_ = descriptor.source_partition_count;
	runtime_source_task_count_ = descriptor.source_task_count;
}

void PhysicalRemoteExchangeSource::ApplyRuntimeSplitQueue(std::shared_ptr<distributed::FteSplitQueue> queue) {
	runtime_split_queue_ = std::move(queue);
}

unique_ptr<GlobalSourceState> PhysicalRemoteExchangeSource::GetGlobalSourceState(ClientContext &context) const {
	if (runtime_source_node_id_.IsValid() && source_handles_.empty() && !runtime_split_queue_) {
		throw InvalidInputException("remote exchange source missing runtime binding for source node " +
		                            std::to_string(runtime_source_node_id_.GetIndex()));
	}

	// Pass worker-side ClientContext to the ExchangeManager so that
	// FlightExchangeSource receives a valid context for ReadPartition().
	// During deserialization the manager is created without a context.
	exchange_mgr_->SetContext(&context);
	auto max_threads =
	    ResolveRemoteExchangeMaxThreads(context, source_handles_, partition_indices_, runtime_source_partition_count_,
	                                    runtime_source_task_count_, runtime_split_queue_);
	return make_uniq<RemoteSourceGlobalState>(exchange_mgr_, source_handles_, runtime_split_queue_, max_threads);
}

unique_ptr<LocalSourceState> PhysicalRemoteExchangeSource::GetLocalSourceState(ExecutionContext &context,
                                                                               GlobalSourceState &gstate) const {
	auto &remote_state = gstate.Cast<RemoteSourceGlobalState>();
	std::lock_guard<std::mutex> lock(remote_state.source_lock);
	auto source = exchange_mgr_->CreateSource();
	if (!source) {
		throw IOException("[RemoteExchangeSource] ExchangeManager::CreateSource returned null for exchange_id=" +
		                  exchange_id_);
	}
	return make_uniq<RemoteSourceLocalState>(std::move(source));
}

SourceResultType PhysicalRemoteExchangeSource::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                               OperatorSourceInput &input) const {
	auto &gstate = input.global_state.Cast<RemoteSourceGlobalState>();
	auto &lstate = input.local_state.Cast<RemoteSourceLocalState>();

	if (lstate.finished) {
		chunk.SetCardinality(0);
		return SourceResultType::FINISHED;
	}

	while (true) {
		if (gstate.runtime_split_queue) {
			auto assign_result = AssignNextDynamicSplit(gstate, lstate, input.interrupt_state);
			if (assign_result == SourceResultType::BLOCKED) {
				chunk.SetCardinality(0);
				return SourceResultType::BLOCKED;
			}
			if (assign_result == SourceResultType::FINISHED && lstate.source->IsFinished()) {
				lstate.finished = true;
				chunk.SetCardinality(0);
				return SourceResultType::FINISHED;
			}
		} else if (lstate.source->IsFinished()) {
			if (!AssignNextStaticHandle(gstate, lstate)) {
				lstate.finished = true;
				chunk.SetCardinality(0);
				return SourceResultType::FINISHED;
			}
		}

		// Block until data is available
		if (lstate.source->IsBlocked()) {
			lstate.source->WaitUnblocked();
		}

		// Check if finished after potential wait. For dynamic queues, this may
		// mean only the current split is done, so loop back and request another.
		if (lstate.source->IsFinished()) {
			if (gstate.runtime_split_queue && !lstate.dynamic_no_more_splits) {
				continue;
			}
			if (!gstate.runtime_split_queue && !gstate.source_handles.empty()) {
				continue;
			}
			lstate.finished = true;
			chunk.SetCardinality(0);
			return SourceResultType::FINISHED;
		}

		bool got_data = lstate.source->ReadChunk(chunk);
		if (got_data) {
			return SourceResultType::HAVE_MORE_OUTPUT;
		}

		if (lstate.source->IsFinished()) {
			continue;
		}
		// Not finished but no data right now — keep trying later.
		if (gstate.runtime_split_queue && !lstate.dynamic_no_more_splits) {
			auto assign_result = AssignNextDynamicSplit(gstate, lstate, input.interrupt_state);
			if (assign_result == SourceResultType::BLOCKED) {
				chunk.SetCardinality(0);
				return SourceResultType::BLOCKED;
			}
		}
		chunk.SetCardinality(0);
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
}

InsertionOrderPreservingMap<string> PhysicalRemoteExchangeSource::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["exchange_id"] = exchange_id_;
	std::string parts_str;
	for (idx_t i = 0; i < partition_indices_.size(); i++) {
		if (i > 0)
			parts_str += ",";
		parts_str += std::to_string(partition_indices_[i]);
	}
	result["partition_indices"] = parts_str;
	result["type"] = "remote_exchange";
	return result;
}

void PhysicalRemoteExchangeSource::SerializeOperatorData(Serializer &serializer) const {
	vector<idx_t> handle_partition_ids;
	vector<string> handle_node_ids;
	vector<string> handle_paths;
	vector<int> handle_flight_ports;
	vector<idx_t> handle_attempt_ids;
	vector<string> local_dirs;
	if (exchange_mgr_) {
		auto flight_mgr = std::dynamic_pointer_cast<distributed::FlightExchangeManager>(exchange_mgr_);
		if (flight_mgr) {
			local_dirs.insert(local_dirs.end(), flight_mgr->config().local_dirs.begin(),
			                  flight_mgr->config().local_dirs.end());
		}
	}
	handle_partition_ids.reserve(source_handles_.size());
	handle_node_ids.reserve(source_handles_.size());
	handle_paths.reserve(source_handles_.size());
	handle_flight_ports.reserve(source_handles_.size());
	handle_attempt_ids.reserve(source_handles_.size());
	for (const auto &handle : source_handles_) {
		handle_partition_ids.push_back(handle.partition_id);
		handle_node_ids.push_back(handle.node_id);
		handle_paths.push_back(handle.files.empty() ? string() : handle.files[0].path);
		handle_flight_ports.push_back(handle.flight_port);
		handle_attempt_ids.push_back(handle.attempt_id);
	}
	serializer.WriteProperty(103, "shuffle_stage_id", exchange_id_);
	serializer.WriteProperty(104, "partition_indices", partition_indices_);
	serializer.WriteProperty(105, "source_nodes", source_nodes_);
	serializer.WriteProperty(106, "flight_location_template",
	                         std::string("grpc://{node}:") +
	                             std::to_string(distributed::ResolveFlightExchangeEnvInt("DUCKDB_FLIGHT_PORT", 0)));
	serializer.WriteProperty(107, "flight_timeout_seconds", 0.0);
	serializer.WriteProperty(108, "source_handle_partition_ids", handle_partition_ids);
	serializer.WriteProperty(109, "source_handle_node_ids", handle_node_ids);
	serializer.WriteProperty(110, "source_handle_paths", handle_paths);
	serializer.WriteProperty(111, "source_handle_flight_ports", handle_flight_ports);
	serializer.WritePropertyWithDefault(112, "runtime_source_node_id", runtime_source_node_id_, optional_idx());
	serializer.WriteProperty(113, "source_handle_attempt_ids", handle_attempt_ids);
	serializer.WriteProperty(114, "local_dirs", local_dirs);
}

} // namespace duckdb
