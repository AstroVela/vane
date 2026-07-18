// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// physical_remote_exchange_sink.cpp — Delegates to ExchangeManager SPI
//
//===----------------------------------------------------------------------===//

#include "duckdb/execution/operator/exchange/physical_remote_exchange_sink.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/distributed/exchange/flight_exchange_manager.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/string_type.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/function/create_sort_key.hpp"

#include <algorithm>
#include <cstring>

namespace duckdb {

namespace {

struct RemoteSinkGlobalState : public GlobalSinkState {
	explicit RemoteSinkGlobalState(std::unique_ptr<distributed::ExchangeSink> sink_p) : sink(std::move(sink_p)) {
	}

	std::unique_ptr<distributed::ExchangeSink> sink;
};

struct RemoteSinkLocalState : public LocalSinkState {
	explicit RemoteSinkLocalState(ClientContext &context, const PhysicalRemoteExchangeSink &op)
	    : executor(context), hash(LogicalType::HASH), sort_key(LogicalType::BLOB) {
		const auto &exprs = op.PartitionBy();
		if (!exprs.empty()) {
			vector<LogicalType> types;
			types.reserve(exprs.size());
			for (auto &expr : exprs) {
				executor.AddExpression(*expr);
				types.push_back(expr->return_type);
			}
			key_chunk.InitializeEmpty(types);
		}
		if (op.RepartitionType() == RepartitionSpec::Type::Range) {
			auto &modifier_strings = op.RangeOrderModifiers();
			if (exprs.size() != modifier_strings.size()) {
				throw InvalidInputException("range repartition requires one order modifier per partition expression");
			}
			range_modifiers.reserve(modifier_strings.size());
			for (auto &modifier : modifier_strings) {
				range_modifiers.push_back(OrderModifiers::Parse(modifier));
			}
		}
		selections.reserve(op.NumPartitions());
		for (idx_t i = 0; i < op.NumPartitions(); i++) {
			selections.emplace_back(STANDARD_VECTOR_SIZE);
		}
		sel_counts.resize(op.NumPartitions(), 0);
	}

	ExpressionExecutor executor;
	DataChunk key_chunk;
	Vector hash;
	Vector sort_key;
	vector<OrderModifiers> range_modifiers;
	vector<SelectionVector> selections;
	vector<idx_t> sel_counts;
	idx_t round_robin = 0;
};

int CompareSortKeys(const string_t &left, const std::string &right) {
	auto right_key = string_t(right.data(), UnsafeNumericCast<uint32_t>(right.size()));
	const auto left_size = left.GetSize();
	const auto right_size = right_key.GetSize();
	const auto min_size = std::min(left_size, right_size);
	auto cmp = std::memcmp(left.GetData(), right_key.GetData(), min_size);
	if (cmp != 0) {
		return cmp < 0 ? -1 : 1;
	}
	if (left_size == right_size) {
		return 0;
	}
	return left_size < right_size ? -1 : 1;
}

} // namespace

idx_t PhysicalRemoteExchangeSink::SelectPartitionHash(const hash_t hash, const idx_t num_partitions) {
	if (num_partitions == 1) {
		return 0;
	}
	return static_cast<idx_t>(hash % num_partitions);
}

idx_t PhysicalRemoteExchangeSink::SelectPartitionRange(const string_t &sort_key, const vector<string> &boundaries,
                                                       const idx_t num_partitions) {
	if (num_partitions <= 1 || boundaries.empty()) {
		return 0;
	}
	idx_t lower = 0;
	idx_t upper = std::min<idx_t>(boundaries.size(), num_partitions - 1);
	while (lower < upper) {
		auto mid = lower + (upper - lower) / 2;
		auto cmp = CompareSortKeys(sort_key, boundaries[mid]);
		if (cmp >= 0) {
			lower = mid + 1;
		} else {
			upper = mid;
		}
	}
	return std::min<idx_t>(lower, num_partitions - 1);
}

PhysicalRemoteExchangeSink::PhysicalRemoteExchangeSink(
    PhysicalPlan &physical_plan, vector<LogicalType> types, idx_t estimated_cardinality, std::string exchange_id,
    idx_t num_partitions, RepartitionSpec::Type repartition_type, vector<unique_ptr<Expression>> partition_by,
    distributed::ExchangeSinkInstanceHandle sink_handle, std::shared_ptr<distributed::ExchangeManager> exchange_mgr,
    vector<string> range_boundaries, vector<string> range_order_modifiers)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXCHANGE_SINK, std::move(types), estimated_cardinality),
      exchange_id_(std::move(exchange_id)), num_partitions_(num_partitions), repartition_type_(repartition_type),
      partition_by_(std::move(partition_by)), sink_handle_(std::move(sink_handle)),
      exchange_mgr_(std::move(exchange_mgr)), range_boundaries_(std::move(range_boundaries)),
      range_order_modifiers_(std::move(range_order_modifiers)) {
	if (num_partitions_ == 0) {
		throw InvalidInputException("remote exchange sink requires at least one partition");
	}
	if (!exchange_mgr_) {
		throw InvalidInputException("remote exchange sink requires a non-null ExchangeManager");
	}
	if (repartition_type_ == RepartitionSpec::Type::Range) {
		if (partition_by_.empty()) {
			throw InvalidInputException("range repartition requires partition expressions");
		}
		if (partition_by_.size() != range_order_modifiers_.size()) {
			throw InvalidInputException("range repartition requires one order modifier per partition expression");
		}
		if (range_boundaries_.size() >= num_partitions_) {
			throw InvalidInputException("range repartition requires fewer boundary keys than partitions");
		}
	}
}

unique_ptr<GlobalSinkState> PhysicalRemoteExchangeSink::GetGlobalSinkState(ClientContext &context) const {
	// Inject context for deserialized managers (context is not available during deserialization)
	exchange_mgr_->SetContext(&context);
	auto sink = exchange_mgr_->CreateSink(sink_handle_);
	if (!sink) {
		throw IOException("[RemoteExchangeSink] ExchangeManager::CreateSink returned null for exchange_id=" +
		                  exchange_id_);
	}
	return make_uniq<RemoteSinkGlobalState>(std::move(sink));
}

unique_ptr<LocalSinkState> PhysicalRemoteExchangeSink::GetLocalSinkState(ExecutionContext &context) const {
	return make_uniq<RemoteSinkLocalState>(context.client, *this);
}

SourceResultType PhysicalRemoteExchangeSink::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                             OperatorSourceInput &input) const {
	chunk.SetCardinality(0);
	return SourceResultType::FINISHED;
}

SinkResultType PhysicalRemoteExchangeSink::Sink(ExecutionContext &context, DataChunk &chunk,
                                                OperatorSinkInput &input) const {
	if (chunk.size() == 0) {
		return SinkResultType::NEED_MORE_INPUT;
	}

	auto &gstate = input.global_state.Cast<RemoteSinkGlobalState>();
	auto &lstate = input.local_state.Cast<RemoteSinkLocalState>();
	const auto count = chunk.size();
	const auto partitions = num_partitions_;

	std::fill(lstate.sel_counts.begin(), lstate.sel_counts.end(), 0);

	switch (repartition_type_) {
	case RepartitionSpec::Type::Hash: {
		if (partition_by_.empty()) {
			throw InvalidInputException("hash repartition requires partition expressions");
		}
		lstate.executor.Execute(chunk, lstate.key_chunk);
		lstate.key_chunk.Hash(lstate.hash);
		lstate.hash.Flatten(count);
		auto hashes = FlatVector::GetData<hash_t>(lstate.hash);
		for (idx_t row_idx = 0; row_idx < count; row_idx++) {
			auto part = SelectPartitionHash(hashes[row_idx], partitions);
			auto &sel_count = lstate.sel_counts[part];
			lstate.selections[part].set_index(sel_count++, row_idx);
		}
		break;
	}
	case RepartitionSpec::Type::Random:
	case RepartitionSpec::Type::IntoPartitions: {
		idx_t current = lstate.round_robin;
		for (idx_t row_idx = 0; row_idx < count; row_idx++) {
			auto part = current++ % partitions;
			auto &sel_count = lstate.sel_counts[part];
			lstate.selections[part].set_index(sel_count++, row_idx);
		}
		lstate.round_robin = current % partitions;
		break;
	}
	case RepartitionSpec::Type::Range: {
		if (partition_by_.empty()) {
			throw InvalidInputException("range repartition requires partition expressions");
		}
		lstate.executor.Execute(chunk, lstate.key_chunk);
		CreateSortKeyHelpers::CreateSortKey(lstate.key_chunk, lstate.range_modifiers, lstate.sort_key);
		lstate.sort_key.Flatten(count);
		auto sort_keys = FlatVector::GetData<string_t>(lstate.sort_key);
		for (idx_t row_idx = 0; row_idx < count; row_idx++) {
			auto part = SelectPartitionRange(sort_keys[row_idx], range_boundaries_, partitions);
			auto &sel_count = lstate.sel_counts[part];
			lstate.selections[part].set_index(sel_count++, row_idx);
		}
		break;
	}
	default:
		throw NotImplementedException("remote exchange sink: unsupported repartition type");
	}

	// Check backpressure from the ExchangeSink
	if (gstate.sink->IsBlocked()) {
		gstate.sink->WaitUnblocked();
	}

	for (idx_t part_idx = 0; part_idx < partitions; part_idx++) {
		auto part_count = lstate.sel_counts[part_idx];
		if (part_count == 0) {
			continue;
		}
		DataChunk part_chunk;
		part_chunk.InitializeEmpty(chunk.GetTypes());
		part_chunk.Slice(chunk, lstate.selections[part_idx], part_count);
		part_chunk.Flatten();
		auto result = gstate.sink->AddChunk(part_idx, part_chunk);
		if (result.is_err()) {
			throw IOException(std::string("[RemoteExchangeSink] AddChunk failed: ") + result.error().what());
		}
	}

	return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType PhysicalRemoteExchangeSink::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                      OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<RemoteSinkGlobalState>();
	auto result = gstate.sink->Finish();
	if (result.is_err()) {
		throw IOException(std::string("[RemoteExchangeSink] Finish failed: ") + result.error().what());
	}
	// Ensure schema.arrow is written even for 0-row exchanges so downstream
	// Flight readers can open the exchange without ENOENT errors.
	vector<string> col_names;
	col_names.reserve(types.size());
	for (idx_t i = 0; i < types.size(); i++) {
		col_names.push_back("col" + std::to_string(i));
	}
	auto schema_result = gstate.sink->EnsureSchema(context, types, col_names);
	return SinkFinalizeType::READY;
}

InsertionOrderPreservingMap<string> PhysicalRemoteExchangeSink::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["exchange_id"] = exchange_id_;
	result["num_partitions"] = std::to_string(num_partitions_);
	result["type"] = "remote_exchange";
	return result;
}

} // namespace duckdb

namespace duckdb {

void PhysicalRemoteExchangeSink::SerializeOperatorData(Serializer &serializer) const {
	// Write the same field layout as PhysicalExchangeSink so the worker-side
	// deserializer (EXCHANGE_SINK case) can reconstruct a working sink.
	serializer.WriteProperty(103, "shuffle_stage_id", exchange_id_);
	serializer.WriteProperty(104, "node_id", distributed::ResolveFlightExchangeNodeIdFromEnv());
	serializer.WriteProperty(105, "num_partitions", num_partitions_);
	serializer.WriteProperty(106, "repartition_type", static_cast<uint8_t>(repartition_type_));
	serializer.WriteProperty(107, "partition_by", partition_by_);
	auto local_dirs = distributed::ResolveFlightExchangeLocalDirsFromEnv();
	serializer.WriteProperty(108, "local_dirs", local_dirs);
	serializer.WriteProperty(109, "flight_bind_host", std::string("0.0.0.0"));
	serializer.WriteProperty(110, "flight_port", distributed::ResolveFlightExchangeEnvInt("DUCKDB_FLIGHT_PORT", 0));
	serializer.WriteProperty(111, "sink_task_partition_id", sink_handle_.sink_handle.task_partition_id);
	serializer.WriteProperty(112, "sink_attempt_id", sink_handle_.attempt_id);
	serializer.WriteProperty(113, "sink_output_location", sink_handle_.output_location);
	serializer.WriteProperty(114, "range_boundaries", range_boundaries_);
	serializer.WriteProperty(115, "range_order_modifiers", range_order_modifiers_);
}

} // namespace duckdb
