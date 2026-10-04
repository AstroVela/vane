// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/distributed/plan/scan_split.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "duckdb/common/open_file_info.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/set.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/distributed_table_function.hpp"

namespace duckdb {

class PhysicalPlan;

namespace distributed {

class FteSplitQueue;

enum class ScanSplitKind : uint8_t { FILE = 0, EXTENSION = 1 };

//! One stable, independently assignable unit of scan work. A split owns source
//! identity and estimates; it never owns an FTE task or a collection of other
//! splits. The explicit empty marker is the no-work assignment used to execute
//! an otherwise empty source exactly once.
struct ScanSplit {
	ScanSplitKind kind = ScanSplitKind::FILE;
	string split_id;
	bool empty = false;
	OpenFileInfo file;
	DistributedExtensionCapabilityReference extension_capability;
	DistributedPayloadCodec split_codec;
	string extension_payload;
	optional_idx estimated_cardinality;
	optional_idx estimated_bytes;

	static ScanSplit File(string split_id, OpenFileInfo file, optional_idx estimated_cardinality = optional_idx(),
	                      optional_idx estimated_bytes = optional_idx());
	static ScanSplit Extension(string split_id, string payload, DistributedExtensionCapabilityReference capability,
	                           DistributedPayloadCodec split_codec, optional_idx estimated_cardinality = optional_idx(),
	                           optional_idx estimated_bytes = optional_idx());
	static ScanSplit EmptyFile();
	static ScanSplit EmptyExtension(DistributedExtensionCapabilityReference capability,
	                                DistributedPayloadCodec split_codec);

	bool IsExtension() const {
		return kind == ScanSplitKind::EXTENSION;
	}

	void Validate() const;
	void Serialize(Serializer &serializer) const;
	static ScanSplit Deserialize(Deserializer &deserializer);
};

//! Transport container for scan splits belonging to one source. Source
//! planning emits one ScanSplit at a time and may wrap it in a singleton batch
//! for transport. Only the scheduler may combine multiple splits into the batch
//! assigned to one FTE task attempt.
struct ScanSplitBatch {
	vector<ScanSplit> splits;

	idx_t split_count() const {
		return static_cast<idx_t>(splits.size());
	}
	idx_t file_count() const;
	bool IsExtension() const;
	idx_t EstimatedCardinality() const;
	idx_t EstimatedBytes() const;

	void Validate() const;
	void Merge(ScanSplitBatch other);
	vector<ScanSplitBatch> Explode() const;
	void Serialize(Serializer &serializer) const;
	static ScanSplitBatch Deserialize(Deserializer &deserializer);

	std::string SerializeToBytes() const;
	std::string SerializeToBase64() const;
	static ScanSplitBatch DeserializeFromBytes(const std::string &bytes);
	static ScanSplitBatch DeserializeFromBase64(const std::string &base64);
};

bool ApplyScanSplitBatchesToPlan(duckdb::PhysicalPlan &plan, const std::unordered_map<idx_t, ScanSplitBatch> &batches,
                                 std::string *error = nullptr);

bool ApplyFteScanSourceQueuesToPlan(duckdb::PhysicalPlan &plan,
                                    const std::unordered_map<idx_t, std::shared_ptr<FteSplitQueue>> &queues,
                                    std::string *error = nullptr);

//! Validate the complete static-plus-FTE assignment domain before either
//! assignment mechanism mutates the worker plan.
bool ValidateScanSplitAssignments(const duckdb::PhysicalPlan &plan, const set<idx_t> &assigned_node_ids,
                                  string *error = nullptr);

//! Require every distributed table scan in a worker plan to have received an
//! explicit split assignment. An explicit empty split is the legal empty scan.
bool ValidateDistributedScanSplitsApplied(const duckdb::PhysicalPlan &plan, string *error = nullptr);

//! Returns true when the plan contains a table scan tagged as a distributed
//! worker scan target.
bool HasDistributedScanSplitTargets(const duckdb::PhysicalPlan &plan);

} // namespace distributed
} // namespace duckdb
