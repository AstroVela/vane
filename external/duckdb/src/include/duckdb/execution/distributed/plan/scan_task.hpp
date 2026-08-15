// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/distributed/plan/scan_task.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "duckdb/common/constants.hpp"
#include "duckdb/common/open_file_info.hpp"
#include "duckdb/common/set.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/function/distributed_table_function.hpp"

namespace duckdb {

class PhysicalPlan;

namespace distributed {

class FteSplitQueue;

enum class ScanTaskKind : uint8_t { FILES = 0, EXTENSION = 1 };

struct ScanTaskDescriptor {
	ScanTaskKind kind = ScanTaskKind::FILES;
	//! Used only by DuckDB's built-in MultiFile scan path.
	vector<OpenFileInfo> files;
	//! Used only by a TableFunction with explicit distributed scan callbacks.
	DistributedExtensionCapabilityReference extension_capability;
	DistributedPayloadCodec task_codec;
	vector<DistributedScanTask> extension_tasks;
	idx_t estimated_cardinality = 0;
	idx_t estimated_bytes = 0;
	//! Stable ordinal within the logical scan source. This distinguishes
	//! repeated occurrences of an otherwise identical file descriptor.
	idx_t source_task_partition_id = DConstants::INVALID_INDEX;

	idx_t file_count() const {
		return static_cast<idx_t>(files.size());
	}
	idx_t task_count() const {
		return kind == ScanTaskKind::FILES ? file_count() : static_cast<idx_t>(extension_tasks.size());
	}
	bool IsExtensionTask() const {
		return kind == ScanTaskKind::EXTENSION;
	}

	void Validate() const;
	void Merge(ScanTaskDescriptor other);
	void Serialize(Serializer &serializer) const;
	static ScanTaskDescriptor Deserialize(Deserializer &deserializer);

	std::string SerializeToBytes() const;
	std::string SerializeToBase64() const;
	static ScanTaskDescriptor DeserializeFromBytes(const std::string &bytes);
	static ScanTaskDescriptor DeserializeFromBase64(const std::string &base64);
};

bool ApplyScanTasksToPlan(duckdb::PhysicalPlan &plan, const std::unordered_map<idx_t, ScanTaskDescriptor> &tasks,
                          std::string *error = nullptr);

bool ApplyFteScanSourceQueuesToPlan(duckdb::PhysicalPlan &plan,
                                    const std::unordered_map<idx_t, std::shared_ptr<FteSplitQueue>> &queues,
                                    std::string *error = nullptr);

//! Validate the complete static-plus-FTE assignment domain before either
//! assignment mechanism mutates the worker plan.
bool ValidateScanTaskAssignments(const duckdb::PhysicalPlan &plan, const set<idx_t> &assigned_node_ids,
                                 std::string *error = nullptr);

//! Require every distributed table scan in a worker plan to have received an
//! explicit task assignment. A legal empty scan is represented by an applied
//! empty descriptor, including through an FTE queue.
bool ValidateDistributedScanTasksApplied(const duckdb::PhysicalPlan &plan, std::string *error = nullptr);

//! Returns true when the plan contains a table scan tagged as a distributed
//! worker scan target. Such a plan must not execute from its task-free worker
//! bind without an explicit static or FTE task assignment.
bool HasDistributedScanTaskTargets(const duckdb::PhysicalPlan &plan);

} // namespace distributed
} // namespace duckdb
