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

#include "duckdb/common/open_file_info.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"

namespace duckdb {

class PhysicalPlan;

namespace distributed {

class FteSplitQueue;

struct ScanTaskDescriptor {
	vector<OpenFileInfo> files;
	idx_t estimated_cardinality = 0;
	idx_t estimated_bytes = 0;

	idx_t file_count() const {
		return static_cast<idx_t>(files.size());
	}

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

} // namespace distributed
} // namespace duckdb
