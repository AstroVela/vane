// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/execution/dynamic_filter_serialization.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/serializer/serialization_data.hpp"
#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/planner/table_filter.hpp"

namespace duckdb {

struct DynamicTableFilterSerializationState : public SerializationData::CustomData {
	static string GetType() {
		return "DynamicTableFilterSerializationState";
	}

	optional_idx GetId(const shared_ptr<DynamicTableFilterSet> &filters) {
		if (!filters) {
			return optional_idx();
		}
		auto it = ids.find(filters.get());
		if (it != ids.end()) {
			return optional_idx(it->second);
		}
		auto id = next_id++;
		ids.emplace(filters.get(), id);
		filters_by_id.emplace(id, filters);
		return optional_idx(id);
	}

	shared_ptr<DynamicTableFilterSet> GetFilters(const optional_idx &id) {
		if (!id.IsValid()) {
			return nullptr;
		}
		auto idx = id.GetIndex();
		auto it = filters_by_id.find(idx);
		if (it != filters_by_id.end()) {
			return it->second;
		}
		auto filters = make_shared_ptr<DynamicTableFilterSet>();
		filters_by_id.emplace(idx, filters);
		return filters;
	}

private:
	unordered_map<const DynamicTableFilterSet *, idx_t> ids;
	unordered_map<idx_t, shared_ptr<DynamicTableFilterSet>> filters_by_id;
	idx_t next_id = 0;
};

} // namespace duckdb
