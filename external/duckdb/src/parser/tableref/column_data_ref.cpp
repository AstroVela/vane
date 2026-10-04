// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/parser/tableref/column_data_ref.hpp"
#include "duckdb/common/string_util.hpp"

namespace duckdb {

ColumnDataRef::ColumnDataRef(optionally_owned_ptr<ColumnDataCollection> collection_p, vector<string> expected_names)
    : TableRef(TableReferenceType::COLUMN_DATA), expected_names(std::move(expected_names)),
      collection(std::move(collection_p)) {
}

string ColumnDataRef::ToString() const {
	// This is a query description, not serialization of its materialized rows.
	// Scanning and stringifying the collection can box millions of Tensor pixels
	// merely to bind a relation or export it to Arrow. Describe its schema instead.
	string result = "ColumnDataCollection [" + to_string(collection->Count()) + " rows; ";
	auto &types = collection->Types();
	for (idx_t i = 0; i < types.size(); i++) {
		result += (i ? ", " : "") + types[i].ToString();
	}
	result += "]";
	return BaseToString(result, expected_names);
}

bool ColumnDataRef::Equals(const TableRef &other_p) const {
	if (!TableRef::Equals(other_p)) {
		return false;
	}
	auto &other = other_p.Cast<ColumnDataRef>();
	auto expected_types = collection->Types();
	auto other_expected_types = other.collection->Types();
	if (expected_types.size() != other_expected_types.size()) {
		return false;
	}
	if (expected_names.size() != other.expected_names.size()) {
		return false;
	}
	D_ASSERT(expected_types.size() == expected_names.size());
	for (idx_t i = 0; i < expected_types.size(); i++) {
		auto &this_type = expected_types[i];
		auto &other_type = other_expected_types[i];

		auto &this_name = expected_names[i];
		auto &other_name = other.expected_names[i];

		if (this_type != other_type) {
			return false;
		}
		if (!StringUtil::CIEquals(this_name, other_name)) {
			return false;
		}
	}
	string unused;
	if (!ColumnDataCollection::ResultEquals(*collection, *other.collection, unused, true)) {
		return false;
	}
	return true;
}

optionally_owned_ptr<ColumnDataCollection> CopyCollection(optionally_owned_ptr<ColumnDataCollection> &collection) {
	auto &unique = collection.get_owned_unique();
	if (unique) {
		// uniquely owned - need to copy over all the data and make a new collection
		auto new_collection = make_uniq<ColumnDataCollection>(collection->GetAllocator(), collection->Types());
		for (auto &chunk : collection->Chunks()) {
			new_collection->Append(chunk);
		}
		return std::move(new_collection);
	}
	auto &shared = collection.get_owned_shared();
	if (shared) {
		// shared ptr - we can directly reference it
		return shared;
	}
	// unowned collection - just return the raw reference
	return collection.get();
}

unique_ptr<TableRef> ColumnDataRef::Copy() {
	auto copied_collection = CopyCollection(collection);
	auto result = make_uniq<ColumnDataRef>(std::move(copied_collection), expected_names);
	CopyProperties(*result);
	return std::move(result);
}

} // namespace duckdb
