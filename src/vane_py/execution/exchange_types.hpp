// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/common/types.hpp"
#include "duckdb/common/exception.hpp"

namespace duckdb {
namespace vane_execution {

inline void CheckExchangeType(const LogicalType &type, idx_t depth = 0) {
	if (depth > 32 || type.HasAlias()) {
		throw NotImplementedException("unsupported exchange type: %s", type.ToString());
	}
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::HUGEINT:
	case LogicalTypeId::UTINYINT:
	case LogicalTypeId::USMALLINT:
	case LogicalTypeId::UINTEGER:
	case LogicalTypeId::UBIGINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::DECIMAL:
	case LogicalTypeId::DATE:
	case LogicalTypeId::TIME:
	case LogicalTypeId::TIMESTAMP:
	case LogicalTypeId::TIMESTAMP_SEC:
	case LogicalTypeId::TIMESTAMP_MS:
	case LogicalTypeId::TIMESTAMP_NS:
	case LogicalTypeId::TIMESTAMP_TZ:
	case LogicalTypeId::INTERVAL:
	case LogicalTypeId::VARCHAR:
	case LogicalTypeId::BLOB:
	case LogicalTypeId::SQLNULL:
		return;
	case LogicalTypeId::LIST:
	case LogicalTypeId::MAP:
		CheckExchangeType(ListType::GetChildType(type), depth + 1);
		return;
	case LogicalTypeId::STRUCT:
		for (auto &child : StructType::GetChildTypes(type)) {
			CheckExchangeType(child.second, depth + 1);
		}
		return;
	case LogicalTypeId::ARRAY:
		CheckExchangeType(ArrayType::GetChildType(type), depth + 1);
		return;
	default:
		throw NotImplementedException("unsupported exchange type: %s", type.ToString());
	}
}

} // namespace vane_execution
} // namespace duckdb
