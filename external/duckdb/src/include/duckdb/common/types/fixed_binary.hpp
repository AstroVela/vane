// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/types/value.hpp"
#include "duckdb/common/extra_type_info.hpp"

namespace duckdb {

//! A BLOB with a known byte width, exported as Arrow FixedSizeBinary.
struct FixedBinaryType {
	static bool IsFixedBinary(const LogicalType &type) {
		return type.id() == LogicalTypeId::BLOB && type.HasAlias() && type.GetAlias() == "FIXEDBINARY";
	}
	static idx_t Size(const LogicalType &type) {
		if (!IsFixedBinary(type) || !type.HasExtensionInfo()) {
			throw InvalidInputException("Expected FIXEDBINARY with a byte width");
		}
		auto &modifiers = type.GetExtensionInfo()->modifiers;
		if (modifiers.size() != 1 || modifiers[0].value.IsNull() ||
		    modifiers[0].value.type() != LogicalType::UINTEGER) {
			throw InvalidInputException("Malformed FIXEDBINARY byte width");
		}
		auto size = modifiers[0].value.GetValue<uint32_t>();
		if (size > 2147483647) {
			throw InvalidInputException("FIXEDBINARY byte width must be between 0 and 2147483647");
		}
		return size;
	}
	static LogicalType Create(idx_t size) {
		if (size > 2147483647) {
			throw InvalidInputException("FIXEDBINARY byte width must be between 0 and 2147483647");
		}
		LogicalType type(LogicalType::BLOB);
		type.SetAlias("FIXEDBINARY");
		auto info = make_uniq<ExtensionTypeInfo>();
		info->modifiers.emplace_back(Value::UINTEGER(uint32_t(size)));
		type.SetExtensionInfo(std::move(info));
		return type;
	}
	static void Validate(const LogicalType &type, idx_t bytes) {
		if (Size(type) != bytes) {
			throw InvalidInputException("FIXEDBINARY(%d) requires exactly %d bytes, got %d", Size(type), Size(type),
			                            bytes);
		}
	}
};

} // namespace duckdb
