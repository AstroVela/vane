// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "image_transform_contract.hpp"

namespace duckdb {

struct ImageCodecContract {
	static string Format(const string &value) {
		auto format = StringUtil::Upper(value);
		if (format != "PNG" && format != "JPEG" && format != "TIFF" && format != "GIF" && format != "BMP") {
			throw InvalidInputException("Image format must be PNG, JPEG, TIFF, GIF, or BMP");
		}
		return format;
	}

	static void CheckEncoding(const string &format, const ImageLayout &layout) {
		auto mode = ImageLogicalType::ModeName(layout.mode);
		bool supported = format == "TIFF" || (format == "PNG" && layout.mode <= 8) ||
		                 ((format == "JPEG" || format == "BMP" || format == "GIF") && (mode == "L" || mode == "RGB"));
		if (!supported) {
			throw InvalidInputException("%s encoding does not support mode %s; use convert_image explicitly", format,
			                            mode);
		}
	}

	static bool OnError(const string &value) {
		if (value != "raise" && value != "null") {
			throw InvalidInputException("decode_image on_error must be 'raise' or 'null'");
		}
		return value == "null";
	}

	static unique_ptr<FunctionData> BindDecode(ClientContext &context, ScalarFunction &function,
	                                           vector<unique_ptr<Expression>> &arguments) {
		auto type = arguments[0]->return_type;
		if (type.id() == LogicalTypeId::UNKNOWN) {
			throw ParameterNotResolvedException();
		}
		if (type.id() != LogicalTypeId::BLOB && type.id() != LogicalTypeId::SQLNULL) {
			throw BinderException("decode_image requires BINARY input, not %s", type);
		}
		function.arguments[0] = LogicalType::BLOB;
		function.return_type = ImageLogicalType::Create();
		for (idx_t i = 1; i < 3; i++) {
			if (arguments[i]->IsFoldable()) {
				auto value = ExpressionExecutor::EvaluateScalar(context, *arguments[i]);
				if (!value.IsNull()) {
					if (i == 1) {
						OnError(value.GetValue<string>());
					} else {
						auto mode = value.GetValue<string>();
						ImageLogicalType::ModeCode(mode);
						function.return_type = ImageLogicalType::Create(mode);
					}
				}
			}
		}
		return nullptr;
	}

	static idx_t OutputSize(const LogicalType &type, const ImageLayout &layout, idx_t remaining) {
		return ImageOperatorContract::CheckSize(layout.width, layout.height, layout.channels, remaining,
		                                        GetTypeIdSize(ImageLogicalType::StorageType(type).InternalType()));
	}
};

} // namespace duckdb
