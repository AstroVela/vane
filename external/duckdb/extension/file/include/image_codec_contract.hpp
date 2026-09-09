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

	static void CheckDecodedBytes(uint64_t width, uint64_t height, idx_t source_width, uint16_t output_channels,
	                              idx_t output_width, idx_t storage_width, idx_t limit) {
		// Reserve decoder working/source pixels, converted pixels, and canonical
		// column storage. This also bounds Python's pixel-to-spool copy, and
		// generic IMAGE's native-dtype conversion scratch before widening.
		auto bytes_per_pixel = source_width * 2 + output_channels * (storage_width + output_width);
		if (width > limit / bytes_per_pixel / height) {
			throw OutOfRangeException("Image decoder exceeds max_decoded_bytes");
		}
	}
};

} // namespace duckdb
