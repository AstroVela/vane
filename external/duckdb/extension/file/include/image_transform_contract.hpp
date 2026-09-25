// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "image_operator_contract.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/expression_executor.hpp"

namespace duckdb {

enum class ImageTransform { RESIZE, CONVERT };

//! Type inference, argument validation and bounded engine output are shared.
//! Each backend supplies its own pixel implementation.
struct ImageTransformContract {
	static uint32_t Dimension(int64_t dimension) {
		if (dimension <= 0 || uint64_t(dimension) > NumericLimits<uint32_t>::Maximum()) {
			throw InvalidInputException("resize width and height must be positive UINTEGER values");
		}
		return uint32_t(dimension);
	}

	static string Mode(const char *data, idx_t size) {
		if (size > 7) {
			throw InvalidInputException("Invalid Image mode");
		}
		auto mode = StringUtil::Upper(string(data, size));
		ImageLogicalType::ModeCode(mode);
		return mode;
	}

	static unique_ptr<FunctionData> BindResize(ClientContext &context, ScalarFunction &function,
	                                           vector<unique_ptr<Expression>> &arguments) {
		auto type = ImageOperatorContract::BindImage(function, arguments);
		auto mode = ImageLogicalType::GetMode(type);
		if (arguments.size() == 4) {
			auto id = arguments[3]->return_type.id();
			if (id == LogicalTypeId::UNKNOWN) {
				throw ParameterNotResolvedException();
			}
			if (id != LogicalTypeId::BOOLEAN && id != LogicalTypeId::SQLNULL) {
				throw BinderException("resize antialias must be a boolean");
			}
			function.arguments[3] = LogicalType::BOOLEAN;
		}
		uint32_t dimensions[2] = {};
		for (idx_t i = 1; i < 3; i++) {
			auto &argument = *arguments[i];
			if (argument.return_type.id() == LogicalTypeId::UNKNOWN) {
				throw ParameterNotResolvedException();
			}
			if (!argument.return_type.IsIntegral() && argument.return_type.id() != LogicalTypeId::SQLNULL) {
				throw BinderException("resize width and height must be integers");
			}
			function.arguments[i] = LogicalType::BIGINT;
			if (argument.IsFoldable()) {
				auto value = ExpressionExecutor::EvaluateScalar(context, argument);
				if (!value.IsNull()) {
					dimensions[i - 1] = Dimension(value.GetValue<int64_t>());
				}
			}
		}
		function.return_type = mode.empty() ? ImageLogicalType::Create() : ImageLogicalType::Create(mode);
		if (dimensions[0] && dimensions[1]) {
			ImageOperatorContract::CheckSize(
			    dimensions[0], dimensions[1], mode.empty() ? 1 : ImageLogicalType::ChannelsForMode(mode),
			    ImageOperatorContract::MAX_BYTES, mode.empty() ? 4 : ImageLogicalType::ElementSize(mode));
			if (!mode.empty()) {
				function.return_type = ImageLogicalType::Create(mode, dimensions[1], dimensions[0]);
			}
		}
		return nullptr;
	}

	static unique_ptr<FunctionData> BindConvert(ClientContext &context, ScalarFunction &function,
	                                            vector<unique_ptr<Expression>> &arguments) {
		auto type = ImageOperatorContract::BindImage(function, arguments);
		auto &argument = *arguments[1];
		if (argument.return_type.id() == LogicalTypeId::UNKNOWN) {
			throw ParameterNotResolvedException();
		}
		if (argument.return_type.id() != LogicalTypeId::VARCHAR &&
		    argument.return_type.id() != LogicalTypeId::SQLNULL) {
			throw BinderException("convert_image mode must be a string");
		}
		function.arguments[1] = LogicalType::VARCHAR;
		function.return_type = ImageLogicalType::Create();
		if (argument.IsFoldable()) {
			auto value = ExpressionExecutor::EvaluateScalar(context, argument);
			if (!value.IsNull()) {
				auto &text = StringValue::Get(value);
				auto mode = Mode(text.data(), text.size());
				function.return_type = ImageLogicalType::Create(mode);
				if (ImageLogicalType::IsFixedShape(type)) {
					auto width = ImageLogicalType::GetWidth(type);
					auto height = ImageLogicalType::GetHeight(type);
					ImageOperatorContract::CheckSize(width, height, ImageLogicalType::ChannelsForMode(mode),
					                                 ImageOperatorContract::MAX_BYTES,
					                                 mode.empty() ? 4 : ImageLogicalType::ElementSize(mode));
					function.return_type = ImageLogicalType::Create(mode, height, width);
				}
			}
		}
		return nullptr;
	}

	template <class EXECUTE>
	static void Execute(DataChunk &args, ClientContext &context, Vector &result, ImageTransform operation,
	                    EXECUTE execute) {
		auto constant = args.AllConstant();
		auto count = constant && args.size() ? idx_t(1) : args.size();
		result.SetVectorType(VectorType::FLAT_VECTOR);
		if (!count) {
			return;
		}
		ImageOperatorContract::Interrupt(context);
		auto &type = result.GetType();
		auto fixed = ImageLogicalType::IsFixedShape(type);
		if (fixed) {
			// ARRAY storage includes NULL rows. Reject an oversized fixed
			// output batch before allocating pixels or doing any pixel work.
			ImageOperatorContract::CheckSize(ImageLogicalType::GetWidth(type), ImageLogicalType::GetHeight(type),
			                                 ImageLogicalType::ChannelsForMode(ImageLogicalType::GetMode(type)),
			                                 ImageOperatorContract::MAX_BYTES / count,
			                                 GetTypeIdSize(ImageLogicalType::StorageType(type).InternalType()));
		}
		ImageOperatorInput images(args.data[0], count, &context);
		UnifiedVectorFormat first;
		UnifiedVectorFormat second;
		UnifiedVectorFormat antialias;
		auto has_antialias = operation == ImageTransform::RESIZE && args.ColumnCount() == 4;
		if (has_antialias) {
			args.data[3].ToUnifiedFormat(count, antialias);
		}
		args.data[1].ToUnifiedFormat(count, first);
		if (operation == ImageTransform::RESIZE) {
			args.data[2].ToUnifiedFormat(count, second);
		}
		idx_t bytes = 0;
		for (idx_t row = 0; row < count; row++) {
			ImageOperatorContract::Interrupt(context);
			auto first_index = first.sel->get_index(row);
			auto second_index = operation == ImageTransform::RESIZE ? second.sel->get_index(row) : 0;
			auto antialias_index = has_antialias ? antialias.sel->get_index(row) : 0;
			if (images.IsNull(row) || !first.validity.RowIsValid(first_index) ||
			    (operation == ImageTransform::RESIZE && !second.validity.RowIsValid(second_index)) ||
			    (has_antialias && !antialias.validity.RowIsValid(antialias_index))) {
				result.SetValue(row, Value(type));
				continue;
			}
			ImagePixelView image;
			images.Read(row, image);
			auto layout = image.layout;
			if (operation == ImageTransform::RESIZE) {
				layout.width = Dimension(UnifiedVectorFormat::GetData<int64_t>(first)[first_index]);
				layout.height = Dimension(UnifiedVectorFormat::GetData<int64_t>(second)[second_index]);
			} else {
				auto text = UnifiedVectorFormat::GetData<string_t>(first)[first_index];
				auto mode = Mode(text.GetData(), text.GetSize());
				layout.mode = ImageLogicalType::ModeCode(mode);
				layout.channels = ImageLogicalType::ChannelsForMode(mode);
			}
			if (!fixed) {
				bytes += ImageOperatorContract::CheckSize(
				    layout.width, layout.height, layout.channels, ImageOperatorContract::MAX_BYTES - bytes,
				    GetTypeIdSize(ImageLogicalType::StorageType(type).InternalType()));
			}
			ImageOperatorOutput output(result, row, layout);
			execute(image, layout, output.Data(),
			        has_antialias && UnifiedVectorFormat::GetData<bool>(antialias)[antialias_index]);
			output.Finish(context);
			ImageOperatorContract::Interrupt(context);
		}
		if (constant) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
	}
};

} // namespace duckdb
