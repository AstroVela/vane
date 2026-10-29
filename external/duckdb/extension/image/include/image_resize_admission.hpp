// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "image_transform.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/managed_media_scalar.hpp"
#include "duckdb/execution/media_buffer.hpp"

namespace duckdb {

//! Shared by the optional backend and the provider-independent executor tests.
struct ManagedImageResize {
	static ManagedMediaWorkingSet WorkingSet(ClientContext &context, const BoundFunctionExpression &expression) {
		const auto &input = expression.children[0]->return_type;
		const auto &output = expression.return_type;
		if (!ImageLogicalType::IsFixedShape(input) || !ImageLogicalType::IsFixedShape(output)) {
			throw NotImplementedException(
			    "Managed native_resize requires fixed-shape IMAGE input and constant dimensions");
		}
		auto mode = ImageLogicalType::GetMode(input);
		if (ImageLogicalType::StorageType(input) != ImageLogicalType::PixelType(mode)) {
			throw NotImplementedException("Managed native_resize requires canonical IMAGE pixel storage");
		}
		for (idx_t i = 1; i < expression.children.size(); i++) {
			if (!expression.children[i]->IsFoldable()) {
				throw NotImplementedException("Managed native_resize dimensions and antialias must be constants");
			}
		}
		const auto width = ImageLogicalType::GetWidth(output);
		const auto height = ImageLogicalType::GetHeight(output);
		const auto channels = ImageLogicalType::ChannelsForMode(mode);
		auto bytes = ImageOperatorContract::CheckSize(width, height, channels, ImageOperatorContract::MAX_BYTES,
		                                              ImageLogicalType::ElementSize(mode));
		bool antialias = false;
		if (expression.children.size() == 4) {
			auto option = ExpressionExecutor::EvaluateScalar(context, *expression.children[3]);
			antialias = !option.IsNull() && option.GetValue<bool>();
		}
		idx_t scratch = 0;
		const auto input_width = ImageLogicalType::GetWidth(input);
		const auto input_height = ImageLogicalType::GetHeight(input);
		if (antialias && (width < input_width || height < input_height)) {
			const bool horizontal = uint64_t(width) * input_height <= uint64_t(input_width) * height;
			scratch =
			    ImageOperatorContract::CheckSize(horizontal ? width : input_width, horizontal ? input_height : height,
			                                     channels, ImageOperatorContract::MAX_BYTES, sizeof(double));
		}
		return {bytes, scratch};
	}

	static void Execute(ClientContext &context, DataChunk &args, Vector &result, MediaResourceLease lease,
	                    shared_ptr<void> lifetime) {
		D_ASSERT(args.size() == 1);
		AllocateMediaArray(result, 1, std::move(lease), std::move(lifetime));
		ImageOperatorInput images(args.data[0], 1, &context);
		bool antialias = false;
		if (args.ColumnCount() == 4) {
			auto option = args.data[3].GetValue(0);
			if (option.IsNull()) {
				FlatVector::SetNull(result, 0, true);
				return;
			}
			antialias = option.GetValue<bool>();
		}
		ImagePixelView image;
		if (!images.Read(0, image)) {
			FlatVector::SetNull(result, 0, true);
			return;
		}
		auto layout = image.layout;
		layout.width = ImageLogicalType::GetWidth(result.GetType());
		layout.height = ImageLogicalType::GetHeight(result.GetType());
		auto target = FlatVector::GetData(ArrayVector::GetEntry(result));
		ResizeImagePixels(
		    image, layout, target, [&context]() { ImageOperatorContract::Interrupt(context); }, antialias);
	}
};

} // namespace duckdb
