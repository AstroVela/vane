// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "image_operator_contract.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/execution/expression_executor_state.hpp"

namespace duckdb {

//! Base Image/Tensor storage conversion. No codec, optional backend or Python callback.
struct ImageToTensor {
	static LogicalType ResultType(const LogicalType &image) {
		auto mode = ImageLogicalType::GetMode(image);
		vector<idx_t> shape(3, TensorType::VARIABLE_DIMENSION);
		if (!mode.empty()) {
			shape[2] = ImageLogicalType::ChannelsForMode(mode);
		}
		if (ImageLogicalType::IsFixedShape(image)) {
			shape[0] = ImageLogicalType::GetHeight(image);
			shape[1] = ImageLogicalType::GetWidth(image);
		}
		return TensorType::Create(LogicalType::UTINYINT, shape);
	}

	static unique_ptr<FunctionData> Bind(ClientContext &, ScalarFunction &function,
	                                     vector<unique_ptr<Expression>> &arguments) {
		function.return_type = ResultType(ImageOperatorContract::BindImage(function, arguments));
		return nullptr;
	}

	static void Execute(DataChunk &args, ClientContext &context, Vector &result) {
		const auto constant = args.AllConstant();
		const auto count = constant && args.size() ? idx_t(1) : args.size();
		result.SetVectorType(VectorType::FLAT_VECTOR);
		if (!count) {
			return;
		}
		ImageOperatorContract::Interrupt(context);
		auto &input = args.data[0];
		ImageOperatorInput images(input, count);
		if (ImageLogicalType::IsFixedShape(input.GetType())) {
			for (idx_t row = 0; row < count; row++) {
				ImageOperatorContract::Interrupt(context);
				ImagePixelView image;
				images.Read(row, image);
			}
			// Identical UInt8 ARRAY storage, including constant/dictionary selection.
			result.Reinterpret(input);
			return;
		}

		auto &source_data = *StructVector::GetEntries(input)[ImageLogicalType::DATA];
		UnifiedVectorFormat source_entries;
		source_data.ToUnifiedFormat(count, source_entries);
		auto &children = StructVector::GetEntries(result);
		auto &data = *children[0];
		auto &shape = *children[1];
		data.SetVectorType(VectorType::FLAT_VECTOR);
		shape.SetVectorType(VectorType::FLAT_VECTOR);
		// Only offsets and three dimensions are produced per row. Pixel buffers
		// remain owned by the referenced LIST auxiliary, including after input release.
		auto *payload = &source_data;
		while (payload->GetVectorType() == VectorType::DICTIONARY_VECTOR) {
			payload = &DictionaryVector::Child(*payload);
		}
		ListVector::ReferenceEntry(data, *payload);
		auto entries = FlatVector::GetData<list_entry_t>(data);
		auto &dimensions = ArrayVector::GetEntryForWrite(shape, count);
		auto sizes = FlatVector::GetData<int32_t>(dimensions);
		FlatVector::Validity(result).SetAllValid(count);
		FlatVector::Validity(data).SetAllValid(count);
		FlatVector::Validity(shape).SetAllValid(count);
		FlatVector::Validity(dimensions).SetAllValid(count * 3);
		for (idx_t row = 0; row < count; row++) {
			ImageOperatorContract::Interrupt(context);
			ImagePixelView image;
			if (!images.Read(row, image)) {
				FlatVector::Validity(result).SetInvalid(row);
				FlatVector::Validity(data).SetInvalid(row);
				entries[row] = list_entry_t(0, 0);
				FlatVector::SetNull(shape, row, true);
				sizes[row * 3] = sizes[row * 3 + 1] = sizes[row * 3 + 2] = 0;
				continue;
			}
			entries[row] =
			    UnifiedVectorFormat::GetData<list_entry_t>(source_entries)[source_entries.sel->get_index(row)];
			sizes[row * 3] = NumericCast<int32_t>(image.layout.height);
			sizes[row * 3 + 1] = NumericCast<int32_t>(image.layout.width);
			sizes[row * 3 + 2] = image.layout.channels;
		}
		if (constant) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
	}

	static ScalarFunction Function() {
		ScalarFunction function(
		    "image_to_tensor", {LogicalType::ANY}, LogicalType::ANY,
		    [](DataChunk &args, ExpressionState &state, Vector &result) { Execute(args, state.GetContext(), result); },
		    Bind);
		function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
		// Keep this buffer operation in execution. Scalar constant folding would
		// expand a dense Tensor into one Value per pixel. Execute still preserves
		// constant input as a single payload per batch.
		function.SetStability(FunctionStability::VOLATILE);
		function.SetFallible();
		return function;
	}
};

} // namespace duckdb
