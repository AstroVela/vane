// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "image_codec.hpp"
#include "image_hash.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

namespace duckdb {
namespace {

static void Decode(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &context = state.GetContext();
	auto constant = args.AllConstant();
	auto count = constant && args.size() ? idx_t(1) : args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	UnifiedVectorFormat input, modes, policies;
	args.data[0].ToUnifiedFormat(count, input);
	args.data[1].ToUnifiedFormat(count, policies);
	args.data[2].ToUnifiedFormat(count, modes);
	idx_t retained = 0;
	for (idx_t row = 0; row < count; row++) {
		MediaInterrupt(context);
		auto source = input.sel->get_index(row), policy_index = policies.sel->get_index(row),
		     mode_index = modes.sel->get_index(row);
		if (!input.validity.RowIsValid(source) || !policies.validity.RowIsValid(policy_index)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto null_on_error =
		    ImageCodecContract::OnError(UnifiedVectorFormat::GetData<string_t>(policies)[policy_index].GetString());
		string mode;
		if (modes.validity.RowIsValid(mode_index)) {
			mode = UnifiedVectorFormat::GetData<string_t>(modes)[mode_index].GetString();
			ImageLogicalType::ModeCode(mode);
		}
		auto bytes = UnifiedVectorFormat::GetData<string_t>(input)[source];
		try {
			auto decoded =
			    NativeImageCodec::Decode(context, const_data_ptr_cast(bytes.GetData()), bytes.GetSize(),
			                             result.GetType(), mode, ImageOperatorContract::MAX_BYTES - retained);
			retained += NativeImageCodec::Write(context, decoded, mode, result, row,
			                                    ImageOperatorContract::MAX_BYTES - retained);
		} catch (const MediaFormatException &) {
			MediaInterrupt(context);
			if (!null_on_error) {
				throw;
			}
			FlatVector::SetNull(result, row, true);
		}
	}
	if (constant && count) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

static void HashImage(DataChunk &args, ExpressionState &state, Vector &result) {
	auto &context = state.GetContext();
	auto &options = state.expr.Cast<BoundFunctionExpression>().bind_info->Cast<ImageHashOptions>();
	auto constant = args.AllConstant();
	auto count = constant && args.size() ? idx_t(1) : args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	ImageOperatorInput images(args.data[0], count, &context);
	NativeImageHash hash(context, options);
	for (idx_t row = 0; row < count; row++) {
		MediaInterrupt(context);
		ImagePixelView image;
		if (!images.Read(row, image)) {
			FlatVector::SetNull(result, row, true);
			continue;
		}
		auto value = hash.Compute(image);
		FlatVector::GetData<string_t>(result)[row] = StringVector::AddStringOrBlob(result, value.data(), value.size());
		FlatVector::SetNull(result, row, false);
	}
	if (constant && count) {
		result.SetVectorType(VectorType::CONSTANT_VECTOR);
	}
}

} // namespace

void RegisterImageComputeFunctions(ExtensionLoader &loader) {
	ScalarFunction decode("native_decode_image", {LogicalType::ANY, LogicalType::VARCHAR, LogicalType::VARCHAR},
	                      LogicalType::ANY, Decode, ImageCodecContract::BindDecode);
	ScalarFunction hash("native_image_hash",
	                    {LogicalType::ANY, LogicalType::VARCHAR, LogicalType::ANY, LogicalType::ANY, LogicalType::ANY},
	                    LogicalType::ANY, HashImage, ImageHashOptions::Bind);
	for (auto function : {&decode, &hash}) {
		function->SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
		function->SetStability(FunctionStability::VOLATILE);
		function->SetFallible();
		loader.RegisterFunction(*function);
	}
}

} // namespace duckdb
