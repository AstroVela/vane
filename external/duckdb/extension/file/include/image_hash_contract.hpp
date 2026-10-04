// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "image_operator_contract.hpp"
#include "duckdb/common/types/fixed_binary.hpp"
#include "duckdb/execution/expression_executor.hpp"

namespace duckdb {

struct ImageHashOptions : FunctionData {
	string method;
	idx_t hash_size;
	idx_t binbits;
	idx_t segments;

	idx_t Bits() const {
		return method == "colorhash" ? 14 * binbits
		                             : hash_size * hash_size * (method == "crop_resistant" ? segments * segments : 1);
	}
	idx_t Bytes() const {
		return (Bits() + 7) / 8;
	}
	unique_ptr<FunctionData> Copy() const override {
		auto copy = make_uniq<ImageHashOptions>();
		copy->method = method;
		copy->hash_size = hash_size;
		copy->binbits = binbits;
		copy->segments = segments;
		return std::move(copy);
	}
	bool Equals(const FunctionData &other) const override {
		auto &value = other.Cast<ImageHashOptions>();
		return method == value.method && hash_size == value.hash_size && binbits == value.binbits &&
		       segments == value.segments;
	}
	static unique_ptr<FunctionData> Bind(ClientContext &context, ScalarFunction &function,
	                                     vector<unique_ptr<Expression>> &arguments) {
		ImageOperatorContract::BindImage(function, arguments);
		vector<Value> values;
		for (idx_t i = 1; i < 5; i++) {
			if (arguments[i]->return_type.id() == LogicalTypeId::UNKNOWN) {
				throw ParameterNotResolvedException();
			}
			if (!arguments[i]->IsFoldable()) {
				throw BinderException("image_hash options must be constant so its binary width is known");
			}
			auto value = ExpressionExecutor::EvaluateScalar(context, *arguments[i]);
			if (value.IsNull()) {
				throw InvalidInputException("image_hash options cannot be NULL");
			}
			if (i > 1 && !value.type().IsIntegral()) {
				throw InvalidInputException("image_hash sizes must be integers");
			}
			values.push_back(std::move(value));
		}
		auto options = make_uniq<ImageHashOptions>();
		options->method = values[0].GetValue<string>();
		bool known = false;
		for (auto method :
		     {"phash", "phash_simple", "dhash", "dhash_vertical", "ahash", "whash", "crop_resistant", "colorhash"}) {
			known |= options->method == method;
		}
		if (!known) {
			throw InvalidInputException("Unsupported image_hash method '%s'", options->method);
		}
		auto size = values[1].GetValue<int64_t>(), bits = values[2].GetValue<int64_t>(),
		     segments = values[3].GetValue<int64_t>();
		if (size < 2 || size > 64 || bits < 1 || bits > 8 || segments < 1 || segments > 16) {
			throw InvalidInputException(
			    "image_hash requires hash_size in [2,64], binbits in [1,8], segments in [1,16]");
		}
		if (options->method == "whash" && (size & (size - 1))) {
			throw InvalidInputException("whash hash_size must be a power of two");
		}
		options->hash_size = idx_t(size);
		options->binbits = idx_t(bits);
		options->segments = idx_t(segments);
		function.return_type = FixedBinaryType::Create(options->Bytes());
		return std::move(options);
	}
};

} // namespace duckdb
