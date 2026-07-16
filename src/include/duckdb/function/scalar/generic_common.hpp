// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/scalar/generic_common.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/built_in_functions.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"

namespace duckdb {
class BoundFunctionExpression;

struct ConstantOrNull {
	static unique_ptr<FunctionData> Bind(Value value);
	static bool IsConstantOrNull(BoundFunctionExpression &expr, const Value &val);
};

struct ExportAggregateFunctionBindData : public FunctionData {
	unique_ptr<BoundAggregateExpression> aggregate;
	explicit ExportAggregateFunctionBindData(unique_ptr<Expression> aggregate_p);
	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other_p) const override;
};

DUCKDB_API void ExportStateAggregateSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                                              const AggregateFunction &function);
DUCKDB_API unique_ptr<FunctionData> ExportStateAggregateDeserialize(Deserializer &deserializer,
                                                                    AggregateFunction &function);
DUCKDB_API void ExportStateScalarSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                                           const ScalarFunction &function);
DUCKDB_API unique_ptr<FunctionData> ExportStateScalarDeserialize(Deserializer &deserializer, ScalarFunction &function);

namespace distributed {

DUCKDB_API void MergeAggregateSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                                        const AggregateFunction &function);
DUCKDB_API unique_ptr<FunctionData> MergeAggregateDeserialize(Deserializer &deserializer, AggregateFunction &function);

} // namespace distributed

struct ExportAggregateFunction {
	static unique_ptr<BoundAggregateExpression> Bind(unique_ptr<BoundAggregateExpression> child_aggregate);
};

} // namespace duckdb
