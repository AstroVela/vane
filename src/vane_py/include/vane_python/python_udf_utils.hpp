// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
//                         DuckDB
//
// vane_python/python_udf_utils.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/value.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "vane_python/pybind11/pybind_wrapper.hpp"
#include "vane_python/python_objects.hpp"
#include "vane_python/pytype.hpp"

namespace duckdb {

unique_ptr<Expression> LowerRegisteredExpressionUDF(FunctionBindExpressionInput &input);
unique_ptr<Expression> LowerRegisteredExpressionUDFPreservingFoldableNulls(FunctionBindExpressionInput &input);
void ValidateSynchronousUDFCallable(const py::object &udf);

Value BuildPythonUDFPayload(
    const string &name, const py::function &udf, const py::object &schema, const shared_ptr<DuckDBPyType> &return_type,
    const string &execution_backend, idx_t default_parallelism, const Optional<py::object> &cpus,
    const Optional<py::object> &gpus, const Optional<py::object> &memory_bytes, const Optional<py::object> &batch_size,
    const Optional<py::object> &output_batch_size, const Optional<py::object> &min_task_batch_size,
    const Optional<py::object> &preserve_compute_batch_boundaries, const Optional<py::object> &actor_number,
    const Optional<py::object> &target_max_batch_bytes, const Optional<py::object> &task_input_max_bytes,
    const Optional<py::object> &output_target_max_bytes, bool flat_map = false);

Value BuildScalarUDFPayload(const string &name, const py::function &udf, const shared_ptr<DuckDBPyType> &return_type,
                            const string &execution_backend, idx_t default_parallelism,
                            const vector<LogicalType> &passthrough_types, const Optional<py::object> &cpus,
                            const Optional<py::object> &gpus, const Optional<py::object> &batch_size,
                            const Optional<py::object> &actor_number);

Value BuildExpressionScalarUDFPayload(const string &name, const py::function &udf,
                                      const shared_ptr<DuckDBPyType> &return_type, const string &execution_backend,
                                      idx_t default_parallelism, idx_t scalar_arg_count,
                                      const Optional<py::object> &expression_id);

Value BuildExpressionMapBatchesUDFPayload(const string &name, const py::function &udf, const py::object &schema,
                                          const string &execution_backend, idx_t default_parallelism,
                                          const vector<string> &input_names, const Optional<py::object> &batch_size,
                                          bool row_preserving, const Optional<py::object> &gpus,
                                          const Optional<py::object> &actor_number,
                                          const Optional<py::object> &expression_id);

Value AddAISQLPayloadMetadata(const Value &payload, const string &provider, const string &model,
                              const string &return_type, const Optional<py::object> &dimensions);

} // namespace duckdb
