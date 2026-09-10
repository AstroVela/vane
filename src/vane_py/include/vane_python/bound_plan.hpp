// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "duckdb/main/client_context.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/planner/planner.hpp"
#include "vane_python/pybind11/pybind_wrapper.hpp"
#include "vane_python/pyresult.hpp"

namespace duckdb {

enum class RunnerPlanKind : uint8_t { READ, COPY, TABLE_WRITE, DATA_SINK };
enum class RunnerPlanAdmission : uint8_t { CONNECTION, TRANSPORT };

//! The executor owns the bound tree and its statement semantics together.
//! A runner only receives the serialized transport produced from this object.
struct RunnerBoundPlan {
	shared_ptr<ClientContext> context;
	shared_ptr<Binder> binder;
	unique_ptr<LogicalOperator> plan;
	vector<string> names;
	vector<LogicalType> types;
	StatementProperties properties;
	StatementType statement_type = StatementType::INVALID_STATEMENT;
	idx_t query_number = 0;
	RunnerPlanKind kind = RunnerPlanKind::READ;
	string operation;
};

//! Reject unsupported SQL wrappers before binding can evaluate their arguments.
void ValidateRunnerStatement(SQLStatement &statement);

//! Connection admission may return nullptr for native execution. Explicit
//! transports must pass runner validation independently of the source runner.
unique_ptr<RunnerBoundPlan> AdmitRunnerBoundPlan(Planner &planner, unique_ptr<LogicalOperator> &plan,
                                                 PreparedStatementData &prepared,
                                                 const case_insensitive_map_t<BoundParameterData> &parameters,
                                                 RunnerPlanAdmission admission = RunnerPlanAdmission::CONNECTION);

//! Serialize without binding, choosing a runner, or starting query execution.
py::object SerializeRunnerBoundPlan(RunnerBoundPlan &plan, const py::object &connection_owner);

struct RunnerExecutionResult {
	unique_ptr<QueryResult> native_result;
	shared_ptr<DuckDBPyResult> distributed_result;
	py::object write_outcome = py::none();
	StatementReturnType return_type = StatementReturnType::NOTHING;

	shared_ptr<DuckDBPyResult> TakeResult();
};

//! Both SQL statements and Relation terminals enter here. Exactly one input is set.
RunnerExecutionResult ExecuteWithRunner(const shared_ptr<ClientContext> &context, unique_ptr<SQLStatement> statement,
                                        const shared_ptr<Relation> &relation,
                                        case_insensitive_map_t<BoundParameterData> parameters,
                                        const py::object &connection_owner, const py::object &interrupt_check,
                                        bool stream_result = false, vector<string> *cleanup_warnings = nullptr,
                                        optional_ptr<unique_ptr<PreparedStatement>> native_prepared_cache = nullptr);

} // namespace duckdb
