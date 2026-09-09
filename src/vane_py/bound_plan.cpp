// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/bound_plan.hpp"

#include "duckdb/common/file_system.hpp"
#include "duckdb/main/relation/query_relation.hpp"
#include "duckdb/planner/operator/logical_copy_to_file.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_data_sink.hpp"
#include "duckdb/planner/operator/logical_explain.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"

#include <filesystem>

namespace duckdb {

static bool IsConnectionPlan(LogicalOperator &plan) {
	switch (plan.type) {
	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
		return plan.children.empty();
	case LogicalOperatorType::LOGICAL_EXPLAIN:
		return plan.Cast<LogicalExplain>().explain_type != ExplainType::EXPLAIN_ANALYZE;
	case LogicalOperatorType::LOGICAL_ALTER:
	case LogicalOperatorType::LOGICAL_CREATE_INDEX:
	case LogicalOperatorType::LOGICAL_CREATE_SEQUENCE:
	case LogicalOperatorType::LOGICAL_CREATE_VIEW:
	case LogicalOperatorType::LOGICAL_CREATE_SCHEMA:
	case LogicalOperatorType::LOGICAL_CREATE_MACRO:
	case LogicalOperatorType::LOGICAL_CREATE_TYPE:
	case LogicalOperatorType::LOGICAL_CREATE_SECRET:
	case LogicalOperatorType::LOGICAL_DROP:
	case LogicalOperatorType::LOGICAL_TRANSACTION:
	case LogicalOperatorType::LOGICAL_ATTACH:
	case LogicalOperatorType::LOGICAL_DETACH:
	case LogicalOperatorType::LOGICAL_SET:
	case LogicalOperatorType::LOGICAL_RESET:
	case LogicalOperatorType::LOGICAL_PRAGMA:
	case LogicalOperatorType::LOGICAL_LOAD:
	case LogicalOperatorType::LOGICAL_UPDATE_EXTENSIONS:
		return true;
	default:
		return false;
	}
}

static LogicalOperator *FindWrite(LogicalOperator &plan) {
	switch (plan.type) {
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
	case LogicalOperatorType::LOGICAL_INSERT:
	case LogicalOperatorType::LOGICAL_UPDATE:
	case LogicalOperatorType::LOGICAL_DELETE:
	case LogicalOperatorType::LOGICAL_MERGE_INTO:
	case LogicalOperatorType::LOGICAL_CREATE_TABLE:
		return &plan;
	default:
		break;
	}
	for (auto &child : plan.children) {
		if (auto write = FindWrite(*child)) {
			return write;
		}
	}
	return nullptr;
}

static void ValidateCopyDestination(ClientContext &context, LogicalCopyToFile &copy) {
	if (FileSystem::IsRemoteFile(copy.file_path)) {
		return;
	}
	auto expanded = FileSystem::GetFileSystem(context).ExpandPath(copy.file_path);
	auto normalized = std::filesystem::path(expanded).lexically_normal().generic_string();
	bool non_file = normalized == "/dev/stdout" || normalized == "/dev/stderr" || normalized == "/dev/stdin" ||
	                StringUtil::StartsWith(normalized, "/dev/fd/") ||
	                StringUtil::StartsWith(normalized, "/proc/self/fd/");
	std::error_code error;
	auto status = std::filesystem::status(expanded, error);
	if (!error && std::filesystem::exists(status)) {
		non_file |= !std::filesystem::is_regular_file(status) && !std::filesystem::is_directory(status);
	}
	if (non_file) {
		throw NotImplementedException("Runner COPY TO requires a file dataset destination; STDOUT, devices and pipes "
		                              "are not supported");
	}
}

unique_ptr<RunnerBoundPlan> AdmitRunnerBoundPlan(Planner &planner, unique_ptr<LogicalOperator> &plan,
                                                 PreparedStatementData &prepared,
                                                 const case_insensitive_map_t<BoundParameterData> &parameters) {
	auto &context = planner.context;
	const auto &runner_type = context.vane_runner_type;
	if (runner_type == "local-fast" || IsConnectionPlan(*plan)) {
		return nullptr;
	}
	if (prepared.statement_type == StatementType::CALL_STATEMENT) {
		// CALL is lowered to a SELECT-like GET, but its function can mutate a
		// catalog (for example dbgen/checkpoint). No distributed effect contract
		// exists for these calls, so never admit them as ordinary reads.
		throw NotImplementedException("Runner execution does not support SQL CALL; use a local-fast connection");
	}
	if (plan->type == LogicalOperatorType::LOGICAL_PREPARE || plan->type == LogicalOperatorType::LOGICAL_EXECUTE ||
	    plan->type == LogicalOperatorType::LOGICAL_EXPLAIN) {
		throw NotImplementedException("Runner execution does not support SQL PREPARE, EXECUTE, or EXPLAIN ANALYZE; "
		                              "use direct SQL with bound parameters or a local-fast connection");
	}
	if (plan->type == LogicalOperatorType::LOGICAL_EXPORT || plan->type == LogicalOperatorType::LOGICAL_COPY_DATABASE ||
	    plan->type == LogicalOperatorType::LOGICAL_VACUUM) {
		throw NotImplementedException("Runner execution does not support logical operator %s", plan->GetName());
	}
	auto kind = RunnerPlanKind::READ;
	string operation = "SELECT";
	auto write = FindWrite(*plan);
	if (prepared.statement_type == StatementType::COPY_STATEMENT && write &&
	    write->type == LogicalOperatorType::LOGICAL_INSERT) {
		throw NotImplementedException("Runner execution does not support SQL COPY FROM");
	}
	if (dynamic_cast<LogicalDataSink *>(plan.get())) {
		kind = RunnerPlanKind::DATA_SINK;
		operation = "DataSink";
	} else if (write) {
		kind = write->type == LogicalOperatorType::LOGICAL_COPY_TO_FILE ? RunnerPlanKind::COPY
		                                                                : RunnerPlanKind::TABLE_WRITE;
		operation = kind == RunnerPlanKind::COPY                               ? "COPY TO"
		            : write->type == LogicalOperatorType::LOGICAL_CREATE_TABLE ? "CTAS"
		                                                                       : write->GetName();
		if (kind == RunnerPlanKind::TABLE_WRITE && runner_type != "ray") {
			throw InvalidInputException("%s requires a ray or local-fast connection", operation);
		}
	} else if (runner_type == "local") {
		// The local FTE backend only supports terminals; reads use native DuckDB.
		return nullptr;
	}
	if (!context.transaction.IsAutoCommit()) {
		// This is a binding restriction, so rejecting it must not abort the
		// caller's transaction before any runner has been initialized.
		throw BinderException("Runner %s requires DuckDB auto-commit mode and cannot participate "
		                      "in an explicit transaction",
		                      operation);
	}
	if (write) {
		// DuckDB marks CTAS as NOTHING even though its result is a Count row.
		bool count_result = prepared.properties.return_type == StatementReturnType::CHANGED_ROWS ||
		                    (write->type == LogicalOperatorType::LOGICAL_CREATE_TABLE &&
		                     prepared.properties.return_type == StatementReturnType::NOTHING);
		if (!count_result || prepared.types != vector<LogicalType> {LogicalType::BIGINT} ||
		    prepared.names != vector<string> {"Count"}) {
			throw NotImplementedException("Runner writes only support the default Count result; RETURNING, "
			                              "RETURN_FILES and RETURN_STATS are not supported");
		}
		if (write->type == LogicalOperatorType::LOGICAL_COPY_TO_FILE) {
			ValidateCopyDestination(context, write->Cast<LogicalCopyToFile>());
		}
		// The SQL binder rewrites INSERT conflict handling into MERGE INTO.
		if ((prepared.statement_type == StatementType::INSERT_STATEMENT &&
		     write->type == LogicalOperatorType::LOGICAL_MERGE_INTO) ||
		    (write->type == LogicalOperatorType::LOGICAL_INSERT &&
		     write->Cast<LogicalInsert>().on_conflict_info.action_type != OnConflictAction::THROW)) {
			throw NotImplementedException("Runner INSERT does not support ON CONFLICT");
		}
		if (write->type == LogicalOperatorType::LOGICAL_CREATE_TABLE) {
			auto &info = write->Cast<LogicalCreateTable>().info->Base();
			if (info.temporary || info.on_conflict != OnCreateConflict::ERROR_ON_CONFLICT) {
				throw NotImplementedException("Runner CTAS does not support TEMPORARY, OR REPLACE or IF NOT EXISTS");
			}
			for (auto &option : info.options) {
				QueryRelation::CaptureParameters(option.second, parameters);
			}
			for (auto &key : info.partition_keys) {
				QueryRelation::CaptureParameters(key, parameters);
			}
			for (auto &key : info.sort_keys) {
				QueryRelation::CaptureParameters(key, parameters);
			}
		}
	}
	// Resolve remaining bound parameter expressions before transport. Native
	// execution performs the same assignment when admitting its prepared statement.
	prepared.Bind(parameters);
	auto result = make_uniq<RunnerBoundPlan>();
	result->context = context.shared_from_this();
	result->binder = planner.binder;
	result->names = prepared.names;
	result->types = prepared.types;
	result->properties = prepared.properties;
	result->statement_type = prepared.statement_type;
	result->query_number = context.transaction.GetActiveQuery();
	result->kind = kind;
	result->operation = std::move(operation);
	result->plan = std::move(plan);
	return result;
}

shared_ptr<DuckDBPyResult> RunnerExecutionResult::TakeResult() {
	if (native_result) {
		return make_shared_ptr<DuckDBPyResult>(std::move(native_result));
	}
	return std::move(distributed_result);
}

} // namespace duckdb
