// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/bound_plan.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_macro_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/function/lambda_functions.hpp"
#include "duckdb/main/relation/query_relation.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/parsed_expression_iterator.hpp"
#include "duckdb/parser/statement/explain_statement.hpp"
#include "duckdb/planner/constraints/bound_check_constraint.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_binder.hpp"
#include "duckdb/planner/client_context_query.hpp"
#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/operator/logical_copy_to_file.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_data_sink.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_explain.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_merge_into.hpp"
#include "duckdb/planner/operator/logical_update.hpp"

#include <filesystem>

namespace duckdb {

void ValidateRunnerStatement(SQLStatement &statement) {
	if (statement.type == StatementType::CALL_STATEMENT) {
		throw NotImplementedException("Runner execution does not support SQL CALL; use a local-fast connection");
	}
	if (statement.type == StatementType::PREPARE_STATEMENT || statement.type == StatementType::EXECUTE_STATEMENT ||
	    (statement.type == StatementType::EXPLAIN_STATEMENT &&
	     statement.Cast<ExplainStatement>().explain_type == ExplainType::EXPLAIN_ANALYZE)) {
		throw NotImplementedException("Runner execution does not support SQL PREPARE, EXECUTE, or EXPLAIN ANALYZE; "
		                              "use direct SQL with bound parameters or a local-fast connection");
	}
	if (statement.type == StatementType::EXPLAIN_STATEMENT) {
		ValidateRunnerStatement(*statement.Cast<ExplainStatement>().stmt);
	}
}

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
	case LogicalOperatorType::LOGICAL_VACUUM:
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

class ValidateRunnerExpressionEffects : public LogicalOperatorVisitor {
public:
	void VisitOperator(LogicalOperator &op) override {
		if (op.type == LogicalOperatorType::LOGICAL_GET) {
			auto &get = op.Cast<LogicalGet>();
			if (auto table = get.GetTable()) {
				ValidateTable(*table);
			}
			auto &function = get.function;
			if (function.RequiresClientContext()) {
				throw NotImplementedException("Runner execution does not support client-context table function %s; "
				                              "use a local-fast connection",
				                              function.name);
			}
		}
		// This is validation only; avoid the rewriting visitor's projection-map
		// repair and cover defaults and constraints outside the usual expression lists.
		for (auto &child : op.children) {
			VisitOperator(*child);
		}
		VisitOperatorExpressions(op);
		if (op.type == LogicalOperatorType::LOGICAL_INSERT) {
			auto &insert = op.Cast<LogicalInsert>();
			VisitWriteExpressions(insert.table, insert.bound_defaults, insert.bound_constraints);
			for (auto &row : insert.insert_values) {
				VisitDefaults(row);
			}
		} else if (op.type == LogicalOperatorType::LOGICAL_UPDATE) {
			auto &update = op.Cast<LogicalUpdate>();
			VisitWriteExpressions(update.table, update.bound_defaults, update.bound_constraints);
		} else if (op.type == LogicalOperatorType::LOGICAL_MERGE_INTO) {
			auto &merge = op.Cast<LogicalMergeInto>();
			VisitWriteExpressions(merge.table, merge.bound_defaults, merge.bound_constraints);
		} else if (op.type == LogicalOperatorType::LOGICAL_DELETE) {
			ValidateTable(op.Cast<LogicalDelete>().table);
		}
	}

private:
	static void ValidateTable(const TableCatalogEntry &table) {
		if (table.temporary || table.catalog.IsTemporaryCatalog()) {
			throw NotImplementedException("Runner plans cannot read or write temporary table %s", table.name);
		}
	}

	void VisitWriteExpressions(TableCatalogEntry &table, vector<unique_ptr<Expression>> &defaults,
	                           vector<unique_ptr<BoundConstraint>> &constraints) {
		ValidateTable(table);
		// Generated columns are rebound and evaluated by append verification,
		// outside the bound write plan. They need an explicit effect contract.
		if (table.HasGeneratedColumns()) {
			throw NotImplementedException("Runner writes do not support generated target columns");
		}
		VisitDefaults(defaults);
		VisitConstraints(constraints);
	}

	void VisitConstraints(vector<unique_ptr<BoundConstraint>> &constraints) {
		for (auto &constraint : constraints) {
			if (constraint->type == ConstraintType::CHECK) {
				VisitExpression(&constraint->Cast<BoundCheckConstraint>().expression);
			}
		}
	}

	void VisitDefaults(vector<unique_ptr<Expression>> &expressions) {
		for (auto &expression : expressions) {
			VisitExpression(&expression);
		}
	}

	unique_ptr<Expression> VisitReplace(BoundFunctionExpression &expression, unique_ptr<Expression> *) override {
		expression.function.VerifyRunnerExecution();
		// Bound list functions store their executable lambda body in bind data,
		// outside the ordinary children visited by LogicalOperatorVisitor.
		auto lambda = dynamic_cast<ListLambdaBindData *>(expression.bind_info.get());
		if (lambda && lambda->lambda_expr) {
			VisitExpression(&lambda->lambda_expr);
		}
		return nullptr;
	}
};

class RunnerCTASMetadataBinder : public ExpressionBinder {
public:
	RunnerCTASMetadataBinder(Binder &binder, ClientContext &context) : ExpressionBinder(binder, context) {
	}

	void ValidateAndCapture(unique_ptr<ParsedExpression> &expression, bool allow_transform) {
		RejectUnsupportedExpression(*expression);
		bool transform = false;
		if (expression->GetExpressionClass() == ExpressionClass::FUNCTION) {
			auto &function = expression->Cast<FunctionExpression>();
			EntryLookupInfo scalar_lookup(CatalogType::SCALAR_FUNCTION_ENTRY, function.function_name);
			auto entry =
			    GetCatalogEntry(function.catalog, function.schema, scalar_lookup, OnEntryNotFound::RETURN_NULL);
			if (entry && entry->type == CatalogType::MACRO_ENTRY) {
				// Validate the entire expansion first, including recursion limits.
				// Keep its expansion in the payload: the driver has no client macros.
				auto copy = expression->Copy();
				auto bound = Bind(copy);
				ValidateRunnerExpressionEffects().VisitExpression(&bound);
				auto alias = expression->GetAlias();
				auto query_location = expression->GetQueryLocation();
				UnfoldMacroExpression(function, entry->Cast<ScalarMacroCatalogEntry>(), expression, 0);
				expression->SetAlias(alias);
				expression->SetQueryLocation(query_location);
				ValidateAndCapture(expression, false);
				return;
			}
			// Extension partition/sort transforms (for example bucket) are a
			// catalog-owned declaration, not necessarily registered SQL functions.
			// Only an unqualified outer transform may use that syntax. Every
			// argument still goes through ordinary SQL binding and effect checks.
			if (allow_transform && !entry && function.catalog.empty() && function.schema.empty() &&
			    !function.children.empty() && !function.distinct && !function.filter &&
			    function.order_bys->orders.empty() && !IsUnnestFunction(function.function_name)) {
				EntryLookupInfo table_lookup(CatalogType::TABLE_FUNCTION_ENTRY, function.function_name);
				transform =
				    !GetCatalogEntry(INVALID_CATALOG, INVALID_SCHEMA, table_lookup, OnEntryNotFound::RETURN_NULL);
			}
		}
		ParsedExpressionIterator::EnumerateChildren(
		    *expression, [&](unique_ptr<ParsedExpression> &child) { ValidateAndCapture(child, false); });
		if (transform) {
			return;
		}
		auto copy = expression->Copy();
		auto bound = Bind(copy);
		ValidateRunnerExpressionEffects().VisitExpression(&bound);
		if (bound->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
			// Binding can capture client values (for example getvariable).
			// Preserve those constants instead of looking them up on the driver.
			auto constant = make_uniq<ConstantExpression>(bound->Cast<BoundConstantExpression>().value);
			constant->SetAlias(expression->GetAlias());
			constant->SetQueryLocation(expression->GetQueryLocation());
			expression = std::move(constant);
		}
	}

protected:
	static void RejectUnsupportedExpression(const ParsedExpression &expression) {
		if (expression.GetExpressionClass() == ExpressionClass::SUBQUERY) {
			throw NotImplementedException("Runner CTAS metadata does not support subqueries");
		}
		if (expression.GetExpressionClass() == ExpressionClass::LAMBDA) {
			throw NotImplementedException("Runner CTAS metadata does not support lambda expressions");
		}
	}

	BindResult BindExpression(unique_ptr<ParsedExpression> &expression, idx_t depth,
	                          bool root_expression = false) override {
		RejectUnsupportedExpression(*expression);
		auto result = ExpressionBinder::BindExpression(expression, depth, root_expression);
		if (!result.HasError()) {
			// Validate children as they bind: a parent's bind callback may fold
			// them away or evaluate a constant argument before the final visit.
			ValidateRunnerExpressionEffects().VisitExpression(&result.expression);
		}
		return result;
	}
};

static void ValidateRunnerCTASMetadata(ClientContext &context, CreateTableInfo &info,
                                       const case_insensitive_map_t<BoundParameterData> &parameters) {
	if (info.options.empty() && info.partition_keys.empty() && info.sort_keys.empty()) {
		return;
	}
	auto binder = Binder::CreateBinder(context);
	// Metadata binds independently of the query, but its callbacks must obey
	// the same runner policy before they can evaluate or change client state.
	binder->SetBindingForRunner(true);
	// These are expression fragments, not SELECT result columns. Preserve
	// untyped NULLs so capturing one child does not change its parent's overload.
	binder->SetCanContainNulls(true);
	RunnerCTASMetadataBinder metadata_binder(*binder, context);
	for (auto &option : info.options) {
		QueryRelation::CaptureParameters(option.second, parameters);
		metadata_binder.ValidateAndCapture(option.second, false);
	}
	// Keys refer to the created table's output columns, not the CTAS query's
	// input bindings (which can have different names or no table at all).
	binder->bind_context.AddGenericBinding(binder->GenerateTableIndex(), info.table, info.columns.GetColumnNames(),
	                                       info.columns.GetColumnTypes());
	for (auto &key : info.partition_keys) {
		QueryRelation::CaptureParameters(key, parameters);
		metadata_binder.ValidateAndCapture(key, true);
	}
	for (auto &key : info.sort_keys) {
		QueryRelation::CaptureParameters(key, parameters);
		metadata_binder.ValidateAndCapture(key, true);
	}
}

static unique_ptr<RunnerBoundPlan>
AdmitRunnerBoundPlanInternal(Planner &planner, unique_ptr<LogicalOperator> &plan, PreparedStatementData &prepared,
                             const case_insensitive_map_t<BoundParameterData> &parameters,
                             RunnerPlanAdmission admission) {
	auto &context = planner.context;
	const bool transport = admission == RunnerPlanAdmission::TRANSPORT;
	const auto runner_type = transport ? "ray" : context.vane_runner_type;
	if (runner_type == "local-fast") {
		return nullptr;
	}
	if (IsConnectionPlan(*plan)) {
		if (transport) {
			throw NotImplementedException("Runner transports cannot include client connection operations");
		}
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
	if (plan->type == LogicalOperatorType::LOGICAL_EXPORT || plan->type == LogicalOperatorType::LOGICAL_COPY_DATABASE) {
		throw NotImplementedException("Runner execution does not support logical operator %s", plan->GetName());
	}
	auto kind = RunnerPlanKind::READ;
	string operation = "SELECT";
	auto write = FindWrite(*plan);
	if (!transport && !write && prepared.properties.modified_databases.empty() &&
	    IsClientContextQuery(*plan,
	                         prepared.properties.captured_client_context || prepared.properties.requires_client_context,
	                         prepared.properties.requires_client_context)) {
		return nullptr;
	}
	if (prepared.properties.requires_client_context) {
		if (write || dynamic_cast<LogicalDataSink *>(plan.get())) {
			throw NotImplementedException("Runner writes cannot include client connection queries or command results");
		}
		if (transport) {
			throw NotImplementedException(
			    "Runner transports cannot include client connection queries or command results");
		}
		throw NotImplementedException("Runner queries cannot combine client connection queries with data scans or "
		                              "unsupported expressions");
	}
	if (context.config.query_verification_enabled) {
		throw NotImplementedException("Native query verification requires a local-fast connection");
	}
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
	if (kind == RunnerPlanKind::READ && !prepared.properties.modified_databases.empty()) {
		throw NotImplementedException("Runner reads do not support database-modifying expressions");
	}
	ValidateRunnerExpressionEffects().VisitOperator(*plan);
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
			ValidateRunnerCTASMetadata(context, info, parameters);
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

unique_ptr<RunnerBoundPlan> AdmitRunnerBoundPlan(Planner &planner, unique_ptr<LogicalOperator> &plan,
                                                 PreparedStatementData &prepared,
                                                 const case_insensitive_map_t<BoundParameterData> &parameters,
                                                 RunnerPlanAdmission admission) {
	try {
		return AdmitRunnerBoundPlanInternal(planner, plan, prepared, parameters, admission);
	} catch (const Exception &exception) {
		ErrorData error(exception);
		if (!planner.context.transaction.IsAutoCommit() &&
		    (error.Type() == ExceptionType::NOT_IMPLEMENTED || error.Type() == ExceptionType::INVALID_INPUT)) {
			// Capability rejection is a binding restriction, not an execution
			// failure. Preserve existing transactional work for the caller.
			throw BinderException(error.RawMessage());
		}
		throw;
	}
}

shared_ptr<DuckDBPyResult> RunnerExecutionResult::TakeResult() {
	if (native_result) {
		return make_shared_ptr<DuckDBPyResult>(std::move(native_result));
	}
	return std::move(distributed_result);
}

} // namespace duckdb
