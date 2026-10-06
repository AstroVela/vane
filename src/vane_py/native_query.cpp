// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_python/native_query.hpp"
#include "duckdb/main/relation/query_relation.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/showref.hpp"

namespace duckdb {

vector<unique_ptr<SQLStatement>> ExtractVaneStatements(ClientContext &context, const string &query) {
	// The Python connection caller serializes this with its execution mutex.
	try {
		Parser parser(context.GetParserOptions());
		parser.ParseQuery(query);
		// Preserve direct commands until execution. Native preprocessing runs
		// there with the transaction state established by preceding statements.
		return std::move(parser.statements);
	} catch (std::exception &exception) {
		ErrorData error(exception);
		context.ProcessError(error, query);
		error.Throw();
	}
}

vector<unique_ptr<SQLStatement>> PreprocessVaneStatement(ClientContext &context, unique_ptr<SQLStatement> statement) {
	auto query = statement->query;
	vector<unique_ptr<SQLStatement>> statements;
	statements.push_back(std::move(statement));
	try {
		context.PreprocessStatements(statements);
		return statements;
	} catch (std::exception &exception) {
		ErrorData error(exception);
		context.ProcessError(error, query);
		error.Throw();
	}
}

static bool IsClientCommandNode(QueryNode &node) {
	if (node.type != QueryNodeType::SELECT_NODE) {
		return false;
	}
	auto &from = node.Cast<SelectNode>().from_table;
	return from && from->type == TableReferenceType::SHOW_REF && from->Cast<ShowRef>().show_type != ShowType::SUMMARY;
}

bool IsDirectClientCommand(SQLStatement &statement) {
	return statement.type == StatementType::PRAGMA_STATEMENT ||
	       (statement.type == StatementType::SELECT_STATEMENT &&
	        IsClientCommandNode(*statement.Cast<SelectStatement>().node));
}

// Keep command dispatch in Vane. Native QueryRelation may wrap its AST with
// replacement-scan CTEs, so inspect the command before its constructor binds it.
class ClientCommandRelation : public QueryRelation {
public:
	using QueryRelation::QueryRelation;
};

shared_ptr<Relation> CreateVaneQueryRelation(const shared_ptr<ClientContext> &context,
                                             unique_ptr<SelectStatement> statement, const string &alias,
                                             const string &query) {
	if (IsDirectClientCommand(*statement)) {
		return make_shared_ptr<ClientCommandRelation>(context, std::move(statement), alias, query);
	}
	return make_shared_ptr<QueryRelation>(context, std::move(statement), alias, query);
}

bool IsDirectClientCommand(Relation &relation) {
	return relation.type == RelationType::MATERIALIZED_RELATION || dynamic_cast<ClientCommandRelation *>(&relation);
}

shared_ptr<DuckDBPyResult> NativeExecutionResult::TakeResult() {
	if (native_result) {
		return make_shared_ptr<DuckDBPyResult>(std::move(native_result));
	}
	return nullptr;
}

} // namespace duckdb
