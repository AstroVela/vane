// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/main/relation/write_file_relation.hpp"
#include "duckdb/main/relation/query_relation.hpp"
#include "duckdb/parser/expression/columnref_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/statement/select_statement.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"
#include "duckdb/parser/statement/copy_statement.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/bound_parameter_map.hpp"

namespace duckdb {

WriteFileRelation::WriteFileRelation(shared_ptr<Relation> child_p, string file_path_p, string format_p,
                                     case_insensitive_map_t<vector<Value>> options_p)
    : Relation(child_p->context, RelationType::WRITE_FILE_RELATION), child(std::move(child_p)),
      file_path(std::move(file_path_p)), format(std::move(format_p)), options(std::move(options_p)) {
	if (format.empty()) {
		throw InvalidInputException("WriteFileRelation requires a non-empty COPY format");
	}
	TryBindRelation(columns);
}

WriteFileRelation::WriteFileRelation(const shared_ptr<ClientContext> &context, unique_ptr<CopyStatement> statement_p,
                                     case_insensitive_map_t<BoundParameterData> parameters_p)
    : Relation(context, RelationType::WRITE_FILE_RELATION), statement(std::move(statement_p)),
      parameters(std::move(parameters_p)) {
	if (!statement || statement->info->is_from) {
		throw InvalidInputException("WriteFileRelation requires a COPY TO statement");
	}
	auto &info = *statement->info;
	auto select = make_uniq<SelectStatement>();
	if (info.select_statement) {
		select->node = std::move(info.select_statement);
	} else {
		auto table = make_uniq<BaseTableRef>();
		table->catalog_name = info.catalog;
		table->schema_name = info.schema;
		table->table_name = info.table;
		auto node = make_uniq<SelectNode>();
		node->from_table = std::move(table);
		for (auto &column : info.select_list) {
			node->select_list.push_back(make_uniq<ColumnRefExpression>(column));
		}
		if (node->select_list.empty()) {
			node->select_list.push_back(make_uniq<StarExpression>());
		}
		select->node = std::move(node);
	}
	// Keep replacement scans and their Python dependencies alive across the
	// initial bind and the runner's later logical-plan extraction.
	child = make_shared_ptr<QueryRelation>(context, std::move(select), "copy_source", "", parameters);
	info.select_relation = child;
	TryBindRelation(columns);
}

WriteFileRelation::~WriteFileRelation() = default;

BoundStatement WriteFileRelation::Bind(Binder &binder) {
	if (statement) {
		BoundParameterMap parameter_map(parameters);
		struct ParameterScope {
			Binder &binder;
			optional_ptr<BoundParameterMap> previous;
			~ParameterScope() {
				binder.SetParameters(previous);
			}
		} scope {binder, binder.GetParameters()};
		binder.SetParameters(parameter_map);
		auto copy = statement->Copy();
		return binder.Bind(*copy);
	}
	CopyStatement copy;
	auto info = make_uniq<CopyInfo>();
	info->select_relation = child;
	info->is_from = false;
	info->file_path = file_path;
	info->format = format;
	info->is_format_auto_detected = false;
	info->options = options;
	copy.info = std::move(info);
	return binder.Bind(copy.Cast<SQLStatement>());
}

unique_ptr<QueryNode> WriteFileRelation::GetQueryNode() {
	throw InternalException("Cannot create a query node from a write file relation");
}

string WriteFileRelation::GetQuery() {
	return string();
}

const vector<ColumnDefinition> &WriteFileRelation::Columns() {
	return columns;
}

string WriteFileRelation::ToString(idx_t depth) {
	if (statement) {
		return RenderWhitespace(depth) + "Write From SQL\n";
	}
	string str = RenderWhitespace(depth) + "Write To " + StringUtil::Upper(format) + " [" + file_path + "]\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
