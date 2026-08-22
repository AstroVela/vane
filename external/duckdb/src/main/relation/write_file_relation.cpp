// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/main/relation/write_file_relation.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"
#include "duckdb/parser/statement/copy_statement.hpp"
#include "duckdb/planner/binder.hpp"

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

BoundStatement WriteFileRelation::Bind(Binder &binder) {
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
	string str = RenderWhitespace(depth) + "Write To " + StringUtil::Upper(format) + " [" + file_path + "]\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
