// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/main/relation/data_sink_relation.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/operator/logical_data_sink.hpp"

namespace duckdb {

DataSinkRelation::DataSinkRelation(shared_ptr<Relation> child_p, string operation_id_p)
    : Relation(child_p->context, RelationType::EXTENSION_RELATION), child(std::move(child_p)),
      operation_id(std::move(operation_id_p)) {
	D_ASSERT(child.get() != this);
	if (operation_id.empty() || operation_id.size() > 256) {
		throw InvalidInputException("DataSink operation identity must contain 1 to 256 UTF-8 bytes");
	}
	TryBindRelation(columns);
}

unique_ptr<QueryNode> DataSinkRelation::GetQueryNode() {
	throw NotImplementedException(
	    "A DataSink relation has no SQL query-node representation; converting it would discard the terminal write");
}

string DataSinkRelation::GetQuery() {
	return string();
}

string DataSinkRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &DataSinkRelation::Columns() {
	return columns;
}

BoundStatement DataSinkRelation::Bind(Binder &binder) {
	// This relation is a terminal marker. Keep it as the logical root instead
	// of introducing the SELECT * projection used by ordinary passthrough
	// relations; the distributed runner requires the corresponding physical
	// DataSink to be the unique terminal root.
	return BindAsInput(binder);
}

BoundStatement DataSinkRelation::BindAsInput(Binder &binder) {
	auto child_ref = BindRelationInput(binder, *child);
	auto child_bound = binder.Bind(*child_ref);
	auto sink = make_uniq<LogicalDataSink>(operation_id);
	sink->children.push_back(std::move(child_bound.plan));
	child_bound.plan = std::move(sink);
	return child_bound;
}

string DataSinkRelation::ToString(idx_t depth) {
	return RenderWhitespace(depth) + "DataSink [" + operation_id + "]\n" + child->ToString(depth + 1);
}

} // namespace duckdb
