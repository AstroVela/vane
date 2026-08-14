#include "duckdb/planner/operator/logical_insert.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/binder.hpp"

namespace duckdb {

BoundOnConflictInfo::BoundOnConflictInfo() : action_type(OnConflictAction::THROW), update_is_del_and_insert(false) {
}

LogicalInsert::LogicalInsert(ClientContext &context, TableCatalogEntry &table, idx_t table_index)
    : LogicalOperator(LogicalOperatorType::LOGICAL_INSERT), write_target(context, table), table(table),
      table_index(table_index), return_chunk(false) {
}

LogicalInsert::LogicalInsert(ClientContext &context, const LogicalWriteTarget &write_target_p)
    : LogicalOperator(LogicalOperatorType::LOGICAL_INSERT), write_target(write_target_p),
      table(write_target_p.Resolve(context)) {
	auto binder = Binder::CreateBinder(context);
	bound_constraints = binder->BindConstraints(table);
}

idx_t LogicalInsert::EstimateCardinality(ClientContext &context) {
	return return_chunk ? LogicalOperator::EstimateCardinality(context) : 1;
}

vector<idx_t> LogicalInsert::GetTableIndex() const {
	return vector<idx_t> {table_index};
}

vector<ColumnBinding> LogicalInsert::GetColumnBindings() {
	if (return_chunk) {
		return GenerateColumnBindings(table_index, table.GetTypes().size());
	}
	return {ColumnBinding(0, 0)};
}

void LogicalInsert::ResolveTypes() {
	if (return_chunk) {
		types = table.GetTypes();
	} else {
		types.emplace_back(LogicalType::BIGINT);
	}
}

string LogicalInsert::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() + StringUtil::Format(" #%llu", table_index);
	}
#endif
	return LogicalOperator::GetName();
}

} // namespace duckdb
