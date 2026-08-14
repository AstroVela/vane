#include "duckdb/planner/operator/logical_update.hpp"

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/binder.hpp"

namespace duckdb {

LogicalUpdate::LogicalUpdate(TableCatalogEntry &table)
    : LogicalOperator(LogicalOperatorType::LOGICAL_UPDATE), write_target(table), table(table), table_index(0),
      return_chunk(false) {
}

LogicalUpdate::LogicalUpdate(ClientContext &context, const LogicalWriteTarget &write_target_p)
    : LogicalOperator(LogicalOperatorType::LOGICAL_UPDATE), write_target(write_target_p),
      table(write_target_p.Resolve(context)) {
	auto binder = Binder::CreateBinder(context);
	bound_constraints = binder->BindConstraints(table);
}

idx_t LogicalUpdate::EstimateCardinality(ClientContext &context) {
	return return_chunk ? LogicalOperator::EstimateCardinality(context) : 1;
}

vector<ColumnBinding> LogicalUpdate::GetColumnBindings() {
	if (return_chunk) {
		return GenerateColumnBindings(table_index, table.GetTypes().size());
	}
	return {ColumnBinding(0, 0)};
}

void LogicalUpdate::ResolveTypes() {
	if (return_chunk) {
		types = table.GetTypes();
	} else {
		types.emplace_back(LogicalType::BIGINT);
	}
}

string LogicalUpdate::GetName() const {
#ifdef DEBUG
	if (DBConfigOptions::debug_print_bindings) {
		return LogicalOperator::GetName() + StringUtil::Format(" #%llu", table_index);
	}
#endif
	return LogicalOperator::GetName();
}

} // namespace duckdb
