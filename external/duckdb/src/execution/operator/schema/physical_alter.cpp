#include "duckdb/execution/operator/schema/physical_alter.hpp"
#include "duckdb/parser/parsed_data/alter_table_info.hpp"
#include "duckdb/parser/parsed_data/alter_database_info.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"

namespace duckdb {

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
SourceResultType PhysicalAlter::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                OperatorSourceInput &input) const {
	if (info->type == AlterType::ALTER_DATABASE) {
		auto &db_info = info->Cast<AlterDatabaseInfo>();
		auto &db_manager = DatabaseManager::Get(context.client);
		db_manager.Alter(context.client, db_info);
	} else {
		auto &catalog = Catalog::GetCatalog(context.client, info->catalog);
		if (catalog.IsDuckCatalog() && info->type == AlterType::ALTER_TABLE) {
			auto &table_info = info->Cast<AlterTableInfo>();
			if (table_info.alter_table_type == AlterTableType::REMOVE_COLUMN) {
				table_info.Cast<RemoveColumnInfo>().new_logical_write_target_identity =
				    CreateTableInfo::GenerateLogicalWriteTargetIdentity();
			}
		}
		catalog.Alter(context.client, *info);
	}
	return SourceResultType::FINISHED;
}

} // namespace duckdb
