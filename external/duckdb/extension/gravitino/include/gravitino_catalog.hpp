// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "gravitino_client.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/common/mutex.hpp"

namespace duckdb {
class GravitinoCatalog : public Catalog {
public:
	GravitinoCatalog(AttachedDatabase &db, GravitinoConfig config);
	~GravitinoCatalog() override;
	GravitinoClient client;
	void Initialize(bool load_builtin) override;
	string GetCatalogType() override;
	bool RequiresAttachmentForPlanDeserialization() const override {
		return false;
	}
	string GetDBPath() override;
	bool InMemory() override;
	DatabaseSize GetDatabaseSize(ClientContext &context) override;
	PhysicalOperator &PlanCreateTableAs(ClientContext &, PhysicalPlanGenerator &, LogicalCreateTable &,
	                                    PhysicalOperator &) override;
	PhysicalOperator &PlanInsert(ClientContext &, PhysicalPlanGenerator &, LogicalInsert &,
	                             optional_ptr<PhysicalOperator>) override;
	PhysicalOperator &PlanDelete(ClientContext &, PhysicalPlanGenerator &, LogicalDelete &,
	                             PhysicalOperator &) override;
	PhysicalOperator &PlanUpdate(ClientContext &, PhysicalPlanGenerator &, LogicalUpdate &,
	                             PhysicalOperator &) override;
	optional_ptr<CatalogEntry> CreateSchema(CatalogTransaction transaction, CreateSchemaInfo &info) override;
	optional_ptr<SchemaCatalogEntry> LookupSchema(CatalogTransaction transaction, const EntryLookupInfo &lookup,
	                                              OnEntryNotFound if_not_found) override;
	void ScanSchemas(ClientContext &context, std::function<void(SchemaCatalogEntry &)> callback) override;
	void DropSchema(ClientContext &context, DropInfo &info) override;
	void RequireMutation(ClientContext &context) const;
	static GravitinoCatalog &Get(ClientContext &context, const string &alias);

private:
	mutex lock;
	unordered_map<string, unique_ptr<SchemaCatalogEntry>> schemas;
	SchemaCatalogEntry &Schema(const string &name);
};
} // namespace duckdb
