// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "gravitino_catalog.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/entry_lookup_info.hpp"
#include "duckdb/common/enums/on_create_conflict.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/attach_info.hpp"
#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "duckdb/storage/database_size.hpp"
#include "duckdb/storage/storage_extension.hpp"
#include "duckdb/transaction/transaction.hpp"
#include "duckdb/transaction/transaction_manager.hpp"

namespace duckdb {
using namespace duckdb_yyjson; // NOLINT

class GravitinoSchema : public SchemaCatalogEntry {
public:
	GravitinoSchema(Catalog &catalog, CreateSchemaInfo &info) : SchemaCatalogEntry(catalog, info) {
	}
	void Scan(ClientContext &, CatalogType, const std::function<void(CatalogEntry &)> &) override {
	}
	void Scan(CatalogType, const std::function<void(CatalogEntry &)> &) override {
	}
	optional_ptr<CatalogEntry> LookupEntry(CatalogTransaction, const EntryLookupInfo &) override {
		return nullptr;
	}
	void DropEntry(ClientContext &, DropInfo &) override {
		Unsupported();
	}
	void Alter(CatalogTransaction, AlterInfo &) override {
		Unsupported();
	}
	optional_ptr<CatalogEntry> CreateIndex(CatalogTransaction, CreateIndexInfo &, TableCatalogEntry &) override {
		Unsupported();
	}
	optional_ptr<CatalogEntry> CreateTable(CatalogTransaction, BoundCreateTableInfo &) override {
		Unsupported();
	}
	optional_ptr<CatalogEntry> CreateView(CatalogTransaction, CreateViewInfo &) override {
		Unsupported();
	}
	optional_ptr<CatalogEntry> CreateFunction(CatalogTransaction, CreateFunctionInfo &) override {
		Unsupported();
	}
	optional_ptr<CatalogEntry> CreateSequence(CatalogTransaction, CreateSequenceInfo &) override {
		Unsupported();
	}
	optional_ptr<CatalogEntry> CreateTableFunction(CatalogTransaction, CreateTableFunctionInfo &) override {
		Unsupported();
	}
	optional_ptr<CatalogEntry> CreateCopyFunction(CatalogTransaction, CreateCopyFunctionInfo &) override {
		Unsupported();
	}
	optional_ptr<CatalogEntry> CreatePragmaFunction(CatalogTransaction, CreatePragmaFunctionInfo &) override {
		Unsupported();
	}
	optional_ptr<CatalogEntry> CreateCollation(CatalogTransaction, CreateCollationInfo &) override {
		Unsupported();
	}
	optional_ptr<CatalogEntry> CreateType(CatalogTransaction, CreateTypeInfo &) override {
		Unsupported();
	}

private:
	[[noreturn]] static void Unsupported() {
		throw NotImplementedException("Gravitino FILESET catalogs support schema and Fileset metadata; "
		                              "relational table operations are unsupported");
	}
};

GravitinoCatalog::GravitinoCatalog(AttachedDatabase &db, GravitinoConfig config)
    : Catalog(db), client(std::move(config)) {
}
GravitinoCatalog::~GravitinoCatalog() = default;
void GravitinoCatalog::Initialize(bool) {
}
string GravitinoCatalog::GetCatalogType() {
	return "gravitino";
}
string GravitinoCatalog::GetDBPath() {
	return client.config.catalog;
}
bool GravitinoCatalog::InMemory() {
	return false;
}
DatabaseSize GravitinoCatalog::GetDatabaseSize(ClientContext &) {
	return {};
}
PhysicalOperator &GravitinoCatalog::PlanCreateTableAs(ClientContext &, PhysicalPlanGenerator &, LogicalCreateTable &,
                                                      PhysicalOperator &) {
	throw NotImplementedException("Gravitino FILESET catalogs do not support relational table writes");
}
PhysicalOperator &GravitinoCatalog::PlanInsert(ClientContext &, PhysicalPlanGenerator &, LogicalInsert &,
                                               optional_ptr<PhysicalOperator>) {
	throw NotImplementedException("Gravitino FILESET catalogs do not support relational table writes");
}
PhysicalOperator &GravitinoCatalog::PlanDelete(ClientContext &, PhysicalPlanGenerator &, LogicalDelete &,
                                               PhysicalOperator &) {
	throw NotImplementedException("Gravitino FILESET catalogs do not support relational table writes");
}
PhysicalOperator &GravitinoCatalog::PlanUpdate(ClientContext &, PhysicalPlanGenerator &, LogicalUpdate &,
                                               PhysicalOperator &) {
	throw NotImplementedException("Gravitino FILESET catalogs do not support relational table writes");
}
GravitinoCatalog &GravitinoCatalog::Get(ClientContext &context, const string &alias) {
	auto &catalog = Catalog::GetCatalog(context, alias);
	if (catalog.GetCatalogType() != "gravitino") {
		throw InvalidInputException("Catalog '%s' is not a Gravitino catalog", alias);
	}
	return catalog.Cast<GravitinoCatalog>();
}
void GravitinoCatalog::RequireMutation(ClientContext &context) const {
	if (GetAttached().IsReadOnly()) {
		throw PermissionException("Gravitino catalog is attached READ_ONLY");
	}
	if (!context.transaction.IsAutoCommit()) {
		throw TransactionException(
		    "Gravitino metadata writes require auto-commit; remote mutations cannot be rolled back");
	}
}
SchemaCatalogEntry &GravitinoCatalog::Schema(const string &name) {
	lock_guard<mutex> guard(lock);
	auto found = schemas.find(name);
	if (found != schemas.end()) {
		return *found->second;
	}
	if (schemas.size() >= 4096) {
		throw InvalidInputException("Gravitino attachment exceeds the 4096-schema limit; reattach the catalog");
	}
	CreateSchemaInfo info;
	info.schema = name;
	auto entry = make_uniq<GravitinoSchema>(*this, info);
	auto &result = *entry;
	schemas.emplace(name, std::move(entry));
	return result;
}
optional_ptr<SchemaCatalogEntry> GravitinoCatalog::LookupSchema(CatalogTransaction transaction,
                                                                const EntryLookupInfo &lookup,
                                                                OnEntryNotFound if_not_found) {
	auto name = lookup.GetEntryName();
	auto response =
	    client.Request(transaction.GetContext(), "GET", "/schemas/" + GravitinoClient::Encode(name), "", 404);
	if (response.status == 404) {
		if (if_not_found == OnEntryNotFound::THROW_EXCEPTION) {
			throw CatalogException("Gravitino schema '%s' does not exist", name);
		}
		return nullptr;
	}
	return &Schema(name);
}
void GravitinoCatalog::ScanSchemas(ClientContext &context, std::function<void(SchemaCatalogEntry &)> callback) {
	for (const auto &name : client.List(context, "/schemas")) {
		callback(Schema(name));
	}
}
optional_ptr<CatalogEntry> GravitinoCatalog::CreateSchema(CatalogTransaction transaction, CreateSchemaInfo &info) {
	auto &context = transaction.GetContext();
	RequireMutation(context);
	GravitinoClient::Identifier(info.schema);
	if (info.on_conflict == OnCreateConflict::REPLACE_ON_CONFLICT) {
		throw NotImplementedException("Gravitino does not support CREATE OR REPLACE SCHEMA");
	}
	// Allocate and enforce the attachment's cache bound before any remote write.
	auto &entry = Schema(info.schema);
	client.Request(context, "POST", "/schemas",
	               "{\"name\":" + GravitinoJson::Quote(info.schema) + ",\"properties\":{}}",
	               info.on_conflict == OnCreateConflict::IGNORE_ON_CONFLICT ? 409 : 0);
	return &entry;
}
void GravitinoCatalog::DropSchema(ClientContext &context, DropInfo &info) {
	RequireMutation(context);
	client.Request(context, "DELETE",
	               "/schemas/" + GravitinoClient::Encode(info.name) + "?cascade=" + (info.cascade ? "true" : "false"),
	               "", info.if_not_found == OnEntryNotFound::RETURN_NULL ? 404 : 0);
}

// These objects satisfy DuckDB's transaction lifecycle only. Metadata writes
// explicitly require auto-commit and are never advertised as remote ACID work.
class GravitinoTransactionManager : public TransactionManager {
public:
	explicit GravitinoTransactionManager(AttachedDatabase &db) : TransactionManager(db) {
	}
	Transaction &StartTransaction(ClientContext &context) override {
		auto transaction = make_uniq<Transaction>(*this, context);
		auto &result = *transaction;
		lock_guard<mutex> guard(lock);
		transactions.emplace(&result, std::move(transaction));
		return result;
	}
	ErrorData CommitTransaction(ClientContext &, Transaction &transaction) override {
		RollbackTransaction(transaction);
		return {};
	}
	void RollbackTransaction(Transaction &transaction) override {
		lock_guard<mutex> guard(lock);
		transactions.erase(&transaction);
	}
	void Checkpoint(ClientContext &, bool) override {
	}

private:
	mutex lock;
	unordered_map<Transaction *, unique_ptr<Transaction>> transactions;
};

static unique_ptr<Catalog> AttachGravitino(optional_ptr<StorageExtensionInfo>, ClientContext &context,
                                           AttachedDatabase &db, const string &, AttachInfo &info,
                                           AttachOptions &options) {
	GravitinoConfig config;
	config.catalog = info.path;
	for (const auto &option : options.options) {
		if (option.second.IsNull()) {
			throw InvalidInputException("Gravitino ATTACH options cannot be NULL");
		}
		auto name = StringUtil::Lower(option.first);
		if (name == "endpoint") {
			config.endpoint = option.second.GetValue<string>();
		} else if (name == "metalake") {
			config.metalake = option.second.GetValue<string>();
		} else if (name == "token") {
			config.token = option.second.GetValue<string>();
		} else if (name == "location_name") {
			config.location_name = option.second.GetValue<string>();
		} else if (name == "timeout_ms") {
			config.timeout_ms = option.second.GetValue<idx_t>();
		} else if (name == "max_response_bytes") {
			config.max_response_bytes = option.second.GetValue<idx_t>();
		} else {
			throw BinderException("Unknown Gravitino ATTACH option: %s", option.first);
		}
	}
	config.Validate();
	auto catalog = make_uniq<GravitinoCatalog>(db, std::move(config));
	auto response = catalog->client.Get(context);
	auto data = yyjson_obj_get(response.Root(), "catalog");
	if (GravitinoJson::String(data, "type") != "FILESET") {
		throw NotImplementedException("This Gravitino extension supports FILESET catalogs; "
		                              "relational catalogs require their format-specific extension");
	}
	return std::move(catalog);
}
void RegisterGravitinoCatalog(ExtensionLoader &loader) {
	auto storage = make_shared_ptr<StorageExtension>();
	storage->attach = AttachGravitino;
	storage->create_transaction_manager = [](optional_ptr<StorageExtensionInfo>, AttachedDatabase &db,
	                                         Catalog &) -> unique_ptr<TransactionManager> {
		return make_uniq<GravitinoTransactionManager>(db);
	};
	StorageExtension::Register(DBConfig::GetConfig(loader.GetDatabaseInstance()), "gravitino", std::move(storage));
}
} // namespace duckdb
