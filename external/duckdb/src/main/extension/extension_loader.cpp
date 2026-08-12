#include "duckdb/main/extension/extension_loader.hpp"

#include "duckdb/function/scalar_function.hpp"
#include "duckdb/parser/parsed_data/create_aggregate_function_info.hpp"
#include "duckdb/parser/parsed_data/create_type_info.hpp"
#include "duckdb/parser/parsed_data/create_copy_function_info.hpp"
#include "duckdb/parser/parsed_data/create_pragma_function_info.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parsed_data/create_macro_info.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/parser/parsed_data/create_collation_info.hpp"
#include "duckdb/main/extension_install_info.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/main/database.hpp"

namespace duckdb {

ExtensionLoader::ExtensionLoader(ExtensionActiveLoad &load_info)
    : db(load_info.db), extension_name(load_info.extension_name), extension_info(load_info.info) {
}

ExtensionLoader::ExtensionLoader(DatabaseInstance &db, const string &name) : db(db), extension_name(name) {
}

DatabaseInstance &ExtensionLoader::GetDatabaseInstance() {
	return db;
}

void ExtensionLoader::SetDescription(const string &description) {
	extension_description = description;
}

void ExtensionLoader::RegisterDistributedExtension(idx_t protocol_version) {
	if (distributed_manifest) {
		throw InvalidInputException("Distributed extension '%s' is already declared by this loader", extension_name);
	}
	auto manifest = make_uniq<DistributedExtensionManifest>();
	manifest->extension_name = extension_name;
	manifest->protocol_version = protocol_version;
	DistributedExtensionManager::ValidateManifest(*manifest);
	distributed_manifest = std::move(manifest);
}

void ExtensionLoader::RegisterDistributedCapability(DistributedExtensionCapabilityKind kind,
                                                    const string &capability_name, idx_t protocol_version) {
	if (kind == DistributedExtensionCapabilityKind::TABLE_FUNCTION) {
		throw InvalidInputException(
		    "Distributed table function '%s' must attach callbacks to its normal RegisterFunction registration",
		    capability_name);
	}
	if (!distributed_manifest) {
		throw InvalidInputException("Distributed extension '%s' must be declared before capability '%s'",
		                            extension_name, capability_name);
	}
	DistributedExtensionCapability capability;
	capability.kind = kind;
	capability.name = capability_name;
	capability.protocol_version = protocol_version;
	distributed_manifest->capabilities.push_back(std::move(capability));
	try {
		DistributedExtensionManager::ValidateManifest(*distributed_manifest);
	} catch (...) {
		distributed_manifest->capabilities.pop_back();
		throw;
	}
}

void ExtensionLoader::RegisterDistributedAggregateFunction(const string &capability_name, idx_t protocol_version) {
	RegisterDistributedCapability(DistributedExtensionCapabilityKind::AGGREGATE_FUNCTION, capability_name,
	                              protocol_version);
}

void ExtensionLoader::RegisterDistributedCopyFunction(const string &capability_name, idx_t protocol_version) {
	RegisterDistributedCapability(DistributedExtensionCapabilityKind::COPY_FUNCTION, capability_name, protocol_version);
}

void ExtensionLoader::RegisterDistributedOperator(const string &capability_name, idx_t protocol_version) {
	RegisterDistributedCapability(DistributedExtensionCapabilityKind::OPERATOR, capability_name, protocol_version);
}

void DistributedWriteOperatorExtension::Register(ExtensionLoader &loader, DistributedWriteOperatorExtension extension) {
	loader.RegisterDistributedWriteOperatorExtension(std::move(extension));
}

void ExtensionLoader::RegisterDistributedWriteOperatorExtension(DistributedWriteOperatorExtension extension) {
	if (!distributed_manifest) {
		throw InvalidInputException("Distributed extension '%s' must be declared before write capability '%s'",
		                            extension_name, extension.name);
	}
	DistributedExtensionCapability capability;
	capability.kind = DistributedExtensionCapabilityKind::OPERATOR;
	capability.name = extension.name;
	capability.protocol_version = extension.protocol_version;
	DistributedExtensionCapabilityReference reference;
	reference.extension_name = extension_name;
	reference.extension_protocol_version = distributed_manifest->protocol_version;
	reference.capability = capability;
	extension.callbacks.Validate(reference.CanonicalIdentity());
	auto candidate_manifest = make_uniq<DistributedExtensionManifest>(*distributed_manifest);
	candidate_manifest->capabilities.push_back(capability);
	DistributedExtensionManager::ValidateManifest(*candidate_manifest);
	distributed_write_callbacks.emplace_back(std::move(capability), std::move(extension.callbacks));
	distributed_manifest = std::move(candidate_manifest);
}

void ExtensionLoader::RegisterDistributedStorage(const string &capability_name, idx_t protocol_version) {
	RegisterDistributedCapability(DistributedExtensionCapabilityKind::STORAGE, capability_name, protocol_version);
}

void ExtensionLoader::RegisterDistributedContext(const string &capability_name, idx_t protocol_version) {
	RegisterDistributedCapability(DistributedExtensionCapabilityKind::CONTEXT, capability_name, protocol_version);
}

void ExtensionLoader::FinalizeLoad() {
	// Set extension description, if provided
	if (!extension_description.empty() && extension_info) {
		auto info = make_uniq<ExtensionLoadedInfo>();
		info->description = extension_description;
		extension_info->load_info = std::move(info);
	}
	if (distributed_manifest) {
		auto &manager = DistributedExtensionManager::Get(db);
		manager.RegisterManifest(*distributed_manifest);
		for (auto &entry : distributed_write_callbacks) {
			DistributedExtensionCapabilityReference reference;
			reference.extension_name = extension_name;
			reference.extension_protocol_version = distributed_manifest->protocol_version;
			reference.capability = entry.first;
			manager.RegisterWriteCallbacks(reference, std::move(entry.second));
		}
	}
}

void ExtensionLoader::RegisterFunction(ScalarFunction function) {
	ScalarFunctionSet set(function.name);
	set.AddFunction(std::move(function));
	RegisterFunction(std::move(set));
}

void ExtensionLoader::RegisterFunction(ScalarFunctionSet function) {
	CreateScalarFunctionInfo info(std::move(function));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	RegisterFunction(std::move(info));
}

void ExtensionLoader::RegisterFunction(CreateScalarFunctionInfo function) {
	D_ASSERT(!function.functions.name.empty());
	auto &system_catalog = Catalog::GetSystemCatalog(db);
	auto data = CatalogTransaction::GetSystemTransaction(db);
	system_catalog.CreateFunction(data, function);
}

void ExtensionLoader::RegisterFunction(AggregateFunction function) {
	AggregateFunctionSet set(function.name);
	set.AddFunction(std::move(function));
	RegisterFunction(std::move(set));
}

void ExtensionLoader::RegisterFunction(AggregateFunctionSet function) {
	CreateAggregateFunctionInfo info(std::move(function));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	RegisterFunction(std::move(info));
}

void ExtensionLoader::RegisterFunction(CreateAggregateFunctionInfo function) {
	D_ASSERT(!function.functions.name.empty());
	auto &system_catalog = Catalog::GetSystemCatalog(db);
	auto data = CatalogTransaction::GetSystemTransaction(db);
	system_catalog.CreateFunction(data, function);
}

void ExtensionLoader::RegisterFunction(CreateSecretFunction function) {
	D_ASSERT(!function.secret_type.empty());
	auto &config = DBConfig::GetConfig(db);
	config.secret_manager->RegisterSecretFunction(std::move(function), OnCreateConflict::ERROR_ON_CONFLICT);
}

void ExtensionLoader::RegisterFunction(TableFunction function) {
	TableFunctionSet set(function.name);
	set.AddFunction(std::move(function));
	RegisterFunction(std::move(set));
}

void ExtensionLoader::RegisterFunction(TableFunctionSet function) {
	D_ASSERT(!function.name.empty());
	CreateTableFunctionInfo info(std::move(function));
	info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
	RegisterFunction(std::move(info));
}

unique_ptr<DistributedExtensionManifest> ExtensionLoader::BindDistributedTableFunctions(TableFunctionSet &functions) {
	idx_t callback_count = 0;
	idx_t protocol_version = 0;
	for (const auto &function : functions.functions) {
		if (!function.HasDistributedScanCallbacks()) {
			continue;
		}
		callback_count++;
		const auto &callbacks = function.GetDistributedScanCallbacks();
		callbacks.ValidateDefinition(functions.name);
		if (protocol_version == 0) {
			protocol_version = callbacks.protocol_version;
		} else if (protocol_version != callbacks.protocol_version) {
			throw InvalidInputException(
			    "Distributed table function '%s' overloads must use one capability protocol version", functions.name);
		}
	}
	if (callback_count != 0 && callback_count != functions.functions.size()) {
		throw InvalidInputException("Distributed table function '%s' must define callbacks for every overload",
		                            functions.name);
	}

	auto existing_entry = TryGetTableFunction(functions.name);
	idx_t existing_callback_count = 0;
	if (existing_entry) {
		for (const auto &function : existing_entry->Cast<TableFunctionCatalogEntry>().functions.functions) {
			if (function.HasDistributedScanCallbacks()) {
				existing_callback_count++;
			}
		}
	}
	if (existing_entry && existing_callback_count != 0 &&
	    existing_callback_count != existing_entry->Cast<TableFunctionCatalogEntry>().functions.functions.size()) {
		throw InternalException("Distributed table function '%s' has a mixed callback registration", functions.name);
	}
	if (callback_count == 0) {
		if (existing_callback_count != 0) {
			throw InvalidInputException("Distributed table function '%s' must define callbacks for every overload",
			                            functions.name);
		}
		return nullptr;
	}
	if (existing_entry && existing_callback_count == 0) {
		throw InvalidInputException("Distributed table function '%s' cannot add callbacks to native-only overloads",
		                            functions.name);
	}
	if (!distributed_manifest) {
		throw InvalidInputException(
		    "Distributed extension '%s' must be declared before distributed table function '%s'", extension_name,
		    functions.name);
	}

	DistributedExtensionCapability capability;
	capability.kind = DistributedExtensionCapabilityKind::TABLE_FUNCTION;
	capability.name = functions.name;
	capability.protocol_version = protocol_version;
	auto candidate_manifest = make_uniq<DistributedExtensionManifest>(*distributed_manifest);
	bool capability_exists = false;
	for (const auto &registered : candidate_manifest->capabilities) {
		if (registered.kind != capability.kind || registered.name != capability.name) {
			continue;
		}
		if (registered.protocol_version != capability.protocol_version) {
			throw InvalidInputException(
			    "Distributed table function '%s' protocol mismatch: callbacks use %llu, manifest uses %llu",
			    functions.name, static_cast<unsigned long long>(capability.protocol_version),
			    static_cast<unsigned long long>(registered.protocol_version));
		}
		capability_exists = true;
	}
	if (!capability_exists) {
		candidate_manifest->capabilities.push_back(capability);
	}
	DistributedExtensionManager::ValidateManifest(*candidate_manifest);

	DistributedExtensionCapabilityReference reference;
	reference.extension_name = extension_name;
	reference.extension_protocol_version = distributed_manifest->protocol_version;
	reference.capability = capability;
	if (existing_entry) {
		for (const auto &function : existing_entry->Cast<TableFunctionCatalogEntry>().functions.functions) {
			const auto &callbacks = function.GetDistributedScanCallbacks();
			callbacks.Validate(function.name);
			if (callbacks.GetCapability() != reference) {
				throw InvalidInputException("Distributed table function '%s' is already owned by '%s'", functions.name,
				                            callbacks.GetCapability().CanonicalIdentity());
			}
		}
	}
	for (auto &function : functions.functions) {
		function.BindDistributedScanCapability(extension_name, distributed_manifest->protocol_version);
	}
	return candidate_manifest;
}

void ExtensionLoader::RegisterFunction(CreateTableFunctionInfo info) {
	D_ASSERT(!info.functions.name.empty());
	auto candidate_manifest = BindDistributedTableFunctions(info.functions);
	auto &system_catalog = Catalog::GetSystemCatalog(db);
	auto data = CatalogTransaction::GetSystemTransaction(db);
	system_catalog.CreateFunction(data, info);
	if (candidate_manifest) {
		distributed_manifest = std::move(candidate_manifest);
	}
}

void ExtensionLoader::RegisterFunction(PragmaFunction function) {
	D_ASSERT(!function.name.empty());
	PragmaFunctionSet set(function.name);
	set.AddFunction(std::move(function));
	RegisterFunction(std::move(set));
}

void ExtensionLoader::RegisterFunction(PragmaFunctionSet function) {
	D_ASSERT(!function.name.empty());
	auto function_name = function.name;
	CreatePragmaFunctionInfo info(std::move(function_name), std::move(function));
	auto &system_catalog = Catalog::GetSystemCatalog(db);
	auto data = CatalogTransaction::GetSystemTransaction(db);
	system_catalog.CreatePragmaFunction(data, info);
}

void ExtensionLoader::RegisterFunction(CopyFunction function) {
	CreateCopyFunctionInfo info(std::move(function));
	auto &system_catalog = Catalog::GetSystemCatalog(db);
	auto data = CatalogTransaction::GetSystemTransaction(db);
	system_catalog.CreateCopyFunction(data, info);
}

void ExtensionLoader::RegisterFunction(CreateMacroInfo &info) {
	auto &system_catalog = Catalog::GetSystemCatalog(db);
	auto data = CatalogTransaction::GetSystemTransaction(db);
	system_catalog.CreateFunction(data, info);
}

void ExtensionLoader::RegisterCollation(CreateCollationInfo &info) {
	auto &system_catalog = Catalog::GetSystemCatalog(db);
	auto data = CatalogTransaction::GetSystemTransaction(db);
	info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
	system_catalog.CreateCollation(data, info);

	// Also register as a function for serialisation
	CreateScalarFunctionInfo finfo(info.function);
	finfo.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
	system_catalog.CreateFunction(data, finfo);
}

void ExtensionLoader::RegisterCoordinateSystem(CreateCoordinateSystemInfo &info) {
	auto &system_catalog = Catalog::GetSystemCatalog(db);
	auto data = CatalogTransaction::GetSystemTransaction(db);
	system_catalog.CreateCoordinateSystem(data, info);
}

void ExtensionLoader::AddFunctionOverload(ScalarFunction function) {
	auto &scalar_function = GetFunction(function.name);
	scalar_function.functions.AddFunction(std::move(function));
}

void ExtensionLoader::AddFunctionOverload(ScalarFunctionSet functions) { // NOLINT
	D_ASSERT(!functions.name.empty());
	auto &scalar_function = GetFunction(functions.name);
	for (auto &function : functions.functions) {
		function.name = functions.name;
		scalar_function.functions.AddFunction(std::move(function));
	}
}

void ExtensionLoader::AddFunctionOverload(TableFunctionSet functions) { // NOLINT
	D_ASSERT(!functions.name.empty());
	for (auto &function : functions.functions) {
		function.name = functions.name;
	}
	auto candidate_manifest = BindDistributedTableFunctions(functions);
	auto &table_function = GetTableFunction(functions.name);
	for (auto &function : functions.functions) {
		table_function.functions.AddFunction(std::move(function));
	}
	if (candidate_manifest) {
		distributed_manifest = std::move(candidate_manifest);
	}
}

static optional_ptr<CatalogEntry> TryGetEntry(DatabaseInstance &db, const string &name, CatalogType type) {
	D_ASSERT(!name.empty());
	auto &system_catalog = Catalog::GetSystemCatalog(db);
	auto data = CatalogTransaction::GetSystemTransaction(db);
	auto &schema = system_catalog.GetSchema(data, DEFAULT_SCHEMA);
	return schema.GetEntry(data, type, name);
}

optional_ptr<CatalogEntry> ExtensionLoader::TryGetFunction(const string &name) {
	return TryGetEntry(db, name, CatalogType::SCALAR_FUNCTION_ENTRY);
}

ScalarFunctionCatalogEntry &ExtensionLoader::GetFunction(const string &name) {
	auto catalog_entry = TryGetFunction(name);
	if (!catalog_entry) {
		throw InvalidInputException("Function with name \"%s\" not found in ExtensionLoader::GetFunction", name);
	}
	return catalog_entry->Cast<ScalarFunctionCatalogEntry>();
}

optional_ptr<CatalogEntry> ExtensionLoader::TryGetTableFunction(const string &name) {
	return TryGetEntry(db, name, CatalogType::TABLE_FUNCTION_ENTRY);
}

TableFunctionCatalogEntry &ExtensionLoader::GetTableFunction(const string &name) {
	auto catalog_entry = TryGetTableFunction(name);
	if (!catalog_entry) {
		throw InvalidInputException("Function with name \"%s\" not found in ExtensionLoader::GetTableFunction", name);
	}
	return catalog_entry->Cast<TableFunctionCatalogEntry>();
}

void ExtensionLoader::RegisterType(string type_name, LogicalType type, bind_logical_type_function_t bind_modifiers) {
	D_ASSERT(!type_name.empty());
	CreateTypeInfo info(std::move(type_name), std::move(type), bind_modifiers);
	info.temporary = true;
	info.internal = true;
	auto &system_catalog = Catalog::GetSystemCatalog(db);
	auto data = CatalogTransaction::GetSystemTransaction(db);
	system_catalog.CreateType(data, info);
}

void ExtensionLoader::RegisterSecretType(SecretType secret_type) {
	auto &config = DBConfig::GetConfig(db);
	config.secret_manager->RegisterSecretType(secret_type);
}

void ExtensionLoader::RegisterCastFunction(const LogicalType &source, const LogicalType &target,
                                           bind_cast_function_t bind_function, int64_t implicit_cast_cost) {
	auto &config = DBConfig::GetConfig(db);
	auto &casts = config.GetCastFunctions();
	casts.RegisterCastFunction(source, target, bind_function, implicit_cast_cost);
}

void ExtensionLoader::RegisterCastFunction(const LogicalType &source, const LogicalType &target, BoundCastInfo function,
                                           int64_t implicit_cast_cost) {
	auto &config = DBConfig::GetConfig(db);
	auto &casts = config.GetCastFunctions();
	casts.RegisterCastFunction(source, target, std::move(function), implicit_cast_cost);
}

} // namespace duckdb
