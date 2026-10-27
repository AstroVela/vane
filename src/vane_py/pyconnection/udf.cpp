// SPDX-FileCopyrightText: 2018-2026 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT AND Apache-2.0
//
// Modified by Vane contributors.

#include "vane_python/pyconnection/pyconnection.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_function_catalog_entry.hpp"
#include "duckdb/common/types/hash.hpp"
#include "duckdb/execution/operator/projection/physical_udf_inout.hpp"
#include "duckdb/function/scalar/udf_functions.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "vane_python/pybind11/gil_wrapper.hpp"
#include "vane_python/python_conversion.hpp"
#include "vane_python/python_input_callback.hpp"
#include "vane_python/python_udf_actor_resources.hpp"
#include "vane_python/python_udf_utils.hpp"
#include <cstdlib>
#include <sstream>

#include "duckdb/main/config.hpp"

namespace duckdb {

namespace {

static string PicklePythonUDFCallable(const py::function &udf) {
	PythonInputCallbackScope callback(nullptr);
	auto pickle_module = py::module_::import("vane.pickle");
	auto dumps = pickle_module.attr("dumps");
	bool annotations_rewritten = false;
	py::object original_annotations;
	if (py::hasattr(udf, "__annotations__") && py::isinstance<py::dict>(udf.attr("__annotations__"))) {
		original_annotations = py::reinterpret_borrow<py::object>(udf.attr("__annotations__"));
		py::dict sanitized_annotations;
		for (auto item : py::reinterpret_borrow<py::dict>(original_annotations)) {
			shared_ptr<DuckDBPyType> pytype;
			// A None annotation (e.g. `-> None`) casts to a null shared_ptr; keep it verbatim
			// instead of dereferencing it below.
			if (py::try_cast<shared_ptr<DuckDBPyType>>(item.second, pytype) && pytype) {
				sanitized_annotations[item.first] = py::str(pytype->ToString());
				annotations_rewritten = true;
			} else {
				sanitized_annotations[item.first] = py::reinterpret_borrow<py::object>(item.second);
			}
		}
		if (annotations_rewritten) {
			udf.attr("__annotations__") = std::move(sanitized_annotations);
		}
	}
	py::object pickled_obj;
	try {
		pickled_obj = dumps(udf);
	} catch (...) {
		if (annotations_rewritten) {
			udf.attr("__annotations__") = original_annotations;
		}
		throw;
	}
	if (annotations_rewritten) {
		udf.attr("__annotations__") = original_annotations;
	}
	auto pickled_bytes = py::reinterpret_borrow<py::bytes>(pickled_obj);
	return py::cast<string>(pickled_bytes);
}

static bool IsPythonClassCallable(const py::object &fun) {
	PythonInputCallbackScope callback(nullptr);
	auto inspect_module = py::module_::import("inspect");
	return py::cast<bool>(inspect_module.attr("isclass")(fun));
}

static string ResolveDefaultUDFExecutionBackend(const py::object &fun, const string &runner_type) {
	const bool use_actor = IsPythonClassCallable(fun);
	return ExpressionUDFExecutionBackendForRunner(runner_type, use_actor);
}

static string HashToHex(hash_t value) {
	std::ostringstream buffer;
	buffer << std::hex << static_cast<uint64_t>(value);
	return buffer.str();
}

static void CombineDigest(hash_t &digest, const string &value) {
	digest = CombineHash(digest, Hash(value.c_str(), value.size()));
}

static void CombineDigest(hash_t &digest, bool value) {
	digest = CombineHash(digest, Hash(static_cast<uint64_t>(value ? 1 : 0)));
}

static void CombineDigest(hash_t &digest, idx_t value) {
	digest = CombineHash(digest, Hash(static_cast<uint64_t>(value)));
}

static void CombineDigest(hash_t &digest, int32_t value) {
	digest = CombineHash(digest, Hash(static_cast<int64_t>(value)));
}

static void CombineDigest(hash_t &digest, double value) {
	digest = CombineHash(digest, Hash(value));
}

static bool GetStructValueField(const Value &payload, const string &name, Value &result) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return false;
	}
	auto &children = StructValue::GetChildren(payload);
	auto child_count = StructType::GetChildCount(payload.type());
	for (idx_t i = 0; i < child_count; i++) {
		if (StructType::GetChildName(payload.type(), i) != name) {
			continue;
		}
		if (i >= children.size() || children[i].IsNull()) {
			return false;
		}
		result = children[i];
		return true;
	}
	return false;
}

static std::pair<bool, string> GetStructStringField(const Value &payload, const string &name) {
	Value child;
	if (!GetStructValueField(payload, name, child)) {
		return std::make_pair(false, string());
	}
	if (child.type().id() == LogicalTypeId::VARCHAR) {
		return std::make_pair(true, StringValue::Get(child));
	}
	return std::make_pair(true, child.ToString());
}

static bool GetStructShapeField(const Value &payload, const string &name, vector<idx_t> &shape) {
	Value child;
	if (!GetStructValueField(payload, name, child)) {
		return false;
	}
	if (child.IsNull() || child.type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("python_udf output_schema tensor field '%s' must be a LIST<BIGINT>", name);
	}
	shape.clear();
	auto &children = ListValue::GetChildren(child);
	shape.reserve(children.size());
	for (auto &entry : children) {
		auto dim = entry.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
		if (dim <= 0) {
			throw InvalidInputException("python_udf output_schema tensor shape dimensions must be positive");
		}
		shape.push_back(NumericCast<idx_t>(dim));
	}
	if (shape.empty()) {
		throw InvalidInputException("python_udf output_schema tensor shape must be non-empty");
	}
	return true;
}

static bool TryParsePayloadOutputSchema(const Value &payload, vector<string> &output_names,
                                        vector<LogicalType> &output_types) {
	Value output_schema;
	if (!GetStructValueField(payload, "output_schema", output_schema)) {
		return false;
	}
	auto &entries = ListValue::GetChildren(output_schema);
	vector<LogicalType> contract_types;
	Value output_contracts;
	if (GetStructValueField(payload, "output_contract_types", output_contracts)) {
		if (output_contracts.type().id() != LogicalTypeId::LIST) {
			throw InvalidInputException("python_udf output_contract_types must be a LIST<VARCHAR>");
		}
		auto &contracts = ListValue::GetChildren(output_contracts);
		if (contracts.size() != entries.size()) {
			throw InvalidInputException("python_udf output contract count does not match output_schema");
		}
		contract_types.reserve(contracts.size());
		for (auto &contract : contracts) {
			if (contract.IsNull() || contract.type().id() != LogicalTypeId::VARCHAR) {
				throw InvalidInputException("python_udf output_contract_types must contain VARCHAR values");
			}
			contract_types.push_back(DBConfig::ParseLogicalType(StringValue::Get(contract)));
		}
	}
	output_names.clear();
	output_types.clear();
	output_names.reserve(entries.size());
	output_types.reserve(entries.size());
	for (idx_t index = 0; index < entries.size(); index++) {
		auto &entry = entries[index];
		if (entry.IsNull() || entry.type().id() != LogicalTypeId::STRUCT) {
			throw InvalidInputException("python_udf output_schema entries must be STRUCT values");
		}
		auto entry_name = GetStructStringField(entry, "name");
		if (!entry_name.first || entry_name.second.empty()) {
			throw InvalidInputException("python_udf output_schema entry is missing name");
		}
		auto entry_kind = GetStructStringField(entry, "kind");
		if (!entry_kind.first || entry_kind.second.empty() || StringUtil::CIEquals(entry_kind.second, "duckdb_type")) {
			auto entry_type = GetStructStringField(entry, "type");
			if (!entry_type.first || entry_type.second.empty()) {
				throw InvalidInputException("python_udf output_schema duckdb_type entry is missing type");
			}
			output_names.push_back(entry_name.second);
			output_types.push_back(contract_types.empty() ? DBConfig::ParseLogicalType(entry_type.second)
			                                              : contract_types[index]);
			continue;
		}
		if (StringUtil::CIEquals(entry_kind.second, "tensor")) {
			auto entry_dtype = GetStructStringField(entry, "dtype");
			if (!entry_dtype.first || entry_dtype.second.empty()) {
				throw InvalidInputException("python_udf output_schema tensor entry is missing dtype");
			}
			vector<idx_t> shape;
			if (!GetStructShapeField(entry, "shape", shape)) {
				throw InvalidInputException("python_udf output_schema tensor entry is missing shape");
			}
			output_names.push_back(entry_name.second);
			output_types.push_back(contract_types.empty()
			                           ? TensorType::Create(DBConfig::ParseLogicalType(entry_dtype.second), shape)
			                           : contract_types[index]);
			continue;
		}
		throw InvalidInputException("Unsupported python_udf output_schema kind '%s'", entry_kind.second);
	}
	return true;
}

static string GetRegistrationDigest(const py::dict &spec) {
	auto digest_obj = spec[py::str("digest")];
	return py::cast<string>(digest_obj);
}

static py::dict BuildDistributedScalarUDFRegistration(const string &name, const py::function &udf,
                                                      const vector<LogicalType> &parameter_types,
                                                      const LogicalType &return_type, PythonUDFType type,
                                                      FunctionNullHandling null_handling,
                                                      PythonExceptionHandling exception_handling, bool side_effects) {
	auto function_pickle = PicklePythonUDFCallable(udf);

	py::list parameters;
	for (auto &entry : parameter_types) {
		parameters.append(py::str(entry.ToString()));
	}

	hash_t digest = Hash("scalar", 6);
	CombineDigest(digest, name);
	CombineDigest(digest, function_pickle);
	for (auto &entry : parameter_types) {
		CombineDigest(digest, entry.ToString());
	}
	CombineDigest(digest, return_type.ToString());
	CombineDigest(digest, static_cast<int32_t>(type));
	CombineDigest(digest, static_cast<int32_t>(null_handling));
	CombineDigest(digest, static_cast<int32_t>(exception_handling));
	CombineDigest(digest, side_effects);

	py::dict spec;
	spec[py::str("kind")] = py::str("scalar");
	spec[py::str("name")] = py::str(name);
	spec[py::str("digest")] = py::str(HashToHex(digest));
	spec[py::str("function_pickle")] = py::bytes(function_pickle);
	spec[py::str("parameters")] = std::move(parameters);
	spec[py::str("return_type")] = py::str(return_type.ToString());
	spec[py::str("udf_type")] = py::str(type == PythonUDFType::ARROW ? "arrow" : "native");
	spec[py::str("null_handling")] = py::int_(static_cast<int32_t>(null_handling));
	spec[py::str("exception_handling")] = py::int_(static_cast<int32_t>(exception_handling));
	spec[py::str("side_effects")] = py::bool_(side_effects);
	return spec;
}

static py::dict BuildDistributedTableUDFRegistration(const string &name, const py::function &udf,
                                                     const vector<string> &output_names,
                                                     const vector<LogicalType> &output_types,
                                                     const Optional<py::object> &batch_size) {
	auto function_pickle = PicklePythonUDFCallable(udf);

	py::dict schema;
	for (idx_t i = 0; i < output_names.size(); i++) {
		schema[py::str(output_names[i])] = py::str(output_types[i].ToString());
	}

	py::object batch_size_obj = batch_size.is_none() ? py::none() : py::reinterpret_borrow<py::object>(batch_size);

	hash_t digest = Hash("table", 5);
	CombineDigest(digest, name);
	CombineDigest(digest, function_pickle);
	for (idx_t i = 0; i < output_names.size(); i++) {
		CombineDigest(digest, output_names[i]);
		CombineDigest(digest, output_types[i].ToString());
	}
	CombineDigest(digest, !batch_size_obj.is_none());
	if (!batch_size_obj.is_none()) {
		CombineDigest(digest, static_cast<idx_t>(py::cast<int64_t>(batch_size_obj)));
	}

	py::dict spec;
	spec[py::str("kind")] = py::str("table");
	spec[py::str("name")] = py::str(name);
	spec[py::str("digest")] = py::str(HashToHex(digest));
	spec[py::str("function_pickle")] = py::bytes(function_pickle);
	spec[py::str("schema")] = std::move(schema);
	spec[py::str("batch_size")] = batch_size_obj;
	return spec;
}

static Value UpsertStructStringField(const Value &payload, const string &field_name, const string &field_value) {
	if (payload.IsNull() || payload.type().id() != LogicalTypeId::STRUCT) {
		return payload;
	}
	auto &existing_children = StructValue::GetChildren(payload);
	auto &existing_type = payload.type();
	child_list_t<Value> new_children;
	bool found = false;
	for (idx_t i = 0; i < StructType::GetChildCount(existing_type); i++) {
		auto child_name = StructType::GetChildName(existing_type, i);
		if (child_name == field_name) {
			new_children.emplace_back(child_name, Value(field_value));
			found = true;
		} else {
			new_children.emplace_back(child_name, existing_children[i]);
		}
	}
	if (!found) {
		new_children.emplace_back(field_name, Value(field_value));
	}
	return Value::STRUCT(std::move(new_children));
}

static vector<LogicalType> ParseVaneSQLParameters(const py::object &parameters, const string &required_message) {
	PythonInputCallbackScope callback(nullptr);
	if (parameters.is_none()) {
		throw InvalidInputException(required_message);
	}
	if (!py::isinstance<py::list>(parameters) && !py::isinstance<py::tuple>(parameters)) {
		throw InvalidInputException("parameters must be a list or tuple of DuckDB types");
	}
	auto params = py::list(parameters);
	vector<LogicalType> result;
	result.reserve(params.size());
	for (auto &param : params) {
		auto type = py::cast<shared_ptr<DuckDBPyType>>(param);
		result.push_back(type->Type());
	}
	return result;
}

static vector<string> ParseVaneSQLInputNames(const py::object &input_names) {
	PythonInputCallbackScope callback(nullptr);
	if (!py::isinstance<py::list>(input_names) && !py::isinstance<py::tuple>(input_names)) {
		throw InvalidInputException("input_names must be a non-empty list or tuple");
	}
	auto names = py::list(input_names);
	if (names.empty()) {
		throw InvalidInputException("input_names must be a non-empty list or tuple");
	}
	vector<string> result;
	result.reserve(names.size());
	for (auto &name_obj : names) {
		if (!py::isinstance<py::str>(name_obj)) {
			throw InvalidInputException("input_names must contain only strings");
		}
		auto name = py::cast<string>(name_obj);
		if (name.empty()) {
			throw InvalidInputException("input_names must contain only non-empty strings");
		}
		for (auto &existing_name : result) {
			if (StringUtil::CIEquals(existing_name, name)) {
				throw InvalidInputException("input_names must be unique (case-insensitive); duplicate name: '%s'",
				                            name);
			}
		}
		result.push_back(std::move(name));
	}
	return result;
}

static void ValidateVaneSQLSingleOutputSchema(const py::object &schema) {
	if (schema.is_none()) {
		throw InvalidInputException("schema is required for batch SQL registration");
	}
	if (!py::isinstance<py::dict>(schema)) {
		throw InvalidInputException("'schema' should be given as a Dict[str, DuckDBType]");
	}
	auto schema_dict = py::reinterpret_borrow<py::dict>(schema);
	if (schema_dict.size() != 1) {
		throw InvalidInputException("map_batches expression requires exactly one output column");
	}
}

static unique_ptr<FunctionData> RegisteredVaneUDFBind(ClientContext &, ScalarFunction &bound_function,
                                                      vector<unique_ptr<Expression>> &) {
	if (!bound_function.function_info) {
		throw BinderException("vane SQL UDF is missing registered payload metadata");
	}
	auto info = dynamic_cast<RegisteredUDFFunctionInfo *>(bound_function.function_info.get());
	if (!info) {
		throw BinderException("vane SQL UDF has invalid registered payload metadata");
	}
	if (info->payload.IsNull()) {
		throw BinderException("vane SQL UDF payload cannot be NULL");
	}
	auto payload = info->payload;
	auto return_type = udf_helpers::ResolvePayloadReturnType(payload);
	bound_function.SetReturnType(return_type);
	return make_uniq<UDFFunctionData>(std::move(payload), std::move(return_type));
}

static void RegisteredVaneUDFExecute(DataChunk &, ExpressionState &, Vector &) {
	throw InvalidInputException("vane SQL UDF can only be used in a projection");
}

static bool InjectVaneCatalogRegistrationFailureForTesting(const string &name) {
	const char *test_hooks = std::getenv("VANE_ENABLE_UDF_TEST_HOOKS");
	if (!test_hooks || string(test_hooks) != "1") {
		return false;
	}
	const char *target_alias = std::getenv("VANE_TEST_FAIL_UDF_CATALOG_REGISTRATION");
	return target_alias && name == string(target_alias);
}

static void RegisterVaneScalarFunctionAtomically(
    ClientContext &context, case_insensitive_map_t<unique_ptr<ExternalDependency>> &registered_functions,
    ScalarFunction scalar_function, unique_ptr<ExternalDependency> new_dependency, bool replace) {
	auto owned_entry = registered_functions.find(scalar_function.name);
	const bool owns_existing_alias = owned_entry != registered_functions.end();
	if (owns_existing_alias && !replace) {
		throw NotImplementedException("A function by the name of '%s' is already created, creating multiple "
		                              "functions with the same name is not supported yet; pass replace=True",
		                              scalar_function.name);
	}

	CreateScalarFunctionInfo info(std::move(scalar_function));
	const auto name = info.name;
	bool dependency_swapped = false;
	bool dependency_inserted = false;
	try {
		context.RunFunctionInTransaction([&]() {
			auto catalog_entry = Catalog::GetEntry<ScalarFunctionCatalogEntry>(context, SYSTEM_CATALOG, DEFAULT_SCHEMA,
			                                                                   name, OnEntryNotFound::RETURN_NULL);
			if (!owns_existing_alias) {
				if (replace && catalog_entry) {
					throw InvalidInputException("Cannot replace function '%s': the catalog entry is not a Vane alias "
					                            "owned by this connection (it may be builtin or registered elsewhere)",
					                            name);
				}
			} else {
				if (!catalog_entry) {
					throw InvalidInputException("Cannot replace owned Vane alias '%s': its catalog entry is missing",
					                            name);
				}
				if (catalog_entry->functions.functions.size() != 1 || info.functions.functions.size() != 1) {
					throw InvalidInputException("Cannot replace owned Vane alias '%s': expected exactly one overload",
					                            name);
				}
				auto &existing_function = catalog_entry->functions.functions.front();
				if (!dynamic_cast<RegisteredUDFFunctionInfo *>(existing_function.function_info.get())) {
					throw InvalidInputException(
					    "Cannot replace function '%s': the existing entry is not a registered Vane "
					    "SQL alias owned by this connection",
					    name);
				}
				if (!info.functions.functions.front().Equal(existing_function)) {
					throw InvalidInputException("Cannot replace owned Vane alias '%s' with a different SQL signature",
					                            name);
				}
				info.on_conflict = OnCreateConflict::ALTER_ON_CONFLICT;
			}

			auto &catalog = Catalog::GetSystemCatalog(context);
			catalog.CreateFunction(context, info);
			if (owns_existing_alias) {
				std::swap(owned_entry->second, new_dependency);
				dependency_swapped = true;
			} else {
				auto insertion = registered_functions.emplace(name, std::move(new_dependency));
				if (!insertion.second) {
					throw InternalException("Failed to record dependency for newly registered Vane alias '%s'", name);
				}
				dependency_inserted = true;
			}
			if (InjectVaneCatalogRegistrationFailureForTesting(name)) {
				throw InternalException("injected Vane catalog registration failure after dependency swap for '%s'",
				                        name);
			}
		});
	} catch (...) {
		// RunFunctionInTransaction commits after its callback. Cover both callback and commit failures so the
		// Python dependency map always describes the catalog version that survived the transaction.
		if (dependency_swapped) {
			std::swap(owned_entry->second, new_dependency);
		}
		if (dependency_inserted) {
			registered_functions.erase(name);
		}
		throw;
	}
}

} // namespace

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::UnregisterUDF(const string &name) {
	auto query_lock = LockForQuery();
	auto entry = registered_functions.find(name);
	if (entry == registered_functions.end()) {
		// Not registered or already unregistered
		throw InvalidInputException("No function by the name of '%s' was found in the list of registered functions",
		                            name);
	}
	auto catalog_type_entry = registered_function_catalog_types.find(name);
	if (catalog_type_entry == registered_function_catalog_types.end()) {
		throw InternalException("Registered Python function '%s' is missing its catalog type", name);
	}
	const auto catalog_type = catalog_type_entry->second;

	auto &connection = con.GetConnection();
	auto &context = *connection.context;

	context.RunFunctionInTransaction([&]() {
		auto &catalog = Catalog::GetCatalog(context, SYSTEM_CATALOG);
		if (catalog_type == PythonUDFCatalogType::SCALAR) {
			auto scalar_entry = Catalog::GetEntry<ScalarFunctionCatalogEntry>(context, SYSTEM_CATALOG, DEFAULT_SCHEMA,
			                                                                  name, OnEntryNotFound::RETURN_NULL);
			if (!scalar_entry) {
				throw InvalidInputException("No scalar function by the name of '%s' was found in the catalog", name);
			}
			DropInfo info;
			info.type = CatalogType::SCALAR_FUNCTION_ENTRY;
			info.name = name;
			info.allow_drop_internal = true;
			info.cascade = false;
			info.if_not_found = OnEntryNotFound::THROW_EXCEPTION;
			catalog.DropEntry(context, info);
		} else {
			auto table_entry = Catalog::GetEntry<TableFunctionCatalogEntry>(context, SYSTEM_CATALOG, DEFAULT_SCHEMA,
			                                                                name, OnEntryNotFound::RETURN_NULL);
			if (!table_entry) {
				throw InvalidInputException("No table function by the name of '%s' was found in the catalog", name);
			}
			DropInfo info;
			info.type = CatalogType::TABLE_FUNCTION_ENTRY;
			info.name = name;
			info.allow_drop_internal = true;
			info.cascade = false;
			info.if_not_found = OnEntryNotFound::THROW_EXCEPTION;
			catalog.DropEntry(context, info);
		}
	});
	registered_functions.erase(entry);
	registered_function_catalog_types.erase(catalog_type_entry);
	distributed_python_udf_registrations.erase(name);
	applied_distributed_python_udf_digests.erase(name);

	return shared_from_this();
}

py::list DuckDBPyConnection::ExportDistributedPythonUDFRegistrations() const {
	py::list registrations;
	for (auto &entry : distributed_python_udf_registrations) {
		registrations.append(entry.second);
	}
	return registrations;
}

void DuckDBPyConnection::ApplyDistributedPythonUDFRegistrations(const py::object &registrations_obj) {
	if (registrations_obj.is_none()) {
		return;
	}
	if (!py::isinstance<py::iterable>(registrations_obj)) {
		throw InvalidInputException("distributed Python UDF registrations must be iterable");
	}

	auto pickle_module = py::module_::import("vane.pickle");
	auto loads = pickle_module.attr("loads");

	for (auto item : py::iterable(registrations_obj)) {
		auto spec = py::reinterpret_borrow<py::dict>(item);
		auto kind = py::cast<string>(spec[py::str("kind")]);
		auto name = py::cast<string>(spec[py::str("name")]);
		auto digest = GetRegistrationDigest(spec);

		auto existing_digest = applied_distributed_python_udf_digests.find(name);
		if (existing_digest != applied_distributed_python_udf_digests.end()) {
			if (existing_digest->second == digest) {
				continue;
			}
			UnregisterUDF(name);
		}

		auto pickled_fn = spec[py::str("function_pickle")].cast<py::bytes>();
		auto udf = py::cast<py::function>(loads(pickled_fn));

		if (StringUtil::CIEquals(kind, "scalar")) {
			py::list parameters;
			for (auto &entry : py::reinterpret_borrow<py::list>(spec[py::str("parameters")])) {
				parameters.append(Type(py::cast<string>(entry)));
			}

			auto return_type_str = py::cast<string>(spec[py::str("return_type")]);
			auto return_type = Type(return_type_str);
			auto udf_type_str = py::cast<string>(spec[py::str("udf_type")]);
			auto udf_type = StringUtil::CIEquals(udf_type_str, "arrow") ? PythonUDFType::ARROW : PythonUDFType::NATIVE;
			auto null_handling = static_cast<FunctionNullHandling>(py::cast<int32_t>(spec[py::str("null_handling")]));
			auto exception_handling =
			    static_cast<PythonExceptionHandling>(py::cast<int32_t>(spec[py::str("exception_handling")]));
			auto side_effects = py::cast<bool>(spec[py::str("side_effects")]);
			RegisterScalarUDF(name, udf, parameters, return_type, udf_type, null_handling, exception_handling,
			                  side_effects);
		} else if (StringUtil::CIEquals(kind, "table")) {
			py::dict schema;
			for (auto &entry : py::reinterpret_borrow<py::dict>(spec[py::str("schema")])) {
				schema[py::reinterpret_borrow<py::object>(entry.first)] = Type(py::cast<string>(entry.second));
			}

			auto batch_size = py::reinterpret_borrow<py::object>(spec[py::str("batch_size")]);

			RegisterTableUDF(name, udf, schema, nullptr, batch_size);
		} else {
			throw InvalidInputException("Unknown distributed Python UDF registration kind '%s'", kind);
		}

		applied_distributed_python_udf_digests[name] = std::move(digest);
	}
}

shared_ptr<DuckDBPyConnection>
DuckDBPyConnection::RegisterScalarUDF(const string &name, const py::function &udf, const py::object &parameters_p,
                                      const shared_ptr<DuckDBPyType> &return_type_p, PythonUDFType type,
                                      FunctionNullHandling null_handling, PythonExceptionHandling exception_handling,
                                      bool side_effects) {
	auto query_lock = LockForQuery();
	ValidateSynchronousUDFCallable(udf);
	auto &connection = con.GetConnection();
	auto &context = *connection.context;

	if (context.transaction.HasActiveTransaction()) {
		context.CancelTransaction();
	}
	if (registered_functions.find(name) != registered_functions.end()) {
		throw NotImplementedException("A function by the name of '%s' is already created, creating multiple "
		                              "functions with the same name is not supported yet, please remove it first",
		                              name);
	}
	auto scalar_function = CreateScalarUDF(name, udf, parameters_p, return_type_p, type == PythonUDFType::ARROW,
	                                       null_handling, exception_handling, side_effects);
	CreateScalarFunctionInfo info(scalar_function);

	context.RegisterFunction(info);

	auto dependency = make_uniq<ExternalDependency>();
	dependency->AddDependency("function", PythonDependencyItem::Create(udf));
	registered_functions[name] = std::move(dependency);
	registered_function_catalog_types[name] = PythonUDFCatalogType::SCALAR;
	auto registration =
	    BuildDistributedScalarUDFRegistration(name, udf, scalar_function.arguments, scalar_function.return_type, type,
	                                          null_handling, exception_handling, side_effects);
	applied_distributed_python_udf_digests[name] = GetRegistrationDigest(registration);
	distributed_python_udf_registrations[name] = std::move(registration);

	return shared_from_this();
}

shared_ptr<DuckDBPyConnection>
DuckDBPyConnection::CreateVaneFunctionInternal(const string &name, const py::object &callable,
                                               const py::object &parameters,
                                               const shared_ptr<DuckDBPyType> &return_type, bool replace) {
	PythonGILWrapper gil;
	CheckCallbackEntry();
	auto helpers = py::module_::import("vane._expression_udf");
	auto unwrap = helpers.attr("_unwrap_vane_function");
	auto normalize_parameters = helpers.attr("_normalize_sql_type_list");

	auto unwrapped = py::cast<py::tuple>(unwrap(callable));
	auto udf = py::cast<py::function>(unwrapped[0]);
	auto decorator_return_type = py::reinterpret_borrow<py::object>(unwrapped[1]);

	shared_ptr<DuckDBPyType> resolved_return_type = return_type;
	if (!resolved_return_type && !decorator_return_type.is_none()) {
		if (py::isinstance<py::str>(decorator_return_type)) {
			resolved_return_type = Type(py::cast<string>(decorator_return_type));
		} else {
			resolved_return_type = py::cast<shared_ptr<DuckDBPyType>>(decorator_return_type);
		}
	}
	if (!resolved_return_type) {
		throw InvalidInputException("return_dtype is required for SQL vane.func registration");
	}

	py::object normalized_parameters = normalize_parameters(parameters);
	auto parameter_types =
	    ParseVaneSQLParameters(normalized_parameters, "parameters is required for SQL vane.func registration");

	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();
	auto &context = *connection.context;
	if (context.transaction.HasActiveTransaction()) {
		context.CancelTransaction();
	}
	auto default_parallelism = static_cast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
	auto payload = BuildExpressionScalarUDFPayload(name, udf, resolved_return_type,
	                                               ExpressionUDFExecutionBackendForRunner(GetRunnerType(), false),
	                                               default_parallelism, parameter_types.size(), py::none());

	ScalarFunction scalar_function(name, std::move(parameter_types), LogicalType::ANY, RegisteredVaneUDFExecute,
	                               RegisteredVaneUDFBind, nullptr, nullptr, nullptr, LogicalType::INVALID,
	                               FunctionStability::VOLATILE);
	scalar_function.SetExtraFunctionInfo(
	    make_shared_ptr<RegisteredUDFFunctionInfo>(payload, vector<string> {}, /*allow_named_arguments=*/false));
	scalar_function.SetBindExpressionCallback(LowerRegisteredExpressionUDF);

	auto dependency = make_uniq<ExternalDependency>();
	dependency->AddDependency("function", PythonDependencyItem::Create(udf));
	RegisterVaneScalarFunctionAtomically(context, registered_functions, std::move(scalar_function),
	                                     std::move(dependency), replace);
	registered_function_catalog_types[name] = PythonUDFCatalogType::SCALAR;

	return shared_from_this();
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::CreateVaneBatchFunctionInternal(
    const string &name, const py::object &udf, const py::object &input_names, const py::object &schema,
    const py::object &parameters, const Optional<py::object> &batch_size, const Optional<py::object> &gpus,
    const Optional<py::object> &actor_number, bool row_preserving, bool replace) {
	PythonGILWrapper gil;
	CheckCallbackEntry();
	if (!row_preserving) {
		throw InvalidInputException("row_preserving=False is supported by the expression API, but SQL attach v1 "
		                            "requires row-preserving batch UDFs");
	}
	auto helpers = py::module_::import("vane._expression_udf");
	auto normalize_parameters = helpers.attr("_normalize_sql_type_list");
	auto normalize_schema = helpers.attr("_normalize_schema");

	auto udf_function = py::cast<py::function>(udf);
	auto parsed_input_names = ParseVaneSQLInputNames(input_names);
	auto normalized_schema = py::reinterpret_borrow<py::object>(normalize_schema(schema));
	ValidateVaneSQLSingleOutputSchema(normalized_schema);

	py::object normalized_parameters = normalize_parameters(parameters);
	auto parameter_types =
	    ParseVaneSQLParameters(normalized_parameters, "parameters is required for batch SQL vane.func registration");
	if (parsed_input_names.size() != parameter_types.size()) {
		throw InvalidInputException("input_names count must match parameters count");
	}

	auto query_lock = LockForQuery();
	auto &connection = con.GetConnection();
	auto &context = *connection.context;
	if (context.transaction.HasActiveTransaction()) {
		context.CancelTransaction();
	}
	const bool use_actor_backend = !actor_number.is_none();
	auto default_parallelism = static_cast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
	auto payload = BuildExpressionMapBatchesUDFPayload(
	    name, udf_function, normalized_schema,
	    ExpressionUDFExecutionBackendForRunner(GetRunnerType(), use_actor_backend), default_parallelism,
	    parsed_input_names, batch_size, row_preserving, gpus, actor_number, py::none());

	ScalarFunction scalar_function(name, std::move(parameter_types), LogicalType::ANY, RegisteredVaneUDFExecute,
	                               RegisteredVaneUDFBind, nullptr, nullptr, nullptr, LogicalType::INVALID,
	                               FunctionStability::VOLATILE);
	scalar_function.SetExtraFunctionInfo(
	    make_shared_ptr<RegisteredUDFFunctionInfo>(payload, parsed_input_names, /*allow_named_arguments=*/true));
	scalar_function.SetBindExpressionCallback(LowerRegisteredExpressionUDF);

	auto dependency = make_uniq<ExternalDependency>();
	dependency->AddDependency("function", PythonDependencyItem::Create(udf_function));
	dependency->AddDependency("schema", PythonDependencyItem::Create(normalized_schema));
	RegisterVaneScalarFunctionAtomically(context, registered_functions, std::move(scalar_function),
	                                     std::move(dependency), replace);
	registered_function_catalog_types[name] = PythonUDFCatalogType::SCALAR;

	return shared_from_this();
}

shared_ptr<DuckDBPyConnection> DuckDBPyConnection::RegisterTableUDF(const string &name, const py::function &udf,
                                                                    const py::object &schema,
                                                                    const shared_ptr<DuckDBPyType> &return_type_p,
                                                                    const Optional<py::object> &batch_size) {
	auto query_lock = LockForQuery();
	ValidateSynchronousUDFCallable(udf);
	auto &connection = con.GetConnection();
	auto &context = *connection.context;

	if (context.transaction.HasActiveTransaction()) {
		context.CancelTransaction();
	}
	if (registered_functions.find(name) != registered_functions.end()) {
		throw NotImplementedException("A function by the name of '%s' is already created, creating multiple "
		                              "functions with the same name is not supported yet, please remove it first",
		                              name);
	}

	auto default_parallelism = static_cast<idx_t>(TaskScheduler::GetScheduler(context).NumberOfThreads());
	auto payload =
	    BuildPythonUDFPayload(name, udf, schema, return_type_p, ResolveDefaultUDFExecutionBackend(udf, GetRunnerType()),
	                          default_parallelism, py::none(), py::none(), py::none(), batch_size, py::none(),
	                          py::none(), py::none(), py::none(), py::none(), py::none(), py::none());

	// Parse output types/names from the payload for MakeUDFRegisteredTableFunction
	vector<LogicalType> output_types;
	vector<string> output_names;
	if (!TryParsePayloadOutputSchema(payload, output_names, output_types)) {
		throw InvalidInputException("python_udf payload is missing output_schema");
	}

	auto registration_output_types = output_types;
	auto registration_output_names = output_names;
	auto registration = BuildDistributedTableUDFRegistration(name, udf, registration_output_names,
	                                                         registration_output_types, batch_size);
	auto registration_digest = GetRegistrationDigest(registration);
	payload = UpsertStructStringField(payload, "udf_id", registration_digest);
	auto function =
	    MakeUDFRegisteredTableFunction(name, std::move(payload), std::move(output_types), std::move(output_names));
	CreateTableFunctionInfo info(std::move(function));
	context.RegisterFunction(info);

	auto dependency = make_uniq<ExternalDependency>();
	dependency->AddDependency("function", PythonDependencyItem::Create(udf));
	if (!schema.is_none()) {
		dependency->AddDependency("schema", PythonDependencyItem::Create(schema));
	}
	registered_functions[name] = std::move(dependency);
	registered_function_catalog_types[name] = PythonUDFCatalogType::TABLE;
	applied_distributed_python_udf_digests[name] = std::move(registration_digest);
	distributed_python_udf_registrations[name] = std::move(registration);

	return shared_from_this();
}

} // namespace duckdb
