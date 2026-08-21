// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/execution/distributed/extension_write_task_provider.hpp"
#include "duckdb/function/distributed_table_function.hpp"
#include "duckdb/function/distributed_write.hpp"
#include "duckdb/main/distributed_extension_manager.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

using namespace duckdb;

namespace {

static void DistributedNativeIdentity(DataChunk &input, ExpressionState &, Vector &result) {
	result.Reference(input.data[0]);
}

static DistributedWriteOperatorExtension FileWriteOperator(string name, idx_t protocol_version = 1) {
	DistributedWriteOperatorExtension result;
	result.name = std::move(name);
	result.protocol_version = protocol_version;
	result.mode = DistributedWriteMode::FILE_ARTIFACT;
	result.fragment_codec = {distributed::DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC,
	                         distributed::DISTRIBUTED_FILE_WRITE_FRAGMENT_CODEC_VERSION};
	return result;
}

class NativeContractExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		loader.RegisterFunction(ScalarFunction("distributed_native_identity", {LogicalType::INTEGER},
		                                       LogicalType::INTEGER, DistributedNativeIdentity));
		DistributedWriteOperatorExtension::Register(loader, FileWriteOperator("native_identity"));
	}

	string Name() override {
		return "native_contract";
	}

	string Version() const override {
		return "test-version";
	}
};

class FailingContractExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		DistributedWriteOperatorExtension::Register(loader, FileWriteOperator("never_published"));
		throw InvalidInputException("intentional distributed extension load failure");
	}

	string Name() override {
		return "failing_contract";
	}
};

class RetriedContractExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		REQUIRE_THROWS_WITH(DistributedWriteOperatorExtension::Register(loader, FileWriteOperator("zero", 0)),
		                    Catch::Matchers::Contains("greater than zero"));
		REQUIRE_THROWS_WITH(
		    DistributedWriteOperatorExtension::Register(loader, FileWriteOperator("Invalid Capability")),
		    Catch::Matchers::Contains("lowercase ASCII"));
		DistributedWriteOperatorExtension::Register(loader, FileWriteOperator("scan"));
	}

	string Name() override {
		return "loader_retry";
	}
};

class IncompleteWriteContractExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		DistributedWriteOperatorExtension extension;
		extension.name = "write";
		extension.protocol_version = 1;
		extension.mode = DistributedWriteMode::CALLBACK;
		extension.fragment_codec = {"incomplete-write.fragment", 1};
		DistributedWriteOperatorExtension::Register(loader, std::move(extension));
	}

	string Name() override {
		return "incomplete_write_contract";
	}
};

struct DistributedOverloadBindData : public FunctionData {
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<DistributedOverloadBindData>();
	}

	bool Equals(const FunctionData &other) const override {
		return dynamic_cast<const DistributedOverloadBindData *>(&other) != nullptr;
	}
};

static unique_ptr<FunctionData> DistributedOverloadBind(ClientContext &, TableFunctionBindInput &,
                                                        vector<LogicalType> &return_types, vector<string> &names) {
	return_types.emplace_back(LogicalType::INTEGER);
	names.emplace_back("value");
	return make_uniq<DistributedOverloadBindData>();
}

static void DistributedOverloadScan(ClientContext &, TableFunctionInput &, DataChunk &output) {
	output.SetCardinality(0);
}

static vector<DistributedScanSplit> DistributedOverloadPlan(const TableFunctionDistributedScanPlanningInput &) {
	return {};
}

static unique_ptr<FunctionData> DistributedOverloadCreateWorkerBind(const TableFunctionDistributedScanInput &input) {
	input.bind_data.Cast<DistributedOverloadBindData>();
	return make_uniq<DistributedOverloadBindData>();
}

static void DistributedOverloadApply(FunctionData &, const vector<DistributedScanSplit> &) {
}

static void DistributedOverloadSerialize(Serializer &serializer, const optional_ptr<FunctionData>,
                                         const TableFunction &) {
	serializer.WriteProperty(100, "marker", true);
}

static unique_ptr<FunctionData> DistributedOverloadDeserialize(Deserializer &deserializer, TableFunction &) {
	if (!deserializer.ReadProperty<bool>(100, "marker")) {
		throw SerializationException("invalid distributed overload marker");
	}
	return make_uniq<DistributedOverloadBindData>();
}

static TableFunction DistributedOverloadFunction(const LogicalType &argument) {
	TableFunction function({argument}, DistributedOverloadScan, DistributedOverloadBind);
	function.serialize = DistributedOverloadSerialize;
	function.deserialize = DistributedOverloadDeserialize;
	TableFunctionDistributedScanCallbacks callbacks;
	callbacks.protocol_version = 1;
	callbacks.split_codec = {"distributed-overload.split", 1};
	callbacks.plan_splits = DistributedOverloadPlan;
	callbacks.create_worker_bind = DistributedOverloadCreateWorkerBind;
	callbacks.apply_splits = DistributedOverloadApply;
	function.SetDistributedScanCallbacks(std::move(callbacks));
	return function;
}

static bool ManagerHasContractIdentity(DistributedExtensionManager &manager, const string &identity) {
	for (const auto &registered : manager.GetContractIdentities()) {
		if (registered == identity) {
			return true;
		}
	}
	return false;
}

class DistributedOverloadExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		TableFunctionSet initial("distributed_overload_scan");
		initial.AddFunction(DistributedOverloadFunction(LogicalType::INTEGER));
		loader.RegisterFunction(std::move(initial));
		TableFunctionSet overloads("distributed_overload_scan");
		overloads.AddFunction(DistributedOverloadFunction(LogicalType::BIGINT));
		loader.AddFunctionOverload(std::move(overloads));
	}

	string Name() override {
		return "distributed_overload";
	}
};

} // namespace

TEST_CASE("Distributed extension manifests are deterministic and exact", "[distributed][extension]") {
	DuckDB db(nullptr);
	auto &manager = DistributedExtensionManager::Get(*db.instance);

	DistributedExtensionManifest registered_manifest;
	registered_manifest.extension_name = "test_manifest";
	registered_manifest.capabilities.push_back({DistributedExtensionCapabilityKind::TABLE_FUNCTION, "scan", 1});
	registered_manifest.capabilities.push_back({DistributedExtensionCapabilityKind::WRITE_OPERATOR, "write", 2});
	manager.RegisterExtension(
	    registered_manifest, {make_shared_ptr<const DistributedWriteOperatorExtension>(FileWriteOperator("write", 2))});

	REQUIRE(ManagerHasContractIdentity(manager, "test_manifest{table_function:scan@1,write_operator:write@2}"));
	auto identities = manager.GetContractIdentities();
	REQUIRE_NOTHROW(manager.ValidateExact(identities));
	DistributedExtensionCapabilityReference reference;
	reference.extension_name = "test_manifest";
	reference.capability = {DistributedExtensionCapabilityKind::TABLE_FUNCTION, "scan", 1};
	REQUIRE_NOTHROW(manager.RequireCapability(reference));

	auto mismatched = identities;
	for (auto &identity : mismatched) {
		if (StringUtil::StartsWith(identity, "test_manifest{")) {
			identity = "test_manifest{table_function:scan@7,write_operator:write@2}";
		}
	}
	REQUIRE_THROWS_WITH(manager.ValidateExact(mismatched), Catch::Matchers::Contains("coordinator and worker"));
	reference.capability.protocol_version = 7;
	REQUIRE_THROWS_WITH(manager.RequireCapability(reference), Catch::Matchers::Contains("protocol mismatch"));
}

TEST_CASE("Distributed extension registration rejects ambiguous declarations", "[distributed][extension]") {
	DuckDB db(nullptr);
	auto &manager = DistributedExtensionManager::Get(*db.instance);

	DistributedExtensionManifest invalid_name {"Invalid-Name", {}};
	REQUIRE_THROWS_WITH(manager.RegisterExtension(invalid_name), Catch::Matchers::Contains("lowercase ASCII"));
	DistributedExtensionManifest empty_contract {"empty_contract", {}};
	REQUIRE_THROWS_WITH(manager.RegisterExtension(empty_contract), Catch::Matchers::Contains("concrete capability"));
	DistributedExtensionManifest zero_version {"zero_version",
	                                           {{DistributedExtensionCapabilityKind::TABLE_FUNCTION, "scan", 0}}};
	REQUIRE_THROWS_WITH(manager.RegisterExtension(zero_version), Catch::Matchers::Contains("greater than zero"));

	DistributedExtensionManifest strict;
	strict.extension_name = "strict";
	strict.capabilities.push_back({DistributedExtensionCapabilityKind::TABLE_FUNCTION, "scan", 1});
	manager.RegisterExtension(strict);
	REQUIRE_THROWS_WITH(manager.RegisterExtension(strict), Catch::Matchers::Contains("already registered"));
	strict.extension_name = "duplicate_capability";
	strict.capabilities.push_back({DistributedExtensionCapabilityKind::TABLE_FUNCTION, "scan", 2});
	REQUIRE_THROWS_WITH(manager.RegisterExtension(strict), Catch::Matchers::Contains("declared more than once"));

	DistributedExtensionManifest missing_write {"missing_write",
	                                            {{DistributedExtensionCapabilityKind::WRITE_OPERATOR, "write", 1}}};
	REQUIRE_THROWS_WITH(manager.RegisterExtension(missing_write),
	                    Catch::Matchers::Contains("no registered implementation"));
	REQUIRE_FALSE(ManagerHasContractIdentity(manager, "missing_write{write_operator:write@1}"));
}

TEST_CASE("ExtensionLoader distributed declarations do not alter native execution", "[distributed][extension]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<NativeContractExtension>();

	Connection connection(db);
	auto result = connection.Query("SELECT distributed_native_identity(42)");
	REQUIRE_NO_FAIL(*result);
	REQUIRE(CHECK_COLUMN(result, 0, {42}));

	REQUIRE(ManagerHasContractIdentity(DistributedExtensionManager::Get(*db.instance),
	                                   "native_contract{write_operator:native_identity@1}"));
}

TEST_CASE("ExtensionLoader publishes a distributed manifest only after successful load", "[distributed][extension]") {
	DuckDB db(nullptr);
	REQUIRE_THROWS_WITH(db.LoadStaticExtension<FailingContractExtension>(),
	                    Catch::Matchers::Contains("intentional distributed extension load failure"));

	auto &manager = DistributedExtensionManager::Get(*db.instance);
	REQUIRE_FALSE(ManagerHasContractIdentity(manager, "failing_contract{write_operator:never_published@1}"));

	REQUIRE_THROWS_WITH(db.LoadStaticExtension<IncompleteWriteContractExtension>(),
	                    Catch::Matchers::Contains("callbacks"));
	REQUIRE_FALSE(ManagerHasContractIdentity(manager, "incomplete_write_contract{write_operator:write@1}"));
}

TEST_CASE("Distributed write operators require an exact concrete capability", "[distributed][extension]") {
	DuckDB db(nullptr);
	auto &manager = DistributedExtensionManager::Get(*db.instance);
	DistributedExtensionManifest manifest;
	manifest.extension_name = "write_contract";
	manifest.capabilities.push_back({DistributedExtensionCapabilityKind::TABLE_FUNCTION, "scan", 1});
	manifest.capabilities.push_back({DistributedExtensionCapabilityKind::WRITE_OPERATOR, "write", 3});
	manager.RegisterExtension(
	    manifest, {make_shared_ptr<const DistributedWriteOperatorExtension>(FileWriteOperator("write", 3))});

	DistributedExtensionCapabilityReference write;
	write.extension_name = "write_contract";
	write.capability = {DistributedExtensionCapabilityKind::WRITE_OPERATOR, "write", 3};
	auto write_operator = manager.GetWriteOperator(write);
	REQUIRE(write_operator->name == "write");
	REQUIRE(write_operator->mode == DistributedWriteMode::FILE_ARTIFACT);

	auto scan = write;
	scan.capability = {DistributedExtensionCapabilityKind::TABLE_FUNCTION, "scan", 1};
	REQUIRE_THROWS_WITH(manager.GetWriteOperator(scan), Catch::Matchers::Contains("write-operator capability"));
}

TEST_CASE("ExtensionLoader distributed declaration validation has strong exception safety",
          "[distributed][extension]") {
	DuckDB db(nullptr);
	REQUIRE_NOTHROW(db.LoadStaticExtension<RetriedContractExtension>());

	REQUIRE(ManagerHasContractIdentity(DistributedExtensionManager::Get(*db.instance),
	                                   "loader_retry{write_operator:scan@1}"));
}

TEST_CASE("ExtensionLoader derives one capability across separately registered table overloads",
          "[distributed][extension]") {
	DuckDB db(nullptr);
	REQUIRE_NOTHROW(db.LoadStaticExtension<DistributedOverloadExtension>());

	REQUIRE(ManagerHasContractIdentity(DistributedExtensionManager::Get(*db.instance),
	                                   "distributed_overload{table_function:distributed_overload_scan@1}"));

	Connection connection(db);
	auto integer_result = connection.Query("SELECT * FROM distributed_overload_scan(1::INTEGER)");
	REQUIRE_NO_FAIL(*integer_result);
	auto bigint_result = connection.Query("SELECT * FROM distributed_overload_scan(1::BIGINT)");
	REQUIRE_NO_FAIL(*bigint_result);
}
