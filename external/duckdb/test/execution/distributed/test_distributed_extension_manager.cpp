// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb.hpp"
#include "duckdb/main/distributed_extension_manager.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

using namespace duckdb;

namespace {

static void DistributedNativeIdentity(DataChunk &input, ExpressionState &, Vector &result) {
	result.Reference(input.data[0]);
}

class NativeContractExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		loader.RegisterDistributedExtension(1);
		loader.RegisterDistributedTableFunction("native_identity", 1);
		loader.RegisterFunction(ScalarFunction("distributed_native_identity", {LogicalType::INTEGER},
		                                       LogicalType::INTEGER, DistributedNativeIdentity));
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
		loader.RegisterDistributedExtension(1);
		loader.RegisterDistributedTableFunction("never_published", 1);
		throw InvalidInputException("intentional distributed extension load failure");
	}

	string Name() override {
		return "failing_contract";
	}
};

class RetriedContractExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override {
		REQUIRE_THROWS_WITH(loader.RegisterDistributedExtension(0), Catch::Matchers::Contains("greater than zero"));
		loader.RegisterDistributedExtension(1);
		REQUIRE_THROWS_WITH(loader.RegisterDistributedTableFunction("Invalid Capability", 1),
		                    Catch::Matchers::Contains("lowercase ASCII"));
		loader.RegisterDistributedTableFunction("scan", 1);
	}

	string Name() override {
		return "loader_retry";
	}
};

} // namespace

TEST_CASE("Distributed extension manifests are deterministic and exact", "[distributed][extension]") {
	DuckDB db(nullptr);
	auto &manager = DistributedExtensionManager::Get(*db.instance);

	DistributedExtensionManifest registered_manifest;
	registered_manifest.extension_name = "test_manifest";
	registered_manifest.protocol_version = 3;
	registered_manifest.capabilities.push_back({DistributedExtensionCapabilityKind::COPY_FUNCTION, "write", 2});
	registered_manifest.capabilities.push_back({DistributedExtensionCapabilityKind::TABLE_FUNCTION, "scan", 1});
	manager.RegisterManifest(registered_manifest);

	auto manifests = manager.GetExtensions();
	DistributedExtensionManifest test_manifest;
	REQUIRE(manager.TryGetExtension("test_manifest", test_manifest));
	REQUIRE(test_manifest.CanonicalIdentity() == "test_manifest@3{table_function:scan@1,copy_function:write@2}");
	REQUIRE_NOTHROW(manager.ValidateExact(manifests));
	REQUIRE_NOTHROW(
	    manager.RequireCapability("test_manifest", DistributedExtensionCapabilityKind::TABLE_FUNCTION, "scan", 1));

	auto mismatched = manifests;
	for (auto &manifest : mismatched) {
		if (manifest.extension_name == "test_manifest") {
			manifest.capabilities[0].protocol_version = 7;
		}
	}
	REQUIRE_THROWS_WITH(manager.ValidateExact(mismatched), Catch::Matchers::Contains("coordinator and worker"));
	REQUIRE_THROWS_WITH(
	    manager.RequireCapability("test_manifest", DistributedExtensionCapabilityKind::TABLE_FUNCTION, "scan", 7),
	    Catch::Matchers::Contains("protocol mismatch"));
	DistributedExtensionCapabilityReference reference;
	reference.extension_name = "test_manifest";
	reference.extension_protocol_version = 4;
	reference.capability.kind = DistributedExtensionCapabilityKind::TABLE_FUNCTION;
	reference.capability.name = "scan";
	reference.capability.protocol_version = 1;
	REQUIRE_THROWS_WITH(manager.RequireCapability(reference), Catch::Matchers::Contains("manifest protocol mismatch"));
}

TEST_CASE("Distributed extension registration rejects ambiguous declarations", "[distributed][extension]") {
	DuckDB db(nullptr);
	auto &manager = DistributedExtensionManager::Get(*db.instance);

	DistributedExtensionManifest invalid_name {"Invalid-Name", 1, {}};
	REQUIRE_THROWS_WITH(manager.RegisterManifest(invalid_name), Catch::Matchers::Contains("lowercase ASCII"));
	DistributedExtensionManifest zero_version {"zero_version", 0, {}};
	REQUIRE_THROWS_WITH(manager.RegisterManifest(zero_version), Catch::Matchers::Contains("greater than zero"));

	DistributedExtensionManifest strict;
	strict.extension_name = "strict";
	strict.protocol_version = 1;
	strict.capabilities.push_back({DistributedExtensionCapabilityKind::TABLE_FUNCTION, "scan", 1});
	manager.RegisterManifest(strict);
	REQUIRE_THROWS_WITH(manager.RegisterManifest(strict), Catch::Matchers::Contains("already registered"));
	strict.extension_name = "duplicate_capability";
	strict.capabilities.push_back({DistributedExtensionCapabilityKind::TABLE_FUNCTION, "scan", 2});
	REQUIRE_THROWS_WITH(manager.RegisterManifest(strict), Catch::Matchers::Contains("declared more than once"));
}

TEST_CASE("ExtensionLoader distributed declarations do not alter native execution", "[distributed][extension]") {
	DuckDB db(nullptr);
	db.LoadStaticExtension<NativeContractExtension>();

	Connection connection(db);
	auto result = connection.Query("SELECT distributed_native_identity(42)");
	REQUIRE_NO_FAIL(*result);
	REQUIRE(CHECK_COLUMN(result, 0, {42}));

	auto manifests = DistributedExtensionManager::Get(*db.instance).GetExtensions();
	DistributedExtensionManifest manifest;
	REQUIRE(DistributedExtensionManager::Get(*db.instance).TryGetExtension("native_contract", manifest));
	REQUIRE(manifest.extension_name == "native_contract");
}

TEST_CASE("ExtensionLoader publishes a distributed manifest only after successful load", "[distributed][extension]") {
	DuckDB db(nullptr);
	REQUIRE_THROWS_WITH(db.LoadStaticExtension<FailingContractExtension>(),
	                    Catch::Matchers::Contains("intentional distributed extension load failure"));

	DistributedExtensionManifest manifest;
	REQUIRE_FALSE(DistributedExtensionManager::Get(*db.instance).TryGetExtension("failing_contract", manifest));
}

TEST_CASE("ExtensionLoader distributed declaration validation has strong exception safety",
          "[distributed][extension]") {
	DuckDB db(nullptr);
	REQUIRE_NOTHROW(db.LoadStaticExtension<RetriedContractExtension>());

	DistributedExtensionManifest manifest;
	REQUIRE(DistributedExtensionManager::Get(*db.instance).TryGetExtension("loader_retry", manifest));
	REQUIRE(manifest.CanonicalIdentity() == "loader_retry@1{table_function:scan@1}");
}
