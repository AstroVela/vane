// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/main/distributed_extension_manager.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/set.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/distributed_write.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"

namespace duckdb {

namespace {

static bool IsValidExtensionName(const string &name) {
	if (name.empty()) {
		return false;
	}
	for (auto character : name) {
		if ((character >= 'a' && character <= 'z') || (character >= '0' && character <= '9') || character == '_') {
			continue;
		}
		return false;
	}
	return true;
}

static bool IsValidCapabilityName(const string &name) {
	if (name.empty()) {
		return false;
	}
	for (auto character : name) {
		if ((character >= 'a' && character <= 'z') || (character >= '0' && character <= '9') || character == '_' ||
		    character == '.' || character == '-') {
			continue;
		}
		return false;
	}
	return true;
}

static void ValidateCapabilityKind(DistributedExtensionCapabilityKind kind) {
	switch (kind) {
	case DistributedExtensionCapabilityKind::TABLE_FUNCTION:
	case DistributedExtensionCapabilityKind::WRITE_OPERATOR:
		return;
	default:
		throw InvalidInputException("Unknown distributed extension capability kind value: %u",
		                            static_cast<unsigned int>(kind));
	}
}

static void ValidateExtensionRegistration(const string &extension_name) {
	if (!IsValidExtensionName(extension_name)) {
		throw InvalidInputException(
		    "Distributed extension name must contain only lowercase ASCII letters, digits, and underscores: '%s'",
		    extension_name);
	}
}

static void ValidateCapabilityRegistration(const string &extension_name, const string &capability_name,
                                           idx_t protocol_version) {
	ValidateExtensionRegistration(extension_name);
	if (!IsValidCapabilityName(capability_name)) {
		throw InvalidInputException("Distributed extension capability name must contain only lowercase ASCII letters, "
		                            "digits, underscores, dots, and hyphens: '%s'",
		                            capability_name);
	}
	if (protocol_version == 0) {
		throw InvalidInputException(
		    "Distributed extension capability '%s.%s' protocol version must be greater than zero", extension_name,
		    capability_name);
	}
}

static string ContractListIdentity(const vector<string> &identities) {
	return "[" + StringUtil::Join(identities, ",") + "]";
}

} // namespace

static string DistributedExtensionCapabilityKindToString(DistributedExtensionCapabilityKind kind) {
	switch (kind) {
	case DistributedExtensionCapabilityKind::TABLE_FUNCTION:
		return "table_function";
	case DistributedExtensionCapabilityKind::WRITE_OPERATOR:
		return "write_operator";
	default:
		throw InternalException("Unknown distributed extension capability kind");
	}
}

string DistributedExtensionCapability::CanonicalIdentity() const {
	return StringUtil::Format("%s:%s@%llu", DistributedExtensionCapabilityKindToString(kind), name,
	                          static_cast<unsigned long long>(protocol_version));
}

bool DistributedExtensionCapability::operator==(const DistributedExtensionCapability &other) const {
	return kind == other.kind && name == other.name && protocol_version == other.protocol_version;
}

bool DistributedExtensionCapability::operator<(const DistributedExtensionCapability &other) const {
	if (kind != other.kind) {
		return static_cast<uint8_t>(kind) < static_cast<uint8_t>(other.kind);
	}
	if (name != other.name) {
		return name < other.name;
	}
	return protocol_version < other.protocol_version;
}

void DistributedExtensionCapabilityReference::Validate() const {
	ValidateCapabilityKind(capability.kind);
	ValidateCapabilityRegistration(extension_name, capability.name, capability.protocol_version);
}

void DistributedExtensionCapabilityReference::Serialize(Serializer &serializer) const {
	Validate();
	serializer.WriteProperty(1, "extension_name", extension_name);
	serializer.WriteProperty(2, "capability_kind", static_cast<uint8_t>(capability.kind));
	serializer.WriteProperty(3, "capability_name", capability.name);
	serializer.WriteProperty(4, "capability_protocol_version", capability.protocol_version);
}

DistributedExtensionCapabilityReference
DistributedExtensionCapabilityReference::Deserialize(Deserializer &deserializer) {
	DistributedExtensionCapabilityReference result;
	result.extension_name = deserializer.ReadProperty<string>(1, "extension_name");
	result.capability.kind =
	    static_cast<DistributedExtensionCapabilityKind>(deserializer.ReadProperty<uint8_t>(2, "capability_kind"));
	result.capability.name = deserializer.ReadProperty<string>(3, "capability_name");
	result.capability.protocol_version = deserializer.ReadProperty<idx_t>(4, "capability_protocol_version");
	result.Validate();
	return result;
}

string DistributedExtensionCapabilityReference::CanonicalIdentity() const {
	return StringUtil::Format("%s.%s", extension_name, capability.CanonicalIdentity());
}

bool DistributedExtensionCapabilityReference::operator==(const DistributedExtensionCapabilityReference &other) const {
	return extension_name == other.extension_name && capability == other.capability;
}

void DistributedPayloadCodec::Validate(const string &description) const {
	if (!IsValidCapabilityName(name) || version == 0) {
		throw InvalidInputException("%s has an invalid codec identity", description);
	}
}

void DistributedPayloadCodec::Serialize(Serializer &serializer) const {
	Validate("Distributed payload");
	serializer.WriteProperty(1, "name", name);
	serializer.WriteProperty(2, "version", version);
}

DistributedPayloadCodec DistributedPayloadCodec::Deserialize(Deserializer &deserializer) {
	DistributedPayloadCodec result;
	result.name = deserializer.ReadProperty<string>(1, "name");
	result.version = deserializer.ReadProperty<idx_t>(2, "version");
	result.Validate("Distributed payload");
	return result;
}

string DistributedPayloadCodec::CanonicalIdentity() const {
	return StringUtil::Format("%s@%llu", name, static_cast<unsigned long long>(version));
}

bool DistributedPayloadCodec::operator==(const DistributedPayloadCodec &other) const {
	return name == other.name && version == other.version;
}

bool DistributedPayloadCodec::operator!=(const DistributedPayloadCodec &other) const {
	return !(*this == other);
}

bool DistributedExtensionCapabilityReference::operator!=(const DistributedExtensionCapabilityReference &other) const {
	return !(*this == other);
}

string DistributedExtensionManifest::CanonicalIdentity() const {
	auto sorted_capabilities = capabilities;
	std::sort(sorted_capabilities.begin(), sorted_capabilities.end());
	vector<string> identities;
	identities.reserve(sorted_capabilities.size());
	for (const auto &capability : sorted_capabilities) {
		identities.push_back(capability.CanonicalIdentity());
	}
	return StringUtil::Format("%s{%s}", extension_name, StringUtil::Join(identities, ","));
}

DistributedExtensionManager::DistributedExtensionManager(DatabaseInstance &) {
}

void DistributedExtensionManager::ValidateManifest(const DistributedExtensionManifest &manifest) {
	ValidateExtensionRegistration(manifest.extension_name);
	if (manifest.capabilities.empty()) {
		throw InvalidInputException("Distributed extension '%s' must contain a concrete capability",
		                            manifest.extension_name);
	}
	set<pair<DistributedExtensionCapabilityKind, string>> capability_identities;
	for (const auto &capability : manifest.capabilities) {
		ValidateCapabilityKind(capability.kind);
		ValidateCapabilityRegistration(manifest.extension_name, capability.name, capability.protocol_version);
		auto identity = make_pair(capability.kind, capability.name);
		if (!capability_identities.insert(identity).second) {
			throw InvalidInputException("Distributed extension capability '%s.%s' is declared more than once",
			                            manifest.extension_name, capability.name);
		}
	}
}

void DistributedExtensionManager::RegisterExtension(
    const DistributedExtensionManifest &manifest_p,
    vector<shared_ptr<const DistributedWriteOperatorExtension>> write_operators_p) {
	ValidateManifest(manifest_p);
	auto manifest = manifest_p;
	std::sort(manifest.capabilities.begin(), manifest.capabilities.end());
	map<string, shared_ptr<const DistributedWriteOperatorExtension>> new_write_operators;
	for (auto &write_operator : write_operators_p) {
		if (!write_operator) {
			throw InvalidInputException("Distributed extension '%s' contains a null write operator",
			                            manifest.extension_name);
		}
		DistributedExtensionCapabilityReference reference;
		reference.extension_name = manifest.extension_name;
		reference.capability = {DistributedExtensionCapabilityKind::WRITE_OPERATOR, write_operator->name,
		                        write_operator->protocol_version};
		reference.Validate();
		write_operator->Validate(reference.CanonicalIdentity());
		if (std::find(manifest.capabilities.begin(), manifest.capabilities.end(), reference.capability) ==
		    manifest.capabilities.end()) {
			throw InvalidInputException("Distributed write operator '%s' has no matching concrete capability",
			                            reference.CanonicalIdentity());
		}
		if (!new_write_operators.emplace(reference.CanonicalIdentity(), std::move(write_operator)).second) {
			throw InvalidInputException("Distributed write operator '%s' is declared more than once",
			                            reference.CanonicalIdentity());
		}
	}
	for (const auto &capability : manifest.capabilities) {
		if (capability.kind != DistributedExtensionCapabilityKind::WRITE_OPERATOR) {
			continue;
		}
		DistributedExtensionCapabilityReference reference {manifest.extension_name, capability};
		if (new_write_operators.find(reference.CanonicalIdentity()) == new_write_operators.end()) {
			throw InvalidInputException("Distributed write capability '%s' has no registered implementation",
			                            reference.CanonicalIdentity());
		}
	}

	lock_guard<mutex> guard(lock);
	if (extensions.find(manifest.extension_name) != extensions.end()) {
		throw InvalidInputException("Distributed extension '%s' is already registered", manifest.extension_name);
	}
	auto next_extensions = extensions;
	auto next_write_operators = write_operators;
	next_extensions.emplace(manifest.extension_name, std::move(manifest));
	for (auto &entry : new_write_operators) {
		if (!next_write_operators.emplace(entry.first, std::move(entry.second)).second) {
			throw InvalidInputException("Distributed write operator '%s' is already registered", entry.first);
		}
	}
	extensions.swap(next_extensions);
	write_operators.swap(next_write_operators);
}

vector<string> DistributedExtensionManager::GetContractIdentities() const {
	lock_guard<mutex> guard(lock);
	vector<string> result;
	result.reserve(extensions.size());
	for (const auto &extension : extensions) {
		result.push_back(extension.second.CanonicalIdentity());
	}
	return result;
}

void DistributedExtensionManager::RequireCapability(const DistributedExtensionCapabilityReference &capability) const {
	capability.Validate();
	lock_guard<mutex> guard(lock);
	auto extension = extensions.find(capability.extension_name);
	if (extension == extensions.end()) {
		throw InvalidInputException("Distributed extension '%s' is not registered", capability.extension_name);
	}
	for (const auto &registered : extension->second.capabilities) {
		if (registered.kind == capability.capability.kind && registered.name == capability.capability.name) {
			if (registered.protocol_version != capability.capability.protocol_version) {
				throw InvalidInputException(
				    "Distributed extension capability '%s.%s' protocol mismatch: required %llu, registered %llu",
				    capability.extension_name, capability.capability.name,
				    static_cast<unsigned long long>(capability.capability.protocol_version),
				    static_cast<unsigned long long>(registered.protocol_version));
			}
			return;
		}
	}
	throw InvalidInputException("Distributed extension capability '%s.%s' is not registered", capability.extension_name,
	                            capability.capability.name);
}

shared_ptr<const DistributedWriteOperatorExtension>
DistributedExtensionManager::GetWriteOperator(const DistributedExtensionCapabilityReference &capability) const {
	RequireCapability(capability);
	if (capability.capability.kind != DistributedExtensionCapabilityKind::WRITE_OPERATOR) {
		throw InvalidInputException("Distributed write requires a write-operator capability: %s",
		                            capability.CanonicalIdentity());
	}
	const auto identity = capability.CanonicalIdentity();
	lock_guard<mutex> guard(lock);
	auto entry = write_operators.find(identity);
	if (entry == write_operators.end()) {
		throw InvalidInputException("Distributed write operator '%s' is not registered", identity);
	}
	return entry->second;
}

shared_ptr<const DistributedWriteOperatorExtension>
DistributedExtensionManager::GetWriteOperator(const string &extension_name, const string &operator_name) const {
	ValidateCapabilityRegistration(extension_name, operator_name, 1);
	lock_guard<mutex> guard(lock);
	auto extension = extensions.find(extension_name);
	if (extension == extensions.end()) {
		throw InvalidInputException("Distributed extension '%s' is not registered", extension_name);
	}
	for (const auto &capability : extension->second.capabilities) {
		if (capability.kind != DistributedExtensionCapabilityKind::WRITE_OPERATOR || capability.name != operator_name) {
			continue;
		}
		DistributedExtensionCapabilityReference reference {extension_name, capability};
		auto entry = write_operators.find(reference.CanonicalIdentity());
		if (entry == write_operators.end()) {
			throw InternalException("Distributed write operator '%s' has no implementation",
			                        reference.CanonicalIdentity());
		}
		return entry->second;
	}
	throw InvalidInputException("Distributed write operator '%s.%s' is not registered", extension_name, operator_name);
}

void DistributedExtensionManager::ValidateExact(const vector<string> &expected_contract_identities) const {
	auto expected = expected_contract_identities;
	set<string> unique_identities;
	for (const auto &identity : expected) {
		if (identity.empty() || !unique_identities.insert(identity).second) {
			throw InvalidInputException("Distributed extension contract identities must be non-empty and unique");
		}
	}
	std::sort(expected.begin(), expected.end());
	auto actual = GetContractIdentities();
	if (actual == expected) {
		return;
	}
	throw InvalidInputException("Distributed extension contracts differ between coordinator and worker: expected %s, "
	                            "worker registered %s",
	                            ContractListIdentity(expected), ContractListIdentity(actual));
}

DistributedExtensionManager &DistributedExtensionManager::Get(DatabaseInstance &db) {
	return db.GetDistributedExtensionManager();
}

DistributedExtensionManager &DistributedExtensionManager::Get(ClientContext &context) {
	return Get(DatabaseInstance::GetDatabase(context));
}

} // namespace duckdb
