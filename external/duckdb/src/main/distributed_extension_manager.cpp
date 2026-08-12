// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/main/distributed_extension_manager.hpp"

#include "duckdb/common/exception.hpp"
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
	case DistributedExtensionCapabilityKind::AGGREGATE_FUNCTION:
	case DistributedExtensionCapabilityKind::COPY_FUNCTION:
	case DistributedExtensionCapabilityKind::OPERATOR:
	case DistributedExtensionCapabilityKind::STORAGE:
	case DistributedExtensionCapabilityKind::CONTEXT:
		return;
	default:
		throw InvalidInputException("Unknown distributed extension capability kind value: %u",
		                            static_cast<unsigned int>(kind));
	}
}

static void ValidateExtensionRegistration(const string &extension_name, idx_t protocol_version) {
	if (!IsValidExtensionName(extension_name)) {
		throw InvalidInputException(
		    "Distributed extension name must contain only lowercase ASCII letters, digits, and underscores: '%s'",
		    extension_name);
	}
	if (protocol_version == 0) {
		throw InvalidInputException("Distributed extension '%s' protocol version must be greater than zero",
		                            extension_name);
	}
}

static void ValidateCapabilityRegistration(const string &extension_name, const string &capability_name,
                                           idx_t protocol_version) {
	ValidateExtensionRegistration(extension_name, 1);
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

static string ManifestListIdentity(const vector<DistributedExtensionManifest> &manifests) {
	vector<string> entries;
	entries.reserve(manifests.size());
	for (const auto &manifest : manifests) {
		entries.push_back(manifest.CanonicalIdentity());
	}
	return "[" + StringUtil::Join(entries, ",") + "]";
}

} // namespace

string DistributedExtensionCapabilityKindToString(DistributedExtensionCapabilityKind kind) {
	switch (kind) {
	case DistributedExtensionCapabilityKind::TABLE_FUNCTION:
		return "table_function";
	case DistributedExtensionCapabilityKind::AGGREGATE_FUNCTION:
		return "aggregate_function";
	case DistributedExtensionCapabilityKind::COPY_FUNCTION:
		return "copy_function";
	case DistributedExtensionCapabilityKind::OPERATOR:
		return "operator";
	case DistributedExtensionCapabilityKind::STORAGE:
		return "storage";
	case DistributedExtensionCapabilityKind::CONTEXT:
		return "context";
	default:
		throw InternalException("Unknown distributed extension capability kind");
	}
}

DistributedExtensionCapabilityKind DistributedExtensionCapabilityKindFromString(const string &value) {
	if (value == "table_function") {
		return DistributedExtensionCapabilityKind::TABLE_FUNCTION;
	}
	if (value == "aggregate_function") {
		return DistributedExtensionCapabilityKind::AGGREGATE_FUNCTION;
	}
	if (value == "copy_function") {
		return DistributedExtensionCapabilityKind::COPY_FUNCTION;
	}
	if (value == "operator") {
		return DistributedExtensionCapabilityKind::OPERATOR;
	}
	if (value == "storage") {
		return DistributedExtensionCapabilityKind::STORAGE;
	}
	if (value == "context") {
		return DistributedExtensionCapabilityKind::CONTEXT;
	}
	throw InvalidInputException("Unknown distributed extension capability kind: '%s'", value);
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

string DistributedExtensionCapabilityReference::CanonicalIdentity() const {
	return StringUtil::Format("%s@%llu.%s", extension_name, static_cast<unsigned long long>(extension_protocol_version),
	                          capability.CanonicalIdentity());
}

bool DistributedExtensionCapabilityReference::operator==(const DistributedExtensionCapabilityReference &other) const {
	return extension_name == other.extension_name && extension_protocol_version == other.extension_protocol_version &&
	       capability == other.capability;
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
	return StringUtil::Format("%s@%llu{%s}", extension_name, static_cast<unsigned long long>(protocol_version),
	                          StringUtil::Join(identities, ","));
}

bool DistributedExtensionManifest::operator==(const DistributedExtensionManifest &other) const {
	if (extension_name != other.extension_name || protocol_version != other.protocol_version ||
	    capabilities.size() != other.capabilities.size()) {
		return false;
	}
	auto left = capabilities;
	auto right = other.capabilities;
	std::sort(left.begin(), left.end());
	std::sort(right.begin(), right.end());
	return left == right;
}

bool DistributedExtensionManifest::operator<(const DistributedExtensionManifest &other) const {
	return extension_name < other.extension_name;
}

DistributedExtensionManager::DistributedExtensionManager(DatabaseInstance &) {
}

void DistributedExtensionManager::ValidateManifest(const DistributedExtensionManifest &manifest) {
	ValidateExtensionRegistration(manifest.extension_name, manifest.protocol_version);
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

void DistributedExtensionManager::RegisterManifest(const DistributedExtensionManifest &manifest_p) {
	ValidateManifest(manifest_p);
	auto manifest = manifest_p;
	std::sort(manifest.capabilities.begin(), manifest.capabilities.end());
	lock_guard<mutex> guard(lock);
	if (extensions.find(manifest.extension_name) != extensions.end()) {
		throw InvalidInputException("Distributed extension '%s' is already registered", manifest.extension_name);
	}
	auto extension_name = manifest.extension_name;
	extensions.emplace(std::move(extension_name), std::move(manifest));
}

bool DistributedExtensionManager::TryGetExtension(const string &extension_name,
                                                  DistributedExtensionManifest &result) const {
	lock_guard<mutex> guard(lock);
	auto extension = extensions.find(extension_name);
	if (extension == extensions.end()) {
		return false;
	}
	result = extension->second;
	std::sort(result.capabilities.begin(), result.capabilities.end());
	return true;
}

vector<DistributedExtensionManifest> DistributedExtensionManager::GetExtensions() const {
	lock_guard<mutex> guard(lock);
	vector<DistributedExtensionManifest> result;
	result.reserve(extensions.size());
	for (const auto &extension : extensions) {
		result.push_back(extension.second);
		std::sort(result.back().capabilities.begin(), result.back().capabilities.end());
	}
	return result;
}

void DistributedExtensionManager::RequireCapability(const string &extension_name,
                                                    DistributedExtensionCapabilityKind kind,
                                                    const string &capability_name, idx_t protocol_version) const {
	ValidateCapabilityKind(kind);
	ValidateCapabilityRegistration(extension_name, capability_name, protocol_version);
	lock_guard<mutex> guard(lock);
	auto extension = extensions.find(extension_name);
	if (extension == extensions.end()) {
		throw InvalidInputException("Distributed extension '%s' is not registered", extension_name);
	}
	for (const auto &capability : extension->second.capabilities) {
		if (capability.kind != kind || capability.name != capability_name) {
			continue;
		}
		if (capability.protocol_version != protocol_version) {
			throw InvalidInputException(
			    "Distributed extension capability '%s.%s' protocol mismatch: required %llu, registered %llu",
			    extension_name, capability_name, static_cast<unsigned long long>(protocol_version),
			    static_cast<unsigned long long>(capability.protocol_version));
		}
		return;
	}
	throw InvalidInputException("Distributed extension capability '%s.%s' is not registered", extension_name,
	                            capability_name);
}

void DistributedExtensionManager::RequireCapability(const DistributedExtensionCapabilityReference &capability) const {
	ValidateExtensionRegistration(capability.extension_name, capability.extension_protocol_version);
	DistributedExtensionManifest manifest;
	if (!TryGetExtension(capability.extension_name, manifest)) {
		throw InvalidInputException("Distributed extension '%s' is not registered", capability.extension_name);
	}
	if (manifest.protocol_version != capability.extension_protocol_version) {
		throw InvalidInputException(
		    "Distributed extension '%s' manifest protocol mismatch: required %llu, registered %llu",
		    capability.extension_name, static_cast<unsigned long long>(capability.extension_protocol_version),
		    static_cast<unsigned long long>(manifest.protocol_version));
	}
	RequireCapability(capability.extension_name, capability.capability.kind, capability.capability.name,
	                  capability.capability.protocol_version);
}

void DistributedExtensionManager::RegisterWriteCallbacks(const DistributedExtensionCapabilityReference &capability,
                                                         DistributedExtensionWriteCallbacks callbacks) {
	RequireCapability(capability);
	if (capability.capability.kind != DistributedExtensionCapabilityKind::OPERATOR) {
		throw InvalidInputException("Distributed write callbacks require an operator capability: %s",
		                            capability.CanonicalIdentity());
	}
	const auto identity = capability.CanonicalIdentity();
	callbacks.Validate(identity);
	auto registered_callbacks = make_shared_ptr<DistributedExtensionWriteCallbacks>(std::move(callbacks));
	lock_guard<mutex> guard(lock);
	if (!write_callbacks.emplace(identity, std::move(registered_callbacks)).second) {
		throw InvalidInputException("Distributed write callbacks for '%s' are already registered", identity);
	}
}

shared_ptr<const DistributedExtensionWriteCallbacks>
DistributedExtensionManager::GetWriteCallbacks(const DistributedExtensionCapabilityReference &capability) const {
	RequireCapability(capability);
	const auto identity = capability.CanonicalIdentity();
	lock_guard<mutex> guard(lock);
	auto entry = write_callbacks.find(identity);
	if (entry == write_callbacks.end()) {
		throw InvalidInputException("Distributed write callbacks for '%s' are not registered", identity);
	}
	return entry->second;
}

void DistributedExtensionManager::ValidateExact(const vector<DistributedExtensionManifest> &expected_p) const {
	auto expected = expected_p;
	set<string> extension_names;
	for (const auto &manifest : expected) {
		ValidateManifest(manifest);
		if (!extension_names.insert(manifest.extension_name).second) {
			throw InvalidInputException(
			    "Distributed extension '%s' appears more than once in the expected manifest set",
			    manifest.extension_name);
		}
	}
	std::sort(expected.begin(), expected.end());
	auto actual = GetExtensions();
	if (actual == expected) {
		return;
	}
	throw InvalidInputException("Distributed extension manifests differ between coordinator and worker: expected %s, "
	                            "worker registered %s",
	                            ManifestListIdentity(expected), ManifestListIdentity(actual));
}

DistributedExtensionManager &DistributedExtensionManager::Get(DatabaseInstance &db) {
	return db.GetDistributedExtensionManager();
}

DistributedExtensionManager &DistributedExtensionManager::Get(ClientContext &context) {
	return Get(DatabaseInstance::GetDatabase(context));
}

} // namespace duckdb
