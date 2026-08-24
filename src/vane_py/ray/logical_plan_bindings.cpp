// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Included by ray_module.cpp inside namespace duckdb.

struct PyPhysicalPlanWrapper;

static constexpr char LOGICAL_PLAN_ENVELOPE_MAGIC[] = {'V', 'A', 'N', 'E', 'P', 'L', 'A', 'N'};
static constexpr idx_t LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE = sizeof(LOGICAL_PLAN_ENVELOPE_MAGIC);
static constexpr uint32_t LOGICAL_PLAN_PROTOCOL_VERSION = 1;
static constexpr uint32_t LOGICAL_PLAN_MAX_SOURCE_ID_SIZE = 4096;

static constexpr uint32_t SCOPED_SECRET_REF_VERSION = 1;
static constexpr uint8_t SCOPED_SECRET_CAPABILITY_READ = 1 << 0;
static constexpr uint8_t SCOPED_SECRET_CAPABILITY_WRITE = 1 << 1;
static constexpr uint8_t SCOPED_SECRET_CAPABILITY_MASK = SCOPED_SECRET_CAPABILITY_READ | SCOPED_SECRET_CAPABILITY_WRITE;
static constexpr idx_t SCOPED_SECRET_MAX_REFERENCES = 4096;
static constexpr idx_t SCOPED_SECRET_MAX_REFERENCE_BYTES = 16 * 1024 * 1024;
static constexpr idx_t SCOPED_SECRET_MAX_USES = 100000;
static constexpr idx_t SCOPED_SECRET_MAX_USE_BYTES = 16 * 1024 * 1024;
static constexpr idx_t SCOPED_SECRET_MAX_IDENTITY_SIZE = 4096;
static constexpr idx_t SCOPED_SECRET_MAX_SCOPE_SIZE = 65536;
static constexpr idx_t SCOPED_SECRET_MAX_METADATA_STORAGES = 4096;
static constexpr idx_t SCOPED_SECRET_MAX_METADATA_ENTRIES = 4096;
static constexpr idx_t SCOPED_SECRET_MAX_METADATA_SCOPES = 4096;
static constexpr idx_t SCOPED_SECRET_MAX_METADATA_BYTES = 16 * 1024 * 1024;
static constexpr idx_t SCOPED_SECRET_MAX_GENERATED_BOUNDARIES = 4096;
static constexpr idx_t SCOPED_SECRET_MAX_GENERATED_BOUNDARY_BYTES = 16 * 1024 * 1024;
static constexpr char SCOPED_SECRET_GENERATED_PROBE_SUFFIX[] = "__vane_generated_copy_scope_probe__";
static constexpr char SCOPED_SECRET_COPY_COMMIT_SUFFIX[] = ".duckdb_commit";

struct ScopedSecretRef {
	uint32_t version = SCOPED_SECRET_REF_VERSION;
	string reference_id;
	string owner_query_id;
	string owner_session_id;
	string secret_type;
	string provider;
	string normalized_scope;
	uint8_t capabilities = 0;
};

struct ScopedSecretUse {
	string uri;
	uint8_t capabilities = 0;
	// Ray COPY rewrites every remote output to generated task-owned paths.
	// This marker stays source-only and proves that every secret scope beneath
	// the generated namespace selects the same opaque identity.
	bool covers_generated_copy_namespace = false;
};

// This source-only binding gives the opaque reference a local identity for
// freshness checks. Secret names, storage names, and resource URIs are never
// copied into logical/physical pickle state.
struct ScopedSecretBinding {
	string reference_id;
	string storage_name;
	string secret_name;
	string secret_type;
	string provider;
	string normalized_scope;
	vector<ScopedSecretUse> uses;
};

struct ScopedSecretDiscovery {
	vector<ScopedSecretRef> references;
	vector<ScopedSecretBinding> source_bindings;
	vector<ScopedSecretUse> source_unmatched_uses;
};

struct ScopedSecretSelection {
	string storage_name;
	string secret_name;
	string secret_type;
	string provider;
	string normalized_scope;
};

class NonPrefixScopedSecretForTest : public BaseSecret {
public:
	explicit NonPrefixScopedSecretForTest(const string &name)
	    : BaseSecret({"s3://non-prefix-test-scope/"}, "s3", "config", name) {
	}

	int64_t MatchScore(const string &path) const override {
		if (!StringUtil::EndsWith(path, ".parquet")) {
			return NumericLimits<int64_t>::Minimum();
		}
		return NumericCast<int64_t>(path.size());
	}

	unique_ptr<const BaseSecret> Clone() const override {
		return make_uniq<NonPrefixScopedSecretForTest>(*this);
	}
};

class NonStandardEmptySecretStorageForTest : public SecretStorage {
public:
	NonStandardEmptySecretStorageForTest() : SecretStorage("vane_nonstandard_empty_test", 1000) {
		unique_ptr<const BaseSecret> secret =
		    make_uniq<KeyValueSecret>(vector<string> {"s3://nonstandard-empty-storage-scope/"}, "s3", "config",
		                              "nonstandard_empty_storage_secret");
		synthetic_entry = make_uniq<SecretEntry>(std::move(secret));
		synthetic_entry->persist_type = SecretPersistType::TEMPORARY;
		synthetic_entry->storage_mode = GetName();
	}

	unique_ptr<SecretEntry> StoreSecret(unique_ptr<const BaseSecret>, OnCreateConflict,
	                                    optional_ptr<CatalogTransaction>) override {
		throw NotImplementedException("Test-only synthetic secret storage does not support writes");
	}

	vector<SecretEntry> AllSecrets(optional_ptr<CatalogTransaction>) override {
		return {};
	}

	bool ScanSecretMetadata(const secret_metadata_callback_t &, optional_ptr<CatalogTransaction>) override {
		return true;
	}

	void DropSecretByName(const string &, OnEntryNotFound, optional_ptr<CatalogTransaction>) override {
		throw NotImplementedException("Test-only synthetic secret storage does not support drops");
	}

	SecretMatch LookupSecret(const string &path, const string &type, optional_ptr<CatalogTransaction>) override {
		if (StringUtil::CIEquals(type, "s3") && StringUtil::EndsWith(path, ".parquet")) {
			return SecretMatch(*synthetic_entry, NumericLimits<int64_t>::Maximum() / 2);
		}
		return SecretMatch();
	}

	unique_ptr<SecretEntry> GetSecretByName(const string &, optional_ptr<CatalogTransaction>) override {
		return nullptr;
	}

private:
	unique_ptr<SecretEntry> synthetic_entry;
};

static void AppendScopedSecretIdentityField(string &identity_material, const string &value) {
	auto value_size = static_cast<uint64_t>(value.size());
	for (idx_t byte_idx = 0; byte_idx < sizeof(value_size); byte_idx++) {
		identity_material.push_back(static_cast<char>((value_size >> (byte_idx * 8)) & 0xff));
	}
	identity_material.append(value);
}

static string CreateScopedSecretSelectionKey(const ScopedSecretSelection &selection) {
	string key;
	AppendScopedSecretIdentityField(key, selection.storage_name);
	AppendScopedSecretIdentityField(key, selection.secret_name);
	AppendScopedSecretIdentityField(key, selection.secret_type);
	AppendScopedSecretIdentityField(key, selection.provider);
	AppendScopedSecretIdentityField(key, selection.normalized_scope);
	return key;
}

static string CreateScopedSecretScopeKey(const ScopedSecretSelection &selection) {
	string key;
	AppendScopedSecretIdentityField(key, selection.secret_type);
	AppendScopedSecretIdentityField(key, selection.provider);
	AppendScopedSecretIdentityField(key, selection.normalized_scope);
	return key;
}

static string CreateScopedSecretReferenceId(const ScopedSecretSelection &selection, const string &owner_query_id,
                                            const string &owner_session_id) {
	string identity_material;
	AppendScopedSecretIdentityField(identity_material, "vane.scoped-secret-ref.v1");
	AppendScopedSecretIdentityField(identity_material, owner_session_id);
	AppendScopedSecretIdentityField(identity_material, owner_query_id);
	AppendScopedSecretIdentityField(identity_material, selection.storage_name);
	AppendScopedSecretIdentityField(identity_material, selection.secret_name);
	AppendScopedSecretIdentityField(identity_material, selection.secret_type);
	AppendScopedSecretIdentityField(identity_material, selection.provider);
	AppendScopedSecretIdentityField(identity_material, selection.normalized_scope);

	char digest[duckdb_mbedtls::MbedTlsWrapper::SHA256_HASH_LENGTH_BYTES];
	duckdb_mbedtls::MbedTlsWrapper::ComputeSha256Hash(identity_material.data(), identity_material.size(), digest);
	data_t identity[16];
	memcpy(identity, digest, sizeof(identity));
	// RFC 9562 UUIDv8 and variant bits. SHA-256 keeps the identifier
	// deterministic, while storage/name inputs remain opaque in plan state.
	identity[6] = static_cast<data_t>((identity[6] & 0x0f) | 0x80);
	identity[8] = static_cast<data_t>((identity[8] & 0x3f) | 0x80);
	return UUID::ToString(BaseUUID::FromBlob(identity));
}

static bool IsSupportedScopedSecretType(const string &secret_type) {
	return secret_type == "s3" || secret_type == "r2" || secret_type == "gcs" || secret_type == "aws";
}

static bool IsScopedSecretSelectionType(const string &secret_type, bool include_http) {
	return StringUtil::CIEquals(secret_type, "s3") || StringUtil::CIEquals(secret_type, "r2") ||
	       StringUtil::CIEquals(secret_type, "gcs") || StringUtil::CIEquals(secret_type, "aws") ||
	       (include_http && StringUtil::CIEquals(secret_type, "http"));
}

static bool IsSupportedScopedSecretProvider(const string &secret_type, const string &provider) {
	return IsSupportedScopedSecretType(secret_type) && (provider == "config" || provider == "credential_chain");
}

static bool ScopedSecretRefLess(const ScopedSecretRef &left, const ScopedSecretRef &right) {
	return std::tie(left.secret_type, left.provider, left.normalized_scope, left.reference_id, left.capabilities) <
	       std::tie(right.secret_type, right.provider, right.normalized_scope, right.reference_id, right.capabilities);
}

static bool ScopedSecretRefsEqual(const vector<ScopedSecretRef> &left, const vector<ScopedSecretRef> &right) {
	if (left.size() != right.size()) {
		return false;
	}
	for (idx_t ref_idx = 0; ref_idx < left.size(); ref_idx++) {
		auto &lhs = left[ref_idx];
		auto &rhs = right[ref_idx];
		if (lhs.version != rhs.version || lhs.reference_id != rhs.reference_id ||
		    lhs.owner_query_id != rhs.owner_query_id || lhs.owner_session_id != rhs.owner_session_id ||
		    lhs.secret_type != rhs.secret_type || lhs.provider != rhs.provider ||
		    lhs.normalized_scope != rhs.normalized_scope || lhs.capabilities != rhs.capabilities) {
			return false;
		}
	}
	return true;
}

static bool ContainsNul(const string &value) {
	return value.find('\0') != string::npos;
}

static void ValidateScopedSecretIdentityField(const string &value, const char *field_name, idx_t max_size,
                                              bool allow_empty = false) {
	if ((!allow_empty && value.empty()) || value.size() > max_size || ContainsNul(value)) {
		throw InvalidInputException("Scoped secret reference contains an invalid %s", field_name);
	}
}

static void ConsumeScopedSecretReferenceBytes(idx_t size, idx_t &total_bytes, const char *boundary) {
	if (total_bytes > SCOPED_SECRET_MAX_REFERENCE_BYTES || size > SCOPED_SECRET_MAX_REFERENCE_BYTES - total_bytes) {
		throw InvalidInputException("Scoped secret reference state at %s exceeds the maximum serialized byte size",
		                            boundary);
	}
	total_bytes += size;
}

static void ConsumeScopedSecretReferenceBytes(const ScopedSecretRef &reference, idx_t &total_bytes,
                                              const char *boundary) {
	ConsumeScopedSecretReferenceBytes(reference.reference_id.size(), total_bytes, boundary);
	ConsumeScopedSecretReferenceBytes(reference.owner_query_id.size(), total_bytes, boundary);
	ConsumeScopedSecretReferenceBytes(reference.owner_session_id.size(), total_bytes, boundary);
	ConsumeScopedSecretReferenceBytes(reference.secret_type.size(), total_bytes, boundary);
	ConsumeScopedSecretReferenceBytes(reference.provider.size(), total_bytes, boundary);
	ConsumeScopedSecretReferenceBytes(reference.normalized_scope.size(), total_bytes, boundary);
}

static void ValidateScopedSecretSelection(const ScopedSecretSelection &selection) {
	ValidateScopedSecretIdentityField(selection.storage_name, "selected storage name", SCOPED_SECRET_MAX_IDENTITY_SIZE);
	ValidateScopedSecretIdentityField(selection.secret_name, "selected secret name", SCOPED_SECRET_MAX_IDENTITY_SIZE);
	ValidateScopedSecretIdentityField(selection.secret_type, "selected secret type", SCOPED_SECRET_MAX_IDENTITY_SIZE);
	ValidateScopedSecretIdentityField(selection.provider, "selected provider", SCOPED_SECRET_MAX_IDENTITY_SIZE);
	ValidateScopedSecretIdentityField(selection.normalized_scope, "selected normalized scope",
	                                  SCOPED_SECRET_MAX_SCOPE_SIZE, true);
}

static idx_t PreflightScopedSecretReferenceBytes(const ScopedSecretSelection &selection, const string &owner_query_id,
                                                 const string &owner_session_id, idx_t total_bytes,
                                                 const char *boundary) {
	ConsumeScopedSecretReferenceBytes(BaseUUID::STRING_SIZE, total_bytes, boundary);
	ConsumeScopedSecretReferenceBytes(owner_query_id.size(), total_bytes, boundary);
	ConsumeScopedSecretReferenceBytes(owner_session_id.size(), total_bytes, boundary);
	ConsumeScopedSecretReferenceBytes(selection.secret_type.size(), total_bytes, boundary);
	ConsumeScopedSecretReferenceBytes(selection.provider.size(), total_bytes, boundary);
	ConsumeScopedSecretReferenceBytes(selection.normalized_scope.size(), total_bytes, boundary);
	return total_bytes;
}

static void ValidateScopedSecretRefs(const vector<ScopedSecretRef> &references, const string &expected_owner_query_id,
                                     const string &expected_session_id, const char *boundary) {
	if (references.size() > SCOPED_SECRET_MAX_REFERENCES) {
		throw InvalidInputException("Scoped secret reference state at %s exceeds the maximum reference count",
		                            boundary);
	}
	if (references.empty()) {
		return;
	}
	ValidateScopedSecretIdentityField(expected_owner_query_id, "expected owner query ID",
	                                  SCOPED_SECRET_MAX_IDENTITY_SIZE);
	ValidateScopedSecretIdentityField(expected_session_id, "expected session ID", SCOPED_SECRET_MAX_IDENTITY_SIZE);

	std::unordered_set<string> reference_ids;
	idx_t total_reference_bytes = 0;
	for (idx_t ref_idx = 0; ref_idx < references.size(); ref_idx++) {
		auto &reference = references[ref_idx];
		if (reference.version != SCOPED_SECRET_REF_VERSION) {
			throw InvalidInputException("Unsupported scoped secret reference version %u at %s (expected %u)",
			                            reference.version, boundary, SCOPED_SECRET_REF_VERSION);
		}
		ValidateScopedSecretIdentityField(reference.reference_id, "opaque ID", BaseUUID::STRING_SIZE);
		hugeint_t parsed_reference_id;
		if (!UUID::FromString(reference.reference_id, parsed_reference_id, true) ||
		    UUID::ToString(parsed_reference_id) != reference.reference_id || reference.reference_id[14] != '8' ||
		    (reference.reference_id[19] != '8' && reference.reference_id[19] != '9' &&
		     reference.reference_id[19] != 'a' && reference.reference_id[19] != 'b')) {
			throw InvalidInputException("Scoped secret reference at %s has an invalid opaque ID", boundary);
		}
		if (!reference_ids.insert(reference.reference_id).second) {
			throw InvalidInputException("Scoped secret reference state at %s contains a duplicate opaque ID", boundary);
		}

		ValidateScopedSecretIdentityField(reference.owner_query_id, "owner query ID", SCOPED_SECRET_MAX_IDENTITY_SIZE);
		if (reference.owner_query_id != expected_owner_query_id) {
			throw InvalidInputException(
			    "Scoped secret reference '%s' is stale or belongs to a different query (expected owner '%s')",
			    reference.reference_id, expected_owner_query_id);
		}
		ValidateScopedSecretIdentityField(reference.owner_session_id, "owner session ID",
		                                  SCOPED_SECRET_MAX_IDENTITY_SIZE);
		if (reference.owner_session_id != expected_session_id) {
			throw InvalidInputException("Scoped secret reference '%s' belongs to a different Vane session",
			                            reference.reference_id);
		}
		ValidateScopedSecretIdentityField(reference.secret_type, "secret type", SCOPED_SECRET_MAX_IDENTITY_SIZE);
		ValidateScopedSecretIdentityField(reference.provider, "provider", SCOPED_SECRET_MAX_IDENTITY_SIZE);
		if (!IsSupportedScopedSecretProvider(reference.secret_type, reference.provider)) {
			throw InvalidInputException(
			    "Scoped secret reference '%s' uses unsupported type/provider '%s/%s'; supported object-storage "
			    "providers are config and credential_chain",
			    reference.reference_id, reference.secret_type, reference.provider);
		}
		ValidateScopedSecretIdentityField(reference.normalized_scope, "normalized scope", SCOPED_SECRET_MAX_SCOPE_SIZE,
		                                  true);
		ConsumeScopedSecretReferenceBytes(reference, total_reference_bytes, boundary);
		if (reference.capabilities == 0 || (reference.capabilities & ~SCOPED_SECRET_CAPABILITY_MASK) != 0) {
			throw InvalidInputException("Scoped secret reference '%s' has invalid capabilities at %s",
			                            reference.reference_id, boundary);
		}

		if (ref_idx > 0) {
			auto &prior = references[ref_idx - 1];
			if (!ScopedSecretRefLess(prior, reference)) {
				throw InvalidInputException("Scoped secret reference state at %s is not in canonical order", boundary);
			}
			if (prior.secret_type == reference.secret_type && prior.provider == reference.provider &&
			    prior.normalized_scope == reference.normalized_scope) {
				throw InvalidInputException("Scoped secret reference state at %s is ambiguous for type/provider/scope",
				                            boundary);
			}
		}
	}
}

static py::tuple ScopedSecretRefsToPython(const vector<ScopedSecretRef> &references) {
	py::tuple result(references.size());
	for (idx_t ref_idx = 0; ref_idx < references.size(); ref_idx++) {
		auto &reference = references[ref_idx];
		result[ref_idx] = py::make_tuple(reference.version, reference.reference_id, reference.owner_query_id,
		                                 reference.owner_session_id, reference.secret_type, reference.provider,
		                                 reference.normalized_scope, reference.capabilities);
	}
	return result;
}

static py::list DescribeScopedSecretRefs(const vector<ScopedSecretRef> &references) {
	py::list result;
	for (auto &reference : references) {
		py::list capabilities;
		if ((reference.capabilities & SCOPED_SECRET_CAPABILITY_READ) != 0) {
			capabilities.append(py::str("read"));
		}
		if ((reference.capabilities & SCOPED_SECRET_CAPABILITY_WRITE) != 0) {
			capabilities.append(py::str("write"));
		}
		py::dict description;
		description[py::str("version")] = py::int_(reference.version);
		description[py::str("reference_id")] = py::str(reference.reference_id);
		description[py::str("owner_query_id")] = py::str(reference.owner_query_id);
		description[py::str("owner_session_id")] = py::str(reference.owner_session_id);
		description[py::str("type")] = py::str(reference.secret_type);
		description[py::str("provider")] = py::str(reference.provider);
		description[py::str("scope")] = py::str(reference.normalized_scope);
		description[py::str("capabilities")] = std::move(capabilities);
		result.append(std::move(description));
	}
	return result;
}

static string ScopedSecretRefTupleString(const py::tuple &reference, idx_t field_idx, const char *field_name,
                                         idx_t max_size, bool allow_empty, idx_t &total_reference_bytes,
                                         const char *boundary) {
	if (!py::isinstance<py::str>(reference[field_idx])) {
		throw InvalidInputException("Scoped secret reference %s must be a string", field_name);
	}
	auto value = py::reinterpret_borrow<py::str>(reference[field_idx]);
	if (py::len(value) > max_size) {
		throw InvalidInputException("Scoped secret reference contains an invalid %s", field_name);
	}
	Py_ssize_t utf8_size = 0;
	auto utf8_data = PyUnicode_AsUTF8AndSize(value.ptr(), &utf8_size);
	if (!utf8_data) {
		throw py::error_already_set();
	}
	auto size = NumericCast<idx_t>(utf8_size);
	if ((!allow_empty && size == 0) || size > max_size || memchr(utf8_data, '\0', size)) {
		throw InvalidInputException("Scoped secret reference contains an invalid %s", field_name);
	}
	ConsumeScopedSecretReferenceBytes(size, total_reference_bytes, boundary);
	return string(utf8_data, size);
}

static vector<ScopedSecretRef> ScopedSecretRefsFromPython(py::handle state, const string &expected_owner_query_id,
                                                          const string &expected_session_id, const char *boundary) {
	if (!py::isinstance<py::tuple>(state)) {
		throw InvalidInputException("Scoped secret reference state at %s must be a tuple", boundary);
	}
	auto state_tuple = py::reinterpret_borrow<py::tuple>(state);
	if (state_tuple.size() > SCOPED_SECRET_MAX_REFERENCES) {
		throw InvalidInputException("Scoped secret reference state at %s exceeds the maximum reference count",
		                            boundary);
	}
	vector<ScopedSecretRef> references;
	references.reserve(state_tuple.size());
	idx_t total_reference_bytes = 0;
	for (auto item : state_tuple) {
		if (!py::isinstance<py::tuple>(item)) {
			throw InvalidInputException("Scoped secret reference entry at %s must be a tuple", boundary);
		}
		auto reference_tuple = py::reinterpret_borrow<py::tuple>(item);
		if (reference_tuple.size() != 8) {
			throw InvalidInputException("Scoped secret reference entry at %s must contain exactly 8 fields", boundary);
		}
		if (!py::isinstance<py::int_>(reference_tuple[0]) || py::isinstance<py::bool_>(reference_tuple[0])) {
			throw InvalidInputException("Scoped secret reference version at %s must be an integer", boundary);
		}
		if (!py::isinstance<py::int_>(reference_tuple[7]) || py::isinstance<py::bool_>(reference_tuple[7])) {
			throw InvalidInputException("Scoped secret reference capabilities at %s must be an integer", boundary);
		}
		auto version = reference_tuple[0].cast<int64_t>();
		auto capabilities = reference_tuple[7].cast<int64_t>();
		if (version < 0 || version > NumericLimits<uint32_t>::Maximum()) {
			throw InvalidInputException("Scoped secret reference version at %s is out of range", boundary);
		}
		if (capabilities < 0 || capabilities > NumericLimits<uint8_t>::Maximum()) {
			throw InvalidInputException("Scoped secret reference capabilities at %s are out of range", boundary);
		}
		ScopedSecretRef reference;
		reference.version = static_cast<uint32_t>(version);
		reference.reference_id = ScopedSecretRefTupleString(reference_tuple, 1, "opaque ID", BaseUUID::STRING_SIZE,
		                                                    false, total_reference_bytes, boundary);
		reference.owner_query_id =
		    ScopedSecretRefTupleString(reference_tuple, 2, "owner query ID", SCOPED_SECRET_MAX_IDENTITY_SIZE, false,
		                               total_reference_bytes, boundary);
		reference.owner_session_id =
		    ScopedSecretRefTupleString(reference_tuple, 3, "owner session ID", SCOPED_SECRET_MAX_IDENTITY_SIZE, false,
		                               total_reference_bytes, boundary);
		reference.secret_type = ScopedSecretRefTupleString(
		    reference_tuple, 4, "secret type", SCOPED_SECRET_MAX_IDENTITY_SIZE, false, total_reference_bytes, boundary);
		reference.provider = ScopedSecretRefTupleString(reference_tuple, 5, "provider", SCOPED_SECRET_MAX_IDENTITY_SIZE,
		                                                false, total_reference_bytes, boundary);
		reference.normalized_scope =
		    ScopedSecretRefTupleString(reference_tuple, 6, "normalized scope", SCOPED_SECRET_MAX_SCOPE_SIZE, true,
		                               total_reference_bytes, boundary);
		reference.capabilities = static_cast<uint8_t>(capabilities);
		references.push_back(std::move(reference));
	}
	ValidateScopedSecretRefs(references, expected_owner_query_id, expected_session_id, boundary);
	return references;
}

static bool IsSupportedObjectStorageURI(const string &uri) {
	static constexpr const char *SUPPORTED_PREFIXES[] = {"s3://", "s3a://", "s3n://", "r2://", "gcs://", "gs://"};
	for (auto prefix : SUPPORTED_PREFIXES) {
		if (StringUtil::CIStartsWith(uri, prefix)) {
			return true;
		}
	}
	return false;
}

static string ScopedSecretUseURIFromPython(py::handle value) {
	if (!py::isinstance<py::str>(value)) {
		throw py::type_error("test scoped secret URIs must be strings");
	}
	auto string_value = py::reinterpret_borrow<py::str>(value);
	if (py::len(string_value) > SCOPED_SECRET_MAX_SCOPE_SIZE) {
		throw InvalidInputException("Distributed object-storage URI exceeds the scoped secret use size limit");
	}
	Py_ssize_t utf8_size = 0;
	auto utf8_data = PyUnicode_AsUTF8AndSize(string_value.ptr(), &utf8_size);
	if (!utf8_data) {
		throw py::error_already_set();
	}
	auto size = NumericCast<idx_t>(utf8_size);
	if (size > SCOPED_SECRET_MAX_SCOPE_SIZE) {
		throw InvalidInputException("Distributed object-storage URI exceeds the scoped secret use size limit");
	}
	if (memchr(utf8_data, '\0', size)) {
		throw InvalidInputException("Distributed object-storage URI contains an invalid NUL byte");
	}
	return string(utf8_data, size);
}

static string NormalizedMatchedSecretScope(const BaseSecret &secret, const string &uri) {
	auto matched_score = secret.MatchScore(uri);
	int64_t longest_scope_size = NumericLimits<int64_t>::Minimum();
	string normalized_scope;
	if (secret.GetScope().empty()) {
		longest_scope_size = 0;
	}
	for (auto &scope : secret.GetScope()) {
		if (!scope.empty() && !StringUtil::StartsWith(uri, scope)) {
			continue;
		}
		auto scope_size = NumericCast<int64_t>(scope.size());
		if (scope_size > longest_scope_size || (scope_size == longest_scope_size && scope < normalized_scope)) {
			longest_scope_size = scope_size;
			normalized_scope = scope;
		}
	}
	if (longest_scope_size == NumericLimits<int64_t>::Minimum() || matched_score != longest_scope_size) {
		throw InvalidInputException(
		    "Distributed scoped secrets require DuckDB's standard longest-prefix scope matching semantics");
	}
	ValidateScopedSecretIdentityField(normalized_scope, "normalized scope", SCOPED_SECRET_MAX_SCOPE_SIZE, true);
	return normalized_scope;
}

static bool MergeHTTPSecretIntoS3Request(ClientContext &context) {
	Value merge_http_secret;
	return context.TryGetCurrentSetting("merge_http_secret_into_s3_request", merge_http_secret) &&
	       !merge_http_secret.IsNull() && merge_http_secret.GetValue<bool>();
}

static void ValidateSecretStorageGeneration(SecretManager &secret_manager, CatalogTransaction transaction,
                                            uint64_t expected_generation) {
	if (secret_manager.GetSecretStorageGeneration(transaction) != expected_generation) {
		throw InvalidInputException("DuckDB secret storage registry changed during scoped secret discovery");
	}
}

static std::optional<ScopedSecretSelection> SelectScopedSecretForURI(ClientContext &context, const string &uri) {
	static constexpr const char *SECRET_TYPE_PRECEDENCE[] = {"s3", "r2", "gcs", "aws"};
	auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
	auto &secret_manager = SecretManager::Get(context);
	auto storage_generation = secret_manager.GetSecretStorageGeneration(transaction);
	for (auto secret_type : SECRET_TYPE_PRECEDENCE) {
		auto match = secret_manager.LookupSecret(transaction, uri, secret_type);
		if (!match.HasMatch()) {
			continue;
		}
		auto &entry = *match.secret_entry;
		auto &secret = *entry.secret;
		ValidateScopedSecretIdentityField(entry.storage_mode, "source storage name", SCOPED_SECRET_MAX_IDENTITY_SIZE);
		ValidateScopedSecretIdentityField(secret.GetName(), "source secret name", SCOPED_SECRET_MAX_IDENTITY_SIZE);
		ValidateScopedSecretIdentityField(secret.GetType(), "source secret type", SCOPED_SECRET_MAX_IDENTITY_SIZE);
		ValidateScopedSecretIdentityField(secret.GetProvider(), "source secret provider",
		                                  SCOPED_SECRET_MAX_IDENTITY_SIZE);
		auto selected_type = StringUtil::Lower(secret.GetType());
		auto provider = StringUtil::Lower(secret.GetProvider());
		if (selected_type != secret_type || secret.GetType() != selected_type || secret.GetProvider() != provider) {
			throw InvalidInputException(
			    "Distributed scoped secret metadata must use canonical lowercase type/provider names");
		}
		if (!IsSupportedScopedSecretProvider(selected_type, provider)) {
			throw InvalidInputException(
			    "Distributed scoped secrets support only config and credential_chain providers for canonical "
			    "object-storage secret types");
		}
		ScopedSecretSelection selection;
		selection.storage_name = entry.storage_mode;
		selection.secret_name = secret.GetName();
		selection.secret_type = std::move(selected_type);
		selection.provider = std::move(provider);
		selection.normalized_scope = NormalizedMatchedSecretScope(secret, uri);
		ValidateSecretStorageGeneration(secret_manager, transaction, storage_generation);
		return selection;
	}

	// KeyValueSecretReader stops at the first matching type. Therefore an HTTP
	// secret participates only when none of the S3-compatible types above match
	// and this setting leaves HTTP in the reader's type list. HTTP secrets are
	// deliberately outside the first distributed object-storage contract, so
	// reject an actually selected one instead of silently omitting it.
	if (MergeHTTPSecretIntoS3Request(context)) {
		auto http_match = secret_manager.LookupSecret(transaction, uri, "http");
		if (http_match.HasMatch()) {
			auto &entry = *http_match.secret_entry;
			auto &secret = *entry.secret;
			ValidateScopedSecretIdentityField(entry.storage_mode, "source storage name",
			                                  SCOPED_SECRET_MAX_IDENTITY_SIZE);
			ValidateScopedSecretIdentityField(secret.GetName(), "source secret name", SCOPED_SECRET_MAX_IDENTITY_SIZE);
			ValidateScopedSecretIdentityField(secret.GetType(), "source secret type", SCOPED_SECRET_MAX_IDENTITY_SIZE);
			ValidateScopedSecretIdentityField(secret.GetProvider(), "source secret provider",
			                                  SCOPED_SECRET_MAX_IDENTITY_SIZE);
			throw InvalidInputException(
			    "Distributed scoped secrets do not support HTTP secret selections for object-storage URIs");
		}
	}
	ValidateSecretStorageGeneration(secret_manager, transaction, storage_generation);
	return std::nullopt;
}

static bool ScopedSecretSelectionMatchesBinding(const ScopedSecretSelection &selection,
                                                const ScopedSecretBinding &binding) {
	return selection.storage_name == binding.storage_name && selection.secret_name == binding.secret_name &&
	       selection.secret_type == binding.secret_type && selection.provider == binding.provider &&
	       selection.normalized_scope == binding.normalized_scope;
}

static bool ScopedSecretSelectionsEqual(const std::optional<ScopedSecretSelection> &left,
                                        const std::optional<ScopedSecretSelection> &right) {
	if (left.has_value() != right.has_value()) {
		return false;
	}
	if (!left) {
		return true;
	}
	return left->storage_name == right->storage_name && left->secret_name == right->secret_name &&
	       left->secret_type == right->secret_type && left->provider == right->provider &&
	       left->normalized_scope == right->normalized_scope;
}

static std::optional<ScopedSecretSelection> SelectScopedSecretForUse(ClientContext &context,
                                                                     const ScopedSecretUse &use) {
	if (!use.covers_generated_copy_namespace) {
		return SelectScopedSecretForURI(context, use.uri);
	}
	if (use.uri.size() > SCOPED_SECRET_MAX_SCOPE_SIZE) {
		throw InvalidInputException("Distributed COPY output namespace exceeds the scoped secret path size limit");
	}

	auto &fs = FileSystem::GetFileSystem(context);
	auto canonical_root = distributed::CanonicalDistributedCopyBasePath(fs, use.uri);
	if (canonical_root.is_err()) {
		throw InvalidInputException("Distributed COPY secret discovery could not canonicalize its output namespace");
	}
	auto canonical_base = std::move(canonical_root).value();
	auto separator = fs.PathSeparator(canonical_base);
	if (separator.empty()) {
		throw InvalidInputException("Distributed COPY secret discovery requires a non-empty path separator");
	}
	const auto commit_suffix_size = sizeof(SCOPED_SECRET_COPY_COMMIT_SUFFIX) - 1;
	if (separator.size() > SCOPED_SECRET_MAX_SCOPE_SIZE ||
	    commit_suffix_size > SCOPED_SECRET_MAX_SCOPE_SIZE - separator.size() ||
	    canonical_base.size() > SCOPED_SECRET_MAX_SCOPE_SIZE - separator.size() - commit_suffix_size) {
		throw InvalidInputException("Distributed COPY output namespace exceeds the scoped secret path size limit");
	}
	auto with_trailing_separator = [&](string path) {
		if (!StringUtil::EndsWith(path, separator)) {
			path += separator;
		}
		return path;
	};
	auto data_namespace_prefix = with_trailing_separator(canonical_base);
	auto commit_namespace_prefix = with_trailing_separator(canonical_base + SCOPED_SECRET_COPY_COMMIT_SUFFIX);
	auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
	auto &secret_manager = SecretManager::Get(context);
	auto storage_generation = secret_manager.GetSecretStorageGeneration(transaction);

	// Selecting at the common child prefix includes a secret scoped exactly to
	// the output directory without inventing a filename that could accidentally
	// match an otherwise unused narrower scope. Direct-write lifecycle,
	// manifest, and committed-marker paths must use that same identity.
	auto baseline = SelectScopedSecretForURI(context, data_namespace_prefix);
	auto commit_selection = SelectScopedSecretForURI(context, commit_namespace_prefix);
	if (!ScopedSecretSelectionsEqual(baseline, commit_selection)) {
		throw InvalidInputException(
		    "Distributed COPY generated output paths can select different scoped secrets; use one invariant "
		    "secret identity for the complete output namespace");
	}

	vector<string> boundary_probes;
	std::unordered_set<string> unique_boundaries;
	idx_t metadata_storage_count = 0;
	idx_t metadata_entry_count = 0;
	idx_t metadata_scope_count = 0;
	idx_t total_metadata_bytes = 0;
	idx_t total_boundary_bytes = 0;
	bool metadata_limit_exceeded = false;
	bool unsupported_storage_lookup = false;
	bool unsupported_prefix_matching = false;
	const bool include_http_boundaries = MergeHTTPSecretIntoS3Request(context);
	auto consume_metadata_bytes = [&](idx_t size) {
		if (size > SCOPED_SECRET_MAX_METADATA_BYTES - total_metadata_bytes) {
			metadata_limit_exceeded = true;
			return false;
		}
		total_metadata_bytes += size;
		return true;
	};
	auto metadata_scan_completed = secret_manager.ScanSecretMetadata(
	    transaction,
	    [&](const SecretStorageMetadata &metadata) {
		    if (metadata_storage_count >= SCOPED_SECRET_MAX_METADATA_STORAGES ||
		        !consume_metadata_bytes(metadata.name.size())) {
			    metadata_limit_exceeded = true;
			    return false;
		    }
		    metadata_storage_count++;
		    if (!metadata.uses_standard_secret_lookup) {
			    unsupported_storage_lookup = true;
			    return false;
		    }
		    return true;
	    },
	    [&](const SecretMetadata &metadata) {
		    if (metadata_entry_count >= SCOPED_SECRET_MAX_METADATA_ENTRIES ||
		        metadata.scope.size() > SCOPED_SECRET_MAX_METADATA_SCOPES - metadata_scope_count) {
			    metadata_limit_exceeded = true;
			    return false;
		    }
		    metadata_entry_count++;
		    metadata_scope_count += metadata.scope.size();
		    if (!consume_metadata_bytes(metadata.storage_mode.size()) ||
		        !consume_metadata_bytes(metadata.name.size()) || !consume_metadata_bytes(metadata.type.size()) ||
		        !consume_metadata_bytes(metadata.provider.size())) {
			    return false;
		    }

		    const bool affects_selection = IsScopedSecretSelectionType(metadata.type, include_http_boundaries);
		    if (affects_selection && !metadata.uses_standard_prefix_matching) {
			    unsupported_prefix_matching = true;
			    return false;
		    }
		    for (auto &scope : metadata.scope) {
			    if (!consume_metadata_bytes(scope.size())) {
				    return false;
			    }
			    if (!affects_selection) {
				    continue;
			    }
			    const bool is_nested_boundary = (scope.size() > data_namespace_prefix.size() &&
			                                     StringUtil::StartsWith(scope, data_namespace_prefix)) ||
			                                    (scope.size() > commit_namespace_prefix.size() &&
			                                     StringUtil::StartsWith(scope, commit_namespace_prefix));
			    if (!is_nested_boundary) {
				    continue;
			    }
			    ValidateScopedSecretIdentityField(scope, "generated COPY scope", SCOPED_SECRET_MAX_SCOPE_SIZE);
			    if (unique_boundaries.find(scope) != unique_boundaries.end()) {
				    continue;
			    }
			    auto boundary_bytes = scope.size() + sizeof(SCOPED_SECRET_GENERATED_PROBE_SUFFIX) - 1;
			    if (unique_boundaries.size() >= SCOPED_SECRET_MAX_GENERATED_BOUNDARIES ||
			        boundary_bytes > SCOPED_SECRET_MAX_GENERATED_BOUNDARY_BYTES - total_boundary_bytes) {
				    metadata_limit_exceeded = true;
				    return false;
			    }
			    unique_boundaries.insert(scope);
			    total_boundary_bytes += boundary_bytes;
			    boundary_probes.push_back(scope + SCOPED_SECRET_GENERATED_PROBE_SUFFIX);
		    }
		    return true;
	    });
	if (!metadata_scan_completed) {
		if (unsupported_storage_lookup || unsupported_prefix_matching) {
			throw InvalidInputException(
			    "Distributed COPY scoped secret discovery requires standard longest-prefix matching semantics");
		}
		if (!metadata_limit_exceeded) {
			throw InternalException("DuckDB secret metadata scan stopped unexpectedly");
		}
		throw InvalidInputException(
		    "Distributed COPY output namespace exceeds the scoped secret metadata validation limit");
	}
	std::sort(boundary_probes.begin(), boundary_probes.end());
	for (auto &probe : boundary_probes) {
		auto selected = SelectScopedSecretForURI(context, probe);
		if (!ScopedSecretSelectionsEqual(baseline, selected)) {
			throw InvalidInputException(
			    "Distributed COPY generated output paths can select different scoped secrets; use one invariant "
			    "secret identity for the complete output namespace");
		}
	}
	ValidateSecretStorageGeneration(secret_manager, transaction, storage_generation);
	return baseline;
}

static void AddScopedSecretUse(vector<ScopedSecretUse> &uses, idx_t &total_use_bytes, const string &uri,
                               uint8_t capability, bool covers_generated_copy_namespace = false) {
	if (!IsSupportedObjectStorageURI(uri)) {
		return;
	}
	if (ContainsNul(uri)) {
		throw InvalidInputException("Distributed object-storage URI contains an invalid NUL byte");
	}
	if (uri.size() > SCOPED_SECRET_MAX_SCOPE_SIZE) {
		if (!covers_generated_copy_namespace) {
			throw InvalidInputException("Distributed object-storage URI exceeds the scoped secret use size limit");
		}
		throw InvalidInputException("Distributed COPY output namespace exceeds the scoped secret path size limit");
	}
	if (uses.size() >= SCOPED_SECRET_MAX_USES) {
		throw InvalidInputException("Distributed plan exceeds the maximum supported remote object-storage URI count");
	}
	if (total_use_bytes > SCOPED_SECRET_MAX_USE_BYTES || uri.size() > SCOPED_SECRET_MAX_USE_BYTES - total_use_bytes) {
		throw InvalidInputException("Distributed plan exceeds the maximum scoped secret URI byte size");
	}
	total_use_bytes += uri.size();
	uses.push_back({uri, capability, covers_generated_copy_namespace});
}

static void CollectLogicalPlanScopedSecretUses(const LogicalOperator &op, vector<ScopedSecretUse> &uses,
                                               idx_t &total_use_bytes) {
	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		auto &get = op.Cast<LogicalGet>();
		if (get.function.get_bind_info) {
			auto bind_info = get.function.get_bind_info(get.bind_data.get());
			if (bind_info.type == ScanType::EXTERNAL) {
				auto file_paths = bind_info.options.find("file_path");
				if (file_paths != bind_info.options.end()) {
					auto &path_list = file_paths->second;
					if (path_list.IsNull() || path_list.type().id() != LogicalTypeId::LIST ||
					    ListType::GetChildType(path_list.type()) != LogicalType::VARCHAR) {
						throw InvalidInputException(
						    "Distributed external scan file_path discovery requires LIST(VARCHAR) bind metadata");
					}
					for (auto &path : ListValue::GetChildren(path_list)) {
						if (path.IsNull() || path.type() != LogicalType::VARCHAR) {
							throw InvalidInputException(
							    "Distributed external scan file_path discovery contains a non-VARCHAR URI");
						}
						AddScopedSecretUse(uses, total_use_bytes, StringValue::Get(path),
						                   SCOPED_SECRET_CAPABILITY_READ);
					}
				}
			}
		}
		// Multi-file bind metadata deliberately displays the original glob. The
		// glob lookup itself and every expanded object can select different
		// scopes, so force expansion before submission and capture both.
		auto multi_file_bind = dynamic_cast<const MultiFileBindData *>(get.bind_data.get());
		if (multi_file_bind && multi_file_bind->file_list) {
			for (auto &file : multi_file_bind->file_list->Files()) {
				AddScopedSecretUse(uses, total_use_bytes, file.path, SCOPED_SECRET_CAPABILITY_READ);
			}
		}
	} else if (op.type == LogicalOperatorType::LOGICAL_COPY_TO_FILE) {
		auto &copy = op.Cast<LogicalCopyToFile>();
		AddScopedSecretUse(uses, total_use_bytes, copy.file_path, SCOPED_SECRET_CAPABILITY_WRITE, true);
	}
	for (auto &child : op.children) {
		CollectLogicalPlanScopedSecretUses(*child, uses, total_use_bytes);
	}
}

static ScopedSecretDiscovery DiscoverScopedSecretRefs(ClientContext &context, vector<ScopedSecretUse> uses,
                                                      const string &owner_query_id, const string &owner_session_id) {
	std::sort(uses.begin(), uses.end(), [](const ScopedSecretUse &left, const ScopedSecretUse &right) {
		return std::tie(left.uri, left.covers_generated_copy_namespace, left.capabilities) <
		       std::tie(right.uri, right.covers_generated_copy_namespace, right.capabilities);
	});
	vector<ScopedSecretUse> canonical_uses;
	canonical_uses.reserve(uses.size());
	for (auto &use : uses) {
		if (!canonical_uses.empty() && canonical_uses.back().uri == use.uri &&
		    canonical_uses.back().covers_generated_copy_namespace == use.covers_generated_copy_namespace) {
			canonical_uses.back().capabilities |= use.capabilities;
		} else {
			canonical_uses.push_back(std::move(use));
		}
	}

	ScopedSecretDiscovery discovery;
	std::unordered_map<string, idx_t> selection_indices;
	std::unordered_set<string> selected_scope_keys;
	selection_indices.reserve(MinValue<idx_t>(canonical_uses.size(), SCOPED_SECRET_MAX_REFERENCES));
	selected_scope_keys.reserve(MinValue<idx_t>(canonical_uses.size(), SCOPED_SECRET_MAX_REFERENCES));
	idx_t total_reference_bytes = 0;
	bool owner_identity_validated = false;
	for (auto &use : canonical_uses) {
		auto selection = SelectScopedSecretForUse(context, use);
		if (!selection) {
			discovery.source_unmatched_uses.push_back(std::move(use));
			continue;
		}
		if (owner_query_id.empty() || owner_session_id.empty()) {
			throw InvalidInputException(
			    "Distributed object-storage secrets require non-empty query and Vane session ownership");
		}
		if (!owner_identity_validated) {
			ValidateScopedSecretIdentityField(owner_query_id, "owner query ID", SCOPED_SECRET_MAX_IDENTITY_SIZE);
			ValidateScopedSecretIdentityField(owner_session_id, "owner session ID", SCOPED_SECRET_MAX_IDENTITY_SIZE);
			owner_identity_validated = true;
		}
		ValidateScopedSecretSelection(*selection);

		auto selection_key = CreateScopedSecretSelectionKey(*selection);
		auto existing_ref = selection_indices.find(selection_key);
		if (existing_ref != selection_indices.end()) {
			auto ref_idx = existing_ref->second;
			discovery.references[ref_idx].capabilities |= use.capabilities;
			discovery.source_bindings[ref_idx].uses.push_back(std::move(use));
			continue;
		}
		auto scope_key = CreateScopedSecretScopeKey(*selection);
		if (selected_scope_keys.find(scope_key) != selected_scope_keys.end()) {
			throw InvalidInputException("Distributed scoped secret selection is ambiguous for one type/provider/scope");
		}
		if (discovery.references.size() >= SCOPED_SECRET_MAX_REFERENCES) {
			throw InvalidInputException("Distributed plan exceeds the maximum scoped secret reference count");
		}
		auto next_total_reference_bytes = PreflightScopedSecretReferenceBytes(
		    *selection, owner_query_id, owner_session_id, total_reference_bytes, "logical plan capture");

		ScopedSecretRef reference;
		reference.reference_id = CreateScopedSecretReferenceId(*selection, owner_query_id, owner_session_id);
		reference.owner_query_id = owner_query_id;
		reference.owner_session_id = owner_session_id;
		reference.secret_type = selection->secret_type;
		reference.provider = selection->provider;
		reference.normalized_scope = selection->normalized_scope;
		reference.capabilities = use.capabilities;
		total_reference_bytes = next_total_reference_bytes;

		ScopedSecretBinding binding;
		binding.reference_id = reference.reference_id;
		binding.storage_name = selection->storage_name;
		binding.secret_name = selection->secret_name;
		binding.secret_type = selection->secret_type;
		binding.provider = selection->provider;
		binding.normalized_scope = selection->normalized_scope;
		binding.uses.push_back(std::move(use));

		auto ref_idx = discovery.references.size();
		discovery.references.push_back(std::move(reference));
		discovery.source_bindings.push_back(std::move(binding));
		selection_indices.emplace(std::move(selection_key), ref_idx);
		selected_scope_keys.insert(std::move(scope_key));
	}

	std::sort(discovery.references.begin(), discovery.references.end(), ScopedSecretRefLess);
	ValidateScopedSecretRefs(discovery.references, owner_query_id, owner_session_id, "logical plan capture");
	return discovery;
}

static void ValidateSourceScopedSecretBindings(ClientContext &context, const vector<ScopedSecretRef> &references,
                                               const vector<ScopedSecretBinding> &bindings,
                                               const vector<ScopedSecretUse> &unmatched_uses) {
	if (bindings.size() != references.size()) {
		throw InternalException("Source scoped secret binding count does not match the reference count");
	}
	for (auto &binding : bindings) {
		auto reference = std::find_if(references.begin(), references.end(), [&](const ScopedSecretRef &candidate) {
			return candidate.reference_id == binding.reference_id;
		});
		if (reference == references.end()) {
			throw InternalException("Source scoped secret binding has no matching opaque reference");
		}
		uint8_t capabilities = 0;
		for (auto &use : binding.uses) {
			auto selection = SelectScopedSecretForUse(context, use);
			if (!selection || !ScopedSecretSelectionMatchesBinding(*selection, binding)) {
				throw InvalidInputException(
				    "Scoped secret reference '%s' is stale because DuckDB now selects a different secret for its scope",
				    binding.reference_id);
			}
			capabilities |= use.capabilities;
		}
		if (capabilities != reference->capabilities) {
			throw InternalException("Source scoped secret binding capabilities do not match the reference");
		}
	}
	for (auto &use : unmatched_uses) {
		if (SelectScopedSecretForUse(context, use)) {
			throw InvalidInputException(
			    "Scoped secret discovery is stale because DuckDB now selects a secret for a previously unmatched "
			    "object-storage resource");
		}
	}
}

static void AppendUInt32LE(string &result, uint32_t value) {
	for (idx_t byte_idx = 0; byte_idx < sizeof(value); byte_idx++) {
		result.push_back(static_cast<char>((value >> (byte_idx * 8)) & 0xff));
	}
}

static void AppendUInt64LE(string &result, uint64_t value) {
	for (idx_t byte_idx = 0; byte_idx < sizeof(value); byte_idx++) {
		result.push_back(static_cast<char>((value >> (byte_idx * 8)) & 0xff));
	}
}

static uint32_t ReadUInt32LE(const string &envelope, idx_t &offset, const char *field_name) {
	if (offset > envelope.size() || envelope.size() - offset < sizeof(uint32_t)) {
		throw InvalidInputException("Logical plan envelope is truncated before %s", field_name);
	}
	uint32_t result = 0;
	for (idx_t byte_idx = 0; byte_idx < sizeof(result); byte_idx++) {
		result |= static_cast<uint32_t>(static_cast<uint8_t>(envelope[offset++])) << (byte_idx * 8);
	}
	return result;
}

static uint64_t ReadUInt64LE(const string &envelope, idx_t &offset, const char *field_name) {
	if (offset > envelope.size() || envelope.size() - offset < sizeof(uint64_t)) {
		throw InvalidInputException("Logical plan envelope is truncated before %s", field_name);
	}
	uint64_t result = 0;
	for (idx_t byte_idx = 0; byte_idx < sizeof(result); byte_idx++) {
		result |= static_cast<uint64_t>(static_cast<uint8_t>(envelope[offset++])) << (byte_idx * 8);
	}
	return result;
}

static string EncodeLogicalPlanEnvelope(const string &logical_payload) {
	if (logical_payload.empty()) {
		throw InternalException("Cannot encode an empty logical plan payload");
	}
	string source_id = DuckDB::SourceID();
	if (source_id.empty() || source_id.size() > LOGICAL_PLAN_MAX_SOURCE_ID_SIZE) {
		throw InternalException("DuckDB SourceID is empty or exceeds the logical plan envelope limit");
	}

	string envelope;
	envelope.reserve(LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE + sizeof(uint32_t) * 2 + sizeof(uint64_t) + source_id.size() +
	                 logical_payload.size());
	envelope.append(LOGICAL_PLAN_ENVELOPE_MAGIC, LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE);
	AppendUInt32LE(envelope, LOGICAL_PLAN_PROTOCOL_VERSION);
	AppendUInt32LE(envelope, static_cast<uint32_t>(source_id.size()));
	AppendUInt64LE(envelope, static_cast<uint64_t>(logical_payload.size()));
	envelope.append(source_id);
	envelope.append(logical_payload);
	return envelope;
}

static string DecodeLogicalPlanEnvelope(const string &envelope) {
	if (envelope.size() < LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE ||
	    envelope.compare(0, LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE, LOGICAL_PLAN_ENVELOPE_MAGIC,
	                     LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE) != 0) {
		throw InvalidInputException("Logical plan payload is not a Vane logical plan envelope");
	}

	idx_t offset = LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE;
	auto protocol_version = ReadUInt32LE(envelope, offset, "protocol version");
	if (protocol_version != LOGICAL_PLAN_PROTOCOL_VERSION) {
		throw InvalidInputException("Unsupported logical plan protocol version %u (expected %u)", protocol_version,
		                            LOGICAL_PLAN_PROTOCOL_VERSION);
	}
	auto source_id_size = ReadUInt32LE(envelope, offset, "SourceID length");
	auto payload_size = ReadUInt64LE(envelope, offset, "payload length");
	if (source_id_size == 0 || source_id_size > LOGICAL_PLAN_MAX_SOURCE_ID_SIZE) {
		throw InvalidInputException("Logical plan envelope contains an invalid SourceID length");
	}
	if (offset > envelope.size() || source_id_size > envelope.size() - offset) {
		throw InvalidInputException("Logical plan envelope is truncated inside the SourceID");
	}

	auto serialized_source_id = envelope.substr(offset, source_id_size);
	offset += source_id_size;
	string current_source_id = DuckDB::SourceID();
	if (current_source_id.empty() || serialized_source_id != current_source_id) {
		throw InvalidInputException("Logical plan SourceID mismatch: payload was built by %s, current engine is %s",
		                            serialized_source_id, current_source_id);
	}
	if (payload_size == 0) {
		throw InvalidInputException("Logical plan envelope contains an empty payload");
	}
	auto remaining_size = envelope.size() - offset;
	if (payload_size != static_cast<uint64_t>(remaining_size)) {
		throw InvalidInputException("Logical plan envelope payload length mismatch: declared %llu bytes, found %llu",
		                            payload_size, static_cast<uint64_t>(remaining_size));
	}
	return envelope.substr(offset, remaining_size);
}

struct PyLogicalPlan {
	string query_id_;
	string serialized_logical_plan_;
	// Driver-local source connection; intentionally omitted from pickle state.
	py::object source_connection_ = py::none();
	py::object udf_registrations_ = py::none();
	py::object connection_snapshot_ = py::none();
	vector<ScopedSecretRef> scoped_secret_refs_;
	// Source secret names, storage identities, and URIs never cross pickle.
	vector<ScopedSecretBinding> source_scoped_secret_bindings_;
	vector<ScopedSecretUse> source_unmatched_scoped_secret_uses_;

	PyLogicalPlan() = default;

	string idx() const {
		return query_id_;
	}

	string session_id() const;
	py::dict session_config() const;
	bool has_explicit_s3_credentials() const;
	py::list scoped_secret_refs() const {
		return DescribeScopedSecretRefs(scoped_secret_refs_);
	}

	PyPhysicalPlanWrapper to_physical_plan(py::object conn_obj, py::object effective_session_config) const;
};

struct SerializedLogicalPlanCapture {
	string serialized_plan;
	ScopedSecretDiscovery scoped_secrets;
};

static SerializedLogicalPlanCapture SerializeLogicalPlanFromRelation(const duckdb::shared_ptr<duckdb::Relation> &rel,
                                                                     const string &owner_query_id,
                                                                     const string &owner_session_id) {
	if (!rel) {
		throw duckdb::InternalException("Relation is null");
	}
	auto client_context = rel->context->GetContext();
	SerializedLogicalPlanCapture capture;
	client_context->RunFunctionInTransaction([&]() {
		auto statement_binder = duckdb::Binder::CreateBinder(*client_context);
		auto relation_stmt = make_uniq<duckdb::RelationStatement>(rel, *statement_binder);
		duckdb::Planner planner(*client_context);
		planner.CreatePlan(std::move(relation_stmt));
		auto logical_plan = std::move(planner.plan);

		vector<ScopedSecretUse> scoped_secret_uses;
		idx_t total_scoped_secret_use_bytes = 0;
		CollectLogicalPlanScopedSecretUses(*logical_plan, scoped_secret_uses, total_scoped_secret_use_bytes);
		capture.scoped_secrets =
		    DiscoverScopedSecretRefs(*client_context, std::move(scoped_secret_uses), owner_query_id, owner_session_id);

		// NOTE: We intentionally do NOT run the Optimizer here.
		// The unoptimized (bound) logical plan is serialized and sent to the Driver,
		// where the Optimizer runs. This avoids needing serialization support for
		// custom LogicalOperator types created by optimizer passes
		// (e.g., LogicalUDFProject, LogicalLocalExchange).

		duckdb::MemoryStream stream(duckdb::Allocator::Get(*client_context));
		duckdb::SerializationOptions options;
		options.serialization_compatibility = duckdb::SerializationCompatibility::Latest();
		options.serialize_default_values = true;
		duckdb::BinarySerializer serializer(stream, options);
		serializer.Begin();
		logical_plan->Serialize(serializer);
		serializer.End();

		auto data_ptr = stream.GetData();
		auto data_size = stream.GetPosition();
		if (data_size == 0) {
			throw duckdb::InternalException("Logical plan serialization returned empty payload");
		}
		auto logical_payload = string(reinterpret_cast<const char *>(data_ptr), data_size);
		capture.serialized_plan = EncodeLogicalPlanEnvelope(logical_payload);
	});
	return capture;
}

static DuckDBPyConnection &ExtractPyConnectionWrapper(py::object conn_obj) {
	if (py::hasattr(conn_obj, "c")) {
		return conn_obj.attr("c").cast<DuckDBPyConnection &>();
	}
	if (py::isinstance<DuckDBPyConnection>(conn_obj)) {
		return conn_obj.cast<DuckDBPyConnection &>();
	}
	throw duckdb::InternalException("Connection object must have 'c' attribute or be a DuckDBPyConnection");
}

static py::dict CopyPyDict(const py::dict &source) {
	py::dict result;
	for (auto item : source) {
		result[item.first] = item.second;
	}
	return result;
}

static bool IsExtensionSecuritySetting(const string &lower_name) {
	return lower_name == "allow_unsigned_extensions" || lower_name == "autoinstall_known_extensions" ||
	       lower_name == "autoload_known_extensions";
}

static bool IsSecretPersistenceSetting(const string &lower_name) {
	return lower_name == "allow_persistent_secrets" || lower_name == "default_secret_storage" ||
	       lower_name == "secret_directory";
}

static bool IsWorkerResourceSetting(const string &lower_name) {
	return lower_name == "max_memory" || lower_name == "memory_limit" || lower_name == "threads" ||
	       lower_name == "worker_threads" || lower_name == "local_exchange_streaming" ||
	       lower_name == "local_exchange_buffer_bytes" || lower_name == "arrow_large_buffer_size";
}

static py::dict SanitizeBootstrapConfig(const py::dict &config, bool disable_persistent_secrets,
                                        bool remove_worker_resource_settings = false) {
	py::dict sanitized;
	for (auto item : config) {
		auto name = duckdb::StringUtil::Lower(py::str(item.first).cast<string>());
		if (IsExtensionSecuritySetting(name)) {
			sanitized[py::str(name)] = py::str("false");
			continue;
		}
		if (disable_persistent_secrets && IsSecretPersistenceSetting(name)) {
			continue;
		}
		if (remove_worker_resource_settings && IsWorkerResourceSetting(name)) {
			continue;
		}
		sanitized[item.first] = item.second;
	}
	if (disable_persistent_secrets) {
		sanitized[py::str("allow_persistent_secrets")] = py::str("false");
	}
	return sanitized;
}

static py::dict ForceReadOnlyAccessMode(const py::dict &config) {
	py::dict result;
	for (auto item : config) {
		auto name = duckdb::StringUtil::Lower(py::str(item.first).cast<string>());
		if (name != "access_mode") {
			result[item.first] = item.second;
		}
	}
	result[py::str("access_mode")] = py::str("read_only");
	return result;
}

static py::object LookupBootstrapSnapshot(const py::object &snapshot_obj) {
	if (snapshot_obj.is_none() || !py::isinstance<py::dict>(snapshot_obj)) {
		return py::none();
	}
	auto snapshot = snapshot_obj.cast<py::dict>();
	if (!snapshot.contains(py::str("bootstrap"))) {
		return py::none();
	}
	auto bootstrap_obj = snapshot[py::str("bootstrap")];
	if (bootstrap_obj.is_none() || !py::isinstance<py::dict>(bootstrap_obj)) {
		return py::none();
	}
	return bootstrap_obj;
}

static py::dict LookupVaneSessionSnapshot(const py::object &snapshot_obj) {
	if (snapshot_obj.is_none() || !py::isinstance<py::dict>(snapshot_obj)) {
		throw duckdb::InvalidInputException("Connection snapshot is missing the required Vane session");
	}
	auto snapshot = snapshot_obj.cast<py::dict>();
	if (!snapshot.contains(py::str("vane_session")) || !py::isinstance<py::dict>(snapshot[py::str("vane_session")])) {
		throw duckdb::InvalidInputException("Connection snapshot is missing the required Vane session");
	}
	return snapshot[py::str("vane_session")].cast<py::dict>();
}

static string VaneSessionIdFromSnapshot(const py::object &snapshot_obj) {
	auto session = LookupVaneSessionSnapshot(snapshot_obj);
	if (!session.contains(py::str("id")) || session[py::str("id")].is_none()) {
		throw duckdb::InvalidInputException("Connection snapshot Vane session is missing id");
	}
	auto session_id = py::str(session[py::str("id")]).cast<string>();
	if (session_id.empty()) {
		throw duckdb::InvalidInputException("Connection snapshot Vane session id must not be empty");
	}
	return session_id;
}

static py::dict VaneSessionConfigFromSnapshot(const py::object &snapshot_obj) {
	auto session = LookupVaneSessionSnapshot(snapshot_obj);
	if (!session.contains(py::str("config")) || session[py::str("config")].is_none()) {
		throw duckdb::InvalidInputException("Connection snapshot Vane session is missing config");
	}
	if (!py::isinstance<py::dict>(session[py::str("config")])) {
		throw duckdb::InvalidInputException("Connection snapshot Vane session config must be a dict");
	}
	return CopyPyDict(session[py::str("config")].cast<py::dict>());
}

static bool HasExplicitS3CredentialsFromSnapshot(const py::object &snapshot_obj) {
	if (snapshot_obj.is_none() || !py::isinstance<py::dict>(snapshot_obj)) {
		throw duckdb::InvalidInputException("Connection snapshot is missing the required Vane session");
	}
	auto snapshot = snapshot_obj.cast<py::dict>();
	if (!snapshot.contains(py::str("settings")) || !py::isinstance<py::list>(snapshot[py::str("settings")])) {
		return false;
	}
	bool has_access_key = false;
	bool has_secret_key = false;
	bool has_session_token = false;
	string access_key;
	string secret_key;
	string session_token;
	for (auto item : snapshot[py::str("settings")].cast<py::list>()) {
		if (!py::isinstance<py::dict>(item)) {
			continue;
		}
		auto setting = py::reinterpret_borrow<py::dict>(item);
		if (!setting.contains(py::str("name")) || !setting.contains(py::str("value"))) {
			continue;
		}
		auto name = duckdb::StringUtil::Lower(py::str(setting[py::str("name")]).cast<string>());
		if (name == "s3_access_key_id") {
			has_access_key = true;
			access_key = py::str(setting[py::str("value")]).cast<string>();
		} else if (name == "s3_secret_access_key") {
			has_secret_key = true;
			secret_key = py::str(setting[py::str("value")]).cast<string>();
		} else if (name == "s3_session_token") {
			has_session_token = true;
			session_token = py::str(setting[py::str("value")]).cast<string>();
		}
	}
	const bool has_access_key_value = !access_key.empty();
	const bool has_secret_key_value = !secret_key.empty();
	if (has_access_key != has_secret_key || has_access_key_value != has_secret_key_value ||
	    (has_session_token && !session_token.empty() && !has_access_key_value)) {
		throw duckdb::InvalidInputException(
		    "Explicit DuckDB S3 credentials must set both s3_access_key_id and s3_secret_access_key");
	}
	return has_access_key;
}

static bool IsDefaultBootstrapSnapshot(const py::object &bootstrap_obj) {
	if (bootstrap_obj.is_none() || !py::isinstance<py::dict>(bootstrap_obj)) {
		return true;
	}
	auto bootstrap = bootstrap_obj.cast<py::dict>();

	string database = ":memory:";
	if (bootstrap.contains(py::str("database")) && !bootstrap[py::str("database")].is_none()) {
		database = py::str(bootstrap[py::str("database")]).cast<string>();
	}

	bool read_only = false;
	if (bootstrap.contains(py::str("read_only")) && !bootstrap[py::str("read_only")].is_none()) {
		read_only = bootstrap[py::str("read_only")].cast<bool>();
	}

	py::dict config = py::dict();
	if (bootstrap.contains(py::str("config")) && !bootstrap[py::str("config")].is_none() &&
	    py::isinstance<py::dict>(bootstrap[py::str("config")])) {
		config = bootstrap[py::str("config")].cast<py::dict>();
	}
	return database == ":memory:" && !read_only && py::len(config) == 0;
}

static py::object NormalizeBootstrapSnapshot(const py::dict &bootstrap_obj) {
	py::dict bootstrap;
	bootstrap[py::str("database")] =
	    bootstrap_obj.contains(py::str("database")) && !bootstrap_obj[py::str("database")].is_none()
	        ? py::object(py::str(bootstrap_obj[py::str("database")]))
	        : py::object(py::str(":memory:"));
	bootstrap[py::str("read_only")] =
	    bootstrap_obj.contains(py::str("read_only")) && !bootstrap_obj[py::str("read_only")].is_none()
	        ? py::object(py::bool_(bootstrap_obj[py::str("read_only")].cast<bool>()))
	        : py::object(py::bool_(false));
	if (bootstrap_obj.contains(py::str("config")) && !bootstrap_obj[py::str("config")].is_none() &&
	    py::isinstance<py::dict>(bootstrap_obj[py::str("config")])) {
		bootstrap[py::str("config")] = CopyPyDict(bootstrap_obj[py::str("config")].cast<py::dict>());
	} else {
		bootstrap[py::str("config")] = py::dict();
	}
	return bootstrap;
}

static py::object PrepareWorkerConnectionSnapshot(const py::object &snapshot_obj) {
	if (snapshot_obj.is_none() || !py::isinstance<py::dict>(snapshot_obj)) {
		return snapshot_obj;
	}
	auto snapshot = CopyPyDict(snapshot_obj.cast<py::dict>());
	// Attached catalogs are replayed only by the isolated coordinator planning
	// connection. The serialized physical worker bind is already self-contained,
	// so forwarding executable ATTACH statements (and their options) to workers
	// would cross an unnecessary credential and metadata boundary.
	snapshot.attr("pop")(py::str("attached_databases"), py::none());
	auto settings_key = py::str("settings");
	if (!snapshot.contains(settings_key) || !py::isinstance<py::list>(snapshot[settings_key])) {
		return snapshot;
	}

	py::list worker_settings;
	for (auto item : snapshot[settings_key].cast<py::list>()) {
		if (py::isinstance<py::dict>(item)) {
			auto setting = py::reinterpret_borrow<py::dict>(item);
			if (setting.contains(py::str("name")) && py::isinstance<py::str>(setting[py::str("name")])) {
				auto name = duckdb::StringUtil::Lower(py::str(setting[py::str("name")]).cast<string>());
				if (IsWorkerResourceSetting(name)) {
					continue;
				}
			}
		}
		worker_settings.append(item);
	}
	snapshot[settings_key] = std::move(worker_settings);
	return snapshot;
}

static bool PythonObjectsEqual(const py::handle &lhs, const py::handle &rhs) {
	int compare_result = PyObject_RichCompareBool(lhs.ptr(), rhs.ptr(), Py_EQ);
	if (compare_result < 0) {
		throw py::error_already_set();
	}
	return compare_result == 1;
}

static string BootstrapDatabasePath(const py::object &bootstrap_obj) {
	if (bootstrap_obj.is_none() || !py::isinstance<py::dict>(bootstrap_obj)) {
		return ":memory:";
	}
	auto bootstrap = bootstrap_obj.cast<py::dict>();
	if (!bootstrap.contains(py::str("database")) || bootstrap[py::str("database")].is_none()) {
		return ":memory:";
	}
	return py::str(bootstrap[py::str("database")]).cast<string>();
}

static bool BootstrapUsesInMemoryDatabase(const py::object &bootstrap_obj) {
	auto database = BootstrapDatabasePath(bootstrap_obj);
	return duckdb::DBConfig::IsInMemoryDatabase(database.c_str());
}

static bool SnapshotHasAttachedDatabases(const py::object &snapshot_obj) {
	if (snapshot_obj.is_none() || !py::isinstance<py::dict>(snapshot_obj)) {
		return false;
	}
	auto snapshot = snapshot_obj.cast<py::dict>();
	if (!snapshot.contains(py::str("attached_databases"))) {
		return false;
	}
	auto attached_obj = snapshot[py::str("attached_databases")];
	if (attached_obj.is_none()) {
		return false;
	}
	if (!py::isinstance<py::list>(attached_obj)) {
		throw InvalidInputException("Connection snapshot attached_databases must be a list");
	}
	return py::len(attached_obj) > 0;
}

static py::object CreateConnectionFromBootstrapSnapshot(const py::object &bootstrap_obj, bool use_instance_cache = true,
                                                        bool force_file_read_only = false,
                                                        bool remove_worker_resource_settings = false) {
	if (IsDefaultBootstrapSnapshot(bootstrap_obj)) {
		py::dict config;
		config[py::str("allow_persistent_secrets")] = py::str("false");
		auto connection = use_instance_cache ? DuckDBPyConnection::Connect(py::str(":memory:"), false, config)
		                                     : DuckDBPyConnection::ConnectUncached(py::str(":memory:"), false, config);
		return py::cast(std::move(connection));
	}

	auto bootstrap = bootstrap_obj.cast<py::dict>();
	auto database = BootstrapDatabasePath(bootstrap_obj);
	auto in_memory_database = duckdb::DBConfig::IsInMemoryDatabase(database.c_str());

	bool source_read_only = false;
	if (bootstrap.contains(py::str("read_only")) && !bootstrap[py::str("read_only")].is_none()) {
		source_read_only = bootstrap[py::str("read_only")].cast<bool>();
	}

	py::dict bootstrap_config = py::dict();
	if (bootstrap.contains(py::str("config")) && !bootstrap[py::str("config")].is_none() &&
	    py::isinstance<py::dict>(bootstrap[py::str("config")])) {
		bootstrap_config = CopyPyDict(bootstrap[py::str("config")].cast<py::dict>());
	}
	const auto disable_persistent_secrets = !use_instance_cache || in_memory_database;
	auto connection_config =
	    SanitizeBootstrapConfig(bootstrap_config, disable_persistent_secrets, remove_worker_resource_settings);
	auto worker_file_read_only = force_file_read_only && !in_memory_database;
	if (worker_file_read_only) {
		connection_config = ForceReadOnlyAccessMode(connection_config);
	}
	auto connection_read_only = source_read_only || worker_file_read_only;
	auto connection =
	    use_instance_cache
	        ? DuckDBPyConnection::Connect(py::str(database), connection_read_only, connection_config)
	        : DuckDBPyConnection::ConnectUncached(py::str(database), connection_read_only, connection_config);
	// Keep the source bootstrap identity for connection matching even though
	// isolated-instance security, actor resources, and worker file access are forced.
	connection->SetConnectionBootstrapConfig(database, source_read_only, bootstrap_config);
	return py::cast(std::move(connection));
}

static py::object CreateSnapshotBaselineConnection(DuckDBPyConnection &source_conn, const py::object &bootstrap_obj) {
	if (BootstrapUsesInMemoryDatabase(bootstrap_obj)) {
		return CreateConnectionFromBootstrapSnapshot(bootstrap_obj);
	}
	// A fresh cursor preserves the existing file-database baseline: database-
	// global settings remain defaults while connection-local overrides differ.
	// It also avoids reopening a live file with sanitized bootstrap settings.
	return py::cast(source_conn.Cursor());
}

static bool ConnectionMatchesBootstrapSnapshot(py::object conn_obj, const py::object &snapshot_obj) {
	auto bootstrap_obj = LookupBootstrapSnapshot(snapshot_obj);
	if (bootstrap_obj.is_none() || IsDefaultBootstrapSnapshot(bootstrap_obj) || conn_obj.is_none()) {
		return true;
	}
	auto actual_bootstrap = ExtractPyConnectionWrapper(conn_obj).ExportConnectionBootstrapConfig();
	auto normalized_required = NormalizeBootstrapSnapshot(bootstrap_obj.cast<py::dict>());
	return PythonObjectsEqual(actual_bootstrap, normalized_required);
}

static bool ConnectionsShareDatabaseInstance(const py::object &lhs_obj, const py::object &rhs_obj) {
	if (lhs_obj.is_none() || rhs_obj.is_none()) {
		return false;
	}
	auto &lhs_wrapper = ExtractPyConnectionWrapper(lhs_obj);
	auto &rhs_wrapper = ExtractPyConnectionWrapper(rhs_obj);
	if (lhs_wrapper.con.ConnectionIsClosed() || rhs_wrapper.con.ConnectionIsClosed()) {
		return false;
	}
	auto &lhs = lhs_wrapper.con.GetConnection();
	auto &rhs = rhs_wrapper.con.GetConnection();
	return &duckdb::DatabaseInstance::GetDatabase(*lhs.context) == &duckdb::DatabaseInstance::GetDatabase(*rhs.context);
}

static py::object ResolveConnectionForSnapshot(py::object conn_obj, const py::object &snapshot_obj) {
	auto bootstrap_obj = LookupBootstrapSnapshot(snapshot_obj);
	if (bootstrap_obj.is_none() || IsDefaultBootstrapSnapshot(bootstrap_obj)) {
		return conn_obj;
	}
	if (!conn_obj.is_none() && ConnectionMatchesBootstrapSnapshot(conn_obj, snapshot_obj)) {
		return conn_obj;
	}
	return CreateConnectionFromBootstrapSnapshot(bootstrap_obj);
}

static py::object ResolvePlanningConnectionForSnapshot(py::object conn_obj, const py::object &source_conn_obj,
                                                       const py::object &snapshot_obj) {
	auto bootstrap_obj = LookupBootstrapSnapshot(snapshot_obj);
	if (source_conn_obj.is_none()) {
		// A transported logical plan must not inherit database-global state from
		// the caller's planning connection, including temporary or persistent
		// secrets that are intentionally absent from the snapshot.
		return CreateConnectionFromBootstrapSnapshot(bootstrap_obj, false);
	}
	if (SnapshotHasAttachedDatabases(snapshot_obj)) {
		if (ConnectionMatchesBootstrapSnapshot(source_conn_obj, snapshot_obj)) {
			// Local execution can keep using the source DatabaseInstance where the
			// attached catalog already exists.
			return py::cast(ExtractPyConnectionWrapper(source_conn_obj).Cursor());
		}
		return CreateConnectionFromBootstrapSnapshot(bootstrap_obj, false);
	}
	if (bootstrap_obj.is_none() || IsDefaultBootstrapSnapshot(bootstrap_obj) ||
	    ConnectionMatchesBootstrapSnapshot(conn_obj, snapshot_obj)) {
		return conn_obj;
	}
	if (!BootstrapUsesInMemoryDatabase(bootstrap_obj) &&
	    ConnectionMatchesBootstrapSnapshot(source_conn_obj, snapshot_obj)) {
		// The source DatabaseInstance may still be alive. Reopening its file with
		// the sanitized worker configuration violates DuckDB's instance cache;
		// a cursor shares that instance without running database initialization.
		return py::cast(ExtractPyConnectionWrapper(source_conn_obj).Cursor());
	}
	return ResolveConnectionForSnapshot(conn_obj, snapshot_obj);
}

struct QueryPythonReplayState {
	string session_id;
	duckdb::distributed::python::ray::SafePyObject session_config;
	duckdb::distributed::python::ray::SafePyObject udf_registrations;
	duckdb::distributed::python::ray::SafePyObject udf_actor_handles;
	duckdb::distributed::python::ray::SafePyObject connection_snapshot;
	vector<ScopedSecretRef> scoped_secret_refs;

	QueryPythonReplayState(string session_id_p, py::object session_config_p, py::object udf_registrations_p,
	                       py::object udf_actor_handles_p, py::object connection_snapshot_p,
	                       vector<ScopedSecretRef> scoped_secret_refs_p)
	    : session_id(std::move(session_id_p)),
	      session_config(duckdb::distributed::python::ray::SafePyObject(std::move(session_config_p))),
	      udf_registrations(duckdb::distributed::python::ray::SafePyObject(std::move(udf_registrations_p))),
	      udf_actor_handles(duckdb::distributed::python::ray::SafePyObject(std::move(udf_actor_handles_p))),
	      connection_snapshot(duckdb::distributed::python::ray::SafePyObject(std::move(connection_snapshot_p))),
	      scoped_secret_refs(std::move(scoped_secret_refs_p)) {
	}
};

static std::mutex g_query_python_replay_states_lock;
static std::unordered_map<string, std::unique_ptr<QueryPythonReplayState>> g_query_python_replay_states;

struct ConnectionSettingRecord {
	string name;
	string value;
	string input_type;
	string scope;
};

static void EnforceExtensionSecuritySettings(duckdb::Connection &conn) {
	// Snapshot replay can run with a query-local result collector installed, so
	// update the configuration without executing SET statements.
	auto &config = duckdb::DBConfig::GetConfig(*conn.context);
	config.SetOptionByName("allow_unsigned_extensions", duckdb::Value::BOOLEAN(false));
	config.SetOptionByName("autoinstall_known_extensions", duckdb::Value::BOOLEAN(false));
	config.SetOptionByName("autoload_known_extensions", duckdb::Value::BOOLEAN(false));
}

static bool ShouldSkipConnectionSettingSnapshot(const string &name, const string &input_type) {
	auto lower_name = duckdb::StringUtil::Lower(name);
	auto upper_input_type = duckdb::StringUtil::Upper(input_type);
	if (lower_name == "duckdb_api" || IsExtensionSecuritySetting(lower_name) ||
	    IsSecretPersistenceSetting(lower_name)) {
		return true;
	}
	if (upper_input_type.find('[') != string::npos) {
		return true;
	}
	return false;
}

static bool IsBooleanConnectionSettingType(const string &input_type) {
	return duckdb::StringUtil::Upper(input_type) == "BOOLEAN";
}

static bool IsNumericConnectionSettingType(const string &input_type) {
	static const std::unordered_set<string> numeric_types = {
	    "TINYINT",   "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT",
	    "USMALLINT", "UINTEGER", "UBIGINT", "FLOAT",  "DOUBLE",  "DECIMAL",
	};
	return numeric_types.find(duckdb::StringUtil::Upper(input_type)) != numeric_types.end();
}

static bool IsVaneSessionBaselineConnectionSetting(const string &name) {
	static const std::unordered_set<string> names {
	    "http_keep_alive",  "http_retries", "http_retry_backoff", "http_retry_wait_ms",
	    "s3_access_key_id", "s3_endpoint",  "s3_region",          "s3_secret_access_key",
	    "s3_session_token", "s3_url_style", "s3_use_ssl",
	};
	return names.find(duckdb::StringUtil::Lower(name)) != names.end();
}

static bool IsS3CredentialConnectionSetting(const string &name) {
	static const std::unordered_set<string> names {
	    "s3_access_key_id",
	    "s3_secret_access_key",
	    "s3_session_token",
	};
	return names.find(duckdb::StringUtil::Lower(name)) != names.end();
}

static string QuoteSQLStringLiteral(const string &value) {
	return "'" + duckdb::StringUtil::Replace(value, "'", "''") + "'";
}

static duckdb::unique_ptr<duckdb::MaterializedQueryResult> ExecuteSnapshotQuery(duckdb::Connection &conn,
                                                                                const string &sql) {
	auto result = conn.Query(sql);
	if (!result) {
		throw duckdb::InternalException("Connection snapshot query returned a null result");
	}
	if (result->HasError()) {
		// Snapshot statements can contain access keys, secret values, or catalog
		// attachment options. DuckDB diagnostics may echo the input line, so only
		// retain the non-sensitive error category in exceptions that can reach
		// worker logs.
		throw duckdb::InvalidInputException("Connection snapshot query failed (" +
		                                    duckdb::Exception::ExceptionTypeToString(result->GetErrorType()) + ")");
	}
	return result;
}

static string StripS3EndpointSchemeForDuckDB(const string &endpoint_url) {
	auto scheme_pos = endpoint_url.find("://");
	if (scheme_pos == string::npos) {
		return endpoint_url;
	}
	return endpoint_url.substr(scheme_pos + 3);
}

static bool S3EndpointUsesSSL(const string &endpoint_url) {
	return duckdb::StringUtil::StartsWith(endpoint_url, "https://");
}

static void ConfigureConnectionForS3Endpoint(duckdb::Connection &conn, const string &endpoint_url,
                                             const string &access_key, const string &secret_key, const string &region) {
	ExecuteSnapshotQuery(conn, "LOAD httpfs");

	const auto endpoint = StripS3EndpointSchemeForDuckDB(endpoint_url);
	const auto use_ssl = S3EndpointUsesSSL(endpoint_url);
	const auto resolved_region = region.empty() ? string("us-east-1") : region;

	ExecuteSnapshotQuery(conn, "SET GLOBAL s3_region=" + QuoteSQLStringLiteral(resolved_region));
	ExecuteSnapshotQuery(conn, "SET GLOBAL s3_access_key_id=" + QuoteSQLStringLiteral(access_key));
	ExecuteSnapshotQuery(conn, "SET GLOBAL s3_secret_access_key=" + QuoteSQLStringLiteral(secret_key));
	ExecuteSnapshotQuery(conn, "SET GLOBAL s3_endpoint=" + QuoteSQLStringLiteral(endpoint));
	ExecuteSnapshotQuery(conn, string("SET GLOBAL s3_use_ssl=") + (use_ssl ? "true" : "false"));
	ExecuteSnapshotQuery(conn, "SET GLOBAL s3_url_style='path'");
	ExecuteSnapshotQuery(conn, "SET GLOBAL http_keep_alive=true");
	ExecuteSnapshotQuery(conn, "SET GLOBAL http_retries=10");
	ExecuteSnapshotQuery(conn, "SET GLOBAL http_retry_wait_ms=100");
	ExecuteSnapshotQuery(conn, "SET GLOBAL http_retry_backoff=1.5");
	ExecuteSnapshotQuery(conn, "CREATE SECRET IF NOT EXISTS __vane_s3_test ("
	                           "TYPE S3, "
	                           "KEY_ID " +
	                               QuoteSQLStringLiteral(access_key) +
	                               ", "
	                               "SECRET " +
	                               QuoteSQLStringLiteral(secret_key) +
	                               ", "
	                               "ENDPOINT " +
	                               QuoteSQLStringLiteral(endpoint) +
	                               ", "
	                               "REGION " +
	                               QuoteSQLStringLiteral(resolved_region) +
	                               ", "
	                               "USE_SSL " +
	                               string(use_ssl ? "true" : "false") +
	                               ", "
	                               "URL_STYLE 'path')");
}

static string VaneSessionConfigValue(const py::dict &config, const char *key) {
	auto py_key = py::str(key);
	if (!config.contains(py_key) || config[py_key].is_none()) {
		return string();
	}
	return py::str(config[py_key]).cast<string>();
}

static void ApplyVaneSessionConfigValues(duckdb::Connection &conn, const py::dict &config) {
	auto endpoint_url = VaneSessionConfigValue(config, "AWS_ENDPOINT_URL");
	auto access_key = VaneSessionConfigValue(config, "AWS_ACCESS_KEY_ID");
	auto secret_key = VaneSessionConfigValue(config, "AWS_SECRET_ACCESS_KEY");
	auto session_token = VaneSessionConfigValue(config, "AWS_SESSION_TOKEN");
	auto region = VaneSessionConfigValue(config, "AWS_REGION");
	if (region.empty()) {
		region = VaneSessionConfigValue(config, "AWS_DEFAULT_REGION");
	}
	if (endpoint_url.empty() && access_key.empty() && secret_key.empty() && session_token.empty() && region.empty()) {
		return;
	}

	ExecuteSnapshotQuery(conn, "LOAD httpfs");
	if (!region.empty()) {
		ExecuteSnapshotQuery(conn, "SET s3_region=" + QuoteSQLStringLiteral(region));
	}
	if (!access_key.empty()) {
		ExecuteSnapshotQuery(conn, "SET s3_access_key_id=" + QuoteSQLStringLiteral(access_key));
	}
	if (!secret_key.empty()) {
		ExecuteSnapshotQuery(conn, "SET s3_secret_access_key=" + QuoteSQLStringLiteral(secret_key));
	}
	if (!access_key.empty() || !secret_key.empty() || !session_token.empty()) {
		ExecuteSnapshotQuery(conn, "SET s3_session_token=" + QuoteSQLStringLiteral(session_token));
	}
	if (!endpoint_url.empty()) {
		ExecuteSnapshotQuery(conn,
		                     "SET s3_endpoint=" + QuoteSQLStringLiteral(StripS3EndpointSchemeForDuckDB(endpoint_url)));
		ExecuteSnapshotQuery(conn, string("SET s3_use_ssl=") + (S3EndpointUsesSSL(endpoint_url) ? "true" : "false"));
		ExecuteSnapshotQuery(conn, "SET s3_url_style='path'");
	}
	ExecuteSnapshotQuery(conn, "SET http_keep_alive=true");
	ExecuteSnapshotQuery(conn, "SET http_retries=10");
	ExecuteSnapshotQuery(conn, "SET http_retry_wait_ms=100");
	ExecuteSnapshotQuery(conn, "SET http_retry_backoff=1.5");
}

static void ApplyVaneSessionConfig(duckdb::Connection &conn, const py::object &snapshot_obj) {
	ApplyVaneSessionConfigValues(conn, VaneSessionConfigFromSnapshot(snapshot_obj));
}

static void CloseOpenPythonConnectionResult(DuckDBPyConnection &conn_wrapper) {
	if (conn_wrapper.con.HasResult()) {
		// A partially consumed StreamQueryResult keeps ClientContext::active_query
		// alive, and StreamQueryResult::Close only drops its weak context handle.
		// Starting a materialized no-op query runs DuckDB's InitialCleanup before
		// snapshot replay enters RunFunctionInTransaction directly.
		(void)ExecuteSnapshotQuery(conn_wrapper.con.GetConnection(), "SELECT NULL WHERE false");
		conn_wrapper.con.GetResult().Close();
	}
	conn_wrapper.con.SetResult(nullptr);
}

static void ApplyEffectiveVaneSessionConfig(DuckDBPyConnection &conn_wrapper, const py::object &config_obj) {
	CloseOpenPythonConnectionResult(conn_wrapper);
	if (config_obj.is_none()) {
		return;
	}
	if (!py::isinstance<py::dict>(config_obj)) {
		throw duckdb::InvalidInputException("Effective Vane session config must be a dict");
	}
	ApplyVaneSessionConfigValues(conn_wrapper.con.GetConnection(), config_obj.cast<py::dict>());
}

static std::vector<string> QueryLoadedNonStaticExtensionNames(DuckDBPyConnection &conn_wrapper) {
	std::vector<string> extensions;
	auto result =
	    ExecuteSnapshotQuery(conn_wrapper.con.GetConnection(), "SELECT extension_name "
	                                                           "FROM duckdb_extensions() "
	                                                           "WHERE loaded AND install_mode <> 'STATICALLY_LINKED' "
	                                                           "ORDER BY extension_name");
	auto &collection = result->Collection();
	extensions.reserve(collection.Count());
	for (auto &row : collection.Rows()) {
		auto value = row.GetValue(0);
		if (value.IsNull()) {
			continue;
		}
		auto extension_name = value.ToString();
		if (!extension_name.empty()) {
			extensions.push_back(std::move(extension_name));
		}
	}
	return extensions;
}

struct StaticExtensionSnapshotEntry {
	string name;
	string version;

	bool operator==(const StaticExtensionSnapshotEntry &other) const {
		return name == other.name && version == other.version;
	}

	bool operator<(const StaticExtensionSnapshotEntry &other) const {
		return name < other.name;
	}
};

static vector<StaticExtensionSnapshotEntry> QueryLoadedStaticExtensions(DuckDBPyConnection &conn_wrapper) {
	vector<StaticExtensionSnapshotEntry> extensions;
	auto result =
	    ExecuteSnapshotQuery(conn_wrapper.con.GetConnection(), "SELECT extension_name, extension_version "
	                                                           "FROM duckdb_extensions() "
	                                                           "WHERE loaded AND install_mode = 'STATICALLY_LINKED' "
	                                                           "ORDER BY extension_name");
	auto &collection = result->Collection();
	extensions.reserve(collection.Count());
	for (auto &row : collection.Rows()) {
		auto name_value = row.GetValue(0);
		if (name_value.IsNull()) {
			continue;
		}
		auto extension_name = name_value.ToString();
		if (!extension_name.empty()) {
			auto version_value = row.GetValue(1);
			extensions.push_back(
			    {std::move(extension_name), version_value.IsNull() ? string() : version_value.ToString()});
		}
	}
	return extensions;
}

static std::vector<ConnectionSettingRecord> QueryConnectionSettings(DuckDBPyConnection &conn_wrapper) {
	std::vector<ConnectionSettingRecord> settings;
	auto result = ExecuteSnapshotQuery(conn_wrapper.con.GetConnection(), "SELECT name, value, input_type, scope "
	                                                                     "FROM duckdb_settings() "
	                                                                     "ORDER BY name");
	auto &collection = result->Collection();
	settings.reserve(collection.Count());
	for (auto &row : collection.Rows()) {
		ConnectionSettingRecord record;
		auto name_val = row.GetValue(0);
		auto value_val = row.GetValue(1);
		auto input_type_val = row.GetValue(2);
		auto scope_val = row.GetValue(3);
		if (name_val.IsNull() || input_type_val.IsNull() || scope_val.IsNull()) {
			continue;
		}
		record.name = name_val.ToString();
		record.value = value_val.IsNull() ? string() : value_val.ToString();
		record.input_type = input_type_val.ToString();
		record.scope = scope_val.ToString();
		settings.push_back(std::move(record));
	}
	return settings;
}

static void RejectNonStaticRayExtensions(const std::vector<string> &extension_names) {
	if (extension_names.empty()) {
		return;
	}
	auto joined_names = extension_names.front();
	for (idx_t index = 1; index < extension_names.size(); index++) {
		joined_names += ", " + extension_names[index];
	}
	throw duckdb::InvalidInputException("Ray distributed execution supports only statically linked extensions; "
	                                    "non-static extensions are not supported: "
	                                    "%s",
	                                    joined_names);
}

static bool IsSafeStaticExtensionName(const string &name) {
	if (name.empty()) {
		return false;
	}
	for (auto character : name) {
		if ((character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
		    (character >= '0' && character <= '9') || character == '_') {
			continue;
		}
		return false;
	}
	return true;
}

static string StaticExtensionListIdentity(const vector<StaticExtensionSnapshotEntry> &extensions) {
	vector<string> identities;
	identities.reserve(extensions.size());
	for (const auto &extension : extensions) {
		identities.push_back(extension.name + "@" + extension.version);
	}
	return "[" + StringUtil::Join(identities, ",") + "]";
}

static vector<StaticExtensionSnapshotEntry> LoadedStaticExtensions(DatabaseInstance &database) {
	vector<StaticExtensionSnapshotEntry> extensions;
	auto &manager = ExtensionManager::Get(database);
	for (const auto &extension_name : manager.GetExtensions()) {
		auto info = manager.GetExtensionInfo(extension_name);
		if (!info) {
			continue;
		}
		lock_guard<mutex> guard(info->lock);
		if (!info->is_loaded || !info->install_info ||
		    info->install_info->mode != ExtensionInstallMode::STATICALLY_LINKED) {
			continue;
		}
		extensions.push_back({extension_name, info->install_info->version});
	}
	std::sort(extensions.begin(), extensions.end());
	return extensions;
}

static void LoadStaticRayExtensions(duckdb::Connection &conn, const vector<StaticExtensionSnapshotEntry> &extensions) {
	auto &database = duckdb::DatabaseInstance::GetDatabase(*conn.context);
	duckdb::DuckDB db(database);
	for (const auto &extension : extensions) {
		if (!IsSafeStaticExtensionName(extension.name)) {
			throw duckdb::InvalidInputException("Invalid static extension name in connection snapshot: %s",
			                                    extension.name);
		}
		auto linked_name = duckdb::StringUtil::Lower(extension.name);
		if (!duckdb::ExtensionHelper::IsExtensionLinked(linked_name)) {
			throw duckdb::InvalidInputException("Ray distributed execution supports only statically linked extensions; "
			                                    "extension '%s' is not statically "
			                                    "linked into this worker",
			                                    extension.name);
		}
		if (duckdb::ExtensionHelper::LoadExtension(db, linked_name) != duckdb::ExtensionLoadResult::LOADED_EXTENSION) {
			throw duckdb::InvalidInputException("Ray distributed execution supports only statically linked extensions; "
			                                    "extension '%s' is not statically "
			                                    "linked into this worker",
			                                    extension.name);
		}
	}
	auto loaded_extensions = LoadedStaticExtensions(database);
	if (loaded_extensions != extensions) {
		throw InvalidInputException(
		    "Static extension identities differ between coordinator and worker: expected %s, worker loaded %s",
		    StaticExtensionListIdentity(extensions), StaticExtensionListIdentity(loaded_extensions));
	}
}

static py::list CaptureDistributedExtensionContracts(DatabaseInstance &database) {
	py::list result;
	for (const auto &identity : DistributedExtensionManager::Get(database).GetContractIdentities()) {
		result.append(py::str(identity));
	}
	return result;
}

static vector<string> ParseDistributedExtensionContracts(const py::dict &snapshot) {
	auto key = py::str("distributed_extension_contracts");
	if (!snapshot.contains(key) || !py::isinstance<py::list>(snapshot[key])) {
		throw InvalidInputException("Connection snapshot distributed_extension_contracts must be a list");
	}
	vector<string> result;
	set<string> identities;
	for (auto item : snapshot[key].cast<py::list>()) {
		if (!py::isinstance<py::str>(item)) {
			throw InvalidInputException("Connection snapshot distributed extension contract must be a string");
		}
		auto identity = item.cast<string>();
		if (identity.empty() || !identities.insert(identity).second) {
			throw InvalidInputException(
			    "Connection snapshot distributed extension contracts must be non-empty and unique");
		}
		result.push_back(std::move(identity));
	}
	std::sort(result.begin(), result.end());
	return result;
}

static py::list CaptureAttachedDatabaseSnapshot(DuckDBPyConnection &conn_wrapper) {
	py::list attached_obj;
	auto &context = *conn_wrapper.con.GetConnection().context;
	auto databases = DatabaseManager::Get(context).GetDatabases(context);
	for (auto &database : databases) {
		if (database->IsSystem() || database->IsTemporary() || database->IsInitialDatabase() ||
		    database->GetVisibility() == AttachVisibility::HIDDEN) {
			continue;
		}

		auto &catalog = database->GetCatalog();
		auto options = database->GetAttachOptions();
		options["type"] = Value(catalog.GetCatalogType());
		if (database->IsReadOnly()) {
			options["read_only"] = Value::BOOLEAN(true);
		}
		if (database->GetRecoveryMode() != RecoveryMode::DEFAULT) {
			options["recovery_mode"] = Value(EnumUtil::ToString(database->GetRecoveryMode()));
		}

		vector<string> option_names;
		option_names.reserve(options.size());
		for (const auto &entry : options) {
			option_names.push_back(entry.first);
		}
		std::sort(option_names.begin(), option_names.end());

		vector<string> serialized_options;
		serialized_options.reserve(option_names.size());
		for (const auto &option_name : option_names) {
			serialized_options.push_back(option_name + " " + options.at(option_name).ToSQLString());
		}

		string attach_sql = "ATTACH DATABASE " + KeywordHelper::WriteQuoted(catalog.GetDBPath(), '\'') + " AS " +
		                    KeywordHelper::WriteOptionallyQuoted(database->GetName());
		if (!serialized_options.empty()) {
			attach_sql += " (" + StringUtil::Join(serialized_options, ", ") + ")";
		}
		attached_obj.append(py::str(attach_sql));
	}
	return attached_obj;
}

static void ApplyAttachedDatabaseSnapshot(duckdb::Connection &conn, const py::dict &snapshot) {
	if (!snapshot.contains(py::str("attached_databases"))) {
		return;
	}
	auto attached_obj = snapshot[py::str("attached_databases")];
	if (attached_obj.is_none()) {
		return;
	}
	if (!py::isinstance<py::list>(attached_obj)) {
		throw duckdb::InvalidInputException("Connection snapshot attached_databases must be a list");
	}
	for (auto item : attached_obj.cast<py::list>()) {
		if (!py::isinstance<py::str>(item)) {
			throw duckdb::InvalidInputException("Connection snapshot attached database entry must be SQL text");
		}
		ExecuteSnapshotQuery(conn, py::str(item).cast<string>());
	}
}

static bool VaneRaySessionLifecycleEnabled() {
	auto native_module = py::module_::import("vane._native");
	auto runner = py::str(native_module.attr("get_or_infer_runner_type")()).cast<string>();
	duckdb::StringUtil::Trim(runner);
	runner = duckdb::StringUtil::Lower(runner);
	return runner == "ray";
}

static py::object CaptureConnectionSnapshot(DuckDBPyConnection &conn_wrapper) {
	auto bootstrap_obj = conn_wrapper.ExportConnectionBootstrapConfig();
	auto non_static_extensions = QueryLoadedNonStaticExtensionNames(conn_wrapper);
	RejectNonStaticRayExtensions(non_static_extensions);
	auto static_extensions = QueryLoadedStaticExtensions(conn_wrapper);
	auto source_settings = QueryConnectionSettings(conn_wrapper);

	auto default_conn_obj = CreateSnapshotBaselineConnection(conn_wrapper, bootstrap_obj);
	auto &default_conn = ExtractPyConnectionWrapper(default_conn_obj);
	auto default_settings = QueryConnectionSettings(default_conn);
	std::unordered_map<string, string> default_setting_values;
	default_setting_values.reserve(default_settings.size());
	for (const auto &record : default_settings) {
		default_setting_values[duckdb::StringUtil::Lower(record.name)] = record.value;
	}

	py::list settings_obj;
	for (const auto &record : source_settings) {
		if (ShouldSkipConnectionSettingSnapshot(record.name, record.input_type)) {
			continue;
		}
		auto lower_name = duckdb::StringUtil::Lower(record.name);
		auto entry = default_setting_values.find(lower_name);
		auto explicitly_local_session_override =
		    duckdb::StringUtil::Lower(record.scope) == "local" && IsVaneSessionBaselineConnectionSetting(record.name);
		if (!explicitly_local_session_override && entry != default_setting_values.end() &&
		    entry->second == record.value) {
			continue;
		}
		py::dict setting_obj;
		setting_obj[py::str("name")] = py::str(record.name);
		setting_obj[py::str("value")] = py::str(record.value);
		setting_obj[py::str("input_type")] = py::str(record.input_type);
		settings_obj.append(std::move(setting_obj));
	}

	bool has_bootstrap = !IsDefaultBootstrapSnapshot(bootstrap_obj);
	py::dict snapshot_obj;
	py::dict session_obj;
	session_obj[py::str("id")] = py::str(conn_wrapper.GetVaneSessionId());
	session_obj[py::str("config")] = conn_wrapper.ExportVaneSessionConfig();
	snapshot_obj[py::str("vane_session")] = std::move(session_obj);
	if (has_bootstrap) {
		snapshot_obj[py::str("bootstrap")] = NormalizeBootstrapSnapshot(bootstrap_obj);
	}
	snapshot_obj[py::str("duckdb_source_id")] = py::str(DuckDB::SourceID());
	py::list extensions_obj;
	for (const auto &extension : static_extensions) {
		py::dict extension_obj;
		extension_obj[py::str("name")] = py::str(extension.name);
		extension_obj[py::str("version")] = py::str(extension.version);
		extensions_obj.append(std::move(extension_obj));
	}
	snapshot_obj[py::str("extensions")] = std::move(extensions_obj);
	auto &source_database = DatabaseInstance::GetDatabase(*conn_wrapper.con.GetConnection().context);
	snapshot_obj[py::str("distributed_extension_contracts")] = CaptureDistributedExtensionContracts(source_database);
	snapshot_obj[py::str("settings")] = std::move(settings_obj);
	snapshot_obj[py::str("attached_databases")] = CaptureAttachedDatabaseSnapshot(conn_wrapper);
	if (VaneRaySessionLifecycleEnabled()) {
		conn_wrapper.MarkVaneRaySessionOpened();
	}
	return snapshot_obj;
}

static PyLogicalPlan LogicalPlanFromDuckDBRelation(py::object relation_obj, py::object query_id_obj,
                                                   bool require_auto_commit) {
	if (!py::isinstance<duckdb::DuckDBPyRelation>(relation_obj)) {
		throw py::type_error("Expected a Vane DuckDBPyRelation object");
	}
	auto &pyrel = relation_obj.cast<duckdb::DuckDBPyRelation &>();
	auto rel = pyrel.GetRelation();
	if (!rel) {
		throw duckdb::InternalException("Relation is null");
	}
	const bool is_write_relation =
	    rel->type == RelationType::CREATE_TABLE_RELATION || rel->type == RelationType::INSERT_RELATION ||
	    rel->type == RelationType::DELETE_RELATION || rel->type == RelationType::UPDATE_RELATION ||
	    rel->type == RelationType::WRITE_FILE_RELATION;
	if (require_auto_commit) {
		if (!rel->context) {
			throw duckdb::InternalException("Cannot validate distributed write transaction: relation has no context");
		}
		auto client_context = rel->context->GetContext();
		if (!client_context->transaction.IsAutoCommit()) {
			throw duckdb::InvalidInputException(
			    "distributed writes require DuckDB auto-commit mode and cannot participate in an explicit transaction");
		}
		if (!is_write_relation) {
			throw duckdb::InvalidInputException("from_duckdb_write_relation requires a write relation");
		}
	} else if (is_write_relation) {
		throw duckdb::InvalidInputException(
		    "from_duckdb_relation does not accept write relations; use from_duckdb_write_relation");
	}

	PyLogicalPlan plan;
	plan.query_id_ = query_id_obj.is_none() ? string() : py::cast<string>(query_id_obj);
	auto connection_owner = pyrel.GetConnectionOwner();
	string owner_session_id;
	if (connection_owner && !connection_owner.is_none() && py::isinstance<DuckDBPyConnection>(connection_owner)) {
		auto &conn_wrapper = connection_owner.cast<DuckDBPyConnection &>();
		plan.source_connection_ = connection_owner;
		owner_session_id = conn_wrapper.GetVaneSessionId();
	}
	auto serialized_capture = SerializeLogicalPlanFromRelation(rel, plan.query_id_, owner_session_id);
	plan.serialized_logical_plan_ = std::move(serialized_capture.serialized_plan);
	plan.scoped_secret_refs_ = std::move(serialized_capture.scoped_secrets.references);
	plan.source_scoped_secret_bindings_ = std::move(serialized_capture.scoped_secrets.source_bindings);
	plan.source_unmatched_scoped_secret_uses_ = std::move(serialized_capture.scoped_secrets.source_unmatched_uses);

	if (!plan.source_connection_.is_none()) {
		auto &conn_wrapper = plan.source_connection_.cast<DuckDBPyConnection &>();
		auto registrations = conn_wrapper.ExportDistributedPythonUDFRegistrations();
		if (py::len(registrations) > 0) {
			plan.udf_registrations_ = std::move(registrations);
		}
		plan.connection_snapshot_ = CaptureConnectionSnapshot(conn_wrapper);
	}
	return plan;
}

struct ConnectionSnapshotApplyOptions {
	bool apply_session_config = true;
	bool enforce_extension_security = true;
	bool apply_s3_credentials = true;
	bool apply_settings = true;
	bool apply_attached_databases = false;
};

static void ApplyConnectionSnapshot(py::object conn_obj, const py::object &snapshot_obj,
                                    const ConnectionSnapshotApplyOptions &options);

static bool ConnectionSnapshotDeclaresStaticExtension(const py::object &snapshot_obj, const string &extension_name) {
	if (snapshot_obj.is_none() || !py::isinstance<py::dict>(snapshot_obj)) {
		return false;
	}
	auto snapshot = snapshot_obj.cast<py::dict>();
	auto extensions_key = py::str("extensions");
	if (!snapshot.contains(extensions_key) || !py::isinstance<py::list>(snapshot[extensions_key])) {
		return false;
	}
	auto expected_name = StringUtil::Lower(extension_name);
	for (auto item : snapshot[extensions_key].cast<py::list>()) {
		if (!py::isinstance<py::dict>(item)) {
			continue;
		}
		auto extension = py::reinterpret_borrow<py::dict>(item);
		auto name_key = py::str("name");
		if (extension.contains(name_key) && py::isinstance<py::str>(extension[name_key]) &&
		    StringUtil::Lower(extension[name_key].cast<string>()) == expected_name) {
			return true;
		}
	}
	return false;
}

static void ValidateConnectionSnapshotExtensions(py::object conn_obj, const py::object &snapshot_obj,
                                                 bool enforce_extension_security) {
	ConnectionSnapshotApplyOptions validation_options;
	validation_options.apply_session_config = false;
	validation_options.enforce_extension_security = enforce_extension_security;
	validation_options.apply_s3_credentials = false;
	validation_options.apply_settings = false;
	validation_options.apply_attached_databases = false;
	ApplyConnectionSnapshot(conn_obj, snapshot_obj, validation_options);
}

static void ApplyConnectionSnapshot(py::object conn_obj, const py::object &snapshot_obj,
                                    const ConnectionSnapshotApplyOptions &options = {}) {
	if (snapshot_obj.is_none()) {
		return;
	}
	if (!py::isinstance<py::dict>(snapshot_obj)) {
		throw duckdb::InvalidInputException("Connection snapshot must be a dict");
	}

	auto snapshot = snapshot_obj.cast<py::dict>();
	if (!snapshot.contains(py::str("duckdb_source_id")) ||
	    !py::isinstance<py::str>(snapshot[py::str("duckdb_source_id")])) {
		throw InvalidInputException("Connection snapshot is missing duckdb_source_id");
	}
	auto expected_source_id = snapshot[py::str("duckdb_source_id")].cast<string>();
	if (expected_source_id != DuckDB::SourceID()) {
		throw InvalidInputException(
		    "DuckDB SourceID differs between coordinator and worker: expected %s, worker has %s", expected_source_id,
		    DuckDB::SourceID());
	}
	if (!snapshot.contains(py::str("extensions")) || !py::isinstance<py::list>(snapshot[py::str("extensions")])) {
		throw InvalidInputException("Connection snapshot extensions must be a list");
	}
	vector<StaticExtensionSnapshotEntry> extensions;
	set<string> extension_names;
	for (auto item : snapshot[py::str("extensions")].cast<py::list>()) {
		if (!py::isinstance<py::dict>(item)) {
			throw InvalidInputException("Connection snapshot extension entry must be a dict");
		}
		auto extension_obj = py::reinterpret_borrow<py::dict>(item);
		if (!extension_obj.contains(py::str("name")) || !py::isinstance<py::str>(extension_obj[py::str("name")]) ||
		    !extension_obj.contains(py::str("version")) ||
		    !py::isinstance<py::str>(extension_obj[py::str("version")])) {
			throw InvalidInputException("Connection snapshot extension entry is missing string name or version");
		}
		StaticExtensionSnapshotEntry extension;
		extension.name = extension_obj[py::str("name")].cast<string>();
		extension.version = extension_obj[py::str("version")].cast<string>();
		if (extension.name.empty() || !extension_names.insert(extension.name).second) {
			throw InvalidInputException("Connection snapshot has an empty or duplicate extension name");
		}
		extensions.push_back(std::move(extension));
	}
	std::sort(extensions.begin(), extensions.end());
	auto distributed_extension_contracts = ParseDistributedExtensionContracts(snapshot);
	auto &conn_wrapper = ExtractPyConnectionWrapper(conn_obj);
	// Snapshot replay starts a new unit of work on this Python cursor. Close a
	// partially consumed DB-API result before mutating connection state.
	CloseOpenPythonConnectionResult(conn_wrapper);
	auto &conn = conn_wrapper.con.GetConnection();
	if (options.enforce_extension_security) {
		// Distributed snapshot replay never inherits settings that permit
		// runtime downloads or unsigned extension binaries.
		EnforceExtensionSecuritySettings(conn);
	}
	LoadStaticRayExtensions(conn, extensions);
	DistributedExtensionManager::Get(DatabaseInstance::GetDatabase(*conn.context))
	    .ValidateExact(distributed_extension_contracts);

	if (options.apply_session_config) {
		ApplyVaneSessionConfig(conn, snapshot_obj);
	}

	if (options.apply_settings && snapshot.contains(py::str("settings"))) {
		auto settings_obj = snapshot[py::str("settings")];
		if (!settings_obj.is_none() && py::isinstance<py::list>(settings_obj)) {
			for (auto item : settings_obj.cast<py::list>()) {
				if (!py::isinstance<py::dict>(item)) {
					continue;
				}
				auto setting_obj = py::reinterpret_borrow<py::dict>(item);
				if (!setting_obj.contains(py::str("name")) || !setting_obj.contains(py::str("value"))) {
					continue;
				}
				auto setting_name = py::str(setting_obj[py::str("name")]).cast<string>();
				auto setting_value = py::str(setting_obj[py::str("value")]).cast<string>();
				auto input_type = setting_obj.contains(py::str("input_type"))
				                      ? py::str(setting_obj[py::str("input_type")]).cast<string>()
				                      : string("VARCHAR");
				auto lower_setting_name = duckdb::StringUtil::Lower(setting_name);
				if (setting_name.empty() || IsExtensionSecuritySetting(lower_setting_name) ||
				    IsSecretPersistenceSetting(lower_setting_name) ||
				    (!options.apply_s3_credentials && IsS3CredentialConnectionSetting(lower_setting_name))) {
					continue;
				}
				string sql_value;
				if (IsBooleanConnectionSettingType(input_type) || IsNumericConnectionSettingType(input_type)) {
					sql_value = setting_value;
				} else {
					sql_value = QuoteSQLStringLiteral(setting_value);
				}
				ExecuteSnapshotQuery(conn, "SET " + setting_name + " = " + sql_value);
			}
		}
	}

	if (options.apply_attached_databases) {
		ApplyAttachedDatabaseSnapshot(conn, snapshot);
	}
}

string PyLogicalPlan::session_id() const {
	return VaneSessionIdFromSnapshot(connection_snapshot_);
}

py::dict PyLogicalPlan::session_config() const {
	return VaneSessionConfigFromSnapshot(connection_snapshot_);
}

bool PyLogicalPlan::has_explicit_s3_credentials() const {
	return HasExplicitS3CredentialsFromSnapshot(connection_snapshot_);
}

enum class QueryPythonReplayField : uint8_t {
	UDFRegistrations,
	UDFActorHandles,
	ConnectionSnapshot,
	ScopedSecretRefs,
};

static py::object LookupQueryPythonReplayState(const string &query_id, QueryPythonReplayField field) {
	if (query_id.empty()) {
		return py::none();
	}
	std::lock_guard<std::mutex> guard(g_query_python_replay_states_lock);
	auto entry = g_query_python_replay_states.find(query_id);
	if (entry == g_query_python_replay_states.end()) {
		return py::none();
	}
	switch (field) {
	case QueryPythonReplayField::UDFRegistrations:
		return entry->second->udf_registrations.get();
	case QueryPythonReplayField::UDFActorHandles:
		return entry->second->udf_actor_handles.get();
	case QueryPythonReplayField::ConnectionSnapshot:
		return entry->second->connection_snapshot.get();
	case QueryPythonReplayField::ScopedSecretRefs:
		return ScopedSecretRefsToPython(entry->second->scoped_secret_refs);
	default:
		throw duckdb::InternalException("Unknown query Python replay field");
	}
}

static bool RegisterQueryPythonReplayState(const string &query_id, const py::object &udf_registrations,
                                           const py::object &udf_actor_handles, const py::object &connection_snapshot,
                                           const vector<ScopedSecretRef> &scoped_secret_refs) {
	if (query_id.empty()) {
		throw duckdb::InternalException("Query Python replay state requires a non-empty query_id");
	}
	auto session_id = VaneSessionIdFromSnapshot(connection_snapshot);
	py::object session_config = VaneSessionConfigFromSnapshot(connection_snapshot);
	ValidateScopedSecretRefs(scoped_secret_refs, query_id, session_id, "query replay registration");
	std::lock_guard<std::mutex> guard(g_query_python_replay_states_lock);
	auto entry = g_query_python_replay_states.find(query_id);
	if (entry == g_query_python_replay_states.end()) {
		g_query_python_replay_states.emplace(
		    query_id, std::make_unique<QueryPythonReplayState>(std::move(session_id), std::move(session_config),
		                                                       py::reinterpret_borrow<py::object>(udf_registrations),
		                                                       py::reinterpret_borrow<py::object>(udf_actor_handles),
		                                                       py::reinterpret_borrow<py::object>(connection_snapshot),
		                                                       scoped_secret_refs));
		return true;
	}
	auto &state = *entry->second;
	if (state.session_id != session_id || !PythonObjectsEqual(state.session_config.get(), session_config)) {
		throw duckdb::InvalidInputException("Query " + query_id + " was registered with a different Vane session");
	}
	if (!PythonObjectsEqual(state.connection_snapshot.get(), connection_snapshot)) {
		throw duckdb::InvalidInputException("Query " + query_id +
		                                    " was registered with a different connection snapshot");
	}
	if (!PythonObjectsEqual(state.udf_registrations.get(), udf_registrations)) {
		throw duckdb::InvalidInputException("Query " + query_id +
		                                    " was registered with different Python UDF registrations");
	}
	if (!PythonObjectsEqual(state.udf_actor_handles.get(), udf_actor_handles)) {
		throw duckdb::InvalidInputException("Query " + query_id +
		                                    " was registered with different Python UDF actor handles");
	}
	if (!ScopedSecretRefsEqual(state.scoped_secret_refs, scoped_secret_refs)) {
		throw duckdb::InvalidInputException("Query " + query_id +
		                                    " was registered with different scoped secret references");
	}
	return false;
}

static py::object LookupQueryUDFRegistrations(const string &query_id) {
	return LookupQueryPythonReplayState(query_id, QueryPythonReplayField::UDFRegistrations);
}

static py::object LookupQueryConnectionSnapshot(const string &query_id) {
	return LookupQueryPythonReplayState(query_id, QueryPythonReplayField::ConnectionSnapshot);
}

static py::object LookupQueryUDFActorHandles(const string &query_id) {
	return LookupQueryPythonReplayState(query_id, QueryPythonReplayField::UDFActorHandles);
}

static py::object LookupQueryScopedSecretRefs(const string &query_id) {
	return LookupQueryPythonReplayState(query_id, QueryPythonReplayField::ScopedSecretRefs);
}

static void CleanupQueryPythonReplayState(const string &query_id) {
	if (query_id.empty()) {
		return;
	}
	std::lock_guard<std::mutex> guard(g_query_python_replay_states_lock);
	g_query_python_replay_states.erase(query_id);
}

static void CleanupAllQueryPythonReplayState() {
	std::lock_guard<std::mutex> guard(g_query_python_replay_states_lock);
	g_query_python_replay_states.clear();
}

static duckdb::unique_ptr<duckdb::LogicalOperator>
RebindAndOptimizeDeserializedLogicalPlan(duckdb::ClientContext &context,
                                         duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan) {
	auto logical_plan_stmt = duckdb::make_uniq<duckdb::LogicalPlanStatement>(std::move(logical_plan));
	duckdb::Planner planner(context);
	planner.CreatePlan(std::move(logical_plan_stmt));
	if (!planner.plan) {
		throw duckdb::InternalException("Planner failed to create logical plan from deserialized LogicalPlanStatement");
	}

	auto rebound_plan = std::move(planner.plan);
	auto &client_config = duckdb::ClientConfig::GetConfig(context);
	if (client_config.enable_optimizer && rebound_plan->RequireOptimizer()) {
		duckdb::Optimizer optimizer(*planner.binder, context);
		rebound_plan = optimizer.Optimize(std::move(rebound_plan));
	}
	return rebound_plan;
}

static duckdb::distributed::DistributedPipelineNodeRef
BuildDistributedPipelineNode(const std::shared_ptr<duckdb::distributed::DistributedPhysicalPlan> &plan,
                             duckdb::ClientContext *client_context = nullptr) {
	using namespace duckdb::distributed;
	if (!plan) {
		throw duckdb::InternalException("DistributedPhysicalPlan is null");
	}
	auto physical_plan = plan->physical_plan();
	if (!physical_plan) {
		throw duckdb::InternalException("DistributedPhysicalPlan has no physical plan");
	}
	if (!physical_plan->HasRoot()) {
		throw duckdb::InternalException("DistributedPhysicalPlan physical plan has no root");
	}
	PlanConfig cfg(plan->idx(), plan->query_id(), plan->execution_config());
	if (client_context && client_context->db) {
		cfg.db = client_context->db;
	}
	auto pipeline_res = physical_plan_to_pipeline_node(std::move(cfg), std::move(physical_plan), client_context);
	if (!pipeline_res.is_ok()) {
		if (pipeline_res.error().type() == DuckDBError::Type::ValueError) {
			throw duckdb::InvalidInputException("Ray runner cannot execute this query: %s",
			                                    pipeline_res.error().what());
		}
		throw duckdb::InternalException(string("Failed to build distributed pipeline node: ") +
		                                pipeline_res.error().what());
	}
	if (!pipeline_res.value()) {
		throw duckdb::InternalException("Distributed pipeline translation returned null root node");
	}
	return pipeline_res.value();
}

static const UDFFunctionData *TryGetUDFBindData(const FunctionData *bind_data) {
	// FunctionData::Cast<T>() only asserts its dynamic type in debug builds and
	// becomes a reinterpret_cast in Release. Generic INOUT functions (for
	// example UNNEST) have unrelated bind data and must not be treated as UDFs.
	return dynamic_cast<const UDFFunctionData *>(bind_data);
}

static const UDFFunctionData *TryGetUDFBindData(const PhysicalTableInOutFunction &inout) {
	return TryGetUDFBindData(inout.GetBindData());
}

static const UDFFunctionData *TryGetUDFBindData(const PhysicalStreamingUDF &streaming) {
	return TryGetUDFBindData(streaming.GetBindData());
}

static UDFFunctionData *TryGetMutableUDFBindData(PhysicalOperator &op) {
	const UDFFunctionData *bind_data = nullptr;
	if (op.type == PhysicalOperatorType::INOUT_FUNCTION) {
		bind_data = TryGetUDFBindData(op.Cast<PhysicalTableInOutFunction>());
	} else if (op.type == PhysicalOperatorType::STREAMING_UDF) {
		bind_data = TryGetUDFBindData(op.Cast<PhysicalStreamingUDF>());
	}
	return const_cast<UDFFunctionData *>(bind_data);
}

static void CollectMutableUDFBindData(PhysicalOperator &op, vector<UDFFunctionData *> &out) {
	if (auto *bind_data = TryGetMutableUDFBindData(op)) {
		out.push_back(bind_data);
	}
}

static duckdb::shared_ptr<void> WrapPyObjectForUDFActorHandles(const py::object &obj) {
	if (obj.is_none()) {
		return nullptr;
	}
	auto *boxed = new py::object(py::reinterpret_borrow<py::object>(obj));
	return duckdb::shared_ptr<void>(boxed, [](void *ptr) {
		if (!ptr) {
			return;
		}
		auto *boxed_obj = static_cast<py::object *>(ptr);
		if (!Py_IsInitialized() || PythonIsFinalizing()) {
			boxed_obj->release();
			delete boxed_obj;
			return;
		}
		PythonGILWrapper gil;
		delete boxed_obj;
	});
}
