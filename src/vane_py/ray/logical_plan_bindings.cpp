// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Included by ray_module.cpp inside namespace duckdb.

struct PyPhysicalPlanWrapper;

static constexpr char LOGICAL_PLAN_ENVELOPE_MAGIC[] = {'V', 'A', 'N', 'E', 'P', 'L', 'A', 'N'};
static constexpr idx_t LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE = sizeof(LOGICAL_PLAN_ENVELOPE_MAGIC);
static constexpr uint32_t LOGICAL_PLAN_PROTOCOL_VERSION = 1;
static constexpr uint32_t LOGICAL_PLAN_MAX_SOURCE_ID_SIZE = 4096;

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

	PyLogicalPlan() = default;

	string idx() const {
		return query_id_;
	}

	string session_id() const;
	py::dict session_config() const;
	bool has_explicit_s3_credentials() const;

	PyPhysicalPlanWrapper to_physical_plan(py::object conn_obj, py::object effective_session_config) const;
};

static string SerializeLogicalPlanFromRelation(const duckdb::shared_ptr<duckdb::Relation> &rel) {
	if (!rel) {
		throw duckdb::InternalException("Relation is null");
	}
	auto client_context = rel->context->GetContext();
	string serialized_plan;
	client_context->RunFunctionInTransaction([&]() {
		auto statement_binder = duckdb::Binder::CreateBinder(*client_context);
		auto relation_stmt = make_uniq<duckdb::RelationStatement>(rel, *statement_binder);
		duckdb::Planner planner(*client_context);
		planner.CreatePlan(std::move(relation_stmt));
		auto logical_plan = std::move(planner.plan);

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
		serializer.GetSerializationData().Set<duckdb::ClientContext &>(*client_context);
		serializer.Begin();
		logical_plan->Serialize(serializer);
		serializer.End();

		auto data_ptr = stream.GetData();
		auto data_size = stream.GetPosition();
		if (data_size == 0) {
			throw duckdb::InternalException("Logical plan serialization returned empty payload");
		}
		auto logical_payload = string(reinterpret_cast<const char *>(data_ptr), data_size);
		serialized_plan = EncodeLogicalPlanEnvelope(logical_payload);
	});
	return serialized_plan;
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

static bool IsWorkerExtensionLocationSetting(const string &lower_name) {
	return lower_name == "extension_directory" || lower_name == "extension_directories" ||
	       lower_name == "home_directory" || lower_name == "custom_extension_repository" ||
	       lower_name == "autoinstall_extension_repository";
}

static bool IsWorkerLocalSetting(const string &lower_name) {
	return IsWorkerResourceSetting(lower_name) || IsWorkerExtensionLocationSetting(lower_name);
}

static py::dict RemoveWorkerLocalBootstrapSettings(const py::dict &config) {
	py::dict worker_config;
	for (auto item : config) {
		auto name = duckdb::StringUtil::Lower(py::str(item.first).cast<string>());
		if (!IsWorkerLocalSetting(name)) {
			worker_config[item.first] = item.second;
		}
	}
	return worker_config;
}

static py::dict SanitizeBootstrapConfig(const py::dict &config, bool disable_persistent_secrets,
                                        bool remove_worker_local_settings = false) {
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
		if (remove_worker_local_settings && IsWorkerLocalSetting(name)) {
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
	auto bootstrap_key = py::str("bootstrap");
	if (snapshot.contains(bootstrap_key) && py::isinstance<py::dict>(snapshot[bootstrap_key])) {
		auto bootstrap = CopyPyDict(snapshot[bootstrap_key].cast<py::dict>());
		auto config_key = py::str("config");
		if (bootstrap.contains(config_key) && py::isinstance<py::dict>(bootstrap[config_key])) {
			bootstrap[config_key] = RemoveWorkerLocalBootstrapSettings(bootstrap[config_key].cast<py::dict>());
		}
		snapshot[bootstrap_key] = std::move(bootstrap);
	}
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
				if (IsWorkerLocalSetting(name)) {
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

static bool SnapshotHasDynamicExtensions(const py::object &snapshot_obj) {
	if (snapshot_obj.is_none() || !py::isinstance<py::dict>(snapshot_obj)) {
		return false;
	}
	auto snapshot = snapshot_obj.cast<py::dict>();
	auto key = py::str("dynamic_extensions");
	return snapshot.contains(key) && py::isinstance<py::list>(snapshot[key]) && py::len(snapshot[key]) > 0;
}

static py::object CreateConnectionFromBootstrapSnapshot(const py::object &bootstrap_obj, bool use_instance_cache = true,
                                                        bool force_file_read_only = false,
                                                        bool remove_worker_local_settings = false) {
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
	    SanitizeBootstrapConfig(bootstrap_config, disable_persistent_secrets, remove_worker_local_settings);
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
	if (SnapshotHasAttachedDatabases(snapshot_obj) || SnapshotHasDynamicExtensions(snapshot_obj)) {
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
	// Driver-only state. Fragment wrappers take isolated cursors from this
	// connection, and DistributedPhysicalPlan pickle state never includes it.
	duckdb::distributed::python::ray::SafePyObject coordinator_connection;

	QueryPythonReplayState(string session_id_p, py::object session_config_p, py::object udf_registrations_p,
	                       py::object udf_actor_handles_p, py::object connection_snapshot_p,
	                       py::object coordinator_connection_p)
	    : session_id(std::move(session_id_p)),
	      session_config(duckdb::distributed::python::ray::SafePyObject(std::move(session_config_p))),
	      udf_registrations(duckdb::distributed::python::ray::SafePyObject(std::move(udf_registrations_p))),
	      udf_actor_handles(duckdb::distributed::python::ray::SafePyObject(std::move(udf_actor_handles_p))),
	      connection_snapshot(duckdb::distributed::python::ray::SafePyObject(std::move(connection_snapshot_p))),
	      coordinator_connection(duckdb::distributed::python::ray::SafePyObject(std::move(coordinator_connection_p))) {
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

	// Every caller establishes and validates the snapshot's extension set before
	// applying connection-local S3 settings. Never turn configuration replay
	// into another extension-loading path.
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

static vector<string> LoadedNonStaticExtensionNames(DuckDBPyConnection &conn_wrapper) {
	vector<string> extensions;
	auto &database = DatabaseInstance::GetDatabase(*conn_wrapper.con.GetConnection().context);
	auto &manager = ExtensionManager::Get(database);
	for (const auto &extension_name : manager.GetExtensions()) {
		auto info = manager.GetExtensionInfo(extension_name);
		if (!info) {
			continue;
		}
		lock_guard<mutex> guard(info->lock);
		if (!info->is_loaded) {
			continue;
		}
		// A loaded extension without install provenance is not demonstrably
		// static. Classify it as dynamic so snapshot capture/admission fails
		// closed unless resolver-owned state declares the same name.
		if (info->install_info && info->install_info->mode == ExtensionInstallMode::STATICALLY_LINKED) {
			continue;
		}
		extensions.push_back(extension_name);
	}
	std::sort(extensions.begin(), extensions.end());
	return extensions;
}

static py::list NormalizeDynamicExtensionSnapshot(const py::object &snapshot_obj) {
	try {
		auto extensions_module = py::module_::import("vane.extensions");
		auto normalized = extensions_module.attr("_normalize_dynamic_extension_snapshot")(snapshot_obj);
		if (!py::isinstance<py::list>(normalized)) {
			throw duckdb::InternalException("Dynamic extension snapshot normalizer did not return a list");
		}
		return normalized.cast<py::list>();
	} catch (const py::error_already_set &exception) {
		throw duckdb::InvalidInputException("Failed to validate dynamic extension snapshot: %s", exception.what());
	}
}

static py::list CaptureDynamicExtensionSnapshot(const py::object &conn_obj) {
	try {
		auto extensions_module = py::module_::import("vane.extensions");
		auto snapshot = extensions_module.attr("_capture_dynamic_extension_snapshot")(conn_obj);
		if (!py::isinstance<py::list>(snapshot)) {
			throw duckdb::InternalException("Dynamic extension snapshot capture did not return a list");
		}
		return snapshot.cast<py::list>();
	} catch (const py::error_already_set &exception) {
		throw duckdb::InvalidInputException("Failed to capture dynamic extension snapshot: %s", exception.what());
	}
}

static vector<string> DynamicExtensionNamesFromSnapshot(const py::list &dynamic_extensions,
                                                        const string &snapshot_description) {
	vector<string> names;
	set<string> unique_names;
	for (auto item : dynamic_extensions) {
		if (!py::isinstance<py::dict>(item)) {
			throw duckdb::InvalidInputException("%s dynamic extension entry must be a descriptor dict",
			                                    snapshot_description);
		}
		auto descriptor = py::reinterpret_borrow<py::dict>(item);
		auto name_key = py::str("name");
		if (!descriptor.contains(name_key) || !py::isinstance<py::str>(descriptor[name_key])) {
			throw duckdb::InvalidInputException("%s dynamic extension descriptor is missing a string name",
			                                    snapshot_description);
		}
		auto name = descriptor[name_key].cast<string>();
		if (name.empty() || !unique_names.insert(name).second) {
			throw duckdb::InvalidInputException("%s has an empty or duplicate dynamic extension name",
			                                    snapshot_description);
		}
		names.push_back(std::move(name));
	}
	std::sort(names.begin(), names.end());
	return names;
}

static string DynamicExtensionListIdentity(const vector<string> &extensions) {
	return "[" + StringUtil::Join(extensions, ",") + "]";
}

static void ValidateCapturedDynamicExtensionNames(const vector<string> &loaded_extensions,
                                                  const vector<string> &snapshot_extensions) {
	if (loaded_extensions == snapshot_extensions) {
		return;
	}
	if (snapshot_extensions.empty() && !loaded_extensions.empty()) {
		throw duckdb::InvalidInputException(
		    "Ray distributed execution rejects dynamic extensions that were not loaded through "
		    "vane.DynamicExtensionResolver: %s",
		    StringUtil::Join(loaded_extensions, ", "));
	}
	throw duckdb::InvalidInputException(
	    "Dynamic extension identities changed while capturing the connection snapshot: loaded %s, snapshot has %s",
	    DynamicExtensionListIdentity(loaded_extensions), DynamicExtensionListIdentity(snapshot_extensions));
}

static void ValidateWorkerDynamicExtensionNames(DuckDBPyConnection &conn_wrapper,
                                                const vector<string> &expected_extensions) {
	auto loaded_extensions = LoadedNonStaticExtensionNames(conn_wrapper);
	if (loaded_extensions != expected_extensions) {
		throw duckdb::InvalidInputException(
		    "Dynamic extension identities differ between coordinator and worker: expected %s, worker loaded %s",
		    DynamicExtensionListIdentity(expected_extensions), DynamicExtensionListIdentity(loaded_extensions));
	}
}

static void ValidateWorkerRecordedDynamicExtensionSnapshot(DuckDBPyConnection &conn_wrapper, const py::object &conn_obj,
                                                           const py::list &expected_extensions) {
	auto recorded_extensions = CaptureDynamicExtensionSnapshot(conn_obj);
	if (!PythonObjectsEqual(recorded_extensions, expected_extensions)) {
		throw duckdb::InvalidInputException(
		    "Worker recorded dynamic extension manifest differs from the coordinator manifest");
	}
	auto recorded_names = DynamicExtensionNamesFromSnapshot(recorded_extensions, "Worker connection dynamic state");
	auto loaded_extensions = LoadedNonStaticExtensionNames(conn_wrapper);
	if (loaded_extensions != recorded_names) {
		throw duckdb::InvalidInputException(
		    "Worker dynamic extension state is not resolver-owned: loaded %s, recorded %s",
		    DynamicExtensionListIdentity(loaded_extensions), DynamicExtensionListIdentity(recorded_names));
	}
}

static void PrepareDynamicExtensionSnapshot(py::object conn_obj, const py::list &dynamic_extensions) {
	try {
		auto extensions_module = py::module_::import("vane.extensions");
		extensions_module.attr("_prepare_dynamic_extension_snapshot")(conn_obj, dynamic_extensions);
	} catch (const py::error_already_set &exception) {
		throw duckdb::InvalidInputException("Failed to prepare dynamic extension snapshot: %s", exception.what());
	}
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

static vector<StaticExtensionSnapshotEntry> ParseStaticExtensionSnapshot(const py::dict &snapshot) {
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
	return extensions;
}

static void
ValidateDistinctStaticAndDynamicExtensionNames(const vector<StaticExtensionSnapshotEntry> &static_extensions,
                                               const vector<string> &dynamic_extension_names) {
	set<string> static_names;
	for (const auto &extension : static_extensions) {
		static_names.insert(StringUtil::Lower(extension.name));
	}
	for (const auto &name : dynamic_extension_names) {
		if (static_names.find(StringUtil::Lower(name)) != static_names.end()) {
			throw InvalidInputException("Connection snapshot declares extension '%s' as both static and dynamic", name);
		}
	}
}

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

static py::object CaptureConnectionSnapshot(DuckDBPyConnection &conn_wrapper, const py::object &conn_obj) {
	auto bootstrap_obj = conn_wrapper.ExportConnectionBootstrapConfig();
	auto non_static_extensions_before = LoadedNonStaticExtensionNames(conn_wrapper);
	auto dynamic_extensions = CaptureDynamicExtensionSnapshot(conn_obj);
	auto non_static_extensions_after = LoadedNonStaticExtensionNames(conn_wrapper);
	if (non_static_extensions_before != non_static_extensions_after) {
		throw duckdb::InvalidInputException(
		    "Dynamic extension identities changed while capturing the connection snapshot: before %s, after %s",
		    DynamicExtensionListIdentity(non_static_extensions_before),
		    DynamicExtensionListIdentity(non_static_extensions_after));
	}
	auto dynamic_extension_names =
	    DynamicExtensionNamesFromSnapshot(dynamic_extensions, "Captured connection snapshot");
	ValidateCapturedDynamicExtensionNames(non_static_extensions_after, dynamic_extension_names);
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
	snapshot_obj[py::str("dynamic_extensions")] = std::move(dynamic_extensions);
	auto &source_database = DatabaseInstance::GetDatabase(*conn_wrapper.con.GetConnection().context);
	snapshot_obj[py::str("distributed_extension_contracts")] = CaptureDistributedExtensionContracts(source_database);
	snapshot_obj[py::str("settings")] = std::move(settings_obj);
	snapshot_obj[py::str("attached_databases")] = CaptureAttachedDatabaseSnapshot(conn_wrapper);
	if (VaneRaySessionLifecycleEnabled()) {
		conn_wrapper.MarkVaneRaySessionOpened();
	}
	return snapshot_obj;
}

enum class DuckDBRelationPlanKind : uint8_t { READ, WRITE, DATA_SINK };

static PyLogicalPlan LogicalPlanFromDuckDBRelation(py::object relation_obj, py::object query_id_obj,
                                                   DuckDBRelationPlanKind plan_kind) {
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
	const bool is_datasink_relation = dynamic_cast<DataSinkRelation *>(rel.get()) != nullptr;
	if (plan_kind == DuckDBRelationPlanKind::WRITE || plan_kind == DuckDBRelationPlanKind::DATA_SINK) {
		if (!rel->context) {
			throw duckdb::InternalException(
			    "Cannot validate distributed terminal transaction: relation has no context");
		}
		auto client_context = rel->context->GetContext();
		if (!client_context->transaction.IsAutoCommit()) {
			const auto terminal_name =
			    plan_kind == DuckDBRelationPlanKind::DATA_SINK ? "DataSink writes" : "distributed writes";
			throw duckdb::InvalidInputException("%s require DuckDB auto-commit mode and cannot participate in an "
			                                    "explicit transaction",
			                                    terminal_name);
		}
	}
	if (plan_kind == DuckDBRelationPlanKind::WRITE && !is_write_relation) {
		throw duckdb::InvalidInputException("from_duckdb_write_relation requires a write relation");
	}
	if (plan_kind == DuckDBRelationPlanKind::DATA_SINK && !is_datasink_relation) {
		throw duckdb::InvalidInputException("from_duckdb_datasink_relation requires a DataSink relation");
	}
	if (plan_kind == DuckDBRelationPlanKind::READ && (is_write_relation || is_datasink_relation)) {
		throw duckdb::InvalidInputException(
		    "from_duckdb_relation does not accept terminal write relations; use the matching terminal factory");
	}

	PyLogicalPlan plan;
	plan.query_id_ = query_id_obj.is_none() ? string() : py::cast<string>(query_id_obj);
	plan.serialized_logical_plan_ = SerializeLogicalPlanFromRelation(rel);
	auto connection_owner = pyrel.GetConnectionOwner();
	if (connection_owner && !connection_owner.is_none() && py::isinstance<DuckDBPyConnection>(connection_owner)) {
		auto &conn_wrapper = connection_owner.cast<DuckDBPyConnection &>();
		plan.source_connection_ = connection_owner;
		auto registrations = conn_wrapper.ExportDistributedPythonUDFRegistrations();
		if (py::len(registrations) > 0) {
			plan.udf_registrations_ = std::move(registrations);
		}
		plan.connection_snapshot_ = CaptureConnectionSnapshot(conn_wrapper, connection_owner);
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

static bool ConnectionSnapshotDeclaresExtension(const py::object &snapshot_obj, const string &extension_name) {
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
	auto dynamic_extensions_key = py::str("dynamic_extensions");
	if (!snapshot.contains(dynamic_extensions_key) || !py::isinstance<py::list>(snapshot[dynamic_extensions_key])) {
		return false;
	}
	for (auto item : snapshot[dynamic_extensions_key].cast<py::list>()) {
		if (!py::isinstance<py::dict>(item)) {
			continue;
		}
		auto descriptor = py::reinterpret_borrow<py::dict>(item);
		auto name_key = py::str("name");
		if (descriptor.contains(name_key) && py::isinstance<py::str>(descriptor[name_key]) &&
		    StringUtil::Lower(descriptor[name_key].cast<string>()) == expected_name) {
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

static py::dict ValidateConnectionSnapshotSourceID(const py::object &snapshot_obj) {
	if (snapshot_obj.is_none() || !py::isinstance<py::dict>(snapshot_obj)) {
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
	return snapshot;
}

static void ApplyConnectionSnapshot(py::object conn_obj, const py::object &snapshot_obj,
                                    const ConnectionSnapshotApplyOptions &options = {}) {
	if (snapshot_obj.is_none()) {
		return;
	}
	auto snapshot = ValidateConnectionSnapshotSourceID(snapshot_obj);
	auto extensions = ParseStaticExtensionSnapshot(snapshot);
	if (!snapshot.contains(py::str("dynamic_extensions")) ||
	    !py::isinstance<py::list>(snapshot[py::str("dynamic_extensions")])) {
		throw InvalidInputException("Connection snapshot dynamic_extensions must be a list");
	}
	auto dynamic_extensions = NormalizeDynamicExtensionSnapshot(snapshot[py::str("dynamic_extensions")]);
	auto dynamic_extension_names = DynamicExtensionNamesFromSnapshot(dynamic_extensions, "Connection snapshot");
	ValidateDistinctStaticAndDynamicExtensionNames(extensions, dynamic_extension_names);
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
	ValidateWorkerRecordedDynamicExtensionSnapshot(conn_wrapper, conn_obj, dynamic_extensions);
	ValidateWorkerDynamicExtensionNames(conn_wrapper, dynamic_extension_names);
	LoadStaticRayExtensions(conn, extensions);
	DistributedExtensionManager::Get(DatabaseInstance::GetDatabase(*conn.context))
	    .ValidateExact(distributed_extension_contracts);

	if (options.apply_session_config && ConnectionSnapshotDeclaresExtension(snapshot_obj, "httpfs")) {
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

static void PrepareConnectionSnapshotExtensions(py::object conn_obj, const py::object &snapshot_obj) {
	auto snapshot = ValidateConnectionSnapshotSourceID(snapshot_obj);
	auto static_extensions = ParseStaticExtensionSnapshot(snapshot);
	if (!snapshot.contains(py::str("dynamic_extensions")) ||
	    !py::isinstance<py::list>(snapshot[py::str("dynamic_extensions")])) {
		throw InvalidInputException("Connection snapshot dynamic_extensions must be a list");
	}
	auto dynamic_extensions = NormalizeDynamicExtensionSnapshot(snapshot[py::str("dynamic_extensions")]);
	auto dynamic_extension_names = DynamicExtensionNamesFromSnapshot(dynamic_extensions, "Connection snapshot");
	ValidateDistinctStaticAndDynamicExtensionNames(static_extensions, dynamic_extension_names);
	(void)ParseDistributedExtensionContracts(snapshot);
	auto &conn_wrapper = ExtractPyConnectionWrapper(conn_obj);
	CloseOpenPythonConnectionResult(conn_wrapper);
	auto &conn = conn_wrapper.con.GetConnection();
	EnforceExtensionSecuritySettings(conn);
	// Preserve the established static-extension bootstrap order before invoking
	// the same resolver-driven dynamic LOAD path used by the source connection.
	LoadStaticRayExtensions(conn, static_extensions);
	PrepareDynamicExtensionSnapshot(conn_obj, dynamic_extensions);
	ValidateConnectionSnapshotExtensions(conn_obj, snapshot_obj, true);
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
	CoordinatorConnection,
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
	case QueryPythonReplayField::CoordinatorConnection:
		return entry->second->coordinator_connection.get();
	default:
		throw duckdb::InternalException("Unknown query Python replay field");
	}
}

static bool RegisterQueryPythonReplayState(const string &query_id, const py::object &udf_registrations,
                                           const py::object &udf_actor_handles, const py::object &connection_snapshot,
                                           const py::object &coordinator_connection) {
	if (query_id.empty()) {
		throw duckdb::InternalException("Query Python replay state requires a non-empty query_id");
	}
	auto session_id = VaneSessionIdFromSnapshot(connection_snapshot);
	py::object session_config = VaneSessionConfigFromSnapshot(connection_snapshot);
	py::object retained_coordinator_connection = SnapshotHasDynamicExtensions(connection_snapshot)
	                                                 ? py::reinterpret_borrow<py::object>(coordinator_connection)
	                                                 : py::none();
	std::lock_guard<std::mutex> guard(g_query_python_replay_states_lock);
	auto entry = g_query_python_replay_states.find(query_id);
	if (entry == g_query_python_replay_states.end()) {
		g_query_python_replay_states.emplace(
		    query_id,
		    std::unique_ptr<QueryPythonReplayState>(new QueryPythonReplayState(
		        std::move(session_id), std::move(session_config), py::reinterpret_borrow<py::object>(udf_registrations),
		        py::reinterpret_borrow<py::object>(udf_actor_handles),
		        py::reinterpret_borrow<py::object>(connection_snapshot), std::move(retained_coordinator_connection))));
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
	if (state.coordinator_connection.get().is_none() && !retained_coordinator_connection.is_none()) {
		state.coordinator_connection =
		    duckdb::distributed::python::ray::SafePyObject(std::move(retained_coordinator_connection));
	}
	return false;
}

static py::object LookupQueryUDFRegistrations(const string &query_id) {
	return LookupQueryPythonReplayState(query_id, QueryPythonReplayField::UDFRegistrations);
}

static py::object LookupQueryConnectionSnapshot(const string &query_id) {
	return LookupQueryPythonReplayState(query_id, QueryPythonReplayField::ConnectionSnapshot);
}

static py::object LookupQueryCoordinatorConnection(const string &query_id) {
	return LookupQueryPythonReplayState(query_id, QueryPythonReplayField::CoordinatorConnection);
}

static py::object LookupQueryUDFActorHandles(const string &query_id) {
	return LookupQueryPythonReplayState(query_id, QueryPythonReplayField::UDFActorHandles);
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
