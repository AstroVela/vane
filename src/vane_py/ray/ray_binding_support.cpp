// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "ray_binding_internal.hpp"

namespace duckdb {
namespace ray_bindings {
namespace py = ::duckdb::py;

PhysicalNonSerializableOperatorForTest::PhysicalNonSerializableOperatorForTest(PhysicalPlan &physical_plan)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::DUMMY_SCAN, {LogicalType::INTEGER}, 1) {
}

auto PhysicalNonSerializableOperatorForTest::GetName() const -> string {
	return "INTENTIONALLY_NON_SERIALIZABLE";
}

auto PhysicalNonSerializableOperatorForTest::SerializeOperatorData(Serializer &) const -> void {
	throw NotImplementedException("INTENTIONALLY_NON_SERIALIZABLE operator cannot be serialized");
}

unique_ptr<DistributedWriteGlobalState>
CoordinatorOnlyWriteInitializeGlobalForTest(ClientContext &, const DistributedExtensionWriteInfo &,
                                            const DistributedWriteTaskContext &) {
	return make_uniq<CoordinatorOnlyWriteGlobalStateForTest>();
}

unique_ptr<DistributedWriteLocalState> CoordinatorOnlyWriteInitializeLocalForTest(ExecutionContext &,
                                                                                  const DistributedExtensionWriteInfo &,
                                                                                  const DistributedWriteTaskContext &,
                                                                                  DistributedWriteGlobalState &) {
	return make_uniq<CoordinatorOnlyWriteLocalStateForTest>();
}

void CoordinatorOnlyWriteSinkForTest(ExecutionContext &, const DistributedExtensionWriteInfo &,
                                     const DistributedWriteTaskContext &, DistributedWriteGlobalState &,
                                     DistributedWriteLocalState &, DataChunk &) {
}

void CoordinatorOnlyWriteCombineForTest(ExecutionContext &, const DistributedExtensionWriteInfo &,
                                        const DistributedWriteTaskContext &, DistributedWriteGlobalState &,
                                        DistributedWriteLocalState &) {
}

vector<DistributedWriteFragment> CoordinatorOnlyWriteFinalizeForTest(ClientContext &,
                                                                     const DistributedExtensionWriteInfo &,
                                                                     const DistributedWriteTaskContext &,
                                                                     DistributedWriteGlobalState &) {
	return {};
}

void RegisterCoordinatorOnlyExtensionWriteForTest(ClientContext &context) {
	DistributedExtensionManifest manifest;
	manifest.extension_name = "vane_test";
	manifest.capabilities.push_back({DistributedExtensionCapabilityKind::WRITE_OPERATOR, "coordinator_only_write", 1});

	DistributedWriteOperatorExtension write_operator;
	write_operator.name = "coordinator_only_write";
	write_operator.protocol_version = 1;
	write_operator.mode = DistributedWriteMode::CALLBACK_SINK;
	write_operator.fragment_codec = {"vane_test.coordinator_only_write", 1};
	write_operator.callbacks.initialize_global = CoordinatorOnlyWriteInitializeGlobalForTest;
	write_operator.callbacks.initialize_local = CoordinatorOnlyWriteInitializeLocalForTest;
	write_operator.callbacks.sink = CoordinatorOnlyWriteSinkForTest;
	write_operator.callbacks.combine = CoordinatorOnlyWriteCombineForTest;
	write_operator.callbacks.finalize = CoordinatorOnlyWriteFinalizeForTest;

	DistributedExtensionManager::Get(context).RegisterExtension(
	    manifest, {make_shared_ptr<const DistributedWriteOperatorExtension>(std::move(write_operator))});
}

PhysicalCoordinatorOnlyExtensionWriteForTest::PhysicalCoordinatorOnlyExtensionWriteForTest(PhysicalPlan &physical_plan,
                                                                                           bool fail_finalize_p)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, {LogicalType::BIGINT}, 1),
      fail_finalize(fail_finalize_p) {
	plan.extension_name = "vane_test";
	plan.operator_name = "coordinator_only_write";
}

auto PhysicalCoordinatorOnlyExtensionWriteForTest::GetExtensionWriteTaskProvider()
    -> optional_ptr<distributed::ExtensionWriteTaskProvider> {
	return this;
}

auto PhysicalCoordinatorOnlyExtensionWriteForTest::WritePlan() const
    -> const distributed::DistributedExtensionWritePlan & {
	return plan;
}

auto PhysicalCoordinatorOnlyExtensionWriteForTest::ValidateDistributedWrite(ClientContext &) const -> void {
}

auto PhysicalCoordinatorOnlyExtensionWriteForTest::FinalizeDistributedWrite(
    ClientContext &, const vector<DistributedWriteTaskResult> &results) const -> idx_t {
	if (fail_finalize) {
		throw IOException("planned coordinator-only extension finalization failure");
	}
	idx_t rows = 0;
	for (const auto &result : results) {
		rows += result.RowCount();
	}
	return rows;
}

auto PhysicalCoordinatorOnlyExtensionWriteForTest::AbortDistributedWrite(
    ClientContext &, const vector<DistributedWriteTaskResult> &) const -> void {
}

auto PhysicalCoordinatorOnlyExtensionWriteForTest::SerializeOperatorData(Serializer &) const -> void {
	throw NotImplementedException("COORDINATOR_ONLY_EXTENSION_WRITE root cannot be serialized");
}

CountingResultCollectorForTest::CountingResultCollectorForTest() : calls(make_shared_ptr<std::atomic<idx_t>>(0)) {
}

auto CountingResultCollectorForTest::operator()(ClientContext &context, PreparedStatementData &data) const
    -> unique_ptr<PhysicalOperator> {
	calls->fetch_add(1, std::memory_order_relaxed);
	return PhysicalResultCollector::GetResultCollector(context, data);
}

ScopedShuffleCacheRegistrationForTest::ScopedShuffleCacheRegistrationForTest(string exchange_id)
    : exchange_id_(std::move(exchange_id)) {
}

ScopedShuffleCacheRegistrationForTest::~ScopedShuffleCacheRegistrationForTest() {
	distributed::ShuffleCacheRegistry::Instance().Remove(exchange_id_);
}

py::object ResolveFlightShuffleCleanupConnection(py::object cleanup_connection, const py::object &connection_snapshot,
                                                 const py::object &effective_session_config,
                                                 bool apply_snapshot_s3_credentials) {
	auto resolved_connection = connection_snapshot.is_none()
	                               ? cleanup_connection
	                               : ResolveConnectionForSnapshot(cleanup_connection, connection_snapshot);
	auto &resolved_wrapper = ExtractPyConnectionWrapper(resolved_connection);
	if (resolved_wrapper.con.ConnectionIsClosed()) {
		throw std::runtime_error("resolved shuffle cleanup connection is closed");
	}
	ConnectionSnapshotApplyOptions snapshot_options;
	snapshot_options.apply_session_config = false;
	snapshot_options.apply_s3_credentials = apply_snapshot_s3_credentials;
	ValidateConnectionSnapshotExtensions(resolved_connection, connection_snapshot,
	                                     snapshot_options.enforce_extension_security);
	if (ConnectionSnapshotDeclaresExtension(connection_snapshot, "httpfs")) {
		ApplyEffectiveVaneSessionConfig(resolved_wrapper, effective_session_config);
	}
	ApplyConnectionSnapshot(resolved_connection, connection_snapshot, snapshot_options);
	return resolved_connection;
}

string QueryConnectionSettingForTest(DuckDBPyConnection &connection, const string &name) {
	auto result = ExecuteSnapshotQuery(connection.con.GetConnection(),
	                                   "SELECT current_setting(" + QuoteSQLStringLiteral(name) + ")");
	for (auto &row : result->Collection().Rows()) {
		auto value = row.GetValue(0);
		return value.IsNull() ? string() : value.ToString();
	}
	throw std::runtime_error("connection setting query returned no rows: " + name);
}

py::list ReadIntegerExchangeSourceForTest(duckdb::ClientContext &context,
                                          duckdb::distributed::FlightExchangeConfig config,
                                          duckdb::distributed::ExchangeSourceHandle handle) {
	using namespace duckdb::distributed;
	config.expected_types = {LogicalType::INTEGER};
	FlightExchangeSource source(config, &context);
	source.AddSourceHandles({std::move(handle)});
	vector<LogicalType> output_types = {LogicalType::INTEGER};
	DataChunk output;
	output.Initialize(Allocator::DefaultAllocator(), output_types);
	py::list values;
	while (source.ReadChunk(output)) {
		for (idx_t row = 0; row < output.size(); row++) {
			values.append(output.GetValue(0, row).GetValue<int32_t>());
		}
	}
	source.Close();
	return values;
}

FakeObjectShuffleStorage::FakeObjectShuffleStorage(FileSystem &fs) : fs_(fs) {
}

auto FakeObjectShuffleStorage::CreateDirectories(const std::string &path) const -> DuckDBResult<void> {
	directories.insert(path);
	return DuckDBResult<void>::ok();
}

auto FakeObjectShuffleStorage::IsRegularFile(const std::string &path) const -> bool {
	if (text_files.find(path) != text_files.end()) {
		return true;
	}
	try {
		return fs_.FileExists(path);
	} catch (...) {
		return false;
	}
}

auto FakeObjectShuffleStorage::FileSize(const std::string &path) const -> DuckDBResult<idx_t> {
	auto text_entry = text_files.find(path);
	if (text_entry != text_files.end()) {
		return DuckDBResult<idx_t>::ok(text_entry->second.size());
	}
	try {
		auto handle = fs_.OpenFile(path, FileFlags::FILE_FLAGS_READ);
		auto size = handle->GetFileSize();
		if (size > std::numeric_limits<idx_t>::max()) {
			return DuckDBResult<idx_t>::err(
			    DuckDBError::io_error("fake object storage file size exceeds idx_t max: " + path));
		}
		return DuckDBResult<idx_t>::ok(static_cast<idx_t>(size));
	} catch (const std::exception &ex) {
		return DuckDBResult<idx_t>::err(
		    DuckDBError::io_error("fake object storage file stat failed: " + std::string(ex.what())));
	} catch (...) {
		return DuckDBResult<idx_t>::err(DuckDBError::io_error("fake object storage file stat failed: " + path));
	}
}

auto FakeObjectShuffleStorage::WriteTextFileAtomically(const std::string &path, const std::string &contents) const
    -> DuckDBResult<void> {
	text_files[path] = contents;
	text_puts++;
	return DuckDBResult<void>::ok();
}

auto FakeObjectShuffleStorage::ReadTextFile(const std::string &path) const -> DuckDBResult<std::string> {
	auto entry = text_files.find(path);
	if (entry == text_files.end()) {
		return DuckDBResult<std::string>::err(DuckDBError::io_error("fake object storage text file missing: " + path));
	}
	return DuckDBResult<std::string>::ok(entry->second);
}

auto FakeObjectShuffleStorage::RemoveAll(const std::string &path) const -> DuckDBResult<idx_t> {
	idx_t removed = 0;
	auto prefix = path;
	if (!prefix.empty() && prefix.back() != '/') {
		prefix += "/";
	}
	for (auto it = text_files.begin(); it != text_files.end();) {
		if (it->first == path || it->first.rfind(prefix, 0) == 0) {
			it = text_files.erase(it);
			removed++;
		} else {
			++it;
		}
	}
	RemoveDistributedCopyDirectoryTree(fs_, path);
	return DuckDBResult<idx_t>::ok(removed);
}

auto FakeObjectShuffleStorage::OpenArrowOutput(const std::string &path) const
    -> DuckDBResult<std::shared_ptr<arrow::io::OutputStream>> {
	(void)path;
	return DuckDBResult<std::shared_ptr<arrow::io::OutputStream>>::err(
	    DuckDBError::invalid_state_error("fake object storage arrow output is not implemented"));
}

auto FakeObjectShuffleStorage::OpenArrowInput(const std::string &path) const
    -> DuckDBResult<std::shared_ptr<arrow::io::InputStream>> {
	(void)path;
	return DuckDBResult<std::shared_ptr<arrow::io::InputStream>>::err(
	    DuckDBError::invalid_state_error("fake object storage arrow input is not implemented"));
}

auto FakeObjectShuffleStorage::HasTextFile(const std::string &path) const -> bool {
	return text_files.find(path) != text_files.end();
}

auto FakeObjectShuffleStorage::TextFile(const std::string &path) const -> std::string {
	auto entry = text_files.find(path);
	return entry == text_files.end() ? std::string() : entry->second;
}
} // namespace ray_bindings
} // namespace duckdb
