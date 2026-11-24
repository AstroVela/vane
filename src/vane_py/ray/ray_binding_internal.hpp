// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ray_binding_includes.hpp"

namespace duckdb {
namespace ray_bindings {
namespace py = ::duckdb::py;

struct DuckDBPyTransactionGuard;
struct NativePartitionMetadata;
struct NativeDistributedTaskResult;
struct FteQueueProgressStats;
struct CopyOutputInfo;
struct ResultPartitionStream;
struct PlanRunState;
struct PyPhysicalPlanWrapper;
struct PyLogicalPlan;
struct SerializedLogicalPlanResult;
struct PythonMemorySourceCacheKey;
struct PythonMemoryScanSource;
struct PendingPythonMemoryScan;
struct PreparedPythonMemorySource;
struct PythonMemorySourceCacheKeyHash;
struct QueryPythonReplayState;
struct ConnectionSettingRecord;
struct StaticExtensionSnapshotEntry;
struct ConnectionSnapshotApplyOptions;
class PyBackendResultOperationGuard;
class PyBackendWorkerManager;
struct PyPhysicalPlanWrapperRunner;
class PhysicalNonSerializableOperatorForTest;
class CoordinatorOnlyWriteGlobalStateForTest;
class CoordinatorOnlyWriteLocalStateForTest;
class PhysicalCoordinatorOnlyExtensionWriteForTest;
class CountingResultCollectorForTest;
class ScopedShuffleCacheRegistrationForTest;
class FakeObjectShuffleStorage;

// native_fragment_executor.cpp
struct DuckDBPyTransactionGuard {
	duckdb::ClientContext &context;
	bool started = false;
	bool finished = false;

	explicit DuckDBPyTransactionGuard(duckdb::ClientContext &ctx);

	void Commit();

	void Rollback();

	~DuckDBPyTransactionGuard();
};

struct NativePartitionMetadata {
	size_t num_rows;
	size_t size_bytes;

	NativePartitionMetadata(size_t num_rows_p = 0, size_t size_bytes_p = 0);
};

struct NativeDistributedTaskResult {
	duckdb::distributed::python::ray::SafePyObject partition_payloads;
	duckdb::distributed::python::ray::SafePyObject partition_metadatas;
	duckdb::distributed::python::ray::SafePyObject result_schema;
	duckdb::distributed::python::ray::SafePyObject stats;
	duckdb::distributed::python::ray::SafePyObject task_stats;
	std::string completion_status;
	int flight_port;
	duckdb::distributed::python::ray::SafePyObject exchange_sink_instance;

	NativeDistributedTaskResult(pybind11::iterable payloads, pybind11::iterable metadatas,
	                            pybind11::object result_schema_p, pybind11::object stats_p,
	                            std::string completion_status_p, int flight_port_p = 0,
	                            pybind11::object exchange_sink_instance_p = pybind11::none(),
	                            pybind11::object task_stats_p = pybind11::none());

	pybind11::object PartitionPayloads() const;

	pybind11::object PartitionMetadatas() const;

	pybind11::object ResultSchema() const;

	pybind11::object Stats() const;

	pybind11::object TaskStats() const;

	pybind11::object ExchangeSinkInstance() const;
};

pybind11::dict BuildNativeResultSchema(const duckdb::vector<string> &names,
                                       const duckdb::vector<duckdb::LogicalType> &types);

size_t GetPyPayloadSizeBytes(const pybind11::object &payload);

pybind11::object BuildNativeTaskResult(pybind11::iterable payloads, pybind11::iterable metadatas,
                                       pybind11::object result_schema, pybind11::object stats,
                                       pybind11::object task_stats, const std::string &completion_status,
                                       int flight_port = 0, pybind11::object exchange_sink_instance = pybind11::none());

idx_t SaturatingAddIdx(idx_t lhs, idx_t rhs);

bool PipelineSnapshotSourceIs(const duckdb::PipelineProgressSnapshot &snapshot, const char *needle);

bool IsCompletionOnlyRemoteExchangeSinkPipeline(const duckdb::PipelineProgressSnapshot &snapshot);

bool IsInternalRemoteExchangeResultCollectorPipeline(const duckdb::PipelineProgressSnapshot &snapshot);

std::string PipelineSnapshotName(const duckdb::PipelineProgressSnapshot &snapshot);

pybind11::list OperatorsToPyList(const duckdb::vector<std::string> &operators);

pybind11::list
OperatorDetailsToPyList(const duckdb::vector<duckdb::InsertionOrderPreservingMap<std::string>> &operator_details);

std::string OperatorListName(const duckdb::vector<std::string> &operators);

duckdb::InsertionOrderPreservingMap<std::string> ProgressOperatorDetails(const duckdb::PhysicalOperator &op);

void CollectPrimaryPipelineOperators(
    const duckdb::PhysicalOperator &op, duckdb::vector<std::string> &operators,
    duckdb::vector<duckdb::InsertionOrderPreservingMap<std::string>> *operator_details = nullptr);

idx_t OperatorDetailIdxValue(const duckdb::InsertionOrderPreservingMap<std::string> &details, const std::string &key);

idx_t SnapshotUDFCounterMax(const duckdb::PipelineProgressSnapshot &snapshot, const std::string &key);

idx_t OperatorDetailsCounterMax(
    const duckdb::vector<duckdb::InsertionOrderPreservingMap<std::string>> &operator_details, const std::string &key);

idx_t ScanSplitInputRows(const std::unordered_map<idx_t, duckdb::distributed::ScanSplitBatch> *scan_split_batch_map);

idx_t ScanSplitInputBytes(const std::unordered_map<idx_t, duckdb::distributed::ScanSplitBatch> *scan_split_batch_map);

idx_t ExchangeSourceInputBytes(
    const std::unordered_map<idx_t, duckdb::distributed::ExchangeSourceTaskDescriptor> *exchange_source_task_map);

idx_t ExchangeSourceInputRows(
    const std::unordered_map<idx_t, duckdb::distributed::ExchangeSourceTaskDescriptor> *exchange_source_task_map);

idx_t FteQueueConsumedRows(
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *queue_map);

idx_t FteQueueConsumedBytes(
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *queue_map);

struct FteQueueProgressStats {
	idx_t submitted_split_count = 0;
	idx_t consumed_split_count = 0;
	idx_t completed_split_count = 0;
	idx_t submitted_split_bytes = 0;
	idx_t consumed_split_bytes = 0;
	idx_t completed_split_bytes = 0;
	idx_t queue_wait_ms = 0;
	pybind11::dict submitted_split_count_by_source;
	pybind11::dict consumed_split_count_by_source;
	pybind11::dict completed_split_count_by_source;
	pybind11::dict submitted_split_bytes_by_source;
	pybind11::dict consumed_split_bytes_by_source;
	pybind11::dict completed_split_bytes_by_source;
	pybind11::dict queue_wait_ms_by_source;
};

void AddFteQueueProgressStats(
    FteQueueProgressStats &stats,
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *queue_map);

void AppendFteQueueProgressStats(
    pybind11::dict &out,
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *scan_queue_map,
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *exchange_queue_map);

pybind11::dict BuildNativeTaskStatsDict(
    const duckdb::vector<duckdb::PipelineProgressSnapshot> &snapshots,
    const std::unordered_map<idx_t, duckdb::distributed::ScanSplitBatch> *scan_split_batch_map,
    const std::unordered_map<idx_t, duckdb::distributed::ExchangeSourceTaskDescriptor> *exchange_source_task_map,
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *fte_scan_source_queue_map,
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *fte_exchange_source_queue_map,
    bool hide_remote_exchange_result_collector = false);

pybind11::dict BuildOutputOnlyTaskStatsDict(
    idx_t rows, idx_t bytes, const duckdb::vector<std::string> *operators = nullptr,
    const duckdb::vector<duckdb::InsertionOrderPreservingMap<std::string>> *operator_details = nullptr,
    idx_t physical_rows = 0, idx_t physical_bytes = 0, idx_t network_rows = 0, idx_t network_bytes = 0,
    idx_t memory_rows = 0, idx_t memory_bytes = 0);

bool TryValueToIdx(const duckdb::Value &value, idx_t &out);

bool ExtractCopyResultProgressStats(duckdb::MaterializedQueryResult &result, idx_t fallback_bytes, idx_t &rows,
                                    idx_t &bytes);

pybind11::dict BuildMaterializedInputTaskStats(
    const duckdb::PhysicalOperator &root_op,
    const std::unordered_map<idx_t, duckdb::distributed::ScanSplitBatch> *scan_split_batch_map,
    const std::unordered_map<idx_t, duckdb::distributed::ExchangeSourceTaskDescriptor> *exchange_source_task_map,
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *fte_scan_source_queue_map,
    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>> *fte_exchange_source_queue_map,
    idx_t fallback_rows, idx_t fallback_bytes, duckdb::MaterializedQueryResult *copy_result = nullptr);

void MarkTaskStatsCompleted(pybind11::dict &stats);

bool NativePipelineMatchesMaterializedOperators(const pybind11::handle &native_obj,
                                                const pybind11::handle &materialized_obj);

void OverlayMaterializedProgressStats(pybind11::dict &native_stats, const pybind11::dict &materialized_stats);

pybind11::dict BuildNativePipelineTopology(const duckdb::vector<duckdb::PipelineProgressSnapshot> &snapshots);

bool NativePlanNeedsResultCollector(const duckdb::PhysicalOperator &root_op);

pybind11::dict BuildNativeProgressTopology(duckdb::ClientContext &context,
                                           const std::shared_ptr<duckdb::PhysicalPlan> &physical_plan);

void AppendDistributedCopyResultMetadata(pybind11::dict &out, const duckdb::distributed::DistributedCopyResult &result);

pybind11::object CollectionToArrowTable(duckdb::ColumnDataCollection &collection, duckdb::ClientContext &context);

using namespace duckdb::distributed::python::ray;

pybind11::dict FteSplitQueueResultToDict(const duckdb::distributed::FteSplitQueue::GetNextResult &result);

pybind11::dict BuildFragmentStatsSummary(
    const std::unordered_map<std::string, std::unordered_map<std::string, duckdb::idx_t>> &stats_by_worker);

struct CopyOutputInfo {
	std::string base;        // staging_root_base (empty for direct-write COPY)
	std::string run_id;      // staging_run_id (UUID)
	std::string remote_base; // spec.file_path (for direct write)
};

void ApplyTaskLocalCopyOutput(duckdb::PhysicalPlan &plan, const CopyOutputInfo *info,
                              duckdb::ClientContext *client_context);

bool PyDictContains(const pybind11::dict &dict, const char *key);

pybind11::object PyDictGet(const pybind11::dict &dict, const char *key);

bool PyObjectTruthy(const pybind11::object &obj);

bool IsDynamicFilterSpecDict(const pybind11::dict &dict);

pybind11::dict CopyDynamicFilterSpecWithColumn(const pybind11::dict &domain, pybind11::handle column);

duckdb::Value PyObjectToDuckValue(const pybind11::object &obj);

bool TryCastDynamicFilterValue(const pybind11::object &obj, const duckdb::LogicalType &target_type, duckdb::Value &out);

bool TryResolveScanFilterIndexForColumn(const duckdb::PhysicalTableScan &scan, duckdb::idx_t requested_column_index,
                                        duckdb::idx_t &filter_column_index, duckdb::idx_t &base_column_index);

bool TryGetScanColumnType(const duckdb::PhysicalTableScan &scan, duckdb::idx_t base_column_index,
                          duckdb::idx_t filter_column_index, duckdb::LogicalType &target_type);

bool TryResolveDynamicFilterColumn(const duckdb::PhysicalTableScan &scan, const pybind11::dict &domain,
                                   duckdb::idx_t &filter_column_index, duckdb::LogicalType &target_type);

bool DynamicFilterSourceMatches(const duckdb::PhysicalTableScan &scan, const std::string &source_node_id);

std::string DynamicFilterSourceFromSpec(const std::string &fallback_source_id, const pybind11::dict &domain);

void PushDynamicTableFilter(duckdb::PhysicalTableScan &scan, duckdb::idx_t filter_column_index,
                            duckdb::unique_ptr<duckdb::TableFilter> filter);

duckdb::idx_t ApplyDynamicFilterSpecToScan(duckdb::PhysicalTableScan &scan, const std::string &fallback_source_id,
                                           const pybind11::dict &domain);

duckdb::idx_t ApplyDynamicFilterDomainsToScan(duckdb::PhysicalTableScan &scan, const pybind11::dict &domains);

duckdb::idx_t ApplyDynamicFilterDomainsToOperator(duckdb::PhysicalOperator &op, const pybind11::dict &domains);

duckdb::idx_t ApplyDynamicFilterDomainsToPlan(duckdb::PhysicalPlan &plan, pybind11::object dynamic_filter_domains_obj);

// result_stream_bindings.cpp
struct ResultPartitionStream : public std::enable_shared_from_this<ResultPartitionStream> {
	std::shared_ptr<duckdb::distributed::PlanResultStream> stream_;
	std::shared_ptr<void> keepalive_;
	std::function<void()> rethrow_pending_error_;
	mutex stream_mutex_;
	mutex callback_mutex_;
	duckdb::distributed::python::ray::SafePyObject callback_loop_;
	duckdb::distributed::python::ray::SafePyObject ready_callback_;

	explicit ResultPartitionStream(std::shared_ptr<duckdb::distributed::PlanResultStream> stream);
	~ResultPartitionStream();

	py::object PartitionToPyObject(const std::shared_ptr<duckdb::distributed::ResultPartition> &part);

	py::object next_nowait();

	void set_ready_callback(py::object loop, py::object callback);

	void arm_ready_notification();

	void clear_ready_callback();

private:
	void NotifyReady() noexcept;
};

struct PlanRunState {
	std::shared_ptr<duckdb::distributed::PlanRunner> runner;
	duckdb::shared_ptr<duckdb::ClientContext> client_context;
	duckdb::distributed::python::ray::SafePyObject py_conn_keepalive;
};

// logical_plan_bindings.cpp
struct PyPhysicalPlanWrapper;

static constexpr char LOGICAL_PLAN_ENVELOPE_MAGIC[] = {'V', 'A', 'N', 'E', 'P', 'L', 'A', 'N'};

static constexpr idx_t LOGICAL_PLAN_ENVELOPE_MAGIC_SIZE = sizeof(LOGICAL_PLAN_ENVELOPE_MAGIC);

static constexpr uint32_t LOGICAL_PLAN_PROTOCOL_VERSION = 1;

static constexpr uint32_t LOGICAL_PLAN_MAX_SOURCE_ID_SIZE = 4096;

void AppendUInt32LE(string &result, uint32_t value);

void AppendUInt64LE(string &result, uint64_t value);

uint32_t ReadUInt32LE(const string &envelope, idx_t &offset, const char *field_name);

uint64_t ReadUInt64LE(const string &envelope, idx_t &offset, const char *field_name);

string EncodeLogicalPlanEnvelope(const string &logical_payload);

string DecodeLogicalPlanEnvelope(const string &envelope);

struct PyLogicalPlan {
	string query_id_;
	string serialized_logical_plan_;
	// Driver-local source connection; intentionally omitted from pickle state.
	py::object source_connection_ = py::none();
	py::object udf_registrations_ = py::none();
	py::object connection_snapshot_ = py::none();
	shared_ptr<VaneFSQueryPins> vane_fs_query_pins_ = CreateVaneFSQueryPins();
	// Query-owned Ray ObjectRefs for Arrow snapshots. The plan contains only
	// compact datasource task descriptors that refer to these objects.
	py::object memory_source_refs_ = py::none();

	PyLogicalPlan() = default;

	string idx() const;

	string session_id() const;
	py::dict session_config() const;
	bool has_explicit_s3_credentials() const;

	PyPhysicalPlanWrapper to_physical_plan(py::object conn_obj, py::object effective_session_config) const;
};

struct SerializedLogicalPlanResult {
	string serialized_plan;
	py::object memory_source_refs = py::none();
	shared_ptr<VaneFSQueryPins> vane_fs_query_pins;
};

struct PythonMemorySourceCacheKey {
	PyObject *source_identity;
	string source_version;

	bool operator==(const PythonMemorySourceCacheKey &other) const;
};

struct PythonMemoryScanSource {
	string source_kind;
	py::object source = py::none();
	py::object source_identity = py::none();
	string source_version;
	bool needs_column_version = false;
};

struct PendingPythonMemoryScan {
	LogicalGet *get;
	PythonMemoryScanSource source;
	vector<idx_t> columns;
};

struct PreparedPythonMemorySource {
	optional_ptr<FunctionData> pandas_bind_data;
	string retention_id;
	string source_kind;
	py::object source = py::none();
	py::object source_identity = py::none();
	vector<LogicalType> source_types;
	vector<string> source_names;
	std::set<idx_t> required_columns;
	bool requires_row_count_column = false;
	vector<idx_t> snapshot_columns;
	optional_idx snapshot_row_count_column;
	string row_count_column_name;
	py::object snapshot_schema = py::none();
	py::object object_refs = py::none();
	idx_t row_count = 0;
};

struct PythonMemorySourceCacheKeyHash {
	size_t operator()(const PythonMemorySourceCacheKey &key) const;
};

using PythonMemorySourceGroups = std::unordered_map<PythonMemorySourceCacheKey, unique_ptr<PreparedPythonMemorySource>,
                                                    PythonMemorySourceCacheKeyHash>;

using PythonMemoryScanGroups = std::unordered_map<LogicalGet *, PreparedPythonMemorySource *>;

using PythonMemoryScanRequirements =
    std::unordered_map<PythonMemorySourceCacheKey, std::set<idx_t>, PythonMemorySourceCacheKeyHash>;

idx_t PythonMemorySourceObjectRefCount(const py::object &memory_source_refs);

RegisteredArrow &PythonArrowScanRegisteredSource(const LogicalGet &get);

string PythonMemorySourceVersionBytes(const py::object &version);

bool TryGetPythonMemoryScanSource(const LogicalGet &get, const py::object &memory_module,
                                  PythonMemoryScanSource &result);

bool PreparePythonMemoryScansForPruning(LogicalOperator &op);

vector<idx_t> PythonMemoryScanColumnIds(const LogicalGet &get);

void CollectPythonMemoryScans(LogicalOperator &op, const py::object &memory_module,
                              vector<PendingPythonMemoryScan> &scans, PythonMemoryScanRequirements &requirements);

void GroupPythonMemoryScans(vector<PendingPythonMemoryScan> &scans, const PythonMemoryScanRequirements &requirements,
                            const py::object &memory_module, PythonMemorySourceGroups &source_groups,
                            PythonMemoryScanGroups &scan_groups);

void PreparePythonMemorySource(PreparedPythonMemorySource &prepared, ClientContext &context,
                               const py::object &memory_module, py::dict &memory_source_refs);

void RewritePythonMemoryScans(unique_ptr<LogicalOperator> &op, ClientContext &context, const py::object &memory_module,
                              py::dict &memory_source_refs, const PythonMemoryScanGroups &scan_groups);

SerializedLogicalPlanResult SerializeBoundLogicalPlan(unique_ptr<LogicalOperator> &logical_plan, Binder &binder,
                                                      const shared_ptr<ClientContext> &client_context);

SerializedLogicalPlanResult SerializeLogicalPlanFromRelation(const shared_ptr<Relation> &rel);

DuckDBPyConnection &ExtractPyConnectionWrapper(py::object conn_obj);

py::dict CopyPyDict(const py::dict &source);

bool IsExtensionSecuritySetting(const string &lower_name);

bool IsSecretPersistenceSetting(const string &lower_name);

bool IsWorkerResourceSetting(const string &lower_name);

bool IsWorkerExtensionLocationSetting(const string &lower_name);

bool IsWorkerLocalSetting(const string &lower_name);

py::dict RemoveWorkerLocalBootstrapSettings(const py::dict &config);

py::dict SanitizeBootstrapConfig(const py::dict &config, bool disable_persistent_secrets,
                                 bool remove_worker_local_settings = false);

py::dict ForceReadOnlyAccessMode(const py::dict &config);

py::object LookupBootstrapSnapshot(const py::object &snapshot_obj);

py::dict LookupVaneSessionSnapshot(const py::object &snapshot_obj);

string VaneSessionIdFromSnapshot(const py::object &snapshot_obj);

py::dict VaneSessionConfigFromSnapshot(const py::object &snapshot_obj);

bool HasExplicitS3CredentialsFromSnapshot(const py::object &snapshot_obj);

bool IsDefaultBootstrapSnapshot(const py::object &bootstrap_obj);

py::object NormalizeBootstrapSnapshot(const py::dict &bootstrap_obj);

py::object PrepareWorkerConnectionSnapshot(const py::object &snapshot_obj);

bool PythonObjectsEqual(const py::handle &lhs, const py::handle &rhs);

string BootstrapDatabasePath(const py::object &bootstrap_obj);

bool BootstrapUsesInMemoryDatabase(const py::object &bootstrap_obj);

bool SnapshotHasAttachedDatabases(const py::object &snapshot_obj);

bool SnapshotHasDynamicExtensions(const py::object &snapshot_obj);

string RunnerTypeFromSnapshot(const py::object &snapshot_obj);

py::object CreateConnectionFromBootstrapSnapshot(const py::object &bootstrap_obj, const string &runner_type,
                                                 bool use_instance_cache = true, bool force_file_read_only = false,
                                                 bool remove_worker_local_settings = false);

py::object CreateSnapshotBaselineConnection(DuckDBPyConnection &source_conn, const py::object &bootstrap_obj);

bool ConnectionMatchesBootstrapSnapshot(py::object conn_obj, const py::object &snapshot_obj);

bool ConnectionsShareDatabaseInstance(const py::object &lhs_obj, const py::object &rhs_obj);

py::object ResolveConnectionForSnapshot(py::object conn_obj, const py::object &snapshot_obj);

py::object ResolvePlanningConnectionForSnapshot(py::object conn_obj, const py::object &source_conn_obj,
                                                const py::object &snapshot_obj);

struct QueryPythonReplayState {
	string session_id;
	duckdb::distributed::python::ray::SafePyObject session_config;
	duckdb::distributed::python::ray::SafePyObject udf_registrations;
	duckdb::distributed::python::ray::SafePyObject udf_actor_handles;
	duckdb::distributed::python::ray::SafePyObject memory_source_refs;
	duckdb::distributed::python::ray::SafePyObject connection_snapshot;
	// Driver-only state. Fragment wrappers take isolated cursors from this
	// connection, and DistributedPhysicalPlan pickle state never includes it.
	duckdb::distributed::python::ray::SafePyObject coordinator_connection;
	shared_ptr<VaneFSQueryPins> vane_fs_query_pins = CreateVaneFSQueryPins();

	QueryPythonReplayState(string session_id_p, py::object session_config_p, py::object udf_registrations_p,
	                       py::object udf_actor_handles_p, py::object memory_source_refs_p,
	                       py::object connection_snapshot_p, py::object coordinator_connection_p);
};

struct ConnectionSettingRecord {
	string name;
	string value;
	bool value_is_null = false;
	string input_type;
	string scope;
};

void EnforceExtensionSecuritySettings(duckdb::Connection &conn);

bool ShouldSkipConnectionSettingSnapshot(const string &name, const string &input_type);

bool IsBooleanConnectionSettingType(const string &input_type);

bool IsNumericConnectionSettingType(const string &input_type);

bool IsVaneSessionBaselineConnectionSetting(const string &name);

bool IsS3CredentialConnectionSetting(const string &name);

string QuoteSQLStringLiteral(const string &value);

duckdb::unique_ptr<duckdb::MaterializedQueryResult> ExecuteSnapshotQuery(duckdb::Connection &conn, const string &sql);

string StripS3EndpointSchemeForDuckDB(const string &endpoint_url);

bool S3EndpointUsesSSL(const string &endpoint_url);

void ConfigureConnectionForS3Endpoint(duckdb::Connection &conn, const string &endpoint_url, const string &access_key,
                                      const string &secret_key, const string &region);

string VaneSessionConfigValue(const py::dict &config, const char *key);

void ApplyVaneSessionConfigValues(duckdb::Connection &conn, const py::dict &config);

void ApplyVaneSessionConfig(duckdb::Connection &conn, const py::object &snapshot_obj);

void CloseOpenPythonConnectionResult(DuckDBPyConnection &conn_wrapper);

void ApplyEffectiveVaneSessionConfig(DuckDBPyConnection &conn_wrapper, const py::object &config_obj);

vector<string> LoadedNonStaticExtensionNames(DuckDBPyConnection &conn_wrapper);

py::list NormalizeDynamicExtensionSnapshot(const py::object &snapshot_obj);

py::list CaptureDynamicExtensionSnapshot(const py::object &conn_obj);

vector<string> DynamicExtensionNamesFromSnapshot(const py::list &dynamic_extensions,
                                                 const string &snapshot_description);

string DynamicExtensionListIdentity(const vector<string> &extensions);

void ValidateCapturedDynamicExtensionNames(const vector<string> &loaded_extensions,
                                           const vector<string> &snapshot_extensions);

void ValidateWorkerDynamicExtensionNames(DuckDBPyConnection &conn_wrapper, const vector<string> &expected_extensions);

void ValidateWorkerRecordedDynamicExtensionSnapshot(DuckDBPyConnection &conn_wrapper, const py::object &conn_obj,
                                                    const py::list &expected_extensions);

void PrepareDynamicExtensionSnapshot(py::object conn_obj, const py::list &dynamic_extensions);

struct StaticExtensionSnapshotEntry {
	string name;
	string version;

	bool operator==(const StaticExtensionSnapshotEntry &other) const;

	bool operator<(const StaticExtensionSnapshotEntry &other) const;
};

vector<StaticExtensionSnapshotEntry> ParseStaticExtensionSnapshot(const py::dict &snapshot);

void ValidateDistinctStaticAndDynamicExtensionNames(const vector<StaticExtensionSnapshotEntry> &static_extensions,
                                                    const vector<string> &dynamic_extension_names);

vector<StaticExtensionSnapshotEntry> QueryLoadedStaticExtensions(DuckDBPyConnection &conn_wrapper);

std::vector<ConnectionSettingRecord> QueryConnectionSettings(DuckDBPyConnection &conn_wrapper);

bool IsSafeStaticExtensionName(const string &name);

string StaticExtensionListIdentity(const vector<StaticExtensionSnapshotEntry> &extensions);

vector<StaticExtensionSnapshotEntry> LoadedStaticExtensions(DatabaseInstance &database);

void LoadStaticRayExtensions(duckdb::Connection &conn, const vector<StaticExtensionSnapshotEntry> &extensions);

py::list CaptureDistributedExtensionContracts(DatabaseInstance &database);

vector<string> ParseDistributedExtensionContracts(const py::dict &snapshot);

py::list CaptureAttachedDatabaseSnapshot(DuckDBPyConnection &conn_wrapper);

void ApplyAttachedDatabaseSnapshot(duckdb::Connection &conn, const py::dict &snapshot);

py::object CaptureConnectionSnapshot(DuckDBPyConnection &conn_wrapper, const py::object &conn_obj,
                                     const shared_ptr<VaneFSQueryPins> &vane_fs_pins = nullptr);

enum class DuckDBRelationPlanKind : uint8_t { READ, WRITE, DATA_SINK };

PyLogicalPlan LogicalPlanFromDuckDBRelation(py::object relation_obj, py::object query_id_obj,
                                            DuckDBRelationPlanKind plan_kind);

struct ConnectionSnapshotApplyOptions {
	bool apply_session_config = true;
	bool enforce_extension_security = true;
	bool apply_s3_credentials = true;
	bool apply_settings = true;
	bool apply_attached_databases = false;
	bool apply_vane_fs = true;
	shared_ptr<VaneFSQueryPins> vane_fs_query_pins;
};

void ApplyConnectionSnapshot(py::object conn_obj, const py::object &snapshot_obj,
                             const ConnectionSnapshotApplyOptions &options);

bool ConnectionSnapshotDeclaresExtension(const py::object &snapshot_obj, const string &extension_name);

void ValidateConnectionSnapshotExtensions(py::object conn_obj, const py::object &snapshot_obj,
                                          bool enforce_extension_security);

py::dict ValidateConnectionSnapshotSourceID(const py::object &snapshot_obj);

void ApplyConnectionSnapshot(py::object conn_obj, const py::object &snapshot_obj,
                             const ConnectionSnapshotApplyOptions &options = {});

void PrepareConnectionSnapshotExtensions(py::object conn_obj, const py::object &snapshot_obj,
                                         const shared_ptr<VaneFSQueryPins> &vane_fs_pins = nullptr);

enum class QueryPythonReplayField : uint8_t {
	UDFRegistrations,
	UDFActorHandles,
	MemorySourceRefs,
	ConnectionSnapshot,
	CoordinatorConnection,
};

py::object LookupQueryPythonReplayState(const string &query_id, QueryPythonReplayField field);

bool RegisterQueryPythonReplayState(const string &query_id, const py::object &udf_registrations,
                                    const py::object &udf_actor_handles, const py::object &memory_source_refs,
                                    const py::object &connection_snapshot, const py::object &coordinator_connection,
                                    const shared_ptr<VaneFSQueryPins> &vane_fs_pins = nullptr);

shared_ptr<VaneFSQueryPins> LookupQueryVaneFSPins(const string &query_id);

py::object LookupQueryUDFRegistrations(const string &query_id);

py::object LookupQueryConnectionSnapshot(const string &query_id);

py::object LookupQueryCoordinatorConnection(const string &query_id);

py::object LookupQueryUDFActorHandles(const string &query_id);

py::object LookupQueryMemorySourceRefs(const string &query_id);

py::object LookupMemorySourceRef(const string &source_id, idx_t partition_index);

void CleanupQueryPythonReplayState(const string &query_id);

void CleanupAllQueryPythonReplayState();

duckdb::unique_ptr<duckdb::LogicalOperator>
RebindAndOptimizeDeserializedLogicalPlan(duckdb::ClientContext &context,
                                         duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan);

duckdb::distributed::DistributedPipelineNodeRef
BuildDistributedPipelineNode(const std::shared_ptr<duckdb::distributed::DistributedPhysicalPlan> &plan,
                             duckdb::ClientContext *client_context = nullptr);

const UDFFunctionData *TryGetUDFBindData(const FunctionData *bind_data);

const UDFFunctionData *TryGetUDFBindData(const PhysicalTableInOutFunction &inout);

const UDFFunctionData *TryGetUDFBindData(const PhysicalStreamingUDF &streaming);

UDFFunctionData *TryGetMutableUDFBindData(PhysicalOperator &op);

void CollectMutableUDFBindData(PhysicalOperator &op, vector<UDFFunctionData *> &out);

// distributed_plan_bindings.cpp
void AssignDataSourceQueryOwner(duckdb::PhysicalOperator &op, const string &query_id,
                                bool allow_unacquired_rebind = false);

struct PyPhysicalPlanWrapper {
	static constexpr uint64_t INIT_MAGIC = 0x445046504C414E31ULL;
	uint64_t init_magic_;
	// CRITICAL: Field order matters for destruction!
	// In C++, members are destroyed in REVERSE order of declaration.
	// worker_connection_ MUST be declared FIRST so it is destroyed LAST,
	// after plan_ is destroyed. This ensures the allocator stays valid.
	py::object worker_connection_; // Keep DuckDB connection alive for worker-side deserialized plans
	duckdb::shared_ptr<duckdb::ClientContext>
	    client_context_; // Keep driver-side context alive when built from relation
	string query_id_;
	string resource_query_id_;
	std::shared_ptr<duckdb::distributed::DistributedPhysicalPlan> plan_;
	py::object arrow_schema_;        // Arrow schema for type information
	py::object udf_registrations_;   // Connection-local Python UDF registrations captured from the source relation
	py::object udf_actor_handles_;   // Resource-unit keyed executor options survive fragment extraction
	py::object memory_source_refs_;  // Query-owned Ray ObjectRefs for Python in-memory scans
	py::object connection_snapshot_; // Connection settings/extensions snapshot captured from the source relation
	shared_ptr<VaneFSQueryPins> vane_fs_query_pins_ = CreateVaneFSQueryPins();
	string serialized_root_; // Deferred serialized PhysicalOperator bytes (for pickle round-trips)

	PyPhysicalPlanWrapper(const PyPhysicalPlanWrapper &) = default;
	PyPhysicalPlanWrapper(PyPhysicalPlanWrapper &&) = default;
	PyPhysicalPlanWrapper &operator=(const PyPhysicalPlanWrapper &) = default;
	PyPhysicalPlanWrapper &operator=(PyPhysicalPlanWrapper &&) = default;

	~PyPhysicalPlanWrapper();

	bool has_root() const;

	void ensure_connection_snapshot(py::object conn_obj, bool apply_session_config = true) const;

	py::object resolve_execution_connection(py::object conn_obj) const;

	void apply_udf_actor_handles();

	void ensure_plan_identity();

	// Materialize deferred serialized_root_ into the plan's PhysicalPlan.
	// Requires a DuckDB connection for deserialization context.
	void materialize_deferred_root(py::object conn_obj, bool apply_session_config = true);

	string serialize_root_for_clone() const;

	void validate_serializable_for_submission() const;

	static string serialize_operator_root(const duckdb::PhysicalOperator &root);

	PyPhysicalPlanWrapper clone(py::object conn_obj) const;

	PyPhysicalPlanWrapper();
	explicit PyPhysicalPlanWrapper(std::shared_ptr<duckdb::distributed::DistributedPhysicalPlan> p);

	PyPhysicalPlanWrapper(std::shared_ptr<duckdb::distributed::DistributedPhysicalPlan> p, string resource_query_id);

	bool IsInitialized() const;

	string idx() const;

	string resource_query_id() const;

	string session_id() const;

	py::dict session_config() const;

	bool has_explicit_s3_credentials() const;

	size_t num_partitions() const;

	static std::pair<bool, string> UDFPayloadStringField(const Value &payload, const string &name);

	static string UDFPayloadStringOrDefault(const Value &payload, const string &name,
	                                        const string &default_value = string());

	static bool UDFPayloadStringEquals(const Value &payload, const string &name, const string &expected);

	static string UDFOperatorIdentityOrEmpty(const Value &payload);

	static Value UDFPayloadWithStringField(const Value &payload, const string &name, const string &value);

	static vector<Value> EnsureStableUDFOperatorIdentities(const vector<UDFFunctionData *> &physical_udfs,
	                                                       const string &query_id);

	py::dict collect_local_resource_graph_metadata(py::object conn_obj);

	py::dict collect_query_resource_graph_metadata(py::object conn_obj);

	py::dict collect_resource_graph_metadata(py::object conn_obj);

	string repr_ascii(bool simple) const;
	string repr_mermaid(bool simple, bool bottom_up) const;
	py::dict datasource_scan_cardinalities_for_test() const;
	py::dict scan_split_batch_map() const;
	// ── UDF node collection ──────────────────────────────────────
	// Walk the physical plan and return metadata for each UDF node.
	// Used by RayQueryDriverActor to pre-create actor pools at the driver level.
	py::list collect_udf_nodes(py::object conn_obj);

	// ── Collect vLLM nodes and pre-inject pool names ─────────────────
	// Walks the physical plan, finds PhysicalVLLM operators, computes a
	// deterministic pool name for each, injects it into the operator's
	// options (so that VLLMProjectNode::produce_tasks reads it back),
	// and returns metadata for driver-side actor pre-creation.
	py::list collect_vllm_nodes(py::object conn_obj);

	// ── Inject actor handles into UDF nodes ─────────────────────
	// Local runtime owners bind only to the current native plan. They must not
	// become worker replay state or add per-query IDs to reusable model payloads.
	void set_local_udf_executor_options(py::dict options_map, py::object);

	// Takes a dict {node_id: executor_options} and injects actor handles into
	// the corresponding UDF bind data for worker execution.
	void set_udf_actor_handles(py::dict handles_map, py::object);
};

class PyBackendResultOperationGuard {
public:
	explicit PyBackendResultOperationGuard(std::function<void()> finish);

	PyBackendResultOperationGuard(const PyBackendResultOperationGuard &) = delete;
	PyBackendResultOperationGuard &operator=(const PyBackendResultOperationGuard &) = delete;

	~PyBackendResultOperationGuard();

private:
	std::function<void()> finish_;
};

class PyBackendWorkerManager : public duckdb::distributed::WorkerManager {
public:
	template <class T>
	using DuckDBResult = duckdb::distributed::DuckDBResult<T>;
	using DuckDBError = duckdb::distributed::DuckDBError;
	using QueryCleanup = std::function<void(const string &)>;
	using QueryLifecycleCoordinator = duckdb::distributed::python::ray::QueryLifecycleCoordinator;

	explicit PyBackendWorkerManager(py::object backend, QueryCleanup query_cleanup);

	void register_query_owner(const string &query_id, const string &owner_query_id,
	                          const std::function<void(const py::error_already_set &)> &publish_registration_error = {},
	                          bool *lifecycle_published = nullptr);

	DuckDBResult<std::vector<duckdb::distributed::WorkerSnapshot>> worker_snapshots() const override;

	DuckDBResult<void> try_autoscale(const std::vector<duckdb::distributed::TaskResourceRequest> &) override;

	DuckDBResult<void> shutdown() override;

	DuckDBResult<void> submit_fte_task_events(std::vector<duckdb::distributed::WorkerTask> tasks) override;

	DuckDBResult<void> task_input_stream_exhausted_for_query(
	    const string &query_id, const std::unordered_set<duckdb::distributed::SourceNodeId> &source_node_ids) override;

	DuckDBResult<void> task_production_finished(const string &query_id) override;

	DuckDBResult<void> materialization_barrier_completed(const string &query_id,
	                                                     duckdb::distributed::NodeID node_id) override;

	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> wait_fte_query(const string &query_id,
	                                                                                  double timeout_s) override;

	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>
	wait_fte_query(const string &query_id, double timeout_s,
	               duckdb::distributed::MaterializedOutputCallback on_output) override;

	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> wait_fte_query(
	    const string &query_id, double timeout_s,
	    const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash> &task_contexts,
	    duckdb::distributed::MaterializedOutputCallback on_output) override;

	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>
	wait_fte_query_streaming(const string &query_id, double timeout_s,
	                         duckdb::distributed::MaterializedOutputCallback on_output,
	                         bool preserve_task_order = true) override;

	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>> WaitFteQuery(
	    const string &query_id, double timeout_s,
	    const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash> &task_contexts,
	    duckdb::distributed::MaterializedOutputCallback on_output, bool stream_outputs,
	    bool preserve_task_order = true);

	DuckDBResult<void> abort_and_quiesce_query(const string &query_id) override;

	void drop_query_fragments(const string &query_id);

	void rethrow_submission_error(const string &query_id, const string &details = string());

	std::unordered_map<string, std::unordered_map<string, idx_t>> fragment_stats_by_worker() const;

private:
	using PythonTaskResultHandle = duckdb::distributed::python::ray::PythonTaskResultHandle;
	using PythonTaskResultHandles = std::vector<std::unique_ptr<PythonTaskResultHandle>>;
	using PythonTaskResultHandleMap = std::unordered_map<string, PythonTaskResultHandles>;

	mutable mutex mutex_;
	duckdb::distributed::python::ray::SafePyObject backend_;
	QueryCleanup query_cleanup_;
	QueryLifecycleCoordinator query_lifecycles_;
	duckdb::distributed::python::ray::PythonExceptionStore submission_errors_;
	PythonTaskResultHandleMap result_handles_by_query_;
	PythonTaskResultHandleMap retained_result_handles_by_query_;
	PythonTaskResultHandleMap cleanup_retry_result_handles_by_query_;
	std::unordered_map<string, std::unordered_map<string, size_t>> result_handle_counts_by_query_;

	static string QueryIdFromTaskEvents(const std::vector<duckdb::distributed::WorkerTask> &tasks);

	distributed::Optional<QueryLifecycleCoordinator::Abort> BeginResultHandleAbort(const string &query_id);

	distributed::Optional<QueryLifecycleCoordinator::Abort>
	BeginResultHandleAbort(const QueryLifecycleCoordinator::Teardown &teardown);

	distributed::Optional<QueryLifecycleCoordinator::Operation>
	BeginResultHandleOperation(const string &query_id, const string &requested_owner_query_id = string());

	void WaitForResultHandleOperations(const QueryLifecycleCoordinator::LifecycleRef &lifecycle);

	void WaitForAllResultHandleOperations();

	distributed::Optional<QueryLifecycleCoordinator::Teardown> BeginResultHandleTeardown(const string &query_id);

	bool BeginResultHandleShutdown();

	DuckDBResult<void>
	ExecuteResultHandleAbort(duckdb::distributed::Optional<QueryLifecycleCoordinator::Abort> active_abort);

	static string PyStringField(const py::dict &dict, const char *field_name, const string &default_value);

	static double PyDoubleField(const py::dict &dict, const char *field_name, double default_value);

	static uint64_t PyUInt64Field(const py::dict &dict, const char *field_name, uint64_t default_value);

	static bool TryBoundedTextValue(const py::object &value, const char *field_name, string &result);

	static bool TryBoundedStatusText(const py::dict &status, const char *field_name, string &result);

	static bool TryBoundedFailedPartitionText(const py::dict &status, string &result);

	static string PyStatusMessage(const py::object &status_obj, bool failed, bool finished, bool canceled, bool matched,
	                              bool registration_pending, size_t selected_attempt_count);

	static bool RequiredStatusBool(const py::object &status_obj, const char *field_name);

	static bool OptionalStatusBool(const py::object &status_obj, const char *field_name, bool default_value);

	static std::unordered_set<string> SelectedAttemptTaskIds(const py::object &status_obj);

	void StorePythonResultHandles(const string &query_id, const QueryLifecycleCoordinator::LifecycleRef &lifecycle,
	                              py::object raw_handles);

	DuckDBResult<void> ValidateResultHandleCoverage(const string &query_id,
	                                                const std::unordered_set<string> &selected_attempt_task_ids) const;

	void StorePythonTaskResultHandles(
	    const string &query_id,
	    std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> handles);

	void RetainPythonTaskResultHandles(
	    const string &query_id,
	    std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> handles);

	void StoreCleanupRetryResultHandles(
	    const string &query_id,
	    std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> handles);

	void ReleaseLatePythonTaskResultHandles(
	    const string &query_id,
	    std::vector<std::unique_ptr<duckdb::distributed::python::ray::PythonTaskResultHandle>> handles);

	void ClearResultHandles(const string &query_id);

	void ClearResultHandlesForLifecycle(const QueryLifecycleCoordinator::Teardown &teardown);

	bool LifecycleHasResultHandles(const QueryLifecycleCoordinator::LifecycleRef &lifecycle) const;

	void ClearAllResultHandles();

	DuckDBResult<std::vector<duckdb::distributed::MaterializedOutput>>
	DrainResultHandles(const string &query_id, double timeout_s,
	                   const std::unordered_set<string> &selected_attempt_task_ids,
	                   const std::unordered_set<duckdb::distributed::TaskContext, duckdb::distributed::TaskContextHash>
	                       *task_context_filter,
	                   bool release_payloads, duckdb::distributed::MaterializedOutputCallback on_output = {},
	                   bool selected_only = false, bool preserve_task_order = true);
};

void CleanupQueryOwnedPythonState(const string &query_id);

struct PyPhysicalPlanWrapperRunner {
	std::shared_ptr<duckdb::distributed::WorkerManager> worker_manager_;
	std::shared_ptr<duckdb::distributed::python::ray::RayWorkerManager> ray_worker_manager_;
	std::shared_ptr<PyBackendWorkerManager> py_backend_worker_manager_;
	// Stored copy plan streaming context for two-phase COPY
	struct StoredCopyContext {
		duckdb::distributed::DistributedCopySpec copy_spec;
		std::string staging_root;
	};
	std::unordered_map<string, std::shared_ptr<StoredCopyContext>> streaming_results_;
	std::atomic<bool> fail_next_extension_result_marshalling_for_test_ {false};

	PyPhysicalPlanWrapperRunner();

	explicit PyPhysicalPlanWrapperRunner(py::object backend);

	void register_query_owner(const string &query_id, const string &owner_query_id,
	                          bool *lifecycle_published = nullptr);

	void drop_query_fragments(const string &query_id);

	void warm_up();

	void shutdown();

	void fail_next_extension_result_marshalling_for_test();

	void close_session(const string &session_id);

	std::unordered_map<string, std::unordered_map<string, idx_t>> fragment_stats_by_worker() const;

	void rethrow_submission_error(const string &query_id, const string &details = string());

	std::shared_ptr<ResultPartitionStream> run_plan(const PyPhysicalPlanWrapper &plan,
	                                                duckdb::shared_ptr<duckdb::ClientContext> client_context = nullptr,
	                                                duckdb::distributed::python::ray::SafePyObject py_conn_keepalive =
	                                                    duckdb::distributed::python::ray::SafePyObject());

	py::object run_copy_plan(const PyPhysicalPlanWrapper &plan,
	                         duckdb::shared_ptr<duckdb::ClientContext> client_context = nullptr,
	                         duckdb::distributed::python::ray::SafePyObject py_conn_keepalive =
	                             duckdb::distributed::python::ray::SafePyObject(),
	                         duckdb::distributed::python::ray::SafePyObject on_execution_started =
	                             duckdb::distributed::python::ray::SafePyObject());

	py::dict run_datasink_plan(const PyPhysicalPlanWrapper &plan,
	                           duckdb::shared_ptr<duckdb::ClientContext> client_context = nullptr,
	                           duckdb::distributed::python::ray::SafePyObject py_conn_keepalive =
	                               duckdb::distributed::python::ray::SafePyObject(),
	                           duckdb::distributed::python::ray::SafePyObject on_execution_started =
	                               duckdb::distributed::python::ray::SafePyObject());

	py::object finalize_copy_impl(py::list file_infos_py, py::str copy_spec_key, py::str,
	                              duckdb::shared_ptr<duckdb::ClientContext> client_context = nullptr);

	// Core implementation of execute_native that works with a raw PhysicalPlan pointer
	py::object execute_native_impl(
	    py::object conn_obj, std::shared_ptr<duckdb::PhysicalPlan> physical_plan, const string &plan_id,
	    const string &resource_query_id,
	    const std::unordered_map<idx_t, duckdb::distributed::ScanSplitBatch> *scan_split_batch_map,
	    const std::unordered_map<idx_t, duckdb::distributed::ExchangeSourceTaskDescriptor> *exchange_source_task_map =
	        nullptr,
	    const duckdb::distributed::ExchangeSinkInstanceTaskDescriptor *exchange_sink_instance_task = nullptr,
	    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>>
	        *fte_scan_source_queue_map = nullptr,
	    const std::unordered_map<idx_t, std::shared_ptr<duckdb::distributed::FteSplitQueue>>
	        *fte_exchange_source_queue_map = nullptr,
	    const CopyOutputInfo *copy_output_info = nullptr, py::object dynamic_filter_domains_obj = py::none(),
	    py::object native_progress_callback = py::none(), py::object runtime_context_obj = py::none(),
	    py::object native_execution_started = py::none());

	// Execute physical plan using DuckDB's native Executor (DistributedPhysicalPlan version)
	py::object execute_native(py::object conn_obj, const PyPhysicalPlanWrapper &plan);
};

py::dict DescribeNativeProgress(py::object conn_obj, const PyPhysicalPlanWrapper &plan);

// ray_binding_support.cpp
class PhysicalNonSerializableOperatorForTest : public PhysicalOperator {
public:
	explicit PhysicalNonSerializableOperatorForTest(PhysicalPlan &physical_plan);

	string GetName() const override;

protected:
	void SerializeOperatorData(Serializer &) const override;
};

class CoordinatorOnlyWriteGlobalStateForTest final : public DistributedWriteGlobalState {};

class CoordinatorOnlyWriteLocalStateForTest final : public DistributedWriteLocalState {};

unique_ptr<DistributedWriteGlobalState>
CoordinatorOnlyWriteInitializeGlobalForTest(ClientContext &, const DistributedExtensionWriteInfo &,
                                            const DistributedWriteTaskContext &);

unique_ptr<DistributedWriteLocalState> CoordinatorOnlyWriteInitializeLocalForTest(ExecutionContext &,
                                                                                  const DistributedExtensionWriteInfo &,
                                                                                  const DistributedWriteTaskContext &,
                                                                                  DistributedWriteGlobalState &);

void CoordinatorOnlyWriteSinkForTest(ExecutionContext &, const DistributedExtensionWriteInfo &,
                                     const DistributedWriteTaskContext &, DistributedWriteGlobalState &,
                                     DistributedWriteLocalState &, DataChunk &);

void CoordinatorOnlyWriteCombineForTest(ExecutionContext &, const DistributedExtensionWriteInfo &,
                                        const DistributedWriteTaskContext &, DistributedWriteGlobalState &,
                                        DistributedWriteLocalState &);

vector<DistributedWriteFragment> CoordinatorOnlyWriteFinalizeForTest(ClientContext &,
                                                                     const DistributedExtensionWriteInfo &,
                                                                     const DistributedWriteTaskContext &,
                                                                     DistributedWriteGlobalState &);

void RegisterCoordinatorOnlyExtensionWriteForTest(ClientContext &context);

class PhysicalCoordinatorOnlyExtensionWriteForTest final : public PhysicalOperator,
                                                           public distributed::ExtensionWriteTaskProvider {
public:
	explicit PhysicalCoordinatorOnlyExtensionWriteForTest(PhysicalPlan &physical_plan, bool fail_finalize_p = false);

	optional_ptr<distributed::ExtensionWriteTaskProvider> GetExtensionWriteTaskProvider() override;

	const distributed::DistributedExtensionWritePlan &WritePlan() const override;

	void ValidateDistributedWrite(ClientContext &) const override;

	idx_t FinalizeDistributedWrite(ClientContext &, const vector<DistributedWriteTaskResult> &results) const override;

	void AbortDistributedWrite(ClientContext &, const vector<DistributedWriteTaskResult> &) const override;

protected:
	void SerializeOperatorData(Serializer &) const override;

private:
	distributed::DistributedExtensionWritePlan plan;
	bool fail_finalize;
};

class CountingResultCollectorForTest {
public:
	CountingResultCollectorForTest();

	unique_ptr<PhysicalOperator> operator()(ClientContext &context, PreparedStatementData &data) const;

	shared_ptr<std::atomic<idx_t>> calls;
};

class ScopedShuffleCacheRegistrationForTest {
public:
	explicit ScopedShuffleCacheRegistrationForTest(string exchange_id);

	~ScopedShuffleCacheRegistrationForTest();

	ScopedShuffleCacheRegistrationForTest(const ScopedShuffleCacheRegistrationForTest &) = delete;
	ScopedShuffleCacheRegistrationForTest &operator=(const ScopedShuffleCacheRegistrationForTest &) = delete;

private:
	string exchange_id_;
};

py::object ResolveFlightShuffleCleanupConnection(py::object cleanup_connection, const py::object &connection_snapshot,
                                                 const py::object &effective_session_config,
                                                 bool apply_snapshot_s3_credentials);

string QueryConnectionSettingForTest(DuckDBPyConnection &connection, const string &name);

template <typename CALLABLE>
static auto WithCopyRecoveryContext(py::object conn_obj, CALLABLE callback)
    -> decltype(callback(std::declval<duckdb::ClientContext &>())) {
	using Result = decltype(callback(std::declval<duckdb::ClientContext &>()));
	auto run_callback = [&](duckdb::ClientContext &context) -> Result {
		distributed::Optional<Result> result;
		std::exception_ptr callback_error;
		{
			py::gil_scoped_release release;
			context.RunFunctionInTransaction(
			    [&]() {
				    // Match normal query startup after a failed query leaves a stale interrupt flag behind.
				    context.ClearInterrupt();
				    try {
					    result.emplace(callback(context));
				    } catch (...) {
					    callback_error = std::current_exception();
				    }
			    },
			    false);
		}
		if (callback_error) {
			std::rethrow_exception(callback_error);
		}
		D_ASSERT(result.has_value());
		return std::move(result.value());
	};
	if (!conn_obj.is_none()) {
		auto &conn_wrapper = ExtractPyConnectionWrapper(std::move(conn_obj));
		return run_callback(*conn_wrapper.con.GetConnection().context);
	}
	duckdb::DuckDB db(nullptr);
	duckdb::Connection conn(db);
	return run_callback(*conn.context);
}

py::list ReadIntegerExchangeSourceForTest(duckdb::ClientContext &context,
                                          duckdb::distributed::FlightExchangeConfig config,
                                          duckdb::distributed::ExchangeSourceHandle handle);

using namespace duckdb::distributed;

class FakeObjectShuffleStorage final : public ShuffleStorage {
public:
	explicit FakeObjectShuffleStorage(FileSystem &fs);

	DuckDBResult<void> CreateDirectories(const std::string &path) const override;

	bool IsRegularFile(const std::string &path) const override;

	DuckDBResult<idx_t> FileSize(const std::string &path) const override;

	DuckDBResult<void> WriteTextFileAtomically(const std::string &path, const std::string &contents) const override;

	DuckDBResult<std::string> ReadTextFile(const std::string &path) const override;

	DuckDBResult<idx_t> RemoveAll(const std::string &path) const override;

	DuckDBResult<std::shared_ptr<arrow::io::OutputStream>> OpenArrowOutput(const std::string &path) const override;

	DuckDBResult<std::shared_ptr<arrow::io::InputStream>> OpenArrowInput(const std::string &path) const override;

	bool HasTextFile(const std::string &path) const;

	std::string TextFile(const std::string &path) const;

	mutable std::unordered_map<std::string, std::string> text_files;
	mutable std::unordered_set<std::string> directories;
	mutable idx_t text_puts = 0;
	mutable idx_t arrow_output_opens = 0;
	mutable idx_t arrow_input_opens = 0;

private:
	FileSystem &fs_;
};

using RayBackedResultPartition = duckdb::distributed::python::ray::RayBackedResultPartition;

void RegisterRayCoreBindings(py::module_ &m);

void RegisterRayPlansBindings(py::module_ &m);

void RegisterRayRunnerBindings(py::module_ &m);

void RegisterRayExecutionTestsBindings(py::module_ &m);

void RegisterRayExchangeTestsBindings(py::module_ &m);

void RegisterRayShuffleTestsBindings(py::module_ &m);

void RegisterRayRecoveryTestsBindings(py::module_ &m);

void RegisterRayDescriptorTestsBindings(py::module_ &m);

void RegisterRayMiscTestsBindings(py::module_ &m);
} // namespace ray_bindings
} // namespace duckdb
