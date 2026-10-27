// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "ray_binding_internal.hpp"

namespace duckdb {
namespace ray_bindings {
namespace py = ::duckdb::py;

void RegisterRayExchangeTestsBindings(py::module_ &m) {
	m.def(
	    "python_task_result_handle_for_test",
	    [](py::object handle) {
		    using namespace duckdb::distributed;
		    using namespace duckdb::distributed::python::ray;

		    auto task_handle = MakePythonTaskResultHandle(std::move(handle));
		    std::string worker_id;
		    bool has_output = false;
		    int flight_port = 0;
		    {
			    pybind11::gil_scoped_release release;
			    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
			    while (std::chrono::steady_clock::now() < deadline) {
				    auto polled = task_handle.poll();
				    if (!polled.first) {
					    std::this_thread::sleep_for(std::chrono::milliseconds(5));
					    continue;
				    }
				    if (polled.second.is_err()) {
					    throw polled.second.error();
				    }
				    auto payload = std::move(polled.second).value();
				    has_output = payload.first;
				    const auto &output = payload.second;
				    if (output.worker_id()) {
					    worker_id = *output.worker_id();
				    }
				    flight_port = output.flight_port();
				    task_handle.AckPollResult();
				    task_handle.ReleasePollResult();
				    break;
			    }
		    }
		    if (worker_id.empty()) {
			    throw std::runtime_error("PythonTaskResultHandle did not produce output before timeout");
		    }
		    pybind11::dict result;
		    result["worker_id"] = worker_id;
		    result["has_output"] = has_output;
		    result["flight_port"] = flight_port;
		    return result;
	    },
	    py::arg("handle"), "Verify PythonTaskResultHandle can poll a backend-neutral Python handle.");

	m.def(
	    "flight_exchange_selected_attempt_dataplane_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};

		    FlightExchangeConfig config;
		    config.node_id = "node-a";
		    config.local_dirs = {local_dir};
		    config.expected_types = types;
		    FlightExchangeManager mgr(config, &context);

		    ExchangeContext exchange_context;
		    exchange_context.query_id = "q";
		    exchange_context.exchange_id =
		        "flight_exchange_selected_attempt_dataplane_" + UUID::ToString(UUID::GenerateRandomUUID());
		    auto exchange = mgr.CreateExchange(exchange_context, 1);
		    auto sink_handle = exchange->AddSink(0);
		    auto lost_instance = exchange->InstantiateSink(sink_handle, 0);
		    auto selected_instance = exchange->InstantiateSink(sink_handle, 1);

		    auto write_attempt = [&](const ExchangeSinkInstanceHandle &instance, std::vector<int32_t> values) {
			    auto sink = mgr.CreateSink(instance);
			    DataChunk chunk;
			    chunk.Initialize(Allocator::DefaultAllocator(), types);
			    chunk.SetCardinality(values.size());
			    for (idx_t row = 0; row < values.size(); row++) {
				    chunk.SetValue(0, row, Value::INTEGER(values[row]));
			    }
			    auto add_res = sink->AddChunk(0, chunk);
			    if (add_res.is_err()) {
				    throw std::runtime_error(add_res.error().what());
			    }
			    auto finish_res = sink->Finish();
			    if (finish_res.is_err()) {
				    throw std::runtime_error(finish_res.error().what());
			    }
			    return mgr.config().node_id;
		    };

		    auto lost_node_id = write_attempt(lost_instance, {101});
		    lost_instance.flight_host = FlightExchangeManager::GetLocalFlightServerHost();
		    lost_instance.flight_server_epoch = FlightExchangeManager::GetLocalFlightServerEpoch();
		    auto selected_node_id = write_attempt(selected_instance, {201, 202});
		    selected_instance.flight_host = FlightExchangeManager::GetLocalFlightServerHost();
		    selected_instance.flight_server_epoch = FlightExchangeManager::GetLocalFlightServerEpoch();
		    const auto flight_port = FlightExchangeManager::GetLocalFlightServerPort();
		    exchange->SinkFinished(selected_instance, selected_node_id, flight_port);
		    exchange->AllRequiredSinksFinished();
		    auto handles = exchange->GetSourceHandles();

		    auto read_values = [&](const std::vector<ExchangeSourceHandle> &source_handles) {
			    auto source = mgr.CreateSource();
			    auto copied_handles = source_handles;
			    source->AddSourceHandles(std::move(copied_handles));
			    DataChunk output;
			    output.Initialize(Allocator::DefaultAllocator(), types);
			    py::list values;
			    while (source->ReadChunk(output)) {
				    for (idx_t row = 0; row < output.size(); row++) {
					    values.append(output.GetValue(0, row).GetValue<int32_t>());
				    }
			    }
			    return values;
		    };

		    auto selected_values_before_late_loser = read_values(handles);
		    exchange->SinkFinished(lost_instance, lost_node_id, flight_port);
		    auto selected_values_after_late_loser = read_values(handles);

		    auto manifest_exists = [&](const ExchangeSinkInstanceHandle &instance, const std::string &node_id) {
			    ShuffleCacheConfig cache_config;
			    cache_config.exchange_id = instance.output_location;
			    cache_config.node_id = node_id;
			    cache_config.num_partitions = 1;
			    cache_config.local_dirs = {local_dir};
			    ShuffleCache cache(cache_config);
			    return cache.HasCommittedManifest();
		    };

		    py::list handle_attempts;
		    py::list handle_paths;
		    py::list handle_node_ids;
		    for (const auto &handle : handles) {
			    handle_attempts.append(handle.attempt_id);
			    handle_paths.append(handle.files.empty() ? string() : handle.files[0].path);
			    handle_node_ids.append(handle.node_id);
		    }

		    py::dict out;
		    out["lost_output_location"] = lost_instance.output_location;
		    out["selected_output_location"] = selected_instance.output_location;
		    out["lost_node_id"] = lost_node_id;
		    out["selected_node_id"] = selected_node_id;
		    out["handle_attempts"] = handle_attempts;
		    out["handle_paths"] = handle_paths;
		    out["handle_node_ids"] = handle_node_ids;
		    out["selected_values_before_late_loser"] = selected_values_before_late_loser;
		    out["selected_values_after_late_loser"] = selected_values_after_late_loser;
		    out["lost_manifest_exists_after_late_loser"] = manifest_exists(lost_instance, lost_node_id);
		    out["selected_manifest_exists_after_late_loser"] = manifest_exists(selected_instance, selected_node_id);
		    exchange->Close();
		    auto cleanup = ShuffleCacheRegistry::Instance().RemoveAndCleanupByQuery(exchange_context.query_id);
		    if (cleanup.cleanup_errors != 0 || cleanup.cleanup_pending != 0) {
			    throw std::runtime_error("failed to clean selected-attempt dataplane test query");
		    }
		    auto retire = ShuffleCacheRegistry::Instance().RetireQuery(exchange_context.query_id);
		    if (retire.is_err()) {
			    throw std::runtime_error(retire.error().what());
		    }
		    return out;
	    },
	    py::arg("local_dir"), "Verify FlightExchangeSource reads only the selected retry attempt data.");

	m.def(
	    "flight_exchange_local_dirs_from_env_for_test",
	    []() {
		    using namespace duckdb::distributed;
		    py::list out;
		    for (const auto &dir : ResolveFlightExchangeLocalDirsFromEnv()) {
			    out.append(dir);
		    }
		    return out;
	    },
	    "Return parsed DUCKDB_SHUFFLE_DIRS values for regression tests.");

	m.def(
	    "flight_exchange_node_id_from_env_for_test",
	    []() { return duckdb::distributed::ResolveFlightExchangeNodeIdFromEnv(); },
	    "Return the resolved FlightExchange node id for regression tests.");

	m.def(
	    "flight_exchange_unselected_attempt_cleanup_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    const std::string node_id = "node-cleanup";
		    vector<LogicalType> types = {LogicalType::INTEGER};

		    FlightExchangeConfig config;
		    config.node_id = node_id;
		    config.local_dirs = {local_dir};
		    FlightExchangeManager mgr(config, &context);

		    ExchangeContext ctx;
		    ctx.query_id = "q";
		    ctx.exchange_id = "flight_exchange_cleanup_test";
		    auto exchange = mgr.CreateExchange(ctx, 2);
		    auto sink_handle = exchange->AddSink(0);
		    auto selected_instance = exchange->InstantiateSink(sink_handle, 0);
		    auto loser_instance = exchange->InstantiateSink(sink_handle, 1);

		    auto write_attempt = [&](const ExchangeSinkInstanceHandle &instance, int32_t value) {
			    auto sink = mgr.CreateSink(instance);
			    DataChunk chunk;
			    chunk.Initialize(Allocator::DefaultAllocator(), types);
			    chunk.SetCardinality(1);
			    chunk.SetValue(0, 0, Value::INTEGER(value));
			    auto add_res = sink->AddChunk(0, chunk);
			    if (add_res.is_err()) {
				    throw std::runtime_error(add_res.error().what());
			    }
			    auto finish_res = sink->Finish();
			    if (finish_res.is_err()) {
				    throw std::runtime_error(finish_res.error().what());
			    }
			    return mgr.config().node_id;
		    };

		    auto selected_node_id = write_attempt(selected_instance, 101);
		    selected_instance.flight_host = FlightExchangeManager::GetLocalFlightServerHost();
		    selected_instance.flight_server_epoch = FlightExchangeManager::GetLocalFlightServerEpoch();
		    const auto flight_port = FlightExchangeManager::GetLocalFlightServerPort();
		    exchange->SinkFinished(selected_instance, selected_node_id, flight_port);
		    auto loser_node_id = write_attempt(loser_instance, 202);
		    loser_instance.flight_host = FlightExchangeManager::GetLocalFlightServerHost();
		    loser_instance.flight_server_epoch = FlightExchangeManager::GetLocalFlightServerEpoch();

		    auto make_cache = [&](const std::string &output_location, const std::string &cache_node_id) {
			    ShuffleCacheConfig cache_config;
			    cache_config.exchange_id = output_location;
			    cache_config.node_id = cache_node_id;
			    cache_config.num_partitions = 2;
			    cache_config.local_dirs = {local_dir};
			    return make_uniq<ShuffleCache>(std::move(cache_config));
		    };

		    auto selected_cache_before = make_cache(selected_instance.output_location, selected_node_id);
		    auto loser_cache_before = make_cache(loser_instance.output_location, loser_node_id);
		    auto selected_manifest_before = selected_cache_before->HasCommittedManifest();
		    auto loser_manifest_before = loser_cache_before->HasCommittedManifest();
		    auto selected_registry_before =
		        ShuffleCacheRegistry::Instance().Get(selected_instance.output_location) != nullptr;
		    auto loser_registry_before =
		        ShuffleCacheRegistry::Instance().Get(loser_instance.output_location) != nullptr;

		    exchange->SinkFinished(loser_instance, loser_node_id, flight_port);
		    exchange->AllRequiredSinksFinished();
		    auto handles = exchange->GetSourceHandles();

		    auto selected_cache_after = make_cache(selected_instance.output_location, selected_node_id);
		    auto loser_cache_after = make_cache(loser_instance.output_location, loser_node_id);
		    auto selected_manifest_after_cleanup = selected_cache_after->HasCommittedManifest();
		    auto loser_manifest_after_cleanup = loser_cache_after->HasCommittedManifest();
		    auto selected_registry_after_cleanup =
		        ShuffleCacheRegistry::Instance().Get(selected_instance.output_location) != nullptr;
		    auto loser_registry_after_cleanup =
		        ShuffleCacheRegistry::Instance().Get(loser_instance.output_location) != nullptr;

		    exchange->Close();
		    auto selected_cache_after_close = make_cache(selected_instance.output_location, selected_node_id);
		    auto selected_manifest_after_close = selected_cache_after_close->HasCommittedManifest();
		    auto selected_registry_after_close =
		        ShuffleCacheRegistry::Instance().Get(selected_instance.output_location) != nullptr;

		    py::list handle_paths;
		    py::list handle_attempts;
		    for (const auto &handle : handles) {
			    handle_attempts.append(handle.attempt_id);
			    handle_paths.append(handle.files.empty() ? string() : handle.files[0].path);
		    }

		    py::dict out;
		    out["selected_manifest_before"] = selected_manifest_before;
		    out["loser_manifest_before"] = loser_manifest_before;
		    out["selected_registry_before"] = selected_registry_before;
		    out["loser_registry_before"] = loser_registry_before;
		    out["selected_manifest_after_cleanup"] = selected_manifest_after_cleanup;
		    out["loser_manifest_after_cleanup"] = loser_manifest_after_cleanup;
		    out["selected_registry_after_cleanup"] = selected_registry_after_cleanup;
		    out["loser_registry_after_cleanup"] = loser_registry_after_cleanup;
		    out["selected_manifest_after_close"] = selected_manifest_after_close;
		    out["selected_registry_after_close"] = selected_registry_after_close;
		    out["selected_output_location"] = selected_instance.output_location;
		    out["loser_output_location"] = loser_instance.output_location;
		    out["handle_paths"] = handle_paths;
		    out["handle_attempts"] = handle_attempts;
		    auto cleanup = ShuffleCacheRegistry::Instance().RemoveAndCleanupByQuery(ctx.query_id);
		    if (cleanup.cleanup_errors != 0 || cleanup.cleanup_pending != 0) {
			    throw std::runtime_error("failed to clean unselected-attempt test query");
		    }
		    auto retire = ShuffleCacheRegistry::Instance().RetireQuery(ctx.query_id);
		    if (retire.is_err()) {
			    throw std::runtime_error(retire.error().what());
		    }
		    return out;
	    },
	    py::arg("local_dir"), "Exercise cleanup of successful unselected FlightExchange attempts.");

	m.def(
	    "shuffle_cache_registry_query_cleanup_for_test",
	    [](const string &local_dir) {
		    using namespace duckdb::distributed;
		    const string query_id = "query-cleanup";
		    const string keep_query_id = "query-keep";
		    const string node_id = "node-a";
		    const string cleanup_exchange = query_id + "_shuffle_1__sink_0__attempt_0";
		    const string keep_exchange = keep_query_id + "_shuffle_1__sink_0__attempt_0";

		    auto make_cache = [&](const string &exchange_id) {
			    ShuffleCacheConfig config;
			    config.exchange_id = exchange_id;
			    config.node_id = node_id;
			    config.num_partitions = 1;
			    config.local_dirs = {local_dir};
			    return std::make_shared<ShuffleCache>(std::move(config));
		    };

		    ShuffleCacheRegistry::Instance().RemoveAndCleanupByPrefix(query_id + "_");
		    ShuffleCacheRegistry::Instance().RemoveAndCleanupByPrefix(keep_query_id + "_");

		    duckdb::LocalFileSystem fs;
		    auto cleanup_node_dir = local_dir + "/shuffle_" + cleanup_exchange + "/node_" + node_id;
		    auto cleanup_partition_dir = cleanup_node_dir + "/partition_0";
		    fs.CreateDirectoriesRecursive(cleanup_partition_dir);
		    {
			    std::ofstream file(cleanup_partition_dir + "/batch.arrow", std::ios::out | std::ios::binary);
			    file << "data";
		    }
		    auto keep_node_dir = local_dir + "/shuffle_" + keep_exchange + "/node_" + node_id;
		    auto keep_partition_dir = keep_node_dir + "/partition_0";
		    fs.CreateDirectoriesRecursive(keep_partition_dir);
		    {
			    std::ofstream file(keep_partition_dir + "/batch.arrow", std::ios::out | std::ios::binary);
			    file << "keep";
		    }

		    auto cleanup_cache = make_cache(cleanup_exchange);
		    auto keep_cache = make_cache(keep_exchange);
		    auto cleanup_register =
		        ShuffleCacheRegistry::Instance().Register(cleanup_exchange, cleanup_cache, query_id);
		    auto keep_register = ShuffleCacheRegistry::Instance().Register(keep_exchange, keep_cache, keep_query_id);
		    if (cleanup_register.is_err() || keep_register.is_err()) {
			    throw std::runtime_error("failed to register query cleanup test caches");
		    }
		    ShuffleCacheRegistry::Instance().RemoveForDeferredCleanup(cleanup_exchange);
		    auto cleanup_registry_after_defer = ShuffleCacheRegistry::Instance().Get(cleanup_exchange) != nullptr;

		    auto cleanup_result = ShuffleCacheRegistry::Instance().RemoveAndCleanupByPrefix(query_id + "_");
		    py::dict out;
		    out["registry_entries_removed"] = cleanup_result.registry_entries_removed;
		    out["storage_entries_removed"] = cleanup_result.storage_entries_removed;
		    out["cleanup_errors"] = cleanup_result.cleanup_errors;
		    out["cleanup_registry_after_defer"] = cleanup_registry_after_defer;
		    out["cleanup_registry_after"] = ShuffleCacheRegistry::Instance().Get(cleanup_exchange) != nullptr;
		    out["keep_registry_after"] = ShuffleCacheRegistry::Instance().Get(keep_exchange) != nullptr;
		    out["cleanup_node_dir_exists_after"] = fs.DirectoryExists(cleanup_node_dir);
		    out["keep_node_dir_exists_after"] = fs.DirectoryExists(keep_node_dir);

		    ShuffleCacheRegistry::Instance().RemoveAndCleanupByPrefix(keep_query_id + "_");
		    return out;
	    },
	    py::arg("local_dir"));

	m.def(
	    "shuffle_cache_attempt_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb::distributed;
		    ShuffleCacheConfig config;
		    config.exchange_id = "shuffle_cache_manifest_test__sink_3__attempt_2";
		    config.node_id = "node-a";
		    config.num_partitions = 2;
		    config.local_dirs = {local_dir};
		    ShuffleCache cache(std::move(config));
		    ShufflePartitionFile file;
		    file.path = local_dir + "/partition_0/batch.arrow";
		    file.rows = 4;
		    file.bytes = 11;
		    auto reg_res = cache.RegisterPartitionFile(0, std::move(file));
		    if (reg_res.is_err()) {
			    throw std::runtime_error(reg_res.error().what());
		    }
		    auto manifest_res = cache.WriteAttemptManifest(3, 2);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }

		    std::ifstream manifest(cache.ManifestFilePath());
		    if (!manifest.good()) {
			    throw std::runtime_error("failed to read shuffle attempt manifest");
		    }
		    std::string contents((std::istreambuf_iterator<char>(manifest)), std::istreambuf_iterator<char>());

		    py::dict out;
		    out["manifest_path"] = cache.ManifestFilePath();
		    out["committed_path"] = cache.CommittedMarkerPath();
		    out["manifest"] = contents;
		    return out;
	    },
	    py::arg("local_dir"), "Exercise ShuffleCache attempt manifest writing for fast tests.");

	m.def(
	    "shuffle_cache_manifest_recovery_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = "shuffle_cache_manifest_recovery_test__sink_0__attempt_1";
		    write_config.node_id = "node-a";
		    write_config.num_partitions = 2;
		    write_config.local_dirs = {local_dir};

		    ShuffleCache writer(write_config);
		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(3);
		    input.SetValue(0, 0, Value::INTEGER(11));
		    input.SetValue(0, 1, Value::INTEGER(12));
		    input.SetValue(0, 2, Value::INTEGER(13));
		    auto write_res = writer.WriteChunk(context, input, 1, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer.FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto manifest_res = writer.WriteAttemptManifest(0, 1);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }

		    auto reader = std::make_shared<ShuffleCache>(write_config);
		    auto memory_files_res = reader->GetPartitionFiles(1);
		    if (memory_files_res.is_err()) {
			    throw std::runtime_error(memory_files_res.error().what());
		    }
		    auto manifest_files_res = reader->GetPartitionFilesFromManifest(1);
		    if (manifest_files_res.is_err()) {
			    throw std::runtime_error(manifest_files_res.error().what());
		    }
		    auto register_res = ShuffleCacheRegistry::Instance().Register(
		        write_config.exchange_id, reader, "shuffle-cache-manifest-recovery-query", "", 1);
		    if (register_res.is_err()) {
			    throw std::runtime_error(register_res.error().what());
		    }
		    ScopedShuffleCacheRegistrationForTest registration(write_config.exchange_id);
		    FlightExchangeConfig source_config;
		    source_config.node_id = write_config.node_id;
		    ExchangeSourceHandle handle;
		    handle.partition_id = 1;
		    handle.attempt_id = 1;
		    handle.node_id = write_config.node_id;
		    handle.files.push_back(ExchangeSourceFile(write_config.exchange_id, 0));
		    auto values = ReadIntegerExchangeSourceForTest(context, source_config, std::move(handle));

		    py::dict out;
		    out["memory_file_count"] = memory_files_res.value().files.size();
		    out["manifest_file_count"] = manifest_files_res.value().files.size();
		    out["row_count"] = values.size();
		    out["values"] = values;
		    out["manifest_path"] = reader->ManifestFilePath();
		    return out;
	    },
	    py::arg("local_dir"), "Exercise ShuffleCache manifest-backed recovery after in-memory registry loss.");

	m.def(
	    "shuffle_cache_uncommitted_files_invisible_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = "shuffle_cache_uncommitted_invisible_test__sink_0__attempt_1";
		    write_config.node_id = "node-a";
		    write_config.num_partitions = 1;
		    write_config.local_dirs = {local_dir};

		    ShuffleCache writer(write_config);
		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(2);
		    input.SetValue(0, 0, Value::INTEGER(51));
		    input.SetValue(0, 1, Value::INTEGER(52));
		    auto write_res = writer.WriteChunk(context, input, 0, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer.FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto partial_files_res = writer.GetPartitionFiles(0);
		    if (partial_files_res.is_err()) {
			    throw std::runtime_error(partial_files_res.error().what());
		    }

		    ShuffleCache reader(write_config);
		    auto manifest_files_res = reader.GetPartitionFilesFromManifest(0);
		    if (manifest_files_res.is_err()) {
			    throw std::runtime_error(manifest_files_res.error().what());
		    }
		    if (!manifest_files_res.value().files.empty()) {
			    throw std::runtime_error("uncommitted shuffle files unexpectedly became visible");
		    }

		    py::dict out;
		    out["partial_file_count"] = partial_files_res.value().files.size();
		    out["committed_manifest"] = reader.HasCommittedManifest();
		    out["recovered_row_count"] = manifest_files_res.value().total_rows;
		    return out;
	    },
	    py::arg("local_dir"), "Exercise that uncommitted shuffle files are invisible without committed manifest.");

	m.def(
	    "shuffle_cache_duckdb_filesystem_storage_roundtrip_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);
		    auto storage = MakeDuckDBFileSystemShuffleStorage(fs);

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    ShuffleCacheConfig config;
		    config.exchange_id = "shuffle_cache_duckdb_fs_storage_test__sink_0__attempt_1";
		    config.node_id = "node-a";
		    config.num_partitions = 1;
		    config.local_dirs = {local_dir};

		    ShuffleCache writer(config, storage);
		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(4);
		    input.SetValue(0, 0, Value::INTEGER(71));
		    input.SetValue(0, 1, Value::INTEGER(72));
		    input.SetValue(0, 2, Value::INTEGER(73));
		    input.SetValue(0, 3, Value::INTEGER(74));
		    auto write_res = writer.WriteChunk(context, input, 0, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer.FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto manifest_res = writer.WriteAttemptManifest(0, 1);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }

		    auto reader = std::make_shared<ShuffleCache>(config, storage);
		    auto manifest_files_res = reader->GetPartitionFilesFromManifest(0);
		    if (manifest_files_res.is_err()) {
			    throw std::runtime_error(manifest_files_res.error().what());
		    }
		    auto register_res = ShuffleCacheRegistry::Instance().Register(
		        config.exchange_id, reader, "shuffle-cache-duckdb-fs-storage-query", "", 1);
		    if (register_res.is_err()) {
			    throw std::runtime_error(register_res.error().what());
		    }
		    ScopedShuffleCacheRegistrationForTest registration(config.exchange_id);
		    FlightExchangeConfig source_config;
		    source_config.node_id = config.node_id;
		    ExchangeSourceHandle handle;
		    handle.partition_id = 0;
		    handle.attempt_id = 1;
		    handle.node_id = config.node_id;
		    handle.files.push_back(ExchangeSourceFile(config.exchange_id, 0));
		    auto values = ReadIntegerExchangeSourceForTest(context, source_config, std::move(handle));

		    py::dict out;
		    out["committed_manifest"] = reader->HasCommittedManifest();
		    out["manifest_file_count"] = manifest_files_res.value().files.size();
		    out["manifest_total_rows"] = manifest_files_res.value().total_rows;
		    out["row_count"] = values.size();
		    out["values"] = values;
		    out["manifest_tmp_exists"] = fs.FileExists(reader->ManifestFilePath() + ".tmp");
		    out["marker_tmp_exists"] = fs.FileExists(reader->CommittedMarkerPath() + ".tmp");
		    return out;
	    },
	    py::arg("local_dir"), "Exercise ShuffleCache roundtrip through the DuckDB FileSystem backed storage backend.");
}
} // namespace ray_bindings
} // namespace duckdb
