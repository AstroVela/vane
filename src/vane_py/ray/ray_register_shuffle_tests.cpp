// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "ray_binding_internal.hpp"

namespace duckdb {
namespace ray_bindings {
namespace py = ::duckdb::py;

void RegisterRayShuffleTestsBindings(py::module_ &m) {
	m.def(
	    "shuffle_cache_duckdb_filesystem_storage_minio_roundtrip_for_test",
	    [](const std::string &base_uri, const std::string &endpoint_url, const std::string &access_key,
	       const std::string &secret_key, const std::string &region) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    ConfigureConnectionForS3Endpoint(conn, endpoint_url, access_key, secret_key, region);
		    auto &context = *conn.context;
		    DuckDBPyTransactionGuard tx_guard(context);
		    auto &fs = FileSystem::GetFileSystem(context);
		    auto storage = MakeDuckDBFileSystemShuffleStorage(fs);

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const auto run_id = UUID::ToString(UUID::GenerateRandomUUID());

		    ShuffleCacheConfig config;
		    config.exchange_id = "shuffle_cache_duckdb_fs_minio_storage_test_" + run_id + "__sink_0__attempt_1";
		    config.node_id = "node-a";
		    config.num_partitions = 1;
		    config.local_dirs = {base_uri};

		    ShuffleCache writer(config, storage);
		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(4);
		    input.SetValue(0, 0, Value::INTEGER(171));
		    input.SetValue(0, 1, Value::INTEGER(172));
		    input.SetValue(0, 2, Value::INTEGER(173));
		    input.SetValue(0, 3, Value::INTEGER(174));
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

		    const auto manifest_path = writer.ManifestFilePath();
		    const auto marker_path = writer.CommittedMarkerPath();
		    const auto manifest_exists_before_cleanup = fs.FileExists(manifest_path);
		    const auto marker_exists_before_cleanup = fs.FileExists(marker_path);
		    const auto manifest_tmp_exists = fs.FileExists(manifest_path + ".tmp");
		    const auto marker_tmp_exists = fs.FileExists(marker_path + ".tmp");

		    ShuffleCache reader(config, storage);
		    const auto committed_manifest_before_cleanup = reader.HasCommittedManifest();
		    auto manifest_files_res = reader.GetPartitionFilesFromManifest(0);
		    if (manifest_files_res.is_err()) {
			    throw std::runtime_error(manifest_files_res.error().what());
		    }
		    FlightExchangeConfig source_config;
		    source_config.local_dirs = {base_uri};
		    source_config.node_id = "reader-node";
		    ExchangeSourceHandle handle;
		    handle.partition_id = 0;
		    handle.attempt_id = 1;
		    handle.node_id = config.node_id;
		    handle.files.push_back(ExchangeSourceFile(config.exchange_id, 0));
		    auto values = ReadIntegerExchangeSourceForTest(context, source_config, std::move(handle));

		    auto cleanup_res = reader.RemoveAttemptStorage();
		    if (cleanup_res.is_err()) {
			    throw std::runtime_error(cleanup_res.error().what());
		    }

		    py::dict out;
		    out["base_uri"] = base_uri;
		    out["exchange_id"] = config.exchange_id;
		    out["manifest_path"] = manifest_path;
		    out["marker_path"] = marker_path;
		    out["committed_manifest"] = committed_manifest_before_cleanup;
		    out["manifest_exists_before_cleanup"] = manifest_exists_before_cleanup;
		    out["marker_exists_before_cleanup"] = marker_exists_before_cleanup;
		    out["manifest_file_count"] = manifest_files_res.value().files.size();
		    out["manifest_total_rows"] = manifest_files_res.value().total_rows;
		    out["manifest_total_bytes"] = manifest_files_res.value().total_bytes;
		    out["row_count"] = values.size();
		    out["values"] = values;
		    out["manifest_tmp_exists"] = manifest_tmp_exists;
		    out["marker_tmp_exists"] = marker_tmp_exists;
		    out["cleanup_removed"] = cleanup_res.value();
		    out["manifest_exists_after_cleanup"] = fs.FileExists(manifest_path);
		    out["marker_exists_after_cleanup"] = fs.FileExists(marker_path);
		    out["committed_manifest_after_cleanup"] = reader.HasCommittedManifest();
		    tx_guard.Commit();
		    return out;
	    },
	    py::arg("base_uri"), py::arg("endpoint_url"), py::arg("access_key"), py::arg("secret_key"),
	    py::arg("region") = "us-east-1",
	    "Exercise ShuffleCache DuckDB FileSystem storage against a real MinIO/S3-compatible endpoint.");

	m.def(
	    "shuffle_cache_duckdb_filesystem_storage_minio_fault_matrix_for_test",
	    [](const std::string &base_uri, const std::string &endpoint_url, const std::string &access_key,
	       const std::string &secret_key, const std::string &region) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    ConfigureConnectionForS3Endpoint(conn, endpoint_url, access_key, secret_key, region);
		    auto &context = *conn.context;
		    DuckDBPyTransactionGuard tx_guard(context);
		    auto &fs = FileSystem::GetFileSystem(context);
		    auto storage = MakeDuckDBFileSystemShuffleStorage(fs);

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const auto run_id = UUID::ToString(UUID::GenerateRandomUUID());
		    const std::string node_id = "node-a";

		    auto make_config = [&](const std::string &scenario) {
			    ShuffleCacheConfig config;
			    config.exchange_id =
			        "shuffle_cache_duckdb_fs_minio_fault_matrix_" + run_id + "_" + scenario + "__sink_0__attempt_1";
			    config.node_id = node_id;
			    config.num_partitions = 1;
			    config.local_dirs = {base_uri};
			    return config;
		    };

		    auto write_attempt = [&](const ShuffleCacheConfig &config, std::vector<int32_t> values) {
			    ShuffleCache writer(config, storage);
			    DataChunk input;
			    input.Initialize(Allocator::DefaultAllocator(), types);
			    input.SetCardinality(values.size());
			    for (idx_t row = 0; row < values.size(); row++) {
				    input.SetValue(0, row, Value::INTEGER(values[row]));
			    }
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
		    };

		    auto first_manifest_file_path = [&](const ShuffleCacheConfig &config) {
			    ShuffleCache cache(config, storage);
			    auto manifest_res =
			        ShuffleCache::ReadAttemptManifest(*storage, cache.ManifestFilePath(), cache.CommittedMarkerPath());
			    if (manifest_res.is_err()) {
				    throw std::runtime_error(manifest_res.error().what());
			    }
			    auto files_res = ShuffleCache::GetPartitionFilesFromManifest(manifest_res.value(), 0);
			    if (files_res.is_err()) {
				    throw std::runtime_error(files_res.error().what());
			    }
			    if (files_res.value().files.empty()) {
				    throw std::runtime_error("expected at least one shuffle file in manifest");
			    }
			    return files_res.value().files[0].path;
		    };

		    auto read_manifest_error_fresh = [&](const ShuffleCacheConfig &config) {
			    DuckDB verify_db(nullptr);
			    Connection verify_conn(verify_db);
			    ConfigureConnectionForS3Endpoint(verify_conn, endpoint_url, access_key, secret_key, region);
			    auto &verify_context = *verify_conn.context;
			    DuckDBPyTransactionGuard verify_guard(verify_context);
			    auto &verify_fs = FileSystem::GetFileSystem(verify_context);
			    auto verify_storage = MakeDuckDBFileSystemShuffleStorage(verify_fs);
			    ShuffleCache cache(config, verify_storage);
			    auto manifest_res = ShuffleCache::ReadAttemptManifest(*verify_storage, cache.ManifestFilePath(),
			                                                          cache.CommittedMarkerPath());
			    std::string error;
			    if (manifest_res.is_err()) {
				    error = manifest_res.error().what();
			    }
			    verify_guard.Commit();
			    return error;
		    };

		    auto read_source_error_fresh = [&](const ShuffleCacheConfig &config) {
			    DuckDB verify_db(nullptr);
			    Connection verify_conn(verify_db);
			    ConfigureConnectionForS3Endpoint(verify_conn, endpoint_url, access_key, secret_key, region);
			    auto &verify_context = *verify_conn.context;
			    DuckDBPyTransactionGuard verify_guard(verify_context);

			    FlightExchangeConfig source_config;
			    source_config.node_id = "reader-node";
			    source_config.local_dirs = {base_uri};
			    source_config.expected_types = types;
			    FlightExchangeSource source(source_config, &verify_context);

			    ExchangeSourceHandle handle;
			    handle.partition_id = 0;
			    handle.attempt_id = 1;
			    handle.node_id = config.node_id;
			    ExchangeSourceFile file;
			    file.path = config.exchange_id;
			    file.file_size = 0;
			    handle.files.push_back(std::move(file));
			    std::vector<ExchangeSourceHandle> handles;
			    handles.push_back(std::move(handle));
			    source.AddSourceHandles(std::move(handles));

			    DataChunk output;
			    output.Initialize(Allocator::DefaultAllocator(), types);
			    std::string error;
			    try {
				    while (source.ReadChunk(output)) {
				    }
			    } catch (const std::exception &ex) {
				    error = ex.what();
			    }
			    verify_guard.Commit();
			    return error;
		    };

		    auto cleanup_attempt = [&](const ShuffleCacheConfig &config) {
			    ShuffleCache cache(config, storage);
			    auto cleanup_res = cache.RemoveAttemptStorage();
			    if (cleanup_res.is_err()) {
				    throw std::runtime_error(cleanup_res.error().what());
			    }
			    return cleanup_res.value();
		    };

		    auto marker_missing_config = make_config("marker_missing");
		    write_attempt(marker_missing_config, {301, 302});
		    {
			    ShuffleCache cache(marker_missing_config, storage);
			    auto remove_res = storage->RemoveAll(cache.CommittedMarkerPath());
			    if (remove_res.is_err()) {
				    throw std::runtime_error(remove_res.error().what());
			    }
		    }

		    auto data_missing_config = make_config("data_missing");
		    write_attempt(data_missing_config, {401, 402});
		    const auto removed_data_file = first_manifest_file_path(data_missing_config);
		    {
			    auto remove_res = storage->RemoveAll(removed_data_file);
			    if (remove_res.is_err()) {
				    throw std::runtime_error(remove_res.error().what());
			    }
		    }

		    auto size_mismatch_config = make_config("size_mismatch");
		    write_attempt(size_mismatch_config, {501, 502});
		    {
			    ShuffleCache cache(size_mismatch_config, storage);
			    auto manifest_text_res = storage->ReadTextFile(cache.ManifestFilePath());
			    if (manifest_text_res.is_err()) {
				    throw std::runtime_error(manifest_text_res.error().what());
			    }
			    auto manifest_text = manifest_text_res.value();
			    auto file_line = manifest_text.find("file=");
			    if (file_line == std::string::npos) {
				    throw std::runtime_error("expected file line in shuffle attempt manifest");
			    }
			    auto first_tab = manifest_text.find('\t', file_line + 5);
			    auto second_tab = manifest_text.find('\t', first_tab + 1);
			    if (first_tab == std::string::npos || second_tab == std::string::npos) {
				    throw std::runtime_error("expected bytes field in shuffle attempt manifest file line");
			    }
			    const auto bytes_text = manifest_text.substr(first_tab + 1, second_tab - first_tab - 1);
			    const auto corrupted_bytes = std::stoull(bytes_text) + 17;
			    auto corrupted_manifest = manifest_text.substr(0, first_tab + 1) + std::to_string(corrupted_bytes) +
			                              manifest_text.substr(second_tab);
			    auto write_res = storage->WriteTextFileAtomically(cache.ManifestFilePath(), corrupted_manifest);
			    if (write_res.is_err()) {
				    throw std::runtime_error(write_res.error().what());
			    }
		    }

		    py::dict out;
		    out["marker_missing_manifest_error"] = read_manifest_error_fresh(marker_missing_config);
		    out["marker_missing_source_error"] = read_source_error_fresh(marker_missing_config);
		    out["marker_missing_cleanup_removed"] = cleanup_attempt(marker_missing_config);
		    out["data_missing_removed_file"] = removed_data_file;
		    out["data_missing_manifest_error"] = read_manifest_error_fresh(data_missing_config);
		    out["data_missing_source_error"] = read_source_error_fresh(data_missing_config);
		    out["data_missing_cleanup_removed"] = cleanup_attempt(data_missing_config);
		    out["size_mismatch_manifest_error"] = read_manifest_error_fresh(size_mismatch_config);
		    out["size_mismatch_source_error"] = read_source_error_fresh(size_mismatch_config);
		    out["size_mismatch_cleanup_removed"] = cleanup_attempt(size_mismatch_config);
		    tx_guard.Commit();
		    return out;
	    },
	    py::arg("base_uri"), py::arg("endpoint_url"), py::arg("access_key"), py::arg("secret_key"),
	    py::arg("region") = "us-east-1",
	    "Exercise MinIO/S3 durable shuffle fault matrix against committed manifest replay.");

	m.def(
	    "flight_exchange_minio_selected_attempt_replay_for_test",
	    [](const std::string &base_uri, const std::string &endpoint_url, const std::string &access_key,
	       const std::string &secret_key, const std::string &region) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    ConfigureConnectionForS3Endpoint(conn, endpoint_url, access_key, secret_key, region);
		    auto &context = *conn.context;
		    DuckDBPyTransactionGuard tx_guard(context);

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    const auto run_id = UUID::ToString(UUID::GenerateRandomUUID());

		    FlightExchangeConfig config;
		    config.node_id = "node-a";
		    config.local_dirs = {base_uri};
		    config.expected_types = types;
		    FlightExchangeManager mgr(config, &context);

		    ExchangeContext exchange_context;
		    exchange_context.query_id = "q";
		    exchange_context.exchange_id = "flight_exchange_minio_selected_attempt_" + run_id;
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
		    };

		    write_attempt(lost_instance, {701});
		    const auto lost_node_id = mgr.config().node_id;
		    write_attempt(selected_instance, {801, 802});
		    const auto selected_node_id = mgr.config().node_id;
		    exchange->SinkFinished(sink_handle, 1, selected_node_id, 0);
		    exchange->AllRequiredSinksFinished();
		    auto handles = exchange->GetSourceHandles();

		    auto read_source_values = [&](const std::vector<ExchangeSourceHandle> &source_handles) {
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

		    auto selected_values_before_cleanup = read_source_values(handles);

		    // The lost attempt committed data, but its status was not selected. If it
		    // is reported later, it must be treated as a loser and cleaned up.
		    exchange->SinkFinished(sink_handle, 0, lost_node_id, 0);

		    auto read_selected_values_fresh = [&]() {
			    DuckDB verify_db(nullptr);
			    Connection verify_conn(verify_db);
			    ConfigureConnectionForS3Endpoint(verify_conn, endpoint_url, access_key, secret_key, region);
			    auto &verify_context = *verify_conn.context;
			    DuckDBPyTransactionGuard verify_guard(verify_context);

			    FlightExchangeConfig verify_config;
			    verify_config.node_id = "reader-node";
			    verify_config.local_dirs = {base_uri};
			    verify_config.expected_types = types;
			    FlightExchangeSource source(verify_config, &verify_context);
			    auto copied_handles = handles;
			    source.AddSourceHandles(std::move(copied_handles));
			    DataChunk output;
			    output.Initialize(Allocator::DefaultAllocator(), types);
			    py::list values;
			    while (source.ReadChunk(output)) {
				    for (idx_t row = 0; row < output.size(); row++) {
					    values.append(output.GetValue(0, row).GetValue<int32_t>());
				    }
			    }
			    verify_guard.Commit();
			    return values;
		    };

		    auto lost_manifest_error_fresh = [&]() {
			    DuckDB verify_db(nullptr);
			    Connection verify_conn(verify_db);
			    ConfigureConnectionForS3Endpoint(verify_conn, endpoint_url, access_key, secret_key, region);
			    auto &verify_context = *verify_conn.context;
			    DuckDBPyTransactionGuard verify_guard(verify_context);
			    auto &verify_fs = FileSystem::GetFileSystem(verify_context);
			    auto verify_storage = MakeDuckDBFileSystemShuffleStorage(verify_fs);
			    ShuffleCacheConfig lost_config;
			    lost_config.exchange_id = lost_instance.output_location;
			    lost_config.node_id = lost_node_id;
			    lost_config.num_partitions = 1;
			    lost_config.local_dirs = {base_uri};
			    ShuffleCache lost_cache(lost_config, verify_storage);
			    auto manifest_res = ShuffleCache::ReadAttemptManifest(*verify_storage, lost_cache.ManifestFilePath(),
			                                                          lost_cache.CommittedMarkerPath());
			    std::string error;
			    if (manifest_res.is_err()) {
				    error = manifest_res.error().what();
			    }
			    verify_guard.Commit();
			    return error;
		    };

		    auto selected_values_after_cleanup = read_selected_values_fresh();
		    auto lost_manifest_after_cleanup_error = lost_manifest_error_fresh();

		    ShuffleCacheConfig selected_config;
		    selected_config.exchange_id = selected_instance.output_location;
		    selected_config.node_id = selected_node_id;
		    selected_config.num_partitions = 1;
		    selected_config.local_dirs = {base_uri};
		    auto &fs = FileSystem::GetFileSystem(context);
		    auto *opener = ClientData::Get(context).file_opener.get();
		    auto storage = MakeDuckDBFileSystemShuffleStorage(fs, opener);
		    ShuffleCache selected_cache(selected_config, storage);
		    auto selected_cleanup_res = selected_cache.RemoveAttemptStorage();
		    if (selected_cleanup_res.is_err()) {
			    throw std::runtime_error(selected_cleanup_res.error().what());
		    }

		    exchange->Close();

		    py::list handle_attempts;
		    py::list handle_paths;
		    py::list handle_node_ids;
		    for (const auto &handle : handles) {
			    handle_attempts.append(handle.attempt_id);
			    handle_node_ids.append(handle.node_id);
			    handle_paths.append(handle.files.empty() ? string() : handle.files[0].path);
		    }

		    py::dict out;
		    out["lost_output_location"] = lost_instance.output_location;
		    out["selected_output_location"] = selected_instance.output_location;
		    out["handle_attempts"] = handle_attempts;
		    out["handle_paths"] = handle_paths;
		    out["handle_node_ids"] = handle_node_ids;
		    out["selected_values_before_cleanup"] = selected_values_before_cleanup;
		    out["selected_values_after_loser_cleanup"] = selected_values_after_cleanup;
		    out["lost_manifest_after_cleanup_error"] = lost_manifest_after_cleanup_error;
		    out["selected_cleanup_removed"] = selected_cleanup_res.value();
		    auto cleanup = ShuffleCacheRegistry::Instance().RemoveAndCleanupByQuery(exchange_context.query_id, storage);
		    if (cleanup.cleanup_errors != 0 || cleanup.cleanup_pending != 0) {
			    throw std::runtime_error("failed to clean MinIO selected-attempt test query");
		    }
		    auto retire = ShuffleCacheRegistry::Instance().RetireQuery(exchange_context.query_id);
		    if (retire.is_err()) {
			    throw std::runtime_error(retire.error().what());
		    }
		    tx_guard.Commit();
		    return out;
	    },
	    py::arg("base_uri"), py::arg("endpoint_url"), py::arg("access_key"), py::arg("secret_key"),
	    py::arg("region") = "us-east-1",
	    "Exercise FTE-style selected-attempt replay and loser cleanup on real MinIO/S3 shuffle storage.");

	m.def(
	    "shuffle_cache_rejects_object_storage_local_dir_for_test",
	    []() {
		    using namespace duckdb::distributed;

		    py::dict out;
		    try {
			    ShuffleCacheConfig config;
			    config.exchange_id = "shuffle_cache_object_storage_reject_test__sink_0__attempt_1";
			    config.node_id = "node-a";
			    config.num_partitions = 1;
			    config.local_dirs = {"s3://bucket/shuffle"};
			    ShuffleCache cache(std::move(config));
			    out["rejected"] = false;
			    out["error"] = "";
		    } catch (const std::exception &ex) {
			    out["rejected"] = true;
			    out["error"] = ex.what();
		    }
		    return out;
	    },
	    "Exercise that object-storage shuffle dirs fail fast until the object backend exists.");

	m.def(
	    "shuffle_cache_duckdb_filesystem_storage_accepts_object_dir_for_test",
	    []() {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    py::dict out;
		    try {
			    DuckDB db(nullptr);
			    Connection conn(db);
			    auto &fs = FileSystem::GetFileSystem(*conn.context);
			    auto storage = MakeDuckDBFileSystemShuffleStorage(fs);

			    ShuffleCacheConfig config;
			    config.exchange_id = "shuffle_cache_duckdb_fs_object_accept_test__sink_0__attempt_1";
			    config.node_id = "node-a";
			    config.num_partitions = 1;
			    config.local_dirs = {"s3://bucket/shuffle"};
			    ShuffleCache cache(std::move(config), storage);
			    out["accepted"] = true;
			    out["error"] = "";
		    } catch (const std::exception &ex) {
			    out["accepted"] = false;
			    out["error"] = ex.what();
		    }
		    return out;
	    },
	    "Exercise that object-capable DuckDB FS storage passes object local_dirs constructor validation.");

	m.def(
	    "shuffle_cache_fake_object_no_rename_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &fs = FileSystem::GetFileSystem(*conn.context);

		    auto storage = std::make_shared<FakeObjectShuffleStorage>(fs);
		    auto data_dir = local_dir + "/partition_0";
		    if (!fs.DirectoryExists(data_dir)) {
			    fs.CreateDirectoriesRecursive(data_dir);
		    }
		    auto data_file = data_dir + "/batch.arrow";
		    const std::string payload = "payload";
		    {
			    std::ofstream output(data_file, std::ios::out | std::ios::binary);
			    output << payload;
			    output.close();
			    if (!output) {
				    throw std::runtime_error("failed to write fake object data file");
			    }
		    }

		    ShuffleCacheConfig config;
		    config.exchange_id = "shuffle_cache_fake_object_manifest_test__sink_0__attempt_7";
		    config.node_id = "node-a";
		    config.num_partitions = 1;
		    config.local_dirs = {local_dir};
		    ShuffleCache writer(config, storage);
		    ShufflePartitionFile file;
		    file.path = data_file;
		    file.rows = 1;
		    file.bytes = payload.size();
		    auto reg_res = writer.RegisterPartitionFile(0, std::move(file));
		    if (reg_res.is_err()) {
			    throw std::runtime_error(reg_res.error().what());
		    }
		    auto manifest_res = writer.WriteAttemptManifest(0, 7);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }

		    auto manifest_path = writer.ManifestFilePath();
		    auto marker_path = writer.CommittedMarkerPath();
		    ShuffleCache reader(config, storage);
		    auto manifest_files_res = reader.GetPartitionFilesFromManifest(0);
		    if (manifest_files_res.is_err()) {
			    throw std::runtime_error(manifest_files_res.error().what());
		    }

		    py::dict out;
		    out["committed_manifest"] = reader.HasCommittedManifest();
		    out["manifest_exists"] = storage->HasTextFile(manifest_path);
		    out["marker_exists"] = storage->HasTextFile(marker_path);
		    out["manifest_tmp_exists"] = storage->HasTextFile(manifest_path + ".tmp");
		    out["marker_tmp_exists"] = storage->HasTextFile(marker_path + ".tmp");
		    out["manifest"] = storage->TextFile(manifest_path);
		    out["manifest_file_count"] = manifest_files_res.value().files.size();
		    out["manifest_total_rows"] = manifest_files_res.value().total_rows;
		    out["manifest_total_bytes"] = manifest_files_res.value().total_bytes;
		    out["text_puts"] = storage->text_puts;
		    out["arrow_output_opens"] = storage->arrow_output_opens;
		    out["arrow_input_opens"] = storage->arrow_input_opens;
		    return out;
	    },
	    py::arg("local_dir"), "Exercise ShuffleCache manifest commit on a fake no-rename object storage backend.");
}
} // namespace ray_bindings
} // namespace duckdb
