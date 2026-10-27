// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "ray_binding_internal.hpp"

namespace duckdb {
namespace ray_bindings {
namespace py = ::duckdb::py;

void RegisterRayRecoveryTestsBindings(py::module_ &m) {
	m.def(
	    "flight_exchange_source_rejects_local_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const std::string output_location = "flight_exchange_source_manifest_recovery_test__sink_0__attempt_1";
		    const std::string node_id = "127.0.0.1";
		    const std::string server_epoch = "unpublished-local-manifest-epoch";

		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = output_location;
		    write_config.node_id = node_id;
		    write_config.num_partitions = 2;
		    write_config.local_dirs = {local_dir};
		    ShuffleCache writer(write_config);

		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(4);
		    input.SetValue(0, 0, Value::INTEGER(21));
		    input.SetValue(0, 1, Value::INTEGER(22));
		    input.SetValue(0, 2, Value::INTEGER(23));
		    input.SetValue(0, 3, Value::INTEGER(24));
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
		    ShuffleCacheRegistry::Instance().Remove(output_location);

		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = server_epoch;
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }

		    FlightExchangeConfig source_config;
		    source_config.node_id = node_id;
		    source_config.local_dirs = {local_dir};
		    source_config.expected_types = types;
		    FlightExchangeSource source(source_config, &context);

		    ExchangeSourceHandle handle;
		    handle.partition_id = 1;
		    handle.attempt_id = 1;
		    handle.node_id = node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = server_epoch;
		    ExchangeSourceFile file;
		    file.path = output_location;
		    file.file_size = 0;
		    handle.files.push_back(std::move(file));
		    std::vector<ExchangeSourceHandle> handles;
		    handles.push_back(std::move(handle));
		    source.AddSourceHandles(std::move(handles));

		    DataChunk output;
		    output.Initialize(Allocator::DefaultAllocator(), types);
		    py::list values;
		    while (source.ReadChunk(output)) {
			    for (idx_t row = 0; row < output.size(); row++) {
				    values.append(output.GetValue(0, row).GetValue<int32_t>());
			    }
		    }

		    py::dict out;
		    out["values"] = values;
		    out["finished"] = source.IsFinished();
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    return out;
	    },
	    py::arg("local_dir"),
	    "Verify that FlightExchangeSource rejects a local manifest missing published server identity.");

	m.def(
	    "flight_exchange_source_rejects_shared_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const std::string output_location =
		        "flight_exchange_source_shared_manifest_recovery_test__sink_0__attempt_1";
		    const std::string writer_node_id = "127.0.0.1";
		    const std::string reader_node_id = "reader-node";
		    const std::string server_epoch = "unpublished-shared-manifest-epoch";

		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = output_location;
		    write_config.node_id = writer_node_id;
		    write_config.num_partitions = 2;
		    write_config.local_dirs = {local_dir};
		    ShuffleCache writer(write_config);

		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(3);
		    input.SetValue(0, 0, Value::INTEGER(71));
		    input.SetValue(0, 1, Value::INTEGER(72));
		    input.SetValue(0, 2, Value::INTEGER(73));
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
		    ShuffleCacheRegistry::Instance().Remove(output_location);

		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = server_epoch;
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }

		    FlightExchangeConfig source_config;
		    source_config.node_id = reader_node_id;
		    source_config.local_dirs = {local_dir};
		    source_config.expected_types = types;
		    FlightExchangeSource source(source_config, &context);

		    ExchangeSourceHandle handle;
		    handle.partition_id = 1;
		    handle.attempt_id = 1;
		    handle.node_id = writer_node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = server_epoch;
		    ExchangeSourceFile file;
		    file.path = output_location;
		    file.file_size = 0;
		    handle.files.push_back(std::move(file));
		    std::vector<ExchangeSourceHandle> handles;
		    handles.push_back(std::move(handle));
		    source.AddSourceHandles(std::move(handles));

		    DataChunk output;
		    output.Initialize(Allocator::DefaultAllocator(), types);
		    py::list values;
		    while (source.ReadChunk(output)) {
			    for (idx_t row = 0; row < output.size(); row++) {
				    values.append(output.GetValue(0, row).GetValue<int32_t>());
			    }
		    }

		    py::dict out;
		    out["values"] = values;
		    out["finished"] = source.IsFinished();
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    out["writer_node_id"] = writer_node_id;
		    out["reader_node_id"] = reader_node_id;
		    return out;
	    },
	    py::arg("local_dir"),
	    "Verify that FlightExchangeSource rejects a shared manifest missing published server identity.");

	m.def(
	    "flight_exchange_source_write_unpublished_shared_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const std::string output_location = "flight_exchange_source_cross_process_manifest_test__sink_0__attempt_1";
		    const std::string writer_node_id = "127.0.0.1";
		    const idx_t sink_partition_id = 0;
		    const idx_t attempt_id = 1;
		    const idx_t output_partition_id = 1;

		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = output_location;
		    write_config.node_id = writer_node_id;
		    write_config.num_partitions = 2;
		    write_config.local_dirs = {local_dir};
		    ShuffleCache writer(write_config);

		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(3);
		    input.SetValue(0, 0, Value::INTEGER(81));
		    input.SetValue(0, 1, Value::INTEGER(82));
		    input.SetValue(0, 2, Value::INTEGER(83));
		    auto write_res = writer.WriteChunk(context, input, output_partition_id, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer.FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto manifest_res = writer.WriteAttemptManifest(sink_partition_id, attempt_id);
		    if (manifest_res.is_err()) {
			    throw std::runtime_error(manifest_res.error().what());
		    }

		    py::dict out;
		    out["output_location"] = output_location;
		    out["writer_node_id"] = writer_node_id;
		    out["partition_id"] = output_partition_id;
		    out["attempt_id"] = attempt_id;
		    out["manifest_path"] = writer.ManifestFilePath();
		    out["committed_path"] = writer.CommittedMarkerPath();
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    return out;
	    },
	    py::arg("local_dir"), "Write a committed but unpublished shared-manifest exchange attempt.");

	m.def(
	    "flight_exchange_source_rejects_unpublished_shared_manifest_for_test",
	    [](const std::string &local_dir, const std::string &output_location, const std::string &writer_node_id,
	       const std::string &reader_node_id, int64_t partition_id, int64_t attempt_id) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    if (partition_id < 0 || attempt_id < 0) {
			    throw std::runtime_error("partition_id and attempt_id must be non-negative");
		    }

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    const std::string server_epoch = "unpublished-cross-process-manifest-epoch";
		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = server_epoch;
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }
		    FlightExchangeConfig source_config;
		    source_config.node_id = reader_node_id;
		    source_config.local_dirs = {local_dir};
		    source_config.expected_types = types;
		    FlightExchangeSource source(source_config, &context);

		    ExchangeSourceHandle handle;
		    handle.partition_id = static_cast<idx_t>(partition_id);
		    handle.attempt_id = static_cast<idx_t>(attempt_id);
		    handle.node_id = writer_node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = server_epoch;
		    ExchangeSourceFile file;
		    file.path = output_location;
		    file.file_size = 0;
		    handle.files.push_back(std::move(file));
		    std::vector<ExchangeSourceHandle> handles;
		    handles.push_back(std::move(handle));
		    source.AddSourceHandles(std::move(handles));

		    DataChunk output;
		    output.Initialize(Allocator::DefaultAllocator(), types);
		    py::list values;
		    while (source.ReadChunk(output)) {
			    for (idx_t row = 0; row < output.size(); row++) {
				    values.append(output.GetValue(0, row).GetValue<int32_t>());
			    }
		    }

		    py::dict out;
		    out["values"] = values;
		    out["finished"] = source.IsFinished();
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    out["writer_node_id"] = writer_node_id;
		    out["reader_node_id"] = reader_node_id;
		    return out;
	    },
	    py::arg("local_dir"), py::arg("output_location"), py::arg("writer_node_id"), py::arg("reader_node_id"),
	    py::arg("partition_id"), py::arg("attempt_id"),
	    "Verify that a fresh process rejects a committed but unpublished shared-manifest exchange attempt.");

	m.def(
	    "flight_server_rejects_unpublished_shared_manifest_for_test",
	    [](const std::string &local_dir, const std::string &output_location, const std::string &writer_node_id,
	       int64_t partition_id) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    if (partition_id < 0) {
			    throw std::runtime_error("partition_id must be non-negative");
		    }

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = "shared-manifest-reader-epoch";
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }

		    FlightExchangeConfig source_config;
		    source_config.local_dirs = {local_dir};
		    source_config.node_id = "reader-node";
		    ExchangeSourceHandle handle;
		    handle.partition_id = static_cast<idx_t>(partition_id);
		    handle.attempt_id = 1;
		    handle.node_id = writer_node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = "shared-manifest-reader-epoch";
		    handle.files.push_back(ExchangeSourceFile(output_location, 0));
		    string read_error;
		    try {
			    ReadIntegerExchangeSourceForTest(context, source_config, std::move(handle));
		    } catch (const std::exception &ex) {
			    read_error = ex.what();
		    }
		    auto stop_res = server.Stop();
		    if (stop_res.is_err()) {
			    throw std::runtime_error(stop_res.error().what());
		    }

		    py::dict out;
		    out["fetch_error"] = !read_error.empty();
		    out["error"] = read_error;
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    return out;
	    },
	    py::arg("local_dir"), py::arg("output_location"), py::arg("writer_node_id"), py::arg("partition_id"),
	    "Verify that an unpublished manifest is not served by a fresh FlightServer process.");

	m.def(
	    "remote_exchange_source_rejects_local_dirs_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    const std::string current_node_id = "127.0.0.1";
		    const std::string server_epoch = "unpublished-serialized-manifest-epoch";

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const std::string output_location = "remote_exchange_source_local_dirs_roundtrip_test__sink_0__attempt_1";

		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = output_location;
		    write_config.node_id = current_node_id;
		    write_config.num_partitions = 2;
		    write_config.local_dirs = {local_dir};
		    ShuffleCache writer(write_config);

		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(2);
		    input.SetValue(0, 0, Value::INTEGER(41));
		    input.SetValue(0, 1, Value::INTEGER(42));
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
		    ShuffleCacheRegistry::Instance().Remove(output_location);

		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = server_epoch;
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }

		    Allocator allocator;
		    PhysicalPlan plan(allocator);
		    vector<idx_t> partition_indices = {1};
		    vector<string> source_nodes = {current_node_id};
		    std::vector<ExchangeSourceHandle> source_handles;
		    ExchangeSourceHandle handle;
		    handle.partition_id = 1;
		    handle.attempt_id = 1;
		    handle.node_id = current_node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = server_epoch;
		    ExchangeSourceFile file;
		    file.path = output_location;
		    file.file_size = 0;
		    handle.files.push_back(std::move(file));
		    source_handles.push_back(std::move(handle));

		    FlightExchangeConfig flight_config;
		    flight_config.node_id = current_node_id;
		    flight_config.local_dirs = {local_dir};
		    auto exchange_mgr = std::make_shared<FlightExchangeManager>(std::move(flight_config));

		    auto &source =
		        plan.Make<PhysicalRemoteExchangeSource>(types, 2, "remote_exchange_source_local_dirs_roundtrip_test",
		                                                partition_indices, source_handles, exchange_mgr, source_nodes);

		    MemoryStream stream(allocator);
		    SerializationOptions options;
		    BinarySerializer serializer(stream, options);
		    serializer.Begin();
		    source.Serialize(serializer);
		    serializer.End();

		    stream.Rewind();
		    BinaryDeserializer deserializer(stream);
		    deserializer.Begin();
		    auto deserialized_op = PhysicalOperator::Deserialize(deserializer, plan);
		    deserializer.End();
		    auto *source_ptr = dynamic_cast<PhysicalRemoteExchangeSource *>(deserialized_op.get());
		    if (!source_ptr) {
			    throw std::runtime_error("deserialized operator is not PhysicalRemoteExchangeSource");
		    }

		    ThreadContext thread_context(context);
		    ExecutionContext execution_context(context, thread_context, nullptr);
		    InterruptState interrupt_state;
		    auto global_state = source_ptr->GetGlobalSourceState(context);
		    auto local_state = source_ptr->GetLocalSourceState(execution_context, *global_state);
		    OperatorSourceInput source_input {*global_state, *local_state, interrupt_state};

		    DataChunk output;
		    output.Initialize(Allocator::DefaultAllocator(), types);
		    py::list values;
		    while (true) {
			    auto result = source_ptr->GetData(execution_context, output, source_input);
			    if (output.size() > 0) {
				    for (idx_t row = 0; row < output.size(); row++) {
					    values.append(output.GetValue(0, row).GetValue<int32_t>());
				    }
			    }
			    if (result == SourceResultType::FINISHED) {
				    break;
			    }
			    if (result == SourceResultType::BLOCKED) {
				    throw std::runtime_error("remote exchange source unexpectedly blocked");
			    }
		    }

		    py::dict out;
		    out["values"] = values;
		    out["node_id"] = current_node_id;
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    return out;
	    },
	    py::arg("local_dir"), "Verify that serialized EXCHANGE_SOURCE cannot bypass catalog identity via local_dirs.");

	m.def(
	    "flight_server_rejects_unpublished_manifest_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const std::string output_location = "flight_server_manifest_recovery_test__sink_0__attempt_1";
		    const std::string node_id = "node-a";

		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = output_location;
		    write_config.node_id = node_id;
		    write_config.num_partitions = 2;
		    write_config.local_dirs = {local_dir};
		    ShuffleCache writer(write_config);

		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(2);
		    input.SetValue(0, 0, Value::INTEGER(31));
		    input.SetValue(0, 1, Value::INTEGER(32));
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
		    ShuffleCacheRegistry::Instance().Remove(output_location);

		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = "unpublished-manifest-epoch";
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }

		    FlightExchangeConfig source_config;
		    source_config.local_dirs = {local_dir};
		    source_config.node_id = "reader-node";
		    ExchangeSourceHandle handle;
		    handle.partition_id = 1;
		    handle.attempt_id = 1;
		    handle.node_id = node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = "unpublished-manifest-epoch";
		    handle.files.push_back(ExchangeSourceFile(output_location, 0));
		    string read_error;
		    try {
			    ReadIntegerExchangeSourceForTest(context, source_config, std::move(handle));
		    } catch (const std::exception &ex) {
			    read_error = ex.what();
		    }
		    auto stop_res = server.Stop();
		    if (stop_res.is_err()) {
			    throw std::runtime_error(stop_res.error().what());
		    }

		    py::dict out;
		    out["fetch_error"] = !read_error.empty();
		    out["error"] = read_error;
		    out["registry_present"] = ShuffleCacheRegistry::Instance().Get(output_location) != nullptr;
		    return out;
	    },
	    py::arg("local_dir"), "Verify FlightServer rejects a manifest with no published catalog entry.");

	m.def(
	    "flight_server_uncommitted_attempt_rejected_for_test",
	    [](const std::string &local_dir) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;

		    vector<LogicalType> types = {LogicalType::INTEGER};
		    vector<string> names = {"value"};
		    const std::string output_location = "flight_server_uncommitted_rejected_test__sink_0__attempt_1";
		    const std::string node_id = "node-a";

		    ShuffleCacheConfig write_config;
		    write_config.exchange_id = output_location;
		    write_config.node_id = node_id;
		    write_config.num_partitions = 1;
		    write_config.local_dirs = {local_dir};
		    auto writer = std::make_shared<ShuffleCache>(write_config);

		    DataChunk input;
		    input.Initialize(Allocator::DefaultAllocator(), types);
		    input.SetCardinality(1);
		    input.SetValue(0, 0, Value::INTEGER(61));
		    auto write_res = writer->WriteChunk(context, input, 0, names);
		    if (write_res.is_err()) {
			    throw std::runtime_error(write_res.error().what());
		    }
		    auto flush_res = writer->FlushAll(context, names);
		    if (flush_res.is_err()) {
			    throw std::runtime_error(flush_res.error().what());
		    }
		    auto partial_files_res = writer->GetPartitionFiles(0);
		    if (partial_files_res.is_err()) {
			    throw std::runtime_error(partial_files_res.error().what());
		    }
		    const std::string server_epoch = "uncommitted-attempt-epoch";
		    auto register_res = ShuffleCacheRegistry::Instance().Register(
		        output_location, writer, "flight-server-uncommitted-rejected-query", server_epoch, 1);
		    if (register_res.is_err()) {
			    throw std::runtime_error(register_res.error().what());
		    }

		    FlightServerConfig server_config;
		    server_config.bind_host = "127.0.0.1";
		    server_config.port = 0;
		    server_config.server_epoch = server_epoch;
		    FlightServer server(std::move(server_config));
		    auto start_res = server.Start();
		    if (start_res.is_err()) {
			    throw std::runtime_error(start_res.error().what());
		    }

		    FlightExchangeConfig source_config;
		    source_config.local_dirs = {local_dir};
		    source_config.node_id = "reader-node";
		    ExchangeSourceHandle handle;
		    handle.partition_id = 0;
		    handle.attempt_id = 1;
		    handle.node_id = node_id;
		    handle.flight_host = "127.0.0.1";
		    handle.flight_port = server.port();
		    handle.flight_server_epoch = server_epoch;
		    handle.files.push_back(ExchangeSourceFile(output_location, 0));
		    string read_error;
		    try {
			    ReadIntegerExchangeSourceForTest(context, source_config, std::move(handle));
		    } catch (const std::exception &ex) {
			    read_error = ex.what();
		    }
		    auto stop_res = server.Stop();
		    ShuffleCacheRegistry::Instance().Remove(output_location);
		    if (stop_res.is_err()) {
			    throw std::runtime_error(stop_res.error().what());
		    }

		    py::dict out;
		    out["partial_file_count"] = partial_files_res.value().files.size();
		    out["committed_manifest"] = writer->HasCommittedManifest();
		    out["fetch_error"] = !read_error.empty();
		    out["error"] = read_error;
		    return out;
	    },
	    py::arg("local_dir"), "Exercise that FlightServer rejects uncommitted attempts without committed manifest.");
}
} // namespace ray_bindings
} // namespace duckdb
