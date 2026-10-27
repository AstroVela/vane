// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "ray_binding_internal.hpp"

namespace duckdb {
namespace ray_bindings {
namespace py = ::duckdb::py;

void RegisterRayExecutionTestsBindings(py::module_ &m) {
	// Merge the independently scheduled splits assigned to one worker attempt

	// into its transport batch.

	m.def(
	    "merge_scan_split_batches",
	    [](const py::list &bytes_list) -> py::bytes {
		    using namespace duckdb::distributed;
		    if (py::len(bytes_list) == 0) {
			    throw py::value_error("merge_scan_split_batches requires at least one batch");
		    }
		    ScanSplitBatch merged;
		    bool has_batch = false;
		    for (auto item : bytes_list) {
			    py::bytes b = item.cast<py::bytes>();
			    string raw(b);
			    auto batch = ScanSplitBatch::DeserializeFromBytes(raw);
			    if (!has_batch) {
				    merged = std::move(batch);
				    has_batch = true;
			    } else {
				    merged.Merge(std::move(batch));
			    }
		    }
		    auto result = merged.SerializeToBytes();
		    return py::bytes(result);
	    },
	    py::arg("bytes_list"), "Merge compatible serialized scan split batches into one worker assignment.");

	m.def(
	    "split_scan_split_batch",
	    [](py::bytes bytes_obj) -> py::list {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto batch = ScanSplitBatch::DeserializeFromBytes(raw);
		    py::list result;
		    for (auto &singleton : batch.Explode()) {
			    const auto &split = singleton.splits[0];
			    py::object estimated_bytes = py::none();
			    if (split.estimated_bytes.IsValid()) {
				    estimated_bytes = py::int_(split.estimated_bytes.GetIndex());
			    }
			    result.append(py::make_tuple(py::str(split.split_id), py::bytes(singleton.SerializeToBytes()),
			                                 std::move(estimated_bytes)));
		    }
		    return result;
	    },
	    py::arg("bytes"),
	    "Explode a serialized scan split batch into (split_id, singleton_batch, estimated_bytes) tuples.");

	m.def(
	    "exchange_source_task_partition_indices",
	    [](py::bytes bytes_obj) {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto desc = ExchangeSourceTaskDescriptor::DeserializeFromBytes(raw);
		    return desc.partition_indices;
	    },
	    py::arg("bytes"), "Return partition_indices from a raw ExchangeSourceTaskDescriptor.");

	m.def(
	    "exchange_source_task_replicated",
	    [](py::bytes bytes_obj) {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto desc = ExchangeSourceTaskDescriptor::DeserializeFromBytes(raw);
		    return desc.replicated;
	    },
	    py::arg("bytes"), "Return whether a raw ExchangeSourceTaskDescriptor is replicated.");

	m.def(
	    "exchange_source_task_mark_join_build_summary_for_test",
	    [](py::bytes bytes_obj) {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto desc = ExchangeSourceTaskDescriptor::DeserializeFromBytes(raw);
		    py::dict out;
		    out["valid"] = desc.mark_join_build_summary.valid;
		    out["has_rows"] = desc.mark_join_build_summary.has_rows;
		    out["has_null"] = desc.mark_join_build_summary.has_null;
		    return out;
	    },
	    py::arg("bytes"), "Return MARK build summary metadata from an exchange source descriptor.");

	m.def(
	    "exchange_source_task_logical_identity",
	    [](py::bytes bytes_obj) {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto desc = ExchangeSourceTaskDescriptor::DeserializeFromBytes(raw);
		    std::set<idx_t> source_task_partition_ids;
		    for (const auto &handle : desc.source_handles) {
			    if (handle.source_task_partition_id == DConstants::INVALID_INDEX) {
				    throw py::value_error(
				        "exchange source handle is missing its stable source task partition identity");
			    }
			    source_task_partition_ids.insert(handle.source_task_partition_id);
		    }
		    if (source_task_partition_ids.empty()) {
			    throw py::value_error("exchange source task has no source task partition identities");
		    }

		    py::dict out;
		    out["partition_indices"] = desc.partition_indices;
		    out["source_task_partition_ids"] =
		        vector<idx_t>(source_task_partition_ids.begin(), source_task_partition_ids.end());
		    out["source_partition_count"] = desc.source_partition_count;
		    out["source_task_count"] = desc.source_task_count;
		    out["replicated"] = desc.replicated;
		    return out;
	    },
	    py::arg("bytes"),
	    "Return completion-order-independent logical identity metadata from an exchange source task.");

	m.def(
	    "read_committed_copy_direct_write_result",
	    [](const std::string &base_path, const std::string &run_id, py::object conn_obj) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    auto read_res = WithCopyRecoveryContext(std::move(conn_obj), [&](ClientContext &context) {
			    auto &fs = FileSystem::GetFileSystem(context);
			    return ReadCommittedDistributedCopyDirectWriteResult(fs, base_path, run_id);
		    });
		    if (read_res.is_err()) {
			    throw py::value_error(read_res.error().what());
		    }
		    auto result = std::move(read_res).value();

		    py::dict out;
		    out["rows_copied"] = result.rows_copied;
		    py::list files;
		    for (const auto &info : result.files) {
			    py::dict entry;
			    entry["staging_path"] = info.staging_path;
			    entry["worker_output_path"] = info.staging_path;
			    entry["final_path"] = info.final_path;
			    entry["row_count"] = info.row_count;
			    entry["file_size_bytes"] = info.file_size_bytes;
			    files.append(entry);
		    }
		    out["files"] = files;
		    AppendDistributedCopyResultMetadata(out, result);
		    return out;
	    },
	    py::arg("base_path"), py::arg("run_id"), py::arg("conn") = py::none(),
	    "Read a committed direct-write distributed COPY result through its committed manifest.");

	m.def(
	    "cleanup_uncommitted_copy_direct_write_run",
	    [](const std::string &base_path, const std::string &run_id) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    auto cleanup_res = CleanupDistributedCopyUncommittedDirectWriteRun(fs, base_path, run_id);
		    if (cleanup_res.is_err()) {
			    throw py::value_error(cleanup_res.error().what());
		    }
		    auto cleanup = std::move(cleanup_res).value();
		    py::dict out;
		    out["skipped_committed"] = cleanup.skipped_committed;
		    out["data_run_dir_existed"] = cleanup.data_run_dir_existed;
		    out["data_run_dir_removed"] = cleanup.data_run_dir_removed;
		    out["commit_dir_existed"] = cleanup.commit_dir_existed;
		    out["commit_dir_removed"] = cleanup.commit_dir_removed;
		    return out;
	    },
	    py::arg("base_path"), py::arg("run_id"),
	    "Cleanup a lifecycle-registered uncommitted direct-write distributed COPY run.");

	m.def(
	    "inspect_copy_direct_write_run",
	    [](const std::string &base_path, const std::string &run_id, py::object conn_obj) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    auto inspection_res = WithCopyRecoveryContext(std::move(conn_obj), [&](ClientContext &context) {
			    auto &fs = FileSystem::GetFileSystem(context);
			    return InspectDistributedCopyDirectWriteRun(fs, base_path, run_id);
		    });
		    if (inspection_res.is_err()) {
			    throw py::value_error(inspection_res.error().what());
		    }
		    auto inspection = std::move(inspection_res).value();
		    py::dict out;
		    switch (inspection.state) {
		    case DistributedCopyDirectWriteRunState::COMMITTED:
			    out["state"] = "COMMITTED";
			    break;
		    case DistributedCopyDirectWriteRunState::UNCOMMITTED:
			    out["state"] = "UNCOMMITTED";
			    break;
		    case DistributedCopyDirectWriteRunState::UNKNOWN:
			    out["state"] = "UNKNOWN";
			    break;
		    }
		    out["safe_to_retry"] = false;
		    out["error"] = inspection.error;
		    out["copy_output_base_path"] = inspection.base_path;
		    out["copy_output_run_id"] = inspection.run_id;
		    out["copy_output_commit_dir"] = inspection.paths.commit_dir;
		    out["copy_output_manifest_path"] = inspection.paths.manifest_path;
		    out["copy_output_committed_marker_path"] = inspection.paths.committed_marker_path;
		    out["copy_output_lifecycle_path"] = inspection.paths.lifecycle_path;
		    py::list files;
		    if (inspection.state == DistributedCopyDirectWriteRunState::COMMITTED) {
			    out["rows_copied"] = inspection.committed_result.rows_copied;
			    for (const auto &info : inspection.committed_result.files) {
				    py::dict entry;
				    entry["staging_path"] = info.staging_path;
				    entry["worker_output_path"] = info.staging_path;
				    entry["final_path"] = info.final_path;
				    entry["row_count"] = info.row_count;
				    entry["file_size_bytes"] = info.file_size_bytes;
				    files.append(entry);
			    }
		    } else {
			    out["rows_copied"] = py::none();
		    }
		    out["files"] = files;
		    return out;
	    },
	    py::arg("base_path"), py::arg("run_id"), py::arg("conn") = py::none(),
	    "Inspect a direct-write COPY run without making an uncommitted run safe to retry.");

	m.def(
	    "force_abort_copy_direct_write_run",
	    [](const std::string &base_path, const std::string &run_id, py::object conn_obj) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    auto abort_res = WithCopyRecoveryContext(std::move(conn_obj), [&](ClientContext &context) {
			    auto &fs = FileSystem::GetFileSystem(context);
			    return ForceAbortDistributedCopyDirectWriteRun(fs, base_path, run_id);
		    });
		    if (abort_res.is_err()) {
			    throw py::value_error(abort_res.error().what());
		    }
		    auto cleanup = std::move(abort_res).value();
		    py::dict out;
		    out["state"] = "ABORTED";
		    out["safe_to_retry"] = true;
		    out["copy_output_base_path"] = base_path;
		    out["copy_output_run_id"] = run_id;
		    out["data_run_dir_existed"] = cleanup.data_run_dir_existed;
		    out["data_run_dir_removed"] = cleanup.data_run_dir_removed;
		    out["commit_dir_existed"] = cleanup.commit_dir_existed;
		    out["commit_dir_removed"] = cleanup.commit_dir_removed;
		    return out;
	    },
	    py::arg("base_path"), py::arg("run_id"), py::arg("conn") = py::none(),
	    "Explicitly discard a shared-storage direct-write COPY run and verify that it is safe to retry.");

	m.def(
	    "register_copy_direct_write_run_lifecycle",
	    [](const std::string &base_path, const std::string &run_id, idx_t created_epoch_ms) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    DuckDB db(nullptr);
		    Connection conn(db);
		    auto &context = *conn.context;
		    auto &fs = FileSystem::GetFileSystem(context);

		    auto canonical_res = CanonicalDistributedCopyBasePath(fs, base_path);
		    if (canonical_res.is_err()) {
			    throw py::value_error(canonical_res.error().what());
		    }
		    auto canonical_base_path = std::move(canonical_res).value();
		    auto register_res =
		        WriteDistributedCopyDirectWriteLifecycle(fs, canonical_base_path, run_id, created_epoch_ms);
		    if (register_res.is_err()) {
			    throw py::value_error(register_res.error().what());
		    }
		    auto paths = BuildDistributedCopyFinalizeCommitPaths(fs, canonical_base_path, run_id);
		    py::dict out;
		    out["copy_output_base_path"] = canonical_base_path;
		    out["copy_output_run_id"] = run_id;
		    out["copy_output_commit_dir"] = paths.commit_dir;
		    out["copy_output_lifecycle_path"] = paths.lifecycle_path;
		    out["copy_output_committed_marker_path"] = paths.committed_marker_path;
		    return out;
	    },
	    py::arg("base_path"), py::arg("run_id"), py::arg("created_epoch_ms") = 0,
	    "Register lifecycle metadata for a direct-write distributed COPY run.");

	m.def(
	    "cleanup_expired_copy_direct_write_runs",
	    [](const std::string &base_path, idx_t min_age_ms, idx_t now_epoch_ms, py::object conn_obj) {
		    using namespace duckdb;
		    using namespace duckdb::distributed;

		    auto cleanup_res = WithCopyRecoveryContext(std::move(conn_obj), [&](ClientContext &context) {
			    auto &fs = FileSystem::GetFileSystem(context);
			    return CleanupExpiredDistributedCopyDirectWriteRuns(fs, base_path, min_age_ms, now_epoch_ms);
		    });
		    if (cleanup_res.is_err()) {
			    throw py::value_error(cleanup_res.error().what());
		    }
		    auto cleanup = std::move(cleanup_res).value();
		    py::dict out;
		    out["scanned_runs"] = cleanup.scanned_runs;
		    out["cleaned_runs"] = cleanup.cleaned_runs;
		    out["committed_runs"] = cleanup.committed_runs;
		    out["active_runs"] = cleanup.active_runs;
		    out["skipped_unregistered_runs"] = cleanup.skipped_unregistered_runs;
		    out["errors"] = cleanup.errors;
		    py::list cleaned_run_ids;
		    for (const auto &run_id : cleanup.cleaned_run_ids) {
			    cleaned_run_ids.append(run_id);
		    }
		    out["cleaned_run_ids"] = cleaned_run_ids;
		    py::list error_messages;
		    for (const auto &message : cleanup.error_messages) {
			    error_messages.append(message);
		    }
		    out["error_messages"] = error_messages;
		    return out;
	    },
	    py::arg("base_path"), py::arg("min_age_ms"), py::arg("now_epoch_ms") = 0, py::arg("conn") = py::none(),
	    "TTL cleanup scan for uncommitted direct-write distributed COPY runs.");

	m.def(
	    "make_exchange_source_task_descriptor_for_test",
	    [](py::list handles, py::list partition_indices, idx_t source_partition_count, idx_t source_task_count,
	       bool replicated, bool mark_join_build_summary_valid, bool mark_join_build_has_rows,
	       bool mark_join_build_has_null) {
		    using namespace duckdb::distributed;
		    ExchangeSourceTaskDescriptor desc;
		    for (auto item : partition_indices) {
			    desc.partition_indices.push_back(py::cast<idx_t>(item));
		    }
		    desc.source_partition_count = source_partition_count;
		    desc.source_task_count = source_task_count;
		    desc.replicated = replicated;
		    if (mark_join_build_summary_valid) {
			    desc.mark_join_build_summary =
			        MarkJoinBuildSummary::Create(mark_join_build_has_rows, mark_join_build_has_null);
			    if (!desc.mark_join_build_summary.IsValid()) {
				    throw py::value_error("invalid MARK join build summary");
			    }
		    } else if (mark_join_build_has_rows || mark_join_build_has_null) {
			    throw py::value_error("MARK join build summary values require a valid summary");
		    }

		    for (auto item : handles) {
			    py::dict d = item.cast<py::dict>();
			    ExchangeSourceHandle handle;
			    handle.partition_id = py::int_(d["partition_id"]).cast<idx_t>();
			    if (d.contains("source_task_partition_id")) {
				    handle.source_task_partition_id = py::int_(d["source_task_partition_id"]).cast<idx_t>();
			    }
			    if (d.contains("attempt_id")) {
				    handle.attempt_id = py::int_(d["attempt_id"]).cast<idx_t>();
			    }
			    if (d.contains("node_id")) {
				    handle.node_id = py::str(d["node_id"]).cast<string>();
			    }
			    if (d.contains("flight_host")) {
				    handle.flight_host = py::str(d["flight_host"]).cast<string>();
			    }
			    if (d.contains("flight_port")) {
				    handle.flight_port = py::int_(d["flight_port"]).cast<int>();
			    }
			    if (d.contains("flight_server_epoch")) {
				    handle.flight_server_epoch = py::str(d["flight_server_epoch"]).cast<string>();
			    }
			    if (d.contains("files")) {
				    auto files = py::reinterpret_borrow<py::iterable>(d["files"]);
				    for (auto file_item : files) {
					    py::dict fd = file_item.cast<py::dict>();
					    ExchangeSourceFile file;
					    file.path = py::str(fd["path"]).cast<string>();
					    if (fd.contains("rows")) {
						    file.rows = py::int_(fd["rows"]).cast<idx_t>();
					    }
					    if (fd.contains("file_size")) {
						    file.file_size = py::int_(fd["file_size"]).cast<size_t>();
					    }
					    handle.files.push_back(std::move(file));
				    }
			    }
			    desc.source_handles.push_back(std::move(handle));
		    }

		    auto bytes = desc.SerializeToBytes();
		    return py::bytes(bytes);
	    },
	    py::arg("handles"), py::arg("partition_indices"), py::arg("source_partition_count"),
	    py::arg("source_task_count"), py::arg("replicated") = false, py::arg("mark_join_build_summary_valid") = false,
	    py::arg("mark_join_build_has_rows") = false, py::arg("mark_join_build_has_null") = false,
	    "Build a raw ExchangeSourceTaskDescriptor for Python fast tests.");

	m.def(
	    "exchange_source_task_source_handles_for_test",
	    [](py::bytes bytes_obj) {
		    using namespace duckdb::distributed;
		    string raw(bytes_obj);
		    auto desc = ExchangeSourceTaskDescriptor::DeserializeFromBytes(raw);
		    py::list out;
		    for (const auto &handle : desc.source_handles) {
			    py::dict d;
			    d["partition_id"] = handle.partition_id;
			    if (handle.source_task_partition_id != DConstants::INVALID_INDEX) {
				    d["source_task_partition_id"] = handle.source_task_partition_id;
			    }
			    d["attempt_id"] = handle.attempt_id;
			    d["node_id"] = handle.node_id;
			    if (!handle.flight_host.empty()) {
				    d["flight_host"] = handle.flight_host;
			    }
			    d["flight_port"] = handle.flight_port;
			    if (!handle.flight_server_epoch.empty()) {
				    d["flight_server_epoch"] = handle.flight_server_epoch;
			    }
			    py::list files;
			    for (const auto &file : handle.files) {
				    py::dict fd;
				    fd["path"] = file.path;
				    if (file.rows > 0) {
					    fd["rows"] = file.rows;
				    }
				    fd["file_size"] = file.file_size;
				    files.append(fd);
			    }
			    d["files"] = files;
			    out.append(d);
		    }
		    return out;
	    },
	    py::arg("bytes"), "Return source handle metadata from a raw ExchangeSourceTaskDescriptor.");

	m.def(
	    "flight_exchange_selected_attempt_handles_for_test",
	    []() {
		    using namespace duckdb::distributed;
		    FlightExchangeConfig config;
		    config.node_id = "coordinator";
		    config.local_dirs = {"."};
		    FlightExchangeManager mgr(config, nullptr);

		    ExchangeContext ctx;
		    ctx.query_id = "q";
		    ctx.exchange_id = "flight_exchange_selected_attempt_test";
		    auto exchange = mgr.CreateExchange(ctx, 2);
		    auto sink0 = exchange->AddSink(0);
		    auto sink1 = exchange->AddSink(1);
		    auto sink0_attempt0 = exchange->InstantiateSink(sink0, 0);
		    auto sink0_attempt1 = exchange->InstantiateSink(sink0, 1);
		    auto sink1_attempt0 = exchange->InstantiateSink(sink1, 0);
		    sink0_attempt1.flight_host = "flight-retry.internal";
		    sink0_attempt0.flight_host = "flight-late.internal";
		    sink1_attempt0.flight_host = "flight-first.internal";
		    sink0_attempt1.flight_server_epoch = "worker-retry-epoch";
		    sink0_attempt0.flight_server_epoch = "worker-late-epoch";
		    sink1_attempt0.flight_server_epoch = "worker-first-epoch";
		    exchange->SinkFinished(sink0_attempt1, "worker-retry", 5010);
		    exchange->SinkFinished(sink0_attempt0, "worker-late", 5011);
		    exchange->SinkFinished(sink1_attempt0, "worker-first", 5012);
		    exchange->AllRequiredSinksFinished();
		    auto handles = exchange->GetSourceHandles();
		    exchange->Close();

		    py::list out;
		    for (const auto &handle : handles) {
			    py::dict d;
			    d["partition_id"] = handle.partition_id;
			    d["attempt_id"] = handle.attempt_id;
			    d["node_id"] = handle.node_id;
			    d["flight_host"] = handle.flight_host;
			    d["flight_port"] = handle.flight_port;
			    d["flight_server_epoch"] = handle.flight_server_epoch;
			    d["path"] = handle.files.empty() ? string() : handle.files[0].path;
			    out.append(d);
		    }
		    return out;
	    },
	    "Exercise FlightExchange selected-attempt source handle generation for fast tests.");

	m.def(
	    "flight_exchange_materialized_output_attempt_metadata_for_test",
	    []() {
		    using namespace duckdb::distributed;
		    FlightExchangeConfig config;
		    config.node_id = "coordinator";
		    config.local_dirs = {"."};
		    FlightExchangeManager mgr(config, nullptr);

		    ExchangeContext ctx;
		    ctx.query_id = "q";
		    ctx.exchange_id = "flight_exchange_output_metadata_test";
		    auto exchange = mgr.CreateExchange(ctx, 2);
		    auto sink0 = exchange->AddSink(0);
		    auto sink1 = exchange->AddSink(1);
		    exchange->InstantiateSink(sink0, 0);
		    auto sink0_attempt1 = exchange->InstantiateSink(sink0, 1);
		    auto sink1_attempt0 = exchange->InstantiateSink(sink1, 0);
		    sink0_attempt1.flight_host = "flight-retry.internal";
		    sink1_attempt0.flight_host = "flight-first.internal";
		    sink0_attempt1.flight_server_epoch = "worker-retry-epoch";
		    sink1_attempt0.flight_server_epoch = "worker-first-epoch";

		    std::vector<MaterializedOutput> outputs;
		    auto retry_output = MaterializedOutput({}, make_worker_id("worker-retry"));
		    retry_output.set_flight_port(5010);
		    retry_output.set_exchange_sink_instance(sink0_attempt1);
		    outputs.push_back(std::move(retry_output));

		    auto first_output = MaterializedOutput({}, make_worker_id("worker-first"));
		    first_output.set_flight_port(5012);
		    first_output.set_exchange_sink_instance(sink1_attempt0);
		    outputs.push_back(std::move(first_output));

		    RecordRemoteExchangeFinishedSinks(*exchange, outputs, "test");
		    exchange->AllRequiredSinksFinished();
		    auto handles = exchange->GetSourceHandles();
		    exchange->Close();

		    py::list out;
		    for (const auto &handle : handles) {
			    py::dict d;
			    d["partition_id"] = handle.partition_id;
			    d["attempt_id"] = handle.attempt_id;
			    d["node_id"] = handle.node_id;
			    d["flight_host"] = handle.flight_host;
			    d["flight_port"] = handle.flight_port;
			    d["flight_server_epoch"] = handle.flight_server_epoch;
			    d["path"] = handle.files.empty() ? string() : handle.files[0].path;
			    out.append(d);
		    }
		    return out;
	    },
	    "Verify MaterializedOutput sink attempt metadata drives remote exchange completion.");

	m.def(
	    "ray_task_result_handle_refreshed_worker_id_for_test",
	    [](py::object handle) {
		    using namespace duckdb::distributed;
		    using namespace duckdb::distributed::python::ray;

		    RayTaskResultHandle task_handle(TaskContext::from_node_context(0, 1, 0), handle,
		                                    make_worker_id("worker-original"));

		    std::string worker_id;
		    std::string exchange_sink_query_id;
		    bool has_output = false;
		    bool has_exchange_sink_instance = false;
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
				    has_exchange_sink_instance = output.has_exchange_sink_instance();
				    if (has_exchange_sink_instance) {
					    exchange_sink_query_id = output.exchange_sink_instance().query_id;
				    }
				    task_handle.AckPollResult();
				    task_handle.ReleasePollResult();
				    break;
			    }
		    }
		    if (worker_id.empty()) {
			    throw std::runtime_error("RayTaskResultHandle did not produce output before timeout");
		    }
		    pybind11::dict result;
		    result["worker_id"] = worker_id;
		    result["has_output"] = has_output;
		    result["flight_port"] = flight_port;
		    result["has_exchange_sink_instance"] = has_exchange_sink_instance;
		    result["exchange_sink_query_id"] = exchange_sink_query_id;
		    return result;
	    },
	    py::arg("handle"),
	    "Verify RayTaskResultHandle records the current Python handle worker_id at completion time.");

	m.def("_prepare_ray_task_result_poller_shutdown_race_for_test",
	      &duckdb::distributed::python::ray::PrepareRayTaskResultPollerShutdownRaceForTest, py::arg("handle"));

	m.def("_shutdown_ray_task_result_poller_for_test", &duckdb::distributed::python::ray::ShutdownRayTaskResultPoller);

	m.def(
	    "_ray_task_result_poller_batch_for_test",
	    [](py::list handles, int timeout_ms) {
		    using namespace duckdb::distributed;
		    using namespace duckdb::distributed::python::ray;

		    if (timeout_ms <= 0) {
			    throw pybind11::value_error("timeout_ms must be positive");
		    }

		    struct PollOutcome {
			    bool terminal = false;
			    bool is_error = false;
			    string error;
			    bool has_output = false;
			    string worker_id;
		    };

		    std::vector<std::unique_ptr<RayTaskResultHandle>> task_handles;
		    task_handles.reserve(handles.size());
		    for (size_t index = 0; index < handles.size(); ++index) {
			    auto handle = pybind11::reinterpret_borrow<pybind11::object>(handles[index]);
			    task_handles.push_back(make_uniq<RayTaskResultHandle>(
			        TaskContext::from_node_context(0, index + 1, index), std::move(handle),
			        make_worker_id("worker-original-" + std::to_string(index)),
			        "poller-test." + std::to_string(index)));
		    }

		    std::vector<PollOutcome> outcomes(task_handles.size());
		    {
			    pybind11::gil_scoped_release release;
			    size_t remaining = task_handles.size();
			    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
			    while (remaining > 0 && std::chrono::steady_clock::now() < deadline) {
				    bool had_progress = false;
				    for (size_t index = 0; index < task_handles.size(); ++index) {
					    auto &outcome = outcomes[index];
					    if (outcome.terminal) {
						    continue;
					    }
					    auto polled = task_handles[index]->poll();
					    if (!polled.first) {
						    continue;
					    }
					    outcome.terminal = true;
					    remaining--;
					    had_progress = true;
					    if (polled.second.is_err()) {
						    outcome.is_error = true;
						    outcome.error = polled.second.error().what();
					    } else {
						    auto payload = std::move(polled.second).value();
						    outcome.has_output = payload.first;
						    if (payload.second.worker_id()) {
							    outcome.worker_id = *payload.second.worker_id();
						    }
					    }
					    task_handles[index]->AckPollResult();
				    }
				    if (!had_progress && remaining > 0) {
					    std::this_thread::sleep_for(std::chrono::milliseconds(1));
				    }
			    }
		    }

		    pybind11::list result;
		    for (const auto &outcome : outcomes) {
			    pybind11::dict item;
			    item["terminal"] = outcome.terminal;
			    item["is_error"] = outcome.is_error;
			    item["error"] = outcome.error;
			    item["has_output"] = outcome.has_output;
			    item["worker_id"] = outcome.worker_id;
			    result.append(std::move(item));
		    }
		    return result;
	    },
	    py::arg("handles"), py::arg("timeout_ms") = 2000,
	    "Poll multiple RayTaskResultHandle instances through the shared native poller.");
}
} // namespace ray_bindings
} // namespace duckdb
