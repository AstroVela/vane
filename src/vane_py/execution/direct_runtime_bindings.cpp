// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "direct_task.hpp"
#include "direct_flight.hpp"
#include "materialized_exchange.hpp"
#include "file_snapshot.hpp"

#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/main/client_context.hpp"
#include "vane_python/pyconnection/pyconnection.hpp"
#include "vane_python/python_conversion.hpp"

#include <pybind11/stl.h>

namespace duckdb {
namespace {

using namespace vane_execution;

MaterializedObject ParseMaterializedObject(const py::dict &value) {
	if (value.size() != 4 || !value.contains("bytes") || !value.contains("rows") || !value.contains("frames") ||
	    !value.contains("sha256")) {
		throw py::value_error("invalid materialized object metadata");
	}
	MaterializedObject object;
	object.bytes = value["bytes"].cast<idx_t>();
	object.rows = value["rows"].cast<idx_t>();
	object.frames = value["frames"].cast<idx_t>();
	object.sha256 = value["sha256"].cast<string>();
	return object;
}

shared_ptr<MaterializedIO> Materialized(bool write, const string &path, shared_ptr<DirectChannel> channel,
                                        const string &identity, idx_t max_bytes, idx_t staging,
                                        MaterializedObject expected = {}) {
	py::gil_scoped_release release;
	return shared_ptr<MaterializedIO>(
	    new MaterializedIO(write, path, std::move(channel), identity, max_bytes, staging, std::move(expected)),
	    [](MaterializedIO *io) {
		    py::gil_scoped_release release;
		    delete io;
	    });
}

// No Python callable is retained by a channel or invoked from an engine thread.
struct DirectTestSignal {
	atomic<idx_t> count {0};
};

ExchangeWakeup TestWakeup(const shared_ptr<DirectTestSignal> &signal) {
	if (!signal) {
		return {};
	}
	auto weak = weak_ptr<DirectTestSignal>(signal);
	return [weak]() {
		auto value = weak.lock();
		if (value) {
			value->count++;
		}
	};
}

struct DirectBatchView {
	DataChunk chunk;
	idx_t sequence = 0;
	string producer;

	py::object Arrow(const vector<string> &names) {
		if (names.size() != chunk.ColumnCount()) {
			throw InvalidInputException("result column names do not match native schema");
		}
		ArrowArray array;
		array.Init();
		ArrowSchema schema;
		schema.Init();
		try {
			DirectFlight::ExportBatch(chunk, names, &array, &schema);
			return py::module_::import("pyarrow")
			    .attr("RecordBatch")
			    .attr("_import_from_c")(reinterpret_cast<uintptr_t>(&array), reinterpret_cast<uintptr_t>(&schema));
		} catch (...) {
			if (array.release) {
				array.release(&array);
			}
			if (schema.release) {
				schema.release(&schema);
			}
			throw;
		}
	}

	py::list Rows() {
		py::list rows;
		for (idx_t row = 0; row < chunk.size(); row++) {
			py::tuple result(chunk.ColumnCount());
			for (idx_t column = 0; column < chunk.ColumnCount(); column++) {
				result[column] = PythonObject::FromValue(chunk.GetValue(column, row), chunk.data[column].GetType(),
				                                         ClientProperties());
			}
			rows.append(std::move(result));
		}
		return rows;
	}
	unique_ptr<DirectBatchView> Slice(idx_t offset, idx_t count) {
		if (offset > chunk.size() || count > chunk.size() - offset) {
			throw InvalidInputException("batch slice is out of bounds");
		}
		auto view = make_uniq<DirectBatchView>();
		view->sequence = sequence;
		view->producer = producer;
		if (count) {
			view->chunk.InitializeEmpty(chunk.GetTypes());
			SelectionVector selection(count);
			for (idx_t row = 0; row < count; row++) {
				selection.set_index(row, offset + row);
			}
			view->chunk.Slice(chunk, selection, count);
		}
		return view;
	}
};

py::tuple Poll(DirectChannel &channel, const string &consumer, const shared_ptr<DirectTestSignal> &signal) {
	shared_ptr<DirectBatch> batch;
	DirectRead result;
	{
		py::gil_scoped_release release;
		result = channel.Poll(consumer, batch, TestWakeup(signal));
	}
	if (result == DirectRead::DATA) {
		auto view = make_uniq<DirectBatchView>();
		view->sequence = batch->sequence;
		view->producer = batch->producer;
		batch->Reference(view->chunk);
		return py::make_tuple("data", std::move(view));
	}
	return py::make_tuple(result == DirectRead::BLOCKED ? "blocked"
	                      : result == DirectRead::END   ? "end"
	                                                    : "closed",
	                      py::none());
}

string WriteRows(DirectChannel &channel, const string &producer, idx_t sequence, const py::list &rows,
                 const shared_ptr<DirectTestSignal> &signal) {
	if (rows.empty() || rows.size() > channel.limits.frame_rows) {
		throw InvalidInputException("test write requires between 1 and frame_rows rows");
	}
	DataChunk chunk;
	chunk.Initialize(Allocator::DefaultAllocator(), channel.types);
	vector<idx_t> selection;
	for (idx_t row = 0; row < rows.size(); row++) {
		auto values = rows[row].cast<py::sequence>();
		if (values.size() != channel.types.size()) {
			throw InvalidInputException("direct test row width mismatch");
		}
		for (idx_t col = 0; col < channel.types.size(); col++) {
			chunk.SetValue(col, row, TransformPythonValue(values[col], channel.types[col], false));
		}
		selection.push_back(row);
	}
	chunk.SetCardinality(rows.size());
	py::gil_scoped_release release;
	auto result = channel.TryWrite(producer, sequence, chunk, selection, 0, selection.size(), TestWakeup(signal));
	return result == DirectWrite::ACCEPTED ? "accepted" : result == DirectWrite::BLOCKED ? "blocked" : "closed";
}

py::dict Snapshot(const DirectChannel &channel) {
	auto value = channel.Snapshot();
	py::dict result;
	result["bytes"] = value.bytes;
	result["peak_bytes"] = value.peak_bytes;
	result["frames"] = value.frames;
	result["leased_bytes"] = value.leased_bytes;
	result["queued_frames"] = value.queued_frames;
	result["outstanding_frames"] = value.outstanding_frames;
	result["producers"] = value.producers;
	result["finished_producers"] = value.finished_producers;
	result["closed_consumers"] = value.closed_consumers;
	result["read_blocks"] = value.read_blocks;
	result["write_blocks"] = value.write_blocks;
	result["accepted_rows"] = value.accepted_rows;
	result["accepted_frames"] = value.accepted_frames;
	result["sealed"] = value.sealed;
	result["error"] = value.error;
	return result;
}

// Entry checks precede locks/GIL release so a Python filesystem callback cannot
// reenter the same native service while its outer task holds a context lock.
struct TaskEntry {
	TaskEntry() {
		DuckDBPyConnection::CheckCallbackEntry();
	}
};

void Prepare(DirectTaskService &service, const string &id, const string &payload, const string &connection_snapshot,
             const string &source_snapshot, const SourceAssignments &assignments, const py::dict &input_values,
             const py::list &output_values) {
	TaskEntry entry;
	std::map<string, vector<DirectInput>> inputs;
	for (auto item : input_values) {
		vector<DirectInput> port;
		for (auto endpoint : item.second.cast<py::list>()) {
			auto pair = endpoint.cast<py::tuple>();
			if (pair.size() != 2) {
				throw InvalidInputException("direct input requires channel and consumer");
			}
			port.push_back({pair[0].cast<shared_ptr<DirectChannel>>(), pair[1].cast<string>()});
		}
		inputs.emplace(item.first.cast<string>(), std::move(port));
	}
	vector<DirectOutput> outputs;
	for (auto item : output_values) {
		auto value = item.cast<py::dict>();
		DirectOutput output;
		output.channels = value["channels"].cast<vector<shared_ptr<DirectChannel>>>();
		output.producer = value["producer"].cast<string>();
		if (value.contains("partitioning") && !value["partitioning"].is_none()) {
			output.partitioning = value["partitioning"].cast<string>();
		}
		outputs.push_back(std::move(output));
	}
	py::gil_scoped_release release;
	service.Prepare(id, payload, connection_snapshot, source_snapshot, assignments, inputs, outputs);
}

} // namespace

void RegisterDirectRuntimeBindings(py::module_ &module) {
	auto runtime = module.def_submodule("execution_runtime", "Internal native DirectExchange and TaskService");
	runtime.def("check_entry", &DuckDBPyConnection::CheckCallbackEntry);
	py::class_<StoreGuard, shared_ptr<StoreGuard>>(runtime, "StoreGuard")
	    .def_static("acquire",
	                [](const string &path, bool exclusive, bool create) {
		                TaskEntry entry;
		                py::gil_scoped_release release;
		                return StoreGuard::Acquire(path, exclusive, create);
	                })
	    .def("close", [](StoreGuard &guard) {
		    TaskEntry entry;
		    py::gil_scoped_release release;
		    guard.Close();
	    });
	py::class_<MaterializedIO, shared_ptr<MaterializedIO>>(runtime, "MaterializedIO")
	    .def_static("staging_bytes", &MaterializedIO::StagingBytes)
	    .def_static("write",
	                [](const string &path, shared_ptr<DirectChannel> channel, const string &consumer, idx_t max_bytes,
	                   idx_t staging) {
		                TaskEntry entry;
		                return Materialized(true, path, std::move(channel), consumer, max_bytes, staging);
	                })
	    .def_static("read",
	                [](const string &path, shared_ptr<DirectChannel> channel, const string &producer, idx_t max_bytes,
	                   idx_t staging, const py::dict &expected) {
		                TaskEntry entry;
		                return Materialized(false, path, std::move(channel), producer, max_bytes, staging,
		                                    ParseMaterializedObject(expected));
	                })
	    .def_static("verify",
	                [](const string &path, const py::dict &expected, const string &schema) {
		                TaskEntry entry;
		                auto object = ParseMaterializedObject(expected);
		                py::gil_scoped_release release;
		                MaterializedIO::Verify(path, object, schema);
	                })
	    .def("status",
	         [](MaterializedIO &io) {
		         TaskEntry entry;
		         MaterializedStatus value;
		         {
			         py::gil_scoped_release release;
			         value = io.Status();
		         }
		         py::dict object;
		         object["bytes"] = value.object.bytes;
		         object["rows"] = value.object.rows;
		         object["frames"] = value.object.frames;
		         object["sha256"] = value.object.sha256;
		         py::dict result;
		         result["done"] = value.done;
		         result["error"] = value.error;
		         result["object"] = std::move(object);
		         return result;
	         })
	    .def("cancel",
	         [](MaterializedIO &io, const string &reason) {
		         TaskEntry entry;
		         py::gil_scoped_release release;
		         io.Cancel(reason);
	         })
	    .def("close", [](MaterializedIO &io) {
		    TaskEntry entry;
		    py::gil_scoped_release release;
		    io.Close();
	    });
	py::class_<DirectLimits>(runtime, "DirectLimits")
	    .def(py::init([](idx_t window_bytes, idx_t frame_bytes, idx_t frame_rows, idx_t frame_slots) {
		    DirectLimits limits {window_bytes, frame_bytes, frame_rows, frame_slots};
		    limits.Validate();
		    return limits;
	    }));
	py::class_<DirectTestSignal, shared_ptr<DirectTestSignal>>(runtime, "_TestSignal")
	    .def(py::init<>())
	    .def_property_readonly("count", [](DirectTestSignal &signal) { return signal.count.load(); });
	py::class_<DirectBatchView>(runtime, "DirectBatch")
	    .def("to_rows", &DirectBatchView::Rows)
	    .def("to_arrow", &DirectBatchView::Arrow)
	    .def("slice", &DirectBatchView::Slice)
	    .def("close", [](DirectBatchView &view) { view.chunk.Destroy(); })
	    .def_property_readonly("num_rows", [](DirectBatchView &view) { return view.chunk.size(); })
	    .def_readonly("producer", &DirectBatchView::producer)
	    .def_readonly("sequence", &DirectBatchView::sequence);
	py::class_<DirectChannel, shared_ptr<DirectChannel>>(runtime, "DirectChannel")
	    .def(py::init([](const string &schema, DirectLimits limits, idx_t producers, const vector<string> &consumers) {
		    return make_shared_ptr<DirectChannel>(DeserializeSchema(schema), limits, producers, consumers);
	    }))
	    .def("add_producer", &DirectChannel::AddProducer)
	    .def("seal_producers", &DirectChannel::SealProducers)
	    .def("finish", &DirectChannel::Finish)
	    .def("abort", &DirectChannel::Abort)
	    .def("close_consumer", &DirectChannel::CloseConsumer)
	    .def("poll", &Poll, py::arg("consumer"), py::arg("signal") = nullptr)
	    .def("_write_rows", &WriteRows, py::arg("producer"), py::arg("sequence"), py::arg("rows"),
	         py::arg("signal") = nullptr)
	    .def("snapshot", &Snapshot)
	    .def("producer_drained", &DirectChannel::ProducerDrained);
	runtime.def("arrow_schema", [](const string &encoded, const vector<string> &names) {
		ClientProperties properties;
		ArrowSchema schema;
		schema.Init();
		auto types = DeserializeSchema(encoded);
		if (types.size() != names.size()) {
			throw InvalidInputException("result column names do not match native schema");
		}
		DirectFlight::ExportSchema(types, names, &schema);
		try {
			return py::module_::import("pyarrow").attr("Schema").attr("_import_from_c")(
			    reinterpret_cast<uintptr_t>(&schema));
		} catch (...) {
			if (schema.release) {
				schema.release(&schema);
			}
			throw;
		}
	});
	py::class_<DirectFlight, shared_ptr<DirectFlight>>(runtime, "DirectFlight")
	    .def(py::init([](const string &host, const string &advertise, idx_t links, idx_t staging, idx_t frame, int port,
	                     const string &certificate, const string &key) {
		         TaskEntry entry;
		         py::gil_scoped_release release;
		         return shared_ptr<DirectFlight>(
		             new DirectFlight(host, advertise, links, staging, frame, port, certificate, key),
		             [](DirectFlight *service) {
			             py::gil_scoped_release release;
			             delete service;
		             });
	         }),
	         py::arg("host"), py::arg("advertise"), py::arg("links"), py::arg("staging"), py::arg("frame"),
	         py::arg("port") = 0, py::arg("certificate") = "", py::arg("private_key") = "")
	    .def_property_readonly("location", &DirectFlight::Location)
	    .def_property_readonly("error", &DirectFlight::Error)
	    .def_property_readonly("active_links", &DirectFlight::ActiveLinks)
	    .def_property_readonly("ready", &DirectFlight::Ready)
	    .def_static("staging_per_link", &DirectFlight::StagingPerLink)
	    .def(
	        "publish",
	        [](DirectFlight &service, const string &ticket, shared_ptr<DirectChannel> channel, const string &consumer) {
		        TaskEntry entry;
		        py::gil_scoped_release release;
		        service.Publish(ticket, std::move(channel), consumer);
	        })
	    .def(
	        "subscribe",
	        [](DirectFlight &service, const string &location, const string &ticket, shared_ptr<DirectChannel> channel,
	           const string &producer, double timeout, const string &roots) {
		        TaskEntry entry;
		        py::gil_scoped_release release;
		        service.Subscribe(location, ticket, std::move(channel), producer, timeout, roots);
	        },
	        py::arg("location"), py::arg("ticket"), py::arg("channel"), py::arg("producer"), py::arg("timeout"),
	        py::arg("root_certificates") = "")
	    .def("revoke",
	         [](DirectFlight &service, const string &ticket) {
		         TaskEntry entry;
		         py::gil_scoped_release release;
		         service.Revoke(ticket);
	         })
	    .def("delivered",
	         [](DirectFlight &service, const string &ticket) {
		         TaskEntry entry;
		         py::gil_scoped_release release;
		         return service.Delivered(ticket);
	         })
	    .def("cancel",
	         [](DirectFlight &service, const string &reason) {
		         TaskEntry entry;
		         py::gil_scoped_release release;
		         service.Cancel(reason);
	         })
	    .def("close", [](DirectFlight &service) {
		    TaskEntry entry;
		    py::gil_scoped_release release;
		    service.Close();
	    });
	py::class_<DirectTaskService, shared_ptr<DirectTaskService>>(runtime, "TaskService")
	    .def(py::init([](DuckDBPyConnection &connection) {
		    auto lock = DuckDBPyConnection::LockConnection(connection.py_connection_lock);
		    return shared_ptr<DirectTaskService>(new DirectTaskService(connection.con.GetConnection().context->db),
		                                         [](DirectTaskService *service) {
			                                         // Destruction can wait for native tasks using Python-backed I/O.
			                                         py::gil_scoped_release release;
			                                         delete service;
		                                         });
	    }))
	    .def("prepare", &Prepare)
	    .def("start",
	         [](DirectTaskService &service, const string &id, const string &token) {
		         TaskEntry entry;
		         py::gil_scoped_release release;
		         service.Start(id, token);
	         })
	    .def("pump",
	         [](DirectTaskService &service, idx_t steps) {
		         TaskEntry entry;
		         py::gil_scoped_release release;
		         return service.Pump(steps);
	         })
	    .def("cancel",
	         [](DirectTaskService &service, const string &reason) {
		         TaskEntry entry;
		         py::gil_scoped_release release;
		         service.Cancel(reason);
	         })
	    .def("expire",
	         [](DirectTaskService &service) {
		         TaskEntry entry;
		         py::gil_scoped_release release;
		         return service.Expire();
	         })
	    .def("release",
	         [](DirectTaskService &service) {
		         TaskEntry entry;
		         py::gil_scoped_release release;
		         service.Release();
	         })
	    .def("status",
	         [](DirectTaskService &service) {
		         TaskEntry entry;
		         vector<DirectTaskStatus> values;
		         {
			         py::gil_scoped_release release;
			         values = service.Status();
		         }
		         py::list result;
		         for (auto &value : values) {
			         py::dict item;
			         item["task_id"] = value.task_id;
			         item["state"] = value.state;
			         item["error"] = value.error;
			         item["released"] = value.released;
			         result.append(std::move(item));
		         }
		         return result;
	         })
	    .def("diagnostics",
	         [](DirectTaskService &service) {
		         TaskEntry entry;
		         vector<DirectTaskStatus> values;
		         {
			         py::gil_scoped_release release;
			         values = service.Diagnostics();
		         }
		         py::list result;
		         for (auto &value : values) {
			         py::dict item;
			         item["task_id"] = value.task_id;
			         item["state"] = value.state;
			         item["error"] = value.error;
			         item["released"] = value.released;
			         item["blocked_on"] = value.blocked_on;
			         item["builds_total"] = value.builds_total;
			         item["builds_ready"] = value.builds_ready;
			         result.append(std::move(item));
		         }
		         return result;
	         })
	    .def("production_status", [](DirectTaskService &service) {
		    TaskEntry entry;
		    DirectProducerStatus status;
		    {
			    py::gil_scoped_release release;
			    status = service.ProductionStatus();
		    }
		    py::dict result;
		    result["finished"] = status.finished;
		    result["error"] = status.error;
		    return result;
	    });
}

} // namespace duckdb
