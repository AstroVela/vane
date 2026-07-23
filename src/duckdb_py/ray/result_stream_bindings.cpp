// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Included by ray_module.cpp inside namespace duckdb.

struct ResultPartitionStream {
	std::shared_ptr<duckdb::distributed::PlanResultStream> stream_;
	std::shared_ptr<void> keepalive_;
	mutex stream_mutex_;

	explicit ResultPartitionStream(std::shared_ptr<duckdb::distributed::PlanResultStream> stream)
	    : stream_(std::move(stream)) {
	}

	py::object PartitionToPyObject(const std::shared_ptr<duckdb::distributed::ResultPartition> &part) {
		return duckdb::distributed::python::ray::ResultPartitionToPyObject(part);
	}

	py::object blocking_next() {
		if (!stream_) {
			throw py::stop_iteration();
		}

		std::pair<bool, duckdb::distributed::ResultPartitionRef> opt;
		{
			// Release the GIL before contending on the stream mutex. The mutex
			// must also be released before Python conversion restores the GIL.
			py::gil_scoped_release release;
			lock_guard<mutex> guard(stream_mutex_);
			opt = stream_->next();
		}
		if (!opt.first) {
			throw py::stop_iteration();
		}
		auto part = opt.second;
		return PartitionToPyObject(part);
	}
};

struct PlanRunState {
	std::shared_ptr<duckdb::distributed::PlanRunner> runner;
	duckdb::shared_ptr<duckdb::ClientContext> client_context;
	duckdb::distributed::python::ray::SafePyObject py_conn_keepalive;
};
