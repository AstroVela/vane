// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Included by ray_module.cpp inside namespace duckdb.

#include <functional>

struct ResultPartitionStream {
	std::shared_ptr<duckdb::distributed::PlanResultStream> stream_;
	std::shared_ptr<void> keepalive_;
	std::function<void()> rethrow_pending_error_;
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
		std::exception_ptr stream_error;
		{
			// Release the GIL before contending on the stream mutex. The mutex
			// must also be released before restoring a Python submission error.
			py::gil_scoped_release release;
			lock_guard<mutex> guard(stream_mutex_);
			try {
				opt = stream_->next();
			} catch (...) {
				stream_error = std::current_exception();
			}
		}
		if (stream_error) {
			if (rethrow_pending_error_) {
				rethrow_pending_error_();
			}
			std::rethrow_exception(stream_error);
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
