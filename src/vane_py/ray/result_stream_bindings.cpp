// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "ray_binding_internal.hpp"

namespace duckdb {
namespace ray_bindings {
namespace py = ::duckdb::py;

ResultPartitionStream::ResultPartitionStream(std::shared_ptr<duckdb::distributed::PlanResultStream> stream)
    : stream_(std::move(stream)) {
}

ResultPartitionStream::~ResultPartitionStream() {
	if (stream_) {
		stream_->ClearReadyCallback();
	}
}

auto ResultPartitionStream::PartitionToPyObject(const std::shared_ptr<duckdb::distributed::ResultPartition> &part)
    -> py::object {
	return duckdb::distributed::python::ray::ResultPartitionToPyObject(part);
}

auto ResultPartitionStream::next_nowait() -> py::object {
	if (!stream_) {
		throw py::stop_iteration();
	}

	duckdb::distributed::PlanResultStream::PollResult poll_result {
	    duckdb::distributed::PlanResultStream::PollState::EXHAUSTED, nullptr};
	std::exception_ptr stream_error;
	{
		py::gil_scoped_release release;
		lock_guard<mutex> guard(stream_mutex_);
		try {
			poll_result = stream_->try_next();
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
	if (poll_result.state == duckdb::distributed::PlanResultStream::PollState::EXHAUSTED) {
		throw py::stop_iteration();
	}
	if (poll_result.state == duckdb::distributed::PlanResultStream::PollState::PENDING) {
		return py::none();
	}
	auto part = poll_result.partition;
	return PartitionToPyObject(part);
}

auto ResultPartitionStream::set_ready_callback(py::object loop, py::object callback) -> void {
	if (loop.is_none() || !py::hasattr(loop, "call_soon_threadsafe")) {
		throw py::type_error("result stream callback loop must provide call_soon_threadsafe()");
	}
	if (callback.is_none() || !PyCallable_Check(callback.ptr())) {
		throw py::type_error("result stream readiness callback must be callable");
	}
	lock_guard<mutex> guard(callback_mutex_);
	callback_loop_ = duckdb::distributed::python::ray::SafePyObject(std::move(loop));
	ready_callback_ = duckdb::distributed::python::ray::SafePyObject(std::move(callback));
}

auto ResultPartitionStream::arm_ready_notification() -> void {
	if (!stream_) {
		NotifyReady();
		return;
	}
	std::weak_ptr<ResultPartitionStream> weak_self = shared_from_this();
	stream_->NotifyWhenReady([weak_self]() {
		if (auto self = weak_self.lock()) {
			self->NotifyReady();
		}
	});
}

auto ResultPartitionStream::clear_ready_callback() -> void {
	if (stream_) {
		stream_->ClearReadyCallback();
	}
	lock_guard<mutex> guard(callback_mutex_);
	callback_loop_.reset_with_gil();
	ready_callback_.reset_with_gil();
}

auto ResultPartitionStream::NotifyReady() noexcept -> void {
	try {
		if (!duckdb::distributed::python::ray::SafePyObjectCanDecRef()) {
			return;
		}
		PythonGILWrapper gil;
		py::object loop;
		py::object callback;
		{
			lock_guard<mutex> guard(callback_mutex_);
			if (callback_loop_.empty() || ready_callback_.empty()) {
				return;
			}
			loop = callback_loop_.get();
			callback = ready_callback_.get();
		}
		loop.attr("call_soon_threadsafe")(callback);
	} catch (...) {
		// A closed Python loop is equivalent to an abandoned result waiter.
	}
}
} // namespace ray_bindings
} // namespace duckdb
