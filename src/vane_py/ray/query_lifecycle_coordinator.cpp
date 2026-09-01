// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "query_lifecycle_coordinator.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace duckdb {
namespace distributed {
namespace python {
namespace ray {

QueryLifecycleCoordinator::QueryLifecycleCoordinator(std::string component_name)
    : component_name_(std::move(component_name)) {
	if (component_name_.empty()) {
		throw std::invalid_argument("query lifecycle component name must not be empty");
	}
}

uint64_t QueryLifecycleCoordinator::NextToken(uint64_t &counter) {
	const auto token = counter++;
	if (counter == 0) {
		counter = 1;
	}
	return token;
}

bool QueryLifecycleCoordinator::SameLifecycle(const LifecycleRef &lhs, const LifecycleRef &rhs) {
	return lhs.owner_query_id == rhs.owner_query_id && lhs.generation == rhs.generation;
}

std::vector<std::string> QueryLifecycleCoordinator::OrderedQueryIds(const LifecycleState &lifecycle) {
	std::vector<std::string> query_ids(lifecycle.query_ids.begin(), lifecycle.query_ids.end());
	std::sort(query_ids.begin(), query_ids.end(), [&](const std::string &lhs, const std::string &rhs) {
		if ((lhs == lifecycle.ref.owner_query_id) != (rhs == lifecycle.ref.owner_query_id)) {
			return lhs != lifecycle.ref.owner_query_id;
		}
		return lhs < rhs;
	});
	return query_ids;
}

std::shared_ptr<QueryLifecycleCoordinator::LifecycleState>
QueryLifecycleCoordinator::FindLifecycleLocked(const LifecycleRef &lifecycle) const {
	auto current = lifecycles_by_owner_.find(lifecycle.owner_query_id);
	if (current == lifecycles_by_owner_.end() || current->second->ref.generation != lifecycle.generation) {
		return nullptr;
	}
	return current->second;
}

std::shared_ptr<QueryLifecycleCoordinator::LifecycleState>
QueryLifecycleCoordinator::ResolveLocked(const std::string &query_id) const {
	auto binding = query_bindings_.find(query_id);
	if (binding == query_bindings_.end()) {
		return nullptr;
	}
	return FindLifecycleLocked(binding->second);
}

std::shared_ptr<QueryLifecycleCoordinator::LifecycleState>
QueryLifecycleCoordinator::ResolveOwnerLocked(const std::string &owner_query_id) const {
	auto lifecycle = lifecycles_by_owner_.find(owner_query_id);
	return lifecycle == lifecycles_by_owner_.end() ? nullptr : lifecycle->second;
}

std::shared_ptr<QueryLifecycleCoordinator::LifecycleState>
QueryLifecycleCoordinator::CreateLifecycleLocked(const std::string &owner_query_id) {
	if (ResolveLocked(owner_query_id) || ResolveOwnerLocked(owner_query_id)) {
		throw std::runtime_error("cannot replace active FTE query lifecycle: " + owner_query_id);
	}
	auto lifecycle = std::make_shared<LifecycleState>();
	lifecycle->ref = LifecycleRef {owner_query_id, NextToken(next_generation_)};
	auto lifecycle_inserted = lifecycles_by_owner_.emplace(owner_query_id, lifecycle);
	if (!lifecycle_inserted.second) {
		throw std::runtime_error("cannot publish FTE query lifecycle: " + owner_query_id);
	}
	bool binding_published = false;
	try {
		auto binding_inserted = query_bindings_.emplace(owner_query_id, lifecycle->ref);
		if (!binding_inserted.second) {
			throw std::runtime_error("cannot publish FTE query lifecycle: " + owner_query_id);
		}
		binding_published = true;
		lifecycle->query_ids.insert(owner_query_id);
	} catch (...) {
		if (binding_published) {
			query_bindings_.erase(owner_query_id);
		}
		lifecycles_by_owner_.erase(lifecycle_inserted.first);
		throw;
	}
	return lifecycle;
}

void QueryLifecycleCoordinator::BindQueryLocked(const std::shared_ptr<LifecycleState> &lifecycle,
                                                const std::string &query_id) {
	auto binding = query_bindings_.find(query_id);
	if (binding != query_bindings_.end() && !SameLifecycle(binding->second, lifecycle->ref)) {
		throw std::runtime_error("FTE query owner changed while active: query=" + query_id + " existing=" +
		                         binding->second.owner_query_id + " requested=" + lifecycle->ref.owner_query_id);
	}
	if (binding != query_bindings_.end()) {
		lifecycle->query_ids.insert(query_id);
		return;
	}
	auto query_id_inserted = lifecycle->query_ids.insert(query_id);
	try {
		auto binding_inserted = query_bindings_.emplace(query_id, lifecycle->ref);
		if (!binding_inserted.second) {
			throw std::runtime_error("cannot publish FTE query binding: " + query_id);
		}
	} catch (...) {
		if (query_id_inserted.second) {
			lifecycle->query_ids.erase(query_id_inserted.first);
		}
		throw;
	}
}

QueryLifecycleCoordinator::Registration
QueryLifecycleCoordinator::BeginRegistration(const std::string &query_id, const std::string &owner_query_id) {
	if (query_id.empty() || owner_query_id.empty()) {
		throw std::invalid_argument("FTE query ownership requires non-empty query and owner IDs");
	}
	std::lock_guard<std::mutex> guard(mutex_);
	if (shutdown_started_) {
		throw std::runtime_error("cannot register FTE query ownership after " + component_name_ +
		                         " shutdown: " + query_id);
	}

	auto query_lifecycle = ResolveLocked(query_id);
	if (query_lifecycle && query_lifecycle->ref.owner_query_id != owner_query_id) {
		throw std::runtime_error("FTE query owner changed while active: query=" + query_id +
		                         " existing=" + query_lifecycle->ref.owner_query_id + " requested=" + owner_query_id);
	}
	auto owner_binding = ResolveLocked(owner_query_id);
	if (owner_binding && owner_binding->ref.owner_query_id != owner_query_id) {
		throw std::runtime_error("FTE resource query is already owned by another query: " + owner_query_id);
	}
	auto owner_lifecycle = ResolveOwnerLocked(owner_query_id);
	if (owner_binding && (!owner_lifecycle || !SameLifecycle(owner_binding->ref, owner_lifecycle->ref))) {
		throw std::runtime_error("FTE resource query lifecycle is inconsistent: " + owner_query_id);
	}
	if (query_lifecycle && owner_lifecycle && !SameLifecycle(query_lifecycle->ref, owner_lifecycle->ref)) {
		throw std::runtime_error("FTE query lifecycle generation changed while registering: " + query_id);
	}

	auto lifecycle = query_lifecycle ? query_lifecycle : owner_lifecycle;
	if (!lifecycle) {
		lifecycle = CreateLifecycleLocked(owner_query_id);
	}
	if (lifecycle->phase != Phase::OPEN) {
		throw std::runtime_error("cannot register closing FTE query lifecycle: " + query_id);
	}
	if (lifecycle->pending_registrations.find(query_id) != lifecycle->pending_registrations.end()) {
		throw std::runtime_error("FTE query lifecycle registration is already in progress: " + query_id);
	}
	if (lifecycle->registered_query_ids.find(query_id) != lifecycle->registered_query_ids.end()) {
		return Registration {lifecycle->ref, query_id, 0, false};
	}

	BindQueryLocked(lifecycle, query_id);
	const auto token = NextToken(next_registration_token_);
	lifecycle->pending_registrations[query_id] = token;
	return Registration {lifecycle->ref, query_id, token, true};
}

void QueryLifecycleCoordinator::CompleteRegistration(const Registration &registration, bool succeeded) {
	if (!registration.publish) {
		return;
	}
	{
		std::lock_guard<std::mutex> guard(mutex_);
		auto lifecycle = FindLifecycleLocked(registration.lifecycle);
		if (!lifecycle) {
			throw std::runtime_error("cannot finish stale FTE query registration: " + registration.query_id);
		}
		auto pending = lifecycle->pending_registrations.find(registration.query_id);
		if (pending == lifecycle->pending_registrations.end() || pending->second != registration.token) {
			throw std::runtime_error("cannot finish stale FTE query registration attempt: " + registration.query_id);
		}
		lifecycle->pending_registrations.erase(pending);
		if (succeeded) {
			lifecycle->registered_query_ids.insert(registration.query_id);
		} else if (lifecycle->phase == Phase::OPEN) {
			lifecycle->phase = Phase::CLOSING;
		}
	}
	condition_.notify_all();
}

void QueryLifecycleCoordinator::RegisterImmediate(const std::string &query_id, const std::string &owner_query_id) {
	auto registration = BeginRegistration(query_id, owner_query_id);
	CompleteRegistration(registration, true);
}

Optional<QueryLifecycleCoordinator::Operation>
QueryLifecycleCoordinator::BeginOperation(const std::string &query_id, const std::string &requested_owner_query_id,
                                          bool create_if_missing) {
	if (query_id.empty()) {
		throw std::invalid_argument("FTE query operation requires non-empty query_id");
	}
	std::lock_guard<std::mutex> guard(mutex_);
	if (shutdown_started_) {
		return nullopt;
	}

	auto lifecycle = ResolveLocked(query_id);
	if (!lifecycle && requested_owner_query_id.empty()) {
		return nullopt;
	}
	if (!lifecycle) {
		if (!create_if_missing) {
			return nullopt;
		}
		auto owner_binding = ResolveLocked(requested_owner_query_id);
		if (owner_binding && owner_binding->ref.owner_query_id != requested_owner_query_id) {
			throw std::runtime_error("FTE resource query is already owned by another query: " +
			                         requested_owner_query_id);
		}
		auto owner_lifecycle = ResolveOwnerLocked(requested_owner_query_id);
		if (owner_binding && (!owner_lifecycle || !SameLifecycle(owner_binding->ref, owner_lifecycle->ref))) {
			throw std::runtime_error("FTE resource query lifecycle is inconsistent: " + requested_owner_query_id);
		}
		lifecycle = owner_lifecycle ? owner_lifecycle : CreateLifecycleLocked(requested_owner_query_id);
		if (lifecycle->phase != Phase::OPEN) {
			return nullopt;
		}
		BindQueryLocked(lifecycle, query_id);
		lifecycle->registered_query_ids.insert(query_id);
	}

	const auto owner_query_id =
	    requested_owner_query_id.empty() ? lifecycle->ref.owner_query_id : requested_owner_query_id;
	if (owner_query_id.empty()) {
		throw std::invalid_argument("FTE query operation requires non-empty resource_query_id");
	}
	if (lifecycle->ref.owner_query_id != owner_query_id) {
		throw std::runtime_error("FTE query owner changed while active: query=" + query_id +
		                         " existing=" + lifecycle->ref.owner_query_id + " requested=" + owner_query_id);
	}
	if (lifecycle->phase != Phase::OPEN) {
		return nullopt;
	}
	lifecycle->active_operations++;
	return Operation {lifecycle->ref};
}

void QueryLifecycleCoordinator::EndOperation(const Operation &operation) noexcept {
	{
		std::lock_guard<std::mutex> guard(mutex_);
		auto lifecycle = FindLifecycleLocked(operation.lifecycle);
		if (!lifecycle || lifecycle->active_operations == 0) {
			std::terminate();
		}
		lifecycle->active_operations--;
	}
	condition_.notify_all();
}

void QueryLifecycleCoordinator::Close(const LifecycleRef &lifecycle_ref) {
	{
		std::lock_guard<std::mutex> guard(mutex_);
		auto lifecycle = FindLifecycleLocked(lifecycle_ref);
		if (lifecycle && lifecycle->phase == Phase::OPEN) {
			lifecycle->phase = Phase::CLOSING;
		}
	}
	condition_.notify_all();
}

void QueryLifecycleCoordinator::WaitForOperations(const LifecycleRef &lifecycle_ref) {
	std::unique_lock<std::mutex> guard(mutex_);
	condition_.wait(guard, [&]() {
		auto lifecycle = FindLifecycleLocked(lifecycle_ref);
		return !lifecycle || lifecycle->active_operations == 0;
	});
}

void QueryLifecycleCoordinator::WaitForAllOperations() {
	std::unique_lock<std::mutex> guard(mutex_);
	condition_.wait(guard, [&]() {
		return std::all_of(lifecycles_by_owner_.begin(), lifecycles_by_owner_.end(),
		                   [](const LifecycleMap::value_type &entry) { return entry.second->active_operations == 0; });
	});
}

std::string QueryLifecycleCoordinator::OwnerForQuery(const std::string &query_id) const {
	std::lock_guard<std::mutex> guard(mutex_);
	auto lifecycle = ResolveLocked(query_id);
	return lifecycle ? lifecycle->ref.owner_query_id : query_id;
}

bool QueryLifecycleCoordinator::IsClosing(const LifecycleRef &lifecycle_ref) const {
	std::lock_guard<std::mutex> guard(mutex_);
	auto lifecycle = FindLifecycleLocked(lifecycle_ref);
	return shutdown_started_ || !lifecycle || lifecycle->phase != Phase::OPEN;
}

bool QueryLifecycleCoordinator::IsQueryClosing(const std::string &query_id) const {
	std::lock_guard<std::mutex> guard(mutex_);
	auto lifecycle = ResolveLocked(query_id);
	return shutdown_started_ || !lifecycle || lifecycle->phase != Phase::OPEN;
}

std::vector<std::string> QueryLifecycleCoordinator::QueryIds(const LifecycleRef &lifecycle_ref) const {
	std::lock_guard<std::mutex> guard(mutex_);
	auto lifecycle = FindLifecycleLocked(lifecycle_ref);
	return lifecycle ? OrderedQueryIds(*lifecycle) : std::vector<std::string> {};
}

std::vector<std::string> QueryLifecycleCoordinator::OwnerQueryIds() const {
	std::lock_guard<std::mutex> guard(mutex_);
	std::vector<std::string> owners;
	owners.reserve(lifecycles_by_owner_.size());
	for (const auto &entry : lifecycles_by_owner_) {
		owners.push_back(entry.first);
	}
	std::sort(owners.begin(), owners.end());
	return owners;
}

void QueryLifecycleCoordinator::EnsureTransitionAllowedLocked(const std::string &query_id,
                                                              const char *operation) const {
	if (!shutdown_started_) {
		return;
	}
	if (shutdown_finished_) {
		throw std::runtime_error("cannot " + std::string(operation) + " FTE query after " + component_name_ +
		                         " shutdown: " + query_id);
	}
	if (shutdown_running_) {
		throw std::runtime_error("cannot " + std::string(operation) + " FTE query while " + component_name_ +
		                         " is shutting down: " + query_id);
	}
	throw std::runtime_error("cannot " + std::string(operation) + " FTE query after " + component_name_ +
	                         " shutdown failed: " + query_id);
}

void QueryLifecycleCoordinator::EnsureAttemptLeader(const Attempt &attempt, const std::string &operation,
                                                    const std::string &owner_query_id) {
	if (attempt.leader != std::this_thread::get_id()) {
		throw std::runtime_error("cannot " + operation + " from a non-leader thread: " + owner_query_id);
	}
}

void QueryLifecycleCoordinator::WaitForAttemptLocked(std::unique_lock<std::mutex> &guard,
                                                     const std::shared_ptr<Attempt> &attempt,
                                                     const std::string &operation, const std::string &owner_query_id) {
	if (attempt->leader == std::this_thread::get_id()) {
		throw std::runtime_error("cannot join a " + operation + " from its leader thread: " + owner_query_id);
	}
	attempt->waiters++;
	try {
		condition_.wait(guard, [&]() { return attempt->complete; });
	} catch (...) {
		attempt->waiters--;
		throw;
	}
	attempt->waiters--;
}

Optional<QueryLifecycleCoordinator::Abort>
QueryLifecycleCoordinator::BeginAbortLocked(std::unique_lock<std::mutex> &guard,
                                            const std::shared_ptr<LifecycleState> &lifecycle) {
	if (lifecycle->phase == Phase::QUIESCED || lifecycle->phase == Phase::DROPPING) {
		return nullopt;
	}
	if (lifecycle->abort_attempt) {
		auto attempt = lifecycle->abort_attempt;
		WaitForAttemptLocked(guard, attempt, "FTE query abort", lifecycle->ref.owner_query_id);
		if (attempt->error) {
			throw std::runtime_error(*attempt->error);
		}
		return nullopt;
	}

	if (lifecycle->phase == Phase::OPEN) {
		lifecycle->phase = Phase::CLOSING;
	}
	auto attempt = std::make_shared<Attempt>();
	attempt->token = NextToken(next_abort_token_);
	attempt->leader = std::this_thread::get_id();
	lifecycle->abort_attempt = attempt;
	lifecycle->phase = Phase::QUIESCING;
	condition_.wait(guard, [&]() { return lifecycle->pending_registrations.empty(); });
	return Abort {lifecycle->ref, OrderedQueryIds(*lifecycle), attempt->token, lifecycle->active_operations != 0};
}

Optional<QueryLifecycleCoordinator::Abort> QueryLifecycleCoordinator::BeginAbort(const std::string &query_id) {
	std::unique_lock<std::mutex> guard(mutex_);
	auto lifecycle = ResolveLocked(query_id);
	if (!lifecycle) {
		return nullopt;
	}
	if (lifecycle->abort_attempt) {
		return BeginAbortLocked(guard, lifecycle);
	}
	EnsureTransitionAllowedLocked(query_id, "abort");
	return BeginAbortLocked(guard, lifecycle);
}

Optional<QueryLifecycleCoordinator::Abort> QueryLifecycleCoordinator::BeginAbort(const Teardown &teardown) {
	std::unique_lock<std::mutex> guard(mutex_);
	auto lifecycle = FindLifecycleLocked(teardown.lifecycle);
	if (!lifecycle || !lifecycle->teardown_attempt || lifecycle->teardown_attempt->token != teardown.token) {
		throw std::runtime_error("cannot abort stale FTE query teardown: " + teardown.lifecycle.owner_query_id);
	}
	EnsureAttemptLeader(*lifecycle->teardown_attempt, "advance FTE query teardown", teardown.lifecycle.owner_query_id);
	return BeginAbortLocked(guard, lifecycle);
}

void QueryLifecycleCoordinator::CompleteAbort(const Abort &abort, const Optional<std::string> &error) {
	std::shared_ptr<Attempt> attempt;
	{
		std::lock_guard<std::mutex> guard(mutex_);
		auto lifecycle = FindLifecycleLocked(abort.lifecycle);
		if (!lifecycle || !lifecycle->abort_attempt || lifecycle->abort_attempt->token != abort.token) {
			throw std::runtime_error("cannot finish stale FTE query abort: " + abort.lifecycle.owner_query_id);
		}
		attempt = lifecycle->abort_attempt;
		EnsureAttemptLeader(*attempt, "finish FTE query abort", abort.lifecycle.owner_query_id);
		attempt->error = error;
		attempt->complete = true;
		lifecycle->abort_attempt.reset();
		lifecycle->phase = error ? Phase::CLOSING : Phase::QUIESCED;
	}
	condition_.notify_all();
}

Optional<QueryLifecycleCoordinator::Teardown> QueryLifecycleCoordinator::BeginTeardown(const std::string &query_id) {
	std::unique_lock<std::mutex> guard(mutex_);
	auto lifecycle = ResolveLocked(query_id);
	if (!lifecycle) {
		return nullopt;
	}
	if (lifecycle->teardown_attempt) {
		auto attempt = lifecycle->teardown_attempt;
		WaitForAttemptLocked(guard, attempt, "FTE query teardown", lifecycle->ref.owner_query_id);
		if (attempt->error) {
			throw std::runtime_error(*attempt->error);
		}
		return nullopt;
	}
	EnsureTransitionAllowedLocked(query_id, "tear down");
	if (lifecycle->phase == Phase::OPEN) {
		lifecycle->phase = Phase::CLOSING;
	}
	auto attempt = std::make_shared<Attempt>();
	attempt->token = NextToken(next_teardown_token_);
	attempt->leader = std::this_thread::get_id();
	lifecycle->teardown_attempt = attempt;
	condition_.wait(guard, [&]() { return lifecycle->pending_registrations.empty(); });
	return Teardown {lifecycle->ref, OrderedQueryIds(*lifecycle), attempt->token};
}

void QueryLifecycleCoordinator::MarkDropping(const Teardown &teardown) {
	std::lock_guard<std::mutex> guard(mutex_);
	auto lifecycle = FindLifecycleLocked(teardown.lifecycle);
	if (!lifecycle || !lifecycle->teardown_attempt || lifecycle->teardown_attempt->token != teardown.token) {
		throw std::runtime_error("cannot start stale FTE query drop: " + teardown.lifecycle.owner_query_id);
	}
	EnsureAttemptLeader(*lifecycle->teardown_attempt, "start FTE query drop", teardown.lifecycle.owner_query_id);
	if (lifecycle->phase != Phase::QUIESCED) {
		throw std::runtime_error("cannot drop FTE query before its abort barrier: " +
		                         teardown.lifecycle.owner_query_id);
	}
	if (lifecycle->active_operations != 0 || lifecycle->abort_attempt) {
		throw std::runtime_error("cannot drop FTE query with an active abort barrier: " +
		                         teardown.lifecycle.owner_query_id);
	}
	lifecycle->phase = Phase::DROPPING;
}

void QueryLifecycleCoordinator::CompleteTeardown(const Teardown &teardown, const Optional<std::string> &error) {
	std::shared_ptr<Attempt> attempt;
	{
		std::lock_guard<std::mutex> guard(mutex_);
		auto lifecycle = FindLifecycleLocked(teardown.lifecycle);
		if (!lifecycle || !lifecycle->teardown_attempt || lifecycle->teardown_attempt->token != teardown.token) {
			throw std::runtime_error("cannot finish stale FTE query teardown: " + teardown.lifecycle.owner_query_id);
		}
		attempt = lifecycle->teardown_attempt;
		EnsureAttemptLeader(*attempt, "finish FTE query teardown", teardown.lifecycle.owner_query_id);
		if (error) {
			attempt->error = error;
			attempt->complete = true;
			lifecycle->teardown_attempt.reset();
			if (lifecycle->phase == Phase::DROPPING) {
				lifecycle->phase = Phase::QUIESCED;
			} else if (lifecycle->phase != Phase::QUIESCED) {
				lifecycle->phase = Phase::CLOSING;
			}
		} else {
			if (lifecycle->phase != Phase::DROPPING) {
				throw std::runtime_error("cannot finish FTE query lifecycle before drop ownership: " +
				                         teardown.lifecycle.owner_query_id);
			}
			if (lifecycle->active_operations != 0) {
				throw std::runtime_error("cannot finish active FTE query lifecycle: " +
				                         teardown.lifecycle.owner_query_id);
			}
			if (!lifecycle->pending_registrations.empty()) {
				throw std::runtime_error("cannot finish FTE query lifecycle with active registrations: " +
				                         teardown.lifecycle.owner_query_id);
			}
			attempt->complete = true;
			for (const auto &owned_query_id : lifecycle->query_ids) {
				auto binding = query_bindings_.find(owned_query_id);
				if (binding != query_bindings_.end() && SameLifecycle(binding->second, lifecycle->ref)) {
					query_bindings_.erase(binding);
				}
			}
			lifecycles_by_owner_.erase(lifecycle->ref.owner_query_id);
		}
	}
	condition_.notify_all();
}

bool QueryLifecycleCoordinator::BeginShutdown() {
	std::unique_lock<std::mutex> guard(mutex_);
	while (shutdown_running_) {
		condition_.wait(guard);
	}
	if (shutdown_finished_) {
		return false;
	}
	shutdown_started_ = true;
	shutdown_running_ = true;
	for (auto &entry : lifecycles_by_owner_) {
		if (entry.second->phase == Phase::OPEN) {
			entry.second->phase = Phase::CLOSING;
		}
	}
	condition_.wait(guard, [&]() {
		return std::all_of(lifecycles_by_owner_.begin(), lifecycles_by_owner_.end(),
		                   [](const LifecycleMap::value_type &entry) {
			                   const auto &lifecycle = *entry.second;
			                   return lifecycle.pending_registrations.empty() && !lifecycle.abort_attempt &&
			                          !lifecycle.teardown_attempt;
		                   });
	});
	return true;
}

void QueryLifecycleCoordinator::FinishShutdown(bool succeeded) {
	{
		std::lock_guard<std::mutex> guard(mutex_);
		if (!shutdown_running_) {
			throw std::runtime_error("cannot finish inactive query lifecycle shutdown");
		}
		if (succeeded) {
			for (const auto &entry : lifecycles_by_owner_) {
				const auto &lifecycle = *entry.second;
				if (!lifecycle.pending_registrations.empty() || lifecycle.abort_attempt || lifecycle.teardown_attempt ||
				    lifecycle.active_operations != 0) {
					throw std::runtime_error("cannot finish query lifecycle shutdown with active transitions");
				}
			}
			query_bindings_.clear();
			lifecycles_by_owner_.clear();
		}
		shutdown_finished_ = succeeded;
		shutdown_running_ = false;
	}
	condition_.notify_all();
}

} // namespace ray
} // namespace python
} // namespace distributed
} // namespace duckdb
