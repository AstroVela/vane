// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/execution/media_resource.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/parallel/interrupt.hpp"

#include <limits>
#include <list>

namespace duckdb {

using MediaRequestList = std::list<shared_ptr<MediaResourceRequestState>>;

struct MediaResourceRequestState {
	MediaResourceRequestState(idx_t scope_id_p, MediaResourceBudget demand_p) : scope_id(scope_id_p), demand(demand_p) {
	}
	idx_t scope_id;
	MediaResourceBudget demand;
	MediaAdmissionStatus status = MediaAdmissionStatus::CANCELLED;
	std::function<void()> wakeup;
	MediaRequestList::iterator live_entry;
	MediaRequestList::iterator pending_entry;
};

struct MediaResourcePoolState {
	explicit MediaResourcePoolState(MediaResourceBudget capacity_p) : capacity(capacity_p) {
	}

	mutex lock;
	const MediaResourceBudget capacity;
	MediaResourceBudget reserved;
	idx_t next_scope = 0;
	idx_t active_leases = 0;
	MediaRequestList live;
	MediaRequestList pending;

	bool Fits(MediaResourceBudget demand) const {
		return demand.decoders <= capacity.decoders - reserved.decoders &&
		       demand.bytes <= capacity.bytes - reserved.bytes;
	}

	void Return(MediaResourceBudget demand) {
		D_ASSERT(demand.decoders <= reserved.decoders);
		D_ASSERT(demand.bytes <= reserved.bytes);
		reserved.decoders -= demand.decoders;
		reserved.bytes -= demand.bytes;
	}

	//! Splice already allocated list nodes: releasing a lease must not allocate
	//! notification storage or lose a wakeup when memory is exhausted.
	void Dispatch(MediaRequestList &notifications) {
		while (!pending.empty()) {
			auto &entry = *pending.front();
			if (!Fits(entry.demand)) {
				break;
			}
			reserved.decoders += entry.demand.decoders;
			reserved.bytes += entry.demand.bytes;
			entry.status = MediaAdmissionStatus::GRANTED;
			notifications.splice(notifications.end(), pending, pending.begin());
		}
	}

	void Cancel(MediaResourceRequestState &entry, MediaRequestList &notifications) {
		if (entry.status == MediaAdmissionStatus::WAITING) {
			pending.erase(entry.pending_entry);
		} else if (entry.status == MediaAdmissionStatus::GRANTED) {
			Return(entry.demand);
		} else {
			return;
		}
		entry.status = MediaAdmissionStatus::CANCELLED;
		notifications.splice(notifications.end(), live, entry.live_entry);
	}

	void Notify(MediaRequestList &notifications) noexcept {
		for (auto &entry : notifications) {
			std::function<void()> callback;
			{
				lock_guard<mutex> guard(lock);
				callback.swap(entry->wakeup);
			}
			if (callback) {
				try {
					callback();
				} catch (...) {
					// One failed/retired callback must not suppress the remaining
					// notifications or throw from resource-owner destruction.
				}
			}
		}
	}
};

MediaResourceLease::MediaResourceLease() {
}

MediaResourceLease::MediaResourceLease(shared_ptr<MediaResourcePoolState> pool_p, MediaResourceBudget budget_p)
    : pool(std::move(pool_p)), budget(budget_p) {
}

MediaResourceLease::~MediaResourceLease() {
	Release();
}

MediaResourceLease::MediaResourceLease(MediaResourceLease &&other) noexcept
    : pool(std::move(other.pool)), budget(other.budget) {
	other.budget = MediaResourceBudget();
}

MediaResourceLease &MediaResourceLease::operator=(MediaResourceLease &&other) noexcept {
	if (this != &other) {
		Release();
		pool = std::move(other.pool);
		budget = other.budget;
		other.budget = MediaResourceBudget();
	}
	return *this;
}

void MediaResourceLease::Release() noexcept {
	if (!pool) {
		return;
	}
	auto owner = std::move(pool);
	MediaRequestList notifications;
	{
		lock_guard<mutex> guard(owner->lock);
		owner->Return(budget);
		budget = MediaResourceBudget();
		D_ASSERT(owner->active_leases > 0);
		owner->active_leases--;
		owner->Dispatch(notifications);
	}
	owner->Notify(notifications);
}

MediaResourceLease::operator bool() const {
	return bool(pool);
}

MediaResourceBudget MediaResourceLease::Budget() const {
	return budget;
}

MediaAdmissionRequest::MediaAdmissionRequest(shared_ptr<MediaResourcePoolState> pool_p,
                                             shared_ptr<MediaResourceRequestState> request_p)
    : pool(std::move(pool_p)), request(std::move(request_p)) {
}

MediaAdmissionRequest::~MediaAdmissionRequest() {
	Cancel();
}

MediaAdmissionStatus MediaAdmissionRequest::PollOrArm(std::function<void()> wakeup) {
	if (!wakeup) {
		throw InvalidInputException("Media admission requires a readiness callback");
	}
	lock_guard<mutex> guard(pool->lock);
	if (request->status == MediaAdmissionStatus::WAITING) {
		request->wakeup.swap(wakeup);
	}
	return request->status;
}

MediaAdmissionStatus MediaAdmissionRequest::PollOrArm(const InterruptState &interrupt) {
	return PollOrArm([interrupt]() { interrupt.Callback(); });
}

MediaResourceLease MediaAdmissionRequest::TakeGrant() {
	lock_guard<mutex> guard(pool->lock);
	if (request->status != MediaAdmissionStatus::GRANTED) {
		throw InternalException("Media admission grant is not available");
	}
	request->status = MediaAdmissionStatus::TAKEN;
	pool->live.erase(request->live_entry);
	pool->active_leases++;
	return MediaResourceLease(pool, request->demand);
}

void MediaAdmissionRequest::Cancel() noexcept {
	MediaRequestList notifications;
	{
		lock_guard<mutex> guard(pool->lock);
		pool->Cancel(*request, notifications);
		pool->Dispatch(notifications);
	}
	pool->Notify(notifications);
}

MediaResourceScope::MediaResourceScope(shared_ptr<MediaResourcePoolState> pool_p, idx_t id_p)
    : pool(std::move(pool_p)), id(id_p) {
}

MediaResourceScope::~MediaResourceScope() {
	Close();
}

unique_ptr<MediaAdmissionRequest> MediaResourceScope::Request(MediaResourceBudget demand) {
	auto entry = make_shared_ptr<MediaResourceRequestState>(id, demand);
	auto result = unique_ptr<MediaAdmissionRequest>(new MediaAdmissionRequest(pool, entry));
	MediaRequestList notifications;
	{
		lock_guard<mutex> guard(pool->lock);
		if (closed) {
			entry->status = MediaAdmissionStatus::CANCELLED;
		} else if (demand.decoders > pool->capacity.decoders || demand.bytes > pool->capacity.bytes) {
			entry->status = MediaAdmissionStatus::IMPOSSIBLE;
		} else {
			pool->live.push_back(entry);
			entry->live_entry = --pool->live.end();
			try {
				pool->pending.push_back(entry);
			} catch (...) {
				pool->live.erase(entry->live_entry);
				entry->status = MediaAdmissionStatus::CANCELLED;
				throw;
			}
			entry->pending_entry = --pool->pending.end();
			entry->status = MediaAdmissionStatus::WAITING;
			pool->Dispatch(notifications);
		}
	}
	pool->Notify(notifications);
	return result;
}

void MediaResourceScope::Close() noexcept {
	MediaRequestList notifications;
	{
		lock_guard<mutex> guard(pool->lock);
		if (closed) {
			return;
		}
		closed = true;
		for (auto it = pool->live.begin(); it != pool->live.end();) {
			auto entry = *it++;
			if (entry->scope_id == id) {
				pool->Cancel(*entry, notifications);
			}
		}
		pool->Dispatch(notifications);
	}
	pool->Notify(notifications);
}

MediaResourcePool::MediaResourcePool(MediaResourceBudget capacity)
    : state(make_shared_ptr<MediaResourcePoolState>(capacity)) {
}

shared_ptr<MediaResourceScope> MediaResourcePool::CreateScope() {
	idx_t id;
	{
		lock_guard<mutex> guard(state->lock);
		if (state->next_scope == std::numeric_limits<idx_t>::max()) {
			throw OutOfRangeException("Media admission scope identifiers exhausted");
		}
		id = ++state->next_scope;
	}
	// shared_ptr construction may delete the scope if its control allocation
	// fails. Scope destruction takes the pool lock, so construct it unlocked.
	return shared_ptr<MediaResourceScope>(new MediaResourceScope(state, id));
}

MediaResourceSnapshot MediaResourcePool::Snapshot() const {
	lock_guard<mutex> guard(state->lock);
	MediaResourceSnapshot result;
	result.capacity = state->capacity;
	result.reserved = state->reserved;
	result.waiting_requests = state->pending.size();
	result.granted_requests = state->live.size() - state->pending.size();
	result.active_leases = state->active_leases;
	return result;
}

static idx_t NativeMediaCapacity(const char *variable) {
	const auto value = FileSystem::GetEnvVariable(variable);
	if (value.empty()) {
		return 0;
	}
	idx_t result = 0;
	for (auto ch : value) {
		if (ch < '0' || ch > '9' || result > (std::numeric_limits<idx_t>::max() - idx_t(ch - '0')) / 10) {
			throw InvalidInputException("%s must be a positive integer", variable);
		}
		result = result * 10 + idx_t(ch - '0');
	}
	if (result == 0) {
		throw InvalidInputException("%s must be a positive integer", variable);
	}
	return result;
}

shared_ptr<MediaResourcePool> GetNativeMediaDecoderPool() {
	static auto pool = []() -> shared_ptr<MediaResourcePool> {
		auto capacity = NativeMediaCapacity("VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES");
		return capacity ? make_shared_ptr<MediaResourcePool>(MediaResourceBudget(capacity, 0)) : nullptr;
	}();
	return pool;
}

shared_ptr<MediaResourcePool> GetNativeMediaOutputPool() {
	static auto pool = []() -> shared_ptr<MediaResourcePool> {
		const auto capacity = NativeMediaCapacity("VANE_NATIVE_MEDIA_MAX_OUTPUT_BYTES");
		return capacity ? make_shared_ptr<MediaResourcePool>(MediaResourceBudget(0, capacity)) : nullptr;
	}();
	return pool;
}

idx_t GetNativeMediaFlowCapacity() {
	static const auto capacity = NativeMediaCapacity("VANE_NATIVE_MEDIA_MAX_INFLIGHT_OUTPUT_BYTES");
	return capacity;
}

struct MediaFlowWindowState {
	explicit MediaFlowWindowState(idx_t bytes) : pool({0, bytes}), scope(pool.CreateScope()) {
	}
	MediaResourcePool pool;
	shared_ptr<MediaResourceScope> scope;
	mutex lock;
	bool closed = false;
	idx_t next_id = 0;
	unordered_map<idx_t, weak_ptr<MediaFlowTicket>> tickets;
};

struct MediaFlowTicket::State {
	State(MediaResourceLease lease_p, weak_ptr<MediaFlowWindowState> window_p, idx_t id_p)
	    : lease(std::move(lease_p)), window(std::move(window_p)), id(id_p) {
	}
	mutable mutex lock;
	MediaResourceLease lease;
	weak_ptr<MediaFlowWindowState> window;
	idx_t id;
};

MediaFlowTicket::MediaFlowTicket(MediaResourceLease lease, weak_ptr<MediaFlowWindowState> window, idx_t id)
    : state(make_uniq<State>(std::move(lease), std::move(window), id)) {
}

MediaFlowTicket::~MediaFlowTicket() {
	Acknowledge();
}

void MediaFlowTicket::Acknowledge() noexcept {
	MediaResourceLease returned;
	{
		lock_guard<mutex> guard(state->lock);
		returned = std::move(state->lease);
	}
	auto window = state->window.lock();
	if (window) {
		lock_guard<mutex> guard(window->lock);
		window->tickets.erase(state->id);
	}
	// Returning capacity may schedule a producer. Never invoke its callback
	// while holding the ticket, window or receiving queue lock.
	returned.Release();
}

bool MediaFlowTicket::IsActive() const {
	lock_guard<mutex> guard(state->lock);
	return bool(state->lease);
}

MediaFlowWindow::MediaFlowWindow(idx_t bytes) : state(make_shared_ptr<MediaFlowWindowState>(bytes)) {
}

MediaFlowWindow::~MediaFlowWindow() {
	Close();
}

unique_ptr<MediaAdmissionRequest> MediaFlowWindow::Request(idx_t bytes) {
	return state->scope->Request({0, bytes});
}

shared_ptr<MediaFlowTicket> MediaFlowWindow::TakeGrant(MediaAdmissionRequest &request) {
	MediaResourceLease lease;
	idx_t id;
	{
		lock_guard<mutex> guard(state->lock);
		if (state->closed) {
			return nullptr;
		}
		// Serialize with Close marking the window closed before it cancels
		// unclaimed grants. The request itself has one polling owner.
		if (state->next_id == std::numeric_limits<idx_t>::max()) {
			throw InternalException("Native media flow ticket identifier exhausted");
		}
		id = state->next_id++;
		lease = request.TakeGrant();
	}
	auto ticket = shared_ptr<MediaFlowTicket>(new MediaFlowTicket(std::move(lease), state, id));
	{
		lock_guard<mutex> guard(state->lock);
		if (!state->closed) {
			state->tickets.emplace(id, ticket);
			return ticket;
		}
	}
	// Close raced with taking a granted request. Do not export a live credit.
	ticket->Acknowledge();
	return nullptr;
}

MediaResourceSnapshot MediaFlowWindow::Snapshot() const {
	return state->pool.Snapshot();
}

void MediaFlowWindow::Close() noexcept {
	unordered_map<idx_t, weak_ptr<MediaFlowTicket>> tickets;
	{
		lock_guard<mutex> guard(state->lock);
		if (state->closed) {
			return;
		}
		state->closed = true;
		tickets.swap(state->tickets);
	}
	state->scope->Close();
	for (auto &entry : tickets) {
		auto ticket = entry.second.lock();
		if (ticket) {
			ticket->Acknowledge();
		}
	}
}

} // namespace duckdb
