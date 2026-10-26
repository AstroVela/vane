// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/shared_ptr.hpp"

#include <functional>

namespace duckdb {

class InterruptState;
struct MediaResourcePoolState;
struct MediaResourceRequestState;

//! Reservations, not a measurement or a limit on the process RSS.
struct MediaResourceBudget {
	MediaResourceBudget(idx_t decoders_p = 0, idx_t bytes_p = 0) : decoders(decoders_p), bytes(bytes_p) {
	}
	idx_t decoders;
	idx_t bytes;
};

enum class MediaAdmissionStatus : uint8_t { WAITING, GRANTED, TAKEN, CANCELLED, IMPOSSIBLE };

struct MediaResourceSnapshot {
	MediaResourceBudget capacity;
	MediaResourceBudget reserved;
	idx_t waiting_requests;
	idx_t granted_requests;
	idx_t active_leases;
};

//! A move-only reservation. Closing a scope never revokes a taken lease.
//! Owners must close the actual decoder/free the allocation before releasing it.
class MediaResourceLease {
public:
	DUCKDB_API MediaResourceLease();
	DUCKDB_API ~MediaResourceLease();
	MediaResourceLease(const MediaResourceLease &) = delete;
	MediaResourceLease &operator=(const MediaResourceLease &) = delete;
	DUCKDB_API MediaResourceLease(MediaResourceLease &&other) noexcept;
	DUCKDB_API MediaResourceLease &operator=(MediaResourceLease &&other) noexcept;
	DUCKDB_API void Release() noexcept;
	DUCKDB_API explicit operator bool() const;
	DUCKDB_API MediaResourceBudget Budget() const;

private:
	friend class MediaAdmissionRequest;
	MediaResourceLease(shared_ptr<MediaResourcePoolState> pool_p, MediaResourceBudget budget_p);
	shared_ptr<MediaResourcePoolState> pool;
	MediaResourceBudget budget;
};

//! Keep one request across blocked polls. Registering readiness and inspecting
//! the grant are atomic with respect to release/cancel; callbacks run unlocked.
class MediaAdmissionRequest {
public:
	DUCKDB_API ~MediaAdmissionRequest();
	MediaAdmissionRequest(const MediaAdmissionRequest &) = delete;
	MediaAdmissionRequest &operator=(const MediaAdmissionRequest &) = delete;
	DUCKDB_API MediaAdmissionStatus PollOrArm(std::function<void()> wakeup);
	DUCKDB_API MediaAdmissionStatus PollOrArm(const InterruptState &interrupt);
	DUCKDB_API MediaResourceLease TakeGrant();
	DUCKDB_API void Cancel() noexcept;

private:
	friend class MediaResourceScope;
	MediaAdmissionRequest(shared_ptr<MediaResourcePoolState> pool_p, shared_ptr<MediaResourceRequestState> request_p);
	shared_ptr<MediaResourcePoolState> pool;
	shared_ptr<MediaResourceRequestState> request;
};

class MediaResourceScope {
public:
	DUCKDB_API ~MediaResourceScope();
	MediaResourceScope(const MediaResourceScope &) = delete;
	MediaResourceScope &operator=(const MediaResourceScope &) = delete;
	DUCKDB_API unique_ptr<MediaAdmissionRequest> Request(MediaResourceBudget demand);
	DUCKDB_API void Close() noexcept;

private:
	friend class MediaResourcePool;
	MediaResourceScope(shared_ptr<MediaResourcePoolState> pool_p, idx_t id_p);
	shared_ptr<MediaResourcePoolState> pool;
	idx_t id;
	bool closed = false; // Protected by pool's lock.
};

//! FIFO admission of whole resource vectors. The pool never waits for capacity
//! on the calling thread and never owns a decoder or an output allocation.
class MediaResourcePool {
public:
	DUCKDB_API explicit MediaResourcePool(MediaResourceBudget capacity);
	DUCKDB_API shared_ptr<MediaResourceScope> CreateScope();
	DUCKDB_API MediaResourceSnapshot Snapshot() const;

private:
	shared_ptr<MediaResourcePoolState> state;
};

//! For the host DatabaseInstance constructor only. Extensions must obtain its
//! pool via DatabaseInstance::GetMediaDecoderPool(), never this local singleton.
//! This first adapter governs native scan decoder slots only. The capacity is
//! fixed on first use from VANE_NATIVE_MEDIA_MAX_CONCURRENT_DECODES. Returns
//! nullptr when that opt-in environment variable is unset.
DUCKDB_API shared_ptr<MediaResourcePool> GetNativeMediaDecoderPool();

//! Separate FIFO for scan pixel storage. A decoder waiter must not prevent an
//! active decoder from obtaining output bytes and reaching EOF. Host use only.
DUCKDB_API shared_ptr<MediaResourcePool> GetNativeMediaOutputPool();

} // namespace duckdb
