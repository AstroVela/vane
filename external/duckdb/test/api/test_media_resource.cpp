// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/execution/media_resource.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/pending_query_result.hpp"
#include "test_helpers.hpp"

#include <atomic>
#include <limits>
#include <stdexcept>
#include <thread>
#include <type_traits>

using namespace duckdb;

static_assert(!std::is_copy_constructible<MediaResourceLease>::value, "Media leases have a single owner");
static_assert(std::is_nothrow_move_constructible<MediaResourceLease>::value, "Lease transfer must not throw");

namespace {

MediaAdmissionStatus Poll(MediaAdmissionRequest &request) {
	return request.PollOrArm([]() {});
}

void CheckEmpty(const MediaResourcePool &pool) {
	auto snapshot = pool.Snapshot();
	CHECK(snapshot.reserved.decoders == 0);
	CHECK(snapshot.reserved.bytes == 0);
	CHECK(snapshot.waiting_requests == 0);
	CHECK(snapshot.granted_requests == 0);
	CHECK(snapshot.active_leases == 0);
}

// Exercise the real scheduler and query cleanup without an optional codec
// provider. The source holds its reservation across output chunks, like a
// video cursor, and can suspend before opening that cursor.
struct AdmissionSourceInfo : public TableFunctionInfo {
	explicit AdmissionSourceInfo(shared_ptr<MediaResourcePool> pool_p) : pool(std::move(pool_p)) {
	}
	shared_ptr<MediaResourcePool> pool;
};

struct AdmissionSourceBind : public TableFunctionData {
	explicit AdmissionSourceBind(shared_ptr<MediaResourcePool> pool_p) : pool(std::move(pool_p)) {
	}
	shared_ptr<MediaResourcePool> pool;
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<AdmissionSourceBind>(pool);
	}
	bool Equals(const FunctionData &other) const override {
		return pool == other.Cast<AdmissionSourceBind>().pool;
	}
};

struct AdmissionSourceGlobal : public GlobalTableFunctionState {
	explicit AdmissionSourceGlobal(const shared_ptr<MediaResourcePool> &pool) : scope(pool->CreateScope()) {
	}
	~AdmissionSourceGlobal() override {
		scope->Close();
	}
	shared_ptr<MediaResourceScope> scope;
};

struct AdmissionSourceLocal : public LocalTableFunctionState {
	unique_ptr<MediaAdmissionRequest> request;
	MediaResourceLease lease;
	idx_t rows = 0;
};

unique_ptr<FunctionData> BindAdmissionSource(ClientContext &, TableFunctionBindInput &input, vector<LogicalType> &types,
                                             vector<string> &names) {
	types.push_back(LogicalType::BIGINT);
	names.push_back("value");
	return make_uniq<AdmissionSourceBind>(input.info->Cast<AdmissionSourceInfo>().pool);
}

unique_ptr<GlobalTableFunctionState> InitAdmissionSource(ClientContext &, TableFunctionInitInput &input) {
	return make_uniq<AdmissionSourceGlobal>(input.bind_data->Cast<AdmissionSourceBind>().pool);
}

unique_ptr<LocalTableFunctionState> InitAdmissionSourceLocal(ExecutionContext &, TableFunctionInitInput &,
                                                             GlobalTableFunctionState *) {
	return make_uniq<AdmissionSourceLocal>();
}

SourceResultType PollAdmissionSource(ClientContext &, TableFunctionInput &input, DataChunk &output,
                                     const InterruptState &interrupt) {
	auto &global = input.global_state->Cast<AdmissionSourceGlobal>();
	auto &local = input.local_state->Cast<AdmissionSourceLocal>();
	if (local.rows == 2) {
		local.lease.Release();
		return SourceResultType::FINISHED;
	}
	if (!local.lease) {
		if (!local.request) {
			local.request = global.scope->Request({1, 64});
		}
		auto status = local.request->PollOrArm(interrupt);
		if (status == MediaAdmissionStatus::WAITING) {
			return SourceResultType::BLOCKED;
		}
		if (status == MediaAdmissionStatus::CANCELLED) {
			return SourceResultType::FINISHED;
		}
		local.lease = local.request->TakeGrant();
		local.request.reset();
	}
	output.SetValue(0, 0, Value::BIGINT(42 + local.rows++));
	output.SetCardinality(1);
	return SourceResultType::HAVE_MORE_OUTPUT;
}

void RegisterAdmissionSource(DuckDB &db, shared_ptr<MediaResourcePool> pool) {
	ExtensionLoader loader(*db.instance, "media_admission_test");
	TableFunction function("test_media_admission", {}, nullptr, BindAdmissionSource, InitAdmissionSource,
	                       InitAdmissionSourceLocal);
	function.poll_function = PollAdmissionSource;
	function.function_info = make_shared_ptr<AdmissionSourceInfo>(std::move(pool));
	loader.RegisterFunction(std::move(function));
}

void RunUntilWaiting(PendingQueryResult &pending, const MediaResourcePool &pool) {
	for (idx_t i = 0; i < 100 && pool.Snapshot().waiting_requests == 0; i++) {
		auto result = pending.ExecuteTask();
		REQUIRE(result != PendingExecutionResult::EXECUTION_ERROR);
		REQUIRE(result != PendingExecutionResult::RESULT_READY);
	}
	REQUIRE(pool.Snapshot().waiting_requests == 1);
}

} // namespace

TEST_CASE("Media admission reserves the complete resource vector", "[media-resource]") {
	MediaResourcePool pool({2, 512});
	auto scope = pool.CreateScope();
	auto first = scope->Request({1, 400});
	REQUIRE(Poll(*first) == MediaAdmissionStatus::GRANTED);
	auto first_lease = first->TakeGrant();
	first.reset();
	auto second = scope->Request({1, 200});
	idx_t wakeups = 0;
	REQUIRE(second->PollOrArm([&]() {
		wakeups++;
		CHECK(pool.Snapshot().reserved.bytes == 200); // The callback runs outside the lock.
	}) == MediaAdmissionStatus::WAITING);
	auto snapshot = pool.Snapshot();
	CHECK(snapshot.reserved.decoders == 1); // No partial slot grant while waiting for bytes.
	CHECK(snapshot.reserved.bytes == 400);
	CHECK(snapshot.waiting_requests == 1);
	first_lease.Release();
	CHECK(wakeups == 1);
	REQUIRE(Poll(*second) == MediaAdmissionStatus::GRANTED);
	auto second_lease = second->TakeGrant();
	REQUIRE_THROWS_AS(second->TakeGrant(), InternalException);
	second->Cancel();
	CHECK(pool.Snapshot().reserved.bytes == 200);
	second_lease.Release();
	second_lease.Release();
	CheckEmpty(pool);
}

TEST_CASE("Media admission distinguishes impossible demand from temporary pressure", "[media-resource]") {
	MediaResourcePool pool({1, 100});
	auto scope = pool.CreateScope();
	auto too_many_decoders = scope->Request({2, 0});
	auto too_many_bytes = scope->Request({0, 101});
	CHECK(Poll(*too_many_decoders) == MediaAdmissionStatus::IMPOSSIBLE);
	CHECK(Poll(*too_many_bytes) == MediaAdmissionStatus::IMPOSSIBLE);
	REQUIRE_THROWS_AS(too_many_bytes->TakeGrant(), InternalException);
	CheckEmpty(pool);

	auto first = scope->Request({1, 100});
	auto waiting = scope->Request({1, 100});
	CHECK(Poll(*first) == MediaAdmissionStatus::GRANTED);
	CHECK(Poll(*waiting) == MediaAdmissionStatus::WAITING);
	first->Cancel();
	CHECK(Poll(*waiting) == MediaAdmissionStatus::GRANTED);
	waiting->Cancel();
	CheckEmpty(pool);

	MediaResourcePool large_pool({0, std::numeric_limits<idx_t>::max()});
	auto large_scope = large_pool.CreateScope();
	auto large = large_scope->Request({0, std::numeric_limits<idx_t>::max()});
	auto overflow = large_scope->Request({0, 1});
	CHECK(Poll(*large) == MediaAdmissionStatus::GRANTED);
	CHECK(Poll(*overflow) == MediaAdmissionStatus::WAITING);
	large_scope->Close();
	CheckEmpty(large_pool);
}

TEST_CASE("Media admission preserves FIFO and cancelling its head admits the next request", "[media-resource]") {
	MediaResourcePool pool({2, 100});
	auto a = pool.CreateScope();
	auto b = pool.CreateScope();
	auto active = a->Request({1, 0});
	auto lease = active->TakeGrant();
	auto large = b->Request({2, 0});
	auto small = a->Request({1, 0});
	CHECK(Poll(*large) == MediaAdmissionStatus::WAITING);
	CHECK(Poll(*small) == MediaAdmissionStatus::WAITING);
	idx_t wakeups = 0;
	small->PollOrArm([&]() { wakeups++; });
	large->Cancel();
	CHECK(wakeups == 1);
	CHECK(Poll(*small) == MediaAdmissionStatus::GRANTED);
	CHECK(pool.Snapshot().reserved.decoders == 2);
	a->Close();
	CHECK(pool.Snapshot().reserved.decoders == 1); // Closing cannot revoke the taken lease.
	lease.Release();
	CheckEmpty(pool);
}

TEST_CASE("Media scope closure returns unclaimed grants but preserves live ownership", "[media-resource]") {
	MediaResourcePool pool({3, 300});
	auto scope = pool.CreateScope();
	auto other_scope = pool.CreateScope();
	auto taken = scope->Request({1, 100});
	auto owner = make_shared_ptr<MediaResourceLease>(taken->TakeGrant());
	auto other_reference = owner;
	auto granted = scope->Request({1, 100});
	auto pending = scope->Request({2, 100});
	auto successor = other_scope->Request({1, 200});
	idx_t cancelled_wakeups = 0;
	pending->PollOrArm([&]() { cancelled_wakeups++; });
	scope->Close();
	scope->Close();
	CHECK(cancelled_wakeups == 1);
	CHECK(Poll(*granted) == MediaAdmissionStatus::CANCELLED);
	CHECK(Poll(*pending) == MediaAdmissionStatus::CANCELLED);
	CHECK(Poll(*successor) == MediaAdmissionStatus::GRANTED);
	auto after_close = scope->Request({0, 1});
	CHECK(Poll(*after_close) == MediaAdmissionStatus::CANCELLED);
	scope.reset();
	taken.reset();
	successor->Cancel();
	owner.reset();
	CHECK(pool.Snapshot().reserved.bytes == 100);
	other_reference.reset();
	CheckEmpty(pool);
}

TEST_CASE("Media request destruction and grant handoff return capacity exactly once", "[media-resource]") {
	MediaResourcePool pool({2, 200});
	auto scope = pool.CreateScope();
	MediaResourceLease lease;
	{
		auto request = scope->Request({1, 100});
		lease = request->TakeGrant();
	}
	CHECK(pool.Snapshot().active_leases == 1);
	auto request = scope->Request({1, 100});
	MediaResourceLease other = request->TakeGrant();
	lease = std::move(other);
	CHECK_FALSE(bool(other));
	CHECK(lease.Budget().bytes == 100);
	CHECK(pool.Snapshot().active_leases == 1);
	lease.Release();
	for (idx_t i = 0; i < 1000; i++) {
		auto unclaimed = scope->Request({2, 200});
		auto pending = scope->Request({1, 1});
		CHECK(Poll(*pending) == MediaAdmissionStatus::WAITING);
	}
	CheckEmpty(pool);
}

TEST_CASE("Media readiness handles release before registration and replaces stale callbacks", "[media-resource]") {
	MediaResourcePool pool({1, 0});
	auto scope = pool.CreateScope();
	auto owner = scope->Request({1, 0});
	auto waiting = scope->Request({1, 0});
	owner.reset();
	idx_t late = 0;
	CHECK(waiting->PollOrArm([&]() { late++; }) == MediaAdmissionStatus::GRANTED);
	CHECK(late == 0); // The caller observes readiness without relying on a callback.
	auto lease = waiting->TakeGrant();
	auto next = scope->Request({1, 0});
	idx_t stale = 0;
	idx_t current = 0;
	CHECK(next->PollOrArm([&]() { stale++; }) == MediaAdmissionStatus::WAITING);
	CHECK(next->PollOrArm([&]() { current++; }) == MediaAdmissionStatus::WAITING);
	lease.Release();
	CHECK(stale == 0);
	CHECK(current == 1);
	next->Cancel();
	CheckEmpty(pool);
}

TEST_CASE("Media callback failure does not suppress another ready task", "[media-resource]") {
	MediaResourcePool pool({2, 0});
	auto scope = pool.CreateScope();
	auto owner = scope->Request({2, 0});
	auto first = scope->Request({1, 0});
	auto second = scope->Request({1, 0});
	idx_t wakeups = 0;
	first->PollOrArm([]() { throw std::runtime_error("retired consumer"); });
	second->PollOrArm([&]() { wakeups++; });
	owner.reset();
	CHECK(wakeups == 1);
	CHECK(Poll(*first) == MediaAdmissionStatus::GRANTED);
	CHECK(Poll(*second) == MediaAdmissionStatus::GRANTED);
	scope->Close();
	CheckEmpty(pool);
}

TEST_CASE("Media cancellation races with resource release without leaking a grant", "[media-resource]") {
	MediaResourcePool pool({1, 64});
	for (idx_t i = 0; i < 100; i++) {
		auto first_scope = pool.CreateScope();
		auto second_scope = pool.CreateScope();
		auto first = first_scope->Request({1, 64});
		auto lease = first->TakeGrant();
		auto second = second_scope->Request({1, 64});
		std::atomic<idx_t> wakeups {0};
		REQUIRE(second->PollOrArm([&]() { wakeups++; }) == MediaAdmissionStatus::WAITING);
		std::thread release([&]() { lease.Release(); });
		std::thread close([&]() { second_scope->Close(); });
		release.join();
		close.join();
		CHECK(Poll(*second) == MediaAdmissionStatus::CANCELLED);
		CHECK(wakeups == 1);
		CheckEmpty(pool);
	}
}

TEST_CASE("Media admission suspends and resumes an engine source", "[media-resource]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=1"));
	auto pool = make_shared_ptr<MediaResourcePool>(MediaResourceBudget(1, 64));
	RegisterAdmissionSource(db, pool);
	auto owner_scope = pool->CreateScope();
	auto owner = owner_scope->Request({1, 64})->TakeGrant();
	auto pending = con.PendingQuery("SELECT sum(value) FROM test_media_admission()");
	REQUIRE_FALSE(pending->HasError());
	RunUntilWaiting(*pending, *pool);
	// The only engine execution thread reached BLOCKED without waiting on a
	// semaphore, so an unrelated query can still run while capacity is held.
	Connection other(db);
	REQUIRE_NO_FAIL(other.Query("SELECT 1"));
	owner.Release();
	auto result = pending->Execute();
	REQUIRE_NO_FAIL(*result);
	REQUIRE(result->Fetch()->GetValue(0, 0) == Value::HUGEINT(85));
	CheckEmpty(*pool);
}

TEST_CASE("Interrupted media source cleans up without another capacity release", "[media-resource]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=1"));
	auto pool = make_shared_ptr<MediaResourcePool>(MediaResourceBudget(1, 64));
	RegisterAdmissionSource(db, pool);
	auto owner_scope = pool->CreateScope();
	auto owner = owner_scope->Request({1, 64})->TakeGrant();
	auto pending = con.PendingQuery("SELECT * FROM test_media_admission()");
	REQUIRE_FALSE(pending->HasError());
	RunUntilWaiting(*pending, *pool);
	con.Interrupt();
	auto result = pending->Execute();
	REQUIRE(result->HasError());
	CHECK(pool->Snapshot().waiting_requests == 0);
	CHECK(pool->Snapshot().granted_requests == 0);
	CHECK(pool->Snapshot().active_leases == 1);
	owner.Release();
	CheckEmpty(*pool);
	REQUIRE_NO_FAIL(con.Query("SELECT * FROM test_media_admission()"));
	CheckEmpty(*pool);
}

TEST_CASE("Media source returns leases on downstream LIMIT and expression errors", "[media-resource]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=1"));
	auto pool = make_shared_ptr<MediaResourcePool>(MediaResourceBudget(1, 64));
	RegisterAdmissionSource(db, pool);
	auto limited = con.Query("SELECT * FROM test_media_admission() LIMIT 1");
	REQUIRE_NO_FAIL(*limited);
	REQUIRE(limited->RowCount() == 1);
	CheckEmpty(*pool);
	auto failed = con.Query("SELECT CAST('invalid' || value::VARCHAR AS BIGINT) FROM test_media_admission()");
	REQUIRE(failed->HasError());
	CheckEmpty(*pool);
	REQUIRE_NO_FAIL(con.Query("SELECT * FROM test_media_admission()"));
	CheckEmpty(*pool);
}

TEST_CASE("Host database instances share the native media admission authority", "[media-resource]") {
	DuckDB first(nullptr);
	DuckDB second(nullptr);
	auto pool = first.instance->GetMediaDecoderPool();
	REQUIRE(pool == second.instance->GetMediaDecoderPool());
	REQUIRE(pool == GetNativeMediaDecoderPool());
	if (pool) {
		auto first_scope = pool->CreateScope();
		auto second_scope = second.instance->GetMediaDecoderPool()->CreateScope();
		auto owner = first_scope->Request(pool->Snapshot().capacity)->TakeGrant();
		auto waiting = second_scope->Request({1, 0});
		CHECK(Poll(*waiting) == MediaAdmissionStatus::WAITING);
		owner.Release();
		CHECK(Poll(*waiting) == MediaAdmissionStatus::GRANTED);
		waiting->Cancel();
		CheckEmpty(*pool);
	}
}
