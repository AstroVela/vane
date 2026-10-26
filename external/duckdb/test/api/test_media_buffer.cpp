// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/types/image.hpp"
#include "duckdb/common/types/vector_buffer.hpp"
#include "duckdb/execution/media_buffer.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/pending_query_result.hpp"
#include "test_helpers.hpp"

#include <cstdlib>

using namespace duckdb;

namespace {
struct AllocationProbe {
	bool freed = false;
	bool destroyed = false;
	bool fail = false;
};
struct ProbeAllocatorData : public PrivateAllocatorData {
	explicit ProbeAllocatorData(shared_ptr<AllocationProbe> probe_p) : probe(std::move(probe_p)) {
	}
	~ProbeAllocatorData() override {
		probe->destroyed = true;
	}
	shared_ptr<AllocationProbe> probe;
};
shared_ptr<Allocator> ProbeAllocator(shared_ptr<AllocationProbe> probe) {
	return make_shared_ptr<Allocator>(
	    [](PrivateAllocatorData *data, idx_t size) -> data_ptr_t {
		    if (data->Cast<ProbeAllocatorData>().probe->fail) {
			    throw OutOfMemoryException("injected media allocation failure");
		    }
		    return static_cast<data_ptr_t>(std::malloc(size));
	    },
	    [](PrivateAllocatorData *data, data_ptr_t pointer, idx_t) {
		    std::free(pointer);
		    data->Cast<ProbeAllocatorData>().probe->freed = true;
	    },
	    nullptr, make_uniq<ProbeAllocatorData>(std::move(probe)));
}

struct OutputInfo : public TableFunctionInfo {
	explicit OutputInfo(shared_ptr<MediaResourcePool> pool_p) : pool(std::move(pool_p)) {
	}
	shared_ptr<MediaResourcePool> pool;
};
struct OutputBind : public TableFunctionData {
	explicit OutputBind(shared_ptr<MediaResourcePool> pool_p) : pool(std::move(pool_p)) {
	}
	shared_ptr<MediaResourcePool> pool;
	unique_ptr<FunctionData> Copy() const override {
		return make_uniq<OutputBind>(pool);
	}
	bool Equals(const FunctionData &other) const override {
		return pool == other.Cast<OutputBind>().pool;
	}
};
struct OutputGlobal : public GlobalTableFunctionState {
	explicit OutputGlobal(const shared_ptr<MediaResourcePool> &pool) : scope(pool->CreateScope()) {
	}
	~OutputGlobal() override {
		scope->Close();
	}
	shared_ptr<MediaResourceScope> scope;
};
struct OutputLocal : public LocalTableFunctionState {
	unique_ptr<MediaAdmissionRequest> request;
	idx_t row = 0;
};
unique_ptr<FunctionData> BindOutput(ClientContext &, TableFunctionBindInput &input, vector<LogicalType> &types,
                                    vector<string> &names) {
	types.push_back(LogicalType::LIST(LogicalType::UBIGINT));
	names.push_back("pixels");
	return make_uniq<OutputBind>(input.info->Cast<OutputInfo>().pool);
}
unique_ptr<GlobalTableFunctionState> InitOutput(ClientContext &, TableFunctionInitInput &input) {
	return make_uniq<OutputGlobal>(input.bind_data->Cast<OutputBind>().pool);
}
unique_ptr<LocalTableFunctionState> InitOutputLocal(ExecutionContext &, TableFunctionInitInput &,
                                                    GlobalTableFunctionState *) {
	return make_uniq<OutputLocal>();
}
SourceResultType PollOutput(ClientContext &, TableFunctionInput &input, DataChunk &output,
                            const InterruptState &interrupt) {
	auto &local = input.local_state->Cast<OutputLocal>();
	auto &global = input.global_state->Cast<OutputGlobal>();
	if (!local.request) {
		local.request = global.scope->Request({0, 64});
	}
	auto status = local.request->PollOrArm(interrupt);
	if (status == MediaAdmissionStatus::WAITING) {
		return SourceResultType::BLOCKED;
	}
	if (status == MediaAdmissionStatus::CANCELLED) {
		return SourceResultType::FINISHED;
	}
	AllocateMediaList(output.data[0], 1, 8, local.request->TakeGrant());
	local.request.reset();
	// Like a codec, discover EOF only after admission for the next read.
	if (local.row == 24) {
		output.Reset();
		return SourceResultType::FINISHED;
	}
	auto &pixels = ListVector::GetEntry(output.data[0]);
	for (idx_t i = 0; i < 8; i++) {
		FlatVector::GetData<uint64_t>(pixels)[i] = local.row;
	}
	ListVector::SetListSize(output.data[0], 8);
	FlatVector::GetData<list_entry_t>(output.data[0])[0] = list_entry_t(0, 8);
	output.SetCardinality(1);
	local.row++;
	return SourceResultType::HAVE_MORE_OUTPUT;
}
void RegisterOutput(DuckDB &db, shared_ptr<MediaResourcePool> pool) {
	ExtensionLoader loader(*db.instance, "media_output_test");
	TableFunction function("test_media_output", {}, nullptr, BindOutput, InitOutput, InitOutputLocal);
	function.poll_function = PollOutput;
	function.function_info = make_shared_ptr<OutputInfo>(std::move(pool));
	loader.RegisterFunction(std::move(function));
}
unique_ptr<QueryResult> Complete(PendingQueryResult &pending) {
	for (idx_t i = 0; i < 10000; i++) {
		auto result = pending.ExecuteTask();
		if (result == PendingExecutionResult::EXECUTION_ERROR) {
			return make_uniq<MaterializedQueryResult>(pending.GetErrorObject());
		}
		if (PendingQueryResult::IsResultReady(result)) {
			return pending.Execute();
		}
	}
	FAIL("Media output pipeline did not complete within its deterministic task budget");
	return nullptr;
}
} // namespace

TEST_CASE("Media pixels retain capacity through references and slices and free before wakeup", "[media-resource]") {
	MediaResourcePool pool({0, 64});
	auto scope = pool.CreateScope();
	auto probe = make_shared_ptr<AllocationProbe>();
	auto allocator = ProbeAllocator(probe);
	auto pixels = AllocateMediaVector(LogicalType::UBIGINT, 8, scope->Request({0, 64})->TakeGrant(), allocator);
	allocator.reset();
	pixels->SetValue(3, Value::UBIGINT(42));
	auto alias = make_uniq<Vector>(*pixels);
	SelectionVector selection(1);
	selection.set_index(0, 3);
	auto slice = make_uniq<Vector>(*pixels, selection, 1);
	auto waiting = scope->Request({0, 64});
	idx_t wakes = 0;
	REQUIRE(waiting->PollOrArm([&]() {
		CHECK(probe->freed);
		wakes++;
	}) == MediaAdmissionStatus::WAITING);
	pixels.reset();
	alias.reset();
	CHECK_FALSE(probe->freed);
	CHECK_FALSE(probe->destroyed);
	CHECK(pool.Snapshot().reserved.bytes == 64);
	CHECK(slice->GetValue(0) == Value::UBIGINT(42));
	slice.reset();
	CHECK(wakes == 1);
	CHECK(probe->destroyed);
	waiting->Cancel();
	CHECK(pool.Snapshot().reserved.bytes == 0);
}

TEST_CASE("Media list ownership survives chunk reset and scope cancellation", "[media-resource]") {
	MediaResourcePool pool({0, 64});
	auto scope = pool.CreateScope();
	DataChunk source;
	source.Initialize(Allocator::DefaultAllocator(), {LogicalType::LIST(LogicalType::UBIGINT)});
	AllocateMediaList(source.data[0], 1, 8, scope->Request({0, 64})->TakeGrant());
	ListVector::GetEntry(source.data[0]).SetValue(0, Value::UBIGINT(42));
	ListVector::SetListSize(source.data[0], 8);
	FlatVector::GetData<list_entry_t>(source.data[0])[0] = list_entry_t(0, 1);
	source.SetCardinality(1);
	auto view = make_uniq<DataChunk>();
	view->InitializeEmpty(source.GetTypes());
	view->Reference(source);
	scope->Close();
	source.Reset();
	CHECK(pool.Snapshot().reserved.bytes == 64);
	CHECK(ListValue::GetChildren(view->GetValue(0, 0))[0] == Value::UBIGINT(42));
	view.reset();
	CHECK(pool.Snapshot().reserved.bytes == 0);
	CHECK(pool.Snapshot().active_leases == 0);
}

TEST_CASE("Failed media allocation and invalid reservations return capacity", "[media-resource]") {
	MediaResourcePool pool({1, 64});
	auto scope = pool.CreateScope();
	auto probe = make_shared_ptr<AllocationProbe>();
	probe->fail = true;
	REQUIRE_THROWS_AS(
	    AllocateMediaVector(LogicalType::UBIGINT, 8, scope->Request({0, 64})->TakeGrant(), ProbeAllocator(probe)),
	    OutOfMemoryException);
	CHECK(pool.Snapshot().reserved.bytes == 0);
	REQUIRE_THROWS_AS(AllocateMediaVector(LogicalType::UBIGINT, 9, scope->Request({0, 64})->TakeGrant()),
	                  InvalidInputException);
	REQUIRE_THROWS_AS(AllocateMediaVector(LogicalType::VARCHAR, 8, scope->Request({0, 64})->TakeGrant()),
	                  InvalidInputException);
	REQUIRE_THROWS_AS(AllocateMediaVector(LogicalType::UBIGINT, 8, scope->Request({1, 64})->TakeGrant()),
	                  InvalidInputException);
	CHECK(pool.Snapshot().reserved.bytes == 0);
	CHECK(pool.Snapshot().reserved.decoders == 0);
	CHECK(pool.Snapshot().active_leases == 0);
}

TEST_CASE("IMAGE pixel writers preserve admitted storage through nested slicing", "[media-resource]") {
	MediaResourcePool pool({0, 24});
	auto scope = pool.CreateScope();
	auto image = make_uniq<Vector>(ImageLogicalType::Create("RGB"), idx_t(2));
	auto &data = *StructVector::GetEntries(*image)[ImageLogicalType::DATA];
	AllocateMediaList(data, 2, 24, scope->Request({0, 24})->TakeGrant());
	for (idx_t row = 0; row < 2; row++) {
		auto pixels = ImageVector::Allocate(*image, row, 2, 2, "RGB");
		for (idx_t i = 0; i < 12; i++) {
			pixels[i] = uint8_t(row + 7);
		}
	}
	auto slice = make_uniq<Vector>(*image, idx_t(1), idx_t(2));
	image.reset();
	scope->Close();
	CHECK(pool.Snapshot().reserved.bytes == 24);
	CHECK(ImageVector::Pixels(slice->GetValue(0))[0] == Value::UTINYINT(8));
	slice.reset();
	CHECK(pool.Snapshot().reserved.bytes == 0);
}

TEST_CASE("One media output allocation suffices for projection filtering sorting and LIMIT", "[media-resource]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=1"));
	auto pool = make_shared_ptr<MediaResourcePool>(MediaResourceBudget(0, 64));
	RegisterOutput(db, pool);
	const vector<string> queries {"SELECT pixels FROM test_media_output()",
	                              "SELECT pixels FROM test_media_output() WHERE pixels[1] >= 0",
	                              "SELECT pixels FROM test_media_output() ORDER BY pixels[1] DESC",
	                              "SELECT pixels FROM test_media_output() LIMIT 1",
	                              "SELECT list_transform(pixels, x -> x + 1) FROM test_media_output()"};
	for (auto &query : queries) {
		INFO(query);
		auto pending = con.PendingQuery(query);
		REQUIRE_FALSE(pending->HasError());
		auto result = Complete(*pending);
		REQUIRE_NO_FAIL(*result);
		idx_t count = 0;
		while (auto chunk = result->Fetch()) {
			for (idx_t row = 0; row < chunk->size(); row++) {
				auto pixels = ListValue::GetChildren(chunk->GetValue(0, row));
				REQUIRE(pixels.size() == 8);
				auto expected = query.find("ORDER BY") != string::npos ? 23 - count : count;
				if (query.find("list_transform") != string::npos) {
					expected++;
				}
				CHECK(pixels[0] == Value::UBIGINT(expected));
				count++;
			}
		}
		CHECK(count == (query.find("LIMIT") == string::npos ? 24 : 1));
		CHECK(pool->Snapshot().reserved.bytes == 0);
	}
}

TEST_CASE("Retained media pixels backpressure another query and cancellation needs no release", "[media-resource]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=1"));
	auto pool = make_shared_ptr<MediaResourcePool>(MediaResourceBudget(0, 64));
	RegisterOutput(db, pool);
	auto scope = pool->CreateScope();
	auto pixels = AllocateMediaVector(LogicalType::UBIGINT, 8, scope->Request({0, 64})->TakeGrant());
	for (bool cancel : {false, true}) {
		auto pending = con.PendingQuery("SELECT sum(pixels[1]) FROM test_media_output()");
		REQUIRE_FALSE(pending->HasError());
		for (idx_t i = 0; i < 1000 && pool->Snapshot().waiting_requests == 0; i++) {
			pending->ExecuteTask();
		}
		REQUIRE(pool->Snapshot().waiting_requests == 1);
		if (cancel) {
			con.Interrupt();
		} else {
			pixels.reset();
		}
		auto result = Complete(*pending);
		CHECK(result->HasError() == cancel);
		CHECK(pool->Snapshot().waiting_requests == 0);
		if (!cancel) {
			REQUIRE(result->Fetch()->GetValue(0, 0) == Value::HUGEINT(276));
			pixels = AllocateMediaVector(LogicalType::UBIGINT, 8, scope->Request({0, 64})->TakeGrant());
		}
	}
	pixels.reset();
	CHECK(pool->Snapshot().reserved.bytes == 0);
}

TEST_CASE("Output pools are shared by host databases independently of decoder queues", "[media-resource]") {
	DuckDB first(nullptr);
	DuckDB second(nullptr);
	auto pool = first.instance->GetMediaOutputPool();
	REQUIRE(pool == second.instance->GetMediaOutputPool());
	REQUIRE(pool == GetNativeMediaOutputPool());
	if (pool) {
		auto scope = pool->CreateScope();
		auto held = scope->Request(pool->Snapshot().capacity)->TakeGrant();
		auto waiting = second.instance->GetMediaOutputPool()->CreateScope();
		auto request = waiting->Request({0, 1});
		CHECK(request->PollOrArm([]() {}) == MediaAdmissionStatus::WAITING);
		held.Release();
		CHECK(request->PollOrArm([]() {}) == MediaAdmissionStatus::GRANTED);
		request->Cancel();
		CHECK(pool->Snapshot().reserved.bytes == 0);
	}
}

TEST_CASE("Streaming results consume media pixels within a one-allocation budget", "[media-resource]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=1"));
	auto pool = make_shared_ptr<MediaResourcePool>(MediaResourceBudget(0, 64));
	RegisterOutput(db, pool);
	auto pending = con.PendingQuery("SELECT pixels FROM test_media_output()", QueryResultOutputType::ALLOW_STREAMING);
	REQUIRE_FALSE(pending->HasError());
	auto result = Complete(*pending);
	REQUIRE_NO_FAIL(*result);
	idx_t count = 0;
	while (auto chunk = result->Fetch()) {
		for (idx_t row = 0; row < chunk->size(); row++) {
			auto pixels = ListValue::GetChildren(chunk->GetValue(0, row));
			REQUIRE(pixels.size() == 8);
			CHECK(pixels[0] == Value::UBIGINT(count));
			count++;
		}
	}
	CHECK(count == 24);
	result.reset();
	pending.reset();
	CHECK(pool->Snapshot().reserved.bytes == 0);
}

TEST_CASE("Downstream errors release pixel allocations and permit query reuse", "[media-resource]") {
	DuckDB db(nullptr);
	Connection con(db);
	REQUIRE_NO_FAIL(con.Query("SET threads=1"));
	auto pool = make_shared_ptr<MediaResourcePool>(MediaResourceBudget(0, 64));
	RegisterOutput(db, pool);
	auto failed = con.PendingQuery("SELECT CAST('bad' || pixels[1]::VARCHAR AS BIGINT) FROM test_media_output()");
	REQUIRE_FALSE(failed->HasError());
	REQUIRE(Complete(*failed)->HasError());
	CHECK(pool->Snapshot().reserved.bytes == 0);
	CHECK(pool->Snapshot().waiting_requests == 0);
	auto retry = con.PendingQuery("SELECT sum(pixels[1]) FROM test_media_output()");
	auto result = Complete(*retry);
	REQUIRE_NO_FAIL(*result);
	CHECK(result->Fetch()->GetValue(0, 0) == Value::HUGEINT(276));
	CHECK(pool->Snapshot().reserved.bytes == 0);
}
