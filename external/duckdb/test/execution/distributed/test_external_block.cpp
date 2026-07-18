// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "catch.hpp"

#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/external_block.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"

using namespace duckdb;

namespace {

static constexpr idx_t NO_SLICE = idx_t(-1);

ExternalBlockDescriptor MakeDescriptor(idx_t rows, idx_t bytes, idx_t start = 0, idx_t end = NO_SLICE) {
	ExternalBlockDescriptor desc;
	desc.object_ref = make_shared_ptr<int>(static_cast<int>(rows));
	desc.metadata.num_rows = rows;
	desc.metadata.size_bytes = bytes;
	if (end != NO_SLICE) {
		desc.has_slice = true;
		desc.slice.start_offset = start;
		desc.slice.end_offset = end;
	}
	return desc;
}

unique_ptr<LazyDataChunk> MakeLazyChunk(vector<ExternalBlockDescriptor> blocks) {
	auto chunk = make_uniq<LazyDataChunk>();
	chunk->logical_types = {LogicalType::BIGINT};
	chunk->names = {"x"};
	chunk->blocks = std::move(blocks);
	chunk->RecomputeCardinality();
	return chunk;
}

unique_ptr<LazyDataChunk> MakeWideLazyChunk(vector<ExternalBlockDescriptor> blocks) {
	auto chunk = make_uniq<LazyDataChunk>();
	chunk->logical_types = {LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::BOOLEAN};
	chunk->names = {"id", "path", "ok"};
	chunk->blocks = std::move(blocks);
	chunk->RecomputeCardinality();
	return chunk;
}

unique_ptr<LazyDataChunk> MakeWrappedStructLazyChunk(vector<ExternalBlockDescriptor> blocks) {
	auto chunk = make_uniq<LazyDataChunk>();
	chunk->logical_types = {LogicalType::STRUCT(
	    {{"id", LogicalType::BIGINT}, {"path", LogicalType::VARCHAR}, {"ok", LogicalType::BOOLEAN}})};
	chunk->names = {"__udf_result"};
	chunk->wrap_columns_as_struct = true;
	chunk->blocks = std::move(blocks);
	chunk->RecomputeCardinality();
	return chunk;
}

struct ExternalBlockBackendGuard {
	ExternalBlockBackendGuard() : previous(GetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE)) {
	}
	~ExternalBlockBackendGuard() {
		SetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE, std::move(previous));
	}

	shared_ptr<ExternalBlockBackendInterface> previous;
};

class TestExternalBlockBackend : public ExternalBlockBackendInterface {
public:
	bool CanMaterialize(const ExternalBlockDescriptor &desc) override {
		return desc.object_ref != nullptr;
	}

	unique_ptr<DataChunk> Materialize(ClientContext &context, const LazyDataChunk &chunk) override {
		materialize_calls++;
		auto result = make_uniq<DataChunk>();
		result->Initialize(context, chunk.logical_types);
		result->SetCardinality(chunk.cardinality);

		idx_t row_idx = 0;
		for (auto &block : chunk.blocks) {
			for (idx_t row = block.StartOffset(); row < block.EndOffset(); row++) {
				result->SetValue(0, row_idx++, Value::BIGINT(static_cast<int64_t>(row)));
			}
		}
		return result;
	}

	idx_t materialize_calls = 0;
};

} // namespace

TEST_CASE("ExternalBlockDescriptor reports slice rows and estimated bytes", "[execution][external_block]") {
	auto full = MakeDescriptor(128, 1280);
	REQUIRE(full.StartOffset() == 0);
	REQUIRE(full.EndOffset() == 128);
	REQUIRE(full.RowCount() == 128);
	REQUIRE(full.EstimatedBytes() == 1280);

	auto slice = MakeDescriptor(128, 1280, 16, 48);
	REQUIRE(slice.StartOffset() == 16);
	REQUIRE(slice.EndOffset() == 48);
	REQUIRE(slice.RowCount() == 32);
	REQUIRE(slice.EstimatedBytes() == 320);
}

TEST_CASE("LazyDataChunkBundler splits one block by descriptor only", "[execution][external_block]") {
	LazyDataChunkBundler bundler(100);
	bundler.Add(MakeLazyChunk({MakeDescriptor(128, 1280)}));

	REQUIRE(bundler.PendingRows() == 28);
	REQUIRE(bundler.ReadyRows() == 100);
	REQUIRE(bundler.HasReady());

	auto first = bundler.PopReady();
	REQUIRE(first);
	REQUIRE(first->cardinality == 100);
	REQUIRE(first->blocks.size() == 1);
	REQUIRE(first->blocks[0].has_slice);
	REQUIRE(first->blocks[0].slice.start_offset == 0);
	REQUIRE(first->blocks[0].slice.end_offset == 100);
	REQUIRE(first->EstimatedBytes() == 1000);

	REQUIRE(!bundler.HasReady());
	bundler.Finish();
	REQUIRE(bundler.PendingRows() == 0);
	REQUIRE(bundler.ReadyRows() == 28);

	auto tail = bundler.PopReady();
	REQUIRE(tail);
	REQUIRE(tail->cardinality == 28);
	REQUIRE(tail->blocks.size() == 1);
	REQUIRE(tail->blocks[0].has_slice);
	REQUIRE(tail->blocks[0].slice.start_offset == 100);
	REQUIRE(tail->blocks[0].slice.end_offset == 128);
	REQUIRE(tail->EstimatedBytes() == 280);
}

TEST_CASE("External block ownership survives descriptor fan-out", "[execution][external_block]") {
	idx_t release_count = 0;
	auto lifetime = shared_ptr<void>(new int(1), [&release_count](void *ptr) {
		delete static_cast<int *>(ptr);
		release_count++;
	});
	auto block = MakeDescriptor(128, 1280);
	block.ownership_tokens.push_back(lifetime);
	auto chunk = MakeLazyChunk({std::move(block)});
	lifetime.reset();

	LazyDataChunkBundler bundler(100);
	bundler.Add(std::move(chunk));
	auto first = bundler.PopReady();
	bundler.Finish();
	auto tail = bundler.PopReady();

	REQUIRE(first);
	REQUIRE(tail);
	REQUIRE(first->blocks[0].ownership_tokens.size() == 1);
	REQUIRE(tail->blocks[0].ownership_tokens.size() == 1);
	REQUIRE(release_count == 0);

	first.reset();
	REQUIRE(release_count == 0);
	tail.reset();
	REQUIRE(release_count == 1);
}

TEST_CASE("LazyDataChunkBundler builds a batch across block descriptors", "[execution][external_block]") {
	LazyDataChunkBundler bundler(100);
	bundler.Add(MakeLazyChunk({MakeDescriptor(60, 600), MakeDescriptor(60, 600)}));

	auto ready = bundler.PopReady();
	REQUIRE(ready);
	REQUIRE(ready->cardinality == 100);
	REQUIRE(ready->blocks.size() == 2);
	REQUIRE(ready->blocks[0].slice.start_offset == 0);
	REQUIRE(ready->blocks[0].slice.end_offset == 60);
	REQUIRE(ready->blocks[1].slice.start_offset == 0);
	REQUIRE(ready->blocks[1].slice.end_offset == 40);
	REQUIRE(ready->EstimatedBytes() == 1000);
	REQUIRE(bundler.PendingRows() == 20);

	bundler.Finish();
	auto tail = bundler.PopReady();
	REQUIRE(tail);
	REQUIRE(tail->cardinality == 20);
	REQUIRE(tail->blocks.size() == 1);
	REQUIRE(tail->blocks[0].slice.start_offset == 40);
	REQUIRE(tail->blocks[0].slice.end_offset == 60);
}

TEST_CASE("LazyDataChunkBundler recomputes cardinality before queueing", "[execution][external_block]") {
	auto chunk = MakeLazyChunk({MakeDescriptor(128, 1280)});
	chunk->cardinality = 0;

	LazyDataChunkBundler bundler(100);
	bundler.Add(std::move(chunk));

	REQUIRE(bundler.HasReady());
	REQUIRE(bundler.PendingRows() == 28);
}

TEST_CASE("LazyDataChunkBundler rejects incompatible schemas", "[execution][external_block]") {
	LazyDataChunkBundler bundler(100);
	bundler.Add(MakeLazyChunk({MakeDescriptor(10, 100)}));

	auto other = MakeLazyChunk({MakeDescriptor(10, 100)});
	other->logical_types = {LogicalType::VARCHAR};
	REQUIRE_THROWS_AS(bundler.Add(std::move(other)), InvalidInputException);
}

TEST_CASE("ProjectLazyDataChunk rewrites schema and descriptor column ids only", "[execution][external_block]") {
	auto block = MakeDescriptor(128, 1280);
	auto object_ref = block.object_ref;
	auto chunk = MakeWideLazyChunk({block});

	auto projected = ProjectLazyDataChunk(*chunk, {2, 0}, {"is_ok", "image_id"});
	REQUIRE(projected);
	REQUIRE(projected->cardinality == 128);
	REQUIRE(projected->logical_types == vector<LogicalType> {LogicalType::BOOLEAN, LogicalType::BIGINT});
	REQUIRE(projected->names == vector<string> {"is_ok", "image_id"});
	REQUIRE(projected->blocks.size() == 1);
	REQUIRE(projected->blocks[0].object_ref == object_ref);
	REQUIRE(projected->blocks[0].column_ids == vector<idx_t> {2, 0});
	REQUIRE(projected->blocks[0].RowCount() == 128);

	auto identity = ProjectLazyDataChunk(*chunk, {0, 1, 2});
	REQUIRE(identity->blocks.size() == 1);
	REQUIRE(identity->blocks[0].object_ref == object_ref);
	REQUIRE(identity->blocks[0].column_ids.empty());
	REQUIRE(identity->names == vector<string> {"id", "path", "ok"});
}

TEST_CASE("ProjectLazyDataChunk composes stacked descriptor projections", "[execution][external_block]") {
	auto chunk = MakeWideLazyChunk({MakeDescriptor(64, 640)});

	auto first = ProjectLazyDataChunk(*chunk, {2, 0});
	auto second = ProjectLazyDataChunk(*first, {1});

	REQUIRE(second->logical_types == vector<LogicalType> {LogicalType::BIGINT});
	REQUIRE(second->names == vector<string> {"id"});
	REQUIRE(second->blocks.size() == 1);
	REQUIRE(second->blocks[0].column_ids == vector<idx_t> {0});
}

TEST_CASE("ProjectLazyDataChunk preserves wrapped STRUCT descriptors", "[execution][external_block]") {
	auto block = MakeDescriptor(64, 640);
	auto object_ref = block.object_ref;
	auto chunk = MakeWrappedStructLazyChunk({block});

	auto projected = ProjectLazyDataChunk(*chunk, {0}, {"renamed_struct"});
	REQUIRE(projected);
	REQUIRE(projected->wrap_columns_as_struct);
	REQUIRE(projected->logical_types == chunk->logical_types);
	REQUIRE(projected->names == vector<string> {"renamed_struct"});
	REQUIRE(projected->blocks.size() == 1);
	REQUIRE(projected->blocks[0].object_ref == object_ref);
	REQUIRE(projected->blocks[0].column_ids.empty());

	auto sliced = SliceLazyDataChunk(*projected, 10, 20);
	REQUIRE(sliced);
	REQUIRE(sliced->wrap_columns_as_struct);
	REQUIRE(sliced->logical_types == projected->logical_types);
	REQUIRE(sliced->names == projected->names);
	REQUIRE(sliced->blocks.size() == 1);
	REQUIRE(sliced->blocks[0].object_ref == object_ref);
	REQUIRE(sliced->blocks[0].column_ids.empty());
	REQUIRE(sliced->blocks[0].has_slice);
	REQUIRE(sliced->blocks[0].slice.start_offset == 10);
	REQUIRE(sliced->blocks[0].slice.end_offset == 30);
}

TEST_CASE("SliceLazyDataChunk rewrites row ranges across descriptors only", "[execution][external_block]") {
	auto first_block = MakeDescriptor(60, 600);
	auto second_block = MakeDescriptor(80, 800, 10, 70);
	auto first_ref = first_block.object_ref;
	auto second_ref = second_block.object_ref;
	auto chunk = MakeWideLazyChunk({first_block, second_block});

	auto sliced = SliceLazyDataChunk(*chunk, 50, 40);
	REQUIRE(sliced);
	REQUIRE(sliced->cardinality == 40);
	REQUIRE(sliced->logical_types == chunk->logical_types);
	REQUIRE(sliced->names == chunk->names);
	REQUIRE(sliced->blocks.size() == 2);
	REQUIRE(sliced->blocks[0].object_ref == first_ref);
	REQUIRE(sliced->blocks[0].has_slice);
	REQUIRE(sliced->blocks[0].slice.start_offset == 50);
	REQUIRE(sliced->blocks[0].slice.end_offset == 60);
	REQUIRE(sliced->blocks[1].object_ref == second_ref);
	REQUIRE(sliced->blocks[1].has_slice);
	REQUIRE(sliced->blocks[1].slice.start_offset == 10);
	REQUIRE(sliced->blocks[1].slice.end_offset == 40);
	REQUIRE(sliced->EstimatedBytes() == 400);
}

TEST_CASE("LazyDataChunk descriptor helpers reject invalid projection and slice", "[execution][external_block]") {
	auto chunk = MakeWideLazyChunk({MakeDescriptor(10, 100)});

	REQUIRE_THROWS_AS(ProjectLazyDataChunk(*chunk, {3}), InvalidInputException);
	REQUIRE_THROWS_AS(ProjectLazyDataChunk(*chunk, {0, 1}, {"only_one_name"}), InvalidInputException);
	REQUIRE_THROWS_AS(ProjectLazyDataChunk(*chunk, {}), InvalidInputException);
	REQUIRE_THROWS_AS(SliceLazyDataChunk(*chunk, 9, 2), InvalidInputException);
	REQUIRE_THROWS_AS(SliceLazyDataChunk(*chunk, 11, 0), InvalidInputException);
}

TEST_CASE("MaterializeExternalBlock dispatches through registered backend", "[execution][external_block]") {
	ExternalBlockBackendGuard guard;
	auto backend = make_shared_ptr<TestExternalBlockBackend>();
	auto backend_raw = backend.get();
	SetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE, backend);

	DuckDB db(nullptr);
	Connection con(db);
	auto chunk = MakeLazyChunk({MakeDescriptor(128, 1280, 16, 20)});

	auto materialized = MaterializeExternalBlock(*con.context, *chunk);
	REQUIRE(materialized);
	REQUIRE(materialized->size() == 4);
	REQUIRE(materialized->GetValue(0, 0).GetValue<int64_t>() == 16);
	REQUIRE(materialized->GetValue(0, 3).GetValue<int64_t>() == 19);
	REQUIRE(backend_raw->materialize_calls == 1);
}

TEST_CASE("MaterializeExternalBlockBarrier reports standard barrier stats", "[execution][external_block]") {
	ExternalBlockBackendGuard guard;
	auto backend = make_shared_ptr<TestExternalBlockBackend>();
	auto backend_raw = backend.get();
	SetExternalBlockBackend(ExternalBlockBackend::RAY_OBJECT_STORE, backend);

	DuckDB db(nullptr);
	Connection con(db);
	auto chunk = MakeLazyChunk({MakeDescriptor(128, 1280, 16, 20), MakeDescriptor(64, 640, 4, 8)});

	auto barrier = MaterializeExternalBlockBarrier(*con.context, *chunk, "unit_test");
	REQUIRE(barrier.chunk);
	REQUIRE(barrier.chunk->size() == 8);
	REQUIRE(barrier.stats.barrier_name == "MaterializeExternalBlock");
	REQUIRE(barrier.stats.reason == "unit_test");
	REQUIRE(barrier.stats.blocks == 2);
	REQUIRE(barrier.stats.rows == 8);
	REQUIRE(barrier.stats.estimated_bytes == 80);
	REQUIRE(backend_raw->materialize_calls == 1);
}
