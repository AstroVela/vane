// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb.hpp"
#include "duckdb/execution/streaming/streaming.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/common/exception.hpp"
#include <iostream>
#include <cassert>

using namespace duckdb;

// Test BoundedChannel basic operations
void TestBoundedChannel() {
	std::cout << "Testing BoundedChannel..." << std::endl;

	BoundedChannel<int> channel(3);

	// Test TrySend
	assert(channel.TrySend(1) == true);
	assert(channel.TrySend(2) == true);
	assert(channel.TrySend(3) == true);
	assert(channel.TrySend(4) == false); // Channel full

	// Test TryRecv
	RecvResult<int> result = channel.TryRecv();
	assert(result.has_value == true);
	assert(result.value == 1);

	result = channel.TryRecv();
	assert(result.has_value == true);
	assert(result.value == 2);

	result = channel.TryRecv();
	assert(result.has_value == true);
	assert(result.value == 3);

	result = channel.TryRecv();
	assert(result.has_value == false); // Channel empty

	std::cout << "BoundedChannel tests passed!" << std::endl;
}

// Test DataChunkChannel with unique_ptr
void TestDataChunkChannel() {
	std::cout << "Testing DataChunkChannel..." << std::endl;

	DataChunkChannel channel(2);

	// Create test DataChunks using unique_ptr
	auto chunk1 = make_uniq<DataChunk>();
	auto chunk2 = make_uniq<DataChunk>();
	DataChunk *raw1 = chunk1.get();
	DataChunk *raw2 = chunk2.get();

	assert(channel.TrySend(std::move(chunk1)) == true);
	assert(channel.TrySend(std::move(chunk2)) == true);

	RecvResult<unique_ptr<DataChunk>> result = channel.TryRecv();
	assert(result.has_value == true);
	assert(result.value.get() == raw1);

	result = channel.TryRecv();
	assert(result.has_value == true);
	assert(result.value.get() == raw2);

	std::cout << "DataChunkChannel tests passed!" << std::endl;
}

// Test AsyncSinkConfig
void TestAsyncSinkConfig() {
	std::cout << "Testing AsyncSinkConfig..." << std::endl;

	AsyncSinkConfig config;
	config.channel_capacity = 100;
	config.max_concurrent_ops = 8;
	config.poll_interval_ms = 5;
	config.batch_size = 16;

	assert(config.channel_capacity == 100);
	assert(config.max_concurrent_ops == 8);
	assert(config.poll_interval_ms == 5);
	assert(config.batch_size == 16);

	std::cout << "AsyncSinkConfig tests passed!" << std::endl;
}

// Test StreamingResult
void TestStreamingResult() {
	std::cout << "Testing StreamingResult..." << std::endl;

	// Create a channel for StreamingResult
	auto channel = make_shared_ptr<DataChunkChannel>(10);

	// Add some chunks to the channel
	auto chunk1 = make_uniq<DataChunk>();
	auto chunk2 = make_uniq<DataChunk>();
	DataChunk *raw1 = chunk1.get();

	channel->TrySend(std::move(chunk1));
	channel->TrySend(std::move(chunk2));

	// Create StreamingResult with channel
	StreamingResult result(channel);

	// Test HasPending
	assert(result.HasPending() == true);

	// Try to get a result (non-blocking)
	unique_ptr<DataChunk> retrieved = result.TryFetchNext();
	assert(retrieved.get() == raw1);

	std::cout << "StreamingResult tests passed!" << std::endl;
}

// Test VLLMConfig
void TestVLLMConfig() {
	std::cout << "Testing VLLMConfig..." << std::endl;

	VLLMConfig config;
	config.base_url = "http://localhost:8000";
	config.model = "llama-7b";
	config.max_tokens = 100;
	config.temperature = 0.7;
	config.timeout_seconds = 30;
	config.max_concurrent_ops = 4;
	config.input_column_idx = 0;

	assert(config.base_url == "http://localhost:8000");
	assert(config.model == "llama-7b");
	assert(config.max_tokens == 100);
	assert(config.temperature == 0.7);
	assert(config.timeout_seconds == 30);
	assert(config.max_concurrent_ops == 4);

	std::cout << "VLLMConfig tests passed!" << std::endl;
}

// Test AsyncSinkResult
void TestAsyncSinkResult() {
	std::cout << "Testing AsyncSinkResult..." << std::endl;

	// Test default constructor
	AsyncSinkResult r1;
	assert(r1.type == AsyncSinkResultType::NEED_MORE_INPUT);

	// Test Error
	AsyncSinkResult r2 = AsyncSinkResult::Error("test error");
	assert(r2.type == AsyncSinkResultType::ERROR);
	assert(r2.error_message == "test error");

	// Test Blocked
	AsyncSinkResult r3 = AsyncSinkResult::Blocked();
	assert(r3.type == AsyncSinkResultType::BLOCKED);

	// Test Finished
	auto chunk = make_uniq<DataChunk>();
	DataChunk *raw = chunk.get();
	AsyncSinkResult r4 = AsyncSinkResult::Finished(std::move(chunk));
	assert(r4.type == AsyncSinkResultType::FINISHED);
	assert(r4.result.get() == raw);

	std::cout << "AsyncSinkResult tests passed!" << std::endl;
}

// Test VLLMGenerateFun scalar function definition
void TestVLLMGenerateFunDefinition() {
	std::cout << "Testing VLLMGenerateFun definition..." << std::endl;

	ScalarFunctionSet funcs = VLLMGenerateFun::GetFunctions();

	// Check that we have function overloads
	assert(funcs.name == "vllm_generate");

	std::cout << "VLLMGenerateFun definition tests passed!" << std::endl;
}

// Test VLLMBatchGenerateFun table function definition
void TestVLLMBatchGenerateFunDefinition() {
	std::cout << "Testing VLLMBatchGenerateFun definition..." << std::endl;

	TableFunction func = VLLMBatchGenerateFun::GetFunction();

	assert(func.name == "vllm_batch_generate");

	std::cout << "VLLMBatchGenerateFun definition tests passed!" << std::endl;
}

// Test using DuckDB Connection (integration test)
void TestDuckDBIntegration() {
	std::cout << "Testing DuckDB integration..." << std::endl;

	// Create an in-memory database
	DuckDB db(nullptr);
	Connection con(db);

	// Test basic query
	auto result = con.Query("SELECT 1 + 1 as result");
	assert(!result->HasError());
	assert(result->Fetch()->GetValue(0, 0) == Value(2));

	// Test creating a table
	result = con.Query("CREATE TABLE test_prompts (id INTEGER, prompt VARCHAR)");
	assert(!result->HasError());

	result = con.Query("INSERT INTO test_prompts VALUES (1, 'Hello'), (2, 'World')");
	assert(!result->HasError());

	result = con.Query("SELECT * FROM test_prompts ORDER BY id");
	assert(!result->HasError());

	auto chunk = result->Fetch();
	assert(chunk->size() == 2);

	std::cout << "DuckDB integration tests passed!" << std::endl;
}

int main() {
	std::cout << "=== DuckDB Streaming Module Tests ===" << std::endl;

	// Core component tests
	TestBoundedChannel();
	TestDataChunkChannel();
	TestAsyncSinkConfig();
	TestStreamingResult();
	TestVLLMConfig();
	TestAsyncSinkResult();

	// Function definition tests
	TestVLLMGenerateFunDefinition();
	TestVLLMBatchGenerateFunDefinition();

	// DuckDB integration test
	TestDuckDBIntegration();

	std::cout << std::endl;
	std::cout << "=== All tests passed! ===" << std::endl;

	return 0;
}
