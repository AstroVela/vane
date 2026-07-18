// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Performance comparison test: Async vs Sync execution
// Shows the benefit of DuckDB-style async execution for AI inference
//===----------------------------------------------------------------------===//

#include "duckdb.hpp"
#include "duckdb/execution/streaming/streaming.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/main/client_context.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <future>
#include <sstream>

using namespace duckdb;
using namespace std::chrono;

// Simulated sync vLLM call (blocking)
static string SyncVLLMCall(const string &prompt) {
	std::this_thread::sleep_for(std::chrono::milliseconds(50)); // Simulate 50ms latency
	return "[Sync] Response to: " + prompt.substr(0, 20);
}

// Benchmark: Sync execution (one by one)
double BenchmarkSync(int num_rows) {
	auto start = high_resolution_clock::now();

	for (int i = 0; i < num_rows; i++) {
		std::ostringstream oss;
		oss << "Prompt " << i;
		SyncVLLMCall(oss.str());
	}

	auto end = high_resolution_clock::now();
	return duration_cast<milliseconds>(end - start).count();
}

// Benchmark: Async execution (all concurrent)
double BenchmarkAsync(int num_rows, int max_concurrent) {
	auto start = high_resolution_clock::now();

	vector<std::future<string>> futures;

	for (int i = 0; i < num_rows; i += max_concurrent) {
		int batch_end = std::min(i + max_concurrent, num_rows);
		futures.clear();

		// Submit batch
		for (int j = i; j < batch_end; j++) {
			std::ostringstream oss;
			oss << "Prompt " << j;
			string prompt = oss.str();

			futures.push_back(std::async(std::launch::async, [prompt]() {
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
				return "[Async] Response to: " + prompt.substr(0, 20);
			}));
		}

		// Wait for batch
		for (auto &f : futures) {
			f.get();
		}
	}

	auto end = high_resolution_clock::now();
	return duration_cast<milliseconds>(end - start).count();
}

void RegisterFunctionsManually(Connection &con) {
	auto &context = *con.context;
	con.Query("BEGIN TRANSACTION");
	auto &catalog = Catalog::GetSystemCatalog(context);

	{
		auto funcs = VLLMGenerateFun::GetFunctions();
		CreateScalarFunctionInfo info(std::move(funcs));
		info.on_conflict = OnCreateConflict::REPLACE_ON_CONFLICT;
		catalog.CreateFunction(context, info);
	}

	{
		auto func = VLLMBatchGenerateFun::GetFunction();
		CreateTableFunctionInfo info(std::move(func));
		info.on_conflict = OnCreateConflict::REPLACE_ON_CONFLICT;
		catalog.CreateTableFunction(context, info);
	}

	con.Query("COMMIT");
}

// Benchmark DuckDB vllm_generate function
double BenchmarkDuckDBAsync(Connection &con, int num_rows) {
	// Create test data
	std::ostringstream create_sql;
	create_sql << "CREATE OR REPLACE TABLE bench_prompts AS SELECT 'Prompt ' || i as prompt FROM range(" << num_rows
	           << ") t(i)";
	con.Query(create_sql.str());

	auto start = high_resolution_clock::now();

	auto result = con.Query("SELECT prompt, vllm_generate(prompt) as response FROM bench_prompts");
	if (result->HasError()) {
		std::cerr << "Error: " << result->GetError() << std::endl;
		return -1;
	}

	// Consume all results
	while (auto chunk = result->Fetch()) {
		// Just consume
	}

	auto end = high_resolution_clock::now();
	return duration_cast<milliseconds>(end - start).count();
}

int main() {
	std::cout << "=============================================" << std::endl;
	std::cout << "  DuckDB Streaming AI - Performance Benchmark" << std::endl;
	std::cout << "  DuckDB-style Async vs Sync Execution" << std::endl;
	std::cout << "=============================================" << std::endl;
	std::cout << std::endl;

	std::cout << "Simulated vLLM latency: 50ms per request" << std::endl;
	std::cout << "Thread pool size: 16 workers" << std::endl;
	std::cout << std::endl;

	// Small test first
	int num_rows = 32;

	std::cout << "--- Benchmark with " << num_rows << " rows ---" << std::endl;

	// Sync benchmark
	std::cout << "\n1. Sync execution (sequential):" << std::endl;
	double sync_time = BenchmarkSync(num_rows);
	std::cout << "   Time: " << sync_time << " ms" << std::endl;
	std::cout << "   Throughput: " << (num_rows * 1000.0 / sync_time) << " rows/sec" << std::endl;

	// Async benchmark (pure C++)
	std::cout << "\n2. Async execution (concurrent, max 16):" << std::endl;
	double async_time = BenchmarkAsync(num_rows, 16);
	std::cout << "   Time: " << async_time << " ms" << std::endl;
	std::cout << "   Throughput: " << (num_rows * 1000.0 / async_time) << " rows/sec" << std::endl;
	std::cout << "   Speedup vs Sync: " << (sync_time / async_time) << "x" << std::endl;

	// DuckDB benchmark
	std::cout << "\n3. DuckDB vllm_generate (DuckDB-style async):" << std::endl;

	DuckDB db(nullptr);
	Connection con(db);
	RegisterFunctionsManually(con);

	double duckdb_time = BenchmarkDuckDBAsync(con, num_rows);
	std::cout << "   Time: " << duckdb_time << " ms" << std::endl;
	std::cout << "   Throughput: " << (num_rows * 1000.0 / duckdb_time) << " rows/sec" << std::endl;
	std::cout << "   Speedup vs Sync: " << (sync_time / duckdb_time) << "x" << std::endl;

	// Summary
	std::cout << "\n=============================================" << std::endl;
	std::cout << "                  SUMMARY" << std::endl;
	std::cout << "=============================================" << std::endl;
	std::cout << "Rows processed: " << num_rows << std::endl;
	std::cout << std::endl;
	std::cout << "| Method               | Time (ms) | Speedup |" << std::endl;
	std::cout << "|---------------------|-----------|---------|" << std::endl;
	std::cout << "| Sync (sequential)   | " << std::setw(9) << sync_time << " | 1.0x    |" << std::endl;
	std::cout << "| Async (C++ threads) | " << std::setw(9) << async_time << " | " << std::fixed << std::setprecision(1)
	          << (sync_time / async_time) << "x    |" << std::endl;
	std::cout << "| DuckDB vllm_generate| " << std::setw(9) << duckdb_time << " | " << std::fixed
	          << std::setprecision(1) << (sync_time / duckdb_time) << "x    |" << std::endl;
	std::cout << "=============================================" << std::endl;

	std::cout << "\n✅ DuckDB-style async execution provides ~" << std::fixed << std::setprecision(0)
	          << (sync_time / duckdb_time) << "x speedup!" << std::endl;

	return 0;
}
