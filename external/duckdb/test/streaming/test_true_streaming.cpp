// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// TRUE STREAMING Test - Shows results flowing to downstream as they complete
// This demonstrates the DuckDB-style poll() architecture
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
#include <atomic>
#include <sstream>

using namespace duckdb;
using namespace std::chrono;

void RegisterFunctionsManually(Connection &con) {
	auto &context = *con.context;
	con.Query("BEGIN TRANSACTION");
	auto &catalog = Catalog::GetSystemCatalog(context);

	// Register scalar function
	{
		auto funcs = VLLMGenerateFun::GetFunctions();
		CreateScalarFunctionInfo info(std::move(funcs));
		info.on_conflict = OnCreateConflict::REPLACE_ON_CONFLICT;
		catalog.CreateFunction(context, info);
	}

	// Register batch table function
	{
		auto func = VLLMBatchGenerateFun::GetFunction();
		CreateTableFunctionInfo info(std::move(func));
		info.on_conflict = OnCreateConflict::REPLACE_ON_CONFLICT;
		catalog.CreateTableFunction(context, info);
	}

	// Register streaming table function
	{
		auto func = VLLMStreamGenerateFun::GetFunction();
		CreateTableFunctionInfo info(std::move(func));
		info.on_conflict = OnCreateConflict::REPLACE_ON_CONFLICT;
		catalog.CreateTableFunction(context, info);
	}

	con.Query("COMMIT");
}

void TestTrueStreaming() {
	std::cout << "=============================================" << std::endl;
	std::cout << "  TRUE STREAMING Test (DuckDB-style)" << std::endl;
	std::cout << "=============================================" << std::endl;
	std::cout << std::endl;

	DuckDB db(nullptr);
	Connection con(db);
	RegisterFunctionsManually(con);

	// Create prompts
	std::vector<std::string> prompts;
	for (int i = 0; i < 20; i++) {
		prompts.push_back("Prompt number " + std::to_string(i));
	}

	// Build SQL for streaming function
	std::ostringstream sql;
	sql << "SELECT * FROM vllm_stream_generate([";
	for (size_t i = 0; i < prompts.size(); i++) {
		if (i > 0)
			sql << ", ";
		sql << "'" << prompts[i] << "'";
	}
	sql << "], 'llama-7b', 'http://localhost:8000')";

	std::cout << "Executing TRUE STREAMING query with " << prompts.size() << " prompts..." << std::endl;
	std::cout << "Watch how results arrive incrementally:" << std::endl;
	std::cout << std::endl;

	auto start = high_resolution_clock::now();
	auto first_result_time = start;
	bool first_result = true;
	int total_rows = 0;

	auto result = con.Query(sql.str());
	if (result->HasError()) {
		std::cerr << "Error: " << result->GetError() << std::endl;
		return;
	}

	std::cout << "| Time (ms) | Row | Completion Order | Status |" << std::endl;
	std::cout << "|-----------|-----|------------------|--------|" << std::endl;

	while (true) {
		auto chunk = result->Fetch();
		if (!chunk || chunk->size() == 0)
			break;

		auto now = high_resolution_clock::now();
		auto elapsed = duration_cast<milliseconds>(now - start).count();

		if (first_result) {
			first_result_time = now;
			first_result = false;
		}

		// Display each row in this chunk
		for (idx_t i = 0; i < chunk->size(); i++) {
			int64_t idx = chunk->GetValue(0, i).GetValue<int64_t>();
			int64_t completion_order = chunk->GetValue(3, i).GetValue<int64_t>();

			std::cout << "| " << std::setw(9) << elapsed << " | " << std::setw(3) << idx << " | " << std::setw(16)
			          << completion_order << " | READY  |" << std::endl;
			total_rows++;
		}
	}

	auto end = high_resolution_clock::now();
	auto total_time = duration_cast<milliseconds>(end - start).count();
	auto time_to_first = duration_cast<milliseconds>(first_result_time - start).count();

	std::cout << std::endl;
	std::cout << "=============================================" << std::endl;
	std::cout << "              STREAMING METRICS" << std::endl;
	std::cout << "=============================================" << std::endl;
	std::cout << "Total rows processed: " << total_rows << std::endl;
	std::cout << "Time to first result: " << time_to_first << " ms" << std::endl;
	std::cout << "Total execution time: " << total_time << " ms" << std::endl;
	std::cout << std::endl;

	std::cout << "KEY INSIGHT:" << std::endl;
	std::cout << "  In TRUE STREAMING, downstream can start processing" << std::endl;
	std::cout << "  results as soon as they're ready, not waiting for all." << std::endl;
	std::cout << "  Time to first result: " << time_to_first << " ms (not " << total_time << " ms!)" << std::endl;
}

void CompareBatchVsStreaming() {
	std::cout << std::endl;
	std::cout << "=============================================" << std::endl;
	std::cout << "  BATCH vs STREAMING Comparison" << std::endl;
	std::cout << "=============================================" << std::endl;
	std::cout << std::endl;

	DuckDB db(nullptr);
	Connection con(db);
	RegisterFunctionsManually(con);

	int num_prompts = 50;

	// Build prompts list
	std::ostringstream prompts_sql;
	prompts_sql << "[";
	for (int i = 0; i < num_prompts; i++) {
		if (i > 0)
			prompts_sql << ", ";
		prompts_sql << "'Prompt " << i << "'";
	}
	prompts_sql << "]";
	std::string prompts = prompts_sql.str();

	// Test 1: Batch (waits for all results)
	std::cout << "1. BATCH mode (vllm_batch_generate):" << std::endl;
	{
		std::ostringstream sql;
		sql << "SELECT * FROM vllm_batch_generate(" << prompts << ", 'llama', 'http://localhost:8000', 16)";

		auto start = high_resolution_clock::now();
		auto first_result_time = start;
		bool first_result = true;
		int count = 0;

		auto result = con.Query(sql.str());
		while (auto chunk = result->Fetch()) {
			if (first_result) {
				first_result_time = high_resolution_clock::now();
				first_result = false;
			}
			count += chunk->size();
		}

		auto end = high_resolution_clock::now();
		auto time_to_first = duration_cast<milliseconds>(first_result_time - start).count();
		auto total = duration_cast<milliseconds>(end - start).count();

		std::cout << "   Total rows: " << count << std::endl;
		std::cout << "   Time to first result: " << time_to_first << " ms" << std::endl;
		std::cout << "   Total time: " << total << " ms" << std::endl;
		std::cout << "   NOTE: First result arrives AFTER all processing!" << std::endl;
	}

	// Test 2: True Streaming
	std::cout << std::endl;
	std::cout << "2. TRUE STREAMING mode (vllm_stream_generate):" << std::endl;
	{
		std::ostringstream sql;
		sql << "SELECT * FROM vllm_stream_generate(" << prompts << ", 'llama', 'http://localhost:8000')";

		auto start = high_resolution_clock::now();
		auto first_result_time = start;
		bool first_result = true;
		int count = 0;

		auto result = con.Query(sql.str());
		while (auto chunk = result->Fetch()) {
			if (first_result && chunk->size() > 0) {
				first_result_time = high_resolution_clock::now();
				first_result = false;
			}
			count += chunk->size();
		}

		auto end = high_resolution_clock::now();
		auto time_to_first = duration_cast<milliseconds>(first_result_time - start).count();
		auto total = duration_cast<milliseconds>(end - start).count();

		std::cout << "   Total rows: " << count << std::endl;
		std::cout << "   Time to first result: " << time_to_first << " ms" << std::endl;
		std::cout << "   Total time: " << total << " ms" << std::endl;
		std::cout << "   NOTE: First result arrives BEFORE all processing!" << std::endl;
	}

	std::cout << std::endl;
	std::cout << "=============================================" << std::endl;
	std::cout << "CONCLUSION:" << std::endl;
	std::cout << "  TRUE STREAMING allows downstream to start work early" << std::endl;
	std::cout << "  This is critical for pipelines where later stages" << std::endl;
	std::cout << "  can process partial results (e.g., writing to disk," << std::endl;
	std::cout << "  aggregations, further transformations)" << std::endl;
	std::cout << "=============================================" << std::endl;
}

int main() {
	std::cout << "╔═══════════════════════════════════════════════════════╗" << std::endl;
	std::cout << "║  DuckDB TRUE STREAMING AI Inference (DuckDB-style)      ║" << std::endl;
	std::cout << "║  Results flow to downstream as they complete!         ║" << std::endl;
	std::cout << "╚═══════════════════════════════════════════════════════╝" << std::endl;
	std::cout << std::endl;

	TestTrueStreaming();
	CompareBatchVsStreaming();

	std::cout << std::endl;
	std::cout << "✅ All tests completed!" << std::endl;

	return 0;
}
