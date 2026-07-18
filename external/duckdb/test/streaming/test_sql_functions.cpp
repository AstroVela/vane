// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// Test SQL functions by registering them through extension mechanism
//===----------------------------------------------------------------------===//

#include "duckdb.hpp"
#include "duckdb/execution/streaming/streaming.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/transaction/transaction_context.hpp"
#include <iostream>

using namespace duckdb;

void RegisterFunctionsManually(Connection &con) {
	// Get the client context
	auto &context = *con.context;

	// Begin transaction for function registration
	con.Query("BEGIN TRANSACTION");

	auto &catalog = Catalog::GetSystemCatalog(context);

	// Register vllm_generate scalar function
	{
		auto funcs = VLLMGenerateFun::GetFunctions();
		CreateScalarFunctionInfo info(std::move(funcs));
		info.on_conflict = OnCreateConflict::REPLACE_ON_CONFLICT;
		catalog.CreateFunction(context, info);
	}

	// Register vllm_batch_generate table function
	{
		auto func = VLLMBatchGenerateFun::GetFunction();
		CreateTableFunctionInfo info(std::move(func));
		info.on_conflict = OnCreateConflict::REPLACE_ON_CONFLICT;
		catalog.CreateTableFunction(context, info);
	}

	// Register async_udf function
	{
		auto funcs = AsyncUDFFun::GetFunctions();
		CreateScalarFunctionInfo info(std::move(funcs));
		info.on_conflict = OnCreateConflict::REPLACE_ON_CONFLICT;
		catalog.CreateFunction(context, info);
	}

	// Commit transaction
	con.Query("COMMIT");
}

int main() {
	std::cout << "=== DuckDB Streaming SQL Function Tests ===" << std::endl;

	// Create an in-memory database
	DuckDB db(nullptr);
	Connection con(db);

	// Register our functions
	std::cout << "\n1. Registering streaming functions..." << std::endl;
	RegisterFunctionsManually(con);
	std::cout << "   Functions registered successfully!" << std::endl;

	// Test vllm_generate with single argument
	std::cout << "\n2. Testing vllm_generate(prompt)..." << std::endl;
	auto result = con.Query("SELECT vllm_generate('Hello, how are you?') as response");
	if (result->HasError()) {
		std::cerr << "   Error: " << result->GetError() << std::endl;
		return 1;
	}
	auto chunk = result->Fetch();
	if (chunk && chunk->size() > 0) {
		std::cout << "   Result: " << chunk->GetValue(0, 0).ToString() << std::endl;
	}

	// Test vllm_generate with model argument
	std::cout << "\n3. Testing vllm_generate(prompt, model)..." << std::endl;
	result = con.Query("SELECT vllm_generate('Tell me a joke', 'llama-7b') as response");
	if (result->HasError()) {
		std::cerr << "   Error: " << result->GetError() << std::endl;
		return 1;
	}
	chunk = result->Fetch();
	if (chunk && chunk->size() > 0) {
		std::cout << "   Result: " << chunk->GetValue(0, 0).ToString() << std::endl;
	}

	// Test vllm_generate with full arguments
	std::cout << "\n4. Testing vllm_generate(prompt, model, url)..." << std::endl;
	result = con.Query("SELECT vllm_generate('What is 2+2?', 'gpt-4', 'http://api.example.com') as response");
	if (result->HasError()) {
		std::cerr << "   Error: " << result->GetError() << std::endl;
		return 1;
	}
	chunk = result->Fetch();
	if (chunk && chunk->size() > 0) {
		std::cout << "   Result: " << chunk->GetValue(0, 0).ToString() << std::endl;
	}

	// Test with a table
	std::cout << "\n5. Testing vllm_generate with table data..." << std::endl;
	con.Query("CREATE TABLE prompts (id INTEGER, prompt VARCHAR)");
	con.Query("INSERT INTO prompts VALUES (1, 'Hello'), (2, 'World'), (3, 'DuckDB rocks!')");

	result = con.Query("SELECT id, prompt, vllm_generate(prompt) as response FROM prompts ORDER BY id");
	if (result->HasError()) {
		std::cerr << "   Error: " << result->GetError() << std::endl;
		return 1;
	}

	std::cout << "   Results:" << std::endl;
	while (true) {
		chunk = result->Fetch();
		if (!chunk || chunk->size() == 0)
			break;
		for (idx_t i = 0; i < chunk->size(); i++) {
			std::cout << "   - ID: " << chunk->GetValue(0, i).ToString()
			          << ", Prompt: " << chunk->GetValue(1, i).ToString()
			          << ", Response: " << chunk->GetValue(2, i).ToString() << std::endl;
		}
	}

	// Test vllm_batch_generate table function
	std::cout << "\n6. Testing vllm_batch_generate table function..." << std::endl;
	result = con.Query(
	    "SELECT * FROM vllm_batch_generate(['Hello', 'World', 'Test'], 'llama-7b', 'http://localhost:8000', 4)");
	if (result->HasError()) {
		std::cerr << "   Error: " << result->GetError() << std::endl;
		return 1;
	}

	std::cout << "   Batch results:" << std::endl;
	while (true) {
		chunk = result->Fetch();
		if (!chunk || chunk->size() == 0)
			break;
		for (idx_t i = 0; i < chunk->size(); i++) {
			std::cout << "   - Idx: " << chunk->GetValue(0, i).ToString()
			          << ", Prompt: " << chunk->GetValue(1, i).ToString()
			          << ", Response: " << chunk->GetValue(2, i).ToString() << std::endl;
		}
	}

	std::cout << "\n=== All SQL function tests passed! ===" << std::endl;

	return 0;
}
