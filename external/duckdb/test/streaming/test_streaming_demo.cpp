// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

//===----------------------------------------------------------------------===//
// STREAMING DEMO - Visual demonstration of TRUE STREAMING behavior
// Shows results arriving incrementally vs batch mode
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
#include <sstream>
#include <atomic>
#include <mutex>
#include <queue>
#include <random>
#include <map>
#include <condition_variable>

using namespace duckdb;
using namespace std::chrono;

//===--------------------------------------------------------------------===//
// Custom Streaming Pool with LONGER delays to demonstrate streaming
//===--------------------------------------------------------------------===//

class DemoStreamingPool {
public:
	struct CompletedTask {
		idx_t row_id;
		string result;
		int64_t completion_time_ms;
	};

	static DemoStreamingPool &Instance() {
		static DemoStreamingPool instance;
		return instance;
	}

	void Reset() {
		lock_guard<mutex> lock(result_lock_);
		while (!completed_tasks_.empty())
			completed_tasks_.pop();
		submitted_count_ = 0;
		start_time_ = high_resolution_clock::now();
	}

	void Submit(idx_t row_id, const string &prompt, int delay_ms) {
		submitted_count_++;

		// Launch async task
		std::thread([this, row_id, prompt, delay_ms]() {
			// Simulate work with specified delay
			std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));

			auto now = high_resolution_clock::now();
			auto elapsed = duration_cast<milliseconds>(now - start_time_).count();

			string result = "Response to: " + prompt + " (took " + std::to_string(delay_ms) + "ms)";

			{
				lock_guard<mutex> lock(result_lock_);
				completed_tasks_.push({row_id, result, elapsed});
			}
			result_cv_.notify_one();
		}).detach();
	}

	vector<CompletedTask> PollWait(int timeout_ms = 100) {
		vector<CompletedTask> results;

		unique_lock<mutex> lock(result_lock_);
		if (completed_tasks_.empty()) {
			result_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
			                    [this]() { return !completed_tasks_.empty(); });
		}

		while (!completed_tasks_.empty()) {
			results.push_back(std::move(completed_tasks_.front()));
			completed_tasks_.pop();
		}

		return results;
	}

private:
	DemoStreamingPool() : submitted_count_(0) {
		start_time_ = high_resolution_clock::now();
	}

	std::queue<CompletedTask> completed_tasks_;
	mutex result_lock_;
	std::condition_variable result_cv_;
	std::atomic<idx_t> submitted_count_;
	high_resolution_clock::time_point start_time_;
};

//===--------------------------------------------------------------------===//
// Visual Demo Functions
//===--------------------------------------------------------------------===//

void PrintProgressBar(int current, int total, int width = 40) {
	float progress = (float)current / total;
	int pos = width * progress;

	std::cout << "[";
	for (int i = 0; i < width; ++i) {
		if (i < pos)
			std::cout << "=";
		else if (i == pos)
			std::cout << ">";
		else
			std::cout << " ";
	}
	std::cout << "] " << std::setw(3) << int(progress * 100.0) << "% ";
	std::cout << "(" << current << "/" << total << ")" << std::flush;
}

void DemoBatchVsStreaming() {
	std::cout << "╔═══════════════════════════════════════════════════════════════╗" << std::endl;
	std::cout << "║     BATCH vs TRUE STREAMING - Visual Demonstration            ║" << std::endl;
	std::cout << "║     Watch how results flow to downstream differently!         ║" << std::endl;
	std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
	std::cout << std::endl;

	const int NUM_TASKS = 10;

	// Assign different delays to each task (simulating variable vLLM response times)
	std::vector<int> delays = {50, 200, 30, 150, 80, 250, 40, 180, 60, 220};

	std::cout << "Task delays (ms): ";
	for (int d : delays)
		std::cout << d << " ";
	std::cout << std::endl << std::endl;

	//===================================================================
	// Demo 1: BATCH mode - downstream waits for ALL results
	//===================================================================
	std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
	std::cout << "📦 BATCH MODE: Downstream waits for ALL results" << std::endl;
	std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;

	auto &pool = DemoStreamingPool::Instance();
	pool.Reset();

	auto batch_start = high_resolution_clock::now();

	// Submit all tasks
	std::vector<std::future<string>> futures;
	for (int i = 0; i < NUM_TASKS; i++) {
		pool.Submit(i, "Task " + std::to_string(i), delays[i]);
	}

	std::cout << "Submitted all " << NUM_TASKS << " tasks..." << std::endl;
	std::cout << "Waiting for ALL to complete before processing..." << std::endl;
	std::cout << std::endl;

	// Wait for ALL results (this is what batch mode does)
	std::map<idx_t, string> all_results;
	while (all_results.size() < NUM_TASKS) {
		auto completed = pool.PollWait(50);
		for (auto &task : completed) {
			all_results[task.row_id] = task.result;
		}

		// Show waiting status
		std::cout << "\r";
		PrintProgressBar(all_results.size(), NUM_TASKS);
	}
	std::cout << std::endl << std::endl;

	auto batch_first_output = high_resolution_clock::now();

	// NOW we can output results (downstream starts here)
	std::cout << "✅ All results ready! Downstream can now process:" << std::endl;
	for (int i = 0; i < NUM_TASKS; i++) {
		std::cout << "  Row " << i << ": " << all_results[i] << std::endl;
	}

	auto batch_end = high_resolution_clock::now();
	auto batch_time_to_first = duration_cast<milliseconds>(batch_first_output - batch_start).count();
	auto batch_total = duration_cast<milliseconds>(batch_end - batch_start).count();

	std::cout << std::endl;
	std::cout << "⏱️  BATCH Metrics:" << std::endl;
	std::cout << "   Time to first output: " << batch_time_to_first << " ms" << std::endl;
	std::cout << "   Total time: " << batch_total << " ms" << std::endl;
	std::cout << "   ⚠️  Downstream blocked for " << batch_time_to_first << " ms!" << std::endl;

	//===================================================================
	// Demo 2: TRUE STREAMING - results flow as they complete
	//===================================================================
	std::cout << std::endl;
	std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;
	std::cout << "🌊 TRUE STREAMING: Results flow as they complete" << std::endl;
	std::cout << "═══════════════════════════════════════════════════════════════" << std::endl;

	pool.Reset();

	auto stream_start = high_resolution_clock::now();
	int64_t stream_time_to_first = -1;

	// Submit all tasks
	for (int i = 0; i < NUM_TASKS; i++) {
		pool.Submit(i, "Task " + std::to_string(i), delays[i]);
	}

	std::cout << "Submitted all " << NUM_TASKS << " tasks..." << std::endl;
	std::cout << "Streaming results to downstream AS THEY COMPLETE:" << std::endl;
	std::cout << std::endl;

	std::map<idx_t, string> stream_results;
	idx_t next_output = 0;
	int output_count = 0;

	std::cout << "| Time (ms) | Row | Status |" << std::endl;
	std::cout << "|-----------|-----|--------|" << std::endl;

	while (output_count < NUM_TASKS) {
		auto completed = pool.PollWait(20);

		auto now = high_resolution_clock::now();
		auto elapsed = duration_cast<milliseconds>(now - stream_start).count();

		for (auto &task : completed) {
			stream_results[task.row_id] = task.result;
		}

		// Output consecutive completed rows (TRUE STREAMING!)
		while (stream_results.find(next_output) != stream_results.end()) {
			if (stream_time_to_first < 0) {
				stream_time_to_first = elapsed;
			}

			std::cout << "| " << std::setw(9) << elapsed << " | " << std::setw(3) << next_output
			          << " | READY  | → Downstream processing!" << std::endl;

			stream_results.erase(next_output);
			next_output++;
			output_count++;
		}
	}

	auto stream_end = high_resolution_clock::now();
	auto stream_total = duration_cast<milliseconds>(stream_end - stream_start).count();

	std::cout << std::endl;
	std::cout << "⏱️  STREAMING Metrics:" << std::endl;
	std::cout << "   Time to first output: " << stream_time_to_first << " ms" << std::endl;
	std::cout << "   Total time: " << stream_total << " ms" << std::endl;
	std::cout << "   🎉 Downstream started " << (batch_time_to_first - stream_time_to_first) << " ms earlier!"
	          << std::endl;

	//===================================================================
	// Summary
	//===================================================================
	std::cout << std::endl;
	std::cout << "╔═══════════════════════════════════════════════════════════════╗" << std::endl;
	std::cout << "║                       SUMMARY                                 ║" << std::endl;
	std::cout << "╠═══════════════════════════════════════════════════════════════╣" << std::endl;
	std::cout << "║  Metric                    │  BATCH    │  STREAMING │ Winner  ║" << std::endl;
	std::cout << "╠════════════════════════════╪═══════════╪════════════╪═════════╣" << std::endl;
	std::cout << "║  Time to first result      │  " << std::setw(6) << batch_time_to_first << " ms │  " << std::setw(7)
	          << stream_time_to_first << " ms │ STREAM  ║" << std::endl;
	std::cout << "║  Total execution time      │  " << std::setw(6) << batch_total << " ms │  " << std::setw(7)
	          << stream_total << " ms │ SAME    ║" << std::endl;
	std::cout << "║  Downstream latency        │  " << std::setw(6) << batch_time_to_first << " ms │  " << std::setw(7)
	          << stream_time_to_first << " ms │ STREAM  ║" << std::endl;
	std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
	std::cout << std::endl;
	std::cout << "💡 KEY INSIGHT:" << std::endl;
	std::cout << "   TRUE STREAMING reduces downstream latency by "
	          << ((float)(batch_time_to_first - stream_time_to_first) / batch_time_to_first * 100) << "%!" << std::endl;
	std::cout << "   This is critical for:" << std::endl;
	std::cout << "   • Real-time dashboards" << std::endl;
	std::cout << "   • Interactive applications" << std::endl;
	std::cout << "   • Pipeline operators that can process partial results" << std::endl;
	std::cout << "   • Memory efficiency (don't buffer all results)" << std::endl;
}

void DemoOutOfOrderCompletion() {
	std::cout << std::endl;
	std::cout << "╔═══════════════════════════════════════════════════════════════╗" << std::endl;
	std::cout << "║     OUT-OF-ORDER Completion Demo                              ║" << std::endl;
	std::cout << "║     Shows how streaming handles variable response times       ║" << std::endl;
	std::cout << "╚═══════════════════════════════════════════════════════════════╝" << std::endl;
	std::cout << std::endl;

	auto &pool = DemoStreamingPool::Instance();
	pool.Reset();

	// Design delays to show out-of-order completion clearly
	// Row 0 is SLOW (200ms), Row 1 is FAST (20ms), etc.
	std::vector<std::pair<int, int>> tasks = {
	    {0, 200}, // Row 0: 200ms (slow)
	    {1, 20},  // Row 1: 20ms (fast)
	    {2, 150}, // Row 2: 150ms (medium)
	    {3, 30},  // Row 3: 30ms (fast)
	    {4, 180}, // Row 4: 180ms (slow)
	};

	std::cout << "Designed scenario:" << std::endl;
	for (auto &t : tasks) {
		std::cout << "  Row " << t.first << ": " << t.second << "ms delay" << std::endl;
	}
	std::cout << std::endl;

	auto start = high_resolution_clock::now();

	// Submit all
	for (auto &t : tasks) {
		pool.Submit(t.first, "Task " + std::to_string(t.first), t.second);
	}

	std::cout << "Completion order (not row order!):" << std::endl;
	std::cout << "| Elapsed (ms) | Completed Row | Notes |" << std::endl;
	std::cout << "|--------------|---------------|-------|" << std::endl;

	int completed = 0;
	std::vector<int> completion_order;

	while (completed < tasks.size()) {
		auto results = pool.PollWait(50);

		for (auto &r : results) {
			completion_order.push_back(r.row_id);
			std::cout << "| " << std::setw(12) << r.completion_time_ms << " | " << std::setw(13) << r.row_id << " | ";

			if (r.row_id == 1 || r.row_id == 3) {
				std::cout << "Fast task!";
			} else {
				std::cout << "Slow task";
			}
			std::cout << " |" << std::endl;
			completed++;
		}
	}

	std::cout << std::endl;
	std::cout << "Actual completion order: ";
	for (int i : completion_order)
		std::cout << i << " ";
	std::cout << std::endl;
	std::cout << "Expected row order:      0 1 2 3 4" << std::endl;
	std::cout << std::endl;
	std::cout << "💡 Even though tasks complete out of order, TRUE STREAMING" << std::endl;
	std::cout << "   can buffer and output in correct row order while still" << std::endl;
	std::cout << "   allowing downstream to start as soon as Row 0 completes!" << std::endl;
}

int main() {
	DemoBatchVsStreaming();
	DemoOutOfOrderCompletion();

	std::cout << std::endl;
	std::cout << "✅ Demo completed!" << std::endl;

	return 0;
}
