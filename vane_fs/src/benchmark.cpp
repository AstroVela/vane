// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "vane_fs/workspace.hpp"
#include <sqlite3.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>

using namespace vane_fs;
using Clock = std::chrono::steady_clock;

namespace {
using Samples = std::map<std::string, std::vector<double>>;
template <class F>
double Measure(F operation) {
	auto start = Clock::now();
	operation();
	return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void Require(bool condition, const std::string &message) {
	if (!condition)
		throw std::runtime_error(message);
}
int64_t Scalar(sqlite3 *db, const std::string &sql) {
	sqlite3_stmt *statement = nullptr;
	Require(sqlite3_prepare_v2(db, sql.c_str(), -1, &statement, nullptr) == SQLITE_OK, sqlite3_errmsg(db));
	auto code = sqlite3_step(statement);
	int64_t value = sqlite3_column_int64(statement, 0);
	sqlite3_finalize(statement);
	Require(code == SQLITE_ROW, sqlite3_errmsg(db));
	return value;
}
std::map<std::string, int64_t> Statistics(const std::string &path) {
	sqlite3 *db = nullptr;
	Require(sqlite3_open_v2(path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) == SQLITE_OK,
	        "Opening benchmark statistics");
	std::map<std::string, int64_t> result;
	try {
		for (const auto *table :
		     {"inode_versions", "dirent_versions", "block_versions", "block_payloads", "snapshots", "branches"}) {
			result[table] = Scalar(db, "SELECT count(*) FROM " + std::string(table));
		}
		result["page_count"] = Scalar(db, "PRAGMA page_count");
		result["free_pages"] = Scalar(db, "PRAGMA freelist_count");
		for (const auto &suffix : {std::string(), std::string("-wal")}) {
			std::error_code error;
			auto size = std::filesystem::file_size(path + suffix, error);
			result[suffix.empty() ? "database_bytes" : "wal_bytes"] = error ? 0 : int64_t(size);
		}
		sqlite3_close(db);
		return result;
	} catch (...) {
		sqlite3_close(db);
		throw;
	}
}
void PrintValues(const std::map<std::string, int64_t> &values) {
	std::cout << '{';
	bool first = true;
	for (const auto &value : values) {
		if (!first)
			std::cout << ',';
		std::cout << std::quoted(value.first) << ':' << value.second;
		first = false;
	}
	std::cout << '}';
}
} // namespace

int main(int argc, char **argv) {
	if (argc != 7) {
		std::cerr << "Usage: vane-fs-benchmark NEW_DATABASE FILES LARGE_MIB FORKS DEPTH ITERATIONS\n";
		return 2;
	}
	try {
		std::string path = argv[1];
		int files = std::stoi(argv[2]), large_mib = std::stoi(argv[3]), forks = std::stoi(argv[4]);
		int depth = std::stoi(argv[5]), iterations = std::stoi(argv[6]);
		Require(files > 0 && files <= 1000000 && large_mib > 0 && large_mib <= 1024 && forks > 0 && forks <= 500 &&
		            depth > 0 && depth <= 64 && iterations > 0 && iterations <= 100000,
		        "Invalid benchmark dimensions");
		Require(!std::filesystem::exists(path), "Benchmark requires a new database");
		Samples timings;
		std::map<std::string, std::map<std::string, int64_t>> observations;
		Workspace workspace(path);
		auto main = workspace.Checkout();
		main->MakeDirectory("/small");
		std::string small(4096, 's'), large(size_t(large_mib) * 1024 * 1024, 'L');
		for (int i = 0; i < files; ++i) {
			timings["small_file_create"].push_back(
			    Measure([&] { main->WriteFile("/small/" + std::to_string(i), small); }));
		}
		double write_ms = Measure([&] { main->WriteFile("/large", large); });
		timings["sequential_write"].push_back(write_ms);
		for (int i = 0; i < 5; ++i) {
			std::string bytes;
			timings["sequential_read"].push_back(Measure([&] { bytes = main->Read("/large"); }));
			Require(bytes == large, "Sequential read validation");
		}
		observations["populated"] = Statistics(path);
		std::vector<std::string> children;
		for (int i = 0; i < forks; ++i) {
			BranchInfo child;
			timings["fork_flat"].push_back(
			    Measure([&] { child = workspace.Fork("main", "flat-" + std::to_string(i)); }));
			children.push_back(child.id);
		}
		observations["after_forks"] = Statistics(path);
		for (const auto *table : {"inode_versions", "dirent_versions", "block_versions", "block_payloads"}) {
			Require(observations["after_forks"][table] == observations["populated"][table], "Fork copied file rows");
		}
		for (const auto &child : children) {
			auto session = workspace.Checkout(child);
			timings["first_write_after_fork"].push_back(Measure([&] { session->Write("/small/0", "changed", 10); }));
			Require(session->Read("/small/0") == small.substr(0, 10) + "changed" + small.substr(17),
			        "Child write validation");
		}
		Require(main->Read("/small/0") == small, "Parent isolation validation");
		observations["after_first_writes"] = Statistics(path);
		auto parent = workspace.GetBranch().id;
		std::string subtree;
		for (int i = 0; i < depth; ++i) {
			BranchInfo child;
			timings["fork_nested"].push_back(
			    Measure([&] { child = workspace.Fork(parent, "depth-" + std::to_string(i)); }));
			if (i == 0)
				subtree = child.id;
			parent = child.id;
			Require(workspace.Checkout(parent)->Read("/small/0") == small, "Nested branch isolation");
		}
		for (int i = 0; i < iterations; ++i) {
			timings["same_block_write"].push_back(
			    Measure([&] { main->Write("/large", std::string(4096, char('a' + i % 26)), 4096); }));
			timings["directory_lookup"].push_back(Measure([&] { main->Stat("/small/" + std::to_string(i % files)); }));
		}
		large.replace(4096, 4096, std::string(4096, char('a' + (iterations - 1) % 26)));
		Require(main->Read("/large") == large, "Repeated block-write validation");
		main->WriteFile("/parallel", std::string(4 * 4096, '\0'));
		double parallel_ms = Measure([&] {
			std::vector<std::future<std::vector<double>>> writers;
			for (int writer = 0; writer < 4; ++writer) {
				writers.push_back(std::async(std::launch::async, [&, writer] {
					Workspace connection(path, 30000);
					auto session = connection.Checkout();
					std::vector<double> samples;
					for (int i = 0; i < iterations; ++i)
						samples.push_back(Measure([&] {
							session->Write("/parallel", std::string(4096, char('A' + (writer + i) % 26)),
							               writer * 4096);
						}));
					return samples;
				}));
			}
			for (auto &writer : writers) {
				auto samples = writer.get();
				timings["contended_write"].insert(timings["contended_write"].end(), samples.begin(), samples.end());
			}
		});
		for (int writer = 0; writer < 4; ++writer)
			Require(main->Read("/parallel", writer * 4096, 4096) ==
			            std::string(4096, char('A' + (writer + iterations - 1) % 26)),
			        "Concurrent writer validation");
		observations["before_gc"] = Statistics(path);
		for (const auto &child : children)
			workspace.DeleteBranch(child);
		workspace.DeleteBranch(subtree, true);
		CollectionResult collected;
		timings["gc"].push_back(Measure([&] { collected = workspace.CollectGarbage(); }));
		observations["after_gc"] = Statistics(path);
		Require(main->Read("/large") == large && main->Read("/small/0") == small, "GC content validation");
		std::cout << std::setprecision(10) << "{\"sqlite\":" << std::quoted(Workspace::SQLiteVersion())
		          << ",\"compiler\":" << std::quoted(VANE_FS_COMPILER)
		          << ",\"build_type\":" << std::quoted(VANE_FS_BUILD_TYPE) << ",\"validated\":true,\"config\":";
		PrintValues({{"files", files},
		             {"large_mib", large_mib},
		             {"forks", forks},
		             {"depth", depth},
		             {"iterations", iterations},
		             {"writers", 4}});
		std::cout << ",\"timings_ms\":{";
		bool first = true;
		for (auto entry : timings) {
			if (!first)
				std::cout << ',';
			first = false;
			auto values = entry.second;
			std::sort(values.begin(), values.end());
			auto percentile = [&](size_t n) {
				return values[(values.size() - 1) * n / 100];
			};
			std::cout << std::quoted(entry.first) << ":{\"count\":" << values.size() << ",\"p50\":" << percentile(50)
			          << ",\"p95\":" << percentile(95)
			          << ",\"total\":" << std::accumulate(values.begin(), values.end(), 0.0) << ",\"samples\":[";
			for (size_t i = 0; i < entry.second.size(); ++i) {
				if (i)
					std::cout << ',';
				std::cout << entry.second[i];
			}
			std::cout << "]}";
		}
		std::cout << "},\"sequential_write_mib_s\":" << large_mib * 1000.0 / write_ms
		          << ",\"parallel_total_ms\":" << parallel_ms << ",\"observations\":{";
		first = true;
		for (const auto &entry : observations) {
			if (!first)
				std::cout << ',';
			first = false;
			std::cout << std::quoted(entry.first) << ':';
			PrintValues(entry.second);
		}
		std::cout << "},\"collected\":";
		PrintValues(
		    {{"versions", collected.versions}, {"payloads", collected.payloads}, {"snapshots", collected.snapshots}});
		std::cout << "}\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
