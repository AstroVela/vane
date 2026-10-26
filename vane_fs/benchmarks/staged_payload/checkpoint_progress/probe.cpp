// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "fixture.hpp"
#include "trace.hpp"
#include <sys/syscall.h>
#include <unistd.h>
#include <iomanip>

struct Phase {
	const char *name;
	long long begin = 0, end = 0;
	double requests = 0, barrier = 0;
	std::vector<double> latencies;
};
static double Seconds(std::chrono::steady_clock::time_point start) {
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected database and state");
		std::string path = argv[1], state = argv[2];
		Require(state == "cold" || state == "reused" || state == "retained", "Unknown replay state");
		request_trace::Install(path.c_str());
		Workspace workspace(path, 5000, Durability::Fsync);
		auto session = workspace.Checkout();
		SQL writer(path), reader(path), progress(path);
		int warm_frames = 0;
		if (state != "cold") {
			request_trace::Scope prepare("prepare");
			writer.Exec("PRAGMA synchronous=NORMAL; CREATE TABLE padding(data BLOB); "
			            "INSERT INTO padding VALUES(zeroblob(20971520))");
			warm_frames = writer.Status().first;
			Require(warm_frames >= 4072, "Missing real large WAL generation");
			if (state == "retained")
				reader.Exec("BEGIN; SELECT count(*) FROM padding");
			session->WriteFile("/seed", "seed");
			if (state == "reused") {
				Wait([&] { return progress.CaughtUp(); }, "Preparation did not backfill");
				Wait(
				    [&] {
					    int frames = -1, copied = -1;
					    auto code =
					        sqlite3_wal_checkpoint_v2(writer.db, "main", SQLITE_CHECKPOINT_RESTART, &frames, &copied);
					    Require(code == SQLITE_OK || code == SQLITE_BUSY, "Reset preparation failed");
					    return code == SQLITE_OK && frames >= 0 && frames == copied;
				    },
				    "Preparation could not make WAL reusable");
			} else {
				Wait(
				    [&] {
					    auto status = progress.Status();
					    return status.second > 0 && status.first > status.second;
				    },
				    "Reader did not retain a partially backfilled generation");
			}
		}
		std::vector<Phase> phases;
		for (int workload = 0; workload < 3; ++workload) {
			const char *name = workload == 0 ? "mkdir" : workload == 1 ? "rename" : "rmdir";
			Phase phase;
			phase.name = name;
			phase.begin = request_trace::Now();
			auto start = std::chrono::steady_clock::now();
			for (int i = 0; i < 128; ++i) {
				auto begin = std::chrono::steady_clock::now();
				request_trace::Scope request(name);
				auto original = "/dir" + std::to_string(i), renamed = "/moved" + std::to_string(i);
				if (workload == 0)
					session->MakeDirectory(original);
				else if (workload == 1)
					session->Rename(original, renamed);
				else
					session->RemoveDirectory(renamed);
				phase.latencies.push_back(Seconds(begin));
			}
			phase.requests = Seconds(start);
			start = std::chrono::steady_clock::now();
			{
				request_trace::Scope barrier("fsync");
				workspace.Sync();
			}
			phase.barrier = Seconds(start);
			phase.end = request_trace::Now();
			phases.push_back(std::move(phase));
		}
		if (state == "retained")
			reader.Exec("ROLLBACK");
		Require(progress.Text("PRAGMA quick_check") == "ok", "SQLite integrity failure");
		Require(session->ListDirectory("/") ==
		            (state == "cold" ? std::vector<std::string>() : std::vector<std::string> {"seed"}),
		        "Namespace replay lost or retained directories");
		if (state != "cold")
			Require(session->Read("/seed") == "seed", "Acknowledged seed changed");
		auto final_status = progress.Status();
		workspace.Close();
		std::cout << std::setprecision(17) << "{\"status\":\"PASS\",\"state\":\"" << state
		          << "\",\"validation\":true,\"main_tid\":" << syscall(SYS_gettid) << ",\"warm_frames\":" << warm_frames
		          << ",\"final_frames\":" << final_status.first << ",\"phases\":[";
		for (size_t i = 0; i < phases.size(); ++i) {
			if (i)
				std::cout << ',';
			auto &phase = phases[i];
			std::cout << "{\"workload\":\"" << phase.name << "\",\"begin_ns\":" << phase.begin
			          << ",\"end_ns\":" << phase.end << ",\"requests_seconds\":" << phase.requests
			          << ",\"fsync_seconds\":" << phase.barrier << ",\"seconds\":" << phase.requests + phase.barrier
			          << ",\"latencies\":[";
			for (size_t j = 0; j < phase.latencies.size(); ++j) {
				if (j)
					std::cout << ',';
				std::cout << phase.latencies[j];
			}
			std::cout << "]}";
		}
		std::cout << "]}\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
