// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#include "../checkpoint_progress/fixture.hpp"
#include <costs.hpp>
#include <array>
#include <fstream>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

namespace {
int64_t Clock(int kind) {
	timespec value {};
	Require(!clock_gettime(kind, &value), "Read measurement clock");
	return int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
}
struct Stamp {
	int64_t wall, thread, process;
	static Stamp Read() {
		return {Clock(CLOCK_MONOTONIC), Clock(CLOCK_THREAD_CPUTIME_ID), Clock(CLOCK_PROCESS_CPUTIME_ID)};
	}
};
struct Phase {
	const char *name;
	Stamp begin {}, end {};
	write_costs::Counters counters {};
	template <class F>
	void Run(int index, const char *label, F operation) {
		name = label;
		begin = Stamp::Read();
#ifdef VANE_WRITE_PROFILE
		write_costs::Begin(index);
		write_costs::Scope root(write_costs::Root);
#else
		(void)index;
#endif
		operation();
#ifdef VANE_WRITE_PROFILE
		root.Finish();
		counters = write_costs::End();
#endif
		end = Stamp::Read();
	}
	void Print(std::ostream &output) const {
		output << "{\"name\":\"" << name << "\",\"begin_ns\":" << begin.wall << ",\"end_ns\":" << end.wall
		       << ",\"wall_ns\":" << end.wall - begin.wall << ",\"thread_cpu_ns\":" << end.thread - begin.thread
		       << ",\"process_cpu_ns\":" << end.process - begin.process << ",\"stages\":";
#ifdef VANE_WRITE_PROFILE
		write_costs::Print(output, counters);
#else
		output << "{}";
#endif
		output << '}';
	}
};
sqlite3 *writer = nullptr;
using WalHook = int (*)(void *, sqlite3 *, const char *, int);
} // namespace
extern "C" void *__real_sqlite3_wal_hook(sqlite3 *, WalHook, void *);
extern "C" void *__wrap_sqlite3_wal_hook(sqlite3 *db, WalHook hook, void *context) {
	if (hook == Checkpointer::OnCommit) {
		Require(writer == nullptr, "Expected one foreground writer");
		writer = db;
	}
	if (!hook && db == writer)
		writer = nullptr;
	return __real_sqlite3_wal_hook(db, hook, context);
}
static std::pair<int, int> Status() {
	int frames = -1, copied = -1;
	auto code = sqlite3_wal_checkpoint_v2(writer, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied);
	Require(code == SQLITE_OK || code == SQLITE_BUSY, "Read WAL status");
	return {frames, copied};
}
int main(int argc, char **argv) {
	try {
		Require(argc == 4, "Expected database, MiB, result JSON");
		std::string path = argv[1];
		int mib = std::stoi(argv[2]);
		Require(mib == 4 || mib == 64 || mib == 256, "Expected 4, 64 or 256 MiB");
		std::string payload(1048576, 'w');
		std::vector<std::array<int64_t, 4>> requests;
		requests.reserve(size_t(mib));
		std::array<Phase, 6> phases;
		std::unique_ptr<Workspace> workspace;
		std::shared_ptr<Session> session;
		int64_t root = -1, inode = -1;
		phases[0].Run(0, "initialization", [&] {
			workspace = std::make_unique<Workspace>(path, 5000, Durability::Fsync);
			session = workspace->Checkout();
			workspace->AcquireMount("main");
			root = session->OpenDirectory("/").inode;
		});
		Require(writer != nullptr, "Missing original WAL hook");
		auto before = Status();
		phases[1].Run(1, "create",
		              [&] { inode = session->CreateNode(root, "file", false, 0600, true, false, 2).inode; });
		phases[2].Run(2, "writes", [&] {
			for (int i = 0; i < mib; ++i) {
				auto begin = Stamp::Read();
				session->WriteInode(inode, payload, int64_t(i) * 1048576);
				auto end = Stamp::Read();
				requests.push_back({begin.wall, end.wall, end.thread - begin.thread, end.process - begin.process});
			}
		});
		phases[3].Run(3, "barrier", [&] {
			session->StatInode(inode);
			workspace->Sync();
		});
		phases[4].Run(4, "release", [&] {
			session->StatInode(inode);
			session->CloseFile(inode);
		});
		auto after = Status();
		auto wal_bytes = std::filesystem::file_size(path + "-wal");
		{
#ifdef VANE_WRITE_PROFILE
			write_costs::Phase(6);
#endif
			Require(session->Read("/file") == std::string(size_t(mib) * 1048576, 'w'), "Content differs");
			session->CloseFile(inode);
			session->CloseFile(root);
			workspace->ReleaseMount("main");
			session->Close();
		}
		phases[5].Run(5, "close", [&] { workspace->Close(); });
		{
			SQL inspect(path);
			Require(inspect.Text("PRAGMA quick_check") == "ok", "Integrity check failed");
			Require(inspect.Text("SELECT version FROM format") == "1004", "Storage format changed");
			Require(inspect.Text("SELECT count(*) FROM block_payloads WHERE length(data)=8") ==
			            std::to_string(int64_t(mib) * 256),
			        "External payload density changed");
			Require(inspect.Text("SELECT count(*) FROM block_payloads WHERE length(data)!=8") == "0",
			        "Unexpected inline payloads");
		}
		std::ofstream output(argv[3]);
		output << "{\"status\":\"PASS\",\"mib\":" << mib << ",\"main_tid\":" << syscall(SYS_gettid)
		       << ",\"content_and_integrity\":true,\"total_ns\":" << phases[4].end.wall - phases[0].begin.wall
		       << ",\"total_thread_cpu_ns\":" << phases[4].end.thread - phases[0].begin.thread
		       << ",\"total_process_cpu_ns\":" << phases[4].end.process - phases[0].begin.process << ",\"wal_before\":["
		       << before.first << ',' << before.second << "],\"wal_after\":[" << after.first << ',' << after.second
		       << "],\"wal_bytes\":" << wal_bytes << ",\"phases\":[";
		for (size_t i = 0; i < phases.size(); ++i) {
			output << (i ? "," : "");
			phases[i].Print(output);
		}
		output << "],\"request_columns\":[\"begin_ns\",\"end_ns\",\"thread_cpu_ns\",\"process_cpu_ns\"],\"requests\":[";
		for (size_t i = 0; i < requests.size(); ++i) {
			auto &r = requests[i];
			output << (i ? "," : "") << '[' << r[0] << ',' << r[1] << ',' << r[2] << ',' << r[3] << ']';
		}
		output << "],\"counter_columns\":[\"calls\",\"inclusive_wall_ns\",\"exclusive_wall_ns\",\"inclusive_cpu_ns\","
		          "\"exclusive_cpu_ns\",\"cpp_allocations\",\"cpp_allocated_bytes\",\"bytes\",\"vm_steps\"],"
		          "\"event_columns\":[\"kind\",\"file\",\"phase\",\"tid\",\"begin_ns\",\"end_ns\",\"cpu_ns\",\"code\","
		          "\"errno\",\"mode\",\"frames\",\"copied\"],\"events\":";
#ifdef VANE_WRITE_PROFILE
		write_costs::PrintEvents(output);
#else
		output << "[]";
#endif
		output << "}\n";
		Require(bool(output), "Write result JSON");
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
