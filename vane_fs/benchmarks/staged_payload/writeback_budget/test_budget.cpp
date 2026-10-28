// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Fault/control executable only. Plain performance runs never use wrappers.
#include "vane_fs/workspace.hpp"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <future>
#include <iostream>
#include <mutex>
#include <thread>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace vane_fs;
using namespace std::chrono_literals;
static std::atomic<int> calls {0}, finished {0}, watched {-1}, fault {0};
static std::atomic<int> payload_syncs {0}, wal_syncs {0};
static std::atomic<bool> watched_closed {false}, fail_barrier {false}, reclamation {false};
static std::atomic<int64_t> largest_range {0}, submitted_bytes {0}, completed_bytes {0}, last_end_ns {0};
static std::atomic<bool> pace_violation {false}, barrier_entered {false};
static bool retry_interrupt = false;
static std::atomic<bool> block_payload_barrier {false}, block_wal_barrier {false}, block_fence_barrier {false},
    fail_fence {false};
static bool barrier_released = false;
static std::mutex gate_mutex;
static std::condition_variable gate;
static bool block_hint = false, released = false;
static pid_t main_thread;

static void Require(bool value, const char *message) {
	if (!value)
		throw std::runtime_error(message);
}
static std::string Name(int fd) {
	char text[4096];
	auto link = "/proc/self/fd/" + std::to_string(fd);
	auto n = readlink(link.c_str(), text, sizeof(text));
	return n > 0 ? std::string(text, size_t(n)) : "";
}
static bool Ends(const std::string &name, const char *suffix) {
	auto n = std::strlen(suffix);
	return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
}
static void Release() {
	std::lock_guard<std::mutex> lock(gate_mutex);
	released = true;
	gate.notify_all();
}
static void ReleaseBarrier() {
	std::lock_guard<std::mutex> lock(gate_mutex);
	barrier_released = true;
	gate.notify_all();
}
struct ReleaseOnExit {
	~ReleaseOnExit() {
		Release();
		ReleaseBarrier();
	}
};
template <class F>
static void Until(F predicate, const char *message) {
	auto end = std::chrono::steady_clock::now() + 5s;
	while (!predicate()) {
		Require(std::chrono::steady_clock::now() < end, message);
		std::this_thread::sleep_for(1ms);
	}
}
struct Task {
	std::future<void> result;
	template <class F>
	explicit Task(F action) : result(std::async(std::launch::async, action)) {
	}
	~Task() {
		Release();
		ReleaseBarrier();
		if (result.valid())
			result.wait();
	}
	void Complete() {
		Require(result.wait_for(5s) == std::future_status::ready, "Foreground blocked behind hint submission");
		result.get();
	}
};
extern "C" int __wrap_sync_file_range(int fd, off64_t offset, off64_t bytes, unsigned flags) {
	Require(gettid() != main_thread, "Hint submitted on calling thread");
	Require(Ends(Name(fd), ".payload"), "Hint must target payload");
	Require(flags == SYNC_FILE_RANGE_WRITE, "Hint must never add WAIT flags");
	Require(offset >= 4096 && offset % 4096 == 0 && bytes > 0 && bytes % 4096 == 0, "Invalid hint range");
	Require(watched < 0 || watched == fd, "Hint changed its error-observation description");
	Require(!watched_closed, "Hint descriptor closed while active");
	watched = fd;
	largest_range = std::max(largest_range.load(), int64_t(bytes));
	auto now = std::chrono::steady_clock::now().time_since_epoch();
	auto previous = last_end_ns.load();
	if (!retry_interrupt && previous &&
	    std::chrono::duration_cast<std::chrono::nanoseconds>(now).count() - previous < 16000000)
		pace_violation = true;
	submitted_bytes += bytes;
	++calls;
	{
		std::unique_lock<std::mutex> lock(gate_mutex);
		gate.wait(lock, [] { return !block_hint || released; });
	}
	auto code = fault.exchange(0);
	last_end_ns =
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
	        .count();
	completed_bytes += bytes;
	++finished;
	if (code) {
		errno = code;
		return -1;
	}
	return 0; // Hint model only: the real final data/WAL barriers still run.
}
extern "C" int __real_fdatasync(int);
extern "C" int __wrap_fdatasync(int fd) {
	bool payload = watched >= 0 && Ends(Name(fd), ".payload");
	if (payload) {
		if (fd != watched) {
			Require(reclamation && payload_syncs > 0 && wal_syncs > 0, "Reopened descriptor before publication");
			return __real_fdatasync(fd); // GC separately flushes hole punching/truncation after FULL deletion.
		}
		Require(fd == watched && !watched_closed, "Final barrier reopened the hint description");
		if (block_payload_barrier) {
			barrier_entered = true;
			std::unique_lock<std::mutex> lock(gate_mutex);
			gate.wait(lock, [] { return barrier_released; });
		}
		if (fail_barrier.exchange(false)) {
			errno = EIO;
			return -1;
		}
	}
	if (block_fence_barrier && Ends(Name(fd), ".stage-fence")) {
		barrier_entered = true;
		std::unique_lock<std::mutex> lock(gate_mutex);
		gate.wait(lock, [] { return barrier_released; });
		if (fail_fence.exchange(false)) {
			errno = EIO;
			return -1;
		}
	}
	auto rc = __real_fdatasync(fd);
	if (payload && !rc)
		++payload_syncs;
	return rc;
}
extern "C" int __real_fsync(int);
extern "C" int __wrap_fsync(int fd) {
	if (watched >= 0 && Ends(Name(fd), "-wal")) {
		Require(payload_syncs > 0, "WAL barrier must follow successful payload barrier");
		if (block_wal_barrier) {
			barrier_entered = true;
			std::unique_lock<std::mutex> lock(gate_mutex);
			gate.wait(lock, [] { return barrier_released; });
		}
		++wal_syncs;
	}
	return __real_fsync(fd);
}
extern "C" int __real_close(int);
extern "C" int __wrap_close(int fd) {
	if (watched >= 0 && fd == watched)
		watched_closed = true;
	return __real_close(fd);
}
static void StorageBarrier(Workspace &ws) {
	bool observed = false;
	Until(
	    [&] {
		    try {
			    ws.Sync();
		    } catch (const Error &e) {
			    Require(e.code == ErrorCode::Storage, "Wrong asynchronous error");
			    observed = true;
		    }
		    return observed;
	    },
	    "Submission error was lost");
}
int main(int argc, char **argv) {
	try {
		Require(argc == 3, "Expected fresh database and case");
		main_thread = gettid();
		std::string mode(argv[2]);
		fault = mode == "unsupported"                                                         ? ENOSYS
		        : mode == "invalid"                                                           ? EINVAL
		        : mode == "not_supported"                                                     ? EOPNOTSUPP
		        : mode == "interrupt"                                                         ? EINTR
		        : mode == "io_error" || mode == "close_error" || mode == "post_barrier_error" ? EIO
		        : mode == "no_space"                                                          ? ENOSPC
		                                                                                      : 0;
		block_hint = mode != "small";
		retry_interrupt = mode == "interrupt";
		std::string bytes(32 * 1024 * 1024, 'a'), expected = "s" + bytes;
		if (mode == "crash") {
			auto child = fork();
			Require(child >= 0, "fork failed");
			if (!child) {
				try {
					main_thread = gettid();
					Workspace ws(argv[1], 5000, Durability::Fsync);
					auto session = ws.Checkout();
					session->WriteFile("/file", "stable");
					ws.Snapshot();
					session->Write("/file", bytes, 1);
					Until([] { return calls > 0; }, "Expected active crash hint");
					Require(finished == 0 && largest_range <= 4 * 1024 * 1024 && session->Read("/file") == expected,
					        "Crash setup");
					_exit(0); // Process crash: no Close, worker join, or final payload/WAL barrier.
				} catch (...) {
					_exit(1);
				}
			}
			int status = -1;
			try {
				Until([&] { return waitpid(child, &status, WNOHANG) == child; }, "Crash child blocked");
			} catch (...) {
				kill(child, SIGKILL);
				waitpid(child, &status, 0);
				throw;
			}
			Require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "Crash child failed");
			int fd = open((std::string(argv[1]) + ".stage-fence").c_str(), O_RDONLY);
			Require(fd >= 0, "Crash fence missing");
			uint64_t newest = 0;
			bool pending = false;
			for (int slot = 0; slot < 2; ++slot) {
				unsigned char fields[16];
				Require(pread(fd, fields, 16, slot * 4096 + 32) == 16, "Crash fence truncated");
				uint64_t sequence = 0;
				for (int i = 0; i < 8; ++i)
					sequence = (sequence << 8) | fields[i];
				if (sequence > newest) {
					newest = sequence;
					pending = fields[15] != 0;
				}
			}
			close(fd);
			Require(newest && pending, "Hint incorrectly published CLEAN");
			for (int i = 0; i < 3; ++i) {
				Workspace recovered(argv[1]);
				Require(recovered.Checkout()->Read("/file") == "stable", "Unacknowledged bytes survived cold recovery");
				recovered.CollectGarbage();
				recovered.Close();
			}
			std::cout << "{\"case\":\"crash\",\"open_fence_verified\":true,\"recovery_and_gc_verified\":true}\n";
			return 0;
		}
		{
			Workspace ws(argv[1], 5000, mode == "strict" ? Durability::Strict : Durability::Fsync);
			ReleaseOnExit unblock;
			auto session = ws.Checkout();
			session->WriteFile("/file", "stable");
			auto saved = ws.Snapshot();
			if (mode == "small") {
				session->WriteFile("/small", std::string(1024 * 1024, 's'));
				Require(calls == 0, "Small writes submitted hints");
				expected = "stable";
			} else {
				Task write([&] { session->Write("/file", bytes, 1); });
				Until([] { return calls > 0; }, "Expected background hint");
				write.Complete();
				Require(finished == 0, "Blocked hint returned before foreground completion");
				Require(session->Read("/file") == expected, "Visible content while hint blocked");
				if (mode == "coalesce") {
					session->WriteFile("/extra", "e");
					session->Write("/extra", bytes, 1);
					Require(calls == 1, "Parallel or unbounded hint queue");
					struct stat state {};
					Require(fstat(watched, &state) == 0, "Payload size missing");
					auto expected_bytes = state.st_size - 4096;
					Release();
					Until([&] { return completed_bytes >= expected_bytes; }, "Coalesced range did not drain");
					Require(submitted_bytes == expected_bytes && calls >= 15,
					        "Budget lost or duplicated pending bytes");
				} else if (mode == "normal_snapshot") {
					ws.Snapshot();
					Require(finished == 0, "NORMAL snapshot waited for optional hint");
					Release();
					Until([] { return calls >= 2; }, "NORMAL snapshot discarded or paused pending hints");
				} else if (mode == "pause_full" || mode == "pause_error" || mode == "pause_wal" ||
				           mode == "pause_fence" || mode == "fence_error") {
					block_payload_barrier = mode == "pause_full" || mode == "pause_error";
					block_fence_barrier = mode == "pause_fence" || mode == "fence_error";
					fail_fence = mode == "fence_error";
					block_wal_barrier = mode == "pause_wal";
					fail_barrier = mode == "pause_error";
					Task sync([&] {
						try {
							ws.Sync();
						} catch (const Error &e) {
							Require((mode == "pause_error" || mode == "fence_error") && e.code == ErrorCode::Storage,
							        "Unexpected paused barrier error");
							return;
						}
						Require(mode != "pause_error" && mode != "fence_error", "Barrier failure hidden");
					});
					Until([] { return barrier_entered.load(); }, "FULL barrier not entered");
					Release();
					Until([] { return finished >= 1; }, "Active hint not released during FULL barrier");
					std::this_thread::sleep_for(80ms);
					Require(calls == 1, "FULL publication claimed another optional hint");
					ReleaseBarrier();
					sync.Complete();
					if (mode == "pause_error" || mode == "fence_error") {
						Until([] { return calls >= 2; }, "Failed FULL barrier did not resume pending hints");
					} else {
						std::this_thread::sleep_for(40ms);
						Require(calls == 1, "Successful FULL barrier did not discard redundant hints");
					}
					block_payload_barrier = block_wal_barrier = block_fence_barrier = false;
				} else if (mode == "fork") {
					// A child can durably clean the shared fence using its own fd.
					// Establish the parent's observed barrier before testing abandonment.
					ws.Sync();
					auto child = fork();
					Require(child >= 0, "fork failed");
					if (!child) {
						try {
							ws.Close(); // Never join the vanished parent's thread or reuse its mutex.
							watched = -1;
							watched_closed = false;
							Workspace reopened(argv[1], 5000, Durability::Fsync);
							Require(reopened.Checkout()->Read("/file") == expected, "Child content");
							reopened.Close();
							_exit(0);
						} catch (...) {
							_exit(1);
						}
					}
					int status = -1;
					try {
						Until([&] { return waitpid(child, &status, WNOHANG) == child; },
						      "Child waited on parent thread");
					} catch (...) {
						kill(child, SIGKILL);
						waitpid(child, &status, 0);
						throw;
					}
					Require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "Child close/reopen failed");
				} else if (mode == "gc") {
					ws.Sync();
					ws.DropSnapshot(saved);
					session->Unlink("/file");
					reclamation = true;
					ws.CollectGarbage();
					reclamation = false;
					Require(finished == 0, "GC waited on hint submission");
					session->WriteFile("/file", "s");
					session->Write("/file", bytes, 1);
					saved = ws.Snapshot();
				} else if (mode == "close_blocked" || mode == "close_error") {
					std::atomic<bool> started {false};
					Task close([&] {
						started = true;
						if (mode == "close_error") {
							try {
								ws.Close();
							} catch (const Error &e) {
								Require(e.code == ErrorCode::Storage, "Close error type");
								return;
							}
							throw std::runtime_error("Close hid hint error");
						}
						ws.Close();
					});
					Until([&] { return started.load(); }, "Close task missing");
					Require(close.result.wait_for(20ms) == std::future_status::timeout && !watched_closed,
					        "Close did not retain active descriptor");
					Release();
					close.Complete();
					if (mode == "close_error") {
						ws.Sync(); // A failed close leaves the workspace usable for retry.
						session->WriteFile("/extra", "e");
						session->Write("/extra", bytes, 1);
						Until([] { return calls >= 2; }, "Worker did not restart after failed close");
					}
				} else if (mode == "blocked" || mode == "post_barrier_error") {
					ws.Sync();
					Require(finished == 0 && payload_syncs > 0 && wal_syncs > 0,
					        "Final durability depended on hint completion");
				}
				Release();
				Until([] { return finished >= 1; }, "Hint did not finish");
				if (mode == "io_error" || mode == "no_space" || mode == "post_barrier_error") {
					StorageBarrier(ws);
					Require(session->Read("/file") == expected, "Error changed prior visible NORMAL commit");
				}
				if (mode == "interrupt")
					Until([] { return calls >= 2 && finished >= 2; }, "EINTR was not retried");
				if (mode == "barrier_error") {
					fail_barrier = true;
					StorageBarrier(ws);
				}
			}
			if (mode != "close_blocked") {
				ws.Sync();
				auto frozen = ws.OpenSnapshot(saved);
				Require(frozen->Read("/file") == (mode == "gc" ? expected : "stable"), "Snapshot content");
				frozen->Close();
				if (mode == "unsupported" || mode == "invalid" || mode == "not_supported") {
					session->WriteFile("/extra", "e");
					session->Write("/extra", bytes, 1);
				}
				session->Close();
				ws.Close();
			}
			if (mode != "small")
				Require(payload_syncs >= 1 && wal_syncs >= 1 && !watched_closed,
				        "Durability ordering or descriptor lifetime");
			if (mode == "unsupported" || mode == "invalid" || mode == "not_supported")
				Require(calls == 1, "Unsupported hint not disabled");
		}
		Require(largest_range <= 4 * 1024 * 1024 && !pace_violation, "Submission budget or pacing violated");
		Require(watched < 0 || watched_closed, "Descriptor leak");
		watched = -1;
		Workspace reopened(argv[1]);
		Require(reopened.Checkout()->Read("/file") == expected, "Reopen content");
		reopened.Close();
		std::cout << "{\"case\":\"" << mode << "\",\"hint_calls\":" << calls << ",\"content_verified\":true}\n";
	} catch (const std::exception &e) {
		Release();
		ReleaseBarrier();
		std::cerr << e.what() << '\n';
		return 1;
	}
}
