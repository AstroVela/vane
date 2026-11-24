// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Reuse the frozen descriptor/error model and lease-aware blocking wrapper.
#define main             unchanged_fault_main
#define __wrap_fdatasync inherited_fdatasync
#include "../payload_range_writeback/test_prewriteback.cpp"
#undef __wrap_fdatasync
#undef main

static thread_local bool testing_barrier = false;
static std::atomic<int> wanted {0};
static std::atomic<bool> barrier_hold {false}, barrier_entered {false}, barrier_failure {false};
static bool Gate(int fd) {
	if (!testing_barrier)
		return false;
	auto name = Name(fd);
	int kind = name.size() >= 4 && name.compare(name.size() - 4, 4, "-wal") == 0              ? 2
	           : name.size() >= 12 && name.compare(name.size() - 12, 12, ".stage-fence") == 0 ? 3
	           : Payload(fd)                                                                  ? 1
	                                                                                          : 0;
	if (!kind || kind != wanted)
		return false;
	barrier_entered = true;
	while (barrier_hold)
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	return barrier_failure.exchange(false);
}
extern "C" int __real_fsync(int);
extern "C" int __wrap_fsync(int fd) {
	if (Gate(fd)) {
		errno = EIO;
		return -1;
	}
	return __real_fsync(fd);
}
extern "C" int __wrap_fdatasync(int fd) {
	if (Gate(fd)) {
		errno = EIO;
		return -1;
	}
	return inherited_fdatasync(fd);
}
struct Join {
	std::thread thread;
	~Join() {
		hold = barrier_hold = false;
		if (thread.joinable())
			thread.join();
	}
};
static void Full(const std::string &path, bool publication, int kind, bool failing) {
	Workspace w(path, 5000, Durability::Fsync);
	auto session = w.Checkout();
	Staging observer(path, 5000);
	observer.Initialize(w.Id(), false);
	observer.Ready();
	std::unique_ptr<Prewriteback> external;
	if (publication) {
		external = std::make_unique<Prewriteback>(path, 5000);
		observer.AttachPrewriteback(external.get());
	}
	entered = barrier_entered = false;
	hold = barrier_hold = true;
	barrier_failure = failing;
	wanted = kind;
	Resume resume;
	session->WriteFile("/file", std::string((publication ? 63 : 64) * MIB, 'p'));
	if (publication)
		external->OnCommit(64 * MIB, false, {4096, 63 * MIB + 4096});
	Wait([] { return entered.load(); });
	std::exception_ptr error;
	Join joining;
	joining.thread = std::thread([&] {
		application = true;
		try {
			if (publication) {
				sqlite3 *raw = nullptr;
				Require(sqlite3_open(path.c_str(), &raw) == SQLITE_OK, "Opening publication pager");
				std::unique_ptr<sqlite3, decltype(&sqlite3_close_v2)> db(raw, sqlite3_close_v2);
				Staging::Configure(raw);
				Staging::Guard guard(observer);
				Staging::CheckpointGuard checkpoint(observer);
				// Guard::Load may adopt an unfamiliar OPEN record first. Gate
				// the actual Publish sequence, including its later CLEAN sync.
				testing_barrier = true;
				observer.Publish(raw);
			} else {
				testing_barrier = true;
				w.Sync();
			}
		} catch (...) {
			error = std::current_exception();
		}
	});
	Wait([] { return barrier_entered.load(); });
	hold = false;
	Wait([&] { return !Calls(path).empty(); });
	std::this_thread::sleep_for(std::chrono::milliseconds(60));
	Require(Calls(path).size() == 1, "A new claim started during a real FULL barrier");
	barrier_hold = false;
	joining.thread.join();
	wanted = 0;
	if (failing) {
		Require(bool(error), "Injected FULL failure was swallowed");
		try {
			std::rethrow_exception(error);
		} catch (const Error &failure) {
			Require(failure.code == ErrorCode::Storage, failure.what());
		}
		Wait([&] { return Calls(path).size() > 1; });
	} else {
		if (error)
			std::rethrow_exception(error);
		std::this_thread::sleep_for(std::chrono::milliseconds(40));
		Require(Calls(path).size() == 1, "Successful FULL did not retire unsent requests");
	}
	if (external) {
		external->Stop();
		observer.AttachPrewriteback(nullptr);
	}
	w.Sync();
	w.Close();
	Workspace reopened(path);
	Require(reopened.Checkout()->Read("/file", 62 * MIB) == std::string((publication ? 1 : 2) * MIB, 'p'),
	        "FULL pause lost data");
	std::cout << "PASS " << (publication ? "background publication" : "foreground FULL") << " barrier " << kind
	          << (failing ? " error/unwind" : " success/retire") << '\n';
}
int main(int argc, char **argv) {
	if (argc != 2)
		return 2;
	application = true;
	try {
		auto root = fs::path(argv[1]);
		for (auto publication : {false, true}) {
			for (int kind = 1; kind <= 3; ++kind)
				Full((root / ((publication ? "publish-" : "full-") + std::to_string(kind) + ".sqlite")).string(),
				     publication, kind, false);
			Full((root / (publication ? "publish-error.sqlite" : "full-error.sqlite")).string(), publication, 1, true);
		}
		Require(unexpected_sync == 0 && bad_range == 0, "Invalid background syscall");
	} catch (const std::exception &error) {
		hold = barrier_hold = false;
		std::cerr << error.what() << '\n';
		return 1;
	}
}
