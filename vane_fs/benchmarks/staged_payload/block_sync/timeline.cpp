#include <sqlite3.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <cerrno>
#include <unistd.h>
#include <array>
#include <cstdlib>
#include <fcntl.h>
namespace timeline {
using Clock = std::chrono::steady_clock;
struct DiskStats {
	std::array<long long, 17> values {};
	long long begin = -1, end = -1;
	bool valid = false;
};
struct Row {
	std::string name, file;
	long long begin, end, offset;
	int rc, frames, copied;
	long long size_before, blocks_before, size_after, blocks_after;
	int error;
	DiskStats disk_before, disk_after;
};
struct Thread {
	long tid = 0;
	std::vector<Row> rows;
};
static Thread threads[8];
static std::atomic<int> next {0};
static thread_local int thread = -1;
static Clock::time_point origin;
static long long epoch;
static std::string output;
static int disk_fd = -1;
static long long Now() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - origin).count();
}
static DiskStats ReadDisk() {
	int saved = errno;
	DiskStats result;
	result.begin = Now();
	char buffer[2048];
	auto count = syscall(SYS_pread64, disk_fd, buffer, sizeof(buffer) - 1, 0);
	if (count > 0) {
		buffer[count] = 0;
		char *next_value = buffer;
		result.valid = true;
		for (auto &value : result.values) {
			char *end;
			value = std::strtoll(next_value, &end, 10);
			if (end == next_value) {
				result.valid = false;
				break;
			}
			next_value = end;
		}
	}
	result.end = Now();
	errno = saved;
	return result;
}
static std::string FileName(int fd) {
	char buffer[4096];
	auto path = "/proc/self/fd/" + std::to_string(fd);
	auto count = readlink(path.c_str(), buffer, sizeof(buffer));
	return count >= 0 ? std::string(buffer, count) : std::string();
}
static void Record(const char *name, long long begin, int rc, int fd = -1, int frames = -1, int copied = -1,
                   long long offset = -1, bool always = false, long long size_before = -1, long long blocks_before = -1,
                   const DiskStats *before = nullptr) {
	int saved = errno;
	auto end = Now();
	if (!always && end - begin < 1000000)
		return;
	DiskStats after;
	if (before)
		after = ReadDisk();
	struct stat state {};
	bool have = fd >= 0 && fstat(fd, &state) == 0;
	if (thread < 0) {
		thread = next.fetch_add(1);
		if (thread >= 8)
			std::abort();
		threads[thread].tid = syscall(SYS_gettid);
	}
	std::string file;
	if (fd >= 0) {
		char text[4096];
		auto link = "/proc/self/fd/" + std::to_string(fd);
		auto n = readlink(link.c_str(), text, sizeof(text));
		if (n >= 0)
			file.assign(text, n);
	}
	threads[thread].rows.push_back({name, std::move(file), begin, end, offset, rc, frames, copied, size_before,
	                                blocks_before, have ? state.st_size : -1, have ? state.st_blocks * 512 : -1,
	                                rc < 0 ? saved : 0, before ? *before : DiskStats {}, after});
	errno = saved;
}
static void PrintDisk(std::ostream &f, const DiskStats &stats) {
	f << "{\"valid\":" << (stats.valid ? "true" : "false") << ",\"begin_ns\":" << stats.begin
	  << ",\"end_ns\":" << stats.end << ",\"values\":[";
	for (size_t i = 0; i < stats.values.size(); ++i) {
		if (i)
			f << ',';
		f << stats.values[i];
	}
	f << "]}";
}
static void Report() {
	std::ofstream f(output);
	f << "{\"origin_epoch_ns\":" << epoch << ",\"threads\":[";
	for (int i = 0; i < next; ++i) {
		if (i)
			f << ',';
		f << "{\"tid\":" << threads[i].tid << ",\"events\":[";
		bool first = true;
		for (auto &r : threads[i].rows) {
			if (!first)
				f << ',';
			first = false;
			f << "{\"name\":" << std::quoted(r.name) << ",\"file\":" << std::quoted(r.file)
			  << ",\"begin_ns\":" << r.begin << ",\"end_ns\":" << r.end << ",\"offset\":" << r.offset
			  << ",\"rc\":" << r.rc << ",\"frames\":" << r.frames << ",\"copied\":" << r.copied
			  << ",\"size_before\":" << r.size_before << ",\"allocated_before\":" << r.blocks_before
			  << ",\"size_after\":" << r.size_after << ",\"allocated_after\":" << r.blocks_after
			  << ",\"errno\":" << r.error << ",\"disk_before\":";
			PrintDisk(f, r.disk_before);
			f << ",\"disk_after\":";
			PrintDisk(f, r.disk_after);
			f << '}';
		}
		f << "]}";
	}
	f << "]}\n";
}
} // namespace timeline
void timeline_install(const char *db) {
	timeline::output = std::string(db) + ".timeline.json";
	timeline::origin = timeline::Clock::now();
	timeline::epoch =
	    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
	        .count();
	const char *disk = std::getenv("VANE_FS_SYNC_STAT_PATH");
	timeline::disk_fd = open(disk ? disk : "/sys/block/sda/stat", O_RDONLY | O_CLOEXEC);
	if (!timeline::ReadDisk().valid)
		std::abort();
	std::atexit(timeline::Report);
}
extern "C" int __real_fsync(int);
extern "C" int __real_fdatasync(int);
static int Sync(int fd, bool data_only) {
	struct stat state {};
	int saved = errno;
	bool have = fstat(fd, &state) == 0;
	errno = saved;
	const auto path = timeline::FileName(fd);
#if VANE_FS_SPLIT_WAL_SYNC
	if (data_only && path.size() >= 4 && path.compare(path.size() - 4, 4, "-wal") == 0) {
		auto before = timeline::ReadDisk();
		auto start = timeline::Now();
		int rc =
		    sync_file_range(fd, 0, 0, SYNC_FILE_RANGE_WAIT_BEFORE | SYNC_FILE_RANGE_WRITE | SYNC_FILE_RANGE_WAIT_AFTER);
		timeline::Record("wal_writeback_wait", start, rc, fd, -1, -1, -1, true, have ? state.st_size : -1,
		                 have ? state.st_blocks * 512 : -1, &before);
		if (rc < 0)
			return rc;
	}
#endif
	auto before = timeline::ReadDisk();
	auto start = timeline::Now();
	auto rc = data_only ? __real_fdatasync(fd) : __real_fsync(fd);
	timeline::Record(data_only ? "fdatasync" : "fsync", start, rc, fd, -1, -1, -1, true, have ? state.st_size : -1,
	                 have ? state.st_blocks * 512 : -1, &before);
	return rc;
}
extern "C" int __wrap_fsync(int fd) {
	return Sync(fd, false);
}
extern "C" int __wrap_fdatasync(int fd) {
	return Sync(fd, true);
}
extern "C" int __real_sqlite3_wal_checkpoint_v2(sqlite3 *, const char *, int, int *, int *);
extern "C" int __wrap_sqlite3_wal_checkpoint_v2(sqlite3 *d, const char *n, int m, int *f, int *b) {
	auto t = timeline::Now();
	auto rc = __real_sqlite3_wal_checkpoint_v2(d, n, m, f, b);
	auto name = "checkpoint_" + std::to_string(m);
	timeline::Record(name.c_str(), t, rc, -1, f ? *f : -1, b ? *b : -1, -1, m != SQLITE_CHECKPOINT_NOOP);
	return rc;
}
extern "C" int __real_sqlite3_step(sqlite3_stmt *);
extern "C" int __wrap_sqlite3_step(sqlite3_stmt *s) {
	const char *sql = sqlite3_sql(s);
	bool control = !std::strcmp(sql, "BEGIN") || !std::strcmp(sql, "BEGIN IMMEDIATE") || !std::strcmp(sql, "COMMIT");
	if (!control)
		return __real_sqlite3_step(s);
	auto t = timeline::Now();
	auto rc = __real_sqlite3_step(s);
	timeline::Record(sql, t, rc);
	return rc;
}
extern "C" ssize_t __real_pread(int, void *, size_t, off_t);
extern "C" ssize_t __real_pwrite(int, const void *, size_t, off_t);
extern "C" ssize_t __wrap_pread(int fd, void *p, size_t n, off_t off) {
	auto t = timeline::Now();
	auto rc = __real_pread(fd, p, n, off);
	timeline::Record("pread", t, rc, fd, -1, -1, off);
	return rc;
}
extern "C" ssize_t __wrap_pwrite(int fd, const void *p, size_t n, off_t off) {
	auto t = timeline::Now();
	auto rc = __real_pwrite(fd, p, n, off);
	timeline::Record("pwrite", t, rc, fd, -1, -1, off);
	return rc;
}
extern "C" ssize_t __real_pread64(int, void *, size_t, off64_t);
extern "C" ssize_t __real_pwrite64(int, const void *, size_t, off64_t);
extern "C" ssize_t __wrap_pread64(int fd, void *p, size_t n, off64_t off) {
	auto t = timeline::Now();
	auto rc = __real_pread64(fd, p, n, off);
	timeline::Record("pread64", t, rc, fd, -1, -1, off);
	return rc;
}
extern "C" ssize_t __wrap_pwrite64(int fd, const void *p, size_t n, off64_t off) {
	auto t = timeline::Now();
	auto rc = __real_pwrite64(fd, p, n, off);
	timeline::Record("pwrite64", t, rc, fd, -1, -1, off);
	return rc;
}

extern "C" int __real_ftruncate(int, off_t);
extern "C" int __wrap_ftruncate(int fd, off_t length) {
	struct stat state {};
	int saved = errno;
	bool have = fstat(fd, &state) == 0;
	errno = saved;
	auto t = timeline::Now();
	auto rc = __real_ftruncate(fd, length);
	timeline::Record("ftruncate", t, rc, fd, -1, -1, length, true, have ? state.st_size : -1,
	                 have ? state.st_blocks * 512 : -1);
	return rc;
}

extern "C" int __real_ftruncate64(int, off64_t);
extern "C" int __wrap_ftruncate64(int fd, off64_t length) {
	struct stat state {};
	int saved = errno;
	bool have = fstat(fd, &state) == 0;
	errno = saved;
	auto t = timeline::Now();
	auto rc = __real_ftruncate64(fd, length);
	timeline::Record("ftruncate64", t, rc, fd, -1, -1, length, true, have ? state.st_size : -1,
	                 have ? state.st_blocks * 512 : -1);
	return rc;
}
