// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Isolated Linux experiment, format 1004. All writers and checkpoints must
// participate. An OPEN fence excludes metadata whose external bytes have not
// passed a data barrier; a CLEAN fence permits ordinary SQLite WAL recovery.
#include "payload_file.hpp"
#include <array>
#include <mutex>

namespace vane_fs {
class Staging {
public:
	static constexpr const char *CLIENT_DATA = "vane_fs.staged_payload_experiment";
	static constexpr const char *PAYLOAD_MAGIC = "VANE-STAGED-1004:";
	static void Check(int code) {
		if (code != SQLITE_OK)
			throw Error((code & 255) == SQLITE_BUSY || (code & 255) == SQLITE_LOCKED ? ErrorCode::Busy
			                                                                         : ErrorCode::Storage,
			            std::string("Staged publication: ") + sqlite3_errstr(code));
	}
	static void Exec(sqlite3 *db, const char *sql) {
		Check(sqlite3_exec(db, sql, nullptr, nullptr, nullptr));
	}
	static void Configure(sqlite3 *db) {
		// A failed constructor/destructor must not checkpoint an OPEN batch.
		Check(sqlite3_db_config(db, SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, 1, nullptr));
		Check(sqlite3_wal_autocheckpoint(db, 0));
	}
	static Staging &Get(sqlite3 *db) {
		auto *stage = static_cast<Staging *>(sqlite3_get_clientdata(db, CLIENT_DATA));
		if (!stage)
			throw Error(ErrorCode::Storage, "Missing staged publication coordinator");
		return *stage;
	}

private:
	[[noreturn]] static void Fail(const char *message) {
		throw Error(ErrorCode::Storage, message);
	}
	[[noreturn]] static void IO(const char *message) {
		throw Error(ErrorCode::Storage, std::string(message) + ": " + std::strerror(errno));
	}
	struct File {
		int fd = -1;
		~File() {
			Close();
		}
		void Close() {
			if (fd >= 0)
				close(fd);
			fd = -1;
		}
		void Open(const std::string &path, bool create) {
			fd = open(path.c_str(), O_RDWR | O_CLOEXEC | O_NOFOLLOW | (create ? O_CREAT : 0), 0600);
			if (fd < 0)
				IO("Opening publication file");
			struct stat st {};
			if (fstat(fd, &st) || !S_ISREG(st.st_mode))
				Fail("Publication file is not regular");
		}
	};
	class Lock {
	public:
		Lock(int fd, int mode, int timeout) : fd(fd), process(getpid()) {
			auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout);
			for (;;) {
				if (!flock(fd, mode | LOCK_NB))
					return;
				if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR)
					IO("Locking publication file");
				if (std::chrono::steady_clock::now() >= end)
					throw Error(ErrorCode::Busy, "Publication is busy");
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		}
		~Lock() {
			if (getpid() == process)
				flock(fd, LOCK_UN);
		}

	private:
		int fd;
		pid_t process;
	};
	static void Read(int fd, void *buffer, size_t size, int64_t offset) {
		auto *data = static_cast<char *>(buffer);
		while (size) {
			auto n = pread(fd, data, size, offset);
			if (n < 0 && errno == EINTR)
				continue;
			if (n < 0)
				IO("Reading publication file");
			if (!n)
				Fail("Truncated publication file");
			data += n;
			size -= size_t(n);
			offset += n;
		}
	}
	static void Sync(int fd) {
		int rc;
		do {
			rc = fdatasync(fd);
		} while (rc && errno == EINTR);
		if (rc)
			IO("Synchronizing publication file");
	}
	static uint64_t Decode(const unsigned char *p, int size = 8) {
		uint64_t value = 0;
		for (int i = 0; i < size; ++i)
			value = (value << 8) | p[i];
		return value;
	}
	static void Encode(unsigned char *p, uint64_t value) {
		for (int i = 7; i >= 0; --i) {
			p[i] = value & 255;
			value >>= 8;
		}
	}
	using Record = std::array<unsigned char, 4096>;
	static constexpr const char *MAGIC = "VANE-STAGE-FENCE-1004";
	static uint64_t Checksum(const Record &record) {
		// CRC64-ECMA detects incomplete records; this is not authentication.
		uint64_t crc = 0;
		// Only the first 152 bytes carry fields. The remaining slot padding
		// separates writes into different sectors and has no recovery meaning.
		for (size_t i = 0; i < 152; ++i) {
			crc ^= uint64_t(record[i]) << 56;
			for (int bit = 0; bit < 8; ++bit)
				crc = (crc << 1) ^ ((crc >> 63) ? UINT64_C(0x42f0e1eba9ea3693) : 0);
		}
		return crc;
	}
	static bool Valid(const Record &record) {
		return !std::memcmp(record.data(), MAGIC, std::strlen(MAGIC)) && Decode(record.data() + 32) &&
		       Decode(record.data() + 40) <= 1 && Checksum(record) == Decode(record.data() + 4088);
	}
	void Load() {
		if (fence.fd < 0)
			return;
		struct stat st {};
		if (fstat(fence.fd, &st))
			IO("Inspecting publication fence");
		Record newest {};
		for (int slot = 0; slot < 2; ++slot) {
			if (st.st_size < (slot + 1) * 4096)
				continue;
			Record candidate {};
			Read(fence.fd, candidate.data(), candidate.size(), slot * 4096);
			if (Valid(candidate) && Decode(candidate.data() + 32) > Decode(newest.data() + 32))
				newest = candidate;
		}
		if (!Valid(newest))
			Fail("No valid publication fence");
		if (!uuid.empty() && std::memcmp(newest.data() + 48, uuid.data(), 32))
			Fail("Publication fence UUID differs");
		// A complete record may still live only in page cache after another
		// connection's failed sync. Validate AND synchronize any unfamiliar
		// sequence before relying on either OPEN or CLEAN across connections.
		auto sequence = Decode(newest.data() + 32);
		if (sequence != synced_sequence) {
			Sync(fence.fd);
			synced_sequence = sequence;
		}
		record = newest;
	}
	void Save(bool pending) {
		auto sequence = Decode(record.data() + 32);
		if (sequence == UINT64_MAX)
			Fail("Publication sequence exhausted");
		Record next = record;
		std::memcpy(next.data(), MAGIC, std::strlen(MAGIC));
		Encode(next.data() + 32, sequence + 1);
		Encode(next.data() + 40, pending ? 1 : 0);
		std::memcpy(next.data() + 48, uuid.data(), 32);
		Encode(next.data() + 4088, Checksum(next));
		auto offset = int64_t(sequence % 2) * 4096;
		size_t written = 0;
		while (written < next.size()) {
			auto n = pwrite(fence.fd, next.data() + written, next.size() - written, offset + int64_t(written));
			if (n < 0 && errno == EINTR)
				continue;
			if (n <= 0)
				IO("Writing publication fence");
			written += size_t(n);
		}
		Sync(fence.fd);
		synced_sequence = sequence + 1;
		record = next;
	}
	void Recover() {
		if (!Pending())
			return;
		File wal;
		if (!std::filesystem::exists(path + "-wal"))
			Fail("Missing WAL for an OPEN publication fence");
		wal.Open(path + "-wal", false);
		struct stat st {};
		if (fstat(wal.fd, &st))
			IO("Inspecting recovery WAL");
		uint64_t frames = Decode(record.data() + 88), keep = 0;
		if (frames && st.st_size) {
			std::array<unsigned char, 32> header {};
			Read(wal.fd, header.data(), header.size(), 0);
			if (!std::memcmp(header.data(), record.data() + 96, header.size())) {
				auto page_size = Decode(header.data() + 8, 4);
				if (page_size < 512 || page_size > 65536 || (page_size & (page_size - 1)))
					Fail("Invalid WAL page size");
				if (frames > (uint64_t(INT64_MAX) - 32) / (page_size + 24))
					Fail("Invalid WAL fence length");
				keep = 32 + frames * (page_size + 24);
				if (uint64_t(st.st_size) < keep)
					Fail("WAL is shorter than its durable fence");
				std::array<unsigned char, 24> last {};
				Read(wal.fd, last.data(), last.size(), int64_t(keep - page_size - 24));
				if (std::memcmp(last.data(), record.data() + 128, last.size()))
					Fail("WAL commit fence differs");
			} else {
				// A managed WAL can restart only after the old generation has
				// reached the database. No checkpoint may run while OPEN.
				auto magic = Decode(header.data(), 4);
				if (magic != 0x377f0682 && magic != 0x377f0683)
					Fail("Invalid WAL generation header");
			}
		}
		if (ftruncate(wal.fd, int64_t(keep)))
			IO("Truncating unpublished WAL tail");
		Sync(wal.fd);
		// Repeated recovery must remain safe if initialization fails later.
		Save(false);
	}

public:
	class CheckpointGuard {
	public:
		CheckpointGuard(Staging &stage, bool exclusive = true) : thread(stage.checkpoint_mutex) {
			// Each lease needs its own open-file description: flock on a shared
			// descriptor would silently replace another thread's lock.
			file.Open(stage.path + ".stage-checkpoint", true);
			lock = std::make_unique<Lock>(file.fd, exclusive ? LOCK_EX : LOCK_SH, stage.timeout);
		}

	private:
		std::unique_lock<std::mutex> thread;
		File file;
		std::unique_ptr<Lock> lock;
	};
	class Guard {
	public:
		explicit Guard(Staging &stage) : thread(stage.mutex), file(stage.publication.fd, LOCK_EX, stage.timeout) {
			stage.Load();
		}

	private:
		std::unique_lock<std::mutex> thread;
		Lock file;
	};
	explicit Staging(const std::string &database, int timeout) : timeout(timeout) {
		path = std::filesystem::weakly_canonical(database).string();
		initialization.Open(path + ".stage-init", true);
		init_lock = std::make_unique<Lock>(initialization.fd, LOCK_EX, timeout);
		liveness.Open(path + ".stage-live", true);
		bool cold = !flock(liveness.fd, LOCK_EX | LOCK_NB);
		if (!cold) {
			if (errno != EAGAIN && errno != EWOULDBLOCK)
				IO("Locking publication liveness");
			if (flock(liveness.fd, LOCK_SH))
				IO("Joining live publication");
		}
		publication.Open(path + ".stage-publish", true);
		// An existing data file provides the identity before SQLite can replay
		// an unsafe WAL tail. Refuse old formats rather than modifying them.
		if (std::filesystem::exists(path + ".payload")) {
			File data;
			data.Open(path + ".payload", false);
			std::array<char, 4096> header {};
			Read(data.fd, header.data(), header.size(), 0);
			if (std::memcmp(header.data(), PAYLOAD_MAGIC, std::strlen(PAYLOAD_MAGIC)))
				throw Error(ErrorCode::Invalid, "Not a staged-payload workspace");
			uuid.assign(header.data() + std::strlen(PAYLOAD_MAGIC), 32);
			if (!std::filesystem::exists(path))
				Fail("Missing staged workspace database");
			fence.Open(path + ".stage-fence", false);
		} else if (std::filesystem::exists(path + ".stage-fence")) {
			Fail("Missing staged payload file");
		}
		startup = std::make_unique<Guard>(*this);
		if (cold && fence.fd >= 0)
			Recover();
	}
	void Initialize(const std::string &identity, bool fresh) {
		if (identity.size() != 32)
			Fail("Invalid workspace identity");
		if (!uuid.empty() && uuid != identity)
			Fail("Payload and database UUID differ");
		uuid = identity;
		if (fresh) {
			if (fence.fd >= 0 || std::filesystem::exists(path + ".stage-fence"))
				Fail("Existing publication fence");
			fence.Open(path + ".stage-fence", true);
			Save(false);
			int directory = open(std::filesystem::path(path).parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
			if (directory < 0)
				IO("Opening publication directory");
			int rc = fsync(directory), saved = errno;
			close(directory);
			if (rc) {
				errno = saved;
				IO("Synchronizing publication directory");
			}
		} else if (fence.fd < 0)
			Fail("Missing publication fence");
	}
	void Ready() {
		if (flock(liveness.fd, LOCK_SH))
			IO("Publishing live workspace");
		startup.reset();
		init_lock.reset();
	}
	void ReleaseLive() {
		liveness.Close();
	}
	void AbandonInherited() {
		// Do not unlock the parent's shared open-file descriptions after fork.
		liveness.Close();
		publication.Close();
		initialization.Close();
		fence.Close();
	}
	bool Pending() const {
		return Decode(record.data() + 40) != 0;
	}
	void BeginBatch(sqlite3 *writer) {
		if (Pending())
			return;
		// A running PASSIVE checkpoint must finish before a new OPEN fence.
		// It holds no publication lock, so FULL barriers can still proceed.
		CheckpointGuard checkpoint(*this, false);
		// NOOP is SQLITE_LOCKED on a connection already in a transaction.
		// A separate pager reports committed mxFrame, excluding spilled pages
		// from the current IMMEDIATE transaction. The publication guard excludes
		// other managed writers and checkpoints throughout this capture.
		sqlite3 *raw = nullptr;
		auto code = sqlite3_open_v2(path.c_str(), &raw,
		                            SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX | SQLITE_OPEN_PRIVATECACHE, nullptr);
		std::unique_ptr<sqlite3, decltype(&sqlite3_close_v2)> observer(raw, sqlite3_close_v2);
		Check(code);
		Configure(raw);
		Exec(raw, "SELECT count(*) FROM sqlite_master");
		int frames = -1, copied = -1;
		Check(sqlite3_wal_checkpoint_v2(raw, "main", SQLITE_CHECKPOINT_NOOP, &frames, &copied));
		if (frames < 0)
			Fail("Missing committed WAL status");
		File wal;
		wal.Open(std::string(sqlite3_db_filename(writer, "main")) + "-wal", false);
		Encode(record.data() + 88, uint64_t(frames));
		if (frames) {
			Read(wal.fd, record.data() + 96, 32, 0);
			auto page_size = Decode(record.data() + 104, 4);
			if (page_size < 512 || page_size > 65536 || (page_size & (page_size - 1)))
				Fail("Invalid WAL page size");
			Read(wal.fd, record.data() + 128, 24, 32 + int64_t(frames - 1) * int64_t(page_size + 24));
			if (!Decode(record.data() + 132, 4) || std::memcmp(record.data() + 136, record.data() + 112, 8))
				Fail("Not a committed WAL fence");
		}
		Sync(wal.fd);
		Save(true);
	}
	void BeforeFullCommit() {
		if (!Pending())
			return;
		// Flush the whole file, including bytes appended by other connections.
		File data;
		data.Open(path + ".payload", false);
		Sync(data.fd);
	}
	void AfterFullCommit() {
		if (Pending())
			Save(false);
	}
	void Publish(sqlite3 *db) {
		if (!Pending())
			return;
		Exec(db, "PRAGMA synchronous=FULL; BEGIN IMMEDIATE");
		try {
			Exec(db, "UPDATE sync_barrier SET value=1-value WHERE id=1");
			BeforeFullCommit();
			Exec(db, "COMMIT");
			AfterFullCommit();
			Exec(db, "PRAGMA synchronous=NORMAL");
		} catch (...) {
			sqlite3_exec(db, "ROLLBACK; PRAGMA synchronous=NORMAL", nullptr, nullptr, nullptr);
			throw;
		}
	}

private:
	const int timeout;
	std::string path, uuid;
	File initialization, liveness, publication, fence;
	std::mutex mutex, checkpoint_mutex;
	std::unique_ptr<Lock> init_lock;
	std::unique_ptr<Guard> startup;
	Record record {};
	uint64_t synced_sequence = 0;
};
} // namespace vane_fs
