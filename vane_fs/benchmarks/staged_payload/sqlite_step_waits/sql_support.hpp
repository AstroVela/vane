// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Inserted inside the isolated recorder namespace, after Local().
struct alignas(64) ActiveSlot {
	std::atomic<uint64_t> sequence {0}, tid {0}, kind {0}, fd {0}, begin {0}, offset {0}, bytes {0}, mode {0};
};
struct alignas(64) ActivePage {
	char magic[8] {};
	uint64_t version = 1, stride = sizeof(ActiveSlot), slots = 16;
	char reserved[32] {};
	ActiveSlot calls[16];
};
static_assert(sizeof(ActiveSlot) == 64 && offsetof(ActivePage, calls) == 64);
static ActivePage *active_page = nullptr;
static void *image_base = nullptr;
static std::string image_path;

static void InitializeObserver() {
	Dl_info image {};
	if (!dladdr(reinterpret_cast<void *>(&Install), &image))
		std::abort();
	image_base = image.dli_fbase;
	image_path = image.dli_fname;
	int fd = int(syscall(SYS_memfd_create, "vane-fs-sql-waits", MFD_CLOEXEC));
	if (fd < 0 || ftruncate(fd, 4096))
		std::abort();
	void *mapping = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (mapping == MAP_FAILED)
		std::abort();
	active_page = new (mapping) ActivePage;
	std::memcpy(active_page->magic, "VNSQLW1", 8);
	void *warmup[16];
	backtrace(warmup, 16);
}

class SqlScope {
public:
	SqlScope(const char *sql, int api) : previous(current), sql(sql), api(api) {
		if (enabled) {
			id = Local().next_sql++;
			current = this;
		}
	}
	~SqlScope() {
		if (enabled)
			current = previous;
	}
	static thread_local SqlScope *current;
	SqlScope *previous;
	const char *sql;
	int api;
	int64_t id = -1;
};
thread_local SqlScope *SqlScope::current = nullptr;

static void TagSql(Event &event) {
	if (!SqlScope::current)
		return;
	auto &scope = *SqlScope::current;
	auto &statements = Local().statements;
	const char *text = scope.sql ? scope.sql : "";
	auto found = std::find(statements.begin(), statements.end(), text);
	if (found == statements.end()) {
		statements.emplace_back(text);
		event.sql = int64_t(statements.size() - 1);
	} else {
		event.sql = found - statements.begin();
	}
	event.sql_context = scope.id;
	event.sql_api = scope.api;
}

static void CaptureStack(Event &event) {
	int saved = errno;
	void *frames[24];
	int count = backtrace(frames, 24);
	for (int i = 0; i < count && event.stack_size < 16; ++i) {
		Dl_info image {};
		if (dladdr(frames[i], &image) && image.dli_fbase == image_base)
			event.stack[event.stack_size++] =
			    reinterpret_cast<uintptr_t>(frames[i]) - reinterpret_cast<uintptr_t>(image_base);
	}
	errno = saved;
}

class ActiveCall {
public:
	explicit ActiveCall(const Event &event) {
		uint64_t kind = event.kind == std::string_view("fsync")             ? 1
		                : event.kind == std::string_view("fdatasync")       ? 2
		                : event.kind == std::string_view("sync_file_range") ? 3
		                : event.kind == std::string_view("nanosleep")       ? 4
		                                                                    : 0;
		if (!kind)
			return;
		auto &owner = Local();
		slot = &active_page->calls[&owner - threads.data()];
		if (slot->kind.load(std::memory_order_relaxed))
			std::abort();
		slot->sequence.fetch_add(1, std::memory_order_acq_rel);
		slot->tid.store(uint64_t(owner.tid), std::memory_order_relaxed);
		slot->kind.store(kind, std::memory_order_relaxed);
		slot->fd.store(uint64_t(event.fd), std::memory_order_relaxed);
		slot->begin.store(uint64_t(event.begin), std::memory_order_relaxed);
		slot->offset.store(uint64_t(event.offset), std::memory_order_relaxed);
		slot->bytes.store(uint64_t(event.bytes), std::memory_order_relaxed);
		slot->mode.store(uint64_t(event.mode), std::memory_order_relaxed);
		slot->sequence.fetch_add(1, std::memory_order_release);
	}
	~ActiveCall() {
		Finish();
	}
	void Finish() {
		if (!slot)
			return;
		slot->sequence.fetch_add(1, std::memory_order_acq_rel);
		slot->kind.store(0, std::memory_order_relaxed);
		slot->sequence.fetch_add(1, std::memory_order_release);
		slot = nullptr;
	}

private:
	ActiveSlot *slot = nullptr;
};

static void JsonString(std::ostream &file, const std::string &text) {
	file << '"';
	for (unsigned char ch : text) {
		if (ch == '"' || ch == '\\')
			file << '\\' << char(ch);
		else if (ch < 32)
			file << "\\u00" << "0123456789abcdef"[ch >> 4] << "0123456789abcdef"[ch & 15];
		else
			file << char(ch);
	}
	file << '"';
}
