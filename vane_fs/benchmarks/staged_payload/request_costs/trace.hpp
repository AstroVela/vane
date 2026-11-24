// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <mutex>

// Diagnostic copies only. Short intervals accumulate in the active request;
// they never perform file inspection, logging, or a shared-counter update.
namespace request_trace {
enum class Stage {
	DatabaseWait,
	Admission,
	PublicationWait,
	PayloadGuard,
	PinsWait,
	Begin,
	Commit,
	StatementAcquire,
	StatementStep,
	StatementRelease,
	Exec,
	WritebackError,
	PauseEnter,
	PauseExit,
	Publication,
	Notify,
	Checkpoint,
	WalHook,
	Pread,
	Pwrite,
	Fsync,
	Fdatasync,
	Hint,
	Count
};
void Install(const char *database);
long long Now();
void Add(Stage stage, long long begin);
void Steps(int value);
void Wal(int mode, int frames, int copied, int rc);
void CommitFrames(int frames);
class Scope {
public:
	explicit Scope(const char *name, int64_t offset = -1, int64_t bytes = -1);
	~Scope();
	Scope(const Scope &) = delete;
	Scope &operator=(const Scope &) = delete;

private:
	void *previous;
	void *record;
};
class Span {
public:
	explicit Span(Stage stage) : stage(stage), begin(Now()) {
	}
	~Span() {
		Add(stage, begin);
	}

private:
	Stage stage;
	long long begin;
};
inline std::unique_lock<std::mutex> Lock(Stage stage, std::mutex &mutex) {
	Span span(stage);
	return std::unique_lock<std::mutex>(mutex);
}
} // namespace request_trace
