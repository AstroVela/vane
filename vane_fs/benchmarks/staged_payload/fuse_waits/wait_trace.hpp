// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <mutex>
#include <optional>

// Isolated diagnostics. No output is written while requests are running.
namespace fuse_waits {
enum Stage {
	RootStage,
	DatabaseWait,
	Admission,
	CheckpointWait,
	PublicationWait,
	LeaseWait,
	PayloadGuard,
	Flock,
	Begin,
	Commit,
	Publish,
	BeforeFull,
	AfterFull,
	BeginBatch,
	PrepareExternal,
	WriteBytes,
	Step,
	Exec,
	Notify,
	Pwrite,
	Pread,
	Fsync,
	Fdatasync,
	Checkpoint,
	Count
};
int64_t Now(int clock);
class Span {
public:
	explicit Span(Stage stage);
	~Span();
	void Finish();
	Span(const Span &) = delete;
	Span &operator=(const Span &) = delete;
	int64_t begin = 0, end = 0, cpu_ns = 0;

private:
	Stage stage;
	Span *parent = nullptr;
	int64_t cpu = 0, child_wall = 0, child_cpu = 0;
	bool running = false;
};
class Root {
public:
	explicit Root(const char *name, int64_t offset = -1, int64_t bytes = -1, int64_t inode = -1, int64_t caller = -1);
	~Root();
	Root(const Root &) = delete;
	Root &operator=(const Root &) = delete;

private:
	std::optional<Span> span;
};
inline std::unique_lock<std::mutex> Lock(Stage stage, std::mutex &mutex) {
	Span span(stage);
	return std::unique_lock<std::mutex>(mutex);
}
void Install(const char *database);
} // namespace fuse_waits
