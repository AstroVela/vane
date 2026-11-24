// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>

// Diagnostic copies only. Exclusive counters subtract complete child spans.
// C++ allocations are counted separately from SQLite's C allocator.
namespace write_costs {
enum Stage {
	Root,
	WriteBytes,
	Batch,
	BuildSql,
	Concatenate,
	BindBatch,
	AcquireStatement,
	ReleaseStatement,
	Prepare,
	StepPayload,
	StepVersion,
	StepOther,
	Exec,
	PayloadAppend,
	Pwrite,
	Pwrite64,
	Pread,
	Sync,
	Checkpoint,
	Count
};
struct Counter {
	int64_t calls = 0, inclusive_wall_ns = 0, exclusive_wall_ns = 0;
	int64_t inclusive_cpu_ns = 0, exclusive_cpu_ns = 0;
	int64_t cpp_allocations = 0, cpp_allocated_bytes = 0, bytes = 0, vm_steps = 0;
};
using Counters = std::array<Counter, Count>;
extern thread_local Counters counters;
extern thread_local bool enabled;
int64_t Now(int clock);
class Scope {
public:
	explicit Scope(Stage stage);
	~Scope();
	void Finish();
	Scope(const Scope &) = delete;
	Scope &operator=(const Scope &) = delete;
	Stage stage;

private:
	Scope *parent = nullptr;
	int64_t wall = 0, cpu = 0, child_wall = 0, child_cpu = 0;
	bool running = false;
};
void Allocated(size_t bytes);
void AddBytes(Stage stage, int64_t bytes);
void Begin(int phase);
Counters End();
void Phase(int phase);
void Print(std::ostream &output, const Counters &values);
void PrintEvents(std::ostream &output);
} // namespace write_costs
