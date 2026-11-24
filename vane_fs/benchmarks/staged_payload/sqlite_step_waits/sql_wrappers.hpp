// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: Apache-2.0

// Replaces only the recorder's existing sqlite3_step and sqlite3_exec wrappers.
static void SqlResult(const char *kind, Span &span, int result, int steps = 0) {
	span.Finish();
	if (!span.begin || span.end - span.begin < 1000000)
		return;
	Event event {kind, "sqlite"};
	event.root = row ? row->id : -1;
	event.begin = span.begin;
	event.end = span.end;
	event.cpu = span.cpu_ns;
	event.result = result;
	event.vm_steps = steps;
	Record(event);
}
extern "C" int __real_sqlite3_step(sqlite3_stmt *);
extern "C" int __wrap_sqlite3_step(sqlite3_stmt *statement) {
	Span span(Step);
	SqlScope sql(sqlite3_sql(statement), 1);
	auto rc = __real_sqlite3_step(statement);
	int saved = errno;
	auto steps = row ? sqlite3_stmt_status(statement, SQLITE_STMTSTATUS_VM_STEP, 1) : 0;
	if (row)
		row->steps += steps;
	SqlResult("sql_step", span, rc, steps);
	errno = saved;
	return rc;
}
extern "C" int __real_sqlite3_exec(sqlite3 *, const char *, int (*)(void *, int, char **, char **), void *, char **);
extern "C" int __wrap_sqlite3_exec(sqlite3 *db, const char *sql, int (*callback)(void *, int, char **, char **),
                                   void *context, char **error) {
	Span span(Exec);
	SqlScope scope(sql, 2);
	auto rc = __real_sqlite3_exec(db, sql, callback, context, error);
	int saved = errno;
	SqlResult("sql_exec", span, rc);
	errno = saved;
	return rc;
}
extern "C" int __real_nanosleep(const timespec *, timespec *);
extern "C" int __wrap_nanosleep(const timespec *request, timespec *remaining) {
	int64_t duration = -1;
	if (request && request->tv_sec >= 0 && request->tv_sec <= (INT64_MAX - 1000000000) / 1000000000)
		duration = int64_t(request->tv_sec) * 1000000000 + request->tv_nsec;
	return Call(Sleep, "nanosleep", -1, duration, 0, -1, [&] { return __real_nanosleep(request, remaining); });
}
