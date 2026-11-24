# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Add SQL identity, sleep stacks and a read-only active-call marker to frozen sources."""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "paced_request_waits/prepare.py"
spec = importlib.util.spec_from_file_location("paced_recorder", PARENT)
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
TRACE, digest, save = parent.TRACE, parent.digest, parent.save
builder = parent.namespace["scope"]["namespace"]
builder["helpers"].WRAPS.append("nanosleep")
original_recorder = parent.namespace["recorder"]


def recorder(file):
    text, pairs = original_recorder(file)

    def replace(before, after):
        nonlocal text
        assert text.count(before) == 1, (file, before)
        text = text.replace(before, after)
        pairs.append((before, after))

    if file.endswith(".hpp"):
        replace("\tRangeWriteback,\n\tCount", "\tRangeWriteback,\n\tSleep,\n\tCount")
    else:
        replace(
            '#include "wait_trace.hpp"',
            '#include "wait_trace.hpp"\n#include <algorithm>\n#include <cstring>\n#include <dlfcn.h>\n#include <execinfo.h>\n#include <linux/memfd.h>\n#include <new>\n#include <sys/mman.h>',
        )
        replace('"range_writeback"};', '"range_writeback", "sleep"};')
        replace(
            "\tint error = 0, fd = -1, mode = -1, frames = -1, copied = -1;",
            "\tint error = 0, fd = -1, mode = -1, frames = -1, copied = -1;\n"
            "\tint64_t sql = -1, sql_context = -1, sql_api = 0, vm_steps = 0;\n"
            "\tstd::array<int64_t, 16> stack {};\n\tint stack_size = 0;",
        )
        replace(
            "\tstd::vector<Event> events;",
            "\tstd::vector<Event> events;\n\tstd::vector<std::string> statements;\n\tint64_t next_sql = 0;",
        )
        replace(
            "static void Record(Event event) {",
            (HERE / "sql_support.hpp").read_text() + "\nstatic void Record(Event event) {\n\tTagSql(event);",
        )
        replace(
            '\tfile << "],\\"counter_columns\\":[',
            '\tfile << "],\\"native_image\\":" << std::quoted(image_path);\n\tfile << ",\\"counter_columns\\":[',
        )
        replace(
            '\\"fd\\",\\"offset\\",\\"bytes\\",\\"mode\\",\\"frames\\",\\"copied\\"],\\"threads\\":[";',
            '\\"fd\\",\\"offset\\",\\"bytes\\",\\"mode\\",\\"frames\\",\\"copied\\",'
            '\\"sql\\",\\"sql_context\\",\\"sql_api\\",\\"vm_steps\\",\\"stack\\"],\\"threads\\":[";',
        )
        replace(
            "\t\t\t     << e.offset << ',' << e.bytes << ',' << e.mode << ',' << e.frames << ',' << e.copied << ']';",
            """\t\t\t     << e.offset << ',' << e.bytes << ',' << e.mode << ',' << e.frames << ',' << e.copied << ','
\t\t\t     << e.sql << ',' << e.sql_context << ',' << e.sql_api << ',' << e.vm_steps << ",[";
\t\t\tfor (int k = 0; k < e.stack_size; ++k) file << (k ? "," : "") << e.stack[k];
\t\t\tfile << "]]";""",
        )
        replace(
            '\t\tfile << "]}";',
            """\t\tfile << "],\\"statements\\":[";
\t\tfor (size_t j = 0; j < t.statements.size(); ++j) { if (j) file << ","; JsonString(file, t.statements[j]); }
\t\tfile << "]}";""",
        )
        replace(
            '\toutput = std::string(database) + ".waits.json";',
            '\tInitializeObserver();\n\toutput = std::string(database) + ".waits.json";',
        )
        replace(
            "\te.begin = Now(CLOCK_MONOTONIC);\n\tauto cpu = Now(CLOCK_THREAD_CPUTIME_ID);\n\tauto result = call();",
            "\tif (stage == Fsync || stage == Fdatasync || stage == RangeWriteback || stage == Sleep) CaptureStack(e);\n"
            "\te.begin = Now(CLOCK_MONOTONIC);\n\tauto cpu = Now(CLOCK_THREAD_CPUTIME_ID);\n"
            "\tActiveCall active_call(e);\n\tauto result = call();",
        )
        replace(
            "\tauto saved = errno;\n\te.cpu = Now(CLOCK_THREAD_CPUTIME_ID) - cpu;\n"
            "\te.end = Now(CLOCK_MONOTONIC);\n\tspan.Finish();\n\tif (stage == Fsync",
            "\tauto saved = errno;\n\tactive_call.Finish();\n\te.cpu = Now(CLOCK_THREAD_CPUTIME_ID) - cpu;\n"
            "\te.end = Now(CLOCK_MONOTONIC);\n\tspan.Finish();\n\tif (stage == Fsync",
        )
        replace("stage == RangeWriteback || e.end", "stage == RangeWriteback || stage == Sleep || e.end")
        replace("\t\te.file = FileKind(fd);", '\t\te.file = stage == Sleep ? "process" : FileKind(fd);')
        begin = text.index('extern "C" int __real_sqlite3_step(')
        end = text.index('extern "C" int __real_sqlite3_wal_checkpoint_v2(')
        replace(text[begin:end], (HERE / "sql_wrappers.hpp").read_text() + "\n")
        replace(
            "\tSpan span(Checkpoint);\n\tEvent e",
            '\tSpan span(Checkpoint);\n\tSqlScope sql("PRAGMA wal_checkpoint", 3);\n\tEvent e',
        )
    return text, pairs


def validate(trace):
    result = parent.validate(trace)
    sleep_calls = slow_sql = 0
    assert trace["event_columns"][14:] == ["sql", "sql_context", "sql_api", "vm_steps", "stack"]
    for thread in trace["threads"]:
        statements = thread["statements"]
        for event in thread["events"]:
            assert len(event) == 19
            if event[14] >= 0:
                assert 0 <= event[14] < len(statements) and event[15] >= 0 and event[16] in (1, 2, 3)
            else:
                assert event[15:17] == [-1, 0]
            assert event[17] >= 0 and len(event[18]) <= 16
            if event[0] == "nanosleep":
                assert event[1] == "process" and event[18]
                sleep_calls += 1
            if event[0] in ("sql_step", "sql_exec"):
                assert event[14] >= 0 and event[4] - event[3] >= 1000000
                slow_sql += 1
    return dict(result, sleep_calls=sleep_calls, slow_sql_calls=slow_sql)


builder.update(HERE=HERE, __file__=__file__, recorder=recorder, validate=validate)

if __name__ == "__main__":
    builder["main"]()
    out = Path(sys.argv[1]).resolve()
    report = json.loads((out / "build.json").read_text())
    for path in [PARENT, HERE.parent / "range_writeback_waits/prepare.py", *HERE.glob("*.py"), *HERE.glob("*.hpp")]:
        report["inputs"][str(path)] = digest(path)
    report["sql_diagnostics"] = dict(
        default_busy_handler_unchanged=True,
        sdk_unchanged=True,
        active_marker="4096-byte memfd, read-only observer, 64-byte seqlocked slots",
        slow_sql_ns=1000000,
    )
    save(out / "build.json", report)
