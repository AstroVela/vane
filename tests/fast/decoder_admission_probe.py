# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, CPU-only end-to-end regression for decoder starvation (#859).

This also runs against unmodified v0.2.0. Four queries share two native threads
and one decoder permit, with asynchronous downstream backpressure. All queries
must finish without increasing threads, releasing live decoders, or using Ray.
"""

from __future__ import annotations

import collections
import faulthandler
import json
import os
import sys
import threading
import time
from pathlib import Path

import av
import numpy as np
import pyarrow as pa

import vane
from vane.datasource import DataSource, read_datasource
from vane.datasource.video_reader import VideoFrameSource


class Source(DataSource):
    def __init__(self, path):
        self.source = VideoFrameSource([path] * 3, height=32, width=32, max_partition_bytes=16_000)

    @property
    def schema(self):
        return self.source.schema

    def get_tasks(self):
        return self.source.get_tasks()


class SlowConsumer:
    def __call__(self, table):
        time.sleep(0.01)
        return pa.table({"frame_index": table.column("frame_index")})


def main(directory):
    faulthandler.dump_traceback_later(75, exit=True)
    path = directory / "decoder-admission.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        for index in range(512):
            pixels = np.full((32, 32, 3), index % 256, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    connection = vane.connect(config={"threads": 2, "video_backend": "python", "preserve_insertion_order": False})
    cursors = [connection.cursor() for _ in range(4)]
    results, errors = [], []

    def run_query(cursor):
        try:
            rows = (
                read_datasource(Source(str(path)), con=cursor)
                .map_batches(
                    SlowConsumer,
                    schema={"frame_index": "BIGINT"},
                    batch_size=8,
                    actor_number=1,
                    execution_backend="subprocess_actor",
                    task_input_max_bytes=32768,
                    output_target_max_bytes=32768,
                )
                .fetchall()
            )
            assert collections.Counter(rows) == collections.Counter({(index,): 3 for index in range(512)})
            results.append(len(rows))
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")

    threads = [threading.Thread(target=run_query, args=(cursor,)) for cursor in cursors]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    try:
        deadline = started + 55
        for thread in threads:
            thread.join(max(0, deadline - time.monotonic()))
        completed = not any(thread.is_alive() for thread in threads)
        if not completed:
            faulthandler.dump_traceback()
    finally:
        for cursor in cursors:
            cursor.interrupt()
        for thread in threads:
            thread.join(2)
        alive = any(thread.is_alive() for thread in threads)
        if not alive:
            for cursor in cursors:
                cursor.close()
            connection.close()

    print(
        json.dumps(
            {"completed": completed, "rows": results, "errors": errors, "elapsed_s": time.monotonic() - started}
        ),
        flush=True,
    )
    if alive:
        # Do not let a regression leave this test's executor/actor processes
        # waiting indefinitely for interpreter shutdown.
        os._exit(2)
    faulthandler.cancel_dump_traceback_later()
    return 0 if completed and results == [1536] * 4 and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
