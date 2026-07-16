import asyncio

from examples.ray_worker_example import SimpleWorker


def test_simple_worker_returns_success():
    w = SimpleWorker()
    h = w.submit_task({"name": "hello"})
    res = asyncio.run(h.get_result())
    # result format: ("Success", parts_list, stats_bytes)
    assert isinstance(res, tuple)
    assert res[0] == "Success"
    parts = res[1]
    assert isinstance(parts, list)
    assert len(parts) == 1
    part = parts[0]
    assert part.get_num_rows() == 1
    assert part.get_size_bytes() >= 0
    # get_object_ref should return the original task data in this example
    assert isinstance(part.get_object_ref(), dict)
