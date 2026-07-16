from __future__ import annotations

from multimodal_inference_benchmarks.video_object_detection.profiling import profile_01_read_resize


def test_vane_reader_timing_parser_summarizes_worker_stage_lines():
    lines = [
        "[vane_video][reader_timing] run_id=test pid=1 call=1 video=a.avi rows=8 "
        "open_s=0.100000 open_ms=100.000 decode_s=0.200000 decode_ms=200.000 "
        "resize_s=0.300000 resize_ms=300.000 flush_s=0.010000 flush_ms=10.000 "
        "total_s=0.700000 total_ms=700.000 rows_per_s=11.43",
        "[vane_video][reader_timing] run_id=test pid=2 call=1 video=b.avi rows=4 "
        "open_s=0.050000 open_ms=50.000 decode_s=0.100000 decode_ms=100.000 "
        "resize_s=0.200000 resize_ms=200.000 flush_s=0.020000 flush_ms=20.000 "
        "total_s=0.400000 total_ms=400.000 rows_per_s=10.00",
    ]

    result = profile_01_read_resize._summarize_vane_reader_timing_lines(lines)

    assert result["batches"] == 2
    assert result["frames"] == 12
    assert result["open_s"]["sum"] == 0.15
    assert result["decode_s"]["sum"] == 0.3
    assert result["resize_s"]["sum"] == 0.5
    assert result["flush_s"]["sum"] == 0.03
    assert result["total_s"]["sum"] == 1.1
