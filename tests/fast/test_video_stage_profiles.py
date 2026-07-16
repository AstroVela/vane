from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_DIR = _REPO_ROOT / "multimodal_inference_benchmarks/video_object_detection/profiling"
_SCRIPTS = [
    "profile_01_read_resize.py",
    "profile_02_tensor_yolo.py",
    "profile_03_crop_png.py",
    "profile_04_write_parquet.py",
    "profile_run_all.py",
]


def test_video_stage_profile_common_helpers_emit_stable_json(tmp_path):
    from multimodal_inference_benchmarks.video_object_detection.profiling.profile_common import (
        build_common_parser,
        make_synthetic_features,
        summarize_values,
        write_json_result,
    )

    assert summarize_values([0.001, 0.003, 0.002]) == {
        "count": 3,
        "sum": 0.006,
        "avg": 0.002,
        "min": 0.001,
        "p50": 0.002,
        "p95": 0.003,
        "max": 0.003,
    }

    features = make_synthetic_features(frame_count=2, width=640, height=480, objects_per_frame=2)
    assert len(features) == 2
    assert len(features[0]) == 2
    assert features[0][0]["label"] == 0
    assert features[0][0]["bbox"] == [0.0, 0.0, 160.0, 120.0]
    assert features[1][1]["label"] == 1

    parser = build_common_parser("unit")
    args = parser.parse_args(
        [
            "--input-path",
            "/tmp/videos",
            "--frames",
            "4096",
            "--height",
            "720",
            "--width",
            "1280",
        ]
    )
    assert args.input_path == "/tmp/videos"
    assert args.frames == 4096
    assert args.height == 720
    assert args.width == 1280

    output_json = tmp_path / "stage.json"
    payload = {"stage": "unit", "frames": 4}
    write_json_result(payload, output_json=output_json, quiet=True)
    assert json.loads(output_json.read_text(encoding="utf-8")) == payload


def test_video_stage_profile_runner_builds_all_stage_commands(tmp_path):
    from multimodal_inference_benchmarks.video_object_detection.profiling.profile_run_all import (
        build_stage_commands,
    )

    commands = build_stage_commands(
        python=sys.executable,
        output_dir=tmp_path,
        input_path="/tmp/videos",
        input_manifest="/tmp/manifest.txt",
        frames=4096,
        height=640,
        width=640,
    )

    assert [command.stage for command in commands] == [
        "read_resize",
        "tensor_yolo",
        "crop_png",
        "write_parquet",
    ]
    for command in commands:
        assert command.output_json.parent == tmp_path
        assert "--frames" in command.argv
        assert "4096" in command.argv
        assert "--input-path" in command.argv
        assert "/tmp/videos" in command.argv
        assert "--input-manifest" in command.argv
        assert "/tmp/manifest.txt" in command.argv


def test_video_stage_profile_help_is_lazy_and_consistent():
    for script_name in _SCRIPTS:
        script = _PROFILE_DIR / script_name
        proc = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=_REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "--frames" in proc.stdout
        assert "--output-json" in proc.stdout
