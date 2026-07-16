from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .profile_common import PROFILE_DIR
    from .profile_common import build_common_parser
    from .profile_common import write_json_result
except ImportError:
    from profile_common import PROFILE_DIR
    from profile_common import build_common_parser
    from profile_common import write_json_result


@dataclass(frozen=True)
class StageCommand:
    stage: str
    script: Path
    output_json: Path
    argv: list[str]


_STAGE_SCRIPTS = [
    ("read_resize", "profile_01_read_resize.py"),
    ("tensor_yolo", "profile_02_tensor_yolo.py"),
    ("crop_png", "profile_03_crop_png.py"),
    ("write_parquet", "profile_04_write_parquet.py"),
]


def build_stage_commands(
    *,
    python: str,
    output_dir: Path,
    input_path: str,
    input_manifest: str | None,
    frames: int,
    height: int,
    width: int,
    batch_size: int = 16,
) -> list[StageCommand]:
    commands: list[StageCommand] = []
    for stage, script_name in _STAGE_SCRIPTS:
        script = PROFILE_DIR / script_name
        output_json = output_dir / f"{stage}.json"
        argv = [
            python,
            str(script),
            "--input-path",
            input_path,
            "--frames",
            str(frames),
            "--height",
            str(height),
            "--width",
            str(width),
            "--batch-size",
            str(batch_size),
            "--output-dir",
            str(output_dir / stage),
            "--output-json",
            str(output_json),
            "--quiet",
        ]
        if input_manifest:
            argv.extend(["--input-manifest", input_manifest])
        commands.append(StageCommand(stage=stage, script=script, output_json=output_json, argv=argv))
    return commands


def _run_command(command: StageCommand) -> dict[str, Any]:
    proc = subprocess.run(
        command.argv,
        cwd=PROFILE_DIR.parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    result: dict[str, Any] = {
        "stage": command.stage,
        "script": str(command.script),
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode == 0 and command.output_json.exists():
        result["result"] = json.loads(command.output_json.read_text(encoding="utf-8"))
    return result


def _nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _markdown_summary(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Video Stage Profile",
        "",
        "| stage | status | Ray rows/s | Vane rows/s | Ray wall(s) | Vane wall(s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        result = item.get("result") or {}
        ray_rows = _nested_get(result, ("ray_data", "rows_per_s"))
        vane_rows = _nested_get(result, ("vane", "rows_per_s"))
        ray_wall = _nested_get(result, ("ray_data", "wall_s"))
        vane_wall = _nested_get(result, ("vane", "wall_s"))
        status = "ok" if item["returncode"] == 0 else f"failed:{item['returncode']}"
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                item["stage"],
                status,
                f"{ray_rows:.2f}" if isinstance(ray_rows, (int, float)) else "",
                f"{vane_rows:.2f}" if isinstance(vane_rows, (int, float)) else "",
                f"{ray_wall:.3f}" if isinstance(ray_wall, (int, float)) else "",
                f"{vane_wall:.3f}" if isinstance(vane_wall, (int, float)) else "",
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = build_common_parser("Run all video stage profiling scripts and aggregate results.")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = build_stage_commands(
        python=sys.executable,
        output_dir=output_dir,
        input_path=args.input_path,
        input_manifest=args.input_manifest,
        frames=args.frames,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
    )
    results = [_run_command(command) for command in commands]
    summary = {
        "stage": "run_all",
        "output_dir": str(output_dir),
        "results": results,
    }
    (output_dir / "summary.md").write_text(_markdown_summary(results), encoding="utf-8")
    write_json_result(summary, output_json=args.output_json or output_dir / "summary.json", quiet=args.quiet)


if __name__ == "__main__":
    main()
