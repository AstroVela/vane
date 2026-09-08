# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce audio parity measurements with independently installed runtimes.

Use generate, run, and compare --help for the individual stages. All generated
media, arrays, versions, and errors are retained under the supplied directory.
Differences are measurements, not pass/fail assertions about either contract.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(root: Path) -> None:
    import soundfile as sf

    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    cases = []

    def add(name: str, frames: int, channels: int, rate: int, targets: list[int], fmt="WAV", subtype="PCM_16"):
        t = np.arange(frames, dtype=np.float64) / rate
        rng = np.random.default_rng(20260908)
        signal = np.stack(
            [
                0.40 * np.sin(2 * np.pi * (317 + 223 * channel) * t)
                + 0.18 * np.cos(2 * np.pi * (0.39 - 0.03 * channel) * rate * t)
                + 0.03 * rng.standard_normal(frames)
                for channel in range(channels)
            ],
            axis=1,
        )
        if frames:
            signal[0] = 0.75
            signal[-1] = -0.625
        path = inputs / f"{name}.{fmt.lower()}"
        sf.write(path, signal, rate, format=fmt, subtype=subtype)
        cases.append(
            {
                "id": name,
                "file": str(path.relative_to(root)),
                "sha256": digest(path),
                "frames": frames,
                "channels": channels,
                "sample_rate": rate,
                "targets": sorted(set([rate, *targets])),
            }
        )

    add("mono_8000", 800, 1, 8000, [4000, 16000])
    add("stereo_8000", 800, 2, 8000, [4000, 16000])
    add("four_channels", 997, 4, 44100, [16000, 48000])
    add("mono_fractional", 1001, 1, 44100, [16000, 48000])
    add("stereo_fractional", 1001, 2, 44100, [16000, 48000])
    add("streaming_stereo", 150001, 2, 48000, [16000, 44100])
    for frames in [0, 1, 2, 7, 31, 32, 33, 65]:
        add(f"short_mono_{frames}", frames, 1, 8000, [4000, 16000])
    for subtype in ["PCM_U8", "PCM_24", "PCM_32", "FLOAT", "DOUBLE", "ULAW", "ALAW"]:
        add(f"wav_{subtype.lower()}", 2049, 2, 44100, [16000], subtype=subtype)
    for fmt, subtype in [
        ("AIFF", "PCM_16"),
        ("FLAC", "PCM_24"),
        ("MP3", "MPEG_LAYER_III"),
        ("OGG", "VORBIS"),
        ("OGG", "OPUS"),
    ]:
        add(f"{fmt.lower()}_{subtype.lower()}", 4801, 2, 48000, [16000], fmt, subtype)
    source = inputs / "stereo_8000.wav"
    for suffix, codec in [("m4a", "aac"), ("webm", "libopus")]:
        path = inputs / f"native_extra.{suffix}"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source), "-c:a", codec, str(path)],
            check=True,
        )
        cases.append(
            {
                "id": f"native_extra_{suffix}",
                "file": str(path.relative_to(root)),
                "sha256": digest(path),
                "frames": 800 if suffix == "m4a" else 4800,
                "channels": 2,
                "sample_rate": 8000 if suffix == "m4a" else 48000,
                "targets": [8000, 16000],
            }
        )
    write_json(root / "manifest.json", {"seed": 20260908, "cases": cases})
    print(f"Generated {len(cases)} inputs and {sum(len(c['targets']) for c in cases)} resample cases", flush=True)


def versions() -> dict[str, Any]:
    import soundfile
    import soxr

    result = {"python": platform.python_version(), "libsndfile": soundfile.__libsndfile_version__}
    result["libsoxr"] = soxr.__libsoxr_version__
    for name in ["vane-ai", "daft", "numpy", "soundfile", "soxr", "librosa", "numba"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def run(root: Path, engine: str, label: str, artifact: Path | None) -> None:
    os.environ.setdefault("VANE_RUNNER", "local-fast")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    import librosa

    out = root / label
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"engine": engine, "versions": versions(), "files": {}, "resamples": {}}
    manifest = json.loads((root / "manifest.json").read_text())

    def capture(key, callback):
        try:
            value = callback()
            if isinstance(value, np.ndarray):
                path = out / f"{key}.npy"
                np.save(path, value, allow_pickle=False)
                return {
                    "array": str(path.relative_to(root)),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "sha256": digest(path),
                }
            if dataclasses.is_dataclass(value):
                value = dataclasses.asdict(value)
            return {"value": value}
        except Exception as error:
            return {"error": type(error).__name__, "message": str(error)[:1500]}

    if engine == "vane":
        import vane

        pycon = vane.connect()
        native = vane.connect(config={"allow_unsigned_extensions": "true"})
        if artifact is None:
            raise ValueError("Vane measurements require --native-extension")
        native.load_extension(str(artifact))
        native.execute("SET audio_backend='native'")
        results["native_artifact"] = {"path": str(artifact), "sha256": digest(artifact)}
        results["native_module"] = {"path": vane._native.__file__, "sha256": digest(Path(vane._native.__file__))}
        results["duckdb_version"] = pycon.execute("PRAGMA version").fetchall()
        results["settings"] = {
            "python": pycon.execute("SELECT current_setting('audio_backend')").fetchone()[0],
            "native": native.execute("SELECT current_setting('audio_backend')").fetchone()[0],
        }
        results["null"] = {
            name: capture(name, lambda con=con: con.execute("SELECT resample(NULL::AUDIOFILE, 16000)").fetchone()[0])
            for name, con in [("sql_python", pycon), ("sql_native", native)]
        }
    else:
        import daft
        from daft.functions import audio_file, audio_metadata, resample

        results["null"] = capture(
            "null",
            lambda: (
                daft.from_pydict({"url": [None]})
                .select(resample(audio_file(daft.col("url").cast(daft.DataType.string())), 16000).alias("wave"))
                .to_pydict()["wave"][0]
            ),
        )

    for case in manifest["cases"]:
        case_id = case["id"]
        path = (root / case["file"]).resolve()
        if digest(path) != case["sha256"]:
            raise ValueError(f"Input changed: {path}")
        file_results = {}
        if engine == "vane":
            file = vane.AudioFile(str(path))
            file_results["metadata_value"] = capture(case_id, lambda: file.metadata(connection=pycon))
            file_results["metadata_sql_python"] = capture(
                case_id, lambda: pycon.execute("SELECT audio_metadata($1)", [file]).fetchone()[0]
            )
            file_results["metadata_sql_native"] = capture(
                case_id, lambda: native.execute("SELECT audio_metadata($1)", [file]).fetchone()[0]
            )
            file_results["metadata_expression_python"] = capture(
                case_id, lambda: pycon.sql("SELECT 1").select(vane.audio_metadata(file)).fetchone()[0]
            )
            file_results["metadata_expression_native"] = capture(
                case_id, lambda: native.sql("SELECT 1").select(vane.audio_metadata(file)).fetchone()[0]
            )
            file_results["decode"] = capture(f"{case_id}_decode", lambda: file.to_numpy(connection=pycon))
        else:
            # Recreate inside each callback so constructor rejection is recorded
            # independently instead of aborting all measurements for that file.
            file_results["metadata_value"] = capture(case_id, lambda: daft.AudioFile(str(path)).metadata())
            file_results["metadata_expression"] = capture(
                case_id,
                lambda: (
                    daft.from_pydict({"url": [str(path)]})
                    .select(audio_metadata(audio_file(daft.col("url"))).alias("metadata"))
                    .to_pydict()["metadata"][0]
                ),
            )
            file_results["decode"] = capture(f"{case_id}_decode", lambda: daft.AudioFile(str(path)).to_numpy())
        results["files"][case_id] = file_results

        for target in case["targets"]:
            key = f"{case_id}_{target}"
            row = {"input": case_id, "target": target}
            if engine == "vane":
                row["value"] = capture(f"{key}_value", lambda: file.resample(target, connection=pycon))
                for name, con in [("sql_python", pycon), ("sql_native", native)]:
                    row[name] = capture(
                        f"{key}_{name}",
                        lambda con=con: con.execute("SELECT resample($1, $2)", [file, target]).fetchone()[0],
                    )
                    row[name + "_expression"] = capture(
                        f"{key}_{name}_expression",
                        lambda con=con: con.sql("SELECT 1").select(vane.resample(file, target)).fetchone()[0],
                    )
                # A connection controls FILE I/O for a value method, but its
                # audio_backend setting does not dispatch that method to native.
                row["value_native_connection"] = capture(
                    f"{key}_value_native_connection", lambda: file.resample(target, connection=native)
                )
                row["native_profile"] = capture(
                    f"{key}_native_profile",
                    lambda: native.execute("SELECT native_audio_resample_profile($1, $2)", [file, target]).fetchone()[
                        0
                    ],
                )
            else:
                row["value"] = capture(f"{key}_value", lambda: daft.AudioFile(str(path)).resample(target))
                row["expression"] = capture(
                    f"{key}_expression",
                    lambda: (
                        daft.from_pydict({"url": [str(path)]})
                        .select(resample(audio_file(daft.col("url")), target).alias("wave"))
                        .to_pydict()["wave"][0]
                    ),
                )
            if "array" in file_results["decode"]:
                decoded = np.load(root / file_results["decode"]["array"], allow_pickle=False)
                if decoded.ndim == 1:
                    decoded = decoded[:, None]
                row["librosa_axis0_reference"] = capture(
                    f"{key}_reference",
                    lambda: librosa.resample(decoded, orig_sr=case["sample_rate"], target_sr=target, axis=0),
                )
            results["resamples"][key] = row
        print(f"{label}: {case_id}", flush=True)
        write_json(out / "results.json", results)
    if engine == "vane":
        pycon.close()
        native.close()


def compare(root: Path, left_label: str, right_label: str) -> None:
    left = json.loads((root / left_label / "results.json").read_text())
    right = json.loads((root / right_label / "results.json").read_text())

    def arrays(a, b):
        if "array" not in a or "array" not in b:
            return {"left_error": a.get("error"), "right_error": b.get("error")}
        x = np.load(root / a["array"], allow_pickle=False)
        y = np.load(root / b["array"], allow_pickle=False)
        result = {
            "left_shape": list(x.shape),
            "right_shape": list(y.shape),
            "dtype_equal": x.dtype == y.dtype,
            "raw_equal": x.dtype == y.dtype and np.array_equal(x, y),
        }
        if x.ndim == 1:
            x = x[:, None]
        if y.ndim == 1:
            y = y[:, None]
        result["shape_equal_after_mono_normalization"] = x.shape == y.shape
        result["equal_after_mono_normalization"] = np.array_equal(x, y)
        result["allclose_1e_6_after_mono_normalization"] = bool(
            result["dtype_equal"] and x.shape == y.shape and np.allclose(x, y, rtol=0, atol=1e-6)
        )
        if x.shape[1:] == y.shape[1:]:
            n = min(len(x), len(y))
            result["overlap_frames"] = n
            if n:
                delta = x[:n] - y[:n]
                result["overlap_max_abs"] = float(np.max(np.abs(delta)))
                result["overlap_rmse"] = float(np.sqrt(np.mean(delta**2)))
                result["overlap_allclose_1e_12"] = bool(np.allclose(x[:n], y[:n], rtol=1e-12, atol=1e-12))
        return result

    comparisons = {"left": left_label, "right": right_label, "resamples": {}, "decode": {}, "metadata": {}}
    for key, row in left["resamples"].items():
        other = right["resamples"][key]
        comparisons["resamples"][key] = {
            "value_vs_sql_python": arrays(row["value"], row["sql_python"]),
            "value_vs_native_connection": arrays(row["value"], row["value_native_connection"]),
            "sql_python_vs_native": arrays(row["sql_python"], row["sql_native"]),
            "sql_python_vs_expression": arrays(row["sql_python"], row["sql_python_expression"]),
            "sql_native_vs_expression": arrays(row["sql_native"], row["sql_native_expression"]),
            "python_vs_daft": arrays(row["value"], other["value"]),
            "native_vs_daft": arrays(row["sql_native"], other["value"]),
            "daft_value_vs_expression": arrays(other["value"], other["expression"]),
            "python_vs_axis0_reference": arrays(row["value"], row.get("librosa_axis0_reference", {})),
        }
    for key, row in left["files"].items():
        comparisons["decode"][key] = arrays(row["decode"], right["files"][key]["decode"])
        a = row["metadata_value"].get("value")
        b = right["files"][key]["metadata_value"].get("value")
        native = row["metadata_sql_native"].get("value")
        python_expression = row.get("metadata_expression_python", {}).get("value")
        native_expression = row.get("metadata_expression_native", {}).get("value")
        comparisons["metadata"][key] = {
            "python_value_equals_sql": a == row["metadata_sql_python"].get("value") if a else None,
            "python_value_equals_native": a == native if a and native else None,
            "python_value_equals_expression": a == python_expression if a and python_expression else None,
            "native_sql_equals_expression": native == native_expression if native and native_expression else None,
            "python_value": a,
            "daft_value": b,
            "daft_expression": right["files"][key]["metadata_expression"].get("value"),
            "native": native,
            "shared_python_daft_fields_equal": all(a[k] == b[k] for k in a.keys() & b.keys()) if a and b else None,
        }
    destination = root / f"comparison-{left_label}-{right_label}.json"
    write_json(destination, comparisons)
    print(destination)
    groups = next(iter(comparisons["resamples"].values())).keys()
    for group in groups:
        rows = [r[group] for r in comparisons["resamples"].values()]
        valid = [r for r in rows if "raw_equal" in r]
        print(
            group,
            json.dumps(
                {
                    "measured": len(valid),
                    "errors": len(rows) - len(valid),
                    "raw_equal": sum(r["raw_equal"] for r in valid),
                    "normalized_equal": sum(r["equal_after_mono_normalization"] for r in valid),
                    "normalized_shape_equal": sum(r["shape_equal_after_mono_normalization"] for r in valid),
                    "normalized_allclose_1e_6": sum(r["allclose_1e_6_after_mono_normalization"] for r in valid),
                }
            ),
        )
    for group in (
        "python_value_equals_sql",
        "python_value_equals_native",
        "python_value_equals_expression",
        "native_sql_equals_expression",
        "shared_python_daft_fields_equal",
    ):
        values = [row[group] for row in comparisons["metadata"].values() if row[group] is not None]
        print("metadata", group, json.dumps({"measured": len(values), "equal": sum(values)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    gen = subparsers.add_parser("generate")
    gen.add_argument("directory", type=Path)
    runner = subparsers.add_parser("run")
    runner.add_argument("directory", type=Path)
    runner.add_argument("--engine", choices=["vane", "daft"], required=True)
    runner.add_argument("--label", required=True)
    runner.add_argument("--native-extension", type=Path)
    comparator = subparsers.add_parser("compare")
    comparator.add_argument("directory", type=Path)
    comparator.add_argument("--left", default="vane")
    comparator.add_argument("--right", default="daft")
    args = parser.parse_args()
    root = args.directory.resolve()
    if args.command == "generate":
        generate(root)
    elif args.command == "run":
        run(root, args.engine, args.label, args.native_extension.resolve() if args.native_extension else None)
    else:
        compare(root, args.left, args.right)


if __name__ == "__main__":
    main()
