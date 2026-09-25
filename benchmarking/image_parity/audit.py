# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproducible, local image comparison; run each engine in its installed environment."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path

import numpy as np

MODES = {
    "L": (1, "uint8"),
    "LA": (2, "uint8"),
    "RGB": (3, "uint8"),
    "RGBA": (4, "uint8"),
    "L16": (1, "uint16"),
    "LA16": (2, "uint16"),
    "RGB16": (3, "uint16"),
    "RGBA16": (4, "uint16"),
    "RGB32F": (3, "float32"),
    "RGBA32F": (4, "float32"),
}
HASHES = ["ahash", "dhash", "dhash_vertical", "phash", "phash_simple", "whash", "colorhash", "crop_resistant"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def generate(root):
    import imagecodecs
    import tifffile
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    manifest = {"seed": 20260911, "arrays": [], "files": []}
    rng = np.random.default_rng(manifest["seed"])

    def save_array(name, mode, array):
        path = inputs / (name + ".npy")
        np.save(path, array, allow_pickle=False)
        manifest["arrays"].append(
            {"id": name, "mode": mode, "file": str(path.relative_to(root)), "sha256": digest(path.read_bytes())}
        )

    def save_file(name, data):
        path = inputs / name
        path.write_bytes(data)
        manifest["files"].append({"id": name, "file": str(path.relative_to(root)), "sha256": digest(data)})

    for mode, (channels, dtype) in MODES.items():
        maximum = 1 if dtype == "float32" else np.iinfo(dtype).max
        for pattern, h, w in [("random", 17, 23), ("ramp", 9, 13), ("checker", 8, 8), ("flat", 9, 9)]:
            if pattern == "random":
                a = rng.random((h, w, channels))
            elif pattern == "ramp":
                a = np.linspace(0, 1, h * w * channels).reshape(h, w, channels)
            elif pattern == "checker":
                a = np.repeat((np.indices((h, w)).sum(axis=0) % 2)[:, :, None], channels, axis=2)
            else:
                a = np.full((h, w, channels), 0.5)
            a = (a * maximum).astype(dtype)
            save_array(f"{mode}_{pattern}", mode, a)
            if pattern != "random":
                continue
            buffer = io.BytesIO()
            tifffile.imwrite(
                buffer,
                a[:, :, 0] if channels == 1 else a,
                photometric="minisblack" if channels < 3 else "rgb",
                extrasamples=["unassalpha"] if channels in (2, 4) else None,
                metadata=None,
            )
            save_file(f"{mode}.tiff", buffer.getvalue())
            if dtype == "uint16":
                save_file(f"{mode}.png", imagecodecs.png_encode(a[:, :, 0] if channels == 1 else a))
            elif dtype == "uint8":
                image = Image.fromarray(a[:, :, 0] if channels == 1 else a)
                for fmt in ["PNG"] + (["JPEG", "GIF", "BMP"] if mode in ("L", "RGB") else []):
                    buffer = io.BytesIO()
                    image.save(buffer, format=fmt)
                    save_file(f"{mode}.{fmt.lower()}", buffer.getvalue())

    save_array("downsample_impulse", "L", np.array([[[0], [0], [0], [240]]], dtype=np.uint8))
    save_array("transparent_edge", "RGBA", np.array([[[255, 0, 0, 255], [0, 0, 255, 0]]], dtype=np.uint8))
    save_array("primaries", "RGB", np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]], dtype=np.uint8))
    rgb = Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
    for name, image, fmt, options in [
        ("palette.png", rgb.quantize(colors=16), "PNG", {}),
        ("palette_alpha.png", rgb.quantize(colors=16), "PNG", {"transparency": 0}),
        ("onebit.png", rgb.convert("1"), "PNG", {}),
        ("cmyk.jpeg", rgb.convert("CMYK"), "JPEG", {}),
        ("oriented.jpeg", rgb, "JPEG", {"exif": Image.Exif()}),
        ("extra.webp", rgb, "WEBP", {"lossless": True}),
        ("extra.ico", rgb, "ICO", {}),
        (
            "animated.gif",
            rgb,
            "GIF",
            {"save_all": True, "append_images": [rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)]},
        ),
    ]:
        if name == "oriented.jpeg":
            options["exif"][274] = 6
        buffer = io.BytesIO()
        image.save(buffer, format=fmt, **options)
        save_file(name, buffer.getvalue())
    save_file("corrupt.bin", b"not an image")
    write_json(root / "manifest.json", manifest)
    print(f"Generated {len(manifest['arrays'])} raw arrays and {len(manifest['files'])} encoded files", flush=True)


def encode_value(value, directory, key):
    if isinstance(value, np.ndarray):
        path = directory / (digest(key.encode()) + ".npy")
        np.save(path, value, allow_pickle=False)
        return {
            "kind": "array",
            "path": path.name,
            "sha256": digest(path.read_bytes()),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sample": value.reshape(-1)[:16].tolist(),
        }
    if isinstance(value, bytes):
        path = directory / (digest(key.encode()) + ".bin")
        path.write_bytes(value)
        return {
            "kind": "bytes",
            "path": path.name,
            "sha256": digest(value),
            "length": len(value),
            "hex": value.hex() if len(value) <= 80 else None,
        }
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    return {"kind": "value", "value": value}


def reference_decode(encoded):
    import imagecodecs
    import tifffile
    from PIL import Image

    if encoded[:4] in (b"II*\0", b"MM\0*", b"II+\0", b"MM\0+"):
        a = tifffile.imread(io.BytesIO(encoded))
    elif encoded[:8] == b"\x89PNG\r\n\x1a\n" and encoded[24] == 16:
        a = imagecodecs.png_decode(encoded)
    else:
        with Image.open(io.BytesIO(encoded)) as image:
            if image.mode == "P":
                image = image.convert("RGBA" if "transparency" in image.info else "RGB")
            a = np.asarray(image).copy()
    return a[:, :, None] if a.ndim == 2 else a


def run(root, engine, artifact):
    manifest = json.loads((root / "manifest.json").read_bytes())
    out = root / engine
    out.mkdir(exist_ok=True)
    records = {}
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pillow", "tifffile", "imagecodecs")}
    if engine.startswith("vane"):
        import vane

        versions["vane-ai"] = importlib.metadata.version("vane-ai")
        con = vane.connect(config={"allow_unsigned_extensions": "true"})
        if engine == "vane-native":
            con.load_extension(str(artifact.resolve()))
            con.execute("SET image_backend='native'")
        else:
            con.execute("SET image_backend='python'")
        versions["engine"] = con.execute("PRAGMA version").fetchone()
        versions["backend"] = con.execute("SELECT current_setting('image_backend')").fetchone()[0]

        def raw_table(a, mode):
            return vane.Value(a, vane.image_type(mode))

        def pixel_op(value, op, arg):
            if op == "resize":
                return con.execute("SELECT resize($1,$2,$3)", [value, *arg]).fetchone()[0]
            if op == "crop":
                return con.execute("SELECT crop($1,$2)", [value, arg]).fetchone()[0]
            if op == "convert":
                return con.execute("SELECT convert_image($1,$2)", [value, arg]).fetchone()[0]
            if op == "tensor":
                return con.execute("SELECT image_to_tensor($1)", [value]).fetchone()[0]
            if op == "hash":
                opts = arg if isinstance(arg, dict) else {"method": arg}
                return con.execute(
                    "SELECT image_hash($1,method=>$2,hash_size=>$3,binbits=>$4,segments=>$5)",
                    [value, opts["method"], opts.get("hash_size", 8), opts.get("binbits", 3), opts.get("segments", 3)],
                ).fetchone()[0]
            return con.execute("SELECT encode_image($1,$2)", [value, arg]).fetchone()[0]

        def file_op(path, data, op, mode, on_error="raise"):
            if op == "metadata":
                return con.execute("SELECT image_file_metadata(image_file($1))", [str(path)]).fetchone()[0]
            if op == "metadata_value":
                return vane.ImageFile(str(path)).metadata()
            if op == "decode_file":
                return con.execute(
                    "SELECT decode_image_file(image_file($1),$2,$3)", [str(path), mode, on_error]
                ).fetchone()[0]
            return con.execute("SELECT decode_image($1,$2,$3)", [data, on_error, mode]).fetchone()[0]
    else:
        import daft
        from daft import col
        from daft import functions as f
        from daft.recordbatch import MicroPartition

        versions["daft"] = daft.__version__

        def raw_table(a, mode):
            s = daft.Series.from_pylist([a], dtype=daft.DataType.python()).cast(daft.DataType.image(mode))
            return MicroPartition.from_pydict({"x": s})

        def evaluate(table, expression):
            return table.eval_expression_list([expression.alias("out")]).to_pydict()["out"][0]

        def pixel_op(value, op, arg):
            x = col("x")
            if op == "resize":
                expression = f.resize(x, *arg)
            elif op == "crop":
                expression = f.crop(x, tuple(arg))
            elif op == "convert":
                expression = f.convert_image(x, arg)
            elif op == "tensor":
                expression = f.image_to_tensor(x)
            elif op == "hash":
                opts = arg if isinstance(arg, dict) else {"method": arg}
                expression = f.image_hash(x, **opts)
            else:
                expression = f.encode_image(x, arg)
            return evaluate(value, expression)

        def file_op(path, data, op, mode, on_error="raise"):
            if op == "metadata_value":
                value = daft.ImageFile(str(path)).metadata()
                return dict(value) if isinstance(value, dict) else vars(value)
            table = MicroPartition.from_pydict({"x": [str(path) if op in ("metadata", "decode_file") else data]})
            x = f.image_file(col("x")) if op in ("metadata", "decode_file") else col("x")
            if op == "metadata":
                expression = f.image_file_metadata(x)
            elif op == "decode_file":
                expression = f.decode_image_file(x, mode=mode, on_error=on_error)
            else:
                expression = f.decode_image(x, mode=mode, on_error=on_error)
            return evaluate(table, expression)

    def capture(key, function):
        nonlocal con
        try:
            value = function()
            records[key] = {"status": "ok", **encode_value(value, out, key)}
            return value
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            records[key] = {"status": "error", "type": type(error).__name__, "message": str(error)[:2000]}
            if engine.startswith("vane"):
                # Some failed statements leave a transaction aborted. Each
                # independent audit case must start with usable engine state.
                con.close()
                con = vane.connect(config={"allow_unsigned_extensions": "true"})
                if engine == "vane-native":
                    con.load_extension(str(artifact.resolve()))
                    con.execute("SET image_backend='native'")
            return None

    for item in manifest["arrays"]:
        data = (root / item["file"]).read_bytes()
        assert digest(data) == item["sha256"]
        a = np.load(io.BytesIO(data), allow_pickle=False)
        value = capture(item["id"] + "/input", lambda: raw_table(a, item["mode"]))
        # Input construction is not itself a portable serializable output.
        if records[item["id"] + "/input"]["status"] == "ok":
            records[item["id"] + "/input"] = {"status": "ok", "kind": "value", "value": item["mode"]}
        if value is None:
            continue
        ops = [
            ("resize", [a.shape[1], a.shape[0]]),
            ("resize", [1, 1]),
            ("resize", [7, 5]),
            ("resize", [31, 19]),
            ("crop", [1, 1, 3, 2]),
            ("crop", [a.shape[1] - 1, a.shape[0] - 1, 4, 3]),
            ("crop", [-1, -1, 4, 3]),
            *[("convert", m) for m in MODES],
            ("tensor", None),
            *[("hash", h) for h in HASHES],
            *[("encode", fmt) for fmt in ("PNG", "JPEG", "TIFF", "GIF", "BMP")],
        ]
        if item["id"] == "RGB_random":
            ops += [("hash", {"method": method, "hash_size": 3}) for method in HASHES]
            ops += [("hash", {"method": "colorhash", "binbits": bits}) for bits in (1, 2, 4, 8)]
            ops += [("hash", {"method": "crop_resistant", "hash_size": 4, "segments": 2})]
        for op, arg in ops:
            key = item["id"] + "/" + op + "/" + json.dumps(arg, separators=(",", ":"))
            result = capture(key, lambda: pixel_op(value, op, arg))
            if op == "encode" and isinstance(result, bytes):
                capture(key + "/decoded", lambda: reference_decode(result))
        print(engine, "raw", item["id"], flush=True)

    for item in manifest["files"]:
        data = (root / item["file"]).read_bytes()
        assert digest(data) == item["sha256"]
        # Every file decoder consumes a private, checksum-verified input copy.
        path = out / ("input_" + item["id"])
        path.write_bytes(data)
        ops = [("metadata", None), ("metadata_value", None)] + [
            (op, mode) for op in ("decode", "decode_file") for mode in (None, "RGB", "RGBA")
        ]
        for op, mode in ops:
            key = item["id"] + "/" + op + "/" + str(mode)
            capture(key, lambda: file_op(path, data, op, mode))
        for op in ("decode", "decode_file"):
            capture(item["id"] + "/" + op + "/on_error_null", lambda: file_op(path, data, op, "RGB", "null"))
        print(engine, "file", item["id"], flush=True)
    capture(
        "missing_file/decode/on_error_null", lambda: file_op(out / "missing.png", b"", "decode_file", "RGB", "null")
    )
    result = {
        "engine": engine,
        "versions": versions,
        "manifest": manifest,
        "manifest_sha256": digest(canonical(manifest)),
        "audit_script_sha256": digest(Path(__file__).read_bytes()),
        "records": records,
    }
    write_json(out / "results.json", result)
    print(engine, len(records), "results saved", flush=True)


def compare(root):
    runs = {
        name: json.loads((root / name / "results.json").read_bytes()) for name in ("vane-python", "vane-native", "daft")
    }
    assert len({r["manifest_sha256"] for r in runs.values()}) == 1
    assert len({r["audit_script_sha256"] for r in runs.values()}) == 1
    output = {}
    for left, right in [("vane-python", "vane-native"), ("vane-python", "daft"), ("vane-native", "daft")]:
        rows = []
        for key in sorted(runs[left]["records"].keys() | runs[right]["records"].keys()):
            a, b = runs[left]["records"].get(key), runs[right]["records"].get(key)
            row = {"key": key}
            if a is None or b is None:
                row["result"] = "input_unsupported"
            elif a["status"] != "ok" or b["status"] != "ok":
                row["result"] = "both_error" if a["status"] == b["status"] else "success_error"
            elif a["kind"] == b["kind"] == "array":
                arrays = []
                for engine, record in [(left, a), (right, b)]:
                    data = (root / engine / record["path"]).read_bytes()
                    assert digest(data) == record["sha256"]
                    array = np.load(io.BytesIO(data), allow_pickle=False)
                    assert list(array.shape) == record["shape"] and str(array.dtype) == record["dtype"]
                    arrays.append(array)
                aa, bb = arrays
                if aa.shape != bb.shape or aa.dtype != bb.dtype:
                    row["result"] = "layout_diff"
                    row["layouts"] = [[list(x.shape), str(x.dtype)] for x in arrays]
                elif np.array_equal(aa, bb):
                    row["result"] = "exact"
                else:
                    delta = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
                    row.update(
                        result="pixels_diff",
                        max_abs=float(delta.max()),
                        mean_abs=float(delta.mean()),
                        different_values=int(np.count_nonzero(delta)),
                    )
            elif a["kind"] == b["kind"] == "bytes":
                for engine, record in [(left, a), (right, b)]:
                    assert digest((root / engine / record["path"]).read_bytes()) == record["sha256"]
                row["result"] = "exact" if a["sha256"] == b["sha256"] else "bytes_diff"
                if "/hash/" in key:
                    aa, bb = bytes.fromhex(a["hex"]), bytes.fromhex(b["hex"])
                    if len(aa) == len(bb):
                        row["hamming_bits"] = sum((x ^ y).bit_count() for x, y in zip(aa, bb, strict=True))
                    else:
                        row["lengths"] = [len(aa), len(bb)]
            else:
                row["result"] = "exact" if a == b else "value_diff"
            rows.append(row)
        output[left + "__" + right] = rows
    write_json(root / "comparison.json", output)
    for pair, rows in output.items():
        from collections import Counter

        print(pair, dict(Counter(row["result"] for row in rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["generate", "run", "compare"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--engine", choices=["vane-python", "vane-native", "daft"])
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("VANE_RUNNER", "local-fast")
    root = args.root.resolve()
    if args.command == "generate":
        generate(root)
    elif args.command == "run":
        run(root, args.engine, args.artifact)
    else:
        compare(root)
