# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Small API boundary probes, separate from the generated pixel corpus."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--engine", choices=["vane-python", "vane-native", "daft"], required=True)
parser.add_argument("--artifact", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
os.environ.setdefault("VANE_RUNNER", "local-fast")
pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
results = {}


def record(key, function):
    global con
    try:
        value = function()
        results[key] = (
            {"status": "ok", "shape": list(value.shape), "dtype": str(value.dtype), "values": value.tolist()}
            if isinstance(value, np.ndarray)
            else {"status": "ok", "python_type": type(value).__name__, "value": value}
        )
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        results[key] = {"status": "error", "type": type(error).__name__, "message": str(error)[:1000]}
        if args.engine.startswith("vane"):
            con.close()
            con = vane.connect(config={"allow_unsigned_extensions": "true"})
            if args.engine == "vane-native":
                con.load_extension(str(args.artifact.resolve()))
                con.execute("SET image_backend='native'")


if args.engine.startswith("vane"):
    import vane

    con = vane.connect(config={"allow_unsigned_extensions": "true"})
    if args.engine == "vane-native":
        con.load_extension(str(args.artifact.resolve()))
        con.execute("SET image_backend='native'")
    value = vane.Value(pixels, vane.image_type("RGB"))
    record("null_metadata", lambda: con.execute("SELECT image_file_metadata(NULL::IMAGEFILE)").fetchone()[0])
    record(
        "generic_tensor",
        lambda: con.execute("SELECT image_to_tensor($1)", [vane.Value(pixels, vane.image_type())]).fetchone()[0],
    )
    record(
        "fixed_tensor",
        lambda: con.execute(
            "SELECT image_to_tensor($1)", [vane.Value(pixels, vane.image_type("RGB", 2, 3))]
        ).fetchone()[0],
    )
    for bbox in ([2, 1, 4, 3], [-1, -1, 4, 3], [99, 99, 4, 3], [0, 0, 0, 1], [0, 0, 1.5, 1]):
        record(f"crop/{bbox}", lambda: con.execute("SELECT crop($1,$2)", [value, bbox]).fetchone()[0])
    record("resize_zero", lambda: con.execute("SELECT resize($1,0,1)", [value]).fetchone()[0])
    for op in (
        "resize($1,2,3)",
        "crop($1,[0,0,1,1])",
        "image_hash($1)",
        "encode_image($1,'PNG')",
        "image_to_tensor($1)",
    ):
        record(
            "null/" + op, lambda: con.execute("SELECT " + op, [vane.Value(None, vane.image_type("RGB"))]).fetchone()[0]
        )
else:
    import daft
    from daft import functions as f
    from daft.recordbatch import MicroPartition

    def table(dtype, value=pixels):
        return MicroPartition.from_pydict(
            {"x": daft.Series.from_pylist([value], dtype=daft.DataType.python()).cast(dtype)}
        )

    def evaluate(t, expression):
        return t.eval_expression_list([expression.alias("out")]).to_pydict()["out"][0]

    t = table(daft.DataType.image("RGB"))
    x = daft.col("x")
    null_path = MicroPartition.from_pydict({"x": daft.Series.from_pylist([None], dtype=daft.DataType.string())})
    record("null_metadata", lambda: evaluate(null_path, f.image_file_metadata(f.image_file(x))))
    record("generic_tensor", lambda: evaluate(table(daft.DataType.image()), f.image_to_tensor(x)))
    record("fixed_tensor", lambda: evaluate(table(daft.DataType.image("RGB", 2, 3)), f.image_to_tensor(x)))
    for bbox in ([2, 1, 4, 3], [-1, -1, 4, 3], [99, 99, 4, 3], [0, 0, 0, 1], [0, 0, 1.5, 1]):
        record(f"crop/{bbox}", lambda: evaluate(t, f.crop(x, tuple(bbox))))
    record("resize_zero", lambda: evaluate(t, f.resize(x, 0, 1)))
    nt = table(daft.DataType.image("RGB"), None)
    for label, expression in [
        ("resize($1,2,3)", f.resize(x, 2, 3)),
        ("crop($1,[0,0,1,1])", f.crop(x, (0, 0, 1, 1))),
        ("image_hash($1)", f.image_hash(x)),
        ("encode_image($1,'PNG')", f.encode_image(x, "PNG")),
        ("image_to_tensor($1)", f.image_to_tensor(x)),
    ]:
        record("null/" + label, lambda: evaluate(nt, expression))

args.output.write_text(json.dumps({"engine": args.engine, "records": results}, indent=2) + "\n")
print(args.engine, len(results), "contract probes saved")
