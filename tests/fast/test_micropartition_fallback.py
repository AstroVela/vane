import importlib


def test_micropartition_pandas_fallback():
    # Create a MicroPartition from a simple pydict and ensure `sum` is synthesized
    rb = importlib.import_module("duckdb.recordbatch")
    mp = rb.MicroPartition.from_pydict({"a": [0, 1, 2], "b": [0, 10, 20]})
    pdf = mp.to_pandas()
    assert "sum" in pdf.columns
    assert list(pdf["sum"]) == [a + b for a, b in zip(pdf["a"], pdf["b"], strict=False)]
