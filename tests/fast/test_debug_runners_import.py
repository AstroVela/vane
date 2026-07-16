def test_print_runners_file():
    import duckdb.runners as r

    print("DUCKDB.RUNNERS FILE:", getattr(r, "__file__", None))
    # ensure test always passes; this is a debug aid
    assert True
