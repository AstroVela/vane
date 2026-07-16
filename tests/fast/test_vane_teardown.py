import duckdb


def _get_vane_module():
    return getattr(duckdb, "vane_runners_cpp", None)


def test_teardown_runner():
    vane_mod = _get_vane_module()
    assert vane_mod is not None, "vane_runners_cpp module not available"

    # Suppress Ray's FutureWarnings during init which are not relevant to the test
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning)

    # Set runner to Ray (no-op if already set in environment)
    vane_mod.set_runner_ray(None, True)

    # Runner should exist now
    r = vane_mod.get_runner()
    assert r is not None

    # Call teardown helper and verify runner cleared
    vane_mod.teardown_runner()
    assert vane_mod.get_runner() is None
