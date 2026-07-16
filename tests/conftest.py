import faulthandler
import os
import signal
import sys
import warnings
from importlib import import_module
from pathlib import Path

import pytest

import duckdb

try:
    # need to ignore warnings that might be thrown deep inside pandas's import tree (from dateutil in this case)
    warnings.simplefilter(action="ignore", category=DeprecationWarning)
    pandas = import_module("pandas")
    warnings.resetwarnings()
except ImportError:
    pandas = None


# Version-aware helpers for Pandas 2.x vs 3.0 compatibility
def _get_pandas_ge_3():
    if pandas is None:
        return False
    from packaging.version import Version

    return Version(pandas.__version__) >= Version("3.0.0")


PANDAS_GE_3 = _get_pandas_ge_3()


@pytest.fixture(autouse=True)
def default_vane_runner_for_tests(monkeypatch):
    """Keep general DuckDB tests local; default-Ray tests explicitly clear this override."""
    if "VANE_RUNNER" not in os.environ:
        monkeypatch.setenv("VANE_RUNNER", "local-fast")


def is_string_dtype(dtype):
    """Check if a dtype is a string dtype (works across Pandas 2.x and 3.0).

    Uses pd.api.types.is_string_dtype() which handles:
    - Pandas 2.x: object dtype for strings
    - Pandas 3.0+: str (StringDtype) for strings
    """
    return pandas.api.types.is_string_dtype(dtype)


def import_pandas():
    if pandas:
        return pandas
    else:
        pytest.skip("Couldn't import pandas")


# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
# https://stackoverflow.com/a/47700320
def pytest_addoption(parser):
    parser.addoption("--skiplist", action="append", nargs="+", type=str, help="skip listed tests")


def pytest_collection_modifyitems(config, items):
    tests_to_skip = config.getoption("--skiplist")
    if not tests_to_skip:
        # --skiplist not given in cli, therefore move on
        return

    # Combine all the lists into one
    skipped_tests = []
    for item in tests_to_skip:
        skipped_tests.extend(item)

    skip_listed = pytest.mark.skip(reason="included in --skiplist")
    for item in items:
        if item.name in skipped_tests:
            # test is named specifically
            item.add_marker(skip_listed)
        elif item.parent is not None and item.parent.name in skipped_tests:
            # the class is named specifically
            item.add_marker(skip_listed)


@pytest.fixture
def duckdb_empty_cursor():
    connection = duckdb.connect("")
    cursor = connection.cursor()
    return cursor


def getTimeSeriesData(nper=None, freq="B"):
    import string
    from datetime import datetime

    import numpy as np
    from pandas import DatetimeIndex, Series, bdate_range

    _N = 30
    _K = 4

    def getCols(k) -> str:
        return string.ascii_uppercase[:k]

    def makeDateIndex(k: int = 10, freq="B", name=None, **kwargs) -> DatetimeIndex:
        dt = datetime(2000, 1, 1)
        dr = bdate_range(dt, periods=k, freq=freq, name=name)
        return DatetimeIndex(dr, name=name, **kwargs)

    def makeTimeSeries(nper=None, freq="B", name=None) -> Series:
        if nper is None:
            nper = _N
        return Series(np.random.randn(nper), index=makeDateIndex(nper, freq=freq), name=name)

    return {c: makeTimeSeries(nper, freq) for c in getCols(_K)}


def pandas_2_or_higher():
    from packaging.version import Version

    return Version(import_pandas().__version__) >= Version("2.0.0")


@pytest.fixture
def require():
    def _require(extension_name, db_name="") -> duckdb.DuckDBPyConnection | None:
        # Paths to search for extensions

        build = Path(__file__).parent.parent / "build"
        extension = "extension/*/*.duckdb_extension"

        extension_search_patterns = [
            build / "release" / extension,
            build / "debug" / extension,
        ]

        # DUCKDB_PYTHON_TEST_EXTENSION_PATH can be used to add a path for the extension test to search for extensions
        if "DUCKDB_PYTHON_TEST_EXTENSION_PATH" in os.environ:
            env_extension_path = os.getenv("DUCKDB_PYTHON_TEST_EXTENSION_PATH")
            env_extension_path = env_extension_path.rstrip("/")
            extension_search_patterns.append(env_extension_path + "/*/*.duckdb_extension")
            extension_search_patterns.append(env_extension_path + "/*.duckdb_extension")

        extension_paths_found = []
        for pattern in extension_search_patterns:
            extension_paths_found.extend(list(Path(pattern).resolve().glob("*")))

        for path in extension_paths_found:
            print(path)
            if path.endswith(extension_name + ".duckdb_extension"):
                conn = duckdb.connect(db_name, config={"allow_unsigned_extensions": "true"})
                conn.execute(f"LOAD '{path}'")
                return conn
        pytest.skip(f"could not load {extension_name}")

    return _require


# By making the scope 'function' we ensure that a new connection gets created for every function that uses the fixture
@pytest.fixture
def spark():
    if not hasattr(spark, "session"):
        # Cache the import
        from spark_namespace.sql import SparkSession as session

        spark.session = session

    return spark.session.builder.appName("pyspark").getOrCreate()


@pytest.fixture
def duckdb_cursor():
    connection = duckdb.connect("")
    yield connection
    connection.close()


@pytest.fixture
def integers(duckdb_cursor):
    cursor = duckdb_cursor
    cursor.execute("CREATE TABLE integers (i integer)")
    cursor.execute(
        """
        INSERT INTO integers VALUES
            (0),
            (1),
            (2),
            (3),
            (4),
            (5),
            (6),
            (7),
            (8),
            (9),
            (NULL)
    """
    )
    yield
    cursor.execute("drop table integers")


@pytest.fixture
def timestamps(duckdb_cursor):
    cursor = duckdb_cursor
    cursor.execute("CREATE TABLE timestamps (t timestamp)")
    cursor.execute("INSERT INTO timestamps VALUES ('1992-10-03 18:34:45'), ('2010-01-01 00:00:01'), (NULL)")
    yield
    cursor.execute("drop table timestamps")


# Per-test timeout (autouse)
TEST_TIMEOUT = int(os.getenv("TEST_TIMEOUT", "300"))


def _alarm_handler(_signum, _frame):
    sys.stderr.write(f"\n=== TEST TIMEOUT ({TEST_TIMEOUT}s) - dumping all thread tracebacks ===\n")
    try:
        faulthandler.dump_traceback(all_threads=True, file=sys.stderr)
    except Exception:
        sys.stderr.write("(failed to dump tracebacks)\n")
    raise TimeoutError(f"Test exceeded timeout of {TEST_TIMEOUT} seconds")


@pytest.fixture(autouse=True)
def test_timeout():
    prev_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(TEST_TIMEOUT)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


@pytest.fixture
def ray_local():
    try:
        import ray
    except Exception:
        pytest.skip("ray not installed")

    def current_ray_node():
        try:
            from ray._private import worker as ray_worker

            return getattr(ray_worker.global_worker, "node", None)
        except Exception:
            return None

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Tip: In future versions of Ray")
        warning_filter = r"ignore:\s*Prefer using device seq_lens directly.*:DeprecationWarning"
        os.environ["PYTHONWARNINGS"] = warning_filter
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        pythonpath = os.environ.get("PYTHONPATH", "")
        try:
            import _duckdb as duckdb_ext

            duckdb_pkg_root = os.path.dirname(duckdb.__file__)
            duckdb_parent = os.path.dirname(duckdb_pkg_root)
            duckdb_ext_root = os.path.dirname(duckdb_ext.__file__)
            pythonpath_entries = []
            if duckdb_ext_root:
                pythonpath_entries.append(duckdb_ext_root)
            if duckdb_parent:
                pythonpath_entries.append(duckdb_parent)
            if pythonpath:
                pythonpath_entries.append(pythonpath)
            pythonpath = os.pathsep.join(dict.fromkeys(pythonpath_entries))
            if pythonpath:
                os.environ["PYTHONPATH"] = pythonpath
        except Exception:
            pythonpath = os.environ.get("PYTHONPATH", "")
        env_vars = {
            "PYTHONWARNINGS": os.environ.get("PYTHONWARNINGS", ""),
            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": os.environ.get("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0"),
        }
        if pythonpath:
            env_vars["PYTHONPATH"] = pythonpath
        ray.init(
            ignore_reinit_error=True,
            logging_level="info",
            log_to_driver=True,
            object_store_memory=int(
                os.environ.get("VANE_TEST_RAY_OBJECT_STORE_BYTES", str(1024**3))
            ),
            runtime_env={"env_vars": env_vars},
        )
    try:
        yield
    finally:
        try:
            vane_mod = getattr(duckdb, "vane_runners_cpp", None)
            if vane_mod is not None and hasattr(vane_mod, "teardown_runner"):
                vane_mod.teardown_runner()
        except Exception as e:
            print(f"WARNING: Exception during Vane runner teardown: {e}", file=sys.stderr)
        ray_node = current_ray_node()
        prev_handler = None
        alarm_set = False
        try:
            import signal

            def timeout_handler(_signum, _frame):
                raise TimeoutError("ray.shutdown() timed out")

            if hasattr(signal, "SIGALRM"):
                prev_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                alarm_set = True

            ray.shutdown()
        except Exception as e:
            print(f"WARNING: Exception during ray.shutdown(): {e}", file=sys.stderr)
        finally:
            if ray_node is not None:
                try:
                    ray_node.kill_all_processes(
                        check_alive=False,
                        allow_graceful=False,
                        wait=True,
                    )
                except Exception as e:
                    print(f"WARNING: Exception during Ray node cleanup: {e}", file=sys.stderr)
            try:
                from duckdb.runners.ray import driver as ray_driver

                ray_driver.shutdown_background_event_loop()
            except Exception:
                pass
            if alarm_set:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prev_handler)


def pytest_configure(config):
    """Enable a faulthandler watchdog as early as pytest initializes so we can
    capture hangs that occur during collection/import (before test fixtures run).
    The watchdog will dump thread tracebacks to stderr after TEST_TIMEOUT seconds.
    """
    try:
        faulthandler.enable()
        timeout = int(os.getenv("TEST_TIMEOUT", "300"))
        faulthandler.dump_traceback_later(timeout, repeat=False)
        # record that we scheduled a dump so we can cancel it in pytest_unconfigure
        config._duckdb_faulthandler_dump_scheduled = True
    except Exception:
        # best-effort; don't fail pytest initialization if this doesn't work
        pass


def pytest_unconfigure(config):
    try:
        if getattr(config, "_duckdb_faulthandler_dump_scheduled", False):
            faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass
