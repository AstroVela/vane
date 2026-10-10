# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import time
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.real_ray, pytest.mark.ray_cluster_owner]


@pytest.mark.parametrize("constructor_failure", [False, True])
def test_failed_initializer_releases_actors_waiting_for_the_same_cpu(monkeypatch, constructor_failure):
    import ray

    from vane.ai import summarize_error
    from vane.execution.udf_ray import wait_for_first_actor_pool_ready

    ray.shutdown()
    ray.init(num_cpus=1, include_dashboard=False)
    monkeypatch.setenv("VANE_RAY_ACTOR_INIT_TIMEOUT_S", "8")
    try:

        @ray.remote(num_cpus=1)
        class Initializer:
            def __init__(self, fail):
                if fail:
                    self.initialize()

            def initialize(self):
                from vane.ai.provider import _safe_provider_execution_error

                raise _safe_provider_execution_error(
                    "sglang",
                    "fixture",
                    "initialization",
                    TypeError("ServerArgs.__init__() got an unexpected keyword argument 'max_model_len'"),
                )

        first = Initializer.remote(constructor_failure)
        # Ensure the first actor occupies the only CPU before scheduling another.
        first_ref = first.initialize.remote()
        ray.wait([first_ref], timeout=30)
        second = Initializer.remote(constructor_failure)
        pool = SimpleNamespace(
            actors=[first, second],
            actor_node_ids=["", ""],
            _init_refs=[first_ref, second.initialize.remote()],
            _confirmed_ready=set(),
            _owns_actors=True,
        )
        started = time.monotonic()
        with pytest.raises(RuntimeError) as caught:
            wait_for_first_actor_pool_ready(pool)
        assert time.monotonic() - started < 4
        assert "max_model_len" in summarize_error(caught.value)
        assert not pool.actors

        @ray.remote(num_cpus=1)
        def available():
            return True

        assert ray.get(available.remote(), timeout=15)
    finally:
        ray.shutdown()
