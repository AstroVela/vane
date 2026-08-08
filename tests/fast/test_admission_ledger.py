# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from vane.runners.ray.admission_ledger import BoundedReplayMap, BoundedSet


def test_bounded_set_retains_only_the_newest_exact_identities():
    identities = BoundedSet[str](capacity=2)
    identities.add("first")
    identities.add("second")
    identities.add("third")

    assert "first" not in identities
    assert list(identities) == ["second", "third"]

    identities.add("second")
    identities.add("fourth")
    assert list(identities) == ["second", "fourth"]


def test_bounded_replay_map_supports_dict_lookup_and_predicate_cleanup():
    tombstones = BoundedReplayMap[str, dict](capacity=2)
    tombstones["request-1"] = {"query_id": "q1"}
    tombstones["request-2"] = {"query_id": "q2"}
    tombstones["request-3"] = {"query_id": "q1"}

    assert tombstones.get("request-1") is None
    assert list(tombstones) == ["request-2", "request-3"]

    tombstones.discard_where(lambda _key, value: value["query_id"] == "q1")
    assert dict(tombstones.items()) == {"request-2": {"query_id": "q2"}}
