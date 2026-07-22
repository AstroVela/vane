# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free wire contract for native vLLM option payloads.

The native vLLM path crosses two otherwise separate layers:

* ``vane.ai.providers.vllm`` builds the constant passed to DuckDB's native
  ``vllm(prompt, model, options)`` expression.
* ``duckdb.execution.vllm`` decodes that constant immediately before creating
  a local vLLM executor or a Ray actor pool.

Keeping the field names here gives both layers one protocol definition without
making the execution layer import Vane's provider/planning classes.

Secret-bearing options use a DuckDB STRUCT with this logical shape::

    {
        "__vane_vllm_payload_version": 1,
        "__vane_vllm_public_options_json": '{"engine_args":{"hf_token":{"__vane_vllm_secret_ref":0}}}',
        "__vane_vllm_secret_payload": b'{"payload_version":1,"values":["..."]}',
    }

The public JSON preserves the option tree but substitutes each sealed value
with an integer reference into the secret payload. The secret payload is a BLOB
so DuckDB does not render its contents as structured plan metadata. It contains
strict JSON data, never pickle or executable Python objects.

Options without sealed values keep the legacy plain-JSON argument, so this
envelope does not change existing non-secret plans. During distributed planning,
C++ may add ``use_ray``, ``ray_worker_only``, and ``ray_actor_pool_name`` beside
the three envelope fields. Those are separately whitelisted execution-routing
fields rather than part of the secret-reference protocol.

``opaque`` does not mean encrypted: callers with access to raw serialized plan
bytes can still recover the payload. This protocol prevents accidental exposure
through EXPLAIN, repr, and ordinary structured logging; environment-based
credentials remain the stronger choice when secrets must not enter a plan.

Incompatible envelope or reference changes must increment
``_NATIVE_OPTIONS_PAYLOAD_VERSION`` and retain an explicit decoder for any older
version that must remain supported.
"""

from __future__ import annotations

import json
from typing import Any

# Version of both the outer envelope and its secret-values payload.
_NATIVE_OPTIONS_PAYLOAD_VERSION = 1

# Keys stored in the DuckDB STRUCT that crosses the native plan boundary.
_NATIVE_OPTIONS_VERSION_KEY = "__vane_vllm_payload_version"
_NATIVE_OPTIONS_PUBLIC_KEY = "__vane_vllm_public_options_json"
_NATIVE_OPTIONS_SECRET_KEY = "__vane_vllm_secret_payload"

# Required top-level fields that identify a value as an opaque options
# envelope. Keeping the set here makes envelope detection and validation use
# exactly the same schema as the encoder.
_NATIVE_OPTIONS_ENVELOPE_KEYS = frozenset(
    {
        _NATIVE_OPTIONS_VERSION_KEY,
        _NATIVE_OPTIONS_PUBLIC_KEY,
        _NATIVE_OPTIONS_SECRET_KEY,
    }
)

# Extra top-level fields that C++ distributed-plan collection may inject into
# an existing envelope. They are non-secret execution-routing metadata. The
# runtime validates them at the envelope boundary, then overlays them onto the
# decoded public options before choosing Local versus Ray execution.
_NATIVE_OPTIONS_DISTRIBUTED_ROUTING_KEYS = frozenset(
    {
        "use_ray",
        "ray_worker_only",
        "ray_actor_pool_name",
    }
)

# Exact one-field marker used inside public JSON in place of a sealed value.
_NATIVE_SECRET_REF_KEY = "__vane_vllm_secret_ref"

# Required fields inside the strict-JSON secret BLOB. References in public JSON
# index into the list stored under ``values``; the nested version is checked
# independently so a future envelope can evolve its secret representation
# without silently misreading old data.
_NATIVE_SECRET_PAYLOAD_VERSION_KEY = "payload_version"
_NATIVE_SECRET_PAYLOAD_VALUES_KEY = "values"
_NATIVE_SECRET_PAYLOAD_KEYS = frozenset(
    {
        _NATIVE_SECRET_PAYLOAD_VERSION_KEY,
        _NATIVE_SECRET_PAYLOAD_VALUES_KEY,
    }
)

# Runtime-only key used after the envelope has been unpacked into normalized
# options. It carries the still-opaque BLOB through validation and is removed
# when secrets are restored at executor/actor creation. The encoder reserves
# this name as well so user options cannot collide with runtime state.
_NATIVE_OPTIONS_NORMALIZED_SECRET_KEY = "_vane_vllm_secret_payload"

# Names that cannot be supplied as ordinary top-level/public options. This set
# covers both the on-wire envelope fields and the transient normalized field;
# secret-reference markers are rejected recursively by the encoder/decoder.
_NATIVE_OPTIONS_RESERVED_KEYS = _NATIVE_OPTIONS_ENVELOPE_KEYS | {
    _NATIVE_OPTIONS_NORMALIZED_SECRET_KEY,
}


def _dump_vllm_protocol_json(value: Any) -> str:
    """Encode protocol data as compact UTF-8-compatible strict JSON.

    ``allow_nan=False`` keeps the encoder consistent with the strict decoder:
    non-finite values never cross the native plan boundary.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _load_vllm_protocol_json(raw: str | bytes, label: str) -> Any:
    """Decode untrusted protocol JSON without accepting ambiguous values.

    The decoder rejects invalid UTF-8, non-finite numeric constants, and
    duplicate object keys. ``label`` identifies the envelope component in the
    stable public error while preserving the parser failure as its cause.
    """

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite number {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} could not be parsed as strict JSON") from exc


def _unpack_native_options_envelope(options: dict[str, Any]) -> dict[str, Any]:
    """Validate and unpack an opaque native-options envelope.

    Secret values remain sealed in the normalized payload returned here. The
    execution layer restores them only at the executor or actor creation
    boundary.
    """
    if not _NATIVE_OPTIONS_ENVELOPE_KEYS.intersection(options):
        return dict(options)

    missing = _NATIVE_OPTIONS_ENVELOPE_KEYS.difference(options)
    if missing:
        raise ValueError(f"vllm native options envelope is missing fields: {', '.join(sorted(missing))}")
    unexpected = set(options).difference(_NATIVE_OPTIONS_ENVELOPE_KEYS | _NATIVE_OPTIONS_DISTRIBUTED_ROUTING_KEYS)
    if unexpected:
        raise ValueError(f"vllm native options envelope has unexpected fields: {', '.join(sorted(unexpected))}")

    version = options[_NATIVE_OPTIONS_VERSION_KEY]
    if isinstance(version, bool) or not isinstance(version, int) or version != _NATIVE_OPTIONS_PAYLOAD_VERSION:
        raise ValueError(f"unsupported vllm native options payload version: {version!r}")
    public_options_json = options[_NATIVE_OPTIONS_PUBLIC_KEY]
    if not isinstance(public_options_json, str):
        raise TypeError("vllm native public options must be a JSON string")
    secret_payload = options[_NATIVE_OPTIONS_SECRET_KEY]
    if not isinstance(secret_payload, bytes):
        raise TypeError("vllm native secret payload must be bytes")

    public_options = _load_vllm_protocol_json(public_options_json, "vllm native public options")
    if not isinstance(public_options, dict):
        raise ValueError("vllm native public options JSON must decode to a dict")
    reserved_public_keys = _NATIVE_OPTIONS_RESERVED_KEYS.intersection(public_options)
    if reserved_public_keys:
        raise ValueError("vllm native public options use reserved fields: " + ", ".join(sorted(reserved_public_keys)))
    for key in _NATIVE_OPTIONS_DISTRIBUTED_ROUTING_KEYS:
        if key in options:
            public_options[key] = options[key]
    public_options[_NATIVE_OPTIONS_NORMALIZED_SECRET_KEY] = secret_payload
    return public_options
