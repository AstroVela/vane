# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pickle

import pytest

from vane.ai._redaction import (
    REDACTED_PLACEHOLDER,
    SENSITIVE_OPTION_KEYS,
    Secret,
    is_sensitive_option_key,
    normalized_option_key,
    unwrap_sensitive_options,
    wrap_sensitive_options,
)

SENTINEL = "sk-INLINE-SECRET-SENTINEL-0123456789"


class TestSecret:
    def test_repr_and_str_are_fixed_placeholder(self):
        for value in ("x", SENTINEL, "a-much-longer-credential-value-" * 8):
            secret = Secret(value)
            assert repr(secret) == REDACTED_PLACEHOLDER
            assert str(secret) == REDACTED_PLACEHOLDER
            assert f"{secret}" == REDACTED_PLACEHOLDER
            assert "{}".format(secret) == REDACTED_PLACEHOLDER
            assert value not in repr(secret)
            assert value not in str(secret)

    def test_placeholder_is_length_hiding(self):
        short = Secret("x")
        long = Secret("y" * 500)
        assert len(repr(short)) == len(repr(long))
        assert repr(short) == repr(long)

    def test_reveal_returns_original_value(self):
        assert Secret(SENTINEL).reveal() == SENTINEL

    def test_reveal_after_pickle_round_trip(self):
        secret = Secret(SENTINEL)
        restored = pickle.loads(pickle.dumps(secret))
        assert isinstance(restored, Secret)
        assert restored.reveal() == SENTINEL
        assert repr(restored) == REDACTED_PLACEHOLDER
        assert SENTINEL not in repr(restored)
        assert SENTINEL not in str(restored)

    def test_equality_between_secrets_compares_real_values(self):
        assert Secret(SENTINEL) == Secret(SENTINEL)
        assert Secret(SENTINEL) != Secret("other")

    def test_equality_with_bare_str_is_always_false(self):
        secret = Secret(SENTINEL)
        assert not (secret == SENTINEL)
        assert not (SENTINEL == secret)
        assert secret != SENTINEL
        assert SENTINEL != secret

    def test_hash_is_consistent_with_equality(self):
        assert hash(Secret(SENTINEL)) == hash(Secret(SENTINEL))
        assert len({Secret(SENTINEL), Secret(SENTINEL)}) == 1
        assert {Secret(SENTINEL): "hit"}[Secret(SENTINEL)] == "hit"

    def test_wrapping_a_secret_keeps_a_single_secret(self):
        double = Secret(Secret(SENTINEL))
        assert isinstance(double, Secret)
        assert not isinstance(double.reveal(), Secret)
        assert double.reveal() == SENTINEL

    def test_non_str_values_are_supported(self):
        secret = Secret(1234)
        assert secret.reveal() == 1234
        assert repr(secret) == REDACTED_PLACEHOLDER

    def test_empty_string_is_still_redacted(self):
        secret = Secret("")
        assert secret.reveal() == ""
        assert repr(secret) == REDACTED_PLACEHOLDER


class TestSensitiveKeyMatching:
    @pytest.mark.parametrize(
        "key",
        [
            "api_key",
            "API-Key",
            "apiKey",
            "openai_api_key",
            "aws_access_key_id",
            "aws_secret_access_key",
            "azure_client_secret_value",
            "client-secret",
            "google-api-token",
            "Proxy-Authorization",
            "access_token",
            "password",
            "credentials",
        ],
    )
    def test_sensitive_keys_match(self, key):
        assert is_sensitive_option_key(key)

    @pytest.mark.parametrize(
        "key",
        [
            "token_budget",
            "max_tokens",
            "max_output_tokens",
            "model",
            "provider",
            "temperature",
            "secretary",
            "passwordless_mode",
        ],
    )
    def test_near_miss_keys_do_not_match(self, key):
        assert not is_sensitive_option_key(key)

    def test_normalization_casefolds_and_strips_non_alphanumerics(self):
        assert normalized_option_key("Proxy-Authorization") == "proxyauthorization"
        assert normalized_option_key("API_KEY") == "apikey"
        assert normalized_option_key(123) == "123"

    def test_key_table_is_normalized(self):
        for key in SENSITIVE_OPTION_KEYS:
            assert key == normalized_option_key(key)


class TestWrapSensitiveOptions:
    def test_wraps_sensitive_keys_at_any_depth(self):
        options = {
            "model": "gpt-4o",
            "api_key": SENTINEL,
            "provider_options": {
                "aws_secret_access_key": SENTINEL,
                "region": "us-east-1",
                "endpoints": [{"authorization": SENTINEL, "url": "https://example.com"}],
            },
            "headers": ({"proxy_authorization": SENTINEL},),
            "token_budget": 42,
        }
        wrapped = wrap_sensitive_options(options)

        assert isinstance(wrapped["api_key"], Secret)
        assert wrapped["api_key"].reveal() == SENTINEL
        assert isinstance(wrapped["provider_options"]["aws_secret_access_key"], Secret)
        assert isinstance(wrapped["provider_options"]["endpoints"][0]["authorization"], Secret)
        assert isinstance(wrapped["headers"], tuple)
        assert isinstance(wrapped["headers"][0]["proxy_authorization"], Secret)
        assert SENTINEL not in repr(wrapped)
        assert SENTINEL not in str(wrapped)

    def test_non_sensitive_and_near_miss_keys_are_untouched(self):
        options = {
            "model": "gpt-4o",
            "token_budget": 42,
            "max_tokens": 128,
            "provider_options": {"region": "us-east-1"},
        }
        wrapped = wrap_sensitive_options(options)
        assert wrapped == options
        assert wrapped["model"] == "gpt-4o"
        assert wrapped["token_budget"] == 42
        assert wrapped["max_tokens"] == 128
        assert wrapped["provider_options"] == {"region": "us-east-1"}

    def test_input_mapping_is_not_mutated(self):
        options = {"api_key": SENTINEL, "provider_options": {"access_token": SENTINEL}}
        wrap_sensitive_options(options)
        assert options["api_key"] == SENTINEL
        assert options["provider_options"]["access_token"] == SENTINEL

    def test_wrap_is_idempotent(self):
        options = {"api_key": SENTINEL, "provider_options": {"access_token": SENTINEL}}
        rewrapped = wrap_sensitive_options(wrap_sensitive_options(options))
        assert isinstance(rewrapped["api_key"], Secret)
        assert not isinstance(rewrapped["api_key"].reveal(), Secret)
        assert rewrapped["api_key"].reveal() == SENTINEL
        inner = rewrapped["provider_options"]["access_token"]
        assert isinstance(inner, Secret)
        assert inner.reveal() == SENTINEL

    def test_container_under_sensitive_key_is_sealed_whole(self):
        options = {"credentials": {"access_token": SENTINEL}}
        wrapped = wrap_sensitive_options(options)
        assert isinstance(wrapped["credentials"], Secret)
        assert wrapped["credentials"].reveal() == {"access_token": SENTINEL}
        assert SENTINEL not in repr(wrapped)

    def test_none_values_stay_none(self):
        wrapped = wrap_sensitive_options({"api_key": None, "model": None})
        assert wrapped["api_key"] is None
        assert wrapped["model"] is None

    def test_non_str_sensitive_values_are_wrapped(self):
        wrapped = wrap_sensitive_options({"secret": 1234})
        assert isinstance(wrapped["secret"], Secret)
        assert wrapped["secret"].reveal() == 1234


class TestUnwrapSensitiveOptions:
    def test_unwrap_restores_equivalent_plain_mapping(self):
        options = {
            "model": "gpt-4o",
            "api_key": SENTINEL,
            "provider_options": {
                "aws_secret_access_key": SENTINEL,
                "endpoints": [{"authorization": SENTINEL, "url": "https://example.com"}],
            },
            "headers": ({"proxy_authorization": SENTINEL},),
            "credentials": {"access_token": SENTINEL},
            "token_budget": 42,
        }
        assert unwrap_sensitive_options(wrap_sensitive_options(options)) == options

    def test_unwrap_of_plain_mapping_is_a_no_op(self):
        options = {"model": "gpt-4o", "provider_options": {"region": "us-east-1"}}
        assert unwrap_sensitive_options(options) == options


class TestRedactionInFailureOutput:
    def test_failing_assertion_diff_contains_no_plaintext(self):
        wrapped = wrap_sensitive_options({"api_key": SENTINEL, "model": "gpt-4o"})
        with pytest.raises(AssertionError) as excinfo:
            assert wrapped == {"api_key": "some-other-value", "model": "gpt-4o"}
        text = str(excinfo.value)
        assert SENTINEL not in text
        assert REDACTED_PLACEHOLDER in text

    def test_exception_message_embedding_a_secret_is_redacted(self):
        secret = Secret(SENTINEL)
        error = ValueError(f"bad option: {secret!r} ({secret})")
        assert SENTINEL not in str(error)
        assert REDACTED_PLACEHOLDER in str(error)


class TestSqlLayerSharesKeyTable:
    def test_sql_binding_imports_the_shared_matcher(self):
        from vane.ai import _redaction, _sql

        assert _sql.is_sensitive_option_key is _redaction.is_sensitive_option_key

    def test_sql_rejection_still_uses_shared_matching(self):
        from vane.ai._sql import _reject_inline_credentials

        _reject_inline_credentials({"model": "gpt-4o", "token_budget": 42})
        with pytest.raises(ValueError, match="inline credential"):
            _reject_inline_credentials({"provider_options": {"aws_secret_access_key": SENTINEL}})
        with pytest.raises(ValueError, match="inline credential"):
            _reject_inline_credentials({"provider_options": [{"google-api-token": SENTINEL}]})
