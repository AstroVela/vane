# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import importlib.util

import pytest

from vane.ai._media import PromptMedia, normalize_media_content_type
from vane.ai.providers._mime import ImageMimePolicy, detect_image_mime_type
from vane.ai.providers.anthropic import _IMAGE_MIME_POLICY as _ANTHROPIC_IMAGE_MIME_POLICY
from vane.ai.providers.google import _IMAGE_MIME_POLICY as _GOOGLE_IMAGE_MIME_POLICY
from vane.ai.providers.openai import _IMAGE_MIME_POLICY as _OPENAI_IMAGE_MIME_POLICY


def _ftyp(
    major_brand: bytes,
    *compatible_brands: bytes,
    extended_size: bool = False,
) -> bytes:
    assert len(major_brand) == 4
    assert all(len(brand) == 4 for brand in compatible_brands)

    body = major_brand + b"\x00\x00\x00\x00" + b"".join(compatible_brands)
    if extended_size:
        size = 16 + len(body)
        return b"\x00\x00\x00\x01ftyp" + size.to_bytes(8, "big") + body
    size = 8 + len(body)
    return size.to_bytes(4, "big") + b"ftyp" + body


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


_PNG = b"\x89PNG\r\n\x1a\nimage"
_JPEG = b"\xff\xd8\xff\xe0image"
_GIF = b"GIF89aimage"
_WEBP_BODY = b"WEBPimage"
_WEBP = b"RIFF" + len(_WEBP_BODY).to_bytes(4, "little") + _WEBP_BODY
_AVIF = _ftyp(b"avif", b"avif", b"mif1")
_HEIC = _ftyp(b"mif1", b"mif1", b"heic")
_HEIF = _ftyp(b"mif1", b"mif1")
_HEIC_SEQUENCE = _ftyp(b"msf1", b"msf1", b"hevc")
_HEIF_SEQUENCE = _ftyp(b"msf1", b"msf1")

_DETECTED_IMAGES = {
    "image/avif": _AVIF,
    "image/gif": _GIF,
    "image/heic": _HEIC,
    "image/heic-sequence": _HEIC_SEQUENCE,
    "image/heif": _HEIF,
    "image/heif-sequence": _HEIF_SEQUENCE,
    "image/jpeg": _JPEG,
    "image/png": _PNG,
    "image/webp": _WEBP,
}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("IMAGE/PNG; charset=binary", "image/png", id="parameters"),
        pytest.param("audio/wav", "audio/wav", id="audio"),
        pytest.param("video/mp4", "video/mp4", id="video"),
    ],
)
def test_prompt_media_normalizes_declared_content_type(value, expected):
    assert normalize_media_content_type(value) == expected


@pytest.mark.parametrize("value", ["invalid", "image/*", "*/*", "image/", "/png", "image/p ng"])
def test_prompt_media_rejects_invalid_declared_content_type(value):
    with pytest.raises(ValueError, match="valid MIME type"):
        PromptMedia(b"payload", value)


def test_prompt_media_repr_does_not_expose_payload():
    payload = b"PRIVATE_MEDIA_BYTES"
    media = PromptMedia(payload, "image/png")

    assert bytes(media) == payload
    assert repr(media) == "PromptMedia(content_type='image/png', size=19)"
    assert "PRIVATE_MEDIA_BYTES" not in repr(media)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param(_PNG, "image/png", id="png"),
        pytest.param(_JPEG, "image/jpeg", id="jpeg"),
        pytest.param(b"GIF87aimage", "image/gif", id="gif87a"),
        pytest.param(_GIF, "image/gif", id="gif89a"),
        pytest.param(_WEBP, "image/webp", id="webp"),
    ],
)
def test_detect_image_mime_type_recognizes_image_signatures(data, expected):
    assert detect_image_mime_type(data) == expected


@pytest.mark.parametrize(
    ("brand", "expected"),
    [
        pytest.param(b"heic", "image/heic", id="heic"),
        pytest.param(b"heix", "image/heic", id="heix"),
        pytest.param(b"heim", "image/heic", id="heim"),
        pytest.param(b"heis", "image/heic", id="heis"),
        pytest.param(b"mif1", "image/heif", id="mif1"),
        pytest.param(b"hevc", "image/heic-sequence", id="hevc"),
        pytest.param(b"hevx", "image/heic-sequence", id="hevx"),
        pytest.param(b"hevm", "image/heic-sequence", id="hevm"),
        pytest.param(b"hevs", "image/heic-sequence", id="hevs"),
        pytest.param(b"msf1", "image/heif-sequence", id="msf1"),
    ],
)
def test_detect_image_mime_type_uses_compatible_ftyp_brands(brand, expected):
    assert detect_image_mime_type(_ftyp(b"isom", brand)) == expected


@pytest.mark.parametrize(
    "compatible_brands",
    [
        pytest.param((b"avif", b"mif1"), id="image-specific-first"),
        pytest.param((b"mif1", b"avif"), id="image-generic-first"),
        pytest.param((b"avis", b"msf1"), id="sequence-specific-first"),
        pytest.param((b"msf1", b"avis"), id="sequence-generic-first"),
        pytest.param((b"avio", b"mif1"), id="intra-only-specific-first"),
        pytest.param((b"mif1", b"avio"), id="intra-only-generic-first"),
    ],
)
def test_detect_image_mime_type_recognizes_avif_compatible_brands(compatible_brands):
    assert detect_image_mime_type(_ftyp(b"isom", *compatible_brands)) == "image/avif"


@pytest.mark.parametrize(
    ("major_brand", "generic_brand"),
    [
        pytest.param(b"avif", b"mif1", id="image"),
        pytest.param(b"avis", b"msf1", id="sequence"),
        pytest.param(b"avio", b"mif1", id="intra-only"),
    ],
)
def test_detect_image_mime_type_recognizes_avif_major_brands(major_brand, generic_brand):
    assert detect_image_mime_type(_ftyp(major_brand, generic_brand)) == "image/avif"


@pytest.mark.parametrize(
    ("major_brand", "expected"),
    [
        pytest.param(b"avif", "image/avif", id="avif-major"),
        pytest.param(b"heic", "image/heic", id="heic-major"),
        pytest.param(b"isom", None, id="ambiguous-generic-major"),
    ],
)
def test_detect_image_mime_type_handles_conflicting_codec_brands(major_brand, expected):
    data = _ftyp(major_brand, b"avif", b"mif1", b"heic")
    assert detect_image_mime_type(data) == expected


def test_detect_image_mime_type_prefers_codec_specific_compatible_brand():
    assert detect_image_mime_type(_ftyp(b"mif1", b"mif1", b"heic")) == "image/heic"
    assert detect_image_mime_type(_ftyp(b"msf1", b"msf1", b"hevc")) == "image/heic-sequence"


def test_detect_image_mime_type_prefers_still_brand_when_sequence_is_also_present():
    # This brand combination is used by libheif's official example.heic.
    assert detect_image_mime_type(_ftyp(b"mif1", b"mif1", b"heic", b"hevc")) == "image/heic"


@pytest.mark.parametrize(
    ("major_brand", "expected"),
    [
        pytest.param(b"mif1", "image/heic", id="still-major"),
        pytest.param(b"hevc", "image/heic-sequence", id="sequence-major"),
        pytest.param(b"isom", None, id="ambiguous-generic-major"),
    ],
)
def test_detect_image_mime_type_uses_major_brand_to_choose_between_kinds(major_brand, expected):
    data = _ftyp(major_brand, b"mif1", b"heic", b"msf1", b"hevc")
    assert detect_image_mime_type(data) == expected


def test_detect_image_mime_type_does_not_treat_major_brand_as_compatible():
    assert detect_image_mime_type(_ftyp(b"heic", b"isom")) is None


def test_detect_image_mime_type_supports_extended_size_ftyp_box():
    assert detect_image_mime_type(_ftyp(b"mif1", b"mif1", b"heic", extended_size=True)) == "image/heic"


def test_detect_image_mime_type_ignores_brands_after_ftyp_box():
    data = _ftyp(b"isom", b"isom") + b"\x00\x00\x00\x0cmdatheic"
    assert detect_image_mime_type(data) is None


def test_detect_image_mime_type_requires_aligned_compatible_brand():
    data = _ftyp(b"isom", b"Xhei", b"cYYY")
    assert detect_image_mime_type(data) is None


def test_detect_image_mime_type_bounds_ftyp_compatible_brand_scanning():
    box_size = 64 * 1024 + 4
    compatible_brand_count = (box_size - 16) // 4
    data = (
        box_size.to_bytes(4, "big")
        + b"ftyp"
        + b"isom"
        + b"\x00\x00\x00\x00"
        + b"heic"
        + b"isom" * (compatible_brand_count - 1)
    )
    assert len(data) == box_size
    assert detect_image_mime_type(data) is None


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"not-an-image", id="unknown"),
        pytest.param(b"%PDF-1.7", id="pdf"),
        pytest.param(b"\xff\xd8", id="truncated-jpeg"),
        pytest.param(b"GIF8", id="truncated-gif"),
        pytest.param(b"RIFF\x00\x00\x00\x00NOPE", id="non-webp-riff"),
        pytest.param(b"RIFF\x04\x00\x00\x00WEBPextra", id="webp-size-too-small"),
        pytest.param(b"RIFF\xff\xff\xff\xffWEBP", id="webp-size-too-large"),
        pytest.param(_ftyp(b"avc1", b"avc1"), id="unknown-ftyp-brand"),
        pytest.param(b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00mif1", id="truncated-ftyp-box"),
        pytest.param(b"\x00\x00\x00\x0cftypmif1\x00\x00\x00\x00mif1", id="undersized-ftyp-box"),
        pytest.param(b"\x00\x00\x00\x15ftypmif1\x00\x00\x00\x00mif1x", id="misaligned-ftyp-brands"),
        pytest.param(b"\x00\x00\x00\x00ftypmif1\x00\x00\x00\x00mif1", id="ftyp-extending-to-end"),
        pytest.param(b"\x00\x00\x00\x01ftyp" + b"\x00" * 12, id="truncated-extended-size"),
        pytest.param(
            b"\x00\x00\x00\x01ftyp\x00\x00\x00\x00\x00\x00\x00\x14mif1\x00\x00\x00\x00",
            id="undersized-extended-ftyp-box",
        ),
    ],
)
def test_detect_image_mime_type_rejects_unknown_or_malformed_data(data):
    assert detect_image_mime_type(data) is None


@pytest.mark.parametrize(
    ("policy", "expected_mime_types"),
    [
        pytest.param(
            _ANTHROPIC_IMAGE_MIME_POLICY,
            frozenset({"image/gif", "image/jpeg", "image/png", "image/webp"}),
            id="anthropic",
        ),
        pytest.param(
            _GOOGLE_IMAGE_MIME_POLICY,
            frozenset({"image/heic", "image/heif", "image/jpeg", "image/png", "image/webp"}),
            id="google",
        ),
        pytest.param(
            _OPENAI_IMAGE_MIME_POLICY,
            frozenset({"image/gif", "image/jpeg", "image/png", "image/webp"}),
            id="openai",
        ),
    ],
)
def test_provider_image_mime_policy_matches_documented_formats(
    policy: ImageMimePolicy,
    expected_mime_types: frozenset[str],
):
    assert policy.supported_mime_types == expected_mime_types
    for mime_type, data in _DETECTED_IMAGES.items():
        if mime_type in expected_mime_types:
            assert policy.require_supported(data) == mime_type
        else:
            with pytest.raises(ValueError, match=f"not supported by {policy.provider_name}"):
                policy.require_supported(data)


@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(_ANTHROPIC_IMAGE_MIME_POLICY, id="anthropic"),
        pytest.param(_GOOGLE_IMAGE_MIME_POLICY, id="google"),
        pytest.param(_OPENAI_IMAGE_MIME_POLICY, id="openai"),
    ],
)
def test_provider_image_mime_policy_rejects_unrecognized_data(policy):
    with pytest.raises(ValueError, match="unrecognized image format"):
        policy.require_supported(b"not-an-image")


@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(_ANTHROPIC_IMAGE_MIME_POLICY, id="anthropic"),
        pytest.param(_OPENAI_IMAGE_MIME_POLICY, id="openai"),
    ],
)
def test_image_provider_policy_uses_file_metadata_without_sniffing(policy):
    assert policy.require_supported(PromptMedia(b"not-an-image", "IMAGE/PNG; source=metadata")) == "image/png"
    with pytest.raises(ValueError, match="FILE MIME type.*not supported"):
        policy.require_supported(PromptMedia(b"payload", "audio/wav"))


def test_anthropic_message_processing_uses_provider_image_policy():
    from vane.ai.providers.anthropic import AnthropicPrompter

    image = AnthropicPrompter._process_message(_PNG)
    assert image["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(_PNG).decode("ascii"),
    }
    with pytest.raises(ValueError, match="not supported by Anthropic"):
        AnthropicPrompter._process_message(_HEIC)

    declared = PromptMedia(b"declared-not-sniffed", "image/png")
    file_image = AnthropicPrompter._process_message(declared)
    assert file_image["source"] == {
        "type": "base64",
        "media_type": "image/png",
        "data": base64.b64encode(bytes(declared)).decode("ascii"),
    }


def test_openai_message_processing_uses_provider_image_policy():
    from vane.ai.providers.openai import OpenAIPrompter

    prompter = OpenAIPrompter.__new__(OpenAIPrompter)
    prompter._use_chat_completions = False
    image = prompter._process_bytes(_PNG)
    assert image["type"] == "input_image"
    assert image["image_url"].startswith("data:image/png;base64,")
    with pytest.raises(ValueError, match="not supported by OpenAI"):
        prompter._process_bytes(_HEIC)

    declared = prompter._process_bytes(PromptMedia(b"declared-not-sniffed", "image/png"))
    assert declared["type"] == "input_image"
    assert declared["image_url"].startswith("data:image/png;base64,")


@pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
def test_google_message_processing_uses_provider_image_policy():
    from vane.ai.providers.google import GooglePrompter

    prompter = GooglePrompter.__new__(GooglePrompter)
    image = prompter._process_message(_HEIC)
    assert image.inline_data.mime_type == "image/heic"
    assert image.inline_data.data == _HEIC
    with pytest.raises(ValueError, match="not supported by Google"):
        prompter._process_message(_GIF)
    with pytest.raises(ValueError, match="not supported by Google"):
        prompter._process_message(_AVIF)


@pytest.mark.skipif(not _has_module("google.genai"), reason="google-genai not installed")
@pytest.mark.parametrize("content_type", ["audio/wav", "video/mp4", "application/pdf"])
def test_google_message_processing_routes_declared_file_media(content_type):
    from vane.ai.providers.google import GooglePrompter

    prompter = GooglePrompter.__new__(GooglePrompter)
    media = prompter._process_message(PromptMedia(b"provider-payload", content_type))

    assert media.inline_data.mime_type == content_type
    assert media.inline_data.data == b"provider-payload"
