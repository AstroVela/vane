# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Image MIME detection and provider-level format policies.

The detector identifies encoded image formats from their byte signatures. It
does not decide whether an AI provider accepts the detected format; each
provider owns that policy through :class:`ImageMimePolicy`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"
_GIF_SIGNATURES: Final = (b"GIF87a", b"GIF89a")
_MAX_FTYP_BOX_SIZE: Final = 64 * 1024

# AVIF brand and media-type specifications:
# https://aomediacodec.github.io/av1-avif/#brands-overview
# https://www.iana.org/assignments/media-types/image/avif
# AVIF is a HEIF-based format and normally also advertises mif1 or msf1. Its
# codec-specific brands must therefore be recognized before generic HEIF.
_AVIF_BRANDS: Final = frozenset({b"avif", b"avis", b"avio"})

# IANA registrations:
# https://www.iana.org/assignments/media-types/image/heic
# https://www.iana.org/assignments/media-types/image/heic-sequence
# The image/heif pages share the corresponding registrations. All four media
# types require one of these brands in the ftyp compatible-brands array.
_HEIC_BRANDS: Final = frozenset({b"heic", b"heix", b"heim", b"heis"})
_HEIF_BRANDS: Final = frozenset({b"mif1"})
_HEIC_SEQUENCE_BRANDS: Final = frozenset({b"hevc", b"hevx", b"hevm", b"hevs"})
_HEIF_SEQUENCE_BRANDS: Final = frozenset({b"msf1"})


def _parse_ftyp_brands(data: bytes) -> tuple[bytes, int, int] | None:
    """Return the major brand and compatible-brand range of an initial ftyp."""
    if len(data) < 16 or data[4:8] != b"ftyp":
        return None

    box_size = int.from_bytes(data[:4], "big")
    major_brand_start = 8
    compatible_brands_start = 16
    if box_size == 1:
        if len(data) < 24:
            return None
        box_size = int.from_bytes(data[8:16], "big")
        major_brand_start = 16
        compatible_brands_start = 24
    elif box_size == 0:
        # ftyp must be followed by the boxes that carry the image. A box that
        # extends to end-of-file cannot describe a usable HEIF payload.
        return None

    # Detection runs before a provider request is built. Bound work here so a
    # malformed BLOB cannot turn its whole payload into a compatible-brand
    # list and consume proportional CPU. Real ftyp boxes are tiny; 64 KiB still
    # permits more than sixteen thousand brand entries.
    if box_size > _MAX_FTYP_BOX_SIZE:
        return None
    if box_size < compatible_brands_start or box_size > len(data):
        return None
    if (box_size - compatible_brands_start) % 4 != 0:
        return None
    return data[major_brand_start : major_brand_start + 4], compatible_brands_start, box_size


def _detect_iso_bmff_image_mime_type(data: bytes) -> str | None:
    brands = _parse_ftyp_brands(data)
    if brands is None:
        return None
    major_brand, start, end = brands

    has_avif = False
    has_heic = False
    has_heif = False
    has_heic_sequence = False
    has_heif_sequence = False
    for offset in range(start, end, 4):
        brand = data[offset : offset + 4]
        if brand in _AVIF_BRANDS:
            has_avif = True
        elif brand in _HEIC_BRANDS:
            has_heic = True
        elif brand in _HEIF_BRANDS:
            has_heif = True
        elif brand in _HEIC_SEQUENCE_BRANDS:
            has_heic_sequence = True
        elif brand in _HEIF_SEQUENCE_BRANDS:
            has_heif_sequence = True

    # The AVIF specification assigns image/avif whenever one of its brands is
    # the major brand. AVIF compatible brands also take precedence over the
    # generic HEIF brands they normally accompany. If the file advertises both
    # AV1 and HEVC codec-specific brands, use a codec-specific major brand to
    # disambiguate; otherwise fail closed instead of treating it as HEIF.
    if major_brand in _AVIF_BRANDS:
        return "image/avif"
    if has_avif and (has_heic or has_heic_sequence):
        major_selects_heic = major_brand in _HEIC_BRANDS and has_heic
        major_selects_heic_sequence = major_brand in _HEIC_SEQUENCE_BRANDS and has_heic_sequence
        if not major_selects_heic and not major_selects_heic_sequence:
            return None
    elif has_avif:
        return "image/avif"

    # A file may advertise both image-item and image-sequence profiles. The
    # major brand expresses its best-use profile, but cannot establish MIME
    # compatibility by itself. Use it only to choose between kinds that were
    # independently established by compatible brands. Within either kind,
    # prefer the codec-specific HEIC type over the generic HEIF type.
    has_still = has_heic or has_heif
    has_sequence = has_heic_sequence or has_heif_sequence
    if has_still and has_sequence:
        major_is_still = major_brand in _HEIC_BRANDS or major_brand in _HEIF_BRANDS
        major_is_sequence = major_brand in _HEIC_SEQUENCE_BRANDS or major_brand in _HEIF_SEQUENCE_BRANDS
        if not major_is_still and not major_is_sequence:
            return None
        has_still = major_is_still

    if has_still:
        return "image/heic" if has_heic else "image/heif"
    if has_sequence:
        return "image/heic-sequence" if has_heic_sequence else "image/heif-sequence"
    return None


def detect_image_mime_type(data: bytes) -> str | None:
    """Detect a canonical image MIME type from encoded bytes.

    The function validates the signature or container fields used for
    detection, but leaves full image decoding and content validation to the
    provider. Unknown and malformed inputs return ``None``.
    """
    if data.startswith(_PNG_SIGNATURE):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(_GIF_SIGNATURES):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        riff_size = int.from_bytes(data[4:8], "little")
        if riff_size + 8 == len(data):
            return "image/webp"
    return _detect_iso_bmff_image_mime_type(data)


@dataclass(frozen=True)
class ImageMimePolicy:
    """Provider-owned allowlist applied after format detection."""

    provider_name: str
    supported_mime_types: frozenset[str]

    def require_supported(self, data: bytes) -> str:
        """Return the detected MIME type or reject the input locally."""
        mime_type = detect_image_mime_type(data)
        if mime_type is None:
            raise ValueError("Prompt image BLOB has an unrecognized image format")
        if mime_type not in self.supported_mime_types:
            supported = ", ".join(sorted(self.supported_mime_types))
            raise ValueError(
                f"Prompt image BLOB format {mime_type!r} is not supported by "
                f"{self.provider_name}; supported MIME types: {supported}"
            )
        return mime_type
