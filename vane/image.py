# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Native multimodal image expressions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vane._expressions import as_expression

if TYPE_CHECKING:
    from vane import Expression


def crop(frame: Any, bbox: Any) -> Expression:
    """Crop an RGB frame into a dynamic-size native RGB image value.

    ``bbox`` contains ``[left, top, right, bottom]`` coordinates and follows
    the benchmark's Pillow crop semantics, including out-of-bounds black
    padding and Python :class:`int` truncation.
    """
    import vane

    return vane.FunctionExpression(
        "image_crop",
        as_expression(frame),
        as_expression(bbox),
    )


def encode(image: Any, format: str = "png") -> Expression:
    """Encode a native image value in the requested format.

    The format is bound as a query constant. ``"png"`` reproduces Pillow
    11.3.0 RGB PNG output with the benchmark's ``compress_level=2`` setting.
    """
    import vane

    if not isinstance(format, str):
        raise TypeError("image encode format must be a string")
    return vane.FunctionExpression(
        "image_encode",
        as_expression(image),
        vane.lit(format),
    )


__all__ = ["crop", "encode"]
