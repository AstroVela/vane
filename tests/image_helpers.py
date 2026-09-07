# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Pixel fixtures and recursive assertions for Image transport tests."""

from collections.abc import Mapping

import numpy as np


def make_image(data, width, height, mode):
    from PIL import Image

    return Image.frombytes(mode, (width, height), data)


def assert_image_equal(actual, expected):
    if hasattr(expected, "getbands"):
        expected = np.asarray(expected)
        if expected.ndim == 2:
            expected = expected[:, :, np.newaxis]
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_image_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_image_equal(left, right)
    else:
        assert actual == expected
