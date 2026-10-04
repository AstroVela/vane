# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared byte budgets for release archives and extension-provider wheels."""

MEBIBYTE = 1024 * 1024

# Bound bytes copied from an untrusted publication artifact before an archive
# parser sees them. This limit also keeps provider wheels within a practical
# index-upload and worker-install envelope.
MAX_PUBLICATION_FILE_BYTES = 128 * MEBIBYTE

# Bound one decompressed member separately from the complete archive. Native
# extension artifacts may legitimately be much larger than their compressed
# wheels, while the aggregate budget still limits decompression amplification.
MAX_ARCHIVE_MEMBER_BYTES = 384 * MEBIBYTE
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 512 * MEBIBYTE
MAX_EXTENSION_ARTIFACT_BYTES = MAX_ARCHIVE_MEMBER_BYTES


def _mibibytes(value: int) -> int:
    if value % MEBIBYTE:
        raise ValueError("artifact byte budgets must be whole MiB values")
    return value // MEBIBYTE


PUBLICATION_FILE_LIMIT_DESCRIPTION = f"the project's {_mibibytes(MAX_PUBLICATION_FILE_BYTES)} MiB publication limit"
ARCHIVE_MEMBER_LIMIT_DESCRIPTION = (
    f"the project's {_mibibytes(MAX_ARCHIVE_MEMBER_BYTES)} MiB per-member uncompressed limit"
)
ARCHIVE_TOTAL_LIMIT_DESCRIPTION = (
    f"the project's {_mibibytes(MAX_ARCHIVE_UNCOMPRESSED_BYTES)} MiB total uncompressed limit"
)
EXTENSION_ARTIFACT_LIMIT_DESCRIPTION = (
    f"the project's {_mibibytes(MAX_EXTENSION_ARTIFACT_BYTES)} MiB extension-artifact limit"
)

__all__ = [
    "ARCHIVE_MEMBER_LIMIT_DESCRIPTION",
    "ARCHIVE_TOTAL_LIMIT_DESCRIPTION",
    "EXTENSION_ARTIFACT_LIMIT_DESCRIPTION",
    "MAX_ARCHIVE_MEMBER_BYTES",
    "MAX_ARCHIVE_UNCOMPRESSED_BYTES",
    "MAX_EXTENSION_ARTIFACT_BYTES",
    "MAX_PUBLICATION_FILE_BYTES",
    "MEBIBYTE",
    "PUBLICATION_FILE_LIMIT_DESCRIPTION",
]
