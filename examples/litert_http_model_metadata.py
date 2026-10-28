# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Create an 8192-token HTTP deployment copy of the reviewed LiteRT 740M model.

Run in an environment with litert-lm==0.18.0. This edits only the embedded
EmbeddingMetadata default; it never changes the original or overwrites a target.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path

SOURCE_SHA256 = "e7a8a2204b91e0f96e92960e84a09a89212e1633dcb7575a9bf3378b4df77f4c"
DEPLOYMENT_SHA256 = "6d738925f2686fa545e50dee7e26e0c4ee8af43b69e63833783ab463e92f63ba"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source: Path, destination: Path) -> dict[str, str | int]:
    from litert_lm_builder import litertlm_peek
    from litert_lm_builder.runtime.proto import embedding_metadata_pb2

    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError("Expected the exact reviewed EmbeddingGemma 2 740M LiteRT model")
    if destination.exists():
        raise FileExistsError("Choose a new deployment path; existing files are never overwritten")
    header = litertlm_peek.read_litertlm_header(str(source), io.StringIO()).SectionMetadata()
    sections = [header.Objects(index) for index in range(header.ObjectsLength())]
    # AnySectionDataType.EmbeddingMetadataProto in the official builder schema.
    metadata_sections = [section for section in sections if section.DataType() == 8]
    if len(metadata_sections) != 1:
        raise ValueError("Expected exactly one EmbeddingMetadata section")
    section = metadata_sections[0]
    with source.open("rb") as stream:
        stream.seek(section.BeginOffset())
        original = stream.read(section.EndOffset() - section.BeginOffset())
    metadata = embedding_metadata_pb2.EmbeddingMetadata()
    metadata.ParseFromString(original)
    if metadata.max_input_length != 1024:
        raise ValueError("Expected the reviewed model's 1024-token default")
    metadata.max_input_length = 8192
    changed = metadata.SerializeToString()
    if len(original) != len(changed) or sum(a != b for a, b in zip(original, changed)) != 1:
        raise ValueError("Expected a fixed-length, one-byte metadata change")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Prepare and verify a private file before atomically publishing a new name.
    # Linking fails if another process has created destination in the meantime.
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".litert-metadata-", delete=False) as target:
        temporary = Path(target.name)
    try:
        shutil.copyfile(source, temporary)
        with temporary.open("r+b") as stream:
            stream.seek(section.BeginOffset())
            stream.write(changed)
        if file_sha256(temporary) != DEPLOYMENT_SHA256:
            raise ValueError("Deployment copy does not match the reviewed whole-file SHA-256")
        if file_sha256(source) != SOURCE_SHA256:
            raise ValueError("Source model changed during preparation")
        os.link(temporary, destination)
    finally:
        temporary.unlink()
    return {
        "source_sha256": SOURCE_SHA256,
        "deployment_sha256": DEPLOYMENT_SHA256,
        "max_input_length": 8192,
        "changed_bytes": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
