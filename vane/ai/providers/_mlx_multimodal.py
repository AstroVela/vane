# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Ordered MLX joint plans and per-part structural checks; no SDK imports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vane.ai._audio_embedding import AudioClip
from vane.ai._embedding_content import EmbeddingContent, EmbeddingPart, EmbeddingPartSpec, MultimodalInputSpec
from vane.ai._embedding_inputs import EmbeddingConfigurationError
from vane.ai._video_embedding import VideoClip
from vane.ai.embedding import Modality
from vane.ai.providers._mlx_media import validate_audio, validate_image, validate_video

# Fixed checkpoint/processor vocabulary, independently qualified by SDK P0.
MEDIA_TOKENS = {
    "image_token": ("<|image|>", 258880),
    "video_token": ("<|video|>", 258884),
    "audio_token": ("<|audio|>", 258881),
    "boi_token": ("<|image>", 255999),
    "eoi_token": ("<image|>", 258882),
    "boa_token": ("<|audio>", 256000),
    "eoa_token": ("<audio|>", 258883),
}


@dataclass(frozen=True)
class JointPart:
    kind: Modality
    index: int = 0
    text: str = ""


@dataclass(frozen=True)
class JointPlan:
    rows: tuple[tuple[JointPart, ...], ...]
    images: tuple[np.ndarray, ...]
    images_by_row: tuple[tuple[np.ndarray, ...], ...]
    videos: tuple[VideoClip, ...]
    audios: tuple[AudioClip, ...]


def joint_plan(inputs: list[EmbeddingContent], spec: MultimodalInputSpec) -> JointPlan:
    """Validate the whole raw batch before copying or processing any media."""
    if not isinstance(inputs, list):
        raise EmbeddingConfigurationError("MLX embed_multimodal requires a list of EmbeddingContent values")
    rows: list[tuple[JointPart, ...]] = []
    images: list[np.ndarray] = []
    images_by_row: list[tuple[np.ndarray, ...]] = []
    videos: list[VideoClip] = []
    audios: list[AudioClip] = []
    for row_index, row in enumerate(inputs):
        if not isinstance(row, EmbeddingContent) or not isinstance(row.parts, tuple):
            raise EmbeddingConfigurationError(f"MLX joint row {row_index} requires an ordered parts tuple")
        if any(
            not isinstance(part, EmbeddingPart) or not isinstance(part.spec, EmbeddingPartSpec) for part in row.parts
        ):
            raise EmbeddingConfigurationError(f"MLX joint row {row_index} requires typed EmbeddingPart values")
        try:
            spec.validate_layout(tuple(part.spec for part in row.parts))
        except EmbeddingConfigurationError:
            raise EmbeddingConfigurationError(
                f"MLX joint row {row_index} requires 1–16 supported decoded parts"
            ) from None
        parts: list[JointPart] = []
        row_images: list[np.ndarray] = []
        text_parts: list[str] = []
        input_bytes = 0
        for part_index, part in enumerate(row.parts):
            value, kind = part.value, part.spec.kind
            if value is None:
                continue
            try:
                if kind == "text":
                    if not isinstance(value, str):
                        raise EmbeddingConfigurationError("text parts require strings")
                    if not value.strip():
                        continue
                    try:
                        input_bytes += len(value.encode("utf-8"))
                    except UnicodeEncodeError:
                        raise EmbeddingConfigurationError("text parts require valid UTF-8") from None
                    text_parts.append(value)
                    parts.append(JointPart(kind, text=value))
                elif kind == "image":
                    validate_image(value)
                    input_bytes += value.nbytes
                    parts.append(JointPart(kind, len(images)))
                    images.append(value)
                    row_images.append(value)
                elif kind == "video":
                    validate_video(value)
                    input_bytes += sum(frame.nbytes for frame in value.frames)
                    parts.append(JointPart(kind, len(videos)))
                    videos.append(value)
                else:
                    validate_audio(value)
                    input_bytes += value.samples.nbytes
                    parts.append(JointPart(kind, len(audios)))
                    audios.append(value)
            except EmbeddingConfigurationError as error:
                raise EmbeddingConfigurationError(
                    f"MLX joint row {row_index}, part {part_index} ({kind}): {error}"
                ) from None
            if input_bytes > spec.max_input_bytes:
                raise EmbeddingConfigurationError(f"MLX joint row {row_index} exceeds the combined input byte budget")
        if not parts:
            raise EmbeddingConfigurationError(f"MLX joint row {row_index} has no nonempty parts")
        # The template concatenates *all* text parts to detect manual markers,
        # including text on opposite sides of media. Detect split markers too.
        text = "".join(text_parts)
        for name, (token, _) in MEDIA_TOKENS.items():
            if token in text:
                raise EmbeddingConfigurationError(
                    f"MLX joint row {row_index} text parts contain a reserved {name} marker"
                )
        rows.append(tuple(parts))
        images_by_row.append(tuple(row_images))
    return JointPlan(tuple(rows), tuple(images), tuple(images_by_row), tuple(videos), tuple(audios))


def validate_structure(
    inputs: dict[str, np.ndarray],
    plan: JointPlan,
    image_tokens: list[int],
    video_tokens: list[list[int]],
    audio_tokens: list[int],
) -> None:
    """Compare media structures for each row, part and frame in original order."""
    structural_ids = tuple(token_id for _, token_id in MEDIA_TOKENS.values())
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    if not np.array_equal(mask, np.arange(ids.shape[1])[None, :] < mask.sum(axis=1)[:, None]):
        raise EmbeddingConfigurationError("MLX joint attention masks must preserve a complete valid token prefix")
    if np.any(ids[mask == 0] != 0):
        raise EmbeddingConfigurationError("MLX joint padding must contain only the fixed padding token")
    for row_index, row in enumerate(plan.rows):
        expected = []
        for part in row:
            if part.kind == "image":
                expected.extend([255999] + [258880] * image_tokens[part.index] + [258882])
            elif part.kind == "video":
                for tokens in video_tokens[part.index]:
                    expected.extend([255999] + [258884] * tokens + [258882])
            elif part.kind == "audio":
                expected.extend([256000] + [258881] * audio_tokens[part.index] + [258883])
        valid_ids = ids[row_index][mask[row_index] == 1]
        if valid_ids[0] != 2 or valid_ids[-1] != 1:
            raise EmbeddingConfigurationError(f"MLX joint row {row_index} lost its fixed beginning/end tokens")
        structural = np.isin(ids[row_index], structural_ids)
        if not (mask[row_index][structural] == 1).all() or ids[row_index][structural].tolist() != expected:
            raise EmbeddingConfigurationError(
                f"MLX joint row {row_index} media structure must preserve each part's tokens, order and boundaries"
            )
