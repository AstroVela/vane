# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

from typing_extensions import assert_type

import vane
from vane.ai import embed, embed_audio, embed_image, embed_video

text = vane.col("text")
relation = cast(vane.Relation, None)

assert_type(embed(text, dimensions=4), vane.Expression)
assert_type(embed(text=text, dimensions=4), vane.Expression)
assert_type(embed(relation, text, dimensions=4), vane.Relation)
assert_type(embed(rel=relation, text=text, dimensions=4), vane.Relation)
assert_type(embed(relation, text, dimensions=4, output_column="vector"), vane.Relation)
assert_type(embed(rel=relation, text=text, dimensions=4, output_column="vector"), vane.Relation)
assert_type(relation.embed(text, dimensions=4), vane.Relation)

assert_type(
    embed(text, dimensions=4, request_batch_size=16, max_concurrency_per_actor=2, supports_overriding_dimensions=False),
    vane.Expression,
)
assert_type(embed(text, provider="transformers", input_type="query", overlength="error"), vane.Expression)

image = vane.col("image")
assert_type(embed_image(image), vane.Expression)
assert_type(embed_image(image=image), vane.Expression)
assert_type(embed_image(relation, image), vane.Relation)
assert_type(embed_image(rel=relation, image=image), vane.Relation)
assert_type(relation.embed_image(image, normalize=True), vane.Relation)

frames = vane.col("frames")
assert_type(embed_video(frames), vane.Expression)
assert_type(embed_video(frames=frames), vane.Expression)
assert_type(embed_video(relation, frames), vane.Relation)
assert_type(embed_video(rel=relation, frames=frames), vane.Relation)
assert_type(relation.embed_video(frames, dtype="float16"), vane.Relation)

audio = vane.col("audio")
assert_type(embed_audio(audio), vane.Expression)
assert_type(embed_audio(audio=audio), vane.Expression)
assert_type(embed_audio(relation, audio), vane.Relation)
assert_type(embed_audio(rel=relation, audio=audio), vane.Relation)
assert_type(relation.embed_audio(audio, normalize=True), vane.Relation)
