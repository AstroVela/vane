# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

from typing_extensions import assert_type

import vane
from vane.ai import AudioInput, ImageInput, MultimodalInput, TextInput, VideoInput, embed

text = vane.col("text")
relation = cast(vane.Relation, None)

assert_type(embed(text, dimensions=4), vane.Expression)
assert_type(embed(input=text, dimensions=4), vane.Expression)
assert_type(embed(relation, text, dimensions=4), vane.Relation)
assert_type(embed(rel=relation, input=text, dimensions=4), vane.Relation)
assert_type(embed(relation, text, dimensions=4, output_column="vector"), vane.Relation)
assert_type(embed(rel=relation, input=text, dimensions=4, output_column="vector"), vane.Relation)
assert_type(relation.embed(text, dimensions=4), vane.Relation)

assert_type(
    embed(text, dimensions=4, request_batch_size=16, max_concurrency_per_actor=2, supports_overriding_dimensions=False),
    vane.Expression,
)
assert_type(embed(text, provider="transformers", input_type="query", overlength="error"), vane.Expression)
assert_type(
    embed(
        text,
        provider="litert",
        model_path="/models/embedding.litertlm",
        model_sha256="a" * 64,
        backend="cpu",
        num_threads=2,
        dimensions=256,
        input_type="query",
    ),
    vane.Expression,
)

image = vane.col("image")
assert_type(embed(ImageInput(image)), vane.Expression)
assert_type(embed(input=ImageInput(image)), vane.Expression)
assert_type(embed(relation, ImageInput(image)), vane.Relation)
assert_type(embed(rel=relation, input=ImageInput(image)), vane.Relation)
assert_type(relation.embed(ImageInput(image), normalize=True), vane.Relation)
assert_type(embed(ImageInput(image), dtype="float16"), vane.Expression)
assert_type(
    embed(
        ImageInput(image),
        provider="litert",
        model_path="/models/embedding.litertlm",
        model_sha256="a" * 64,
        vision_tokens_per_image=70,
    ),
    vane.Expression,
)

frames = vane.col("frames")
assert_type(embed(VideoInput(frames)), vane.Expression)
assert_type(embed(input=VideoInput(frames)), vane.Expression)
assert_type(embed(relation, VideoInput(frames)), vane.Relation)
assert_type(embed(rel=relation, input=VideoInput(frames)), vane.Relation)
assert_type(relation.embed(VideoInput(frames), dtype="float16"), vane.Relation)

audio = vane.col("audio")
assert_type(embed(AudioInput(audio)), vane.Expression)
assert_type(embed(input=AudioInput(audio)), vane.Expression)
assert_type(embed(relation, AudioInput(audio)), vane.Relation)
assert_type(embed(rel=relation, input=AudioInput(audio)), vane.Relation)
assert_type(relation.embed(AudioInput(audio), normalize=True), vane.Relation)

parts = MultimodalInput([VideoInput(frames), TextInput(text), AudioInput(audio), ImageInput(image)])
assert_type(embed(parts, provider="google", dimensions=768), vane.Expression)
assert_type(embed(relation, parts, provider="google", dimensions=768), vane.Relation)
assert_type(relation.embed(parts, provider="google", dimensions=768), vane.Relation)
assert_type(embed(VideoInput(frames, mime_type="video/mp4"), provider="google"), vane.Expression)
