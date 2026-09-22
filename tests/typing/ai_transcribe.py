# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from typing_extensions import assert_type

import vane
from vane.ai import TranscribeOptions, transcribe


def check(rel: vane.Relation, audio: vane.Expression) -> None:
    options: TranscribeOptions = {"language": "en", "max_retries": 0}
    assert_type(transcribe(audio, **options), vane.Expression)
    assert_type(transcribe(audio=audio, model="whisper-1"), vane.Expression)
    assert_type(transcribe(rel, audio), vane.Relation)
    assert_type(transcribe(rel=rel, audio=audio, output_column="speech"), vane.Relation)
    assert_type(rel.transcribe(audio), vane.Relation)
