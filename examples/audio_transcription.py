# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Run explicitly against a configured OpenAI endpoint; may incur API charges.

Install vane-ai[openai], export OPENAI_API_KEY, then run this file with a local
or shared audio path. Ray workers must be able to read the path. The default
runner is Ray; transcribe returns model-aligned segments in the input language.
"""

import argparse

import vane
from vane.ai import transcribe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    args = parser.parse_args()
    with vane.connect() as conn:
        source = conn.sql("SELECT $audio AS audio", params={"audio": args.audio})
        result = transcribe(source, vane.col("audio"), model="whisper-1", max_retries=0)
        print(result.to_arrow_table().to_pylist())


if __name__ == "__main__":
    main()
