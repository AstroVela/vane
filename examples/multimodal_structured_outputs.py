#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Multimodal structured-output evaluation with Vane.

This example adapts the evaluation shape from Daft's multimodal structured
outputs tutorial to Vane:
https://docs.daft.ai/en/stable/examples/mm_structured_outputs/

1. Ask a vision-language model multiple-choice questions with an image.
2. Ask the same questions without the image.
3. Compare the two runs to identify when the image helped or hurt.
4. Optionally run a VLM-as-a-judge pass on failure cases.

The default data source is synthetic and self-contained. Use
``--source hf-ai2d`` to sample the AI2D subset of The Cauldron dataset.

Install the runtime pieces you need, for example:

    pip install "vane-ai[all]" openai pydantic

For the Hugging Face AI2D source, also install:

    pip install datasets pillow

The default provider is the Hugging Face OpenAI-compatible router. Set
``HF_TOKEN`` before running, or pass ``--api-key-env`` for another endpoint.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import zlib
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, Field

import vane
from vane.ai import prompt

DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
CHOICE_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


class ChoiceResponse(BaseModel):
    """Structured answer for a multiple-choice question."""

    choice: str = Field(
        ...,
        description="The selected answer letter, such as A, B, C, or D.",
    )


class JudgeResponse(BaseModel):
    """Structured diagnostic feedback for a failed example."""

    reasoning: str = Field(..., description="Why the model likely answered that way.")
    hypothesis: str = Field(..., description="A concise cause of the error.")
    attribution: str = Field(
        ...,
        description="One of: question, image, model, or other.",
    )


VISION_SYSTEM_PROMPT = (
    "You answer visual multiple-choice questions. Use the attached image and "
    "the text question. Return exactly one JSON object, such as "
    '{"choice": "A"}. The choice must be one of A, B, C, or D.'
)

TEXT_ONLY_SYSTEM_PROMPT = (
    "You answer multiple-choice questions using only the text. No image is "
    "available. Return exactly one JSON object, such as "
    '{"choice": "A"}. The choice must be one of A, B, C, or D.'
)

JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluator for a vision-language benchmark. Review "
    "the question, image, correct answer, model answer with the image, and "
    "model answer without the image. Explain the likely failure mode and "
    "attribute it to question, image, model, or other. Return exactly one JSON "
    "object with reasoning, hypothesis, and attribution fields."
)


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Create a small RGB PNG using only the Python standard library."""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )


def synthetic_rows(limit: int) -> list[dict[str, Any]]:
    """Return a tiny local vision QA set that needs the image to answer."""
    color_rows = [
        ("red", "A", (220, 32, 32)),
        ("blue", "B", (32, 96, 220)),
        ("green", "C", (32, 160, 80)),
        ("yellow", "D", (230, 200, 32)),
    ]
    rows: list[dict[str, Any]] = []
    choices = "A. red\nB. blue\nC. green\nD. yellow"
    for color, answer, rgb in color_rows:
        rows.append(
            {
                "id": f"synthetic-{color}",
                "source": "synthetic",
                "question": (f"Which color fills the attached square?\n{choices}\nReturn the answer letter."),
                "answer": answer,
                "image": solid_png(96, 96, rgb),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def normalize_choice(text: Any) -> str | None:
    """Extract an A-D answer letter from text-like input."""
    if text is None:
        return None
    match = CHOICE_RE.search(str(text))
    if match:
        return match.group(1).upper()
    return None


def image_to_bytes(value: Any) -> bytes | None:
    """Normalize common dataset image representations to image bytes."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, dict):
        if isinstance(value.get("bytes"), bytes):
            return value["bytes"]
        if value.get("path"):
            with open(value["path"], "rb") as image_file:
                return image_file.read()
    if hasattr(value, "save"):
        import io

        buffer = io.BytesIO()
        image_format = getattr(value, "format", None) or "PNG"
        value.save(buffer, format=image_format)
        return buffer.getvalue()
    raise TypeError(f"Unsupported image value: {type(value)!r}")


def first_or_none(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def load_hf_ai2d_rows(limit: int, dataset_path: str, dataset_config: str) -> list[dict[str, Any]]:
    """Load a small AI2D sample from The Cauldron via Hugging Face datasets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install optional dataset dependencies first: pip install datasets pillow") from exc

    try:
        dataset = load_dataset(
            dataset_path,
            dataset_config,
            split="train",
            streaming=True,
        )
    except Exception:
        dataset = load_dataset(
            f"{dataset_path}/{dataset_config}",
            split="train",
            streaming=True,
        )

    rows: list[dict[str, Any]] = []
    for raw in dataset:
        texts = raw.get("texts") or []
        user_text = None
        answer = None

        for turn in texts:
            if not isinstance(turn, dict):
                continue
            if user_text is None and turn.get("user"):
                user_text = str(turn["user"])
            if answer is None and turn.get("assistant"):
                answer = normalize_choice(turn["assistant"])

        image = image_to_bytes(first_or_none(raw.get("images")))
        if user_text and answer and image:
            rows.append(
                {
                    "id": f"hf-ai2d-{len(rows)}",
                    "source": "hf-ai2d",
                    "question": user_text,
                    "answer": answer,
                    "image": image,
                }
            )
        if len(rows) >= limit:
            break

    if not rows:
        raise SystemExit("No usable AI2D rows were loaded.")
    return rows


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def relation_from_rows(
    conn: Any,
    rows: list[dict[str, Any]],
    schema: dict[str, str],
) -> Any:
    """Build a VALUES relation whose data can travel with a Ray plan."""
    if not rows:
        raise RuntimeError("Cannot create a relation from zero rows.")
    columns = list(schema)
    constant = vane.ConstantExpression
    raw = conn.values(
        *(tuple(constant(row[column]) for column in columns) for row in rows),
    )
    projections = [
        f"{quote_ident(source)}::{schema[column]} AS {quote_ident(column)}"
        for source, column in zip(raw.columns, columns, strict=True)
    ]
    return raw.query("input_rows", f"select {', '.join(projections)} from input_rows")


def rows_to_relation(conn: Any, rows: list[dict[str, Any]]) -> Any:
    return relation_from_rows(
        conn,
        rows,
        {
            "id": "VARCHAR",
            "source": "VARCHAR",
            "question": "VARCHAR",
            "answer": "VARCHAR",
            "image": "BLOB",
        },
    )


def collect_relation(rel: Any) -> pa.Table:
    """Materialize a relation through the configured default runner."""
    tables = list(vane.runners.get_or_create_runner().run_iter_tables(rel))
    if not tables:
        return pa.table({column: pa.array([]) for column in rel.columns})
    table = pa.concat_tables(tables)
    expected_columns = list(rel.columns)
    if table.column_names != expected_columns:
        table = table.rename_columns(expected_columns)
    return table


def count_rows(rel: Any) -> int:
    table = collect_relation(rel.query("r", "select count(*) as row_count from r"))
    return int(table["row_count"][0].as_py())


def relation_alias(prefix: str, rel: Any) -> str:
    return f"__vane_{prefix}_{id(rel):x}"


def append_prompt_output(conn: Any, rel: Any, prompted: Any, output_column: str) -> Any:
    """Append a prompt() result column back to the input relation."""
    base = collect_relation(rel)
    output = collect_relation(prompted)
    if base.num_rows != output.num_rows:
        raise RuntimeError(f"Prompt output row count mismatch: {output.num_rows} vs {base.num_rows}")
    if output_column not in output.column_names:
        raise RuntimeError(f"Prompt output column {output_column!r} was not returned.")
    combined = base.append_column(output_column, output[output_column])
    schema = {column: str(type_) for column, type_ in zip(rel.columns, rel.types, strict=True)}
    output_index = list(prompted.columns).index(output_column)
    schema[output_column] = str(prompted.types[output_index])
    return relation_from_rows(conn, combined.to_pylist(), schema)


def print_relation(title: str, rel: Any, sql: str) -> None:
    print(f"\n{title}")
    rel.query("r", sql).show()


def add_choice_eval(rel: Any, response_col: str, choice_col: str, correct_col: str) -> Any:
    alias = relation_alias("choice", rel)
    normalized_choice = (
        "upper(substr(trim(coalesce("
        f"try(json_extract_string({response_col}, '$.choice')), "
        f"nullif(regexp_extract(coalesce({response_col}, ''), '\"choice\"\\s*:\\s*\"([A-D])\"', 1), ''), "
        f"nullif(regexp_extract(coalesce({response_col}, ''), '\\b([A-D])\\b', 1), ''), "
        "'')), 1, 1))"
    )
    return rel.query(
        alias,
        f"""
        select
        {alias}.*,
        {normalized_choice} as {choice_col},
        upper(trim(answer)) = {normalized_choice} as {correct_col}
        from {alias}
        """,
    )


def classify_quadrants(rel: Any) -> Any:
    alias = relation_alias("quadrant", rel)
    return rel.query(
        alias,
        f"""
        select
        {alias}.*,
        case
            when is_correct_with_image and is_correct_without_image then 'Both Correct'
            when is_correct_with_image and not is_correct_without_image then 'Image Helped'
            when not is_correct_with_image and is_correct_without_image then 'Image Hurt'
            else 'Both Incorrect'
        end as quadrant
        from {alias}
        """,
    )


def run(args: argparse.Namespace) -> None:
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1.")

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set {args.api_key_env} before running this example.")

    base_url = args.base_url
    if base_url is None:
        base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    if base_url.lower() in {"", "none", "default"}:
        base_url = None

    rows = (
        synthetic_rows(args.limit)
        if args.source == "synthetic"
        else load_hf_ai2d_rows(args.limit, args.dataset_path, args.dataset_config)
    )

    conn = vane.connect()
    rel = rows_to_relation(conn, rows)

    common_prompt_options = {
        "provider": "openai",
        "model": args.model,
        "api_key": api_key,
        "use_chat_completions": True,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if base_url:
        common_prompt_options["base_url"] = base_url

    choice_return_format = ChoiceResponse if args.structured_output_mode == "parse" else None
    judge_return_format = JudgeResponse if args.structured_output_mode == "parse" else None

    with_image_responses = prompt(
        rel,
        "question",
        image_columns=["image"],
        system_message=VISION_SYSTEM_PROMPT,
        return_format=choice_return_format,
        output_column="response_with_image",
        **common_prompt_options,
    )
    with_image = append_prompt_output(
        conn,
        rel,
        with_image_responses,
        "response_with_image",
    )
    eval_with_image = add_choice_eval(
        with_image,
        "response_with_image",
        "predicted_with_image",
        "is_correct_with_image",
    )

    without_image_responses = prompt(
        eval_with_image,
        "question",
        system_message=TEXT_ONLY_SYSTEM_PROMPT,
        return_format=choice_return_format,
        output_column="response_without_image",
        **common_prompt_options,
    )
    without_image = append_prompt_output(
        conn,
        eval_with_image,
        without_image_responses,
        "response_without_image",
    )
    evaluated = add_choice_eval(
        without_image,
        "response_without_image",
        "predicted_without_image",
        "is_correct_without_image",
    )
    classified = classify_quadrants(evaluated)

    total = count_rows(classified)
    with_correct = count_rows(classified.filter("is_correct_with_image"))
    without_correct = count_rows(classified.filter("is_correct_without_image"))

    print(f"\nRows: {total}")
    print(f"Accuracy with image:    {with_correct / total:.1%}")
    print(f"Accuracy without image: {without_correct / total:.1%}")
    print(f"Delta:                  {(with_correct - without_correct) / total:+.1%}")

    print_relation(
        "Predictions",
        classified,
        """
        select
            id,
            answer,
            predicted_with_image,
            predicted_without_image,
            is_correct_with_image,
            is_correct_without_image,
            quadrant
        from r
        order by id
        """,
    )
    print_relation(
        "Quadrants",
        classified,
        """
        select quadrant, count(*) as count
        from r
        group by quadrant
        order by count desc, quadrant
        """,
    )

    if args.skip_judge:
        return

    failures = classified.filter("quadrant in ('Image Hurt', 'Both Incorrect')")
    if count_rows(failures) == 0:
        print("\nNo failure rows to judge.")
        return

    judge_alias = relation_alias("judge", failures)
    judge_input = failures.query(
        judge_alias,
        f"""
        select
        {judge_alias}.*,
        'Question:\n' || question ||
        '\n\nCorrect answer: ' || answer ||
        '\nModel answer with image: ' || predicted_with_image ||
        '\nModel answer without image: ' || predicted_without_image ||
        '\n\nProvide diagnostic feedback.' as judge_prompt
        from {judge_alias}
        """,
    )

    judge_responses = prompt(
        judge_input,
        "judge_prompt",
        image_columns=["image"],
        system_message=JUDGE_SYSTEM_PROMPT,
        return_format=judge_return_format,
        output_column="judge_response",
        max_tokens=args.judge_max_tokens,
        **{key: value for key, value in common_prompt_options.items() if key != "max_tokens"},
    )
    judged = append_prompt_output(
        conn,
        judge_input,
        judge_responses,
        "judge_response",
    )

    print_relation(
        "Judge Feedback",
        judged,
        """
        select
            id,
            quadrant,
            json_extract_string(judge_response, '$.attribution') as attribution,
            json_extract_string(judge_response, '$.hypothesis') as hypothesis,
            json_extract_string(judge_response, '$.reasoning') as reasoning
        from r
        order by id
        """,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Vane multimodal structured-output evaluation.",
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "hf-ai2d"],
        default="synthetic",
        help="Data source. synthetic is local; hf-ai2d samples The Cauldron AI2D.",
    )
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="HF_TOKEN")
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--structured-output-mode",
        choices=["parse", "prompt"],
        default="parse",
        help=(
            "parse uses SDK structured-output parsing; prompt asks the model to "
            "emit JSON directly for OpenAI-compatible servers without parse()."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--dataset-path", default="HuggingFaceM4/the_cauldron")
    parser.add_argument("--dataset-config", default="ai2d")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
