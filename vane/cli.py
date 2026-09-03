# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for declarative Vane pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from vane import Error as VaneError
from vane.pipeline import PipelineConfigError, execute_pipeline, load_pipeline


def _parameter(value: str) -> tuple[str, str]:
    name, separator, parameter_value = value.partition("=")
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError("parameters must use NAME=VALUE")
    return name.strip(), parameter_value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vane", description="Run declarative Vane pipelines.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="validate and run a YAML pipeline")
    run.add_argument("pipeline", help="path to pipeline.yaml")
    run.add_argument("--param", action="append", default=[], type=_parameter, metavar="NAME=VALUE")
    run.add_argument("--check", action="store_true", help="validate and render configuration without executing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Vane command-line interface."""
    args = _parser().parse_args(argv)
    try:
        parameters = dict(args.param)
        if len(parameters) != len(args.param):
            raise PipelineConfigError("pipeline parameters must not be repeated")
        config = load_pipeline(args.pipeline, parameters)
        if args.check:
            print(json.dumps(config, indent=2, sort_keys=True))
            return 0
        execute_pipeline(config)
    except (OSError, PipelineConfigError, VaneError) as error:
        print(f"vane: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
