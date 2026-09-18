# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""CLI for the snapshot synchronizer and its OpenWiki MCP transport."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from contextlib import closing
from pathlib import Path

from vane.knowledge.config import Config
from vane.knowledge.state import State


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize Iceberg documents to OpenWiki and GBrain")
    commands = parser.add_subparsers(dest="command", required=True)
    sync = commands.add_parser("sync", help="Poll snapshots and deliver document changes")
    sync.add_argument("--config", type=Path, required=True)
    sync.add_argument("--once", action="store_true")
    serve = commands.add_parser("serve", help="Serve the pending batch over MCP stdio")
    serve.add_argument("--state", type=Path, required=True)
    snippet = commands.add_parser("openwiki-config", help="Print a sourceInstances entry for OpenWiki onboarding.json")
    snippet.add_argument("--config", type=Path, required=True)
    snippet.add_argument("--target", required=True)
    args = parser.parse_args()
    if args.command == "serve":
        from vane.knowledge.mcp import create_server

        create_server(args.state.absolute()).run()
        return 0
    config = Config.load(args.config)
    if args.command == "openwiki-config":
        from vane.knowledge.mcp import openwiki_source

        target = next((t for t in config.targets if t.name == args.target and t.kind == "openwiki"), None)
        if target is None:
            raise ValueError("Requested OpenWiki target is not configured")
        print(json.dumps(openwiki_source(config, target), ensure_ascii=False, indent=2))
        return 0

    from vane.knowledge.service import sync_once
    from vane.knowledge.targets import DeliveryError

    def interrupt(_signal: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    with closing(State(config.state_dir, create=True)) as state, state.exclusive():
        while True:
            try:
                changed = sync_once(config, state)
                print("Batch delivered" if changed else "No document changes", file=sys.stderr)
            except DeliveryError as exc:
                if args.once:
                    raise
                print(str(exc), file=sys.stderr)
            if args.once:
                return 0
            time.sleep(config.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
