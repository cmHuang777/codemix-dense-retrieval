#!/usr/bin/env python3
"""Compatibility dispatcher for the old combined CLI.

Use these dedicated scripts directly:
- case_miner.py
- case_inspector.py
"""

from __future__ import annotations

import sys
from typing import List, Optional

from case_inspector import main as inspector_main
from case_miner import main as miner_main


def _print_usage() -> None:
    print("usage: micro_case_tool.py {mine|inspect} [args...]")
    print("  mine     -> delegates to case_miner.py")
    print("  inspect  -> delegates to case_inspector.py")


def main(argv: Optional[List[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        _print_usage()
        return

    command, rest = args[0], args[1:]
    if command == "mine":
        miner_main(rest)
        return
    if command == "inspect":
        inspector_main(rest)
        return

    print(f"Unknown command: {command}")
    _print_usage()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
