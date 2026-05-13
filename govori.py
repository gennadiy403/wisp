"""Compatibility shim for the package entry point."""

from __future__ import annotations

import sys

from govori.__main__ import main


if __name__ == "__main__":
    print("[deprecated] `python govori.py` - use `python -m govori` instead.", file=sys.stderr)
    main()
