#!/usr/bin/env python3
"""Convenience entry point so `python main.py` works without installation."""

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contrastive_decoding.main import main as _main


if __name__ == "__main__":
    _main()
