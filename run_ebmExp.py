#!/usr/bin/env python3
"""
Backward-compatible entry point: forwards to ``scripts/run_ebmExp.py``.

Always run from the repository root (or use ``python scripts/run_ebmExp.py``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    script = root / "scripts" / "run_ebmExp.py"
    raise SystemExit(
        subprocess.call([sys.executable, str(script)] + sys.argv[1:], cwd=str(root))
    )


if __name__ == "__main__":
    main()
