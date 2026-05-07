#!/usr/bin/env python3
"""
Dream Cycle CLI — thin wrapper that imports from src/dream_cycle.py

For direct usage, you can also run:
    python3 src/dream_cycle.py [options]
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import and run main
from dream_cycle import main  # noqa: E402

if __name__ == "__main__":
    main()
