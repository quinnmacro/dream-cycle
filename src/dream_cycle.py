#!/usr/bin/env python3
"""
Backward-compatible wrapper for Dream Cycle.

Usage:
    python3 dream_cycle.py [args...]

Equivalent to:
    python3 -m dream_cycle [args...]

This wrapper exists for cron jobs and scripts that reference the old monolith path.
"""
import sys
import os

# Ensure the parent directory is on the path so `dream_cycle` package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dream_cycle.__main__ import main

if __name__ == "__main__":
    main()
