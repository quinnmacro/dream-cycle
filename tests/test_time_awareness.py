#!/usr/bin/env python3
"""Tests for time-awareness module."""

import sys
sys.path.insert(0, "src")
from dream_cycle import _compute_memory_age_days, _is_time_sensitive

def test_age_days():
    assert _compute_memory_age_days("2026-05-07T10:00:00+08:00") < 1.0
    assert _compute_memory_age_days("2026-04-23T10:00:00+08:00") > 10.0
    assert _compute_memory_age_days(None) is None

def test_time_sensitive():
    assert _is_time_sensitive("CGB 10Y yield at 1.65%") is True
    assert _is_time_sensitive("Docker compose restart needed") is False
    assert _is_time_sensitive("利差走阔25bp") is True
    assert _is_time_sensitive("The project uses Polars") is False

if __name__ == "__main__":
    test_age_days()
    test_time_sensitive()
    print("✅ All tests passed")
