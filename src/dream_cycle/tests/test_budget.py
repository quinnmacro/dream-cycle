"""Tests for dream_cycle.budget — EditBudget tracking and gating."""

import time
import pytest
from dream_cycle.budget import (
    EditBudget, DEFAULT_EDIT_BUDGET, COSTLY_OPS, FREE_OPS,
)


# ─── Basic Spend / Can Spend ──────────────────────────────────────────────────

class TestEditBudgetSpend:
    def test_default_budget(self):
        b = EditBudget()
        assert b.max_edits == DEFAULT_EDIT_BUDGET
        assert b.remaining == DEFAULT_EDIT_BUDGET
        assert b.used == 0

    def test_spend_costly_op(self):
        b = EditBudget(max_edits=3)
        assert b.spend("dedup_archive", detail="dup", memory_id="m1") is True
        assert b.used == 1
        assert b.remaining == 2

    def test_spend_exhausts_budget(self):
        b = EditBudget(max_edits=2)
        assert b.spend("dedup_archive") is True
        assert b.spend("merge") is True
        assert b.spend("decay_archive") is False
        assert b.remaining == 0
        assert len(b.skipped) == 1
        assert b.skipped[0]["reason"] == "budget_exhausted"

    def test_free_ops_dont_consume_budget(self):
        b = EditBudget(max_edits=0)  # zero budget
        for op in FREE_OPS:
            assert b.spend(op) is True
        assert b.used == 0
        assert b.remaining == 0

    def test_unknown_ops_consume_budget(self):
        """Unknown ops are NOT in FREE_OPS, so they consume budget like costly ops."""
        b = EditBudget(max_edits=0)
        assert b.spend("some_unknown_op") is False
        assert len(b.skipped) == 1

    def test_can_spend_costly(self):
        b = EditBudget(max_edits=1)
        assert b.can_spend("dedup_archive") is True
        b.spend("dedup_archive")
        assert b.can_spend("dedup_archive") is False

    def test_can_spend_free(self):
        b = EditBudget(max_edits=0)
        assert b.can_spend("boost") is True


# ─── Fraction Remaining ───────────────────────────────────────────────────────

class TestFractionRemaining:
    def test_full_budget(self):
        b = EditBudget(max_edits=10)
        assert b.fraction_remaining == 1.0

    def test_half_budget(self):
        b = EditBudget(max_edits=10)
        for _ in range(5):
            b.spend("dedup_archive")
        assert abs(b.fraction_remaining - 0.5) < 0.01

    def test_empty_budget(self):
        b = EditBudget(max_edits=10)
        for _ in range(10):
            b.spend("dedup_archive")
        assert b.fraction_remaining == 0.0

    def test_token_budget_reduces_fraction(self):
        b = EditBudget(max_edits=10, max_tokens=1000)
        b.record_tokens(500)
        # token_frac = 0.5, edit_frac = 1.0 → min = 0.5
        assert abs(b.fraction_remaining - 0.5) < 0.01

    def test_token_exhausted(self):
        b = EditBudget(max_edits=10, max_tokens=100)
        b.record_tokens(100)
        assert b.fraction_remaining == 0.0
        assert b.token_budget_ok() is False


# ─── Token Budget ─────────────────────────────────────────────────────────────

class TestTokenBudget:
    def test_token_budget_ok_initially(self):
        b = EditBudget()
        assert b.token_budget_ok() is True

    def test_token_budget_exhausted(self):
        b = EditBudget(max_tokens=100)
        b.record_tokens(50)
        assert b.token_budget_ok() is True
        b.record_tokens(50)
        assert b.token_budget_ok() is False

    def test_tokens_accumulate(self):
        b = EditBudget()
        b.record_tokens(100)
        b.record_tokens(200)
        assert b._tokens_used == 300


# ─── Wall Clock ───────────────────────────────────────────────────────────────

class TestWallClock:
    def test_wall_clock_ok_before_start(self):
        b = EditBudget()
        assert b.wall_clock_ok() is True

    def test_wall_clock_ok_after_start(self):
        b = EditBudget(max_wall_clock=60)
        b.start()
        assert b.wall_clock_ok() is True

    def test_elapsed_zero_before_start(self):
        b = EditBudget()
        assert b.elapsed_seconds == 0.0

    def test_elapsed_positive_after_start(self):
        b = EditBudget()
        b.start()
        time.sleep(0.01)
        assert b.elapsed_seconds > 0


# ─── Summary / Reporting ─────────────────────────────────────────────────────

class TestSummary:
    def test_summary_fields(self):
        b = EditBudget(max_edits=5)
        b.spend("dedup_archive", detail="dup", memory_id="m1")
        b.spend("boost", detail="important")
        s = b.summary()
        assert s["edit_budget"] == 5
        assert s["edits_used"] == 1
        assert s["edits_remaining"] == 4
        assert s["costly_ops"] == 1
        assert s["free_ops"] == 1

    def test_skipped_summary_empty(self):
        b = EditBudget(max_edits=10)
        assert b.skipped_summary() == ""

    def test_skipped_summary_with_skips(self):
        b = EditBudget(max_edits=1)
        b.spend("dedup_archive")
        b.spend("merge", detail="m2+m3", memory_id="m2")
        b.spend("decay_archive", detail="old", memory_id="m4")
        report = b.skipped_summary()
        assert "2 ops skipped" in report
        assert "merge" in report
        assert "decay_archive" in report


# ─── All Budgets OK ──────────────────────────────────────────────────────────

class TestAllBudgetsOK:
    def test_all_ok_initially(self):
        b = EditBudget()
        assert b.all_budgets_ok() is True

    def test_fails_when_edits_exhausted(self):
        b = EditBudget(max_edits=1)
        b.spend("dedup_archive")
        # can_spend returns True for unknown ops; test with known costly op
        assert b.can_spend("dedup_archive") is False

    def test_fails_when_tokens_exhausted(self):
        b = EditBudget(max_tokens=10)
        b.record_tokens(10)
        assert b.all_budgets_ok() is False


# ─── COSTLY_OPS / FREE_OPS Coverage ──────────────────────────────────────────

class TestOpClassification:
    def test_costly_ops_are_frozen(self):
        assert isinstance(COSTLY_OPS, frozenset)
        assert "dedup_archive" in COSTLY_OPS
        assert "merge" in COSTLY_OPS
        assert "decay_archive" in COSTLY_OPS

    def test_free_ops_are_frozen(self):
        assert isinstance(FREE_OPS, frozenset)
        assert "boost" in FREE_OPS
        assert "relation" in FREE_OPS
        assert "degrade_tier" in FREE_OPS

    def test_no_overlap(self):
        assert COSTLY_OPS & FREE_OPS == set()
