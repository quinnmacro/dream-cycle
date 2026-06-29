"""Tests for dream_cycle.types — v7 dataclass contracts."""

import pytest
from dream_cycle.types import (
    DreamMemory, MemoryOp, PrepareResult, BudgetSummary, ExecuteResult,
)


# ─── DreamMemory ──────────────────────────────────────────────────────────────

class TestDreamMemory:
    def test_from_dict_basic(self):
        d = {"id": "abc-123", "text": "hello world", "created_at": "2026-06-01"}
        m = DreamMemory.from_dict(d)
        assert m.id == "abc-123"
        assert m.text == "hello world"
        assert m.created_at == "2026-06-01"
        assert m._split == "train"  # default
        assert m.importance == 0.5  # default

    def test_from_dict_with_v6_fields(self):
        d = {"id": "x", "text": "t", "_split": "val", "_outcome": "success", "_recalled": True}
        m = DreamMemory.from_dict(d)
        assert m._split == "val"
        assert m._outcome == "success"
        assert m._recalled is True

    def test_to_dict_roundtrip(self):
        d = {"id": "abc", "text": "hello", "source": "session", "custom_field": 42}
        m = DreamMemory.from_dict(d)
        result = m.to_dict()
        assert result["id"] == "abc"
        assert result["text"] == "hello"
        assert result["custom_field"] == 42  # preserved from _raw

    def test_from_dict_empty(self):
        m = DreamMemory.from_dict({})
        assert m.id == ""
        assert m.text == ""
        assert m._raw == {}


# ─── MemoryOp ──────────────────────────────────────────────────────────────────

class TestMemoryOp:
    def test_archive_is_costly(self):
        op = MemoryOp(op="archive", memory_id="m1", stage="dedup")
        assert op.is_costly() is True

    def test_update_text_is_costly(self):
        op = MemoryOp(op="update_text", memory_id="m1", new_text="merged")
        assert op.is_costly() is True

    def test_delete_is_costly(self):
        op = MemoryOp(op="delete", memory_id="m1")
        assert op.is_costly() is True

    def test_extend_is_costly(self):
        op = MemoryOp(op="extend", memory_id="m1", payload_patch={"extended": True})
        assert op.is_costly() is True

    def test_boost_is_free(self):
        op = MemoryOp(op="boost", memory_id="m1", payload_patch={"importance": 0.9})
        assert op.is_costly() is False

    def test_degrade_is_free(self):
        op = MemoryOp(op="degrade", memory_id="m1", new_text="short", payload_patch={"dream_tier": "2"})
        assert op.is_costly() is False

    def test_default_fields(self):
        op = MemoryOp(op="boost", memory_id="m1")
        assert op.stage == ""
        assert op.payload_patch == {}
        assert op.new_text == ""
        assert op.reason == ""
        assert op.superseded_by == ""

    def test_payload_patch_independent(self):
        """Each MemoryOp gets its own dict (no shared default)."""
        op1 = MemoryOp(op="boost", memory_id="a")
        op2 = MemoryOp(op="boost", memory_id="b")
        op1.payload_patch["key"] = "val"
        assert "key" not in op2.payload_patch


# ─── BudgetSummary ─────────────────────────────────────────────────────────────

class TestBudgetSummary:
    def test_can_spend_costly_when_budget_available(self):
        bs = BudgetSummary(edits_remaining=5, fraction_remaining=0.8)
        assert bs.can_spend_costly() is True

    def test_cannot_spend_when_no_edits(self):
        bs = BudgetSummary(edits_remaining=0, fraction_remaining=0.8)
        assert bs.can_spend_costly() is False

    def test_cannot_spend_when_low_fraction(self):
        bs = BudgetSummary(edits_remaining=5, fraction_remaining=0.02)
        assert bs.can_spend_costly() is False

    def test_deep_planning_ok_above_threshold(self):
        bs = BudgetSummary(fraction_remaining=0.25)
        assert bs.deep_planning_ok() is True

    def test_deep_planning_denied_below_threshold(self):
        bs = BudgetSummary(fraction_remaining=0.15)
        assert bs.deep_planning_ok() is False

    def test_deep_planning_custom_threshold(self):
        bs = BudgetSummary(fraction_remaining=0.35)
        assert bs.deep_planning_ok(threshold=0.30) is True
        assert bs.deep_planning_ok(threshold=0.40) is False


# ─── PrepareResult / ExecuteResult ─────────────────────────────────────────────

class TestPipelineResults:
    def test_prepare_result_defaults(self):
        pr = PrepareResult(memories=[], sessions=[], signals={})
        assert pr.feedback == {}

    def test_execute_result_defaults(self):
        er = ExecuteResult()
        assert er.clusters == {}
        assert er.stats == {}
        assert er.staging_info == {}
