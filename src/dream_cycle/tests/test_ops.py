"""Tests for dream_cycle.ops — v7 MemoryBackend routing."""

import pytest
from unittest.mock import patch, MagicMock

from dream_cycle.types import MemoryOp
from dream_cycle.ops import DirectBackend, StagingBackend, create_backend
from dream_cycle.budget import EditBudget


# ─── StagingBackend ───────────────────────────────────────────────────────────

class TestStagingBackend:
    def test_archive_staged(self):
        sb = StagingBackend()
        op = MemoryOp(op="archive", memory_id="m1", stage="dedup", reason="duplicate")
        assert sb.execute(op) is True
        assert len(sb.proposals) == 1
        assert sb.proposals[0].op == "archive"

    def test_update_text_staged(self):
        sb = StagingBackend()
        op = MemoryOp(op="update_text", memory_id="m2", new_text="merged", stage="merge")
        assert sb.execute(op) is True
        assert len(sb.proposals) == 1

    def test_delete_staged(self):
        sb = StagingBackend()
        op = MemoryOp(op="delete", memory_id="m3", stage="merge")
        assert sb.execute(op) is True
        assert len(sb.proposals) == 1

    def test_boost_staged(self):
        sb = StagingBackend()
        op = MemoryOp(op="boost", memory_id="m4", payload_patch={"importance": 0.9})
        assert sb.execute(op) is True
        assert len(sb.proposals) == 1

    def test_extend_staged(self):
        sb = StagingBackend()
        op = MemoryOp(op="extend", memory_id="m5", payload_patch={"extended_by": "x"})
        assert sb.execute(op) is True
        assert len(sb.proposals) == 1

    def test_degrade_staged(self):
        sb = StagingBackend()
        op = MemoryOp(op="degrade", memory_id="m6", new_text="short",
                      payload_patch={"dream_tier": "2"}, stage="tier_degradation")
        assert sb.execute(op) is True
        assert len(sb.proposals) == 1

    def test_buffer_stats(self):
        sb = StagingBackend()
        sb.execute(MemoryOp(op="archive", memory_id="a1", stage="dedup"))
        sb.execute(MemoryOp(op="boost", memory_id="b1", payload_patch={"x": 1}))
        stats = sb.stats()
        assert stats["staged"] == 2
        assert stats["skipped"] == 0

    def test_budget_gating_blocks_costly(self):
        budget = EditBudget(max_edits=1)
        sb = StagingBackend(budget=budget)
        # First costly op should pass
        op1 = MemoryOp(op="archive", memory_id="m1", stage="dedup", reason="r1")
        assert sb.execute(op1) is True
        # Second costly op should be blocked
        op2 = MemoryOp(op="archive", memory_id="m2", stage="dedup", reason="r2")
        assert sb.execute(op2) is False
        assert len(sb.skipped) == 1

    def test_budget_does_not_block_free_ops(self):
        budget = EditBudget(max_edits=0)
        sb = StagingBackend(budget=budget)
        # boost is free
        op = MemoryOp(op="boost", memory_id="m1", payload_patch={"importance": 0.9})
        assert sb.execute(op) is True
        assert len(sb.skipped) == 0


# ─── DirectBackend (mocked PG) ────────────────────────────────────────────────

class TestDirectBackend:
    @patch("dream_cycle.ops._pg_query")
    def test_archive_calls_pg(self, mock_pg):
        mock_pg.return_value = []
        db = DirectBackend()
        op = MemoryOp(op="archive", memory_id="m1", stage="dedup")
        assert db.execute(op) is True
        assert mock_pg.called
        sql = mock_pg.call_args[0][0]
        assert "UPDATE mem0" in sql
        assert "m1" in sql

    @patch("dream_cycle.ops._delete_mem")
    def test_delete_calls_delete_mem(self, mock_del):
        mock_del.return_value = True
        db = DirectBackend()
        op = MemoryOp(op="delete", memory_id="m1")
        assert db.execute(op) is True
        mock_del.assert_called_once_with("m1")

    @patch("dream_cycle.ops._update_text")
    def test_update_text_calls_helper(self, mock_upd):
        mock_upd.return_value = True
        db = DirectBackend()
        op = MemoryOp(op="update_text", memory_id="m1", new_text="new")
        assert db.execute(op) is True
        mock_upd.assert_called_once_with("m1", "new")

    @patch("dream_cycle.ops._pg_query")
    def test_boost_calls_pg(self, mock_pg):
        mock_pg.return_value = []
        db = DirectBackend()
        op = MemoryOp(op="boost", memory_id="m1", payload_patch={"importance": 0.9})
        assert db.execute(op) is True
        assert mock_pg.called

    @patch("dream_cycle.ops._pg_query")
    def test_extend_calls_pg(self, mock_pg):
        mock_pg.return_value = []
        db = DirectBackend()
        op = MemoryOp(op="extend", memory_id="m1", payload_patch={"extended_by": "x"})
        assert db.execute(op) is True
        sql = mock_pg.call_args[0][0]
        assert "extended" in sql

    @patch("dream_cycle.ops._pg_query")
    def test_degrade_updates_text_and_metadata(self, mock_pg):
        mock_pg.return_value = []
        db = DirectBackend()
        op = MemoryOp(op="degrade", memory_id="m1", new_text="short summary",
                      payload_patch={"dream_tier": "2", "degraded_at": "2026-06-28"})
        assert db.execute(op) is True
        sql = mock_pg.call_args[0][0]
        assert "jsonb_set" in sql  # updates data field
        assert "dream_tier" in sql  # updates metadata

    def test_unknown_op_returns_false(self):
        db = DirectBackend()
        # Create an op with invalid type by bypassing type checking
        op = MemoryOp(op="archive", memory_id="m1")
        op.op = "nonexistent"  # type: ignore
        assert db.execute(op) is False

    @patch("dream_cycle.ops._pg_query")
    def test_pg_error_returns_false(self, mock_pg):
        mock_pg.side_effect = Exception("PG down")
        db = DirectBackend()
        op = MemoryOp(op="archive", memory_id="m1", stage="dedup")
        assert db.execute(op) is False

    def test_proposals_always_empty(self):
        db = DirectBackend()
        assert db.proposals == []


# ─── Factory ──────────────────────────────────────────────────────────────────

class TestCreateBackend:
    def test_staging_by_default(self):
        b = create_backend(use_staging=True)
        assert isinstance(b, StagingBackend)

    def test_direct_when_no_staging(self):
        b = create_backend(use_staging=False)
        assert isinstance(b, DirectBackend)

    def test_budget_passed_to_staging(self):
        budget = EditBudget(max_edits=5)
        b = create_backend(use_staging=True, budget=budget)
        assert isinstance(b, StagingBackend)
        assert b._budget is budget

    def test_budget_ignored_for_direct(self):
        budget = EditBudget(max_edits=5)
        b = create_backend(use_staging=False, budget=budget)
        assert isinstance(b, DirectBackend)
