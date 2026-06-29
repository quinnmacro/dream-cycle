"""Tests for dream_cycle.staging — StagingBuffer collection and stats."""

import pytest
from dream_cycle.staging import StagingBuffer, PGProposal, StagingResult


# ─── StagingBuffer Basic Ops ──────────────────────────────────────────────────

class TestStagingBufferOps:
    def test_add_update_payload(self):
        buf = StagingBuffer()
        buf.add_update_payload("m1", {"boost": True}, reason="important", stage="boost")
        assert len(buf.proposals) == 1
        p = buf.proposals[0]
        assert p.op == "update_payload"
        assert p.memory_id == "m1"
        assert p.payload_patch == {"boost": True}

    def test_add_update_text(self):
        buf = StagingBuffer()
        buf.add_update_text("m2", "new text", reason="merge", stage="merge")
        assert len(buf.proposals) == 1
        assert buf.proposals[0].op == "update_text"
        assert buf.proposals[0].new_text == "new text"

    def test_add_update_text_with_payload_patch(self):
        buf = StagingBuffer()
        buf.add_update_text("m2", "short", stage="degrade",
                            payload_patch={"dream_tier": "2"})
        assert buf.proposals[0].payload_patch == {"dream_tier": "2"}

    def test_add_archive(self):
        buf = StagingBuffer()
        buf.add_archive("m3", reason="duplicate", stage="dedup")
        assert len(buf.proposals) == 1
        p = buf.proposals[0]
        assert p.op == "update_payload"
        assert p.payload_patch["archived"] is True
        assert "m3" in buf.archived_ids

    def test_add_archive_with_extra_patch(self):
        buf = StagingBuffer()
        buf.add_archive("m3", reason="dup", stage="dedup",
                        payload_patch={"superseded_by": "m4"})
        p = buf.proposals[0]
        assert p.payload_patch["archived"] is True
        assert p.payload_patch["superseded_by"] == "m4"

    def test_add_delete(self):
        buf = StagingBuffer()
        buf.add_delete("m4", reason="merge secondary", stage="merge")
        assert len(buf.proposals) == 1
        assert buf.proposals[0].op == "delete"
        assert "m4" in buf.merged_ids


# ─── Removed IDs ──────────────────────────────────────────────────────────────

class TestRemovedIDs:
    def test_archived_are_removed(self):
        buf = StagingBuffer()
        buf.add_archive("a1")
        assert "a1" in buf.removed_ids

    def test_merged_are_removed(self):
        buf = StagingBuffer()
        buf.add_delete("d1")
        assert "d1" in buf.removed_ids

    def test_removed_is_union(self):
        buf = StagingBuffer()
        buf.add_archive("a1")
        buf.add_delete("d1")
        assert buf.removed_ids == {"a1", "d1"}

    def test_update_payload_not_removed(self):
        buf = StagingBuffer()
        buf.add_update_payload("b1", {"boost": True})
        assert "b1" not in buf.removed_ids

    def test_update_text_not_removed(self):
        buf = StagingBuffer()
        buf.add_update_text("t1", "new")
        assert "t1" not in buf.removed_ids


# ─── Stats ────────────────────────────────────────────────────────────────────

class TestStagingBufferStats:
    def test_empty_stats(self):
        buf = StagingBuffer()
        s = buf.stats()
        assert s["total_proposals"] == 0
        assert s["by_stage"] == {}
        assert s["by_op"] == {}
        assert s["archived_ids"] == 0
        assert s["merged_ids"] == 0

    def test_stats_counts(self):
        buf = StagingBuffer()
        buf.add_archive("a1", stage="dedup")
        buf.add_archive("a2", stage="dedup")
        buf.add_update_payload("b1", {"x": 1}, stage="boost")
        buf.add_delete("d1", stage="merge")
        s = buf.stats()
        assert s["total_proposals"] == 4
        assert s["by_stage"]["dedup"] == 2
        assert s["by_stage"]["boost"] == 1
        assert s["by_stage"]["merge"] == 1
        assert s["by_op"]["update_payload"] == 3  # 2 archives + 1 boost
        assert s["by_op"]["delete"] == 1
        assert s["archived_ids"] == 2
        assert s["merged_ids"] == 1


# ─── PGProposal Dataclass ─────────────────────────────────────────────────────

class TestPGProposal:
    def test_defaults(self):
        p = PGProposal(op="update_payload", memory_id="m1")
        assert p.payload_patch == {}
        assert p.new_text == ""
        assert p.reason == ""
        assert p.source_op == ""
        assert p.stage == ""

    def test_payload_patch_independent(self):
        p1 = PGProposal(op="update_payload", memory_id="a")
        p2 = PGProposal(op="update_payload", memory_id="b")
        p1.payload_patch["key"] = "val"
        assert "key" not in p2.payload_patch


# ─── StagingResult ────────────────────────────────────────────────────────────

class TestStagingResult:
    def test_basic_creation(self):
        r = StagingResult(
            staging_dir="/tmp/test",
            n_proposals=5,
            n_by_stage={"dedup": 3, "boost": 2},
            validation_accepted=True,
            validation_reason="all good",
        )
        assert r.adopted is False
        assert r.adopted_at == ""

    def test_adopted_state(self):
        r = StagingResult(
            staging_dir="/tmp/test",
            n_proposals=1,
            n_by_stage={},
            validation_accepted=True,
            validation_reason="ok",
            adopted=True,
            adopted_at="2026-06-28T04:00:00",
        )
        assert r.adopted is True
