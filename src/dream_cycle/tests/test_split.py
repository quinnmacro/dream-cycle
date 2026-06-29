"""Tests for dream_cycle.split — deterministic train/val/test splitting."""

import pytest
from dream_cycle.split import (
    _hash_bucket, assign_split, split_memories, split_stats,
    get_val_memories, get_test_memories, get_train_memories,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)


# ─── Hash Bucket ──────────────────────────────────────────────────────────────

class TestHashBucket:
    def test_returns_0_to_9(self):
        for i in range(100):
            b = _hash_bucket(f"memory-{i}")
            assert 0 <= b <= 9

    def test_deterministic(self):
        """Same ID always maps to same bucket."""
        for _ in range(3):
            assert _hash_bucket("abc-123") == _hash_bucket("abc-123")

    def test_different_ids_vary(self):
        """Not all IDs land in the same bucket."""
        buckets = {_hash_bucket(f"mem-{i}") for i in range(100)}
        assert len(buckets) > 5  # at least 6 distinct buckets

    def test_empty_string(self):
        b = _hash_bucket("")
        assert 0 <= b <= 9


# ─── Assign Split ─────────────────────────────────────────────────────────────

class TestAssignSplit:
    def test_returns_valid_split(self):
        for i in range(50):
            s = assign_split(f"m-{i}")
            assert s in ("train", "val", "test")

    def test_deterministic(self):
        assert assign_split("fixed-id") == assign_split("fixed-id")

    def test_distribution_approximately_correct(self):
        """With 1000 IDs, distribution should be roughly 70/20/10."""
        counts = {"train": 0, "val": 0, "test": 0}
        n = 1000
        for i in range(n):
            s = assign_split(f"dist-test-{i}")
            counts[s] += 1

        # Allow ±15% tolerance (hash isn't perfectly uniform on small n)
        assert 550 <= counts["train"] <= 850
        assert 100 <= counts["val"] <= 350
        assert 30 <= counts["test"] <= 200


# ─── Split Memories ───────────────────────────────────────────────────────────

class TestSplitMemories:
    def test_tags_each_memory(self):
        mems = [{"id": f"m{i}", "text": f"text-{i}"} for i in range(10)]
        splits = split_memories(mems)
        assert set(splits.keys()) == {"train", "val", "test"}
        total = sum(len(v) for v in splits.values())
        assert total == 10

    def test_adds_split_key(self):
        mems = [{"id": "x1", "text": "t"}]
        splits = split_memories(mems)
        for split_name, mem_list in splits.items():
            for m in mem_list:
                assert m["_split"] == split_name

    def test_empty_input(self):
        splits = split_memories([])
        assert splits == {"train": [], "val": [], "test": []}


# ─── Filter Functions ─────────────────────────────────────────────────────────

class TestFilterFunctions:
    def setup_method(self):
        self.mems = [{"id": f"filt-{i}", "text": "x"} for i in range(100)]

    def test_get_val_memories(self):
        vals = get_val_memories(self.mems)
        assert all(assign_split(m["id"]) == "val" for m in vals)
        assert len(vals) > 0

    def test_get_test_memories(self):
        tests = get_test_memories(self.mems)
        assert all(assign_split(m["id"]) == "test" for m in tests)
        assert len(tests) > 0

    def test_get_train_memories(self):
        trains = get_train_memories(self.mems)
        assert all(assign_split(m["id"]) == "train" for m in trains)
        assert len(trains) > 0

    def test_filters_partition_complete(self):
        t = get_train_memories(self.mems)
        v = get_val_memories(self.mems)
        te = get_test_memories(self.mems)
        assert len(t) + len(v) + len(te) == len(self.mems)


# ─── Split Stats ──────────────────────────────────────────────────────────────

class TestSplitStats:
    def test_stats_fields(self):
        mems = [{"id": f"s{i}"} for i in range(50)]
        s = split_stats(mems)
        assert s["total"] == 50
        assert s["train"] + s["val"] + s["test"] == 50
        assert abs(s["train_pct"] + s["val_pct"] + s["test_pct"] - 100.0) < 0.5

    def test_empty_stats(self):
        s = split_stats([])
        assert s["total"] == 0
        assert s["train_pct"] == 0.0


# ─── Ratio Constants ──────────────────────────────────────────────────────────

class TestRatios:
    def test_ratios_sum_to_one(self):
        assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

    def test_train_is_majority(self):
        assert TRAIN_RATIO > VAL_RATIO
        assert TRAIN_RATIO > TEST_RATIO
