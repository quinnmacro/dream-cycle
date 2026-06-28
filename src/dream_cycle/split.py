"""split.py — Train/Val/Test split discipline (Dream Cycle v6 "Safe Sleep")

Ported from SkillOpt-Sleep: deterministic hash-based split that prevents
train/val leakage across nights. Same memory always lands in the same bucket.

Split ratios:
  - train (70%): SHMR clustering, dedup, merge, decay — all consolidation happens here
  - val   (20%): held-out validation — candidate changes must improve val score
  - test  (10%): final evaluation — never touched during consolidation

Anti-overfitting contract:
  - val/test drawn ONLY from real memories, never from dream-generated ones
  - train may include SHMR-harmonized beliefs, never in val/test
  - Stable hash-based assignment keeps same memory in same split across nights
"""
from __future__ import annotations

import hashlib
from typing import Literal

Split = Literal["train", "val", "test"]

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

# Bucket boundaries (0-9 scale)
# Buckets 0-6 = train, 7-8 = val, 9 = test
TRAIN_BUCKETS = frozenset(range(0, 7))   # 0,1,2,3,4,5,6
VAL_BUCKETS = frozenset(range(7, 9))     # 7,8
TEST_BUCKETS = frozenset({9})            # 9


def _hash_bucket(memory_id: str) -> int:
    """Deterministic bucket assignment via SHA256.

    Returns 0-9. Same memory_id always maps to the same bucket.
    """
    h = hashlib.sha256(memory_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 10


def assign_split(memory_id: str) -> Split:
    """Assign a memory to train/val/test based on its ID hash."""
    bucket = _hash_bucket(memory_id)
    if bucket in TRAIN_BUCKETS:
        return "train"
    elif bucket in VAL_BUCKETS:
        return "val"
    else:
        return "test"


def split_memories(memories: list[dict]) -> dict[str, list[dict]]:
    """Split a list of memory dicts into {train: [...], val: [...], test: [...]}.

    Each memory gets a '_split' key added.
    """
    result: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for mem in memories:
        mid = mem.get("id", "")
        split = assign_split(mid)
        mem["_split"] = split
        result[split].append(mem)
    return result


def get_val_memories(memories: list[dict]) -> list[dict]:
    """Return only val-split memories."""
    return [m for m in memories if assign_split(m.get("id", "")) == "val"]


def get_test_memories(memories: list[dict]) -> list[dict]:
    """Return only test-split memories."""
    return [m for m in memories if assign_split(m.get("id", "")) == "test"]


def get_train_memories(memories: list[dict]) -> list[dict]:
    """Return only train-split memories."""
    return [m for m in memories if assign_split(m.get("id", "")) == "train"]


def split_stats(memories: list[dict]) -> dict:
    """Return split distribution stats."""
    splits = split_memories(memories)
    total = len(memories)
    return {
        "total": total,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "train_pct": round(len(splits["train"]) / max(total, 1) * 100, 1),
        "val_pct": round(len(splits["val"]) / max(total, 1) * 100, 1),
        "test_pct": round(len(splits["test"]) / max(total, 1) * 100, 1),
    }
