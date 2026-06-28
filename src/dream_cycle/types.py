"""
Dream Cycle — Type Definitions
==============================
Dataclasses for the dream cycle pipeline. Structured contracts between modules.

Replaces ad-hoc dicts with typed, validated data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ─── Memory Types ────────────────────────────────────────────────────────────

Split = Literal["train", "val", "test"]
Outcome = Literal["success", "fail", "mixed", "unknown", ""]


@dataclass
class DreamMemory:
    """A memory record flowing through the dream cycle pipeline.

    Wraps the raw dict from PG with typed fields. The original dict is
    preserved in `_raw` for backward compatibility with modules that
    haven't been migrated yet.
    """
    id: str
    text: str
    created_at: str = ""
    source: str = ""
    session_id: str = ""
    session_title: str = ""
    hash: str | None = None

    # v6 fields (set by orchestrator)
    _split: Split = "train"
    _outcome: Outcome = ""
    _recalled: bool = False

    # Signal fields (set by session.py)
    signal_type: str = ""
    importance: float = 0.5

    # Original dict for backward compat
    _raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: dict) -> "DreamMemory":
        """Construct from a raw PG dict."""
        return cls(
            id=d.get("id", ""),
            text=d.get("text", ""),
            created_at=d.get("created_at", ""),
            source=d.get("source", ""),
            session_id=d.get("session_id", ""),
            session_title=d.get("session_title", ""),
            hash=d.get("hash"),
            _split=d.get("_split", "train"),
            _outcome=d.get("_outcome", ""),
            _recalled=d.get("_recalled", False),
            signal_type=d.get("signal_type", ""),
            importance=d.get("importance", 0.5),
            _raw=d,
        )

    def to_dict(self) -> dict:
        """Convert back to dict for backward-compatible modules."""
        d = dict(self._raw) if self._raw else {}
        d.update({
            "id": self.id,
            "text": self.text,
            "created_at": self.created_at,
            "source": self.source,
            "session_id": self.session_id,
            "session_title": self.session_title,
            "_split": self._split,
            "_outcome": self._outcome,
            "_recalled": self._recalled,
            "signal_type": self.signal_type,
            "importance": self.importance,
        })
        return d


# ─── Operation Types ─────────────────────────────────────────────────────────

OpType = Literal["archive", "update_text", "delete", "boost", "extend"]


@dataclass
class MemoryOp:
    """A proposed memory operation with structured intent.

    Replaces raw SQL strings in stage3. The backend interprets this
    into the appropriate PG/SQLite operation.
    """
    op: OpType
    memory_id: str
    stage: str = ""                    # "dedup" | "merge" | "decay" | "boost" | "supersede" | "extend"
    payload_patch: dict = field(default_factory=dict)
    new_text: str = ""                 # for update_text
    reason: str = ""
    superseded_by: str = ""           # for dedup: which memory replaces this one

    def is_costly(self) -> bool:
        """Whether this op consumes edit budget."""
        return self.op in ("archive", "update_text", "delete", "extend")


# ─── Pipeline Result Types ───────────────────────────────────────────────────

@dataclass
class PrepareResult:
    """Phase 1 output: memories + sessions + signals + feedback."""
    memories: list[dict]               # still dicts for stage1/2 compat
    sessions: list[dict]
    signals: dict
    feedback: dict = field(default_factory=dict)


@dataclass
class BudgetSummary:
    """Snapshot of edit/token/time budgets."""
    edit_budget: int = 8
    edits_used: int = 0
    edits_remaining: int = 8
    edits_skipped: int = 0
    tokens_used: int = 0
    token_budget: int = 200_000
    elapsed_seconds: float = 0.0
    wall_clock_budget: int = 1200
    fraction_remaining: float = 1.0
    costly_ops: int = 0
    free_ops: int = 0

    def can_spend_costly(self) -> bool:
        return self.edits_remaining > 0 and self.fraction_remaining > 0.05

    def deep_planning_ok(self, threshold: float = 0.20) -> bool:
        """Whether budget allows expensive stages (SHMR, Contrastive)."""
        return self.fraction_remaining >= threshold


@dataclass
class ExecuteResult:
    """Phase 2 output: clusters + rem_results + stats + staging."""
    clusters: dict = field(default_factory=dict)
    rem_results: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    staging_info: dict = field(default_factory=dict)
