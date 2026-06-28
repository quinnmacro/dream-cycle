"""budget.py — Edit budget (textual learning rate) for Dream Cycle v6

Ported from SkillOpt-Sleep: limits how many destructive operations (dedup,
merge, decay/archive) can be applied per night. Prevents one bad cycle from
corrupting the entire memory store.

Design:
  - EDIT_BUDGET = max destructive ops per night (default 8)
  - Shared across dedup + merge + decay + slot-supersede
  - When exhausted: skip remaining ops, report what was skipped
  - Non-destructive ops (boost, relations, SHMR, vault suggestions) are FREE
  - Cosine decay scheduler: more budget early in pipeline, less at end
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Config ──────────────────────────────────────────────────────────────────

DEFAULT_EDIT_BUDGET = 8      # max destructive ops per night
MIN_EDIT_BUDGET = 2          # minimum when scheduler decays
MAX_TOKENS_PER_NIGHT = 200_000  # token budget for LLM calls
MAX_WALL_CLOCK = 1200        # seconds (20 min hard cap)

# Ops that cost budget (destructive)
COSTLY_OPS = frozenset({
    "dedup_archive",   # archive a duplicate
    "merge",           # merge two memories (update + delete)
    "decay_archive",   # archive a decayed memory
    "slot_supersede",  # archive a superseded memory
    "slot_extend",     # extend both memories (modifies both)
})

# Ops that are free (non-destructive)
FREE_OPS = frozenset({
    "boost",           # just add metadata
    "relation",        # add Neo4j edge
    "shmr_belief",     # add harmonic belief
    "vault_suggest",   # suggest vault page
    "shy_downscale",   # adjust edge weights
    "hebbian",         # strengthen edges
    "degrade_tier",    # compress old text (not destructive)
})


@dataclass
class EditBudget:
    """Tracks remaining edit budget for a dream cycle run."""

    max_edits: int = DEFAULT_EDIT_BUDGET
    used: int = 0
    skipped: list[dict] = field(default_factory=list)
    log: list[dict] = field(default_factory=list)
    _start_time: Optional[float] = None
    _tokens_used: int = 0
    max_tokens: int = MAX_TOKENS_PER_NIGHT
    max_wall_clock: int = MAX_WALL_CLOCK

    def start(self):
        """Mark the start of the budget window."""
        self._start_time = time.time()

    def can_spend(self, op: str) -> bool:
        """Check if an operation can be executed."""
        if op in FREE_OPS:
            return True
        if op in COSTLY_OPS:
            return self.remaining > 0
        return True  # unknown ops are free by default

    def spend(self, op: str, detail: str = "", memory_id: str = "") -> bool:
        """Spend one edit budget unit. Returns True if spent, False if skipped."""
        if op in FREE_OPS:
            self.log.append({"op": op, "detail": detail, "cost": 0, "ts": time.time()})
            return True

        if self.remaining <= 0:
            self.skipped.append({
                "op": op, "detail": detail, "memory_id": memory_id,
                "reason": "budget_exhausted", "ts": time.time(),
            })
            return False

        self.used += 1
        self.log.append({
            "op": op, "detail": detail, "cost": 1,
            "memory_id": memory_id, "ts": time.time(),
        })
        return True

    def record_tokens(self, count: int):
        """Track token usage."""
        self._tokens_used += count

    def token_budget_ok(self) -> bool:
        """Check if token budget allows more LLM calls."""
        return self._tokens_used < self.max_tokens

    def wall_clock_ok(self) -> bool:
        """Check if wall clock budget allows more work."""
        if self._start_time is None:
            return True
        elapsed = time.time() - self._start_time
        return elapsed < self.max_wall_clock

    def all_budgets_ok(self) -> bool:
        """Check all budgets (edits + tokens + time)."""
        return self.can_spend("any_costly") and self.token_budget_ok() and self.wall_clock_ok()

    @property
    def remaining(self) -> int:
        return max(0, self.max_edits - self.used)

    @property
    def fraction_remaining(self) -> float:
        """Minimum across all budget axes."""
        edit_frac = self.remaining / max(self.max_edits, 1)
        token_frac = max(0, 1.0 - self._tokens_used / max(self.max_tokens, 1))
        time_frac = 1.0
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            time_frac = max(0, 1.0 - elapsed / max(self.max_wall_clock, 1))
        return min(edit_frac, token_frac, time_frac)

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def summary(self) -> dict:
        """Return budget summary for reporting."""
        return {
            "edit_budget": self.max_edits,
            "edits_used": self.used,
            "edits_remaining": self.remaining,
            "edits_skipped": len(self.skipped),
            "tokens_used": self._tokens_used,
            "token_budget": self.max_tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "wall_clock_budget": self.max_wall_clock,
            "fraction_remaining": round(self.fraction_remaining, 3),
            "costly_ops": sum(1 for e in self.log if e.get("cost", 0) > 0),
            "free_ops": sum(1 for e in self.log if e.get("cost", 0) == 0),
        }

    def skipped_summary(self) -> str:
        """Human-readable summary of skipped operations."""
        if not self.skipped:
            return ""
        lines = [f"⏸️ {len(self.skipped)} ops skipped (budget exhausted):"]
        # Group by op type
        by_op: dict[str, int] = {}
        for s in self.skipped:
            op = s["op"]
            by_op[op] = by_op.get(op, 0) + 1
        for op, count in sorted(by_op.items(), key=lambda x: -x[1]):
            lines.append(f"  • {op}: {count}")
        return "\n".join(lines)
