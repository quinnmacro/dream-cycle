"""
Dream Cycle — Memory Operations Backend
=======================================
Structured abstraction for PG memory writes. Replaces monkey-patching
interceptors with a Backend interface.

Two implementations:
  - DirectBackend: executes operations against live PG
  - StagingBackend: collects operations into a buffer (no PG writes)

Usage:
    backend = StagingBackend() if use_staging else DirectBackend()
    backend.execute(MemoryOp(op="archive", memory_id="...", stage="dedup", ...))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from dream_cycle.config import log, HKT
from dream_cycle.types import MemoryOp
from dream_cycle.staging import StagingBuffer
from dream_cycle.budget import EditBudget, COSTLY_OPS

# Import PG functions (DirectBackend uses these)
from dream_cycle.db import pg_query as _pg_query, update_memory_text as _update_text, delete_memory as _delete_mem


# ─── Backend Protocol ────────────────────────────────────────────────────────

class MemoryBackend(Protocol):
    """Abstract interface for memory operations."""

    def execute(self, op: MemoryOp) -> bool:
        """Execute a memory operation. Returns True on success."""
        ...

    @property
    def proposals(self) -> list[MemoryOp]:
        """Return collected proposals (StagingBackend) or empty list."""
        ...


# ─── Direct Backend ─────────────────────────────────────────────────────────

class DirectBackend:
    """Execute operations directly against live PG.

    This is the "normal" backend used when staging is disabled (dry-run mode)
    or when the user has explicitly adopted staging proposals.
    """

    def execute(self, op: MemoryOp) -> bool:
        """Execute a single memory operation against PG."""
        try:
            if op.op == "archive":
                patch = dict(op.payload_patch)
                patch.setdefault("archived", True)
                patch.setdefault("archived_reason", op.stage)
                if op.superseded_by:
                    patch["superseded_by"] = op.superseded_by
                patch_json = json.dumps(patch, ensure_ascii=False)
                _pg_query(
                    f"UPDATE mem0 SET payload = payload || '{patch_json}' "
                    f"WHERE id::text = '{op.memory_id}'"
                )
                log.debug(f"  ✓ Archived {op.memory_id[:8]} ({op.stage})")
                return True

            elif op.op == "update_text":
                return _update_text(op.memory_id, op.new_text)

            elif op.op == "delete":
                return _delete_mem(op.memory_id)

            elif op.op == "boost":
                patch = dict(op.payload_patch)
                patch_json = json.dumps(patch, ensure_ascii=False)
                _pg_query(
                    f"UPDATE mem0 SET payload = payload || '{patch_json}' "
                    f"WHERE id::text = '{op.memory_id}'"
                )
                return True

            elif op.op == "extend":
                patch = dict(op.payload_patch)
                patch.setdefault("extended", True)
                patch_json = json.dumps(patch, ensure_ascii=False)
                _pg_query(
                    f"UPDATE mem0 SET payload = payload || '{patch_json}' "
                    f"WHERE id::text = '{op.memory_id}'"
                )
                return True

            elif op.op == "degrade":
                # Tier degradation: update both text (data field) and metadata
                patch = dict(op.payload_patch)
                patch_json = json.dumps(patch, ensure_ascii=False)
                new_data = json.dumps(op.new_text, ensure_ascii=False)
                _pg_query(
                    f"UPDATE mem0 SET payload = payload || '{patch_json}' "
                    f"|| jsonb_set(payload, '{{data}}', '{new_data}') "
                    f"WHERE id::text = '{op.memory_id}'"
                )
                log.debug(f"  ✓ Degraded {op.memory_id[:8]} ({op.stage})")
                return True

            else:
                log.warning(f"  ⚠️ Unknown op type: {op.op}")
                return False

        except Exception as e:
            log.warning(f"  ⚠️ DirectBackend.execute failed for {op.memory_id[:8]}: {e}")
            return False

    @property
    def proposals(self) -> list[MemoryOp]:
        return []


# ─── Staging Backend ─────────────────────────────────────────────────────────

class StagingBackend:
    """Collect operations into a StagingBuffer instead of executing.

    Used when use_staging=True (the default for non-dry-run cycles).
    All operations are recorded but NOT executed against PG.
    After the cycle, the buffer is written to staging files and the user
    reviews and adopts manually.
    """

    def __init__(self, budget: EditBudget | None = None):
        self._buffer = StagingBuffer()
        self._budget = budget
        self._executed: list[MemoryOp] = []
        self._skipped: list[MemoryOp] = []

    def execute(self, op: MemoryOp) -> bool:
        """Record operation into staging buffer. Budget-gated for costly ops."""
        # Budget check for costly operations
        if op.is_costly() and self._budget is not None:
            op_name = f"{op.stage}_archive" if op.op == "archive" else op.op
            if not self._budget.spend(op_name, detail=op.reason[:100], memory_id=op.memory_id):
                log.info(f"  ⏸️ Budget skip: {op.op} on {op.memory_id[:8]}")
                self._skipped.append(op)
                return False

        # Route to staging buffer
        if op.op == "archive":
            self._buffer.add_archive(
                op.memory_id,
                reason=op.reason or f"{op.stage} archive",
                stage=op.stage,
                payload_patch=op.payload_patch,
            )
        elif op.op == "update_text":
            self._buffer.add_update_text(
                op.memory_id, op.new_text,
                reason=op.reason, stage=op.stage,
            )
        elif op.op == "delete":
            self._buffer.add_delete(
                op.memory_id,
                reason=op.reason, stage=op.stage,
            )
        elif op.op in ("boost", "extend"):
            self._buffer.add_update_payload(
                op.memory_id, op.payload_patch,
                reason=op.reason, stage=op.stage,
            )
        elif op.op == "degrade":
            # Degrade = update_text + metadata patch
            self._buffer.add_update_text(
                op.memory_id, op.new_text,
                reason=op.reason, stage=op.stage,
                payload_patch=op.payload_patch,
            )

        self._executed.append(op)
        log.debug(f"  📝 Staged: {op.op} {op.memory_id[:8]} ({op.stage})")
        return True

    @property
    def proposals(self) -> list[MemoryOp]:
        return list(self._executed)

    @property
    def buffer(self) -> StagingBuffer:
        return self._buffer

    @property
    def skipped(self) -> list[MemoryOp]:
        return list(self._skipped)

    def stats(self) -> dict:
        return {
            "staged": len(self._executed),
            "skipped": len(self._skipped),
            "buffer_stats": self._buffer.stats(),
        }


# ─── Factory ─────────────────────────────────────────────────────────────────

def create_backend(
    use_staging: bool = True,
    budget: EditBudget | None = None,
) -> MemoryBackend:
    """Factory: create the appropriate backend for this dream cycle run.

    Args:
        use_staging: If True, use StagingBackend (default). If False, DirectBackend.
        budget: Edit budget for StagingBackend. Ignored for DirectBackend.
    """
    if use_staging:
        return StagingBackend(budget=budget)
    return DirectBackend()
