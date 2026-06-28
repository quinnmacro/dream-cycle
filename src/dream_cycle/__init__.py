"""
Dream Cycle — Modular Memory Consolidation Engine
===================================================
Bio-inspired sleep cycle for AI agent memory.

Architecture (v7.0 "Clean Sleep", 2026-06-29):
  config.py         — Constants & parameters
  types.py          — Dataclasses: DreamMemory, MemoryOp, PrepareResult, BudgetSummary
  ops.py            — MemoryBackend (DirectBackend, StagingBackend) — replaces monkey-patching
  db.py             — Database operations (PG/SQLite/Neo4j)
  similarity.py     — Similarity functions & vector ops
  llm.py            — LLM API calls (DashScope/Infini) + cache + dual-backend + JSON retry
  entities.py       — Entity extraction & topic keywords
  vault.py          — Vault integration
  stage1.py         — Shallow Sleep (clustering)
  stage2.py         — REM (scoring, contradiction, vault candidates)
  stage3.py         — Deep Sleep (dedup, decay, relations) — uses MemoryBackend
  dream_engine.py   — REM walk v2 + SHY + threat simulation
  session.py        — Session mining & signal scanning + feedback detection
  health.py         — Dashboard & adaptive trigger
  shmr.py           — Self-Harmonizing Memory Reasoning + contrastive + associative recall
  split.py          — Train/Val/Test split discipline
  budget.py         — Edit budget / textual learning rate
  validation.py     — Held-out replay validation + safety probe
  staging.py        — Staging + Adopt safety contract
  orchestrator.py   — Main pipeline, lock, report — uses MemoryBackend (no monkey-patching)
"""

from dream_cycle.config import (
    DREAM_DB,
    DREAM_LOCK,
    DREAM_LOCK_TIMEOUT,
    STATE_DB,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASS,
    INFINI_BASE_URL,
    INFINI_MODEL,
    HKT,
)
from dream_cycle.orchestrator import run_dream_cycle, cmd_adopt

__version__ = "7.0.0"
__all__ = ["run_dream_cycle", "cmd_adopt", "DREAM_DB", "__version__"]
