"""
Dream Cycle — Modular Memory Consolidation Engine
===================================================
Bio-inspired sleep cycle for AI agent memory.

Architecture (v6.0 "Safe Sleep", 2026-06-28):
  config.py         — Constants & parameters
  db.py             — Database operations (PG/SQLite/Neo4j)
  similarity.py     — Similarity functions & vector ops
  llm.py            — LLM API calls (DashScope/Infini)
  entities.py       — Entity extraction & topic keywords
  vault.py          — Vault integration
  stage1.py         — Shallow Sleep (clustering)
  stage2.py         — REM (scoring, contradiction, vault candidates)
  stage3.py         — Deep Sleep (dedup, decay, relations)
  dream_engine.py   — REM walk v2 + SHY + threat simulation
  session.py        — Session mining & signal scanning
  health.py         — Dashboard & adaptive trigger
  shmr.py           — Self-Harmonizing Memory Reasoning
  orchestrator.py   — Main pipeline, lock, report, staging interceptors
  split.py          — Train/Val/Test split discipline (v6)
  budget.py         — Edit budget / textual learning rate (v6)
  validation.py     — Held-out replay validation (v6)
  staging.py        — Staging + Adopt safety contract (v6)
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

__version__ = "6.0.0"
__all__ = ["run_dream_cycle", "cmd_adopt", "DREAM_DB", "__version__"]
