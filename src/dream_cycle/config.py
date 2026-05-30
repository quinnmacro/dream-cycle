"""
Dream Cycle — Configuration & Constants
========================================
All tunable parameters, paths, and shared constants.
Source of truth for every magic number in the pipeline.
"""



__all__ = [
    "HKT",
    "PG_CONTAINER",
    "PG_USER",
    "PG_DB",
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASS",
    "STATE_DB",
    "DREAM_DB",
    "VAULT_DIR",
    "DREAM_LOCK",
    "DREAM_LOCK_TIMEOUT",
    "DEDUP_DIST",
    "MERGE_DIST",
    "CLUSTER_DIST",
    "DEDUP_THRESHOLD",
    "MERGE_THRESHOLD",
    "CLUSTER_THRESHOLD",
    "IMPORTANCE_WEIGHTS",
    "TRIGGER_MAX_IDLE_HOURS",
    "TRIGGER_MIN_NEW_MEMORIES",
    "TRIGGER_CONFLICT_DENSITY",
    "TRIGGER_MEMORY_ENTROPY",
    "HEBBIAN_LEARNING_RATE",
    "HEBBIAN_DOWNSCALE",
    "REM_WALK_LENGTH",
    "REM_SEED_COUNT",
    "REM_MAX_NEW_EDGES",
    "REM_JUMP_PROBABILITY",
    "WAKING_THRESHOLD",
    "SHY_PROTECTION_PCT",
    "SHY_DOWNSCALE_FACTOR",
    "SHY_PRUNE_THRESHOLD",
    "PROMOTION_MIN_SCORE",
    "PROMOTION_MIN_RECALLS",
    "PROMOTION_MIN_SESSIONS",
    "PERMANENT_MARKERS",
    "DECAY_HALF_LIVES",
    "FADEMEM_BETA",
    "RETENTION_FLOOR",
    "SIGNAL_CORRECTIONS",
    "SIGNAL_PREFERENCES",
    "SIGNAL_DECISIONS",
    "SIGNAL_PATTERNS",
    "ARCHIVE_THRESHOLD_DAYS",
    "ARCHIVE_MIN_SCORE",
    "INFINI_BASE_URL",
    "INFINI_MODEL",
    "safe_float",
    "log",
]

import json
import hashlib
import re
import sqlite3
import time
import logging
import logging.handlers
import argparse
import sys
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

# ─── Timezone ──────────────────────────────────────────────────────────

HKT = timezone(timedelta(hours=8))

# ─── Database connections ──────────────────────────────────────────────

# PG (via docker exec)
PG_CONTAINER = "postgres"
PG_USER = "postgres"
PG_DB = "mem0_v2"

# Neo4j Playground
NEO4J_URI = "bolt://100.69.76.69:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "knowledge2026"

# ─── Paths ─────────────────────────────────────────────────────────────

STATE_DB = Path("/root/.hermes/state.db")
DREAM_DB = Path("/root/data/dream_cycle.db")
VAULT_DIR = Path("/root/vault")
DREAM_LOCK = Path("/tmp/dream_cycle.lock")
DREAM_LOCK_TIMEOUT = 3600  # 1h timeout (normal run: 15-25 min)

# ─── Similarity thresholds (cosine distance) ───────────────────────────
# 0 = identical, 0.5 = orthogonal, 1.0 = opposite

DEDUP_DIST = 0.10          # <0.10 = duplicate (similarity > 0.90)
MERGE_DIST = 0.18          # 0.10-0.18 = mergeable (similarity 0.82-0.90)
CLUSTER_DIST = 0.30        # 0.18-0.30 = same group (similarity 0.70-0.82)

# Text similarity thresholds (n-gram fallback, no vector)
DEDUP_THRESHOLD = 0.92
MERGE_THRESHOLD = 0.85
CLUSTER_THRESHOLD = 0.70

# ─── Importance scoring weights (7-dim, v3 + novelty from SCM) ─────────

IMPORTANCE_WEIGHTS = {
    "recency": 0.12,          # Time decay (FadeMem dual-layer)
    "frequency": 0.18,        # Short-term signal accumulation
    "query_diversity": 0.12,  # Different query contexts hit
    "domain": 0.18,           # Investment > Tech > Daily
    "consolidation": 0.14,    # Multi-day recurrence strength
    "confidence": 0.08,       # High confidence bonus
    "novelty": 0.18,          # Semantic distance from existing memories (SCM core)
}

# ─── Adaptive trigger (SCM + SleepGate) ────────────────────────────────

TRIGGER_MAX_IDLE_HOURS = 6       # Force trigger after 6h idle
TRIGGER_MIN_NEW_MEMORIES = 10    # Trigger if ≥10 new memories
TRIGGER_CONFLICT_DENSITY = 0.3   # Conflict density threshold (SCM: θ_c=0.3)
TRIGGER_MEMORY_ENTROPY = 0.9     # Memory entropy threshold (SCM: θ_e=0.9)

# ─── NREM Hebbian parameters (from SCM) ─────────────────────────────────

HEBBIAN_LEARNING_RATE = 0.1     # η: connection strength increment
HEBBIAN_DOWNSCALE = 0.8         # α: global downscale factor (20%/cycle)

# ─── REM Dream Engine v2 (from claude-brain) ───────────────────────────

REM_WALK_LENGTH = 5              # Random walk steps per seed
REM_SEED_COUNT = 5               # High-importance seed nodes
REM_MAX_NEW_EDGES = 10           # Max new edges per dream cycle
REM_JUMP_PROBABILITY = 0.30      # 30% creative teleport probability
WAKING_THRESHOLD = 1             # Shared neighbors ≥1 to promote edge
SHY_PROTECTION_PCT = 0.20        # SHY: top 20% edges protected
SHY_DOWNSCALE_FACTOR = 0.15      # SHY: max downscale for unprotected
SHY_PRUNE_THRESHOLD = 0.08       # SHY: prune edges below this

# ─── Vault promotion gates (from OpenClaw Dreaming) ────────────────────

PROMOTION_MIN_SCORE = 0.65       # Minimum combined score
PROMOTION_MIN_RECALLS = 3        # Minimum recall count
PROMOTION_MIN_SESSIONS = 2       # Minimum cross-session appearances

# ─── Permanent markers (never archive) ─────────────────────────────────

PERMANENT_MARKERS = ['⚠️ PERMANENT', '🔥 HIGH', '📌 PIN']

# ─── FadeMem dual-layer decay: R = e^(-λ·t^β) ────────────────────────
# β>1 = super-linear fast decay (volatile)
# β<1 = sub-linear slow decay (stable)

DECAY_HALF_LIVES = {
    "volatile": 3,    # days — market data
    "normal": 7,      # days — projects/tools
    "stable": 30,     # days — user prefs/infra
}
FADEMEM_BETA = {
    "volatile": 1.2,  # super-linear: market data fast decay
    "normal": 1.0,    # linear: standard exponential
    "stable": 0.8,    # sub-linear: user prefs slow decay
}
RETENTION_FLOOR = 0.20  # Minimum retention baseline (from PowerMem)

# ─── Archive strategy (from Auto-Dream: never delete, archive) ─────────

ARCHIVE_THRESHOLD_DAYS = 90   # >90d + low importance → archive
ARCHIVE_MIN_SCORE = 0.25      # Below this score → eligible for archive

# ─── Session Signal Scanning (from Anthropic autoDream) ────────────────

SIGNAL_CORRECTIONS = [
    "不对", "错了", "actually", "wrong", "别这样", "stop doing", "不是",
    "incorrect", "I said", "I meant", "don't do", "correction", "修改",
]
SIGNAL_PREFERENCES = [
    "我喜欢", "prefer", "always use", "从今以后", "记住", "I like",
    "I want", "going forward", "keep in mind", "make sure", "我的偏好",
]
SIGNAL_DECISIONS = [
    "决定", "我们用", "let's go with", "chosen", "decision", "agreed",
    "the plan is", "switch to", "move to", "pick", "选定", "方案是",
]
SIGNAL_PATTERNS = [
    "又是", "每次", "every time", "again", "keep forgetting", "as usual",
    "same as before", "like last time", "老问题", "反复",
]

# ─── LLM API (DashScope) ──────────────────────────────────────────────

INFINI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
INFINI_MODEL = "qwen3.7-max"

# ─── Logging ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            "/var/log/dream-cycle.log", maxBytes=5_000_000, backupCount=3
        ),
    ],
)
log = logging.getLogger("dream")


# ─── Utility functions ─────────────────────────────────────────────────

def safe_float(val, default=None) -> float | None:
    """Safe float conversion — PG returns empty strings, don't crash."""
    if val is None or val == '':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
