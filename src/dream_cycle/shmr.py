"""
Dream Cycle — SHMR: Self-Harmonizing Memory Reasoning
======================================================
Inspired by Mnemosyne's SHMR. Memories "echo" each other, negotiating
contradictions, surfacing hidden patterns, and converging into stable beliefs.

Core idea: related memories are clustered by vector similarity, then an LLM
harmonizes each cluster — resolving contradictions, extracting higher-order
beliefs, dampening noise, and amplifying corroborated signal.

Output: harmonic_beliefs table in dream_cycle.db — stable truths with
confidence scores and provenance chains.
"""

__all__ = [
    "run_shmr",
]

import json
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from dream_cycle.config import (
    DREAM_DB,
    HKT,
    log,
)
from dream_cycle.similarity import combined_similarity
from dream_cycle.llm import _call_infini
from dream_cycle.db import pg_query

# ─── Config ────────────────────────────────────────────────────────────

SHMR_SIMILARITY_THRESHOLD = 0.70  # Cluster members must be ≥0.70 similar
SHMR_MIN_CLUSTER_SIZE = 3  # Need ≥3 memories to form a belief
SHMR_MAX_BELIEFS_PER_CLUSTER = 5
SHMR_CONFIDENCE_HIGH = 0.9  # Highly corroborated
SHMR_CONFIDENCE_MID = 0.6  # Reasonable inference
SHMR_CONFIDENCE_LOW = 0.4  # Speculative
SHMR_DAMPEN_FACTOR = 0.3  # Reduce confidence of contradicted facts

# ─── Schema ────────────────────────────────────────────────────────────

SHMR_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS harmonic_beliefs (
    belief_id TEXT PRIMARY KEY,
    cluster_id TEXT,
    subject TEXT,
    predicate TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    provenance TEXT,
    action TEXT DEFAULT 'create',
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dream_run_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_beliefs_subject ON harmonic_beliefs(subject);
CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON harmonic_beliefs(confidence);
CREATE INDEX IF NOT EXISTS idx_beliefs_cluster ON harmonic_beliefs(cluster_id);
"""

HARMONY_PROMPT = """You are a memory harmonizer. These memories belong to the same semantic cluster — they all relate to the same entities, topics, or events. Your job:

1. **Resolve contradictions**: If two memories conflict, determine which is more likely true based on recency, specificity, and internal consistency. Flag the weaker one as dampened (action: "dampen"), not deleted.

2. **Extract higher-order beliefs**: Find patterns spanning multiple memories. What does this cluster as a whole tell us? What's the stable truth?

3. **Dampen noise, amplify signal**: Low-confidence or stale memories get lower weight. Corroborated facts get reinforced.

4. **Output ONLY stable beliefs**: Return NEW or UPDATED facts with confidence scores. Don't regurgitate every input — synthesize.

Output as JSON array of belief objects:
[{{"subject": "...", "predicate": "...", "confidence": 0.0-1.0,
  "action": "create"|"update"|"dampen", "target_belief_id": null|"existing_id",
  "rationale": "one sentence explaining why"}}]

RULES:
- Confidence 0.9+ = highly corroborated (multiple sources agree)
- Confidence 0.5-0.8 = reasonable inference from the cluster
- Confidence <0.4 = speculative, mark as such
- Use "dampen" to reduce confidence of contradicted facts (never delete)
- Use "update" to modify an existing belief with new information
- Output 1-5 beliefs per cluster (don't over-generate)
- Subject should be a concrete entity or concept (person, project, tool, fact)
- Predicate should be a clear statement about the subject

MEMORIES:
{memories_text}"""


def _init_schema(conn: sqlite3.Connection):
    """Ensure SHMR tables exist."""
    conn.executescript(SHMR_SCHEMA_SQL)
    conn.commit()


def _cluster_by_similarity(
    memories: List[Dict],
    threshold: float = SHMR_SIMILARITY_THRESHOLD,
) -> List[List[Dict]]:
    """
    Greedy connected-components clustering by combined_similarity.

    Each memory must have 'text' and 'id' keys.
    Returns list of clusters (each cluster is a list of memories).
    """
    if not memories:
        return []

    n = len(memories)
    adj: Dict[int, set] = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            sim = combined_similarity(memories[i]["text"], memories[j]["text"])
            if sim >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # BFS connected components
    visited = set()
    clusters = []
    for i in range(n):
        if i in visited:
            continue
        cluster = []
        stack = [i]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.append(memories[node])
            stack.extend(adj[node] - visited)
        clusters.append(cluster)

    return clusters


def _format_cluster_for_llm(cluster: List[Dict]) -> str:
    """Format a memory cluster as prompt text for the LLM harmonizer."""
    lines = []
    for i, mem in enumerate(cluster):
        text = mem.get("text", "")[:300]
        created = mem.get("created_at", "unknown")
        source = mem.get("source", "unknown")
        mid = mem.get("id", "?")[:8]
        lines.append(f"[{i}] (id={mid}, source={source}, created={created[:10] if created else '?'}) {text}")
    return "\n".join(lines)


def _harmonize_cluster(cluster: List[Dict], cluster_id: str) -> List[Dict]:
    """
    Call LLM to harmonize a cluster of related memories.

    Returns list of belief dicts with: subject, predicate, confidence,
    action, rationale.
    """
    memories_text = _format_cluster_for_llm(cluster)
    prompt = HARMONY_PROMPT.format(memories_text=memories_text)

    try:
        response = _call_infini(prompt, system="You are a memory harmonizer. Output only valid JSON.", max_tokens=1024)
        if not response:
            return []

        # Parse JSON from response (handle markdown code blocks)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        beliefs = json.loads(text)
        if not isinstance(beliefs, list):
            beliefs = [beliefs]

        # Validate and clean
        valid = []
        for b in beliefs[:SHMR_MAX_BELIEFS_PER_CLUSTER]:
            if not isinstance(b, dict):
                continue
            if not b.get("predicate"):
                continue
            b["confidence"] = max(0.0, min(1.0, float(b.get("confidence", 0.5))))
            b["action"] = b.get("action", "create")
            if b["action"] not in ("create", "update", "dampen"):
                b["action"] = "create"
            valid.append(b)

        return valid

    except (json.JSONDecodeError, ValueError, TypeError) as e:
        log.debug(f"SHMR: LLM parse failed for cluster {cluster_id}: {e}")
        return []
    except Exception as e:
        log.debug(f"SHMR: harmonize failed for cluster {cluster_id}: {e}")
        return []


def run_shmr(
    memories: List[Dict],
    dream_run_id: int,
    dry_run: bool = False,
) -> Dict:
    """
    Run Self-Harmonizing Memory Reasoning.

    1. Cluster memories by vector similarity (connected components)
    2. LLM harmonizes each cluster → resolve contradictions, extract beliefs
    3. Write harmonic_beliefs to dream_cycle.db

    Returns stats dict.
    """
    log.info(f"🔔 SHMR: Self-Harmonizing Memory Reasoning{' (dry-run)' if dry_run else ''}")

    stats = {
        "clusters_formed": 0,
        "beliefs_created": 0,
        "contradictions_dampened": 0,
        "beliefs_updated": 0,
    }

    # Filter to memories with enough text
    candidates = [m for m in memories if len(m.get("text", "")) > 20]
    if len(candidates) < SHMR_MIN_CLUSTER_SIZE:
        log.info(f"  ⏭ SHMR: 候选记忆太少 ({len(candidates)} < {SHMR_MIN_CLUSTER_SIZE})")
        return stats

    # Cluster
    clusters = _cluster_by_similarity(candidates, SHMR_SIMILARITY_THRESHOLD)
    # Only keep clusters with ≥SHMR_MIN_CLUSTER_SIZE members
    clusters = [c for c in clusters if len(c) >= SHMR_MIN_CLUSTER_SIZE]

    if not clusters:
        log.info("  ⏭ SHMR: 没有形成有效聚类")
        return stats

    stats["clusters_formed"] = len(clusters)
    log.info(f"  📦 形成 {len(clusters)} 个聚类 (从 {len(candidates)} 条记忆)")

    if dry_run:
        for i, c in enumerate(clusters[:3]):
            log.info(f"  🔔 Cluster {i}: {len(c)} 条 — {c[0]['text'][:60]}...")
        return stats

    # Init schema
    conn = sqlite3.connect(str(DREAM_DB), timeout=30.0)
    _init_schema(conn)

    # Harmonize each cluster
    for i, cluster in enumerate(clusters):
        cluster_id = f"shmr_{dream_run_id}_{i}"
        log.info(f"  🔔 Harmonizing cluster {i+1}/{len(clusters)}: {len(cluster)} memories")

        beliefs = _harmonize_cluster(cluster, cluster_id)
        if not beliefs:
            log.info(f"    ⚠️ No beliefs produced")
            continue

        provenance_ids = [m.get("id", "") for m in cluster]

        for b in beliefs:
            belief_id = str(uuid.uuid4())[:12]
            try:
                conn.execute(
                    """INSERT INTO harmonic_beliefs
                    (belief_id, cluster_id, subject, predicate, confidence,
                     provenance, action, rationale, dream_run_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        belief_id,
                        cluster_id,
                        b.get("subject", ""),
                        b["predicate"],
                        b["confidence"],
                        json.dumps(provenance_ids[:10]),
                        b["action"],
                        b.get("rationale", ""),
                        dream_run_id,
                    ),
                )

                if b["action"] == "create":
                    stats["beliefs_created"] += 1
                elif b["action"] == "update":
                    stats["beliefs_updated"] += 1
                elif b["action"] == "dampen":
                    stats["contradictions_dampened"] += 1

            except sqlite3.Error as e:
                log.debug(f"    DB insert failed: {e}")

        log.info(
            f"    ✅ {len(beliefs)} beliefs "
            f"(create={sum(1 for b in beliefs if b['action']=='create')}, "
            f"dampen={sum(1 for b in beliefs if b['action']=='dampen')})"
        )

    conn.commit()
    conn.close()

    log.info(
        f"  📊 SHMR 完成: {stats['clusters_formed']} clusters → "
        f"{stats['beliefs_created']} new + {stats['beliefs_updated']} updated + "
        f"{stats['contradictions_dampened']} dampened beliefs"
    )
    return stats
