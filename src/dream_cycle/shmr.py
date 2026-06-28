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
    "contrastive_beliefs",
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


# ─── P1-2: Contrastive Reflection (SkillOpt-inspired) ────────────────
# Ported from SkillOpt-Sleep rollout.py contrastive_reflect():
# "What did the good attempts do that the bad ones didn't?"
#
# For SHMR: find memories where the same topic/entity appeared in both
# successful and failed sessions. Extract the decisive factors.

CONTRASTIVE_PROMPT = """You are a contrastive analyst. These memories all relate to the same topic or entity, but some come from sessions where the user was satisfied and some from sessions where the user was frustrated or corrected the agent.

Your job: identify what distinguishes the GOOD outcomes from the BAD ones.

**Good outcomes (user satisfied):**
{good_memories}

**Bad outcomes (user corrected/frustrated):**
{bad_memories}

Find 1-3 decisive factors — what made the difference?
Output as JSON array:
[{{"factor": "one-sentence description of the decisive factor",
  "evidence_good": "what good outcomes did",
  "evidence_bad": "what bad outcomes did differently",
  "confidence": 0.0-1.0,
  "actionable": true/false}}]

RULES:
- Focus on concrete behavioral differences, not vague generalizations
- "actionable" = can we encode this as a rule for future sessions?
- Confidence 0.8+ = clear pattern across multiple examples
- Confidence <0.5 = speculative, based on few examples
- Output 1-3 factors maximum"""


def contrastive_beliefs(
    memories: List[Dict],
    session_outcomes: Dict[str, str],
    dream_run_id: int,
    dry_run: bool = False,
) -> Dict:
    """Run contrastive reflection on memories grouped by session outcome.

    Ported from SkillOpt-Sleep: finds topics that appeared in both
    successful and failed sessions, extracts what made the difference.

    Args:
        memories: List of memory dicts (must have 'session_id' or source info)
        session_outcomes: {session_id: 'success'|'fail'|'mixed'|'unknown'}
        dream_run_id: Current dream run ID for provenance
        dry_run: If True, only analyze without LLM calls

    Returns stats dict.
    """
    stats = {
        "contrastive_clusters": 0,
        "factors_extracted": 0,
        "actionable_rules": 0,
    }

    # Group memories by topic (entity/title prefix) and outcome
    # Use first 30 chars of text as a rough topic key
    topic_groups: Dict[str, Dict[str, List[Dict]]] = {}  # topic → {"good": [...], "bad": [...]}

    for mem in memories:
        text = mem.get("text", "")
        if len(text) < 20:
            continue

        # Get session outcome for this memory
        session_id = mem.get("session_id", "")
        outcome = session_outcomes.get(session_id, "unknown")
        if outcome == "unknown":
            continue  # Skip memories without clear outcomes

        # Use topic keywords (first meaningful noun phrase) as grouping key
        topic_key = text[:40].strip().lower()
        # Remove common prefixes to improve grouping
        for prefix in ["user prefers", "user corrected", "[session_", "the user"]:
            if topic_key.startswith(prefix):
                topic_key = topic_key[len(prefix):].strip()

        if topic_key not in topic_groups:
            topic_groups[topic_key] = {"good": [], "bad": []}

        if outcome == "success":
            topic_groups[topic_key]["good"].append(mem)
        elif outcome == "fail":
            topic_groups[topic_key]["bad"].append(mem)

    # Filter: need both good AND bad examples for contrastive analysis
    contrastive_topics = {
        k: v for k, v in topic_groups.items()
        if len(v["good"]) >= 1 and len(v["bad"]) >= 1
    }

    if not contrastive_topics:
        log.info("  ⏭ Contrastive: no topics with both good and bad outcomes")
        return stats

    stats["contrastive_clusters"] = len(contrastive_topics)
    log.info(f"  🔍 Contrastive: {len(contrastive_topics)} topics with mixed outcomes")

    if dry_run:
        for topic, groups in list(contrastive_topics.items())[:3]:
            log.info(f"    • '{topic[:40]}': {len(groups['good'])} good, {len(groups['bad'])} bad")
        return stats

    # Init schema (reuse SHMR table)
    conn = sqlite3.connect(str(DREAM_DB), timeout=30.0)
    _init_schema(conn)

    for topic, groups in contrastive_topics.items():
        good_texts = "\n".join(
            f"  - {m['text'][:200]}" for m in groups["good"][:5]
        )
        bad_texts = "\n".join(
            f"  - {m['text'][:200]}" for m in groups["bad"][:5]
        )

        prompt = CONTRASTIVE_PROMPT.format(
            good_memories=good_texts,
            bad_memories=bad_texts,
        )

        try:
            response = _call_infini(
                prompt,
                system="You are a contrastive analyst. Output only valid JSON.",
                max_tokens=800,
            )
            if not response:
                continue

            # Parse JSON
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            factors = json.loads(text)
            if not isinstance(factors, list):
                factors = [factors]

            for f in factors[:3]:
                if not isinstance(f, dict) or not f.get("factor"):
                    continue

                belief_id = str(uuid.uuid4())[:12]
                confidence = max(0.0, min(1.0, float(f.get("confidence", 0.5))))
                actionable = bool(f.get("actionable", False))

                provenance_ids = (
                    [m.get("id", "") for m in groups["good"][:5]] +
                    [m.get("id", "") for m in groups["bad"][:5]]
                )

                conn.execute(
                    """INSERT INTO harmonic_beliefs
                    (belief_id, cluster_id, subject, predicate, confidence,
                     provenance, action, rationale, dream_run_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        belief_id,
                        f"contrastive_{dream_run_id}",
                        topic[:60],
                        f["factor"],
                        confidence,
                        json.dumps(provenance_ids),
                        "create",
                        f"GOOD: {f.get('evidence_good', '')[:100]} | BAD: {f.get('evidence_bad', '')[:100]}",
                        dream_run_id,
                    ),
                )

                stats["factors_extracted"] += 1
                if actionable:
                    stats["actionable_rules"] += 1

            log.info(
                f"    ✅ '{topic[:30]}': {len(factors)} factors "
                f"({sum(1 for f in factors if f.get('actionable'))} actionable)"
            )

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            log.debug(f"    Contrastive parse failed for '{topic[:30]}': {e}")
        except Exception as e:
            log.debug(f"    Contrastive failed for '{topic[:30]}': {e}")

    conn.commit()
    conn.close()

    log.info(
        f"  📊 Contrastive 完成: {stats['contrastive_clusters']} topics → "
        f"{stats['factors_extracted']} factors ({stats['actionable_rules']} actionable)"
    )
    return stats
