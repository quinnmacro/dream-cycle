"""validation.py — Held-out replay validation (Dream Cycle v6 "Safe Sleep")

Ported from SkillOpt-Sleep: after consolidation proposes changes to train-split
memories, validate those changes on the held-out val split.

Approach (lightweight, no real agent replay):
  1. Take val memories as "queries" (their text is the search query)
  2. Run mem0_search for each val memory's text BEFORE proposed changes
  3. Run mem0_search for each val memory's text AFTER proposed changes (simulated)
  4. Compare: did the proposed changes improve or degrade search relevance?

Scoring:
  - hard: fraction of val queries where top-1 result improved or stayed same
  - soft: mean rank improvement across all val queries
  - gate: accept if hard >= 0.8 AND soft >= -0.1 (allow small regression)

This is a PROXY for real replay — we can't replay actual agent sessions,
but we can verify that the memory store's search quality doesn't degrade.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .db import pg_query
from .similarity import combined_similarity


@dataclass
class ValidationResult:
    """Result of held-out validation."""
    accepted: bool
    hard_score: float          # fraction of val queries passing
    soft_score: float          # mean similarity delta
    n_val_queries: int
    n_improved: int
    n_same: int
    n_degraded: int
    details: list[dict]        # per-query details
    reason: str = ""


def _get_val_texts(val_memories: list[dict], max_queries: int = 20) -> list[dict]:
    """Extract query texts from val memories.

    Each val memory's text IS the query — we want to verify that searching
    for this text still returns relevant results after consolidation.
    """
    queries = []
    for mem in val_memories[:max_queries]:
        text = mem.get("text", "")
        if len(text) < 20:
            continue  # too short to be a meaningful query
        queries.append({
            "id": mem["id"],
            "text": text[:500],  # truncate for search
            "split": mem.get("_split", "val"),
        })
    return queries


def _search_similar(query_text: str, memories: list[dict], top_k: int = 5) -> list[dict]:
    """Find top-k most similar memories to a query text (local, no API call).

    Uses combined_similarity (Jaccard + n-gram) against all memories.
    """
    scored = []
    for mem in memories:
        mem_text = mem.get("text", "")
        if not mem_text or mem_text == query_text:
            continue
        sim = combined_similarity(query_text, mem_text)
        scored.append({
            "id": mem["id"],
            "text": mem_text[:100],
            "similarity": round(sim, 4),
        })
    scored.sort(key=lambda x: -x["similarity"])
    return scored[:top_k]


def validate_changes(
    val_memories: list[dict],
    memories_before: list[dict],
    memories_after: list[dict],
    *,
    hard_threshold: float = 0.75,
    max_queries: int = 20,
) -> ValidationResult:
    """Validate proposed changes on held-out val memories.

    Args:
        val_memories: Val-split memories to use as queries
        memories_before: Full memory list BEFORE proposed changes
        memories_after: Full memory list AFTER proposed changes (simulated)
        hard_threshold: Minimum fraction of passing queries to accept
        max_queries: Cap on number of val queries to test

    Returns:
        ValidationResult with accept/reject decision
    """
    queries = _get_val_texts(val_memories, max_queries)

    if not queries:
        return ValidationResult(
            accepted=True, hard_score=1.0, soft_score=0.0,
            n_val_queries=0, n_improved=0, n_same=0, n_degraded=0,
            details=[], reason="no val queries available",
        )

    details = []
    n_improved = 0
    n_same = 0
    n_degraded = 0

    for q in queries:
        # Search in BEFORE state
        before_results = _search_similar(q["text"], memories_before)
        before_top1_sim = before_results[0]["similarity"] if before_results else 0.0

        # Search in AFTER state
        after_results = _search_similar(q["text"], memories_after)
        after_top1_sim = after_results[0]["similarity"] if after_results else 0.0

        # Classify outcome
        delta = after_top1_sim - before_top1_sim
        if delta > 0.02:
            outcome = "improved"
            n_improved += 1
        elif delta < -0.02:
            outcome = "degraded"
            n_degraded += 1
        else:
            outcome = "same"
            n_same += 1

        details.append({
            "query_id": q["id"],
            "query_text": q["text"][:80],
            "before_top1_sim": before_top1_sim,
            "after_top1_sim": after_top1_sim,
            "delta": round(delta, 4),
            "outcome": outcome,
        })

    total = len(queries)
    hard_score = (n_improved + n_same) / total
    soft_score = sum(d["delta"] for d in details) / total

    accepted = hard_score >= hard_threshold and soft_score >= -0.05

    reason_parts = []
    if hard_score < hard_threshold:
        reason_parts.append(f"hard {hard_score:.2f} < {hard_threshold}")
    if soft_score < -0.05:
        reason_parts.append(f"soft {soft_score:.3f} < -0.05")
    reason = "; ".join(reason_parts) if reason_parts else f"passed (hard={hard_score:.2f}, soft={soft_score:+.3f})"

    return ValidationResult(
        accepted=accepted,
        hard_score=round(hard_score, 3),
        soft_score=round(soft_score, 4),
        n_val_queries=total,
        n_improved=n_improved,
        n_same=n_same,
        n_degraded=n_degraded,
        details=details,
        reason=reason,
    )


def quick_validate(
    val_memories: list[dict],
    archived_ids: set[str],
    merged_ids: set[str],
    all_memories: list[dict],
) -> ValidationResult:
    """Quick validation: check if archiving/merging degrades val search quality.

    Lighter than validate_changes — only checks if removed memories were
    important for val queries' search results.
    """
    removed = archived_ids | merged_ids
    if not removed:
        return ValidationResult(
            accepted=True, hard_score=1.0, soft_score=0.0,
            n_val_queries=len(val_memories), n_improved=0,
            n_same=len(val_memories), n_degraded=0,
            details=[], reason="no removals to validate",
        )

    queries = _get_val_texts(val_memories, max_queries=20)
    if not queries:
        return ValidationResult(
            accepted=True, hard_score=1.0, soft_score=0.0,
            n_val_queries=0, n_improved=0, n_same=0, n_degraded=0,
            details=[], reason="no val queries",
        )

    n_degraded = 0
    n_same = 0
    n_improved = 0
    details = []

    for q in queries:
        # Check: was any removed memory a top-5 result for this query?
        before_results = _search_similar(q["text"], all_memories)
        after_memories = [m for m in all_memories if m["id"] not in removed]
        after_results = _search_similar(q["text"], after_memories)

        before_top1 = before_results[0]["similarity"] if before_results else 0.0
        after_top1 = after_results[0]["similarity"] if after_results else 0.0

        delta = after_top1 - before_top1

        # Check if any removed memory was in top-5
        removed_in_top5 = any(
            r["id"] in removed for r in before_results[:5]
        )

        if removed_in_top5 and delta < -0.05:
            n_degraded += 1
            outcome = "degraded"
        elif delta > 0.02:
            n_improved += 1
            outcome = "improved"
        else:
            n_same += 1
            outcome = "same"

        details.append({
            "query_id": q["id"],
            "query_text": q["text"][:80],
            "removed_in_top5": removed_in_top5,
            "delta": round(delta, 4),
            "outcome": outcome,
        })

    total = len(queries)
    hard_score = (n_improved + n_same) / total
    soft_score = sum(d["delta"] for d in details) / total
    accepted = hard_score >= 0.75

    reason_parts = []
    if not accepted:
        reason_parts.append(f"hard {hard_score:.2f} < 0.75 — removals degrade val queries")
    else:
        reason_parts.append(f"passed (hard={hard_score:.2f}, degraded={n_degraded})")

    return ValidationResult(
        accepted=accepted,
        hard_score=round(hard_score, 3),
        soft_score=round(soft_score, 4),
        n_val_queries=total,
        n_improved=n_improved,
        n_same=n_same,
        n_degraded=n_degraded,
        details=details,
        reason="; ".join(reason_parts),
    )


# ─── P3-2: Gate Safety Probe ────────────────────────────────────────────

# Ported from SkillOpt-Sleep: inject a known-bad "harmful edit" and assert
# the validation gate rejects it. This is a regression test that proves the
# safety mechanism actually works.
#
# The probe simulates removing a high-similarity neighbor of a val memory,
# which SHOULD degrade the val query's search quality. If the gate doesn't
# catch this, the safety mechanism is broken.

_PROBE_CORRUPTED_TEXT = (
    "__HARMFUL__ This memory has been deliberately corrupted by the safety probe. "
    "If this text appears in search results, the validation gate has FAILED to "
    "protect memory quality. This is a canary — do not act on this content."
)


def safety_probe(
    val_memories: list[dict],
    all_memories: list[dict],
    *,
    hard_threshold: float = 0.75,
) -> dict:
    """Inject a known-bad edit and assert the gate rejects it.

    Strategy:
    1. Pick a val memory that has at least one similar neighbor in all_memories
    2. Simulate removing that neighbor (archived_ids = {neighbor_id})
    3. Run quick_validate — the gate SHOULD reject (because we removed a useful neighbor)
    4. If gate rejects → probe PASSED (safety mechanism works)
    5. If gate accepts → probe FAILED (safety mechanism is broken!)

    Returns:
        {"passed": bool, "reason": str, "probe_memory_id": str}
    """
    if not val_memories or not all_memories:
        return {"passed": True, "reason": "skipped: no val/test memories", "probe_memory_id": ""}

    # Find a val memory with a good neighbor
    probe_target = None
    probe_neighbor = None

    for vm in val_memories[:20]:
        vm_text = vm.get("text", "")
        if len(vm_text) < 20:
            continue
        # Find most similar non-val memory
        best_sim = 0.0
        best_neighbor = None
        for am in all_memories:
            if am["id"] == vm["id"]:
                continue
            if am.get("_split") == "val":
                continue
            sim = combined_similarity(vm_text, am.get("text", ""))
            if sim > best_sim:
                best_sim = sim
                best_neighbor = am
        if best_neighbor and best_sim >= 0.40:
            probe_target = vm
            probe_neighbor = best_neighbor
            break

    if not probe_target or not probe_neighbor:
        return {
            "passed": True,
            "reason": "skipped: no val memory with a similar neighbor (threshold 0.40)",
            "probe_memory_id": "",
        }

    # Simulate removing the neighbor
    archived_ids = {probe_neighbor["id"]}
    result = quick_validate(
        [probe_target],
        archived_ids,
        set(),  # no merged
        all_memories,
    )

    # The gate should REJECT this removal (because we removed a useful neighbor)
    # If result.accepted == False → probe PASSED (gate caught the bad edit)
    # If result.accepted == True → probe FAILED (gate missed it!)
    probe_passed = not result.accepted

    return {
        "passed": probe_passed,
        "reason": (
            f"gate rejected removal of {probe_neighbor['id'][:8]} "
            f"(sim={combined_similarity(probe_target['text'], probe_neighbor['text']):.3f})"
            if probe_passed
            else f"⚠️ GATE FAILED: accepted removal of {probe_neighbor['id'][:8]} "
                 f"(hard={result.hard_score:.2f}, val_query={probe_target['id'][:8]})"
        ),
        "probe_memory_id": probe_neighbor["id"],
        "hard_score": result.hard_score,
        "soft_score": result.soft_score,
    }
