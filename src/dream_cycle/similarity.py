"""
Dream Cycle — Similarity functions
===================================
Text similarity (Jaccard, n-gram, combined), pgvector nearest-neighbor,
batch vector clustering, and query–memory matching.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

from dream_cycle.config import (
    DEDUP_DIST, MERGE_DIST, CLUSTER_DIST,
    DEDUP_THRESHOLD, MERGE_THRESHOLD, CLUSTER_THRESHOLD,
    safe_float, log,
)
# pg_query imported lazily inside functions that need it

__all__ = [
    "text_hash",
    "jaccard_similarity",
    "ngram_similarity",
    "combined_similarity",
    "get_vector_neighbors",
    "batch_vector_clustering",
    "match_memory_to_queries",
]


# ─── Text fingerprints & similarity ─────────────────────────────────




def jaccard_similarity(s1: str, s2: str) -> float:
    """Word-level Jaccard similarity (fast, no vectors needed)."""
    w1 = set(s1.lower().split())
    w2 = set(s2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def ngram_similarity(s1: str, s2: str, n: int = 3) -> float:
    """Character n-gram similarity (more sensitive than Jaccard)."""
    def ngrams(text: str, n: int) -> set[str]:
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    ng1 = ngrams(s1.lower(), n)
    ng2 = ngrams(s2.lower(), n)
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)


def combined_similarity(s1: str, s2: str) -> float:
    """Blended similarity: 40 % Jaccard + 60 % n-gram."""
    return 0.4 * jaccard_similarity(s1, s2) + 0.6 * ngram_similarity(s1, s2)


# ─── pgvector nearest-neighbor ────────────────────────────────────────


def get_vector_neighbors(
    memory_id: str,
    limit: int = 10,
    max_dist: float = CLUSTER_DIST,
) -> list[dict]:
    """
    Find nearest neighbors via pgvector cosine distance.

    Returns list of ``{"id": str, "text": str, "distance": float}``.
    """
    from dream_cycle.db import pg_query
    sql = f"""\
        SELECT b.id::text,
               LEFT(b.payload->>'data', 200) AS text,
               ROUND((a.vector <=> b.vector)::numeric, 4) AS dist
          FROM mem0 a, mem0 b
         WHERE a.id::text = '{memory_id}'
           AND a.id != b.id
           AND (a.vector <=> b.vector) < {max_dist}
           AND b.payload->>'data' IS NOT NULL
      ORDER BY a.vector <=> b.vector
         LIMIT {limit}
    """
    rows = pg_query(sql)
    neighbors: list[dict] = []
    for r in rows:
        if len(r) >= 3 and r[1]:
            dist = safe_float(r[2])
            if dist is None:
                continue
            neighbors.append({"id": r[0], "text": r[1], "distance": dist})
    return neighbors


def batch_vector_clustering(
    memory_ids: list[str],
    max_dist: float = CLUSTER_DIST,
) -> dict[str, list[str]]:
    """
    Build an adjacency list of vector-similar pairs among *memory_ids*.

    Returns ``{memory_id: [neighbor_id, ...]}`` (symmetric).
    """
    from dream_cycle.db import pg_query
    if len(memory_ids) < 2:
        return {}

    id_list = "','".join(memory_ids)
    sql = f"""\
        SELECT a.id::text AS source,
               b.id::text AS neighbor,
               ROUND((a.vector <=> b.vector)::numeric, 4) AS dist
          FROM mem0 a, mem0 b
         WHERE a.id::text IN ('{id_list}')
           AND b.id::text IN ('{id_list}')
           AND a.id < b.id
           AND (a.vector <=> b.vector) < {max_dist}
      ORDER BY dist
    """
    rows = pg_query(sql)

    graph: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if len(r) >= 3:
            dist = safe_float(r[2])
            if dist is None:
                continue
            src, nbr = r[0], r[1]
            graph[src].add(nbr)
            graph[nbr].add(src)

    return {k: list(v) for k, v in graph.items()}


# ─── Query ↔ memory matching ────────────────────────────────────────


def match_memory_to_queries(
    memory_text: str,
    query_stats: dict[str, int],
) -> tuple[int, int]:
    """
    Match a memory's text against search query statistics.

    Returns ``(recall_count, session_count)``:

    * **recall_count** – total search calls that matched this memory.
    * **session_count** – distinct queries that matched (diversity proxy).
    """
    text_lower = memory_text.lower()
    recall_count = 0
    matched_queries: set[str] = set()

    for query, count in query_stats.items():
        query_words = [w for w in query.lower().split() if len(w) > 2]
        if not query_words:
            continue
        matched_words = sum(1 for w in query_words if w in text_lower)
        if matched_words / len(query_words) >= 0.5:
            recall_count += count
            matched_queries.add(query)

    return recall_count, len(matched_queries)
