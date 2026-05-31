"""
Dream Cycle — Stage 1: Shallow Sleep — topic clustering + vector clustering + singleton detection
"""

__all__ = [
    "stage1_shallow_sleep",
]

from collections import defaultdict
from dream_cycle.config import (
    CLUSTER_THRESHOLD,
    CLUSTER_DIST,
    log,
)
from dream_cycle.similarity import (
    text_hash,
    combined_similarity,
    batch_vector_clustering,
)
from dream_cycle.entities import extract_topic_key


def stage1_shallow_sleep(memories: list[dict]) -> dict[str, list[dict]]:
    """
    浅睡: 将记忆按主题聚类分组

    方法:
    1. 实体级聚类 (相同项目/技能/话题 → 同组)
    2. 精确去重 (hash)
    3. 文本相似度聚类 (n-gram)
    4. 提取每组的主题关键词
    """
    log.info(f"💤 Stage 1: Shallow Sleep — 聚类 {len(memories)} 条记忆")

    clusters: dict[str, list[dict]] = {}  # cluster_id → [memories]
    assigned: dict[str, str] = {}  # memory_id → cluster_id

    # ── Phase A: 实体级聚类 (最强信号) ──
    topic_groups: dict[str, list[dict]] = defaultdict(list)
    no_topic: list[dict] = []

    for m in memories:
        topic_key = extract_topic_key(m["text"])
        if topic_key:
            topic_groups[topic_key].append(m)
        else:
            no_topic.append(m)

    for topic_key, group in topic_groups.items():
        ck = f"topic_{topic_key[:40]}"
        clusters[ck] = group
        for m in group:
            assigned[m["id"]] = ck

    log.info(
        f"  📌 实体聚类: {len(topic_groups)} 主题组, {sum(len(v) for v in topic_groups.values())} 条"
    )

    # ── Phase B: 精确去重 (hash) ──
    remaining = [m for m in no_topic if m["id"] not in assigned]
    hash_groups: dict[str, list[dict]] = defaultdict(list)
    for m in remaining:
        h = text_hash(m["text"])
        hash_groups[h].append(m)

    exact_dedup_count = 0
    for h, group in hash_groups.items():
        if len(group) > 1:
            ck = f"exact_{h}"
            clusters[ck] = group
            exact_dedup_count += len(group) - 1
            for m in group:
                assigned[m["id"]] = ck
        elif group[0]["id"] not in assigned:
            remaining_unassigned = [m for m in remaining if m["id"] not in assigned]

    # ── Phase C: 向量相似度聚类 (pgvector, 精度最高) ──
    remaining = [m for m in memories if m["id"] not in assigned]
    if len(remaining) >= 2:
        remaining_ids = [m["id"] for m in remaining]
        vector_graph = batch_vector_clustering(remaining_ids, max_dist=CLUSTER_DIST)

        # 从向量图构建连通分量
        visited: set[str] = set()
        vec_cluster_id = 0
        id_to_mem = {m["id"]: m for m in remaining}

        for mid in remaining_ids:
            if mid in visited or mid not in vector_graph:
                continue
            # BFS 找连通分量
            queue = [mid]
            component = []
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for nbr in vector_graph.get(node, []):
                    if nbr not in visited:
                        queue.append(nbr)

            if len(component) >= 2:
                ck = f"vec_{vec_cluster_id}"
                clusters[ck] = [id_to_mem[cid] for cid in component if cid in id_to_mem]
                for cid in component:
                    assigned[cid] = ck
                vec_cluster_id += 1

        vec_count = sum(1 for k in clusters if k.startswith("vec_"))
        vec_mem_count = sum(len(v) for k, v in clusters.items() if k.startswith("vec_"))
        if vec_count > 0:
            log.info(f"  🔢 向量聚类: {vec_count} 组, {vec_mem_count} 条")

    remaining = [m for m in memories if m["id"] not in assigned]

    # ── Phase D: 文本相似度聚类 (fallback) ──
    cluster_id = 0
    for i, m1 in enumerate(remaining):
        if m1["id"] in assigned:
            continue
        ck = f"sim_{cluster_id}"
        clusters[ck] = [m1]
        assigned[m1["id"]] = ck

        for j in range(i + 1, len(remaining)):
            m2 = remaining[j]
            if m2["id"] in assigned:
                continue
            sim = combined_similarity(m1["text"], m2["text"])
            if sim >= CLUSTER_THRESHOLD:
                clusters[ck].append(m2)
                assigned[m2["id"]] = ck

        cluster_id += 1

    # Singleton
    for m in memories:
        if m["id"] not in assigned:
            clusters[f"singleton_{m['id'][:8]}"] = [m]

    multi_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    log.info(
        f"  ✅ 聚类完成: {len(clusters)} 组, {len(multi_clusters)} 多条组, "
        f"{exact_dedup_count} 精确重复, {len(topic_groups)} 主题组"
    )

    # 统计最有价值的组
    for k, v in sorted(multi_clusters.items(), key=lambda x: len(x[1]), reverse=True)[
        :5
    ]:
        log.info(f"     {k}: {len(v)} 条")

    return clusters


# ─── Stage 2: REM (快速眼动) — 重要性评分 ────────────────────────────
