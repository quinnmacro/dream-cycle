"""
Dream Cycle — Stage 2: REM — FadeMem dual-layer scoring, contradiction detection, vault candidates
"""

__all__ = [
    "stage2_rem",
    "score_importance",
    "classify_decay_tier",
]

import re
import math
from datetime import datetime, timezone
from collections import defaultdict
from dream_cycle.config import (
    RETENTION_FLOOR,
    FADEMEM_BETA,
    DECAY_HALF_LIVES,
    ARCHIVE_THRESHOLD_DAYS,
    PROMOTION_MIN_SCORE,
    PROMOTION_MIN_RECALLS,
    PROMOTION_MIN_SESSIONS,
    PERMANENT_MARKERS,
    IMPORTANCE_WEIGHTS,
    DEDUP_THRESHOLD,
    MERGE_THRESHOLD,
    DEDUP_DIST,
    MERGE_DIST,
    log,
)
from dream_cycle.config import safe_float
from dream_cycle.db import pg_query, get_recall_stats
from dream_cycle.similarity import (
    combined_similarity,
    match_memory_to_queries,
    get_vector_neighbors,
    batch_vector_clustering,
)
from dream_cycle.entities import _is_valid_entity, extract_entities_with_fallback
from dream_cycle.llm import llm_verify_contradiction
from dream_cycle.vault import _compute_memory_age_days


def classify_decay_tier(text: str) -> str:
    """
    FadeMem: 分类记忆衰减层级

    - volatile (β=1.2): 市场数据、价格、实时新闻 → 快速衰减
    - stable (β=0.8): 用户偏好、基础设施、个人信息 → 慢衰减
    - normal (β=1.0): 默认
    """
    text_lower = text.lower()

    # Volatile: market data, prices, time-sensitive
    volatile_kw = [
        "yield",
        "spread",
        "bp",
        "bps",
        "price",
        "价格",
        "利率",
        "利差",
        "today",
        "yesterday",
        "今天",
        "昨天",
        "收盘",
        "开盘",
        "实时",
        "breaking",
        "突发",
        "just announced",
        "刚发布",
        "非农",
        "GDP",
        "CPI",
        "PMI",
        "NFP",
        "Fed meeting",
        "央行",
        "data release",
        "stock",
        "股价",
        "ticker",
        "market close",
        "intraday",
    ]

    # Stable: user preferences, infrastructure, personal info
    stable_kw = [
        "prefer",
        "偏好",
        "喜欢",
        "like",
        "always",
        "never",
        "from now on",
        "server",
        "服务器",
        "config",
        "setup",
        "部署",
        "infrastructure",
        "password",
        "key",
        "credential",
        "user_id",
        "api_key",
        "my name",
        "I am",
        "我是",
        "my role",
        "我的角色",
        "architecture",
        "framework",
        "design pattern",
        "convention",
        "standard",
        "规范",
    ]

    volatile_hits = sum(1 for kw in volatile_kw if kw in text_lower)
    stable_hits = sum(1 for kw in stable_kw if kw in text_lower)

    if volatile_hits >= 2:
        return "volatile"
    elif stable_hits >= 2:
        return "stable"
    elif volatile_hits > stable_hits:
        return "volatile"
    elif stable_hits > volatile_hits:
        return "stable"
    return "normal"


def score_importance(
    memory: dict, recall_count: int = 0, session_count: int = 0
) -> float:
    """
    REM: 6维重要性评分 (对齐 OpenClaw Dreaming)

    维度:
    - recency(15%): Ebbinghaus 时间衰减, 14天半衰期
    - frequency(24%): 被搜索/引用次数 (recall_count)
    - query_diversity(15%): 不同 session 命中数
    - domain(20%): 投资(0.9) > 技术(0.6) > 日常(0.3)
    - consolidation(16%): 跨 session 重现强度
    - confidence(10%): 信息密度代理
    """
    scores = {}

    # Recency (FadeMem dual-layer: R = e^(-λ·t^β))
    # volatile β=1.2 (fast), stable β=0.8 (slow), normal β=1.0
    try:
        created = memory.get("created_at", "")
        if created:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
            text = memory.get("text", "")
            tier = classify_decay_tier(text)
            half_life = DECAY_HALF_LIVES.get(tier, 7)
            beta = FADEMEM_BETA.get(tier, 1.0)
            # λ = ln(2) / half_life, then R = e^(-λ · t^β)
            lam = math.log(2) / half_life
            retention = math.exp(-lam * (age_days**beta))
            scores["recency"] = max(RETENTION_FLOOR, retention)
            scores["decay_tier"] = tier  # track for reporting
        else:
            scores["recency"] = 0.5
            scores["decay_tier"] = "normal"
    except Exception:
        scores["recency"] = 0.5
        scores["decay_tier"] = "normal"

    # Frequency (被召回次数, 来自 mem0_search 统计)
    scores["frequency"] = (
        min(1.0, math.log2(recall_count + 1) / 4) if recall_count > 0 else 0.1
    )

    # Query diversity (跨 session 命中)
    scores["query_diversity"] = (
        min(1.0, session_count / 5) if session_count > 0 else 0.1
    )

    # Domain (关键词判定)
    text = memory.get("text", "")
    investment_kw = [
        "债券",
        "利率",
        "利差",
        "bonds",
        "yield",
        "spread",
        "CGB",
        "UST",
        "信用",
        "carry",
        "duration",
        "PM",
        "trade",
        "signal",
        "CME",
        "COMEX",
    ]
    tech_kw = [
        "docker",
        "python",
        "git",
        "MCP",
        "plugin",
        "skill",
        "API",
        "config",
        "Neo4j",
        "mem0",
        "cron",
        "deploy",
    ]

    text_lower = text.lower()
    if any(kw.lower() in text_lower for kw in investment_kw):
        scores["domain"] = 0.9
    elif any(kw.lower() in text_lower for kw in tech_kw):
        scores["domain"] = 0.6
    else:
        scores["domain"] = 0.3

    # Consolidation (跨天/跨 session 重现)
    # 用 recall_count 和 session_count 的几何平均
    if (
        recall_count >= PROMOTION_MIN_RECALLS
        and session_count >= PROMOTION_MIN_SESSIONS
    ):
        scores["consolidation"] = min(1.0, math.sqrt(recall_count * session_count) / 4)
    elif session_count >= 2:
        scores["consolidation"] = 0.4
    else:
        scores["consolidation"] = 0.1

    # Confidence (长度+唯一词比)
    word_count = len(text.split())
    unique_words = len(set(text.lower().split()))
    if word_count > 50:
        scores["confidence"] = min(1.0, (unique_words / max(word_count, 1)) * 2)
    else:
        scores["confidence"] = 0.3

    # Novelty (SCM ValueTagger 核心 — 与已有记忆的语义距离)
    # 新的、意想不到的信息得分更高
    # 用 pgvector 最近邻距离作为 novelty 代理
    # 如果没有向量，用文本相似度 fallback
    try:
        neighbors = get_vector_neighbors(memory["id"], limit=3, max_dist=0.50)
        if neighbors:
            min_dist = min(n["distance"] for n in neighbors)
            # distance 越大 = 越不相似 = 越新奇 → novelty 越高
            scores["novelty"] = min(1.0, min_dist / 0.50)
        else:
            # 无近邻 = 高度独特 = 高 novelty
            scores["novelty"] = 0.9
    except Exception:
        # fallback: 用文本长度和唯一词比估算
        if word_count > 20:
            scores["novelty"] = min(1.0, (unique_words / max(word_count, 1)))
        else:
            scores["novelty"] = 0.5

    # 优先标记检测 (永不归档)
    for marker in PERMANENT_MARKERS:
        if marker in text:
            scores["confidence"] = 1.0  # 满分
            break

    # 加权平均
    total = sum(IMPORTANCE_WEIGHTS[k] * scores.get(k, 0.1) for k in IMPORTANCE_WEIGHTS)
    return total


# ─── P0: 自适应触发机制 (SCM + SleepGate) ─────────────────────────────


def stage2_rem(clusters: dict[str, list[dict]], neo4j_connections: dict = None) -> dict:
    """
    REM: 7维评分 + 三重门限提升 + 矛盾检测

    对每条记忆打分, 对每组:
    - 标记最重要的记忆 (keep)
    - 标记可合并/删除的候选
    - 标记 Vault 沉淀候选 (三重门限: 分数+召回+跨session)
    - 检测矛盾事实 (KektorDB Gardener 模式)
    """
    log.info(f"👁️ Stage 2: REM — 评分 {sum(len(v) for v in clusters.values())} 条记忆")

    results = {
        "boosted": [],  # 高重要性, 建议 Boost
        "dedup_candidates": [],  # 可去重
        "merge_candidates": [],  # 可合并
        "vault_candidates": [],  # 建议沉淀到 Vault (三重门限)
        "decay_candidates": [],  # 建议归档 (永不删除)
        "contradictions": [],  # 矛盾事实对
    }

    # P2-1: 加载真实 recall 统计
    recall_stats = get_recall_stats()

    for cluster_key, group in clusters.items():
        # 评分: 优先用真实 recall 统计, fallback 到启发式
        scored = []
        for m in group:
            # P2-1: 真实 recall 匹配
            real_rc, real_sc = match_memory_to_queries(m.get("text", ""), recall_stats)

            if real_rc > 0:
                # 有真实搜索命中 → 用真实数据
                rc = real_rc
                sc = real_sc
            else:
                # No real recall hit — don't fabricate signal from cluster size
                rc = 0
                sc = 0

            s = score_importance(m, recall_count=rc, session_count=sc)
            scored.append((m, s))
        scored.sort(key=lambda x: x[1], reverse=True)

        # 矛盾检测 (两阶段: 关键词预筛 + LLM 验证)
        # Phase 1: 关键词预筛 — 只匹配中文高置信矛盾（收紧英文模式）
        # P2-3: 加主题重叠过滤 — 两条记忆必须共享关键实体才视为矛盾候选
        CONTRADICTION_MARKERS = [
            ("并非", "而是"),
            ("不再", "改为"),
            ("已从", "变为"),
            ("已从", "迁到"),
            ("不再是", "现在是"),
        ]

        def _extract_key_nouns(text: str) -> set[str]:
            """提取文本中的关键名词/实体（简易版）"""
            # 英文: 大写开头的词 + 全大写的缩写
            en_nouns = set(re.findall(r"\b[A-Z][a-z]{2,}\b|\b[A-Z]{2,}\b", text))
            # 中文: 提取2-4字的中文词组（粗粒度实体）
            cn_nouns = set(re.findall(r"[\u4e00-\u9fff]{2,4}", text))
            # 过滤常见停用词
            stop = {
                "The",
                "This",
                "That",
                "What",
                "How",
                "When",
                "Where",
                "Which",
                "可以",
                "但是",
                "因为",
                "所以",
                "如果",
                "已经",
                "不是",
                "而是",
                "通过",
                "使用",
                "进行",
                "需要",
                "目前",
                "现在",
                "之前",
                "之后",
            }
            return (en_nouns | cn_nouns) - stop

        def _has_subject_overlap(text1: str, text2: str, min_overlap: int = 1) -> bool:
            """检查两条记忆是否共享关键实体（主题重叠）"""
            nouns1 = _extract_key_nouns(text1)
            nouns2 = _extract_key_nouns(text2)
            overlap = nouns1 & nouns2
            return len(overlap) >= min_overlap

        if len(group) >= 2:
            texts = [m["text"] for m, _ in scored]
            texts_lower = [t.lower() for t in texts]
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    t1, t2 = texts_lower[i], texts_lower[j]
                    matched_marker = None
                    for marker_pair in CONTRADICTION_MARKERS:
                        if isinstance(marker_pair, tuple) and len(marker_pair) == 2:
                            if marker_pair[0] in t1 and marker_pair[1] in t2:
                                matched_marker = f"{marker_pair[0]} vs {marker_pair[1]}"
                                break
                            elif marker_pair[1] in t1 and marker_pair[0] in t2:
                                matched_marker = f"{marker_pair[1]} vs {marker_pair[0]}"
                                break
                    # P2-3: 只有主题重叠的对才标记矛盾
                    if matched_marker and _has_subject_overlap(texts[i], texts[j]):
                        results["contradictions"].append(
                            {
                                "mem1": scored[i][0],
                                "mem2": scored[j][0],
                                "marker": matched_marker,
                                "score_diff": abs(scored[i][1] - scored[j][1]),
                                "verified": False,  # 待 LLM 验证
                            }
                        )

        if len(group) == 1:
            m, s = scored[0]
            # P10: 年龄驱动衰减旁路 — >90天且非高重要性 → 强制衰减
            try:
                created = m.get("created_at", "")
                if created:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
                    if age_days > ARCHIVE_THRESHOLD_DAYS and s < 0.50:
                        results["decay_candidates"].append(
                            {
                                "memory": m,
                                "score": s,
                                "reason": f"age_based({age_days:.0f}d>{ARCHIVE_THRESHOLD_DAYS}d,s={s:.2f})",
                            }
                        )
                        continue
            except Exception:
                pass
            if s > 0.7:
                results["boosted"].append(
                    {"memory": m, "score": s, "reason": "singleton_high_importance"}
                )
            elif s < 0.25:
                results["decay_candidates"].append(
                    {"memory": m, "score": s, "reason": "singleton_low_importance"}
                )
            continue

        # 多条组: 向量相似度(优先) + 文本相似度(fallback) 判定
        best, best_score = scored[0]

        # P4: 多条组也产生boost和decay (原来只给singleton产生)
        for m, s in scored:
            if s >= 0.7:
                results["boosted"].append(
                    {"memory": m, "score": s, "reason": "high_importance_in_cluster"}
                )
            elif s < 0.25:
                results["decay_candidates"].append(
                    {"memory": m, "score": s, "reason": "low_importance_in_cluster"}
                )

        # P2 批量向量去重
        group_ids = [m["id"] for m, _ in scored]
        vec_dedup_cache: dict[str, dict] = {}
        if len(group_ids) >= 2:
            vec_graph = batch_vector_clustering(group_ids, max_dist=DEDUP_DIST)
            # 构建 id→neighbors 映射 (含距离)
            for mid in group_ids:
                vec_dedup_cache[mid] = {}
            # 改用: 一次批量查询 best 对所有 group 成员的向量距离
            best_id = best["id"]
            batch_sql = f"""
                SELECT b.id::text,
                       ROUND((a.vector <=> b.vector)::numeric, 4) as dist
                FROM mem0 a, mem0 b
                WHERE a.id::text = '{best_id}'
                AND b.id::text IN ('{"','".join(group_ids)}')
                AND a.id != b.id
                AND (a.vector <=> b.vector) < 0.20
            """
            batch_rows = pg_query(batch_sql)
            for r in batch_rows:
                if len(r) >= 2:
                    dist_val = safe_float(r[1])
                    if dist_val is not None:
                        vec_dedup_cache[r[0]]["vec_dist_to_best"] = dist_val

        for m, s in scored[1:]:
            # P2: 优先用向量距离判断 (pgvector, 精度最高)
            vec_dist = vec_dedup_cache.get(m["id"], {}).get("vec_dist_to_best")

            if vec_dist is not None:
                # 余弦距离 → 相似度: sim = 1 - dist
                vec_sim = 1.0 - vec_dist
                if vec_dist < DEDUP_DIST:  # dist<0.10 → sim>0.90 → 精确重复
                    results["dedup_candidates"].append(
                        {
                            "keep": best,
                            "remove": m,
                            "similarity": vec_sim,
                            "distance": vec_dist,
                            "method": "vector",
                            "keep_score": best_score,
                            "remove_score": s,
                        }
                    )
                    continue
                elif vec_dist < MERGE_DIST:  # dist<0.18 → sim>0.82 → 可合并
                    results["merge_candidates"].append(
                        {
                            "primary": best,
                            "secondary": m,
                            "similarity": vec_sim,
                            "distance": vec_dist,
                            "method": "vector",
                            "primary_score": best_score,
                            "secondary_score": s,
                        }
                    )
                    continue

            # Fallback: 文本相似度 (ngram+jaccard)
            sim_val = combined_similarity(best["text"], m["text"])

            if sim_val >= DEDUP_THRESHOLD:
                results["dedup_candidates"].append(
                    {
                        "keep": best,
                        "remove": m,
                        "similarity": sim_val,
                        "method": "ngram",
                        "keep_score": best_score,
                        "remove_score": s,
                    }
                )
            elif sim_val >= MERGE_THRESHOLD:
                results["merge_candidates"].append(
                    {
                        "primary": best,
                        "secondary": m,
                        "similarity": sim_val,
                        "method": "ngram",
                        "primary_score": best_score,
                        "secondary_score": s,
                    }
                )

        # 整组重要性高 → Vault 候选
        # Three-gate check: score + recalls + cross-session
        passes_all = (
            best_score >= PROMOTION_MIN_SCORE
            and len(group) >= PROMOTION_MIN_RECALLS
            and len(set(mem.get("created_at", "")[:10] for mem, _ in scored))
            >= PROMOTION_MIN_SESSIONS
        )

        # Use LLM entity extraction (preferred) + rule fallback
        all_texts = [mem["text"] for mem, _ in scored]
        top_keywords = extract_entities_with_fallback(all_texts, max_entities=5)

        # Time-aware sample selection: prefer freshest memory text
        freshest = min(
            scored,
            key=lambda ms: _compute_memory_age_days(ms[0].get("created_at")) or 9999,
        )
        sample_text = freshest[0]["text"][:200]
        sample_age = _compute_memory_age_days(freshest[0].get("created_at"))

        results["vault_candidates"].append(
            {
                "cluster": cluster_key,
                "memories": [mem["id"] for mem, _ in scored],
                "best_score": best_score,
                "recall_count": len(group),
                "session_count": len(
                    set(mem.get("created_at", "")[:10] for mem, _ in scored)
                ),
                "keywords": top_keywords,
                "sample_text": sample_text,
                "sample_age_days": sample_age,
                "promotion_pass": "all_3_gates" if passes_all else "below_gates",
                "priority": "high" if passes_all else "normal",
            }
        )

    # ── Phase 2: LLM 验证预筛矛盾 (限10个，避免 API 过载) ──
    unverified = [c for c in results["contradictions"] if not c.get("verified", False)]
    if unverified:
        to_verify = unverified[:10]  # 最多验证10个
        verified_count = 0
        false_positive_count = 0
        api_fail_count = 0
        for c in to_verify:
            v = llm_verify_contradiction(
                c["mem1"]["text"], c["mem2"]["text"], c["marker"]
            )
            if v is None:
                api_fail_count += 1
                c["verified"] = "api_failed"
            elif v.get("type") == "FALSE_POSITIVE" or not v.get(
                "is_contradiction", False
            ):
                c["verified"] = "false_positive"
                c["llm_explanation"] = v.get("explanation", "")
                false_positive_count += 1
            else:
                c["verified"] = True
                c["contradiction_type"] = v.get("type", "SUPERSEDE")
                c["llm_explanation"] = v.get("explanation", "")
                verified_count += 1

        # 过滤掉误报，只保留确认的真矛盾和 API 失败的
        results["contradictions"] = [
            c
            for c in results["contradictions"]
            if c.get("verified") is True or c.get("verified") == "api_failed"
        ]
        log.info(
            f"  🔍 LLM 矛盾验证: {verified_count} 确认, "
            f"{false_positive_count} 误报, {api_fail_count} API失败"
        )

    log.info(
        f"  ✅ REM 完成: {len(results['boosted'])} boost, "
        f"{len(results['dedup_candidates'])} dedup, "
        f"{len(results['merge_candidates'])} merge, "
        f"{len(results['vault_candidates'])} vault, "
        f"{len(results['decay_candidates'])} decay, "
        f"{len(results['contradictions'])} contradictions"
    )

    # P2-2: 跨聚类实体共现 → 关系推断
    # 扫描所有 cluster 的实体，找出在 ≥2 个 cluster 共现的实体对
    cross_cluster_relations = []
    entity_clusters: dict[str, set[str]] = defaultdict(set)
    cluster_entities_cache: dict[
        str, list[str]
    ] = {}  # P11: cache for orchestrator reuse
    for cluster_key, group in clusters.items():
        cluster_texts = [m["text"][:200] for m in group]
        entities = extract_entities_with_fallback(cluster_texts, max_entities=5)
        valid_entities = [ent for ent in entities if _is_valid_entity(ent)]
        cluster_entities_cache[cluster_key] = (
            valid_entities  # P11: store for dream walk
        )
        for ent in valid_entities:
            entity_clusters[ent].add(cluster_key)

    entity_list = list(entity_clusters.keys())
    co_pairs = []
    for i in range(len(entity_list)):
        for j in range(i + 1, len(entity_list)):
            e1, e2 = entity_list[i], entity_list[j]
            shared = entity_clusters[e1] & entity_clusters[e2]
            if len(shared) >= 1:  # P3: 降低阈值，1个共享cluster即可
                co_pairs.append((e1, e2, len(shared)))

    co_pairs.sort(key=lambda x: x[2], reverse=True)
    max_rels = 50  # P3: 增加到50条
    for e1, e2, count in co_pairs[:max_rels]:
        conf = min(0.7, 0.3 + count * 0.1)
        cross_cluster_relations.append(
            {
                "source": e1,
                "target": e2,
                "type": "RELATED_TO",
                "confidence": conf,
            }
        )

    results["cross_cluster_relations"] = cross_cluster_relations
    results["cluster_entities"] = cluster_entities_cache  # P11: expose for dream walk
    if cross_cluster_relations:
        log.info(
            f"  🔗 P2-2 跨聚类共现: {len(co_pairs)} 对, 取 top {len(cross_cluster_relations)}"
        )

    return results


# ─── Stage 3: Deep Sleep (深睡) — 整合行动 ────────────────────────────
