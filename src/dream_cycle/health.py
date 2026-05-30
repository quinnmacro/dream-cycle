"""
Dream Cycle — Health — adaptive trigger, online dedup check, 7-day dashboard
"""



__all__ = [
    "check_dream_trigger",
    "show_health_dashboard",
    "online_dedup_check",
]

import json
import sqlite3
import time
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dream_cycle.config import (
    DREAM_DB, STATE_DB, NEO4J_URI, NEO4J_USER, NEO4J_PASS, HKT, log,
    TRIGGER_MAX_IDLE_HOURS, TRIGGER_MIN_NEW_MEMORIES, TRIGGER_CONFLICT_DENSITY,
    TRIGGER_MEMORY_ENTROPY, DEDUP_THRESHOLD, MERGE_THRESHOLD,
)
from dream_cycle.db import pg_query, get_recent_memories, get_incremental_memories, get_all_memories_with_embeddings
from dream_cycle.config import text_hash
from dream_cycle.similarity import combined_similarity

def check_dream_trigger() -> dict:
    """
    检查是否应该触发梦循环
    
    触发条件 (满足任一):
    1. 时间间隔: 距上次梦循环 > TRIGGER_MAX_IDLE_HOURS
    2. 新数据量: 新增未处理记忆 >= TRIGGER_MIN_NEW_MEMORIES
    3. 冲突密度: contradiction_log 中 pending 的比例 > TRIGGER_CONFLICT_DENSITY
    4. 记忆熵: 重要度分布集中度过高 (SCM H > θ_e)
    
    Returns: {"should_trigger": bool, "reasons": [...], "urgency": "low"|"medium"|"high"}
    """
    reasons = []
    urgency = "low"
    
    # 1. 时间间隔检查
    conn = sqlite3.connect(str(DREAM_DB))
    last_run = conn.execute(
        "SELECT finished_at FROM dream_runs WHERE error IS NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    
    if last_run and last_run[0]:
        try:
            last_dt = datetime.fromisoformat(last_run[0])
            idle_hours = (datetime.now(HKT) - last_dt).total_seconds() / 3600
            if idle_hours >= TRIGGER_MAX_IDLE_HOURS:
                reasons.append(f"idle_{idle_hours:.1f}h>={TRIGGER_MAX_IDLE_HOURS}h")
                urgency = "medium"
        except Exception:
            pass
    else:
        reasons.append("no_previous_run")
        urgency = "high"
    
    # 2. 新数据量检查
    new_memories = get_incremental_memories(hours=24)
    if len(new_memories) >= TRIGGER_MIN_NEW_MEMORIES:
        reasons.append(f"new_memories_{len(new_memories)}>={TRIGGER_MIN_NEW_MEMORIES}")
        urgency = "high" if len(new_memories) >= TRIGGER_MIN_NEW_MEMORIES * 3 else urgency
    
    # 3. 冲突密度检查 (从 contradiction_log)
    try:
        conn = sqlite3.connect(str(DREAM_DB))
        total_contra = conn.execute("SELECT COUNT(*) FROM contradiction_log").fetchone()[0]
        pending_contra = conn.execute("SELECT COUNT(*) FROM contradiction_log WHERE resolution='pending'").fetchone()[0]
        conn.close()
        if total_contra > 0:
            conflict_density = pending_contra / total_contra
            if conflict_density >= TRIGGER_CONFLICT_DENSITY:
                reasons.append(f"conflict_density_{conflict_density:.2f}>={TRIGGER_CONFLICT_DENSITY}")
                urgency = "high"
    except Exception:
        pass  # 表可能不存在 (首次运行)
    
    # 4. 记忆熵检查 (重要度分布 — 用 PG 记忆量估算)
    try:
        recent = get_recent_memories(hours=24)
        if len(recent) > 50:
            # 简化熵: 用记忆长度分布的标准差/均值作为 concentration 代理
            lengths = [len(m["text"]) for m in recent]
            mean_len = sum(lengths) / len(lengths)
            if mean_len > 0:
                var_len = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
                cv = (var_len ** 0.5) / mean_len  # coefficient of variation
                # CV 低 = 所有记忆长度相似 = 高集中度 = 高熵
                concentration = 1.0 - min(1.0, cv)
                if concentration >= TRIGGER_MEMORY_ENTROPY:
                    reasons.append(f"entropy_{concentration:.2f}>={TRIGGER_MEMORY_ENTROPY}")
                    urgency = max(urgency, "medium")
    except Exception:
        pass
    
    return {
        "should_trigger": len(reasons) > 0,
        "reasons": reasons,
        "urgency": urgency,
        "new_memory_count": len(new_memories) if new_memories else 0,
    }


# ─── P1: REM 梦游 (SCM 核心 — 随机游走发现隐含关联) ──────────────────

def show_health_dashboard():
    """
    P8: 显示梦循环健康仪表盘 — 7天趋势 + 当前状态
    """
    conn = sqlite3.connect(str(DREAM_DB))
    
    # 7天趋势
    cutoff = (datetime.now(HKT) - timedelta(days=7)).isoformat()
    runs = conn.execute("""
        SELECT id, started_at, stage1_clusters, stage2_boosted, stage3_deduped,
               stage3_inferred, stage3_decayed, stage3_vault_suggestions, summary, error
        FROM dream_runs WHERE started_at > ?
        ORDER BY id ASC
    """, (cutoff,)).fetchall()
    
    # Manifest 统计
    total_manifest = conn.execute("SELECT COUNT(*) FROM processed_manifest").fetchone()[0]
    active_manifest = conn.execute("SELECT COUNT(*) FROM processed_manifest WHERE status='active'").fetchone()[0]
    archived_manifest = conn.execute("SELECT COUNT(*) FROM processed_manifest WHERE status='archived'").fetchone()[0]
    
    # Relation 统计
    total_relations = conn.execute("SELECT COUNT(*) FROM relation_log").fetchone()[0]
    useful_relations = conn.execute("SELECT COUNT(*) FROM relation_log WHERE confidence >= 0.5").fetchone()[0]
    boosted_relations = conn.execute("SELECT COUNT(*) FROM relation_log WHERE confidence >= 0.6").fetchone()[0]
    cross_cluster_rels = conn.execute("SELECT COUNT(*) FROM relation_log WHERE method = 'cross_cluster_cooccurrence'").fetchone()[0]
    
    # Neo4j 实际关系数 (查询 Playground)
    neo4j_total = 0
    neo4j_dream = 0
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            neo4j_total = session.run("MATCH ()-[r]->() RETURN count(r)").single()[0]
            neo4j_dream = session.run("MATCH ()-[r]->() WHERE r.source = 'dream_cycle' RETURN count(r)").single()[0]
        driver.close()
    except Exception:
        pass
    
    # Vault suggestion 统计
    total_suggestions = conn.execute("SELECT COUNT(*) FROM vault_suggestion").fetchone()[0]
    pending_suggestions = conn.execute("SELECT COUNT(*) FROM vault_suggestion WHERE status='pending'").fetchone()[0]
    auto_created = conn.execute("SELECT COUNT(*) FROM vault_suggestion WHERE status='auto_created'").fetchone()[0]
    reviewed = conn.execute("SELECT COUNT(*) FROM vault_suggestion WHERE status='reviewed'").fetchone()[0]
    
    # Contradiction 统计
    total_contra = conn.execute("SELECT COUNT(*) FROM contradiction_log").fetchone()[0]
    resolved_contra = conn.execute("SELECT COUNT(*) FROM contradiction_log WHERE resolution != 'pending'").fetchone()[0]
    
    conn.close()
    
    # 记忆总量
    recent_memories = get_recent_memories(hours=168)
    all_memories = get_all_memories_with_embeddings()
    
    print("=" * 50)
    print("🏥 **Dream Cycle Health Dashboard**")
    print("=" * 50)
    
    print(f"\n📊 **记忆状态**")
    print(f"  PG 总量: {len(all_memories)} | 7天新增: {len(recent_memories)}")
    print(f"  Manifest: {active_manifest} active / {archived_manifest} archived / {total_manifest} total")
    
    print(f"\n🔗 **关系网络**")
    print(f"  总关系: {total_relations} | 高置信(>=0.5): {useful_relations} | LLM Boost(>=0.6): {boosted_relations}")
    print(f"  跨聚类: {cross_cluster_rels} | Neo4j: {neo4j_dream}/{neo4j_total}")
    
    print(f"\n📝 **Vault 沉淀**")
    print(f"  建议: {total_suggestions} | pending: {pending_suggestions} | auto_created: {auto_created} | reviewed: {reviewed}")
    
    print(f"\n⚡ **冲突检测**")
    print(f"  总冲突: {total_contra} | 已解决: {resolved_contra} | pending: {total_contra - resolved_contra}")
    
    print(f"\n📈 **7天趋势** ({len(runs)} runs)")
    if runs:
        # Sparkline-style 趋势
        clusters_trend = [str(r[2]) for r in runs]
        dedup_trend = [str(r[4]) for r in runs]     # stage3_deduped (index 4)
        inferred_trend = [str(r[5]) for r in runs]   # stage3_inferred (index 5)
        vault_trend = [str(r[8]) for r in runs]
        
        print(f"  Clusters: {' → '.join(clusters_trend)}")
        print(f"  Deduped:  {' → '.join(dedup_trend)}")
        print(f"  Inferred: {' → '.join(inferred_trend)}")
        print(f"  Vault:    {' → '.join(vault_trend)}")
        
        # 最近一次详情
        last = runs[-1]
        last_status = "✅" if not last[9] else "❌"
        print(f"\n  最近一次: #{last[0]} [{last_status}] {last[1]}")
        summary = last[8]  # summary is column 8 (index 8)
        if summary and summary != "None":
            try:
                s = json.loads(summary)
                print(f"    扫描: {s.get('memories_scanned', '?')} | "
                      f"聚类: {s.get('clusters', '?')} | "
                      f"去重候选: {s.get('dedup_candidates', '?')} | "
                      f"合并候选: {s.get('merge_candidates', '?')}")
            except Exception:
                print(f"    {summary[:100]}")
    else:
        print("  无最近7天记录")
    
    # 健康评分 (简化版)
    # coverage: Neo4j dream关系覆盖了多少记忆 (目标: 10% 记忆有dream关系)
    mem_count = max(len(all_memories), 1)
    coverage = min(1.0, neo4j_dream / max(mem_count * 0.1, 1)) if neo4j_dream > 0 else min(1.0, (useful_relations + auto_created) / max(total_manifest * 0.5, 1))
    
    # coherence: 矛盾解决率
    coherence = min(1.0, 1.0 - (total_contra - resolved_contra) / max(total_contra, 1)) if total_contra > 0 else 1.0
    
    # efficiency: 处理效率
    efficiency = min(1.0, 1.0 - max(0, total_manifest - active_manifest) / max(total_manifest, 1))
    
    # reachability: Neo4j总关系 vs 记忆量 (目标: 30% 连接有向边)
    reachability = min(1.0, neo4j_total / max(mem_count * 0.3, 1)) if neo4j_total > 0 else min(1.0, useful_relations / max(total_manifest * 0.3, 1))
    
    health = {
        "coverage": coverage,
        "coherence": coherence,
        "efficiency": efficiency,
        "reachability": reachability,
    }
    score = sum(health[k] * w for k, w in [("coverage", 0.30), ("coherence", 0.25), ("efficiency", 0.20), ("reachability", 0.25)]) * 100
    
    print(f"\n💊 **综合健康: {score:.0f}/100**")
    print(f"  cov={health['coverage']:.0%} coh={health['coherence']:.0%} eff={health['efficiency']:.0%} reach={health['reachability']:.0%}")
    print("=" * 50)


# ─── P9: Vault Suggestion Review ───────────────────────────────────────

def online_dedup_check(text: str, threshold: float = 0.85) -> dict:
    """
    写入前去冗余检查 — 不等 04:00 梦循环，写入时立刻检查
    
    被 mem0 plugin 的 mem0_conclude 调用:
    1. 计算 text hash → 精确去重
    2. pgvector 查最近邻 → 语义去重
    3. 返回建议: SKIP(重复) / MERGE(近似) / ADD(新增)
    
    Args:
        text: 待写入的文本
        threshold: 语义相似度阈值 (余弦距离, 越小越相似)
    
    Returns:
        {"action": "ADD"|"SKIP"|"MERGE", "reason": "...", "similar_id": "...", "similarity": float}
    """
    # 1. 精确去重: hash 检查
    h = text_hash(text)
    hash_match = pg_query(f"""
        SELECT id::text, LEFT(payload->>'data', 200) as sample
        FROM mem0
        WHERE payload->>'hash' = '{h}'
        AND payload->>'archived' IS NULL
        LIMIT 1
    """)
    if hash_match:
        return {
            "action": "SKIP",
            "reason": f"exact_hash_match",
            "similar_id": hash_match[0][0],
            "similarity": 1.0,
        }
    
    # 2. 语义去重: pgvector 查最近邻
    # 先找最近插入的向量做 anchor
    # (需要先插入向量才能查，所以改用文本相似度快速筛选)
    # 快速文本扫描: 最近100条记忆
    recent = get_recent_memories(hours=72)  # 3天窗口
    best_sim = 0.0
    best_match = None
    
    for m in recent:
        sim = combined_similarity(text, m["text"])
        if sim > best_sim:
            best_sim = sim
            best_match = m
    
    # 余弦距离阈值换算: distance < 0.15 ≈ similarity > 0.85
    if best_sim >= DEDUP_THRESHOLD and best_match:
        return {
            "action": "SKIP",
            "reason": f"semantic_exact (sim={best_sim:.3f})",
            "similar_id": best_match["id"],
            "similarity": best_sim,
        }
    elif best_sim >= MERGE_THRESHOLD and best_match:
        return {
            "action": "MERGE",
            "reason": f"semantic_merge (sim={best_sim:.3f})",
            "similar_id": best_match["id"],
            "similarity": best_sim,
        }
    
    return {"action": "ADD", "reason": "new_unique_memory", "similar_id": None, "similarity": 0.0}


# ─── 主循环 ────────────────────────────────────────────────────────────

