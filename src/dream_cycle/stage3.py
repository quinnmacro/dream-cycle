"""
Dream Cycle — Stage 3: Deep Sleep — dedup execution, decay archival, relation inference, Neo4j write
"""



__all__ = [
    "stage3_deep_sleep",
    "detect_slot_conflicts",
    "resolve_slot_conflicts",
]

import json
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from dream_cycle.config import (
    DREAM_DB, HKT, ARCHIVE_THRESHOLD_DAYS, PERMANENT_MARKERS, VAULT_DIR, log,
)
from dream_cycle.db import (
    pg_query, update_manifest, mark_manifest_archived,
    delete_memory, update_memory_text,
    dedup_neo4j_relations, write_relations_to_neo4j,
)
from dream_cycle.similarity import combined_similarity, get_vector_neighbors
from dream_cycle.dream_engine import rem_shy_downscale, rem_threat_simulation
from dream_cycle.llm import llm_merge_memories, llm_verify_contradiction
from dream_cycle.entities import _is_valid_entity, extract_entities_with_fallback, _KEYWORD_DOMAIN_BOOST
from dream_cycle.vault import create_vault_stub

def stage3_deep_sleep(rem_results: dict, dream_run_id: int, dry_run: bool = False, total_memories: int = 0, total_clusters: int = 0) -> dict:
    """
    Deep Sleep: 执行整合行动
    
    1. 去重: 删除重复记忆
    2. 合并: 合并近似记忆
    3. 关系推断: 识别实体间新关系 → 写入 Neo4j
    4. 衰减清理: 标记低价值记忆
    5. Vault 建议: 生成沉淀建议
    """
    log.info(f"🌊 Stage 3: Deep Sleep — 执行整合{' (dry-run)' if dry_run else ''}")
    
    stats = {
        "deduped": 0,
        "merged": 0,
        "inferred": 0,
        "decayed": 0,
        "vault_suggestions": 0,
    }
    
    conn = sqlite3.connect(str(DREAM_DB))
    
    # P4: 0. Boost — 高重要性记忆标记 (原来只产生不执行)
    # P5: 同时标注 freshness=fresh (和 mem0 plugin Ebbinghaus 联动)
    for item in rem_results.get("boosted", []):
        m = item["memory"]
        score = item["score"]
        if not dry_run:
            # 写入 PG payload: dream_boost + freshness=fresh (P5 联动)
            reason = item.get('reason', '').replace('"', '').replace("'", '')
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{"dream_boost": true, "boost_score": {score:.3f}, "boost_reason": "{reason}", "boosted_at": "{datetime.now(HKT).isoformat()}", "freshness": "fresh", "freshness_source": "dream_cycle_boost"}}' WHERE id::text = '{m['id']}'""")
        log.info(f"  🔥 Boost: {m['id'][:8]} score={score:.3f} ({item.get('reason', '')})")
    stats["boosted"] = len(rem_results.get("boosted", []))
    
    # 1. 去重 → 归档 (永不删除, Auto-Dream 模式)
    for item in rem_results.get("dedup_candidates", []):
        remove_id = item["remove"]["id"]
        keep_id = item["keep"]["id"]
        sim = item["similarity"]
        
        # 优先标记检测 — 带标记的永不归档
        remove_text = item["remove"].get("text", "")
        is_permanent = any(marker in remove_text for marker in PERMANENT_MARKERS)
        if is_permanent:
            log.info(f"  📌 优先标记保护, 跳过归档: {remove_id[:8]}")
            continue
        
        log.info(f"  🗑️ 去重→归档: remove={remove_id[:8]} (sim={sim:.3f}, keep={keep_id[:8]})")
        
        if not dry_run:
            # 归档而非删除: 写入归档标记到 payload
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{\"archived\": true, \"archived_reason\": \"dedup\", \"archived_at\": \"{datetime.now(HKT).isoformat()}\", \"superseded_by\": \"{keep_id}\"}}' WHERE id::text = '{remove_id}'""")
            stats["deduped"] += 1
            conn.execute(
                "INSERT INTO dedup_log (dream_run_id, kept_id, removed_id, similarity) VALUES (?, ?, ?, ?)",
                (dream_run_id, keep_id, remove_id, sim)
            )
        else:
            stats["deduped"] += 1
    
    # 2. 合并 (用 LLM 生成摘要, 删除被合并的)
    merged_texts_log = []
    for item in rem_results.get("merge_candidates", []):
        primary = item["primary"]
        secondary = item["secondary"]
        
        log.info(f"  🔄 合并候选: {primary['id'][:8]} ← {secondary['id'][:8]} "
                 f"(dist={item.get('distance', 'N/A')}, method={item.get('method', 'N/A')})")
        
        if not dry_run:
            merged_text = llm_merge_memories([primary["text"], secondary["text"]])
            if merged_text:
                # 更新 primary 的文本
                if update_memory_text(primary["id"], merged_text):
                    # 删除 secondary
                    if delete_memory(secondary["id"]):
                        stats["merged"] += 1
                        conn.execute(
                            "INSERT INTO dedup_log (dream_run_id, kept_id, removed_id, similarity, merged_text) VALUES (?, ?, ?, ?, ?)",
                            (dream_run_id, primary["id"], secondary["id"],
                             item.get("similarity", 0), merged_text[:500])
                        )
                        merged_texts_log.append(f"{primary['id'][:8]}←{secondary['id'][:8]}: {merged_text[:80]}")
            else:
                log.info(f"    ⏭️ LLM 合并失败, 保留两条")
    
    # 3. 关系推断 + Neo4j 回写
    neo4j_relations = []
    
    # P2-2: 跨聚类实体共现 → 关系推断 (大幅增加关系产出)
    # 从 rem_results 获取预计算的跨聚类关系
    for rel in rem_results.get("cross_cluster_relations", []):
        conn.execute(
            "INSERT INTO relation_log (dream_run_id, source_entity, target_entity, relation_type, confidence, method) VALUES (?, ?, ?, ?, ?, ?)",
            (dream_run_id, rel["source"], rel["target"], rel["type"], rel["confidence"], "cross_cluster_cooccurrence")
        )
        stats["inferred"] += 1
        neo4j_relations.append(rel)
    if rem_results.get("cross_cluster_relations"):
        log.info(f"  🔗 跨聚类共现: {len(rem_results['cross_cluster_relations'])} 条新关系")
    
    # 原有: vault_candidates 关键词关系
    for item in rem_results.get("vault_candidates", []):
        keywords = item.get("keywords", [])
        # 过滤无效实体
        valid_keywords = [k for k in keywords if _is_valid_entity(k)]
        if len(valid_keywords) < 2:
            continue  # 至少需要两个有效实体才能建关系
        # 两两配对推断关系
        for i in range(len(valid_keywords)):
            for j in range(i + 1, min(i + 3, len(valid_keywords))):
                rel_type = "RELATED_TO"  # 通用关系
                conn.execute(
                    "INSERT INTO relation_log (dream_run_id, source_entity, target_entity, relation_type, confidence, method) VALUES (?, ?, ?, ?, ?, ?)",
                    (dream_run_id, valid_keywords[i], valid_keywords[j], rel_type, 0.4, "dream_keyword_cooccurrence")
                )
                stats["inferred"] += 1
                neo4j_relations.append({
                    "source": valid_keywords[i], "target": valid_keywords[j],
                    "type": rel_type, "confidence": 0.4
                })
    
    # P2: 关系去重 — 同源同目标的已有关系不重复写入
    if neo4j_relations and not dry_run:
        # 去重: 查 Neo4j 已有关系，过滤重复
        deduped_relations = dedup_neo4j_relations(neo4j_relations)
        neo4j_written = write_relations_to_neo4j(deduped_relations)
        log.info(f"  🔗 Neo4j 回写: {neo4j_written}/{len(deduped_relations)} 关系 "
                 f"(去重 {len(neo4j_relations)-len(deduped_relations)} 条)")
    
    # 3b. SHY Downscaling (Synaptic Homeostasis — from claude-brain)
    # After writing new edges, downscale weak edges globally to prevent unbounded growth
    if not dry_run:
        shy_stats = rem_shy_downscale()
        if shy_stats["total"] > 0:
            log.info(f"  🧬 SHY: {shy_stats['protected']} protected, "
                     f"{shy_stats['downscaled']} downscaled, {shy_stats['pruned']} pruned "
                     f"(of {shy_stats['total']} total edges)")
    
    # 3c. Threat Simulation (from claude-brain)
    # Scan for contradiction edges between high-confidence nodes
    threats = rem_threat_simulation()
    if threats:
        log.info(f"  ⚠️ Threat: {len(threats)} contradiction edges detected")
        for t in threats[:3]:
            log.info(f"    {t['node']} ↔ {t['contradicts']} (severity={t['severity']})")
    
    # 4. 衰减候选 → 归档 (永不删除)
    # P5: 同时更新 PG payload 的 freshness 字段 (和 mem0 plugin Ebbinghaus 联动)
    for item in rem_results.get("decay_candidates", []):
        m = item["memory"]
        score = item["score"]
        # P5: 根据 score 判断 freshness 标签 (和 mem0 plugin 对齐)
        if score < 0.15:
            freshness = "outdated"
        elif score < 0.25:
            freshness = "stale"
        else:
            freshness = "aging"
        log.info(f"  📉 衰减→归档候选: {m['id'][:8]} (score={score:.3f}, freshness={freshness})")
        if not dry_run:
            # 归档: 写入归档标记到 payload + P5 freshness 联动
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{"archived": true, "archived_reason": "decay", "archived_at": "{datetime.now(HKT).isoformat()}", "decay_score": {score:.3f}, "freshness": "{freshness}", "freshness_source": "dream_cycle_decay"}}' WHERE id::text = '{m['id']}'""")
            mark_manifest_archived([m["id"]])
        stats["decayed"] += 1
    
# 4. 矛盾报告 + P7 自动处理
    contradictions = rem_results.get("contradictions", [])
    if contradictions:
        log.info(f"  ⚡ 矛盾检测: 发现 {len(contradictions)} 对矛盾")
        for c in contradictions[:5]:
            log.info(f"    {c['marker']}: {c['mem1']['id'][:8]} vs {c['mem2']['id'][:8]}")
        
        # P7: 语义签名冲突自动处理 (slot_conflict 类型)
        slot_conflicts = rem_results.get("slot_conflicts_list", [])
        if slot_conflicts and not dry_run:
            processed_conflicts = resolve_slot_conflicts(slot_conflicts, dream_run_id)
            stats["conflicts_resolved"] = len(processed_conflicts)
    
    # 6. 健康评分 (来自 Auto-Dream 5维)
    archived_count = sum(1 for _ in rem_results.get("decay_candidates", []))
    vault_count = len(rem_results.get("vault_candidates", []))
    contradiction_count = len(contradictions)
    
    health = {
        "freshness": min(1.0, (total_memories - archived_count) / max(total_memories, 1)),
        "coverage": min(1.0, vault_count / max(total_clusters, 1)),
        "coherence": min(1.0, 1.0 - contradiction_count / max(total_memories, 1)),
        "efficiency": min(1.0, 1.0 - stats.get("deduped", 0) / max(total_memories, 1)),
        "reachability": min(1.0, stats.get("inferred", 0) / max(total_memories * 0.5, 1)),
    }
    health_score = sum(health[k] * w for k, w in [
        ("freshness", 0.25), ("coverage", 0.25), ("coherence", 0.20),
        ("efficiency", 0.15), ("reachability", 0.15)
    ]) * 100
    
    log.info(f"  💊 健康评分: {health_score:.0f}/100 "
             f"(fresh={health['freshness']:.0%} cov={health['coverage']:.0%} "
             f"coh={health['coherence']:.0%} eff={health['efficiency']:.0%} "
             f"reach={health['reachability']:.0%})")
    
    stats["health_score"] = round(health_score, 1)
    stats["contradictions"] = contradiction_count
    
    # 6. Vault 建议 + 自动沉淀
    for item in rem_results.get("vault_candidates", []):
        # 使用高质量关键词 — 只取有效实体
        keywords = item.get("keywords", [])
        valid_keywords = [k for k in keywords if _is_valid_entity(k)]
        if not valid_keywords:
            continue  # 无有效关键词, 跳过
        
        # 推断 category — 基于关键词的领域加权
        sample = item.get("sample_text", "")
        sample_age = item.get("sample_age_days")
        keywords_lower = [k.lower() for k in valid_keywords]
        investment_kw = {'bonds', 'yield', 'spread', 'cgb', 'ust', 'carry', 'duration',
                        'credit', 'curve', 'swap', 'basis', 'delivery', 'bond', 'rate',
                        'inflation', 'fed', 'ecb', 'boj', 'macro', 'fiscal', 'monetary',
                        'hedge', 'position', 'flow', 'premium', 'sovereign', 'cme', 'comex'}
        tech_kw = {'docker', 'mcp', 'plugin', 'mem0', 'neo4j', 'config', 'deploy', 'cron', 'hermes'}
        
        if any(k in investment_kw for k in keywords_lower):
            category = "markets"
        elif any(k in tech_kw for k in keywords_lower):
            category = "projects"
        else:
            category = "concepts"
        
        # Vault 建议实体名筛选: 优先选有领域加权的词
        domain_keywords = [k for k in valid_keywords if k.lower() in _KEYWORD_DOMAIN_BOOST]
        entity_name = domain_keywords[0] if domain_keywords else valid_keywords[0]
        
        # 如果唯一的实体名太通用 (不在 domain boost 且不是复合词/缩写), 跳过
        if (entity_name.lower() not in _KEYWORD_DOMAIN_BOOST 
            and '-' not in entity_name 
            and not entity_name.isupper()
            and len(entity_name) < 6):
            continue
        
        conn.execute(
            "INSERT INTO vault_suggestion (dream_run_id, entity, category, frequency, reason) VALUES (?, ?, ?, ?, ?)",
            (dream_run_id, entity_name, category,
             len(item.get("memories", [])), f"score={item.get('best_score', 0):.2f}")
        )
        stats["vault_suggestions"] += 1
        
        # 自动沉淀: 创建 Vault 页面骨架 (P3: 门槛从3条降到2条)
        if not dry_run and len(item.get("memories", [])) >= 2:
            vault_path = create_vault_stub(
                entity=entity_name,
                category=category,
                keywords=valid_keywords,
                sample=sample,
                sample_age_days=sample_age,
            )
            if vault_path:
                stats["vault_created"] = stats.get("vault_created", 0) + 1
                # P3: 自动沉淀后标记状态
                conn.execute(
                    "UPDATE vault_suggestion SET status = 'auto_created' WHERE entity = ? AND status = 'pending'",
                    (entity_name,)
                )
    
    conn.commit()
    conn.close()
    
    log.info(f"  ✅ Deep Sleep 完成: deduped={stats['deduped']}, merged={stats['merged']}, "
             f"inferred={stats['inferred']}, decay={stats['decayed']}, vault={stats['vault_suggestions']}")
    
    return stats


# ─── 在线去冗余 (写入时) ──────────────────────────────────────────────

# ─── Conflict detection & resolution (moved from orchestrator to break circular dep) ───

def detect_slot_conflicts() -> list[dict]:
    """
    语义签名冲突检测: 用 HNSW 索引逐条查找最近邻, 避免全表 cross-join
    
    Returns: [{"mem1": ..., "mem2": ..., "slot_similarity": ..., "value_diff": ...}]
    """
    conflicts = []
    
    # 1. 取最近 7 天有文本的记忆 ID (最多 200 条)
    recent_ids = pg_query("""
        SELECT id::text
        FROM mem0
        WHERE payload->>'archived' IS NULL
        AND LENGTH(payload->>'data') > 30
        AND payload->>'created_at' IS NOT NULL
        AND payload->>'created_at' >= (NOW() - INTERVAL '7 days')::text
        ORDER BY id
        LIMIT 200
    """)
    
    if not recent_ids:
        log.info("  ✅ 无近期记忆, 跳过冲突检测")
        return []
    
    # 2. 对每条记忆, 用 HNSW 索引查最近邻 (高效)
    seen_pairs = set()
    for row in recent_ids:
        mem_id = row[0] if isinstance(row, list) else row
        neighbors = get_vector_neighbors(mem_id, limit=5, max_dist=0.15)
        for n in neighbors:
            nid = n["id"]
            pair_key = tuple(sorted([mem_id, nid]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            # 获取两条记忆的文本
            text_rows = pg_query(f"""
                SELECT a.id::text, LEFT(a.payload->>'data', 300) as t1,
                       b.id::text, LEFT(b.payload->>'data', 300) as t2
                FROM mem0 a, mem0 b
                WHERE a.id::text = '{mem_id}' AND b.id::text = '{nid}'
            """)
            
            if not text_rows or len(text_rows[0]) < 4:
                continue
            r = text_rows[0]
            text1, text2 = r[1], r[3]
            if not text1 or not text2:
                continue
            
            text_sim = combined_similarity(text1, text2)
            vec_dist = n["distance"]
            
            if text_sim < 0.5:  # 语义近但文本差异大 = 槽位冲突
                conflicts.append({
                    "mem1_id": mem_id,
                    "mem1_text": text1,
                    "mem2_id": nid,
                    "mem2_text": text2,
                    "slot_similarity": 1.0 - vec_dist,
                    "value_diff": 1.0 - text_sim,
                    "type": "slot_conflict",
                })
                
                if len(conflicts) >= 10:
                    break
        
        if len(conflicts) >= 10:
            break
    
    if conflicts:
        log.info(f"  🔍 语义签名冲突: {len(conflicts)} 对'同槽不同值'")
        for c in conflicts[:3]:
            log.info(f"    [{c['slot_similarity']:.2f}槽似, {c['value_diff']:.2f}值差] "
                     f"{c['mem1_text'][:60]}... vs {c['mem2_text'][:60]}...")
    else:
        log.info("  ✅ 无语义签名冲突")
    
    return conflicts

def resolve_slot_conflicts(conflicts: list[dict], dream_run_id: int, max_resolve: int = 5) -> list[dict]:
    """
    P7: 语义签名冲突自动处理
    
    对高值差(>0.5)的 slot_conflict:
    1. LLM 判断类型: SUPERSEDE / EXTEND / FALSE_POSITIVE
    2. SUPERSEDE → 旧记忆归档，保留新记忆
    3. EXTEND → 两条都保留，标记 extended
    4. FALSE_POSITIVE → 标记忽略
    
    成本: ~200 tokens/conflict, 最多处理5个
    """
    resolved = []
    
    # 只处理高值差的冲突 (值差>0.5 说明真的不同)
    high_diff = [c for c in conflicts if c.get("value_diff", 0) > 0.5]
    if not high_diff:
        log.info("  🔍 P7: 无高值差冲突需要处理")
        return resolved
    
    log.info(f"  🔍 P7: {len(high_diff)} 个高值差冲突待处理 (限{max_resolve})")
    
    for c in high_diff[:max_resolve]:
        mem1_id = c.get("mem1_id", "")
        mem2_id = c.get("mem2_id", "")
        mem1_text = c.get("mem1_text", "")
        mem2_text = c.get("mem2_text", "")
        
        # LLM 判断
        v = llm_verify_contradiction(mem1_text, mem2_text, f"slot_conflict(sim={c.get('slot_similarity', 0):.2f},diff={c.get('value_diff', 0):.2f})")
        
        if v is None:
            log.info(f"    ⏭️ API失败, 跳过 {mem1_id[:8]} vs {mem2_id[:8]}")
            continue
        
        ctype = v.get("type", "FALSE_POSITIVE")
        explanation = v.get("explanation", "")
        
        if ctype == "SUPERSEDE":
            # 新事实取代旧事实 → 归档旧记忆 (按创建时间判断)
            # 获取两条记忆的创建时间
            rows1 = pg_query(f"SELECT id::text, payload->>'created_at' FROM mem0 WHERE id::text = '{mem1_id}'")
            rows2 = pg_query(f"SELECT id::text, payload->>'created_at' FROM mem0 WHERE id::text = '{mem2_id}'")
            
            older_id = mem1_id  # 默认归档第一条
            newer_id = mem2_id
            if rows1 and rows2:
                # 比较创建时间
                t1 = rows1[0][1] if len(rows1[0]) > 1 else ""
                t2 = rows2[0][1] if len(rows2[0]) > 1 else ""
                if t2 < t1:  # mem2更早
                    older_id, newer_id = mem2_id, mem1_id
            
            log.info(f"    ✅ SUPERSEDE: 归档 {older_id[:8]}, 保留 {newer_id[:8]} ({explanation[:60]})")
            # 归档旧记忆
            safe_explanation = explanation[:100].replace('"', '').replace("'", "''")
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{"archived": true, "archived_reason": "slot_supersede", "superseded_by": "{newer_id}", "supersede_explanation": "{safe_explanation}", "freshness": "outdated", "freshness_source": "dream_cycle_supersede"}}' WHERE id::text = '{older_id}'""")
            mark_manifest_archived([older_id])
            resolved.append({"type": "SUPERSEDE", "older": older_id, "newer": newer_id, "explanation": explanation})
        
        elif ctype == "EXTEND":
            # 新事实扩展旧事实 → 两条都保留，标记 extended
            log.info(f"    🔗 EXTEND: 两条都保留 ({explanation[:60]})")
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{"extended": true, "extended_by": "{mem2_id}", "extension_type": "{ctype}"}}' WHERE id::text = '{mem1_id}'""")
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{"extended": true, "extends": "{mem1_id}", "extension_type": "{ctype}"}}' WHERE id::text = '{mem2_id}'""")
            resolved.append({"type": "EXTEND", "mem1": mem1_id, "mem2": mem2_id, "explanation": explanation})
        
        else:  # FALSE_POSITIVE
            log.info(f"    ⏭️ FALSE_POSITIVE: 不处理 ({explanation[:60]})")
            resolved.append({"type": "FALSE_POSITIVE", "explanation": explanation})
    
    # 记录到 contradiction_log
    conn_cl = sqlite3.connect(str(DREAM_DB))
    for r in resolved:
        ctype = r["type"]
        ids = [r.get("older", r.get("mem1", "")), r.get("newer", r.get("mem2", ""))]
        conn_cl.execute(
            "UPDATE contradiction_log SET contradiction_type = ?, resolution = ?, llm_explanation = ?, verified = 1 WHERE mem1_id = ? OR mem2_id = ?",
            (ctype, ctype, r.get("explanation", ""), ids[0], ids[1])
        )
    conn_cl.commit()
    conn_cl.close()
    
    log.info(f"  ✅ P7 冲突处理: {len(resolved)} 条 (SUPERSEDE={sum(1 for r in resolved if r['type']=='SUPERSEDE')}, "
             f"EXTEND={sum(1 for r in resolved if r['type']=='EXTEND')}, "
             f"FALSE_POSITIVE={sum(1 for r in resolved if r['type']=='FALSE_POSITIVE')})")
    
    return resolved


# ─── 报告生成 ──────────────────────────────────────────────────────────
