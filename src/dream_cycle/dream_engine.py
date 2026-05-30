"""
Dream Cycle — Dream Engine — REM walk v2 (teleport+waking), SHY downscale, threat simulation, Hebbian
"""



__all__ = [
    "rem_dream_walk",
    "rem_shy_downscale",
    "rem_threat_simulation",
    "llm_boost_relations",
    "nrem_hebbian_consolidation",
]

import math
import random
import json
import logging
from collections import defaultdict
from dream_cycle.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASS,
    REM_WALK_LENGTH, REM_SEED_COUNT, REM_MAX_NEW_EDGES,
    REM_JUMP_PROBABILITY, WAKING_THRESHOLD,
    SHY_PROTECTION_PCT, SHY_DOWNSCALE_FACTOR, SHY_PRUNE_THRESHOLD,
    HEBBIAN_LEARNING_RATE, HEBBIAN_DOWNSCALE, log,
)
from dream_cycle.entities import extract_entities_with_fallback
from dream_cycle.llm import _call_infini
from dream_cycle.vault import _is_time_sensitive

def rem_dream_walk(cluster_entities: list[str] = None) -> list[dict]:
    """
    REM 梦游 v2: Bio-inspired dream engine (from claude-brain)
    
    Upgrades over v1:
    - Creative Teleport (30%): jump to topologically distant but label-shared nodes
    - Revisit Penalty: exp(-visit_count) refractory period
    - Waking Gate: shared neighbor check before promoting provisional edges
    - Edge confidence scoring: path length + shared neighbors → variable confidence
    
    Returns: [{"source": ..., "target": ..., "path": [...], "type": "DREAM_CONNECTION", 
               "confidence": float, "jump_points": [...]}]
    """
    import random, math
    try:
        from neo4j import GraphDatabase
    except ImportError:
        log.warning("⚠️ neo4j driver 未安装, 跳过 REM 梦游")
        return []
    
    # Noise filter shared with v1
    NOISE_NAMES = {'user', 'assistant', 'system', 'from', 'with',
        'that', 'this', 'trade', 'framework', 'data', 'system', 'project',
        'file', 'update', 'change', 'feature', 'lines', 'order', 'skill',
        'memory', 'limit', 'service', 'command', 'parameter', 'value',
        'result', 'output', 'error', 'warning', 'status', 'version',
        'method', 'function', 'code', 'line', 'step', 'process', 'task',
        'action', 'check', 'active', 'repository', 'Related'}
    
    def _revisit_penalty(visit_count: int) -> float:
        """Exponential penalty for revisited nodes (neural refractory period)."""
        if visit_count <= 0:
            return 1.0
        return math.exp(-visit_count)
    
    provisional_edges = []
    promoted_edges = []
    total_jumps = 0
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            # --- Seed selection (v1 logic preserved) ---
            cluster_seed_names = []
            if cluster_entities:
                for ent in cluster_entities[:3]:
                    r = session.run(
                        "MATCH (n:Concept {name: $name}) RETURN n.name as name, COUNT { (n)--() } as degree",
                        name=ent
                    ).data()
                    if r and r[0]["degree"] > 0:
                        cluster_seed_names.append(r[0])
            
            remaining_slots = REM_SEED_COUNT - len(cluster_seed_names)
            neo4j_seeds = []
            if remaining_slots > 0:
                exclude_names = [s["name"] for s in cluster_seed_names] if cluster_seed_names else []
                neo4j_seeds = session.run("""
                    MATCH (n:Concept)
                    WHERE n.name IS NOT NULL AND size(n.name) > 3
                    AND NOT n.name =~ '.*\\d{4}.*'
                    AND NOT n.name ENDS WITH 'ing'
                    AND NOT n.name CONTAINS ','
                    AND NOT n.name IN $noise
                    AND NOT n.name IN $exclude
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n, COUNT(r) as degree
                    ORDER BY degree DESC
                    LIMIT $seed_count
                    RETURN n.name as name, degree
                """, seed_count=remaining_slots, exclude=exclude_names, noise=list(NOISE_NAMES)).data()
            
            seeds = cluster_seed_names + neo4j_seeds
            if not seeds:
                log.info("  💭 REM: 无种子节点，跳过梦游")
                driver.close()
                return []
            
            log.info(f"  💭 REM 梦游 v2: {len(seeds)} 个种子 (jump_p={REM_JUMP_PROBABILITY})")
            
            # --- Walk each seed ---
            for seed in seeds:
                seed_name = seed["name"]
                path = [seed_name]
                visit_counts = {seed_name: 1}
                jump_points = []
                prev_node = None
                current = seed_name
                
                for step in range(REM_WALK_LENGTH):
                    do_jump = random.random() < REM_JUMP_PROBABILITY
                    
                    # Get neighbors
                    neighbors = session.run("""
                        MATCH (c:Concept {name: $name})-[r]-(n:Concept)
                        WHERE n.name IS NOT NULL AND size(n.name) > 3
                        AND NOT n.name =~ '.*\\d{4}.*'
                        AND NOT n.name IN $noise
                        AND n.name <> $prev
                        RETURN n.name as name, COALESCE(r.confidence, 0.5) as weight
                        LIMIT 8
                    """, name=current, prev=prev_node or "", noise=list(NOISE_NAMES)).data()
                    
                    if do_jump or not neighbors:
                        # --- Creative Teleport ---
                        # Jump to a node that shares labels but is NOT a direct neighbor
                        distant = session.run("""
                            MATCH (c:Concept {name: $name})
                            WITH c, labels(c) as my_labels
                            UNWIND my_labels as lbl
                            MATCH (distant:Concept)
                            WHERE lbl IN labels(distant)
                            AND distant.name <> $name
                            AND NOT exists((c)--(distant))
                            AND distant.name IS NOT NULL AND size(distant.name) > 3
                            AND NOT distant.name IN $noise
                            AND NOT distant.name IN $visited
                            OPTIONAL MATCH (distant)-[r2]-()
                            WITH distant, COUNT(r2) as degree
                            WHERE degree > 0
                            RETURN distant.name as name, degree as weight
                            ORDER BY rand()
                            LIMIT 3
                        """, name=current, noise=list(NOISE_NAMES), visited=list(visit_counts.keys())).data()
                        
                        if distant:
                            chosen = random.choice(distant)
                            next_node = chosen["name"]
                            jump_points.append(step)
                            total_jumps += 1
                        elif neighbors:
                            # Fallback to normal neighbor if no distant node found
                            weights = [n["weight"] * _revisit_penalty(visit_counts.get(n["name"], 0))
                                       for n in neighbors]
                            total_w = sum(weights)
                            if total_w <= 0:
                                chosen = random.choice(neighbors)
                            else:
                                r_val = random.random() * total_w
                                cumulative = 0
                                chosen = neighbors[0]
                                for n, w in zip(neighbors, weights):
                                    cumulative += w
                                    if cumulative >= r_val:
                                        chosen = n
                                        break
                            next_node = chosen["name"]
                        else:
                            break  # Dead end
                    else:
                        # --- Normal walk with revisit penalty ---
                        weights = [n["weight"] * _revisit_penalty(visit_counts.get(n["name"], 0))
                                   for n in neighbors]
                        total_w = sum(weights)
                        if total_w <= 0:
                            chosen = random.choice(neighbors)
                        else:
                            r_val = random.random() * total_w
                            cumulative = 0
                            chosen = neighbors[0]
                            for n, w in zip(neighbors, weights):
                                cumulative += w
                                if cumulative >= r_val:
                                    chosen = n
                                    break
                        next_node = chosen["name"]
                    
                    if next_node not in path:
                        path.append(next_node)
                    visit_counts[next_node] = visit_counts.get(next_node, 0) + 1
                    prev_node = current
                    current = next_node
                
                # --- Collect provisional edges (non-adjacent pairs) ---
                if len(path) >= 3:
                    for i in range(len(path)):
                        for j in range(i + 2, len(path)):
                            provisional_edges.append({
                                "source": path[i],
                                "target": path[j],
                                "path": "->".join(path[i:j+1]),
                                "path_len": j - i,
                                "jump_points": jump_points,
                            })
            
            # --- Waking Gate: re-evaluate provisional edges ---
            # Only promote edges where source and target share at least 1 neighbor
            # (topological evidence of real association)
            seen_pairs = set()
            for pe in provisional_edges:
                pair = tuple(sorted((pe["source"], pe["target"])))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                
                # Count shared neighbors
                shared = session.run("""
                    MATCH (a:Concept {name: $src})--(shared:Concept)--(b:Concept {name: $tgt})
                    RETURN count(shared) as cnt
                """, src=pe["source"], tgt=pe["target"]).data()
                
                shared_count = shared[0]["cnt"] if shared else 0
                if shared_count >= WAKING_THRESHOLD:
                    # Confidence based on path length (shorter = stronger) + shared neighbors bonus
                    base_conf = max(0.25, 0.5 - pe["path_len"] * 0.05)
                    neighbor_bonus = min(0.2, shared_count * 0.05)
                    jump_bonus = 0.1 if pe["jump_points"] else 0.0
                    confidence = min(0.7, base_conf + neighbor_bonus + jump_bonus)
                    
                    promoted_edges.append({
                        "source": pe["source"],
                        "target": pe["target"],
                        "path": pe["path"],
                        "type": "DREAM_CONNECTION",
                        "confidence": round(confidence, 3),
                        "shared_neighbors": shared_count,
                        "jump_points": pe["jump_points"],
                    })
        
        driver.close()
    except Exception as e:
        log.warning(f"⚠️ REM 梦游 v2 失败: {e}")
    
    # --- Dedup + limit ---
    seen = set()
    unique_edges = []
    for e in promoted_edges:
        key = f"{e['source']}|{e['target']}"
        rkey = f"{e['target']}|{e['source']}"
        if key not in seen and rkey not in seen:
            seen.add(key)
            unique_edges.append(e)
            if len(unique_edges) >= REM_MAX_NEW_EDGES:
                break
    
    if unique_edges:
        log.info(f"  💭 REM v2: {len(unique_edges)} promoted edges "
                 f"(from {len(provisional_edges)} provisional, {total_jumps} jumps)")
        for e in unique_edges[:5]:
            log.info(f"    {e['source']} → {e['target']} "
                     f"(conf={e['confidence']}, shared={e['shared_neighbors']}, "
                     f"via {e['path'][:60]})")
    else:
        log.info(f"  💭 REM v2: 0 promoted (from {len(provisional_edges)} provisional, "
                 f"{total_jumps} jumps — waking gate filtered all)")
    
    return unique_edges


def rem_shy_downscale() -> dict:
    """
    SHY (Synaptic Homeostasis) Downscaling — from claude-brain.
    
    After dream cycle, rank all edges by weight:
    - Top SHY_PROTECTION_PCT (20%) are protected
    - Remaining edges get gradient downscale (weaker = more downscale)
    - Edges below SHY_PRUNE_THRESHOLD after downscale are removed
    
    This prevents unbounded edge growth while preserving important connections.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return {"downscaled": 0, "pruned": 0, "protected": 0}
    
    stats = {"downscaled": 0, "pruned": 0, "protected": 0, "total": 0}
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            # Get all edges with confidence, sorted by weight desc
            all_edges = session.run("""
                MATCH (a)-[r]->(b)
                WHERE r.confidence IS NOT NULL
                RETURN elementId(r) as rid, COALESCE(r.confidence, 0.5) as weight,
                       a.name as source, b.name as target, type(r) as rel_type
                ORDER BY weight DESC
            """).data()
            
            stats["total"] = len(all_edges)
            if not all_edges:
                driver.close()
                return stats
            
            protected_count = max(1, int(len(all_edges) * SHY_PROTECTION_PCT))
            unprotected_count = max(1, len(all_edges) - protected_count)
            
            for rank, edge in enumerate(all_edges):
                if rank < protected_count:
                    stats["protected"] += 1
                    continue  # Protected by rank
                
                # Gradient: weakest edges lose the most
                position = (rank - protected_count + 1) / unprotected_count
                scale = 1.0 - SHY_DOWNSCALE_FACTOR * position
                new_weight = edge["weight"] * max(0.0, scale)
                
                if new_weight < SHY_PRUNE_THRESHOLD:
                    # Prune this edge
                    session.run("MATCH ()-[r]->() WHERE elementId(r) = $rid DELETE r", rid=edge["rid"])
                    stats["pruned"] += 1
                else:
                    # Downscale
                    session.run("""
                        MATCH ()-[r]->() WHERE elementId(r) = $rid
                        SET r.confidence = $new_w, r.shy_downscaled = true
                    """, rid=edge["rid"], new_w=round(new_weight, 4))
                    stats["downscaled"] += 1
        
        driver.close()
    except Exception as e:
        log.warning(f"⚠️ SHY downscale 失败: {e}")
    
    return stats


def rem_threat_simulation() -> list[dict]:
    """
    Threat Simulation — from claude-brain.
    
    Scan high-confidence nodes for CONTRADICTS edges and flag them.
    These contradiction alerts surface conflicting information that needs resolution.
    
    Returns: [{"node": ..., "contradicts": ..., "severity": float}]
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return []
    
    threats = []
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            # Find nodes with CONTRADICTS edges
            results = session.run("""
                MATCH (a:Concept)-[r:CONTRADICTS]-(b:Concept)
                WHERE a.name IS NOT NULL AND b.name IS NOT NULL
                WITH a, b, COALESCE(r.confidence, 0.5) as severity
                ORDER BY severity DESC
                LIMIT 20
                RETURN a.name as node, b.name as contradicts, severity
            """).data()
            
            seen = set()
            for r in results:
                pair = tuple(sorted((r["node"], r["contradicts"])))
                if pair in seen:
                    continue
                seen.add(pair)
                threats.append({
                    "node": r["node"],
                    "contradicts": r["contradicts"],
                    "severity": round(r["severity"], 3),
                })
        
        driver.close()
    except Exception as e:
        log.warning(f"⚠️ Threat simulation 失败: {e}")
    
    return threats


def llm_boost_relations(walk_edges: list[dict], clusters: dict[str, list[dict]], 
                         max_boost: int = 10) -> list[dict]:
    """
    LLM Boost: 对梦游发现的低置信度关系(0.3)进行LLM验证并提升
    
    触发条件: 两个实体在clusters中有>=2条共享记忆 (说明关联有事实基础)
    
    流程:
    1. 遍历 walk_edges，检查每对实体的cluster共现频率
    2. 高共现的对 → 调LLM确认关联类型并生成一句话解释
    3. LLM确认 → conf 从 0.3 提升到 0.6
    4. LLM失败或低共现 → 保持 0.3
    
    成本: ~200 tokens/relation, 最多boost 10条
    """
    boosted = []
    not_boosted = []
    
    # 构建实体→记忆映射 (从clusters提取所有记忆文本)
    entity_memories: dict[str, list[str]] = defaultdict(list)
    for ck, group in clusters.items():
        texts = [m["text"][:200] for m in group]
        # 用 extract_entities_with_fallback 提取每个cluster的实体
        entities = extract_entities_with_fallback(texts, max_entities=3)
        for ent in entities:
            entity_memories[ent].extend(texts)
    
    for edge in walk_edges:
        src, tgt = edge.get("source", ""), edge.get("target", "")
        if not src or not tgt:
            not_boosted.append(edge)
            continue
        
        # 检查共现: 两个实体有多少条共享记忆
        src_mems = set(entity_memories.get(src, []))
        tgt_mems = set(entity_memories.get(tgt, []))
        shared = src_mems & tgt_mems
        
        # 也检查: src和tgt是否在同一个cluster出现
        co_cluster_count = 0
        for ck, group in clusters.items():
            texts = [m["text"][:200] for m in group]
            combined_text = " ".join(texts)
            if src.lower() in combined_text.lower() and tgt.lower() in combined_text.lower():
                co_cluster_count += 1
        
        # 触发条件: >=2个cluster共现 OR >=3条共享记忆
        should_boost = co_cluster_count >= 2 or len(shared) >= 3
        
        if not should_boost:
            # 保持低置信度
            edge["boost_attempted"] = False
            edge["boost_reason"] = f"low_cooccurrence(clusters={co_cluster_count},shared_mem={len(shared)})"
            not_boosted.append(edge)
            continue
        
        # LLM 验证
        # 时间感知：检查共享记忆中是否有过期市场数据
        shared_texts = list(shared)[:5] if shared else []
        has_stale_data = any(_is_time_sensitive(t) for t in shared_texts)
        stale_warning = "\n注意：上述记忆片段中可能包含过期的市场数据（价格/利率/利差等），不要基于具体数字建立因果关系，只判断概念层面的结构性关联。" if has_stale_data else ""
        
        prompt = f"""判断以下两个概念之间是否存在有意义的知识关联，并给出关联类型。

概念A: {src}
概念B: {tgt}

相关记忆片段:
{chr(10).join(f'- {t[:150]}...' for t in shared_texts) or '无直接共享记忆'}{stale_warning}

请判断:
1. 关联类型: CAUSE(因果)/PART_OF(包含)/DEPENDS_ON(依赖)/CONTRAST(对比)/EVOLUTION(演变)/RELATED(通用关联)
2. 一句话解释为什么它们关联

只输出一行JSON:
{{"type": "关联类型", "explanation": "一句话解释", "is_valid": true/false}}"""
        
        result = _call_infini(prompt, max_tokens=150, temperature=0.1)
        if result:
            import re
            json_match = re.search(r'\{[^{}]+\}', result)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if parsed.get("is_valid", False):
                        edge["confidence"] = 0.6
                        edge["boost_attempted"] = True
                        edge["boost_reason"] = f"llm_verified({parsed.get('type', 'RELATED')}): {parsed.get('explanation', '')[:80]}"
                        edge["relation_type"] = parsed.get("type", "RELATED_TO").replace(" ", "_")
                        boosted.append(edge)
                        log.info(f"  🔥 LLM Boost: {src} → {tgt} conf 0.3→0.6 ({parsed.get('type', '')})")
                        if len(boosted) >= max_boost:
                            break
                        continue
                except json.JSONDecodeError:
                    pass
        
        # LLM失败或判定无效 → 保持 0.3
        edge["boost_attempted"] = True
        edge["boost_reason"] = "llm_failed_or_invalid"
        not_boosted.append(edge)
    
    log.info(f"  🔥 LLM Boost 结果: {len(boosted)} 条提升→0.6, {len(not_boosted)} 条保持0.3")
    return boosted + not_boosted


# ─── P2: NREM Hebbian 强化 (SCM — 强化重要连接) ────────────────────────

def nrem_hebbian_consolidation() -> dict:
    """
    NREM Hebbian 强化: 高重要性概念之间的连接被强化，同时全局缩减
    
    来自 SCM:
    - Hebbian: Δs_ij = η · I(c_i) · I(c_j)
    - Downscaling: s_ij ← α · s_ij (保留相对排名，创建新空间)
    
    Returns: {"strengthened": N, "downscaled": N}
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return {"strengthened": 0, "downscaled": 0}
    
    strengthened = 0
    downscaled = 0
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            # 1. Hebbian 强化: 高度数节点之间的关系加强
            # Neo4j 5.26: 用 COUNT {} 替代 size((a)-[]-())
            result = session.run("""
                MATCH (a:Concept)-[r]->(b:Concept)
                WHERE r.confidence IS NOT NULL
                AND r.source = 'dream_cycle'
                AND COUNT { (a)--() } >= 5
                AND COUNT { (b)--() } >= 5
                RETURN a.name as src, b.name as tgt, 
                       r.confidence as conf, type(r) as rel_type
                LIMIT 50
            """).data()
            
            for row in result:
                old_conf = row.get("conf", 0.5)
                # Hebbian: Δs = η · I(a) · I(b)
                # I(a), I(b) 用 degree 的 log 作为代理
                delta = HEBBIAN_LEARNING_RATE  # 简化: 统一增量
                new_conf = min(1.0, old_conf + delta)
                
                safe_rel_type = ''.join(c for c in row.get("rel_type", "RELATED_TO") 
                                       if c.isalnum() or c == '_') or "RELATED_TO"
                
                session.run(f"""
                    MATCH (a:Concept {{name: $src}})-[r:{safe_rel_type}]->(b:Concept {{name: $tgt}})
                    SET r.confidence = $new_conf,
                        r.last_reinforced = datetime()
                """, src=row["src"], tgt=row["tgt"], new_conf=new_conf)
                strengthened += 1
            
            # 2. 全局缩减: 所有 dream_cycle 关系的 confidence × α
            session.run("""
                MATCH ()-[r]->()
                WHERE r.source = 'dream_cycle' AND r.confidence IS NOT NULL
                SET r.confidence = r.confidence * $alpha
            """, alpha=HEBBIAN_DOWNSCALE)
            
            # 计算被缩减的数量
            downscaled = session.run("""
                MATCH ()-[r]->()
                WHERE r.source = 'dream_cycle'
                RETURN COUNT(r) as cnt
            """).data()
            downscaled = downscaled[0]["cnt"] if downscaled else 0
        
        driver.close()
    except Exception as e:
        log.warning(f"⚠️ NREM Hebbian 失败: {e}")
    
    log.info(f"  🧠 NREM Hebbian: 强化 {strengthened} 条, 缩减 {downscaled} 条")
    return {"strengthened": strengthened, "downscaled": downscaled}


# ─── P3: 语义签名冲突检测 (SleepGate — "同槽不同值") ──────────────────

