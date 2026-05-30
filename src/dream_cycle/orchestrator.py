"""
Dream Cycle — Orchestrator — main pipeline, lock management, report formatting, batch operations
"""



__all__ = [
    "run_dream_cycle",
    "format_report",
    "send_telegram_report",
    "review_vault_suggestions",
    "batch_resolve_all_conflicts",
    "batch_review_all_vault",
]

import os
import json
import time
import sqlite3
import logging
import signal
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dream_cycle.config import (
    DREAM_DB, DREAM_LOCK, DREAM_LOCK_TIMEOUT, STATE_DB,
    NEO4J_URI, NEO4J_USER, NEO4J_PASS, VAULT_DIR, PERMANENT_MARKERS,
    HKT, log,
)
from dream_cycle.db import (
    init_dream_db, pg_query, get_recent_memories, get_incremental_memories,
    update_manifest, mark_manifest_archived, write_relations_to_neo4j,
    get_recall_stats,
)
from dream_cycle.stage1 import stage1_shallow_sleep
from dream_cycle.stage2 import stage2_rem
from dream_cycle.stage3 import stage3_deep_sleep, detect_slot_conflicts, resolve_slot_conflicts
from dream_cycle.dream_engine import (
    rem_dream_walk, llm_boost_relations, nrem_hebbian_consolidation,
)
from dream_cycle.similarity import combined_similarity
from dream_cycle.entities import extract_entities_with_fallback
from dream_cycle.session import (
    mine_recent_sessions, scan_session_signals, generate_session_digest,
)
from dream_cycle.vault import create_vault_stub
from dream_cycle.health import online_dedup_check
from dream_cycle.llm import llm_verify_contradiction

def _acquire_lock() -> bool:
    """获取并发锁，防止多个 dream cycle 同时运行"""
    import os, signal
    
    if DREAM_LOCK.exists():
        try:
            pid = int(DREAM_LOCK.read_text().strip())
            # 检查进程是否还活着
            os.kill(pid, 0)
            # 检查是否超时
            lock_age = time.time() - DREAM_LOCK.stat().st_mtime
            if lock_age > DREAM_LOCK_TIMEOUT:
                log.warning(f"🔓 锁超时 ({lock_age:.0f}s > {DREAM_LOCK_TIMEOUT}s), PID {pid} 可能是僵尸, 强制接管")
            else:
                log.error(f"🔒 另一个 dream cycle 正在运行 (PID {pid}, {lock_age:.0f}s 前)")
                return False
        except (ProcessLookupError, ValueError):
            log.warning(f"🔓 发现过期锁文件 (进程已死), 清理并继续")
    
    DREAM_LOCK.write_text(str(os.getpid()))
    return True

def _release_lock():
    """释放并发锁"""
    try:
        DREAM_LOCK.unlink(missing_ok=True)
    except Exception:
        pass

def _cleanup_zombie_runs():
    """清理僵尸 run (started 但 finished_at=NULL 超过 1 小时)"""
    conn = sqlite3.connect(str(DREAM_DB))
    cutoff = (datetime.now(HKT) - timedelta(hours=1)).isoformat()
    cursor = conn.execute("""
        UPDATE dream_runs 
        SET finished_at = ?, error = 'zombie: no finish after 1h'
        WHERE finished_at IS NULL AND started_at < ?
    """, (datetime.now(HKT).isoformat(), cutoff))
    cleaned = cursor.rowcount
    conn.commit()
    conn.close()
    if cleaned > 0:
        log.info(f"🧹 清理了 {cleaned} 个僵尸 run")

def _prepare_memories(hours: int) -> tuple[list[dict], list[dict], dict] | None:
    """
    Phase 1: Fetch memories and session signals.

    Returns ``(memories, sessions, signals)`` or *None* if nothing to process.
    """
    memories = get_incremental_memories(hours)
    log.info(f"📊 获取到 {len(memories)} 条新记忆 (最近 {hours} 小时, 增量)")

    # Fallback to full scan if incremental is too thin
    if len(memories) < 5:
        conn_check = sqlite3.connect(str(DREAM_DB))
        manifest_count = conn_check.execute(
            "SELECT COUNT(*) FROM processed_manifest WHERE status='active'"
        ).fetchone()[0]
        conn_check.close()

        if manifest_count > 10 and len(memories) == 0:
            return None  # genuinely nothing new

        all_recent = get_recent_memories(hours)
        if len(all_recent) > len(memories) * 3:
            log.info(f"  ⚠️ 增量太少({len(memories)}), 回退到全量({len(all_recent)})")
            memories = all_recent

    # Session mining
    sessions = mine_recent_sessions(hours)
    session_digest = generate_session_digest(sessions)
    log.info(f"📋 近期 session: {len(sessions)} 个")
    log.info(session_digest)

    # Session signal scanning (from Anthropic autoDream)
    signals = scan_session_signals(hours)
    total_signals = sum(len(v) for v in signals.values())
    if total_signals > 0:
        log.info(
            f"📡 Session 信号: {total_signals} 条 "
            f"(纠正={len(signals['corrections'])}, "
            f"偏好={len(signals['preferences'])}, "
            f"决策={len(signals['decisions'])}, "
            f"模式={len(signals['patterns'])})"
        )
        for sig_type, sigs in signals.items():
            for sig in sigs[:5]:
                memories.append({
                    "id": f"signal_{sig_type}_{sig['timestamp']:.0f}",
                    "text": f"[SESSION_{sig_type.upper()}] {sig['text']}",
                    "created_at": datetime.fromtimestamp(
                        sig["timestamp"], tz=timezone.utc
                    ).isoformat(),
                    "source": "session_signal",
                    "signal_type": sig_type,
                    "session_title": sig.get("session_title", ""),
                })
    else:
        log.info("📡 Session 信号: 0 条")

    return memories, sessions, signals


def _execute_stages(
    memories: list[dict],
    stages: str,
    dry_run: bool,
    dream_run_id: int,
) -> tuple[dict, dict, dict, list[dict]]:
    """
    Phase 2: Run pipeline stages 1-3 + dream engine + Hebbian + slot conflicts.

    Returns ``(clusters, rem_results, stats, dream_walk_edges)``.
    """
    clusters: dict = {}
    if "1" in stages:
        clusters = stage1_shallow_sleep(memories)

    rem_results: dict = {}
    if "2" in stages:
        rem_results = stage2_rem(clusters)

    stats: dict = {}
    if "3" in stages:
        stats = stage3_deep_sleep(
            rem_results, dream_run_id, dry_run,
            total_memories=len(memories),
            total_clusters=len(clusters),
        )

    # REM dream walk (Neo4j random walk v2)
    dream_walk_edges: list[dict] = []
    if "2" in stages and not dry_run:
        all_cluster_entities: list[str] = []
        for _ck, group in clusters.items():
            texts = [m["text"][:200] for m in group]
            ents = extract_entities_with_fallback(texts, max_entities=3)
            all_cluster_entities.extend(ents)

        dream_walk_edges = rem_dream_walk(cluster_entities=all_cluster_entities)
        if dream_walk_edges:
            dream_walk_edges = llm_boost_relations(dream_walk_edges, clusters, max_boost=10)
            written = write_relations_to_neo4j(dream_walk_edges)
            stats["dream_walk"] = written
            conn_rl = sqlite3.connect(str(DREAM_DB))
            for e in dream_walk_edges:
                conn_rl.execute(
                    "INSERT INTO relation_log "
                    "(dream_run_id, source_entity, target_entity, relation_type, confidence, method) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (dream_run_id, e["source"], e["target"],
                     "DREAM_WALK", e["confidence"], "rem_dream_walk"),
                )
            conn_rl.commit()
            conn_rl.close()

    # NREM Hebbian consolidation
    if "3" in stages and not dry_run:
        hebbian_stats = nrem_hebbian_consolidation()
        stats["hebbian_strengthened"] = hebbian_stats.get("strengthened", 0)
        stats["hebbian_downscaled"] = hebbian_stats.get("downscaled", 0)

    # Slot conflict detection
    if "2" in stages:
        slot_conflicts = detect_slot_conflicts()
        if slot_conflicts:
            conn_sc = sqlite3.connect(str(DREAM_DB))
            for c in slot_conflicts:
                conn_sc.execute(
                    "INSERT INTO contradiction_log "
                    "(dream_run_id, mem1_id, mem2_id, marker, contradiction_type, llm_explanation, verified) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (dream_run_id, c["mem1_id"], c["mem2_id"],
                     f"slot_conflict(sim={c['slot_similarity']:.2f},diff={c['value_diff']:.2f})",
                     "SLOT_CONFLICT",
                     f"同槽不同值: 槽相似度={c['slot_similarity']:.2f}, 值差异={c['value_diff']:.2f}",
                     0),
                )
            conn_sc.commit()
            conn_sc.close()
            stats["slot_conflicts"] = len(slot_conflicts)
            rem_results["slot_conflicts_list"] = slot_conflicts

    return clusters, rem_results, stats, dream_walk_edges


def _finalize_run(
    dream_run_id: int,
    memories: list[dict],
    clusters: dict,
    rem_results: dict,
    stats: dict,
    dry_run: bool,
    start_time: datetime,
) -> dict:
    """
    Phase 3: Update manifest, record results, return summary.
    """
    # Update manifest (incremental tracking)
    if memories and not dry_run:
        update_manifest(memories, dream_run_id)
        log.info(f"  📋 Manifest 已更新: {len(memories)} 条标记为已处理")

        archived_ids = [
            item["remove"]["id"]
            for item in rem_results.get("dedup_candidates", [])
        ]
        if archived_ids:
            mark_manifest_archived(archived_ids)

    # Record in dream_runs
    end_time = datetime.now(HKT)
    dream_conn = sqlite3.connect(str(DREAM_DB))
    dream_conn.execute("""        UPDATE dream_runs SET
            finished_at = ?,
            stage1_clusters = ?,
            stage2_boosted = ?,
            stage3_deduped = ?,
            stage3_inferred = ?,
            stage3_decayed = ?,
            stage3_vault_suggestions = ?,
            summary = ?
        WHERE id = ?
    """, (
        end_time.isoformat(),
        len(clusters),
        len(rem_results.get("boosted", [])),
        stats.get("deduped", 0),
        stats.get("inferred", 0),
        stats.get("decayed", 0),
        stats.get("vault_suggestions", 0),
        json.dumps({
            "memories_scanned": len(memories),
            "clusters": len(clusters),
            "dedup_candidates": len(rem_results.get("dedup_candidates", [])),
            "merge_candidates": len(rem_results.get("merge_candidates", [])),
            "vault_candidates": len(rem_results.get("vault_candidates", [])),
            "decay_candidates": len(rem_results.get("decay_candidates", [])),
        }, ensure_ascii=False),
        dream_run_id,
    ))
    dream_conn.commit()
    dream_conn.close()

    result = {
        "status": "success",
        "dream_run_id": dream_run_id,
        "memories_scanned": len(memories),
        "clusters": len(clusters),
        "deduped": stats.get("deduped", 0),
        "inferred": stats.get("inferred", 0),
        "decayed": stats.get("decayed", 0),
        "vault_suggestions": stats.get("vault_suggestions", 0),
        "boosted": stats.get("boosted", 0),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "rem_results": rem_results,
        "stats": stats,
    }

    log.info(f"🌅 梦循环完成 — {result['duration_seconds']:.1f}s")
    return result


def run_dream_cycle(hours: int = 48, dry_run: bool = False, stages: str = "123") -> dict:
    """
    Execute the full dream cycle pipeline.

    Args:
        hours: how many hours of memories to scan
        dry_run: analyze only, don't execute
        stages: which stages to run (1/2/3/12/23/123)
    """
    start_time = datetime.now(HKT)
    log.info(
        f"🌙 梦循环启动 @ {start_time.strftime('%Y-%m-%d %H:%M:%S')} HKT "
        f"(hours={hours}, dry_run={dry_run}, stages={stages})"
    )

    if not _acquire_lock():
        return {"status": "skipped", "reason": "another_instance_running"}

    _cleanup_zombie_runs()

    dream_conn = init_dream_db()
    cursor = dream_conn.execute(
        "INSERT INTO dream_runs (started_at) VALUES (?)",
        (start_time.isoformat(),),
    )
    dream_run_id = cursor.lastrowid
    dream_conn.commit()
    dream_conn.close()

    try:
        # Phase 1: Prepare
        prepared = _prepare_memories(hours)
        if prepared is None:
            log.info("  ✅ 无新记录, 跳过梦循环")
            _skip_run(dream_run_id, "no_new_memories_incremental")
            _release_lock()
            return {"status": "skipped", "reason": "no_new_memories_incremental"}

        memories, sessions, signals = prepared
        if not memories and not sessions:
            log.warning("⚠️ 没有新记忆或session, 跳过梦循环")
            _release_lock()
            return {"status": "skipped", "reason": "no_memories"}

        # Phase 2: Execute stages
        clusters, rem_results, stats, dream_walk_edges = _execute_stages(
            memories, stages, dry_run, dream_run_id,
        )

        # Phase 3: Finalize
        result = _finalize_run(
            dream_run_id, memories, clusters, rem_results, stats,
            dry_run, start_time,
        )
        _release_lock()
        return result

    except Exception as e:
        log.error(f"❌ 梦循环失败: {e}", exc_info=True)
        dream_conn = sqlite3.connect(str(DREAM_DB))
        dream_conn.execute(
            "UPDATE dream_runs SET error = ?, finished_at = ? WHERE id = ?",
            (str(e), datetime.now(HKT).isoformat(), dream_run_id),
        )
        dream_conn.commit()
        dream_conn.close()
        _release_lock()
        return {"status": "error", "error": str(e)}


def _skip_run(dream_run_id: int, reason: str) -> None:
    """Mark a dream run as skipped in the database."""
    conn = sqlite3.connect(str(DREAM_DB))
    conn.execute(
        "UPDATE dream_runs SET finished_at = ?, summary = ? WHERE id = ?",
        (datetime.now(HKT).isoformat(), json.dumps({"status": reason}), dream_run_id),
    )
    conn.commit()
    conn.close()


def format_report(result: dict, rem_results: dict = None, stats: dict = None) -> str:
    """
    P6: 生成 Telegram 友好的增强报告
    
    新增: Top3 boost + Top3 vault + Top3 语义冲突 + 健康评分
    """
    if result["status"] == "skipped":
        return "💤 Dream Cycle — 无新记忆，跳过"
    
    if result["status"] == "error":
        return f"💤 Dream Cycle — ❌ 错误: {result['error'][:100]}"
    
    lines = [
        "💤 **Dream Cycle v3 完成**",
        "",
        f"📊 扫描: {result['memories_scanned']} 条 | 聚类: {result['clusters']} 组",
        f"🔥 Boost: {result.get('boosted', 0)} | 🔗 推断: {result['inferred']} 关系",
        f"🗑️ 去重: {result['deduped']} | 📉 衰减: {result['decayed']} 候选",
        f"📝 Vault: {result['vault_suggestions']} 建议",
        f"⏱️ {result['duration_seconds']:.1f}s",
    ]
    
    # P6: Top3 Boost 详情
    if rem_results:
        boosted = rem_results.get("boosted", [])
        if boosted:
            lines.append("")
            lines.append("🔥 **Top3 Boost (高重要性记忆)**")
            for b in sorted(boosted, key=lambda x: x["score"], reverse=True)[:3]:
                text_preview = b["memory"]["text"][:60].replace("\n", " ")
                lines.append(f"  • {b['score']:.2f} — {text_preview}... ({b.get('reason', '')})")
        
        # P6: Top3 Vault 候选
        vault = rem_results.get("vault_candidates", [])
        if vault:
            lines.append("")
            lines.append("📝 **Top3 Vault 候选**")
            for v in sorted(vault, key=lambda x: x.get("priority", "normal"), reverse=True)[:3]:
                kw = ", ".join(v.get("keywords", [])[:3])
                priority = v.get("priority", "normal")
                gate = v.get("promotion_pass", "")
                lines.append(f"  • [{priority}] {kw} (score={v['best_score']:.2f}, gate={gate})")
        
        # P6: Top3 语义冲突
        contradictions = rem_results.get("contradictions", [])
        if contradictions:
            lines.append("")
            lines.append("⚡ **Top3 语义冲突**")
            for c in contradictions[:3]:
                m1 = c.get("mem1", {}).get("text", "")[:50].replace("\n", " ")
                m2 = c.get("mem2", {}).get("text", "")[:50].replace("\n", " ")
                vtype = c.get("contradiction_type", c.get("verified", "?"))
                lines.append(f"  • [{vtype}] {m1}... vs {m2}...")
    
    # P6: 健康评分
    if stats and stats.get("health_score"):
        lines.append("")
        lines.append(f"💊 健康评分: **{stats['health_score']:.0f}/100**")
    
    return "\n".join(lines)


def send_telegram_report(report: str):
    """通过 Telegram Bot 发送报告"""
    import subprocess
    # 用 hermes send_message 或直接 curl
    # 从 config 读 bot token + chat_id
    try:
        config_path = Path("/root/.hermes/config.yaml")
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            token = config.get("telegram", {}).get("bot_token", "")
            chat_id = config.get("telegram", {}).get("home_chat_id", "")
            if token and chat_id:
                import urllib.parse
                encoded = urllib.parse.quote(report, safe='')
                url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={encoded}&parse_mode=Markdown"
                subprocess.run(["curl", "-s", url], timeout=10, capture_output=True)
                log.info("📱 Telegram 报告已发送")
                return
        log.warning("⚠️ Telegram 配置不完整，跳过发送")
    except Exception as e:
        log.warning(f"⚠️ Telegram 发送失败: {e}")


# ─── Session 挖掘 — 从 state.db 提取近期会话主题 ──────────────────────

def review_vault_suggestions(max_review: int = 5) -> list[dict]:
    """
    P9: 处理 pending vault suggestion
    
    1. 对 auto_created 的 stub 页 — 用 wiki-ingest --llm 充实内容
    2. 对 pending 的 — LLM 判断是否值得创建stub
    3. 不值得的 → 标记 rejected
    
    Returns: 处理结果列表
    """
    conn = sqlite3.connect(str(DREAM_DB))
    
    # 1. 处理 pending 建议
    pending = conn.execute("""
        SELECT id, entity, category, frequency, reason, dream_run_id
        FROM vault_suggestion WHERE status = 'pending'
        ORDER BY frequency DESC
        LIMIT ?
    """, (max_review,)).fetchall()
    
    # 2. 处理 auto_created 但内容还是stub的
    auto_created = conn.execute("""
        SELECT id, entity, category
        FROM vault_suggestion WHERE status = 'auto_created'
        ORDER BY frequency DESC
    """).fetchall()
    
    conn.close()
    
    reviewed = []
    
    # Process pending: LLM 判断是否值得
    for p in pending:
        sid, entity, category, freq, reason, drid = p
        
        # 检查是否已有 Vault 页面
        slug = entity.lower().replace(" ", "-")[:50]
        cat_map = {"markets": "markets", "investment": "markets", "projects": "projects", 
                   "technology": "concepts", "concepts": "concepts"}
        vault_cat = cat_map.get(category, "concepts")
        filepath = VAULT_DIR / vault_cat / f"{slug}.md"
        
        if filepath.exists():
            # 已有页面 → 标记 reviewed
            conn = sqlite3.connect(str(DREAM_DB))
            conn.execute("UPDATE vault_suggestion SET status = 'reviewed' WHERE id = ?", (sid,))
            conn.commit()
            conn.close()
            reviewed.append({"entity": entity, "action": "already_exists", "status": "reviewed"})
            continue
        
        # 频率太低(<2) → reject
        if freq < 2:
            conn = sqlite3.connect(str(DREAM_DB))
            conn.execute("UPDATE vault_suggestion SET status = 'rejected' WHERE id = ?", (sid,))
            conn.commit()
            conn.close()
            reviewed.append({"entity": entity, "action": "rejected_low_freq", "status": "rejected"})
            continue
        
        # 频率>=2 → 创建 stub (LLM充实概述)
        keywords = [entity]
        sample = f"{entity} (出现 {freq} 次, {reason})"
        vault_path = create_vault_stub(entity, category, keywords, sample, sample_age_days=None)
        if vault_path:
            conn = sqlite3.connect(str(DREAM_DB))
            conn.execute("UPDATE vault_suggestion SET status = 'auto_created' WHERE id = ?", (sid,))
            conn.commit()
            conn.close()
            reviewed.append({"entity": entity, "action": "stub_created", "path": vault_path, "status": "auto_created"})
        else:
            reviewed.append({"entity": entity, "action": "stub_failed", "status": "pending"})
    
    # Process auto_created: 检查内容是否需要充实
    for ac in auto_created:
        sid, entity, category = ac
        slug = entity.lower().replace(" ", "-")[:50]
        cat_map = {"markets": "markets", "investment": "markets", "projects": "projects",
                   "technology": "concepts", "concepts": "concepts"}
        vault_cat = cat_map.get(category, "concepts")
        filepath = VAULT_DIR / vault_cat / f"{slug}.md"
        
        if filepath.exists():
            # 检查内容长度: <500字 = 还是stub → 标记需要充实
            content = filepath.read_text()
            word_count = len(content.split())
            if word_count < 100:
                reviewed.append({"entity": entity, "action": "needs_enrichment", "words": word_count})
            else:
                # 内容已充实 → 标记 reviewed
                conn = sqlite3.connect(str(DREAM_DB))
                conn.execute("UPDATE vault_suggestion SET status = 'reviewed' WHERE id = ?", (sid,))
                conn.commit()
                conn.close()
                reviewed.append({"entity": entity, "action": "enriched", "words": word_count, "status": "reviewed"})
    
    log.info(f"  📝 P9 Vault Review: {len(reviewed)} 条处理")
    for r in reviewed[:5]:
        log.info(f"    {r['entity']}: {r['action']}")
    
    return reviewed


# ─── P10: 批量积压处理 ─────────────────────────────────────────────────

def batch_resolve_all_conflicts(max_per_run: int = 50) -> dict:
    """
    批量处理所有 pending 矛盾 — 每次最多处理 max_per_run 个
    
    从 contradiction_log 读取 pending 记录，获取记忆文本，调用 LLM 分类
    """
    conn = sqlite3.connect(str(DREAM_DB))
    pending = conn.execute("""
        SELECT id, mem1_id, mem2_id, marker
        FROM contradiction_log WHERE resolution = 'pending'
        ORDER BY id
        LIMIT ?
    """, (max_per_run,)).fetchall()
    total_pending = conn.execute("SELECT COUNT(*) FROM contradiction_log WHERE resolution = 'pending'").fetchone()[0]
    conn.close()
    
    if not pending:
        log.info("✅ 无 pending 矛盾需要处理")
        return {"resolved": 0, "remaining": 0}
    
    log.info(f"🔍 批量矛盾处理: {len(pending)}/{total_pending} pending")
    
    resolved_count = 0
    failed = 0
    superseded = 0
    extended = 0
    false_pos = 0
    
    for row in pending:
        cid, mem1_id, mem2_id, marker = row
        
        # 获取记忆文本
        rows1 = pg_query(f"SELECT payload->>'data' FROM mem0 WHERE id::text = '{mem1_id}'")
        rows2 = pg_query(f"SELECT payload->>'data' FROM mem0 WHERE id::text = '{mem2_id}'")
        
        text1 = rows1[0][0] if rows1 and rows1[0][0] else ""
        text2 = rows2[0][0] if rows2 and rows2[0][0] else ""
        
        if not text1 or not text2:
            # 记忆已被删除 → 标记为 FALSE_POSITIVE
            conn = sqlite3.connect(str(DREAM_DB))
            conn.execute("UPDATE contradiction_log SET resolution = 'false_positive', llm_explanation = ? WHERE id = ?",
                        ("memory_deleted", cid))
            conn.commit()
            conn.close()
            false_pos += 1
            resolved_count += 1
            continue
        
        # LLM 判断
        v = llm_verify_contradiction(text1[:300], text2[:300], marker)
        
        if v is None:
            log.warning(f"  ⚠️ API 失败: {mem1_id[:8]} vs {mem2_id[:8]}")
            failed += 1
            continue
        
        ctype = v.get("type", "FALSE_POSITIVE")
        explanation = v.get("explanation", "")[:200].replace('"', '').replace("'", "")
        
        conn = sqlite3.connect(str(DREAM_DB))
        
        if ctype == "SUPERSEDE":
            # 归档较旧的记忆
            rows_t1 = pg_query(f"SELECT payload->>'created_at' FROM mem0 WHERE id::text = '{mem1_id}'")
            rows_t2 = pg_query(f"SELECT payload->>'created_at' FROM mem0 WHERE id::text = '{mem2_id}'")
            t1 = rows_t1[0][0] if rows_t1 and rows_t1[0][0] else ""
            t2 = rows_t2[0][0] if rows_t2 and rows_t2[0][0] else ""
            
            older_id = mem1_id if t1 <= t2 else mem2_id
            newer_id = mem2_id if older_id == mem1_id else mem1_id
            
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{"archived": true, "archived_reason": "slot_supersede", "superseded_by": "{newer_id}"}}' WHERE id::text = '{older_id}'""")
            mark_manifest_archived([older_id])
            superseded += 1
            log.info(f"  ✅ SUPERSEDE: 归档 {older_id[:8]}")
        
        elif ctype == "EXTEND":
            extended += 1
            log.info(f"  🔗 EXTEND: {mem1_id[:8]} ↔ {mem2_id[:8]}")
        
        else:  # FALSE_POSITIVE
            false_pos += 1
        
        conn.execute("""
            UPDATE contradiction_log SET contradiction_type = ?, resolution = ?, 
                   llm_explanation = ?, verified = 1 WHERE id = ?
        """, (f"SLOT_CONFLICT->{ctype}", ctype, explanation, cid))
        conn.commit()
        conn.close()
        resolved_count += 1
    
    remaining = total_pending - resolved_count
    log.info(f"📊 批量矛盾完成: {resolved_count} resolved (S:{superseded} E:{extended} FP:{false_pos} fail:{failed}), {remaining} remaining")
    return {"resolved": resolved_count, "remaining": remaining, "superseded": superseded, "extended": extended, "false_positive": false_pos, "failed": failed}


def batch_review_all_vault(max_per_run: int = 100) -> dict:
    """
    批量处理所有 pending vault 建议
    
    freq>=2 → 创建 stub; freq<2 → reject; 已有页面 → reviewed
    """
    conn = sqlite3.connect(str(DREAM_DB))
    pending = conn.execute("""
        SELECT id, entity, category, frequency, reason
        FROM vault_suggestion WHERE status = 'pending'
        ORDER BY frequency DESC
        LIMIT ?
    """, (max_per_run,)).fetchall()
    total_pending = conn.execute("SELECT COUNT(*) FROM vault_suggestion WHERE status='pending'").fetchone()[0]
    conn.close()
    
    if not pending:
        log.info("✅ 无 pending vault 建议需要处理")
        return {"processed": 0, "remaining": 0}
    
    log.info(f"📝 批量 Vault 审核: {len(pending)}/{total_pending} pending")
    
    created = 0
    rejected = 0
    exists = 0
    failed = 0
    
    for row in pending:
        sid, entity, category, freq, reason = row
        
        # 检查是否已有 Vault 页面
        slug = entity.lower().replace(" ", "-")[:50]
        cat_map = {"markets": "markets", "investment": "markets", "projects": "projects",
                   "technology": "concepts", "concepts": "concepts"}
        vault_cat = cat_map.get(category, "concepts")
        filepath = VAULT_DIR / vault_cat / f"{slug}.md"
        
        conn = sqlite3.connect(str(DREAM_DB))
        
        if filepath.exists():
            conn.execute("UPDATE vault_suggestion SET status = 'reviewed' WHERE id = ?", (sid,))
            conn.commit()
            conn.close()
            exists += 1
            continue
        
        if freq < 2:
            conn.execute("UPDATE vault_suggestion SET status = 'rejected' WHERE id = ?", (sid,))
            conn.commit()
            conn.close()
            rejected += 1
            continue
        
        # 创建 stub
        keywords = [entity]
        sample = f"{entity} (出现 {freq} 次, {reason or ''})"
        vault_path = create_vault_stub(entity, category, keywords, sample, sample_age_days=None)
        if vault_path:
            conn.execute("UPDATE vault_suggestion SET status = 'auto_created' WHERE id = ?", (sid,))
            conn.commit()
            conn.close()
            created += 1
            log.info(f"  📄 创建: {entity} → {vault_path}")
        else:
            conn.close()
            failed += 1
    
    remaining = total_pending - (created + rejected + exists + failed)
    log.info(f"📊 批量 Vault 完成: {created} created, {rejected} rejected, {exists} exists, {failed} failed, {remaining} remaining")
    return {"created": created, "rejected": rejected, "exists": exists, "failed": failed, "remaining": remaining}


# ─── CLI ───────────────────────────────────────────────────────────────

