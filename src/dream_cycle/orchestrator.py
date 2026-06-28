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
    "cmd_adopt",
]

import os
import json
import time
import sqlite3
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dream_cycle.config import (
    DREAM_DB,
    DREAM_LOCK,
    DREAM_LOCK_TIMEOUT,
    VAULT_DIR,
    HKT,
    log,
)
from dream_cycle.db import (
    init_dream_db,
    pg_query,
    get_recent_memories,
    get_incremental_memories,
    claim_memories,
    update_manifest,
    mark_manifest_archived,
    write_relations_to_neo4j,
)
from dream_cycle.stage1 import stage1_shallow_sleep
from dream_cycle.stage2 import stage2_rem
from dream_cycle.stage3 import stage3_deep_sleep, detect_slot_conflicts, resolve_slot_conflicts
from dream_cycle.dream_engine import (
    rem_dream_walk,
    llm_boost_relations,
    nrem_hebbian_consolidation,
)
from dream_cycle.entities import extract_entities_with_fallback
from dream_cycle.session import (
    mine_recent_sessions,
    scan_session_signals,
    generate_session_digest,
    detect_feedback_signals,
)
from dream_cycle.vault import create_vault_stub
from dream_cycle.llm import llm_verify_contradiction
from dream_cycle.shmr import contrastive_beliefs

# ── v6 "Safe Sleep" imports ──────────────────────────────────────────────────
from dream_cycle.split import split_memories, get_val_memories, split_stats
from dream_cycle.budget import EditBudget, COSTLY_OPS
from dream_cycle.staging import StagingBuffer, adopt_staging, latest_staging, staging_status
from dream_cycle.validation import quick_validate

import dream_cycle.db as _db_module  # for monkey-patching PG writes


def _acquire_lock() -> bool:
    """获取并发锁，防止多个 dream cycle 同时运行"""

    if DREAM_LOCK.exists():
        try:
            pid = int(DREAM_LOCK.read_text().strip())
            # 检查进程是否还活着
            os.kill(pid, 0)
            # 检查是否超时
            lock_age = time.time() - DREAM_LOCK.stat().st_mtime
            if lock_age > DREAM_LOCK_TIMEOUT:
                log.warning(
                    f"🔓 锁超时 ({lock_age:.0f}s > {DREAM_LOCK_TIMEOUT}s), PID {pid} 可能是僵尸, 强制接管"
                )
            else:
                log.error(
                    f"🔒 另一个 dream cycle 正在运行 (PID {pid}, {lock_age:.0f}s 前)"
                )
                return False
        except (ProcessLookupError, ValueError):
            log.warning("🔓 发现过期锁文件 (进程已死), 清理并继续")

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
    conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
    cutoff = (datetime.now(HKT) - timedelta(hours=1)).isoformat()
    cursor = conn.execute(
        """
        UPDATE dream_runs 
        SET finished_at = ?, error = 'zombie: no finish after 1h'
        WHERE finished_at IS NULL AND started_at < ?
    """,
        (datetime.now(HKT).isoformat(), cutoff),
    )
    cleaned = cursor.rowcount
    conn.commit()
    conn.close()
    if cleaned > 0:
        log.info(f"🧹 清理了 {cleaned} 个僵尸 run")


# ── v6 "Safe Sleep": Write Interceptor ───────────────────────────────────────
# Monkey-patches db module functions to route PG writes into StagingBuffer
# instead of executing them directly. Stage3 code is unchanged.

_staging_active = False
_staging_buffer: StagingBuffer | None = None
_edit_budget: EditBudget | None = None
_original_pg_query = None
_original_update_memory_text = None
_original_delete_memory = None


def _install_staging_interceptors(buffer: StagingBuffer, budget: EditBudget):
    """Install write interceptors that route PG writes to staging buffer."""
    global _staging_active, _staging_buffer, _edit_budget
    global _original_pg_query, _original_update_memory_text, _original_delete_memory

    _staging_active = True
    _staging_buffer = buffer
    _edit_budget = budget

    # Save originals
    _original_pg_query = _db_module.pg_query
    _original_update_memory_text = _db_module.update_memory_text
    _original_delete_memory = _db_module.delete_memory

    # Install interceptors
    _db_module.pg_query = _intercepted_pg_query
    _db_module.update_memory_text = _intercepted_update_memory_text
    _db_module.delete_memory = _intercepted_delete_memory

    log.info("🔒 Staging interceptors installed — PG writes redirected to buffer")


def _remove_staging_interceptors():
    """Restore original db functions."""
    global _staging_active, _staging_buffer, _edit_budget
    global _original_pg_query, _original_update_memory_text, _original_delete_memory

    if _original_pg_query:
        _db_module.pg_query = _original_pg_query
    if _original_update_memory_text:
        _db_module.update_memory_text = _original_update_memory_text
    if _original_delete_memory:
        _db_module.delete_memory = _original_delete_memory

    _staging_active = False
    _staging_buffer = None
    _edit_budget = None
    _original_pg_query = None
    _original_update_memory_text = None
    _original_delete_memory = None

    log.info("🔓 Staging interceptors removed — PG writes restored to direct")


def _intercepted_pg_query(sql: str) -> list:
    """Intercept pg_query calls and route destructive writes to staging."""
    sql_upper = sql.strip().upper()

    # Non-destructive queries pass through
    if sql_upper.startswith("SELECT"):
        return _original_pg_query(sql)

    # Destructive UPDATE on mem0 — intercept
    if "UPDATE MEM0" in sql_upper:
        # Parse memory_id from WHERE clause
        import re
        id_match = re.search(r"id::text\s*=\s*'([^']+)'", sql)
        mem_id = id_match.group(1) if id_match else "unknown"

        # Parse payload patch from SET clause
        patch_match = re.search(r"payload\s*\|\|\s*'(\{[^}]+\})'", sql)
        payload_patch = {}
        if patch_match:
            try:
                payload_patch = json.loads(patch_match.group(1).replace("''", "'"))
            except json.JSONDecodeError:
                payload_patch = {"raw_sql_snippet": sql[:200]}

        # Determine stage from payload content
        stage = "unknown"
        if "dream_boost" in sql:
            stage = "boost"
        elif "archived" in sql and "dedup" in sql:
            stage = "dedup"
        elif "archived" in sql and "decay" in sql:
            stage = "decay"
        elif "archived" in sql and "supersede" in sql:
            stage = "supersede"
        elif "archived" in sql and "slot" in sql:
            stage = "supersede"
        elif "extended" in sql:
            stage = "extend"
        elif "freshness" in sql:
            stage = "boost"

        # Budget check for costly ops
        op_name = f"{stage}_archive" if "archived" in sql else stage
        if _edit_budget and op_name in COSTLY_OPS:
            if not _edit_budget.spend(op_name, detail=sql[:100], memory_id=mem_id):
                log.info(f"  ⏸️ Budget skip: {op_name} on {mem_id[:8]}")
                return []

        # Route to staging buffer
        if "archived" in sql:
            _staging_buffer.add_archive(
                mem_id, reason=sql[:150], stage=stage,
                payload_patch=payload_patch,
            )
        else:
            _staging_buffer.add_update_payload(
                mem_id, payload_patch, reason=sql[:150], stage=stage,
            )

        log.info(f"  📝 Staged: UPDATE mem0 {stage} {mem_id[:8]}...")
        return []

    # DELETE — intercept
    if sql_upper.startswith("DELETE"):
        import re
        id_match = re.search(r"id::text\s*=\s*'([^']+)'", sql)
        mem_id = id_match.group(1) if id_match else "unknown"

        if _edit_budget and not _edit_budget.spend("merge", detail=f"delete {mem_id[:8]}", memory_id=mem_id):
            log.info(f"  ⏸️ Budget skip: delete {mem_id[:8]}")
            return []

        _staging_buffer.add_delete(mem_id, reason="merge secondary", stage="merge")
        log.info(f"  📝 Staged: DELETE mem0 {mem_id[:8]}...")
        return []

    # Other writes (INSERT, CREATE TABLE, etc.) — pass through
    return _original_pg_query(sql)


def _intercepted_update_memory_text(memory_id: str, new_text: str) -> bool:
    """Intercept update_memory_text and route to staging."""
    if _edit_budget and not _edit_budget.spend("merge", detail=f"update text {memory_id[:8]}", memory_id=memory_id):
        log.info(f"  ⏸️ Budget skip: update text {memory_id[:8]}")
        return False

    _staging_buffer.add_update_text(memory_id, new_text, reason="merge primary", stage="merge")
    log.info(f"  📝 Staged: UPDATE TEXT {memory_id[:8]}...")
    return True


def _intercepted_delete_memory(memory_id: str) -> bool:
    """Intercept delete_memory and route to staging."""
    if _edit_budget and not _edit_budget.spend("merge", detail=f"delete {memory_id[:8]}", memory_id=memory_id):
        log.info(f"  ⏸️ Budget skip: delete {memory_id[:8]}")
        return False

    _staging_buffer.add_delete(memory_id, reason="merge secondary", stage="merge")
    log.info(f"  📝 Staged: DELETE {memory_id[:8]}...")
    return True


# ── v6: Adopt command ────────────────────────────────────────────────────────

def cmd_adopt(staging_dir: str = "") -> dict:
    """Apply staged proposals to live PG database."""
    if not staging_dir:
        staging_dir = latest_staging()
        if not staging_dir:
            return {"error": "no staging directory found"}

    log.info(f"📋 Adopting from: {staging_dir}")
    result = adopt_staging(staging_dir)
    log.info(f"✅ Adopt complete: {result.get('applied', 0)}/{result.get('total', 0)} applied")
    return result


def _prepare_memories(hours: int) -> tuple[list[dict], list[dict], dict, dict] | None:
    """
    Phase 1: Fetch memories and session signals.

    Returns ``(memories, sessions, signals, feedback)`` or *None* if nothing to process.
    """
    memories = get_incremental_memories(hours)
    log.info(f"📊 获取到 {len(memories)} 条新记忆 (最近 {hours} 小时, 增量)")

    # Fallback to full scan if incremental is too thin
    if len(memories) < 5:
        conn_check = sqlite3.connect(str(DREAM_DB), timeout=5.0)
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
            f"模式={len(signals['patterns'])}, "
            f"身份={len(signals.get('identity', []))})"
        )
        for sig_type, sigs in signals.items():
            for sig in sigs[:5]:
                # Identity signals get higher importance (from Mnemosyne)
                importance = 0.85 if sig_type == "identity" else 0.5
                source = "identity" if sig_type == "identity" else "session_signal"
                memories.append(
                    {
                        "id": f"signal_{sig_type}_{sig['timestamp']:.0f}",
                        "text": f"[SESSION_{sig_type.upper()}] {sig['text']}",
                        "created_at": datetime.fromtimestamp(sig["timestamp"], tz=HKT).isoformat(),
                        "source": source,
                        "signal_type": sig_type,
                        "importance": importance,
                        "session_title": sig.get("session_title", ""),
                    }
                )
    else:
        log.info("📡 Session 信号: 0 条")

    # P1-1: Feedback signal detection (SkillOpt-inspired outcome classification)
    feedback = detect_feedback_signals(hours)
    if feedback["positive"] or feedback["negative"]:
        log.info(
            f"💬 Feedback: {len(feedback['positive'])} positive, "
            f"{len(feedback['negative'])} negative, "
            f"{len(feedback['session_outcomes'])} sessions classified"
        )
    # Tag memories with session outcome for contrastive reflection
    for mem in memories:
        sid = mem.get("session_id", "")
        if sid in feedback.get("session_outcomes", {}):
            mem["_outcome"] = feedback["session_outcomes"][sid]

    return memories, sessions, signals, feedback


def _execute_stages(
    memories: list[dict],
    stages: str,
    dry_run: bool,
    dream_run_id: int,
    use_staging: bool = True,
    feedback: dict = None,
) -> tuple[dict, dict, dict, list[dict], dict]:
    """
    Phase 2: Run pipeline stages 1-3 + dream engine + Hebbian + slot conflicts.

    v6 "Safe Sleep": When use_staging=True, PG writes from stage3 are intercepted
    and routed to a StagingBuffer. Held-out validation runs on val split before
    staging files are written. Nothing touches live PG until adopt().

    Returns ``(clusters, rem_results, stats, dream_walk_edges, staging_info)``.
    """
    # ── v6: Initialize budget and split ──────────────────────────────────────
    budget = EditBudget()
    budget.start()
    splits = split_memories(memories)
    ss = split_stats(memories)
    log.info(f"  📊 Split: train={ss['train']} val={ss['val']} test={ss['test']}")

    # Stage 3 interceptors setup
    staging_buffer = StagingBuffer() if use_staging else None
    staging_info: dict = {
        "use_staging": use_staging,
        "split": ss,
        "budget": {},
        "validation": {},
        "staging_dir": "",
    }

    clusters: dict = {}
    if "1" in stages:
        clusters = stage1_shallow_sleep(memories)

    rem_results: dict = {}
    if "2" in stages:
        rem_results = stage2_rem(clusters)

    stats: dict = {}
    if "3" in stages:
        # Install interceptors if staging is active
        if use_staging and staging_buffer is not None:
            _install_staging_interceptors(staging_buffer, budget)

        try:
            stats = stage3_deep_sleep(
                rem_results,
                dream_run_id,
                dry_run,
                total_memories=len(memories),
                total_clusters=len(clusters),
            )
        finally:
            # Always remove interceptors, even if stage3 fails
            if use_staging:
                _remove_staging_interceptors()

    # REM dream walk (Neo4j random walk v2)
    dream_walk_edges: list[dict] = []
    if "2" in stages and not dry_run:
        # P11: reuse cluster entities from stage2_rem instead of re-extracting
        cluster_entities_cache = rem_results.get("cluster_entities", {})
        all_cluster_entities: list[str] = []
        if cluster_entities_cache:
            for ents in cluster_entities_cache.values():
                all_cluster_entities.extend(ents[:3])  # take top 3 per cluster
        else:
            # Fallback: extract if stage2 didn't run or cache missing
            for _ck, group in clusters.items():
                texts = [m["text"][:200] for m in group]
                ents = extract_entities_with_fallback(texts, max_entities=3)
                all_cluster_entities.extend(ents)

        dream_walk_edges = rem_dream_walk(cluster_entities=all_cluster_entities)
        if dream_walk_edges:
            dream_walk_edges = llm_boost_relations(
                dream_walk_edges, clusters, max_boost=10
            )
            written = write_relations_to_neo4j(dream_walk_edges)
            stats["dream_walk"] = written
            conn_rl = sqlite3.connect(str(DREAM_DB), timeout=5.0)
            for e in dream_walk_edges:
                conn_rl.execute(
                    "INSERT INTO relation_log "
                    "(dream_run_id, source_entity, target_entity, relation_type, confidence, method) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        dream_run_id,
                        e["source"],
                        e["target"],
                        "DREAM_WALK",
                        e["confidence"],
                        "rem_dream_walk",
                    ),
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
            conn_sc = sqlite3.connect(str(DREAM_DB), timeout=5.0)
            for c in slot_conflicts:
                conn_sc.execute(
                    "INSERT INTO contradiction_log "
                    "(dream_run_id, mem1_id, mem2_id, marker, contradiction_type, llm_explanation, verified) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        dream_run_id,
                        c["mem1_id"],
                        c["mem2_id"],
                        f"slot_conflict(sim={c['slot_similarity']:.2f},diff={c['value_diff']:.2f})",
                        "SLOT_CONFLICT",
                        f"同槽不同值: 槽相似度={c['slot_similarity']:.2f}, 值差异={c['value_diff']:.2f}",
                        0,
                    ),
                )
            conn_sc.commit()
            conn_sc.close()
            stats["slot_conflicts"] = len(slot_conflicts)
            rem_results["slot_conflicts_list"] = slot_conflicts

    # SHMR: Self-Harmonizing Memory Reasoning (from Mnemosyne)
    if "3" in stages:
        from dream_cycle.shmr import run_shmr
        shmr_stats = run_shmr(memories, dream_run_id, dry_run=dry_run)
        stats["shmr_beliefs"] = shmr_stats.get("beliefs_created", 0)
        stats["shmr_dampened"] = shmr_stats.get("contradictions_dampened", 0)

        # P1-2: Contrastive Reflection (SkillOpt-inspired)
        # Run after SHMR — uses session outcomes from feedback detection
        if feedback and feedback.get("session_outcomes"):
            contrastive_stats = contrastive_beliefs(
                memories,
                feedback["session_outcomes"],
                dream_run_id,
                dry_run=dry_run,
            )
            stats["contrastive_clusters"] = contrastive_stats.get("contrastive_clusters", 0)
            stats["contrastive_factors"] = contrastive_stats.get("factors_extracted", 0)
            stats["contrastive_actionable"] = contrastive_stats.get("actionable_rules", 0)

    # Three-tier degradation (from Mnemosyne BEAM)
    if "3" in stages and not dry_run:
        from dream_cycle.stage3 import degrade_tiers
        tier_stats = degrade_tiers(dry_run=dry_run)
        stats["tier1_to_tier2"] = tier_stats.get("tier1_to_tier2", 0)
        stats["tier2_to_tier3"] = tier_stats.get("tier2_to_tier3", 0)

    # ── v6: Validation + Staging ─────────────────────────────────────────────
    if use_staging and staging_buffer is not None and not dry_run:
        val_mems = splits.get("val", [])

        # Held-out validation: check if proposed removals degrade val search quality
        validation_result = None
        if val_mems and staging_buffer.removed_ids:
            validation_result = quick_validate(
                val_mems,
                staging_buffer.archived_ids,
                staging_buffer.merged_ids,
                memories,
            )
            staging_info["validation"] = {
                "accepted": validation_result.accepted,
                "hard_score": validation_result.hard_score,
                "soft_score": validation_result.soft_score,
                "n_val_queries": validation_result.n_val_queries,
                "n_improved": validation_result.n_improved,
                "n_same": validation_result.n_same,
                "n_degraded": validation_result.n_degraded,
                "reason": validation_result.reason,
            }
            gate_icon = "✅" if validation_result.accepted else "❌"
            log.info(f"  {gate_icon} Validation: {validation_result.reason}")
        else:
            log.info("  ℹ️ Validation skipped (no val memories or no removals)")

        # Write staging files (always, even if validation fails — for review)
        staging_result = staging_buffer.write_staging(
            dream_run_id,
            validation_result=validation_result,
            budget_summary=budget.summary(),
            split_stats=ss,
        )
        staging_info["staging_dir"] = staging_result.staging_dir
        staging_info["n_proposals"] = staging_result.n_proposals

        gate_icon = "✅" if staging_result.validation_accepted else "❌"
        log.info(f"  📋 Staged {staging_result.n_proposals} proposals → {staging_result.staging_dir}")
        log.info(f"  {gate_icon} Gate: {staging_result.validation_reason}")
        log.info(f"  💰 Budget: {budget.summary()}")
        if budget.skipped:
            log.info(f"  {budget.skipped_summary()}")

    elif use_staging and staging_buffer is not None and dry_run:
        # Dry-run: just log what would be staged
        s = staging_buffer.stats()
        staging_info["dry_run_proposals"] = s
        log.info(f"  📋 Dry-run: {s['total_proposals']} proposals would be staged")

    staging_info["budget"] = budget.summary()

    return clusters, rem_results, stats, dream_walk_edges, staging_info


def _finalize_run(
    dream_run_id: int,
    memories: list[dict],
    clusters: dict,
    rem_results: dict,
    stats: dict,
    dry_run: bool,
    start_time: datetime,
    staging_info: dict = None,
) -> dict:
    """
    Phase 3: Update manifest, record results, return summary.
    """
    # Update manifest (incremental tracking)
    if memories and not dry_run:
        update_manifest(memories, dream_run_id)
        log.info(f"  📋 Manifest 已更新: {len(memories)} 条标记为已处理")

        # Atomic claim: prevent concurrent dream cycles from re-processing
        memory_ids = [m["id"] for m in memories]
        claimed = claim_memories(memory_ids, dream_run_id)
        if len(claimed) < len(memory_ids):
            log.info(f"  🔒 Atomic claim: {len(claimed)}/{len(memory_ids)} claimed")

        archived_ids = [
            item["remove"]["id"] for item in rem_results.get("dedup_candidates", [])
        ]
        if archived_ids:
            mark_manifest_archived(archived_ids)

    # Record in dream_runs
    end_time = datetime.now(HKT)
    dream_conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
    dream_conn.execute(
        """        UPDATE dream_runs SET
            finished_at = ?,
            stage1_clusters = ?,
            stage2_boosted = ?,
            stage3_deduped = ?,
            stage3_inferred = ?,
            stage3_decayed = ?,
            stage3_vault_suggestions = ?,
            summary = ?
        WHERE id = ?
    """,
        (
            end_time.isoformat(),
            len(clusters),
            len(rem_results.get("boosted", [])),
            stats.get("deduped", 0),
            stats.get("inferred", 0),
            stats.get("decayed", 0),
            stats.get("vault_suggestions", 0),
            json.dumps(
                {
                    "memories_scanned": len(memories),
                    "clusters": len(clusters),
                    "dedup_candidates": len(rem_results.get("dedup_candidates", [])),
                    "merge_candidates": len(rem_results.get("merge_candidates", [])),
                    "vault_candidates": len(rem_results.get("vault_candidates", [])),
                    "decay_candidates": len(rem_results.get("decay_candidates", [])),
                },
                ensure_ascii=False,
            ),
            dream_run_id,
        ),
    )
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

    # v6: Add staging info to result
    if staging_info:
        result["staging"] = staging_info
        si = staging_info
        if si.get("staging_dir"):
            val = si.get("validation", {})
            gate_icon = "✅" if val.get("accepted", True) else "❌"
            result["status"] = "staged"
            log.info(
                f"🌅 梦循环完成 (v6 Safe Sleep) — {result['duration_seconds']:.1f}s | "
                f"📋 {si.get('n_proposals', 0)} proposals staged → {si['staging_dir']} | "
                f"{gate_icon} Gate: {val.get('reason', 'n/a')}"
            )
        else:
            log.info(f"🌅 梦循环完成 — {result['duration_seconds']:.1f}s")
    else:
        log.info(f"🌅 梦循环完成 — {result['duration_seconds']:.1f}s")

    return result


def run_dream_cycle(
    hours: int = 48, dry_run: bool = False, stages: str = "123"
) -> dict:
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

        memories, sessions, signals, feedback = prepared
        if not memories and not sessions:
            log.warning("⚠️ 没有新记忆或session, 跳过梦循环")
            _skip_run(dream_run_id, "no_memories")
            _release_lock()
            return {"status": "skipped", "reason": "no_memories"}

        # Phase 2: Execute stages (with staging interceptors)
        clusters, rem_results, stats, dream_walk_edges, staging_info = _execute_stages(
            memories,
            stages,
            dry_run,
            dream_run_id,
            use_staging=not dry_run,  # staging active unless dry-run
            feedback=feedback,        # P1-1: feedback outcomes for contrastive reflection
        )

        # Phase 3: Finalize
        result = _finalize_run(
            dream_run_id,
            memories,
            clusters,
            rem_results,
            stats,
            dry_run,
            start_time,
            staging_info=staging_info,
        )
        _release_lock()
        return result

    except Exception as e:
        log.error(f"❌ 梦循环失败: {e}", exc_info=True)
        dream_conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
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
    conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
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
                lines.append(
                    f"  • {b['score']:.2f} — {text_preview}... ({b.get('reason', '')})"
                )

        # P6: Top3 Vault 候选
        vault = rem_results.get("vault_candidates", [])
        if vault:
            lines.append("")
            lines.append("📝 **Top3 Vault 候选**")
            for v in sorted(
                vault, key=lambda x: x.get("priority", "normal"), reverse=True
            )[:3]:
                kw = ", ".join(v.get("keywords", [])[:3])
                priority = v.get("priority", "normal")
                gate = v.get("promotion_pass", "")
                lines.append(
                    f"  • [{priority}] {kw} (score={v['best_score']:.2f}, gate={gate})"
                )

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

                encoded = urllib.parse.quote(report, safe="")
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
    conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)

    # 1. 处理 pending 建议
    pending = conn.execute(
        """
        SELECT id, entity, category, frequency, reason, dream_run_id
        FROM vault_suggestion WHERE status = 'pending'
        ORDER BY frequency DESC
        LIMIT ?
    """,
        (max_review,),
    ).fetchall()

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
        slug = entity.lower().replace(" ", "-").replace("|", "-")[:50]
        cat_map = {
            "markets": "markets",
            "investment": "markets",
            "projects": "projects",
            "technology": "concepts",
            "concepts": "concepts",
        }
        vault_cat = cat_map.get(category, "concepts")
        filepath = VAULT_DIR / vault_cat / f"{slug}.md"

        if filepath.exists():
            # 已有页面 → 标记 reviewed
            conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
            conn.execute(
                "UPDATE vault_suggestion SET status = 'reviewed' WHERE id = ?", (sid,)
            )
            conn.commit()
            conn.close()
            reviewed.append(
                {"entity": entity, "action": "already_exists", "status": "reviewed"}
            )
            continue

        # 频率太低(<2) → reject
        if freq < 2:
            conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
            conn.execute(
                "UPDATE vault_suggestion SET status = 'rejected' WHERE id = ?", (sid,)
            )
            conn.commit()
            conn.close()
            reviewed.append(
                {"entity": entity, "action": "rejected_low_freq", "status": "rejected"}
            )
            continue

        # 频率>=2 → 创建 stub (LLM充实概述)
        keywords = [entity]
        sample = f"{entity} (出现 {freq} 次, {reason})"
        vault_path = create_vault_stub(
            entity, category, keywords, sample, sample_age_days=None
        )
        if vault_path:
            conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
            conn.execute(
                "UPDATE vault_suggestion SET status = 'auto_created' WHERE id = ?",
                (sid,),
            )
            conn.commit()
            conn.close()
            reviewed.append(
                {
                    "entity": entity,
                    "action": "stub_created",
                    "path": vault_path,
                    "status": "auto_created",
                }
            )
        else:
            reviewed.append(
                {"entity": entity, "action": "stub_failed", "status": "pending"}
            )

    # Process auto_created: 检查内容是否需要充实
    for ac in auto_created:
        sid, entity, category = ac
        slug = entity.lower().replace(" ", "-").replace("|", "-")[:50]
        cat_map = {
            "markets": "markets",
            "investment": "markets",
            "projects": "projects",
            "technology": "concepts",
            "concepts": "concepts",
        }
        vault_cat = cat_map.get(category, "concepts")
        filepath = VAULT_DIR / vault_cat / f"{slug}.md"

        if filepath.exists():
            # 检查内容长度: <500字 = 还是stub → 标记需要充实
            content = filepath.read_text()
            word_count = len(content.split())
            if word_count < 100:
                reviewed.append(
                    {
                        "entity": entity,
                        "action": "needs_enrichment",
                        "words": word_count,
                    }
                )
            else:
                # 内容已充实 → 标记 reviewed
                conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
                conn.execute(
                    "UPDATE vault_suggestion SET status = 'reviewed' WHERE id = ?",
                    (sid,),
                )
                conn.commit()
                conn.close()
                reviewed.append(
                    {
                        "entity": entity,
                        "action": "enriched",
                        "words": word_count,
                        "status": "reviewed",
                    }
                )

    log.info(f"  📝 P9 Vault Review: {len(reviewed)} 条处理")
    for r in reviewed[:5]:
        log.info(f"    {r['entity']}: {r['action']}")

    return reviewed


# ─── P10: 批量积压处理 ─────────────────────────────────────────────────


def batch_resolve_all_conflicts(max_per_run: int = 50) -> dict:
    """
    批量处理所有 pending 矛盾 — 每次最多处理 max_per_run 个

    从 contradiction_log 读取 pending 记录，获取记忆文本，调用 LLM 分类
    优化: 批量预取所有记忆文本和时间戳 (2 次 docker exec), 而非逐条查询
    """
    conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
    pending = conn.execute(
        """
        SELECT id, mem1_id, mem2_id, marker
        FROM contradiction_log WHERE resolution = 'pending'
        ORDER BY id
        LIMIT ?
    """,
        (max_per_run,),
    ).fetchall()
    total_pending = conn.execute(
        "SELECT COUNT(*) FROM contradiction_log WHERE resolution = 'pending'"
    ).fetchone()[0]
    conn.close()

    if not pending:
        log.info("✅ 无 pending 矛盾需要处理")
        return {"resolved": 0, "remaining": 0}

    log.info(f"🔍 批量矛盾处理: {len(pending)}/{total_pending} pending")

    # 批量预取: 收集所有 memory IDs, 一次性查出文本和时间戳
    all_mem_ids = set()
    for _, mem1_id, mem2_id, _ in pending:
        all_mem_ids.add(mem1_id)
        all_mem_ids.add(mem2_id)

    id_values = ",".join(f"'{mid}'" for mid in all_mem_ids)
    # 1 次 docker exec: 取所有文本
    text_rows = pg_query(f"""
        SELECT id::text, LEFT(payload->>'data', 300), payload->>'created_at'
        FROM mem0 WHERE id::text IN ({id_values})
    """)

    # Build lookup maps
    text_map: dict[str, str] = {}
    time_map: dict[str, str] = {}
    for r in text_rows:
        if len(r) >= 3:
            text_map[r[0]] = r[1] or ""
            time_map[r[0]] = r[2] or ""

    resolved_count = 0
    failed = 0
    superseded = 0
    extended = 0
    false_pos = 0

    for row in pending:
        cid, mem1_id, mem2_id, marker = row

        # 从预取 map 查文本 (0 次 docker exec)
        text1 = text_map.get(mem1_id, "")
        text2 = text_map.get(mem2_id, "")

        if not text1 or not text2:
            # 记忆已被删除 → 标记为 FALSE_POSITIVE
            conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
            conn.execute(
                "UPDATE contradiction_log SET resolution = 'false_positive', llm_explanation = ? WHERE id = ?",
                ("memory_deleted", cid),
            )
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
        explanation = v.get("explanation", "")[:200].replace('"', "").replace("'", "")

        conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)

        if ctype == "SUPERSEDE":
            # 从预取 map 查时间戳 (0 次 docker exec)
            t1 = time_map.get(mem1_id, "")
            t2 = time_map.get(mem2_id, "")

            older_id = mem1_id if t1 <= t2 else mem2_id
            newer_id = mem2_id if older_id == mem1_id else mem1_id

            pg_query(
                f"""UPDATE mem0 SET payload = payload || '{{"archived": true, "archived_reason": "slot_supersede", "superseded_by": "{newer_id}"}}' WHERE id::text = '{older_id}'"""
            )
            mark_manifest_archived([older_id])
            superseded += 1
            log.info(f"  ✅ SUPERSEDE: 归档 {older_id[:8]}")

        elif ctype == "EXTEND":
            extended += 1
            log.info(f"  🔗 EXTEND: {mem1_id[:8]} ↔ {mem2_id[:8]}")

        else:  # FALSE_POSITIVE
            false_pos += 1

        conn.execute(
            """
            UPDATE contradiction_log SET contradiction_type = ?, resolution = ?,
                   llm_explanation = ?, verified = 1 WHERE id = ?
        """,
            (f"SLOT_CONFLICT->{ctype}", ctype, explanation, cid),
        )
        conn.commit()
        conn.close()
        resolved_count += 1

    remaining = total_pending - resolved_count
    log.info(
        f"📊 批量矛盾完成: {resolved_count} resolved (S:{superseded} E:{extended} FP:{false_pos} fail:{failed}), {remaining} remaining"
    )
    return {
        "resolved": resolved_count,
        "remaining": remaining,
        "superseded": superseded,
        "extended": extended,
        "false_positive": false_pos,
        "failed": failed,
    }


def batch_review_all_vault(max_per_run: int = 100) -> dict:
    """
    批量处理所有 pending vault 建议

    freq>=2 → 创建 stub; freq<2 → reject; 已有页面 → reviewed
    """
    conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
    pending = conn.execute(
        """
        SELECT id, entity, category, frequency, reason
        FROM vault_suggestion WHERE status = 'pending'
        ORDER BY frequency DESC
        LIMIT ?
    """,
        (max_per_run,),
    ).fetchall()
    total_pending = conn.execute(
        "SELECT COUNT(*) FROM vault_suggestion WHERE status='pending'"
    ).fetchone()[0]
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
        slug = entity.lower().replace(" ", "-").replace("|", "-")[:50]
        cat_map = {
            "markets": "markets",
            "investment": "markets",
            "projects": "projects",
            "technology": "concepts",
            "concepts": "concepts",
        }
        vault_cat = cat_map.get(category, "concepts")
        filepath = VAULT_DIR / vault_cat / f"{slug}.md"

        conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)

        if filepath.exists():
            conn.execute(
                "UPDATE vault_suggestion SET status = 'reviewed' WHERE id = ?", (sid,)
            )
            conn.commit()
            conn.close()
            exists += 1
            continue

        if freq < 2:
            conn.execute(
                "UPDATE vault_suggestion SET status = 'rejected' WHERE id = ?", (sid,)
            )
            conn.commit()
            conn.close()
            rejected += 1
            continue

        # 创建 stub
        keywords = [entity]
        sample = f"{entity} (出现 {freq} 次, {reason or ''})"
        vault_path = create_vault_stub(
            entity, category, keywords, sample, sample_age_days=None
        )
        if vault_path:
            conn.execute(
                "UPDATE vault_suggestion SET status = 'auto_created' WHERE id = ?",
                (sid,),
            )
            conn.commit()
            conn.close()
            created += 1
            log.info(f"  📄 创建: {entity} → {vault_path}")
        else:
            conn.close()
            failed += 1

    remaining = total_pending - (created + rejected + exists + failed)
    log.info(
        f"📊 批量 Vault 完成: {created} created, {rejected} rejected, {exists} exists, {failed} failed, {remaining} remaining"
    )
    return {
        "created": created,
        "rejected": rejected,
        "exists": exists,
        "failed": failed,
        "remaining": remaining,
    }


# ─── CLI ───────────────────────────────────────────────────────────────
