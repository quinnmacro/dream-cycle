"""
Dream Cycle — CLI entry point — argparse, command routing
"""

import sys
import json
import sqlite3
import argparse
import logging
from datetime import datetime, timezone, timedelta
from dream_cycle.config import DREAM_DB, HKT, log
from dream_cycle.orchestrator import (
    run_dream_cycle, _cleanup_zombie_runs,
    resolve_slot_conflicts, format_report, send_telegram_report,
    review_vault_suggestions, batch_resolve_all_conflicts, batch_review_all_vault,
    cmd_adopt,
)
from dream_cycle.staging import staging_status, latest_staging
from dream_cycle.health import show_health_dashboard, check_dream_trigger, online_dedup_check
from dream_cycle.session import mine_recent_sessions, generate_session_digest

def main():
    parser = argparse.ArgumentParser(description="Hermes Dream Cycle — 记忆自主整理")
    parser.add_argument("--hours", type=int, default=48, help="回溯小时数")
    parser.add_argument("--dry-run", action="store_true", help="只分析不执行")
    parser.add_argument("--stages", default="123", help="执行阶段 (1/2/3/123)")
    parser.add_argument("--report", action="store_true", help="只输出报告")
    parser.add_argument("--history", type=int, default=0, help="查看最近N次梦循环记录")
    parser.add_argument("--notify", action="store_true", help="发送 Telegram 报告")
    parser.add_argument("--dedup-check", type=str, help="在线去冗余检查: 传入文本，返回ADD/SKIP/MERGE建议")
    parser.add_argument("--manifest-stats", action="store_true", help="查看manifest统计")
    parser.add_argument("--trigger-check", action="store_true", help="检查是否应该触发梦循环 (自适应触发)")
    parser.add_argument("--health", action="store_true", help="P8: 显示梦循环健康仪表盘 (7天趋势)")
    parser.add_argument("--vault-review", action="store_true", help="P9: 处理 pending vault suggestion")
    parser.add_argument("--resolve-all", action="store_true", help="P10: 批量处理所有 pending 矛盾")
    parser.add_argument("--vault-all", action="store_true", help="P10: 批量处理所有 pending vault 建议")
    parser.add_argument("--backlog", action="store_true", help="P10: 一次性清理所有积压(矛盾+vault)")
    parser.add_argument("--auto", action="store_true", help="自适应模式: 先检查触发条件，满足才执行")
    # v6 Safe Sleep
    parser.add_argument("--adopt", nargs="?", const="latest", default=None,
                        help="采纳 staging 的修改 → 写入 PG (默认最新 staging, 可指定目录)")
    parser.add_argument("--staging-status", action="store_true",
                        help="查看 staging 目录状态 (pending/adopted)")
    args = parser.parse_args()

    # v6: Adopt command
    if args.adopt is not None:
        staging_dir = args.adopt if args.adopt != "latest" else ""
        result = cmd_adopt(staging_dir)
        if "error" in result:
            print(f"❌ Adopt failed: {result['error']}")
        else:
            print(f"✅ Adopt: {result.get('applied', 0)}/{result.get('total', 0)} proposals applied")
            if result.get("errors"):
                print(f"⚠️  {len(result['errors'])} errors:")
                for e in result["errors"][:5]:
                    print(f"   • {e['memory_id'][:8]}... {e['op']}: {e['error'][:80]}")
        return result

    # v6: Staging status
    if args.staging_status:
        status = staging_status()
        print(f"📋 Staging 状态:")
        print(f"  总计: {status['staging_dirs']} 个目录")
        print(f"  待采纳: {status['pending']}")
        print(f"  已采纳: {status['adopted']}")
        if status.get("latest"):
            lat = status["latest"]
            print(f"  最新: {lat['dir']} ({lat['n_proposals']} proposals, adopted={lat['adopted']})")
        return status
    
    if args.dedup_check:
        result = online_dedup_check(args.dedup_check)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    
    if args.manifest_stats:
        conn = sqlite3.connect(str(DREAM_DB))
        total = conn.execute("SELECT COUNT(*) FROM processed_manifest").fetchone()[0]
        active = conn.execute("SELECT COUNT(*) FROM processed_manifest WHERE status='active'").fetchone()[0]
        archived = conn.execute("SELECT COUNT(*) FROM processed_manifest WHERE status='archived'").fetchone()[0]
        avg_count = conn.execute("SELECT AVG(process_count) FROM processed_manifest").fetchone()[0]
        print(f"📋 Manifest 统计:")
        print(f"  总计: {total}")
        print(f"  活跃: {active}")
        print(f"  已归档: {archived}")
        print(f"  平均处理次数: {avg_count:.1f}")
        conn.close()
        return
    
    if args.trigger_check:
        result = check_dream_trigger()
        status = "🟢 触发" if result["should_trigger"] else "⚪ 不触发"
        print(f"🔍 自适应触发检查: {status}")
        print(f"  紧急度: {result['urgency']}")
        print(f"  新记忆数: {result['new_memory_count']}")
        if result["reasons"]:
            print(f"  触发原因: {', '.join(result['reasons'])}")
        else:
            print(f"  无触发条件满足")
        return result
    
    # P8: 健康仪表盘
    if args.health:
        show_health_dashboard()
        return
    
    # P9: Vault Review
    if args.vault_review:
        results = review_vault_suggestions()
        print(f"📝 Vault Review: {len(results)} 条处理")
        for r in results:
            print(f"  {r['entity']}: {r['action']} → {r.get('status', '?')}")
        return results
    
    # P10: 批量矛盾处理
    if args.resolve_all:
        result = batch_resolve_all_conflicts(max_per_run=50)
        print(f"🔍 批量矛盾处理: {result}")
        return result
    
    # P10: 批量 Vault 审核
    if args.vault_all:
        result = batch_review_all_vault(max_per_run=100)
        print(f"📝 批量 Vault 审核: {result}")
        return result
    
    # P10: 一键清理积压
    if args.backlog:
        print("🧹 开始清理积压...")
        cr = batch_resolve_all_conflicts(max_per_run=50)
        vr = batch_review_all_vault(max_per_run=100)
        print(f"\n📊 积压清理完成:")
        print(f"  矛盾: {cr}")
        print(f"  Vault: {vr}")
        return {"conflicts": cr, "vault": vr}
    
    # 自适应模式: 先检查触发条件
    if args.auto:
        trigger = check_dream_trigger()
        if not trigger["should_trigger"]:
            print(f"💤 自适应模式: 无需触发 ({', '.join(trigger['reasons']) if trigger['reasons'] else '无触发条件'})")
            return {"status": "skipped", "reason": "adaptive_no_trigger"}
        print(f"🌙 自适应模式: 触发! 原因: {', '.join(trigger['reasons'])}, 紧急度: {trigger['urgency']}")
    
    if args.history > 0:
        conn = sqlite3.connect(str(DREAM_DB))
        rows = conn.execute("""
            SELECT id, started_at, stage1_clusters, stage3_deduped, stage3_inferred,
                   stage3_decayed, stage3_vault_suggestions, summary, error
            FROM dream_runs ORDER BY id DESC LIMIT ?
        """, (args.history,)).fetchall()
        for r in rows:
            status = "❌" if r[8] else "✅"
            print(f"#{r[0]} [{status}] {r[1]} clusters={r[2]} dedup={r[3]} infer={r[4]} decay={r[5]} vault={r[6]}")
        conn.close()
        return
    
    result = run_dream_cycle(
        hours=args.hours,
        dry_run=args.dry_run,
        stages=args.stages,
    )
    
    report = format_report(result, rem_results=result.get("rem_results"), stats=result.get("stats"))
    print(report)
    
    # 发送 Telegram 报告
    if args.notify:
        # 附加 session digest
        sessions = mine_recent_sessions(hours=args.hours)
        if sessions:
            session_report = generate_session_digest(sessions)
            report = report + "\n\n" + session_report
        send_telegram_report(report)
    
    if args.report:
        return result
    
    return result


if __name__ == "__main__":
    main()
