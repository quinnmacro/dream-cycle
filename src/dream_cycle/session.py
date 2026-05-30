"""
Dream Cycle — Session mining — recent sessions, signal scanning (corrections/preferences/decisions/patterns)
"""

import json
import sqlite3
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from dream_cycle.config import (
    STATE_DB, HKT,
    SIGNAL_CORRECTIONS, SIGNAL_PREFERENCES, SIGNAL_DECISIONS, SIGNAL_PATTERNS,
    log,
)

def mine_recent_sessions(hours: int = 24) -> list[dict]:
    """
    从 state.db 提取近期会话主题, 作为梦循环的额外输入
    
    返回:
    - 高频话题
    - 未被 mem0 捕获的知识
    - 跨 session 反复出现的模式
    """
    if not STATE_DB.exists():
        return []
    
    cutoff = time.time() - hours * 3600
    conn = sqlite3.connect(str(STATE_DB))
    
    # 最近 session 的标题和统计
    sessions = conn.execute("""
        SELECT id, title, message_count, tool_call_count, started_at, estimated_cost_usd
        FROM sessions
        WHERE started_at > ? AND title IS NOT NULL
        ORDER BY started_at DESC
    """, (cutoff,)).fetchall()
    
    conn.close()
    
    topics = []
    for s in sessions:
        sid, title, msg_count, tool_count, started, cost = s
        if title and len(title) > 5:
            topics.append({
                "session_id": sid,
                "title": title,
                "message_count": msg_count,
                "tool_call_count": tool_count,
                "cost": cost or 0,
            })
    
    return topics



def scan_session_signals(hours: int = 72) -> dict:
    """
    Session Transcript Scanning (from Anthropic autoDream / dream-skill).
    
    Scan state.db user messages for 4 signal types:
    - corrections: user corrected the agent (highest priority)
    - preferences: explicit preference statements
    - decisions: architectural/tool choices
    - patterns: recurring complaints or repeated requests
    
    Returns: {"corrections": [...], "preferences": [...], "decisions": [...], "patterns": [...]}
    Each item: {"text": str, "session_id": str, "timestamp": float, "signal_type": str}
    """
    if not STATE_DB.exists():
        return {"corrections": [], "preferences": [], "decisions": [], "patterns": []}
    
    cutoff = time.time() - hours * 3600
    signals = {"corrections": [], "preferences": [], "decisions": [], "patterns": []}
    
    try:
        conn = sqlite3.connect(str(STATE_DB))
        cursor = conn.execute("""
            SELECT m.content, m.session_id, m.timestamp, s.title
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE m.role = 'user'
            AND m.timestamp > ?
            AND m.content IS NOT NULL
            AND length(m.content) > 10
            AND length(m.content) < 500
            AND m.content NOT LIKE '[IMPORTANT%'
            AND m.content NOT LIKE 'Tool results%'
            AND m.content NOT LIKE '%SYSTEM:%'
            ORDER BY m.timestamp DESC
            LIMIT 2000
        """, (cutoff,))
        
        for row in cursor:
            text, session_id, ts, title = row
            text_stripped = text.strip()
            text_lower = text_stripped.lower()
            
            # Skip very short or system-like messages
            if len(text_stripped) < 15:
                continue
            
            # Check each signal type (priority order)
            matched_type = None
            matched_kw = None
            
            for kw in SIGNAL_CORRECTIONS:
                if kw.lower() in text_lower:
                    matched_type = "corrections"
                    matched_kw = kw
                    break
            
            if not matched_type:
                for kw in SIGNAL_PREFERENCES:
                    if kw.lower() in text_lower:
                        matched_type = "preferences"
                        matched_kw = kw
                        break
            
            if not matched_type:
                for kw in SIGNAL_DECISIONS:
                    if kw.lower() in text_lower:
                        matched_type = "decisions"
                        matched_kw = kw
                        break
            
            if not matched_type:
                for kw in SIGNAL_PATTERNS:
                    if kw.lower() in text_lower:
                        matched_type = "patterns"
                        matched_kw = kw
                        break
            
            if matched_type:
                signals[matched_type].append({
                    "text": text_stripped[:300],
                    "session_id": session_id,
                    "session_title": title or "untitled",
                    "timestamp": ts,
                    "signal_type": matched_type,
                    "trigger_keyword": matched_kw,
                })
        
        conn.close()
    except Exception as e:
        log.warning(f"⚠️ session signal scan failed: {e}")
    
    return signals


def generate_session_digest(sessions: list[dict]) -> str:
    """生成近期 session 摘要"""
    if not sessions:
        return "无近期 session"
    
    # 按 cost 排序 (高 cost = 深度工作)
    top = sorted(sessions, key=lambda x: x["cost"], reverse=True)[:5]
    
    lines = [f"📋 近期 Top {len(top)} Session:"]
    for s in top:
        lines.append(f"  • {s['title'][:50]} ({s['message_count']}msg, ${s['cost']:.3f})")
    
    return "\n".join(lines)


# ─── P8: 健康仪表盘 ──────────────────────────────────────────────────────

