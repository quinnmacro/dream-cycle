"""
Dream Cycle — Session mining — recent sessions, signal scanning, feedback detection
"""



__all__ = [
    "mine_recent_sessions",
    "scan_session_signals",
    "generate_session_digest",
    "detect_feedback_signals",
]

import json
import sqlite3
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from dream_cycle.config import (
    STATE_DB, HKT,
    SIGNAL_CORRECTIONS, SIGNAL_PREFERENCES, SIGNAL_DECISIONS, SIGNAL_PATTERNS,
    SIGNAL_IDENTITY,
    FEEDBACK_POSITIVE, FEEDBACK_NEGATIVE,
    FEEDBACK_POS_IMPORTANCE_BOOST, FEEDBACK_NEG_IMPORTANCE_BOOST,
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
        return {"corrections": [], "preferences": [], "decisions": [], "patterns": [], "identity": []}
    
    cutoff = time.time() - hours * 3600
    signals = {"corrections": [], "preferences": [], "decisions": [], "patterns": [], "identity": []}
    
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
            
            # Identity signals — lowest priority, checked after patterns
            if not matched_type:
                for kw in SIGNAL_IDENTITY:
                    if kw.lower() in text_lower:
                        matched_type = "identity"
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


# ─── P1-1: Feedback Signal Detection (SkillOpt-inspired) ─────────────

def _detect_feedback(text: str) -> list[str]:
    """Detect positive/negative feedback signals in user text.

    Returns list like ['neg:wrong', 'pos:thanks'].
    Ported from SkillOpt-Sleep harvest.py _detect_feedback().
    """
    signals = []
    text_lower = text.lower()
    for kw in FEEDBACK_NEGATIVE:
        if kw.lower() in text_lower:
            signals.append(f"neg:{kw}")
    for kw in FEEDBACK_POSITIVE:
        if kw.lower() in text_lower:
            signals.append(f"pos:{kw}")
    return signals


def _classify_outcome(feedback_signals: list[str], n_user_turns: int) -> str:
    """Infer task outcome from feedback signals.

    Returns: 'success' | 'fail' | 'mixed' | 'unknown'
    """
    has_neg = any(s.startswith("neg:") for s in feedback_signals)
    has_pos = any(s.startswith("pos:") for s in feedback_signals)

    if has_neg and not has_pos:
        return "fail"
    if has_pos and not has_neg:
        return "success"
    if n_user_turns >= 3 and not has_pos and not has_neg:
        return "mixed"   # lots of turns without resolution = probably struggling
    return "unknown"


def detect_feedback_signals(hours: int = 72) -> dict:
    """Scan recent sessions for feedback signals and classify outcomes.

    Ported from SkillOpt-Sleep: detects user satisfaction/dissatisfaction
    and injects importance modifiers for nearby memories.

    Returns: {
        "positive": [{"text", "session_id", "session_title", "keyword", "timestamp"}],
        "negative": [...],
        "session_outcomes": {"session_id": "success"|"fail"|"mixed"|"unknown"},
        "importance_modifiers": {"memory_text_prefix": float_boost},
    }
    """
    if not STATE_DB.exists():
        return {"positive": [], "negative": [], "session_outcomes": {}, "importance_modifiers": {}}

    cutoff = time.time() - hours * 3600
    positive = []
    negative = []
    session_feedback: dict[str, list[str]] = {}  # session_id → all feedback signals

    try:
        conn = sqlite3.connect(str(STATE_DB))
        cursor = conn.execute("""
            SELECT m.content, m.session_id, m.timestamp, s.title
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE m.role = 'user'
            AND m.timestamp > ?
            AND m.content IS NOT NULL
            AND length(m.content) > 5
            AND length(m.content) < 300
            AND m.content NOT LIKE '[IMPORTANT%'
            AND m.content NOT LIKE 'Tool results%'
            AND m.content NOT LIKE '%SYSTEM:%'
            ORDER BY m.timestamp DESC
            LIMIT 3000
        """, (cutoff,))

        for row in cursor:
            text, session_id, ts, title = row
            text_stripped = text.strip()
            if len(text_stripped) < 8:
                continue

            signals = _detect_feedback(text_stripped)
            if not signals:
                continue

            session_feedback.setdefault(session_id, []).extend(signals)

            for sig in signals:
                entry = {
                    "text": text_stripped[:200],
                    "session_id": session_id,
                    "session_title": title or "untitled",
                    "keyword": sig.split(":", 1)[1] if ":" in sig else sig,
                    "timestamp": ts,
                }
                if sig.startswith("pos:"):
                    positive.append(entry)
                elif sig.startswith("neg:"):
                    negative.append(entry)

        conn.close()
    except Exception as e:
        log.warning(f"⚠️ feedback signal scan failed: {e}")
        return {"positive": [], "negative": [], "session_outcomes": {}, "importance_modifiers": {}}

    # Classify per-session outcomes
    # Count user turns per session
    session_turns: dict[str, int] = {}
    try:
        conn2 = sqlite3.connect(str(STATE_DB))
        for sid in session_feedback:
            row = conn2.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id=? AND role='user' AND timestamp>?",
                (sid, cutoff)
            ).fetchone()
            session_turns[sid] = row[0] if row else 0
        conn2.close()
    except Exception:
        pass

    session_outcomes = {}
    for sid, sigs in session_feedback.items():
        outcome = _classify_outcome(sigs, session_turns.get(sid, 0))
        session_outcomes[sid] = outcome

    # Build importance modifiers: for memories whose text overlaps with
    # feedback sessions, boost their importance score
    importance_modifiers: dict[str, float] = {}
    for entry in positive[:50]:
        prefix = entry["text"][:30].strip()
        if prefix:
            importance_modifiers[prefix] = FEEDBACK_POS_IMPORTANCE_BOOST
    for entry in negative[:50]:
        prefix = entry["text"][:30].strip()
        if prefix:
            importance_modifiers[prefix] = FEEDBACK_NEG_IMPORTANCE_BOOST

    log.info(
        f"  📡 Feedback signals: {len(positive)} positive, {len(negative)} negative "
        f"across {len(session_outcomes)} sessions"
    )

    return {
        "positive": positive,
        "negative": negative,
        "session_outcomes": session_outcomes,
        "importance_modifiers": importance_modifiers,
    }


# ─── P8: 健康仪表盘 ──────────────────────────────────────────────────────

