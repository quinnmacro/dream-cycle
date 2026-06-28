"""
Dream Cycle — LLM API calls — DashScope (qwen3.7-max) for merge/verify/entity extraction

v6 P1 upgrades:
  - P1-3: SHA256(prompt) SQLite response cache (24h TTL)
  - P1-4: Dual-backend — call_quick() for mining, call_deep() for consolidation
"""

__all__ = [
    "call_llm",
    "call_quick",
    "call_deep",
    "call_json",
    "llm_merge_memories",
    "llm_verify_contradiction",
    "llm_extract_entities",
]

import hashlib
import json
import sqlite3
import subprocess
import time
from pathlib import Path
from dream_cycle.config import (
    INFINI_BASE_URL,
    INFINI_MODEL,
    INFINI_MODEL_QUICK,
    LLM_CACHE_DB,
    LLM_CACHE_TTL_HOURS,
    log,
)

# P11: cache API key to avoid reading config file on every call
_api_key_cache: str | None = None

# P1-3: LLM response cache — module-level DB connection
_cache_conn: sqlite3.Connection | None = None


def _get_cache_conn() -> sqlite3.Connection:
    """Get or create the LLM cache SQLite connection."""
    global _cache_conn
    if _cache_conn is not None:
        return _cache_conn
    LLM_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LLM_CACHE_DB), timeout=5.0)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_cache (
            prompt_hash TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at REAL NOT NULL,
            hits INTEGER DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_created ON llm_cache(created_at)")
    conn.commit()
    _cache_conn = conn
    return conn


def _cache_get(prompt: str, model: str) -> str | None:
    """Check cache for a response. Returns cached response or None."""
    try:
        conn = _get_cache_conn()
        prompt_hash = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()[:24]
        cutoff = time.time() - LLM_CACHE_TTL_HOURS * 3600
        row = conn.execute(
            "SELECT response FROM llm_cache WHERE prompt_hash=? AND created_at>?",
            (prompt_hash, cutoff),
        ).fetchone()
        if row:
            conn.execute("UPDATE llm_cache SET hits=hits+1 WHERE prompt_hash=?", (prompt_hash,))
            conn.commit()
            return row[0]
    except Exception:
        pass
    return None


def _cache_put(prompt: str, model: str, response: str):
    """Store a response in cache."""
    try:
        conn = _get_cache_conn()
        prompt_hash = hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()[:24]
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (prompt_hash, model, response, created_at, hits) VALUES (?,?,?,?,0)",
            (prompt_hash, model, response, time.time()),
        )
        conn.commit()
    except Exception:
        pass


def _cache_stats() -> dict:
    """Return cache statistics."""
    try:
        conn = _get_cache_conn()
        total = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        cutoff = time.time() - LLM_CACHE_TTL_HOURS * 3600
        active = conn.execute("SELECT COUNT(*) FROM llm_cache WHERE created_at>?", (cutoff,)).fetchone()[0]
        total_hits = conn.execute("SELECT COALESCE(SUM(hits),0) FROM llm_cache").fetchone()[0]
        return {"total": total, "active": active, "total_hits": total_hits}
    except Exception:
        return {"total": 0, "active": 0, "total_hits": 0}


def _get_infini_api_key() -> str:
    """读取 DashScope API key (优先) → Infini AI (fallback), 带缓存"""
    global _api_key_cache
    if _api_key_cache is not None:
        return _api_key_cache

    try:
        import yaml

        with open("/root/.hermes/config.yaml") as f:
            config = yaml.safe_load(f)
        # 优先 DashScope
        key = config.get("credentials", {}).get("dashscope_api_key", "")
        if key:
            _api_key_cache = key
            return key
        # fallback Infini
        key = config.get("credentials", {}).get("infini_api_key", "")
        if key:
            _api_key_cache = key
            return key
    except Exception:
        pass
    try:
        with open("/root/projects/mem0-selfhost/.env") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    key = line.strip().split("=", 1)[1]
                    _api_key_cache = key
                    return key
    except Exception:
        pass
    _api_key_cache = ""
    return ""


def _call_infini(
    prompt: str, max_tokens: int = 300, temperature: float = 0.3,
    system: str = "", model: str = "",
) -> str | None:
    """调用 DashScope API 的通用函数 (with P1-3 cache + P1-4 model routing)"""
    if not model:
        model = INFINI_MODEL

    # P1-3: Check cache first
    cached = _cache_get(prompt, model)
    if cached is not None:
        log.debug(f"  💾 LLM cache hit ({model})")
        return cached

    api_key = _get_infini_api_key()
    if not api_key:
        return None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "enable_thinking": False,  # 不需要推理，节省 tokens 和延迟
        }
    )

    auth_header = f"Authorization: Bearer {api_key}"
    try:
        result = subprocess.run(
            [
                "curl",
                "-s",
                INFINI_BASE_URL + "/chat/completions",
                "-H",
                auth_header,
                "-H",
                "Content-Type: application/json",
                "-d",
                payload,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        resp = json.loads(result.stdout)
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        content = content.strip() if content else None

        # P1-3: Store in cache
        if content:
            _cache_put(prompt, model, content)

        return content
    except Exception as e:
        log.warning(f"⚠️ Infini API 调用失败: {e}")
        return None


def call_llm(prompt: str, max_tokens: int = 300, temperature: float = 0.3, system: str = "") -> str | None:
    """Standard LLM call (uses default model = qwen3.7-max)."""
    return _call_infini(prompt, max_tokens=max_tokens, temperature=temperature, system=system)


def call_quick(prompt: str, max_tokens: int = 200, temperature: float = 0.3, system: str = "") -> str | None:
    """P1-4: Quick/cheap LLM call — uses flash model for mining/simple tasks.

    Use for: session mining, entity extraction, simple classification.
    ~10x cheaper than call_deep().
    """
    return _call_infini(prompt, model=INFINI_MODEL_QUICK, max_tokens=max_tokens, temperature=temperature, system=system)


def call_deep(prompt: str, max_tokens: int = 1024, temperature: float = 0.3, system: str = "") -> str | None:
    """P1-4: Deep/expensive LLM call — uses max model for consolidation/analysis.

    Use for: SHMR harmonization, contrastive reflection, merge, contradiction verification.
    """
    return _call_infini(prompt, model=INFINI_MODEL, max_tokens=max_tokens, temperature=temperature, system=system)


# ─── P2-4: JSON Retry Escalation (SkillOpt-inspired) ─────────────────

_JSON_RETRY_SUFFIX = "\n\nIMPORTANT: Your previous reply was not valid JSON. Output ONLY a valid JSON object/array, no markdown, no explanation."


def call_json(prompt: str, max_tokens: int = 512, temperature: float = 0.1,
              system: str = "", model: str = "") -> dict | list | None:
    """P2-4: LLM call with JSON retry escalation.

    First attempt: normal call.
    If JSON parse fails: retry with escalation suffix appended to prompt.
    Returns parsed JSON (dict or list) or None.
    """
    import re

    if not model:
        model = INFINI_MODEL

    # First attempt
    response = _call_infini(prompt, model=model, max_tokens=max_tokens,
                            temperature=temperature, system=system)
    if not response:
        return None

    parsed = _try_parse_json(response)
    if parsed is not None:
        return parsed

    # Retry with escalation
    log.debug(f"  🔄 JSON retry: first attempt not valid JSON, escalating")
    escalated = prompt + _JSON_RETRY_SUFFIX
    response2 = _call_infini(escalated, model=model, max_tokens=max_tokens,
                             temperature=temperature, system=system)
    if not response2:
        return None

    parsed2 = _try_parse_json(response2)
    if parsed2 is not None:
        return parsed2

    log.debug(f"  ⚠️ JSON retry failed after escalation")
    return None


def _try_parse_json(text: str) -> dict | list | None:
    """Try to parse JSON from LLM response, handling markdown code blocks."""
    import re
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, (dict, list)):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find JSON in the response
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    match = re.search(r"\[[^\]]*\]", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def llm_merge_memories(texts: list[str]) -> str | None:
    """
    用 LLM 将多条近似记忆合并为一条

    使用 qwen3.7-max (DashScope), 独立配额不受 Hermes 影响
    """
    numbered = "\n".join(f"{i + 1}. {t[:300]}" for i, t in enumerate(texts))
    prompt = f"""将以下{len(texts)}条相关记忆合并为1条精炼记忆。保留所有关键事实，去除重复，保持时间线。

记忆列表:
{numbered}

合并后的记忆（一句话，中文）:"""

    merged = _call_infini(prompt, max_tokens=300, temperature=0.3)
    if merged and len(merged) > 10:
        return merged
    return None


def llm_verify_contradiction(text1: str, text2: str, marker: str) -> dict | None:
    """
    LLM 矛盾验证 — 确认关键词预筛的矛盾是否为真矛盾

    返回:
        {"is_contradiction": True/False, "type": "SUPERSEDE|EXTEND|FALSE_POSITIVE",
         "explanation": "..."}
        或 None (API 失败)

    矛盾类型:
        SUPERSEDE: 新事实完全取代旧事实 (旧→invalid)
        EXTEND: 新事实扩展/修正旧事实 (保留旧+加新)
        FALSE_POSITIVE: 不是真矛盾 (关键词匹配但语义无冲突)
    """
    prompt = f"""判断以下两条记忆是否存在真实矛盾。

记忆A: {text1[:500]}
记忆B: {text2[:500]}
检测标记: {marker}

请判断:
1. 是否存在真实的矛盾（不是简单的措辞差异，而是事实性冲突）？
2. 如果存在，矛盾类型是什么？
   - SUPERSEDE: 新事实完全取代旧事实
   - EXTEND: 新事实扩展/修正旧事实（新旧都有价值）
   - FALSE_POSITIVE: 不是真矛盾（关键词匹配但语义无冲突）

只输出一行JSON，格式:
{{"is_contradiction": true/false, "type": "SUPERSEDE/EXTEND/FALSE_POSITIVE", "explanation": "一句话解释"}}"""

    result = _call_infini(prompt, max_tokens=200, temperature=0.1)
    if not result:
        return None

    # 从结果中提取 JSON（LLM 可能输出额外文字）
    import re

    json_match = re.search(r"\{[^{}]+\}", result)
    if not json_match:
        return None

    try:
        parsed = json.loads(json_match.group())
        # 标准化
        if not parsed.get("is_contradiction", False):
            parsed["type"] = "FALSE_POSITIVE"
        return parsed
    except json.JSONDecodeError:
        return None


# ─── Neo4j 关系回写 ──────────────────────────────────────────────────
