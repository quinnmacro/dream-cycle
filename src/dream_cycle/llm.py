"""
Dream Cycle — LLM API calls — DashScope (qwen3.7-max) for merge/verify/entity extraction
"""

import json
import subprocess
import logging
from pathlib import Path
from dream_cycle.config import (
    INFINI_BASE_URL, INFINI_MODEL, HKT, log,
)

def _get_infini_api_key() -> str:
    """读取 DashScope API key (优先) → Infini AI (fallback)"""
    try:
        import yaml
        with open("/root/.hermes/config.yaml") as f:
            config = yaml.safe_load(f)
        # 优先 DashScope
        key = config.get("credentials", {}).get("dashscope_api_key", "")
        if key:
            return key
        # fallback Infini
        key = config.get("credentials", {}).get("infini_api_key", "")
        if key:
            return key
    except:
        pass
    try:
        with open("/root/projects/mem0-selfhost/.env") as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    return line.strip().split("=", 1)[1]
    except:
        pass
    return ""


def _call_infini(prompt: str, max_tokens: int = 300, temperature: float = 0.3) -> str | None:
    """调用 DashScope API (qwen3.7-max) 的通用函数"""
    api_key = _get_infini_api_key()
    if not api_key:
        return None
    
    payload = json.dumps({
        "model": INFINI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "enable_thinking": False,  # 不需要推理，节省 tokens 和延迟
    })
    
    import subprocess
    auth_header = "Authorization: Bearer " + api_key
    try:
        result = subprocess.run(
            ["curl", "-s", INFINI_BASE_URL + "/chat/completions",
             "-H", auth_header,
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True, text=True, timeout=60,
        )
        resp = json.loads(result.stdout)
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip() if content else None
    except Exception as e:
        log.warning(f"⚠️ Infini API 调用失败: {e}")
        return None


def llm_merge_memories(texts: list[str]) -> str | None:
    """
    用 LLM 将多条近似记忆合并为一条
    
    使用 qwen3.7-max (DashScope), 独立配额不受 Hermes 影响
    """
    numbered = "\n".join(f"{i+1}. {t[:300]}" for i, t in enumerate(texts))
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
    json_match = re.search(r'\{[^{}]+\}', result)
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

