"""
Dream Cycle — Vault integration — age computation, time-sensitivity, vault stub creation, topic keys
"""



__all__ = [
    "create_vault_stub",
    "is_time_sensitive",
    "compute_memory_age_days",
]

import os
import re
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dream_cycle.config import HKT, VAULT_DIR, log

def _compute_memory_age_days(created_at: str | None) -> float | None:
    """计算记忆的天数年龄"""
    if not created_at:
        return None
    try:
        # 兼容多种时间格式
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                     "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                     "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(created_at[:26], fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=HKT)
                return (datetime.now(HKT) - dt).total_seconds() / 86400
            except ValueError:
                continue
        return None
    except Exception:
        return None


# 时间敏感关键词 — 出现这些词的记忆中的具体数字容易过期
_TIME_SENSITIVE_PATTERNS = [
    r'\d+\.\d+%',        # 利率/利差 4.25%, 3.7%
    r'CGB\s+\d+Y',       # CGB 10Y
    r'UST\s+\d+Y',       # UST 10Y
    r'bp\b',             # 25bp, 100bp
    r'\$\d+',            # $4.2B 价格
    r'yield.*\d+\.\d+',  # yield 4.25
    r'spread.*\d+',      # spread 120
    r'rates?\s+(at|of|are)\s',  # rate at / rates of
    r'Selic\s+\d+',      # Selic 14.74%
    r'Shibor\s+\d+',     # Shibor
    r'DR\d{3}\s',        # DR007
    r' hikes?|cuts?\b',  # hike/cut
    r' pricing\b',       # market pricing
    r' position\w*\b',   # positioning
    r' carry\b',         # carry trade
    r' 跌|涨|升|降',     # 中文市场方向
    r'利率|收益率|利差',  # 中文市场术语
]


def _is_time_sensitive(text: str) -> bool:
    """判断文本是否包含时间敏感的市场数据"""
    if not text:
        return False
    for pattern in _TIME_SENSITIVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def create_vault_stub(entity: str, category: str, keywords: list[str], sample: str,
                      sample_age_days: float | None = None) -> str | None:
    """
    为高频实体创建 Vault 页面骨架
    
    P3 增强: 
    - 用LLM充实"概述"段落（不再是纯sample文本）
    - 反论自动注入（和wiki-ingest --llm对齐）
    - 时间感知：时间敏感数据不编入概述
    
    Args:
        entity: 实体名
        category: 类别
        keywords: 关键词
        sample: 样本文本
        sample_age_days: 样本文本的天数年龄
    Returns: 文件路径 or None
    """
    # 类别映射
    cat_map = {
        "markets": "markets",
        "investment": "markets", 
        "projects": "projects",
        "technology": "concepts",
        "concepts": "concepts",
        "articles": "articles",
    }
    vault_cat = cat_map.get(category, "concepts")
    
    # 生成 slug
    slug = entity.lower().replace(" ", "-").replace("|", "-")[:50]
    filepath = VAULT_DIR / vault_cat / f"{slug}.md"
    
    if filepath.exists():
        log.info(f"  📄 Vault 页面已存在: {filepath}")
        return None
    
    # P3: 用 LLM 充实概述段落 (取代纯 sample 拼贴)
    # 时间感知：如果sample含时间敏感数据且>7天，明确告知LLM不要引用具体数字
    is_stale_data = (sample_age_days is not None and sample_age_days > 7 
                     and _is_time_sensitive(sample))
    
    if is_stale_data:
        overview_prompt = (
            f"用1-2句话简明概述'{entity}'这个概念的核心定义和结构性特征。中文回答，不要用列表，不要'值得注意的是'。"
            f"重要：不要引用任何具体数字（利率、利差、价格、百分比等），这些数据已过时。"
            f"只描述框架、机制、结构性关系。"
        )
    else:
        overview_prompt = (
            f"用1-2句话简明概述'{entity}'这个概念的核心定义和关键特征。中文回答，不要用列表，不要'值得注意的是'。"
        )
    
    overview = sample[:300]  # fallback
    llm_overview = _call_infini(overview_prompt, max_tokens=150, temperature=0.3)
    if llm_overview and len(llm_overview) > 20:
        overview = llm_overview
    
    # 生成 frontmatter
    now = datetime.now(HKT).strftime("%Y-%m-%d")
    # 数据新鲜度标记
    if is_stale_data:
        data_freshness = "stale"
        freshness_note = f"\n> ⚠️ 源数据已过期（{sample_age_days:.0f}天前），待更新"
    elif sample_age_days is not None and sample_age_days <= 1:
        data_freshness = "fresh"
        freshness_note = ""
    elif sample_age_days is not None and sample_age_days <= 7:
        data_freshness = "recent"
        freshness_note = ""
    else:
        data_freshness = "unknown"
        freshness_note = ""
    
    frontmatter = f"""---
title: "{entity}"
date: {now}
category: {vault_cat}
tags: [{', '.join(f'"{k}"' for k in keywords[:5])}]
explored: false
confidence: 0.5
provenance: dream_cycle
data_freshness: {data_freshness}
---

# {entity}

> 🌱 由 Dream Cycle 自动生成，待人工充实{freshness_note}

## 概述

{overview}

## 待探索

- [ ] 核心定义
- [ ] 关键指标
- [ ] 当前状态
- [ ] 与其他概念的关系

## 来源

- 自动检测: Dream Cycle ({now})
- 关联关键词: {', '.join(keywords[:5])}
"""
    
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(frontmatter)
        log.info(f"  📄 Vault 页面已创建: {filepath}")
        return str(filepath)
    except Exception as e:
        log.warning(f"⚠️ Vault 创建失败: {e}")
        return None


# ─── Stage 1: Shallow Sleep (浅睡) — 记忆聚类 ────────────────────────

