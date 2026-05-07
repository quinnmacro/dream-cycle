# Dream Cycle 架构设计

> ADR-023 的技术实现细节。本文档描述 Dream Cycle v3.0+ 的完整架构。

## 概述

Dream Cycle 是 Hermes 记忆系统的自主整理引擎，每日凌晨 04:00 HKT 自动运行。它从 Mem0 PG 读取近期记忆，经过三阶段处理（聚类→评分→行动），输出5类结果：Boost/Decay/关系/Vault沉淀/Telegram日报。

**核心定位**：L1→L2 知识沉淀管道。不是独立存储层，是 Mem0 碎片到 Vault 文章的转换器。

## 三阶段流程

```
┌─────────────────────────────────────────────────────────┐
│  Mem0 PG (2200+ memories with pgvector)                 │
│  SELECT id, payload->>'data', payload->>'created_at'    │
│  WHERE created_at > NOW() - INTERVAL '48 hours'         │
└─────────────┬───────────────────────────────────────────┘
              │ incremental (skip processed_manifest)
              ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Shallow Sleep (浅睡) — 聚类分组               │
│                                                         │
│  1. extract_keywords() — 200+停用词+标点+bigram+领域加权  │
│  2. 三路聚类:                                            │
│     a) 关键词Jaccard → 语义相近的词聚类                    │
│     b) pgvector余弦距离 → 向量近邻聚类                     │
│     c) 项目正则(r'.*/(hermes-config|mem0-stack)/')       │
│  3. 合并: 关键词∪向量∪项目 → 最终cluster集合               │
│  4. Singleton(独居) vs Group(群居) 分类                   │
└─────────────┬───────────────────────────────────────────┘
              │ clusters: {cluster_key: [memory, ...]}
              ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2: REM (快速眼动) — 评分+分析                     │
│                                                         │
│  1. importance_score() — 词频×领域加权×独占性             │
│  2. Boost (score≥0.7) / Decay (score<0.25)              │
│  3. 去重检测:                                            │
│     a) pgvector (余弦距离<0.10 → 精确重复)               │
│     b) ngram+jaccard fallback (≥0.85 → 重复)            │
│  4. 合并检测 (距离0.10-0.18 / 相似度0.75-0.85)           │
│  5. 矛盾检测 — 18对反义词(上涨/下跌, 升级/降级...)        │
│  6. LLM验证矛盾 (限10条, SUPERSEDE/EXTEND/FALSE_POSITIVE) │
│  7. 梦游 (REM Dream Walk):                               │
│     a) Neo4j high-degree节点 → seed list                 │
│     b) cluster实体优先seed                               │
│     c) 邻居遍历 → 新关系推断 (conf=0.3)                  │
│  8. LLM Boost (co_cluster≥2 → LLM验证 → conf 0.3→0.6)  │
│  9. 实体提取: extract_entities_with_fallback()           │
│     a) LLM优先 (Infini AI, deepseek-v3.2)               │
│     b) 规则降级 (关键词+bigram)                           │
│ 10. 时间感知选sample: freshest而非best score             │
└─────────────┬───────────────────────────────────────────┘
              │ rem_results: {boosted, decayed, dedup, merge,
              │              vault_candidates, contradictions, ...}
              ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3: Deep Sleep (深睡) — 执行行动                   │
│                                                         │
│  1. 去重: 标记PG重复记忆 (hash+向量)                      │
│  2. 合并: 近似记忆合并建议                                │
│  3. 关系推断: 梦游结果 → Neo4j MERGE写入                  │
│     - _is_valid_entity() 过滤: 长度≤3/停用词/-ing/纯数字  │
│     - dedup_neo4j_relations() 同源去重                   │
│  4. 衰减归档:                                            │
│     - decay_candidates → PG freshness=stale/aging        │
│     - Ebbinghaus联动: boost→fresh, decay→outdated        │
│  5. 冲突自动处理 (P7):                                    │
│     - resolve_slot_conflicts() — LLM判断                 │
│     - SUPERSEDE → 旧边设invalid_at                       │
│     - EXTEND → 保留+EXTENDS关系                          │
│     - FALSE_POSITIVE → 忽略                              │
│  6. Vault沉淀:                                           │
│     - 门槛: score≥0.65 OR recalls≥2 OR sessions≥1       │
│     - create_vault_stub(): LLM充实概述+data_freshness    │
│     - 时间感知: stale数据→"不要引用具体数字"              │
│  7. 健康评分: 5维 (freshness/coverage/coherence/          │
│     efficiency/reachability) × 权重                      │
│  8. Telegram日报: Top3 boost/vault/conflict + 健康评分   │
└─────────────────────────────────────────────────────────┘
```

## 数据流图

```
                    ┌──────────┐
                    │ Mem0 PG  │ 2200+ vectors
                    │ (L1瞬时) │
                    └────┬─────┘
                         │ get_incremental_memories(48h)
                         ▼
              ┌─────────────────────┐
              │    Dream Cycle      │ 04:00 HKT daily
              │   (dream_cycle.py)  │ 3067行, 44函数
              └──┬──┬──┬──┬──┬──────┘
                 │  │  │  │  │
     ┌───────────┘  │  │  │  └──────────────┐
     ▼              ▼  │  ▼                  ▼
┌─────────┐  ┌──────┐ │ ┌──────┐    ┌───────────────┐
│PG boost │  │Neo4j │ │ │Vault │    │Telegram日报    │
│freshness│  │关系图│ │ │stub  │    │Top3+健康评分   │
│=fresh   │  │(L1)  │ │ │(L2)  │    └───────────────┘
└─────────┘  └──────┘ │ └──┬───┘
┌─────────┐  ┌──────┐ │    │
│PG decay │  │Neo4j │ │    ▼
│freshness│  │Playg.│ │ ┌──────────────────────┐
│=stale   │  │(独立) │ │ │vault_to_notebooklm   │
└─────────┘  └──────┘ │ │_sync.py              │
                      │ │  ├─ stale跳过         │
                      │ │  ├─ daily>14天不推送    │
                      │ │  └─ --prune-stale CLI  │
                      │ └──────────┬────────────┘
                      │            ▼
                      │ ┌──────────────────────┐
                      │ │NotebookLM            │
                      │ │  ├─ audio description │
                      │ │  │  自动注入今日日期   │
                      │ │  │  +框架引导          │
                      │ │  └─ mind-map/quiz等   │
                      │ └──────────────────────┘
                      │
                      └─→ dream_cycle.db (SQLite)
                           ├─ dream_runs (21次)
                           ├─ relation_log (57 useful)
                           ├─ vault_suggestion (7 valid)
                           └─ processed_manifest (增量标记)
```

## 时间感知7层防护

投资研究记忆的核心问题：4月的市场数据到5月就是垃圾。不处理→NotebookLM把4月CGB 1.65%当现状。

| # | 层 | 位置 | 机制 | 触发条件 |
|---|---|------|------|----------|
| 1 | 记忆年龄 | `_compute_memory_age_days()` | 解析created_at→天数 | 所有记忆 |
| 2 | 数据敏感检测 | `_is_time_sensitive()` | 18种正则：\d+\.\d+%/CGB\s+\d+Y/bp/利差/利率... | 含市场关键词的文本 |
| 3 | Sample选择 | `stage2_rem()` | `min(scored, key=age)` 而非 `max(scored, key=score)` | Vault候选 |
| 4 | LLM概览 | `create_vault_stub()` | stale prompt: "不要引用具体数字，只描述框架/机制" | age>7天+时间敏感 |
| 5 | Frontmatter | `data_freshness` | stale/recent/fresh 标记 | 自动生成页面 |
| 6 | Sync过滤 | `vault_to_notebooklm_sync.py` | stale跳过 + daily>14天不推送 + `--prune-stale` | NotebookLM同步 |
| 7 | Audio引导 | `get_category_description()` | "Today is {date}. Treat prices as historical snapshots..." | audio生成 |

### 时间敏感模式列表

```python
_TIME_SENSITIVE_PATTERNS = [
    r'\d+\.\d+%',        # 4.25%, 3.7%
    r'CGB\s+\d+Y',       # CGB 10Y
    r'UST\s+\d+Y',       # UST 10Y
    r'bp\b',             # 25bp
    r'\$\d+',            # $4.2B
    r'yield.*\d+\.\d+',  # yield 4.25
    r'spread.*\d+',      # spread 120
    r'Selic\s+\d+',      # Selic 14.74%
    r'Shibor\s+\d+',     # Shibor
    r'DR\d{3}\s',        # DR007
    r' hikes?|cuts?\b',  # hike/cut
    r' pricing\b',       # market pricing
    r' carry\b',         # carry trade
    r' 跌|涨|升|降',     # 中文市场方向
    r'利率|收益率|利差',  # 中文市场术语
]
```

## 关键函数清单

| 函数 | 作用 | LLM依赖 |
|------|------|----------|
| `extract_keywords()` | 停用词+bigram+领域加权 | ❌ 纯规则 |
| `extract_entities_with_fallback()` | LLM提取实体→规则降级 | ✅ Infini AI |
| `importance_score()` | 词频×领域×独占性 | ❌ 纯规则 |
| `llm_boost_relations()` | co_cluster≥2→LLM验证→conf 0.3→0.6 | ✅ Infini AI |
| `llm_verify_contradiction()` | LLM判断是否真矛盾 | ✅ Infini AI |
| `resolve_slot_conflicts()` | SUPERSEDE/EXTEND/FALSE_POSITIVE | ✅ Infini AI |
| `rem_dream_walk()` | Neo4j种子+cluster实体→新关系 | ❌ (读Neo4j) |
| `create_vault_stub()` | LLM充实概述+data_freshness | ✅ Infini AI |
| `get_category_description()` | NotebookLM audio description | ❌ 纯规则 |
| `_is_time_sensitive()` | 18种市场数据模式 | ❌ 纯规则 |
| `show_health_dashboard()` | 5维健康评分+7天趋势 | ❌ 读SQLite |
| `review_vault_suggestions()` | pending→auto_created→reviewed | ✅ (create_vault_stub) |
| `online_dedup_check()` | 写入前去冗余(被mem0_conclude调用) | ❌ pgvector |

## 数据库 Schema

### dream_cycle.db (SQLite)

```sql
-- 梦循环运行记录
CREATE TABLE dream_runs (
    id INTEGER PRIMARY KEY,
    run_date TEXT,
    total_memories INTEGER,
    stage1_clusters INTEGER,
    stage2_boosted INTEGER,
    stage3_deduped INTEGER,
    stage3_inferred INTEGER,
    stage3_decayed INTEGER,
    stage3_vault_suggestions INTEGER,
    health_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 关系日志 (LLM提取+梦游推断)
CREATE TABLE relation_log (
    id INTEGER PRIMARY KEY,
    dream_run_id INTEGER,
    source TEXT,        -- 实体A
    target TEXT,        -- 实体B
    relation TEXT,      -- 关系类型
    confidence REAL,    -- 0.3(梦游)/0.6(LLM验证)
    method TEXT,        -- llm_extract/dream_walk/llm_boost
    memory_ids TEXT,    -- JSON array
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Vault沉淀建议
CREATE TABLE vault_suggestion (
    id INTEGER PRIMARY KEY,
    dream_run_id INTEGER,
    entity TEXT,
    category TEXT,
    frequency INTEGER,
    reason TEXT,
    status TEXT DEFAULT 'pending'  -- pending/auto_created/reviewed/rejected
);

-- 增量处理标记
CREATE TABLE processed_manifest (
    memory_id TEXT PRIMARY KEY,
    processed_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

## 配置参考

```yaml
# Cron
schedule: "0 4 * * *"  # 04:00 HKT daily
command: "python3 /root/scripts/dream_cycle.py"

# 环境变量
INFINI_API_KEY: <from config.yaml>
INFINI_BASE_URL: https://cloud.infini-ai.com/maas/coding/v1
INFINI_MODEL: deepseek-v3.2

# 数据库路径
DREAM_DB: /root/data/dream_cycle.db
MEM0_PG: postgresql://mem0:***@localhost:8432/mem0_v2
NEO4J: bolt://localhost:7687 (neo4j/knowledge2026)

# 日志
LOG_FILE: /var/log/dream-cycle.log (5MB RotatingFileHandler)

# 阈值
DEDUP_DIST: 0.10       # pgvector余弦距离, <0.10=精确重复
MERGE_DIST: 0.18       # 0.10-0.18=可合并
PROMOTION_MIN_SCORE: 0.65  # Vault沉淀最低分
LLM_MAX_ENTITIES: 10   # 每次LLM提取最大实体数
LLM_MAX_BOOST: 10      # 每次最大boost数
STALE_THRESHOLD_DAYS: 7 # 超过7天=stale
DAILY_RESEARCH_MAX_AGE: 14  # NotebookLM daily>14天不推送
```

## CLI 命令参考

```bash
# 日常运行 (cron自动)
python3 dream_cycle.py

# Dry-run测试
python3 dream_cycle.py --dry-run

# 扩大时间窗口
python3 dream_cycle.py --hours 168  # 7天

# 健康仪表盘 (P8)
python3 dream_cycle.py --health

# Vault建议review (P9)
python3 dream_cycle.py --vault-review

# NotebookLM过期source清理
python3 vault_to_notebooklm_sync.py --prune-stale --dry-run
python3 vault_to_notebooklm_sync.py --prune-stale

# NotebookLM音频生成 (自动注入时间感知description)
python3 notebooklm_parallel_generate.py --types audio
python3 notebooklm_ops.py generate --notebook <id> --type audio
```

## 和其他系统的交互

| 系统 | 方向 | 交互方式 |
|------|------|----------|
| Mem0 PG | 读 | `get_incremental_memories()` → 48h增量 |
| Mem0 Plugin | 写 | boost→freshness=fresh, decay→freshness=stale |
| Vault (L2) | 写 | `create_vault_stub()` → ~/vault/{category}/{slug}.md |
| Neo4j Playground | 写 | `rem_dream_walk()` → MERGE节点+关系 |
| Neo4j Playground | 读 | high-degree节点 → seed list (P10) |
| NotebookLM | 间接 | vault_to_notebooklm_sync.py → 跳过stale + audio description |
| Telegram | 写 | format_report() → 日报推送 |
| mem0_conclude | 读 | `online_dedup_check()` 写入前去冗余 |

## 健康评分体系

5维评分，满分100：

| 维度 | 权重 | 计算 |
|------|------|------|
| freshness | 25% | (fresh+recent) / total_memories |
| coverage | 25% | memories_in_clusters / total |
| coherence | 20% | 1 - contradictions / total |
| efficiency | 15% | useful_relations / total_relations |
| reachability | 15% | avg_degree / max_degree |

**当前基准**：44/100（旧数据清理后关系较少，待积累）

## 版本演进

| 版本 | 日期 | 关键变更 |
|------|------|----------|
| v1.0 | 2026-04-19 | 三阶段骨架+规则提取+Telegram日报 |
| v1.1 | 2026-04-20 | 修关键词提取bug(79%垃圾) |
| v2.0 | 2026-04-27 | LLM实体提取+规则降级+Neo4j写入 |
| v3.0 | 2026-05-03 | P1-P5: boost+向量去重+Vault沉淀+触发修复+Ebbinghaus联动 |
| v3.1 | 2026-05-04 | P6-P10: 日报增强+冲突处理+健康仪表盘+Vault review+Neo4j双向 |
| v3.2 | 2026-05-07 | 时间感知7层防护+NotebookLM audio description |
