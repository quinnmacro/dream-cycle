<div align="center">

# 🌙 Dream Cycle

### 自主记忆整理引擎 — 让 AI 在睡梦中巩固知识

**Autonomous Memory Consolidation for AI Agents**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-v3.3-green.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()

[English](#english) · [架构文档](docs/architecture.md) · [CLI 参考](docs/cli-reference.md) · [ADR-001](docs/adr/001-dream-cycle-architecture.md)

</div>

---

## 🇨🇳 中文

### 为什么需要 Dream Cycle？

AI Agent 每天积累上千条记忆碎片。如果不整理：

| 问题 | 后果 |
|:-----|:-----|
| 🔴 **重复堆积** | 同一条事实存了 5 次，搜索时 5 条全弹出来 |
| 🔴 **过期数据投毒** | 4 月的 CGB 1.65% 被当成今天的利率，NotebookLM 音频概览里张口就来 |
| 🔴 **关联埋没** | 「利差走阔」和「basis trade unwind」明明是同一件事，但永远连不上 |
| 🔴 **高价值洞察蒸发** | 深度研究的结论停留在 L1 碎片层，从来没有沉淀成长期知识 |

Dream Cycle 像**大脑的睡眠**一样工作——凌晨 4 点自动运行，把碎片记忆**聚类、评分、去重、推断关系、沉淀到知识库**。

### 核心架构：三阶段睡眠模型

> 灵感来源：睡眠神经科学的 **浅睡→REM→深睡** 三阶段模型 [\[1\]](#references)

```
                          ┌───────────────────┐
                          │   Mem0 PG (L1)    │
                          │   2200+ vectors   │
                          └────────┬──────────┘
                                   │ 48h 增量
                    ┌──────────────▼──────────────┐
                    │    🌙 Dream Cycle v3.2       │
                    │                              │
   ┌────────────────┼──────────────────────────────┼────────────────┐
   │                │                              │                │
   ▼                ▼                              ▼                ▼
┌──────────┐  ┌──────────┐                  ┌──────────┐     ┌──────────┐
│ Stage 1  │  │ Stage 2  │                  │ Stage 3  │     │  时间    │
│ 浅睡     │→ │ REM      │ ──────────────→  │ 深睡     │ ←── │  感知    │
│ 聚类     │  │ 评分+分析│                  │ 执行行动 │     │  7层防护 │
└──────────┘  └──────────┘                  └──────────┘     └──────────┘
                   │                              │
     ┌─────────────┼─────────────┐    ┌───────────┼───────────┐
     ▼             ▼             ▼    ▼           ▼           ▼
  Boost        Dedup         Vault  Neo4j      Decay      Telegram
  (强化)       (去重)       (沉淀)  (关系图)   (衰减)      (日报)
```

#### Stage 1：浅睡 — 聚类分组

把一天的记忆按主题分堆，就像睡前把今天的经历分类归档。

- **三路聚类**：关键词 Jaccard + pgvector 余弦距离 + 项目正则
- 向量邻居：余弦距离 < 0.30 → 同一簇
- 关键词重叠：共享实体加强簇内连接
- 产出：20-50 个簇 / 每次循环约 200 条新记忆

#### Stage 2：REM — 评分 + 分析

给每条记忆打分、找矛盾、识别高价值内容。

| 动作 | 触发条件 | 说明 |
|:-----|:---------|:-----|
| 📈 **Boost** | score ≥ 0.7 | 高重要性记忆在 Mem0 中强化 |
| 🗑️ **Dedup** | 向量距离 < 0.10 | 精确重复，删除 |
| 🔗 **Merge** | 距离 0.10-0.18 | 近似重复，合并 |
| ⚡ **矛盾检测** | 18 对反义词匹配 + LLM 验证 | 「上涨/下跌」「升级/降级」→ 标记 SUPERSEDE/EXTEND |
| 🏛️ **Vault 候选** | 综合评分高 | 整簇记忆→建议沉淀为知识库页面 |
| 🧠 **梦游** | Neo4j 种子 + cluster 实体 | 遍历知识图谱，推断新关系 |

#### Stage 3：深睡 — 执行行动

REM 阶段的所有决策在这里**真正执行**——写数据库、创建文件、归档旧数据。

- 去重 → 从 Mem0 PG 删除
- 合并 → 合并文本，保留主条
- 关系 → 推断实体关系 → 写入 Neo4j
- 衰减 → 标记低价值记忆，应用 Ebbinghaus 遗忘曲线
- Vault → 创建 wiki 骨架，LLM 充实概述
- NotebookLM → 同步新知识，清理过期 source

### ⏰ 时间感知：7 层防护

> 投资研究的记忆有半衰期。4 月的利率到 5 月就是垃圾。不处理→NotebookLM 把过期数字当现状念出来。

```
  ┌──────────────────────────────────────────────────────────────┐
  │                    7-Layer Time Protection                   │
  │                                                              │
  │  L1  记忆年龄 ──────── _compute_memory_age_days()            │
  │  L2  数据敏感检测 ──── _is_time_sensitive() · 18 种模式       │
  │  L3  Sample 选择 ────── freshest 而非 best score             │
  │  L4  LLM 概览 ──────── stale → "描述框架，不要引用具体数字"    │
  │  L5  Vault 标记 ─────── data_freshness: stale/recent/fresh   │
  │  L6  Sync 过滤 ──────── stale 跳过 + daily > 14天不推送       │
  │  L7  Audio 引导 ─────── "Today is {date}. Data = snapshots"  │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

| 层 | 组件 | 防护机制 |
|:---|:-----|:---------|
| 1 | `_compute_memory_age_days()` | 解析 `created_at`，计算天数年龄 |
| 2 | `_is_time_sensitive()` | 18 种正则模式检测市场数据：`\d+\.\d+%` / `CGB \d+Y` / `bp` / `利差` / `利率`… |
| 3 | Sample 选择 | `min(scored, key=age)` 最新记忆优先做样本 |
| 4 | LLM 概览 prompt | age > 7 天 + 时间敏感 → "不要引用具体数字，只描述框架/机制" |
| 5 | Vault frontmatter | `data_freshness: stale/recent/fresh` 元数据标记 |
| 6 | NotebookLM sync | stale 跳过 + `research/daily > 14 天` 不推送 + `--prune-stale` CLI |
| 7 | Audio description | 自动注入 `"Today is May 07, 2026. Market data = historical snapshots"` |

### 📊 健康评分

Dream Cycle 计算 5 维健康评分（0-100），像体检报告一样监控记忆系统状态：

```
  ┌─────────────────────────────────────────┐
  │        Dream Cycle Health Score         │
  │                                         │
  │  Freshness  ████████░░  25%   ■ <7天    │
  │  Coverage   ███░░░░░░░  25%   ■ 有页面  │
  │  Coherence  ██████░░░░  20%   ■ 非独居  │
  │  Efficiency █████░░░░░  15%   ■ 已去重  │
  │  Reachable  █████░░░░░  15%   ■ 有连接  │
  │                                         │
  │  Total: ████████████░░  67/100          │
  └─────────────────────────────────────────┘
```

### 🚀 快速开始

```bash
# 运行梦循环（默认处理最近 48 小时）
python3 dream_cycle.py

# Dry run（不写入，只预览）
python3 dream_cycle.py --dry-run

# 扩大时间窗口
python3 dream_cycle.py --hours 168    # 7 天

# 健康仪表盘
python3 dream_cycle.py --health

# 处理 pending 的 Vault 建议
python3 dream_cycle.py --vault-review

# NotebookLM 过期 source 清理
python3 vault_to_notebooklm_sync.py --prune-stale --dry-run

# NotebookLM 音频生成（自动注入时间感知 description）
python3 notebooklm_parallel_generate.py --types audio
```

### 🔌 数据源

| 数据源 | 类型 | 用途 |
|:-------|:-----|:-----|
| Mem0 PostgreSQL | 向量数据库 | 记忆存储 + 相似度搜索 (pgvector) |
| Neo4j | 图数据库 | 实体关系 + 知识图谱 |
| Vault | Markdown Wiki | 持久知识页面 (L2 存储) |
| NotebookLM | Google API | 音频概览 + 跨文档问答 |
| Infini AI | LLM API | 实体提取、关系推断、概述生成 |
| SQLite | 本地数据库 | 运行日志 + 关系记录 + Vault 建议 |

### 🔗 与 Hermes Agent 集成

Dream Cycle 作为 cron job（04:00 HKT）运行在 [Hermes Agent](https://github.com/NousResearch/hermes-agent) 生态中：

- **Memory Plugin**：Hermes `mem0_conclude` 调用 `online_dedup_check()` 实时去重
- **Skill**：`meta/dream-cycle` 提供快速参考
- **Vault Pipeline**：Dream Cycle → wiki-ingest → Obsidian → NotebookLM
- **Telegram**：每日梦循环报告通过 Hermes gateway 推送

### 🧪 测试

```bash
# 运行全部 45 个单元测试
cd /root/repos/dream-cycle
python3 -m pytest src/dream_cycle/tests/ -v

# 运行特定模块
python3 -m pytest src/dream_cycle/tests/test_stage2.py -v
```

测试覆盖：config 参数验证、similarity 函数（Jaccard/n-gram/combined）、stage2 衰减层级分类和重要性评分逻辑。当前无集成测试（依赖外部 PG/Neo4j）。

### ⚡ 性能优化 (v3.3)

批量查询模式避免 N+1 问题，显著减少 `docker exec` 调用次数：

| 操作 | 优化前 | 优化后 | 提升 |
|:-----|:-------|:-------|:-----|
| 语义签名冲突检测 (200条记忆) | ~200次 docker exec | 2次 docker exec | **100x** |
| 批量矛盾处理 (50条冲突) | ~150次 docker exec | 1次 docker exec | **150x** |
| 实体提取 (stage2→orchestrator) | 2次 LLM 调用/cluster | 1次 LLM 调用/cluster | **2x** |
| API key 读取 | 每次 LLM 调用读文件 | 首次读取后缓存 | **~20x I/O** |

### 📜 版本历史

| 版本 | 日期 | 关键变更 |
|:-----|:-----|:---------|
| v1.0 | 2026-04-19 | 三阶段骨架 + 规则提取 + Telegram 日报 |
| v2.0 | 2026-04-27 | LLM 实体提取 + 规则降级 + Neo4j 写入 + 垃圾清理 |
| v3.0 | 2026-05-03 | P1-P5：LLM boost + 向量去重 + Vault OR-gate + Ebbinghaus 联动 |
| v3.1 | 2026-05-04 | P6-P10：Telegram 日报增强 + 冲突处理 + 健康仪表盘 + Neo4j 双向 |
| v3.2 | 2026-05-07 | 时间感知 7 层防护 + NotebookLM audio description |
| v3.3 | 2026-05-31 | 10个关键bug修复 + 批量查询性能优化 (100-150x) + 45个单元测试 |

---

<a id="english"></a>

## 🇬🇧 English

### Why Dream Cycle?

AI agents accumulate thousands of memory fragments daily. Without consolidation:

| Problem | Consequence |
|:--------|:------------|
| 🔴 **Duplicates pile up** | The same fact stored 5 times, all 5 returned on every search |
| 🔴 **Stale data poisons reasoning** | April's CGB yield presented as today's rate, NotebookLM reads it as current |
| 🔴 **Connections stay buried** | "Spread widening" and "basis trade unwind" are the same event but never linked |
| 🔴 **High-value insights evaporate** | Research conclusions stay as L1 fragments, never promoted to durable knowledge |

Dream Cycle works like **biological sleep** — running automatically at 4 AM, it **consolidates fragmented memories into structured knowledge** through clustering, scoring, deduplication, and promotion.

### Core Architecture: Three-Stage Sleep Model

> Inspired by the **Shallow Sleep → REM → Deep Sleep** model from sleep neuroscience [\[1\]](#references)

```
                          ┌───────────────────┐
                          │   Mem0 PG (L1)    │
                          │   2200+ vectors   │
                          └────────┬──────────┘
                                   │ 48h incremental
                    ┌──────────────▼──────────────┐
                    │    🌙 Dream Cycle v3.2       │
                    │                              │
   ┌────────────────┼──────────────────────────────┼────────────────┐
   │                │                              │                │
   ▼                ▼                              ▼                ▼
┌──────────┐  ┌──────────┐                  ┌──────────┐     ┌──────────┐
│ Stage 1  │  │ Stage 2  │                  │ Stage 3  │     │  Time    │
│ Shallow  │→ │ REM      │ ──────────────→  │ Deep     │ ←── │  Aware   │
│ Cluster  │  │ Evaluate │                  │ Execute  │     │  7-Layer  │
└──────────┘  └──────────┘                  └──────────┘     └──────────┘
                   │                              │
     ┌─────────────┼─────────────┐    ┌───────────┼───────────┐
     ▼             ▼             ▼    ▼           ▼           ▼
  Boost        Dedup         Vault  Neo4j      Decay      Telegram
  (reinforce)  (remove dup)  (wiki)  (graph)   (ebbinghaus) (report)
```

#### Stage 1: Shallow Sleep — Clustering

Groups similar memories into thematic piles, like sorting the day's experiences before filing.

- **Three-way clustering**: keyword Jaccard + pgvector cosine + project regex
- Vector neighbors: cosine distance < 0.30 → same cluster
- Keyword overlap: shared entities strengthen cluster bonds
- Output: 20-50 clusters per cycle from ~200 new memories

#### Stage 2: REM — Evaluation

Scores memories for importance, detects contradictions, and identifies high-value clusters.

| Action | Trigger | Description |
|:-------|:--------|:------------|
| 📈 **Boost** | score ≥ 0.7 | Reinforce high-importance memories in Mem0 |
| 🗑️ **Dedup** | vector distance < 0.10 | Exact duplicate, remove |
| 🔗 **Merge** | distance 0.10-0.18 | Near-duplicate, combine |
| ⚡ **Contradiction** | 18 antonym pairs + LLM verify | "increased/decreased" → SUPERSEDE/EXTEND |
| 🏛️ **Vault candidates** | High aggregate score | Cluster → promote to wiki page |
| 🧠 **Dream Walk** | Neo4j seeds + cluster entities | Traverse knowledge graph, infer new relations |

#### Stage 3: Deep Sleep — Execution

Actually executes all REM decisions — writes to databases, creates files, archives stale data.

- Dedup → delete from Mem0 PG
- Merge → combine texts, keep primary
- Relations → infer entity relationships → write to Neo4j
- Decay → mark low-value memories, apply Ebbinghaus forgetting curve
- Vault → create wiki stubs with LLM-enriched overviews
- NotebookLM → sync fresh knowledge, prune stale sources

### ⏰ Time-Awareness: 7-Layer Protection

> Market data has a half-life. A yield quoted in April is dangerous in May. Without protection → NotebookLM narrates stale numbers as current.

| Layer | Component | Protection |
|:------|:----------|:-----------|
| 1 | `_compute_memory_age_days()` | Parse `created_at`, compute age in days |
| 2 | `_is_time_sensitive()` | 18 regex patterns detect market data: `\d+\.\d+%` / `CGB \d+Y` / `bp` / `利差`… |
| 3 | Sample selection | Freshest memory → sample text, not highest-scored |
| 4 | LLM overview prompt | Stale data → "describe framework, not numbers" |
| 5 | Vault frontmatter | `data_freshness: stale/recent/fresh` metadata |
| 6 | NotebookLM sync | Skip stale sources, prune daily research > 14 days |
| 7 | Audio description | Auto-inject: `"Today is {date}. Market data = historical snapshots"` |

### 📊 Health Score

5-dimensional health score (0-100), like an annual checkup for your memory system:

| Dimension | Weight | Measures |
|:----------|:-------|:---------|
| Freshness | 25% | % of memories < 7 days old |
| Coverage | 25% | % of clusters with vault pages |
| Coherence | 20% | % of non-singleton clusters |
| Efficiency | 15% | % of memories deduplicated |
| Reachability | 15% | % of entities with Neo4j connections |

### 🚀 Quick Start

```bash
# Run dream cycle (default: last 48 hours)
python3 dream_cycle.py

# Dry run (preview only, no writes)
python3 dream_cycle.py --dry-run

# Extended range
python3 dream_cycle.py --hours 168    # 7 days

# Health dashboard
python3 dream_cycle.py --health

# Review pending vault suggestions
python3 dream_cycle.py --vault-review

# NotebookLM stale source cleanup
python3 vault_to_notebooklm_sync.py --prune-stale --dry-run

# NotebookLM audio generation (auto-injects time-aware description)
python3 notebooklm_parallel_generate.py --types audio
```

### 🔌 Data Sources

| Source | Type | Purpose |
|:-------|:-----|:---------|
| Mem0 PostgreSQL | Vector DB | Memory storage + similarity search (pgvector) |
| Neo4j | Graph DB | Entity relationships + knowledge graph |
| Vault | Markdown Wiki | Durable knowledge pages (L2 storage) |
| NotebookLM | Google API | Audio overviews + cross-document Q&A |
| Infini AI | LLM API | Entity extraction, relation inference, overview generation |
| SQLite | Local DB | Run logs + relation records + vault suggestions |

### 🔗 Integration with Hermes Agent

Dream Cycle runs as a cron job (04:00 HKT) within the [Hermes Agent](https://github.com/NousResearch/hermes-agent) ecosystem:

- **Memory Plugin**: Hermes `mem0_conclude` calls `online_dedup_check()` for real-time dedup
- **Skill**: `meta/dream-cycle` provides quick reference
- **Vault Pipeline**: Dream Cycle → wiki-ingest → Obsidian → NotebookLM
- **Telegram**: Daily dream report via Hermes gateway

### 🧪 Testing

```bash
# Run all 45 unit tests
cd /root/repos/dream-cycle
python3 -m pytest src/dream_cycle/tests/ -v

# Run specific module
python3 -m pytest src/dream_cycle/tests/test_stage2.py -v
```

Test coverage: config parameter validation, similarity functions (Jaccard/n-gram/combined), stage2 decay tier classification and importance scoring logic. No integration tests currently (external PG/Neo4j dependencies).

### ⚡ Performance Optimizations (v3.3)

Batch query patterns eliminate N+1 problems, dramatically reducing `docker exec` calls:

| Operation | Before | After | Improvement |
|:----------|:-------|:------|:------------|
| Semantic slot conflict detection (200 memories) | ~200 docker exec | 2 docker exec | **100x** |
| Batch contradiction resolution (50 conflicts) | ~150 docker exec | 1 docker exec | **150x** |
| Entity extraction (stage2→orchestrator) | 2 LLM calls/cluster | 1 LLM call/cluster | **2x** |
| API key reading | File read per LLM call | Cached after first read | **~20x I/O** |

---

<a id="references"></a>

## 📚 References

| # | Reference | Relevance |
|:--|:----------|:----------|
| \[1\] | Walker, M. P. (2017). *Why We Sleep: Unlocking the Power of Sleep and Dreams*. Scribner. | Three-stage sleep model (Shallow → REM → Deep) that inspired Dream Cycle's architecture |
| \[2\] | Ebbinghaus, H. (1885). *Memory: A Contribution to Experimental Psychology*. | Forgetting curve formula used in decay/boost calculations: `score(t) = e^(-λ·Δt)` |
| \[3\] | Graphiti (2025). *Bi-Temporal Knowledge Graphs for LLM Agents*. [GitHub](https://github.com/getzep/graphiti) | Dual-temporal model (valid_at/invalid_at) referenced in ADR-013 time-series design |
| \[4\] | FluxMemo (2025). *Fact Lifecycle Management with SUPERSEDE/EXTEND/ADD*. | Contradiction resolution protocol adopted in `resolve_slot_conflicts()` |
| \[5\] | KektorDB (2025). *Ebbinghaus Decay in Vector Databases*. | Decay formula with access-count reinforcement: `e^(-age / (halfLife × (1 + ln(1 + accessCount))))` |
| \[6\] | Thoth (2025). *4-Stage Dream Cycle for Memory Consolidation*. | 90-day linear decay + dream cycle pattern (dedup → enrich → infer → decay) |
| \[7\] | MemGPT / Letta (2024). *Memory Management for LLM Agents*. [Paper](https://arxiv.org/abs/2310.08560) | Tiered memory architecture (core → archival → recall) informing L0/L1/L2 design |
| \[8\] | Karpathy, A. (2024). *Compiled Knowledge > RAG*. [Blog](https://karpathy.bearblog.dev/) | Compiled wiki over retrieval — Vault (L2) stores curated knowledge, not raw fragments |
| \[9\] | OpenAI (2026). *Harness Engineering: From Prompts to Runtime*. | Constraint > instruction principle; harness components have lifecycles; validation loops |
| \[10\] | Decagon (ICLR 2026). *Production Memory Systems at Scale*. | 20-100 samples optimal; reflection model must be frontier; 1500-word length constraint as regularization |
| \[11\] | Bordes, A. et al. (2025). *Memory Layers at Scale*. [Paper](https://arxiv.org/abs/2312.14930) | Memory-augmented neural networks; key-value lookup patterns informing vector search design |
| \[12\] | Ratcliff, R. (1990). *Connectionist Models of Recognition Memory*. | Signal-detection approach to memory matching — influences dedup threshold design (0.10/0.18) |

---

## 🤝 Related Projects

| Project | Description |
|:--------|:------------|
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | The AI agent framework Dream Cycle runs in |
| [Vyakarana](https://github.com/quinnmacro/Vyakarana) | Architecture decisions (ADR-023) |
| [Vault](https://github.com/quinnmacro/vault) | Knowledge base (L2 durable storage) |
| [mem0-stack](https://github.com/quinnmacro/mem0-stack) | Self-hosted Mem0 + Neo4j |
| [Graphiti](https://github.com/getzep/graphiti) | Bi-temporal knowledge graphs for LLM agents |
| [Letta (MemGPT)](https://github.com/letta-ai/letta) | Tiered memory management for LLM agents |

## 📄 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

*月光不问赶路人，记忆自有整理时*

*The moonlight asks not the traveler's haste — memories find their order in the dark*

</div>
