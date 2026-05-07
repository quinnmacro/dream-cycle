# ADR-023: Dream Cycle — 记忆自主整理架构

| Field | Value |
|-------|-------|
| Status | Accepted |
| Date | 2026-05-07 |
| Decision Drivers | Quinn, Hermes Agent |
| Local Commit | `264db5e` (time-awareness), `c319211` (P1-P5) |

## Context

4层时态记忆架构 (ADR-010) 定义了 L0-L3，但 L1→L2 的知识沉淀缺少自动化引擎：

- Mem0 PG 积累 2200+ 条碎片化记忆，从不整理
- 高频实体反复出现但从未沉淀为 Vault 页面
- 矛盾事实共存（"项目在Vultr" vs "项目搬到阿里云"），无检测
- 过期市场数据（4月CGB 1.65%）被 NotebookLM 当现状叙述
- 关系图（Neo4j）依赖手动导入，无自动推断

参考 OpenClaw Dreaming 三阶段模型（Shallow Sleep → REM → Deep Sleep）和 Thoth 4阶段梦循环（去重→丰富→推断→衰减）。

## Decision

实现 Dream Cycle 三阶段自动整理引擎，作为 L1→L2 的知识沉淀管道。

### 核心架构

```
Mem0 PG (L1) → Dream Cycle → 5 Outputs
  ├─ Stage 1: Shallow Sleep — 聚类分组
  ├─ Stage 2: REM — 评分+Boost+矛盾+梦游+实体提取
  └─ Stage 3: Deep Sleep — 去重/关系/Vault/衰减/冲突解决
```

**5个输出通道**：
1. **Boost** → PG freshness=fresh (Ebbinghaus联动)
2. **Decay** → PG freshness=outdated/stale/aging
3. **关系** → Neo4j Playground
4. **Vault** → ~/vault/ 自动沉淀stub
5. **日报** → Telegram

### 时间感知7层防护（v3.0 关键决策）

投资研究记忆含时间敏感的市场数据。不处理 → NotebookLM 把4月价格当现状。

| 层 | 机制 | 效果 |
|---|------|------|
| 1 | `_compute_memory_age_days()` | 解析created_at计算天数 |
| 2 | `_is_time_sensitive()` (18种模式) | 识别利率/利差/bp/价格/CGB/UST |
| 3 | sample选最新而非最高分 | 避免旧数据当选代表 |
| 4 | stale-aware LLM prompt | "不要引用具体数字，只描述框架" |
| 5 | `data_freshness` frontmatter | stale/recent/fresh标记 |
| 6 | NotebookLM sync跳过stale | daily>14天不推送 |
| 7 | audio description自动注入 | 含今日日期+框架引导 |

### P6-P10 增强

- **P6**: Telegram日报增强（Top3 boost/vault/conflict + 健康评分）
- **P7**: 语义冲突自动处理（SUPERSEDE→归档旧 / EXTEND→保留 / FALSE_POSITIVE→忽略）
- **P8**: `--health` CLI（7天趋势 + 5维健康评分）
- **P9**: `--vault-review` CLI（pending→auto_created→reviewed pipeline）
- **P10**: Neo4j双向同步（cluster实体优先seed）

## Options Considered

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| 纯规则引擎 | 零成本 | 关键词永远追不完，79%垃圾 | ❌ v1已验证失败 |
| 纯LLM提取 | 质量高 | 每条都调API成本太高 | ❌ 不经济 |
| LLM优先+规则降级 | 关键路径高质量，fallback保底 | 需两层维护 | ✅ 已采用 |
| 人工审核所有 | 最准 | 不可扩展 | ❌ |

## Consequences

### Positive
- L1碎片自动沉淀为L2文章（知识不再只存在于对话中）
- 矛盾事实主动检测和标记
- 过期市场数据不污染下游（NotebookLM/Vault）
- Neo4j 关系图自动生长
- 每日Telegram日报提供可操作的洞察

### Negative
- LLM调用成本（每次~300 tokens/实体，10实体~3K tokens/天）
- dream_cycle.db 额外维护（SQLite，4表）
- Neo4j写入可能产生噪音（需 _is_valid_entity 过滤）

### Risks
- LLM API 429限速（Infini共享配额）→ circuit breaker 120s
- 记忆提取质量问题→三层防御（prompt+.lower()移除+_normalize_entity_type）

## Confirmation

- [x] Dream Cycle cron 04:00 HKT 每日自动运行
- [x] Telegram日报正常推送
- [x] Neo4j 关系质量（79%垃圾→0% after LLM提取）
- [x] Vault stub 自动创建含 data_freshness
- [x] NotebookLM audio description 自动注入

## Revisit Conditions

- Neo4j Playground 关闭时→关系输出需改路径
- Mem0 v3+ 移除 PG→需改数据源
- LLM成本>5元/天→考虑本地模型

## Related ADRs

- [ADR-010](010-4-layer-temporal-architecture.md) — 4层时态架构（Dream Cycle是L1→L2引擎）
- [ADR-013](013-neo4j-temporal-awareness.md) — Neo4j时序感知（Dream Cycle写入关系）
- [ADR-014](014-mem0-time-series-awareness.md) — Mem0 Ebbinghaus衰减（Dream Cycle联动freshness）
- [ADR-021](021-neo4j-playground.md) — Neo4j Playground（Dream Cycle写入目标）
