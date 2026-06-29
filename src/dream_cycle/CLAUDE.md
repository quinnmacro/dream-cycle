# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 运行命令

```bash
# 从 ops/ 目录运行 (需要 sys.path 包含 ops/)
cd /root/scripts/ops
python3 -m dream_cycle                    # 完整梦循环 (默认48h回溯)
python3 -m dream_cycle --hours 24         # 自定义回溯窗口
python3 -m dream_cycle --dry-run          # 只分析不执行
python3 -m dream_cycle --stages 12        # 只跑 Stage 1+2 (跳过 Deep Sleep)
python3 -m dream_cycle --auto             # 自适应模式: 检查触发条件再决定是否执行
python3 -m dream_cycle --notify           # 执行后发 Telegram 报告

# 运维命令
python3 -m dream_cycle --health           # 7天健康仪表盘
python3 -m dream_cycle --trigger-check    # 检查是否应该触发
python3 -m dream_cycle --history 5        # 查看最近5次运行记录
python3 -m dream_cycle --manifest-stats   # manifest 统计
python3 -m dream_cycle --backlog          # 一键清理所有积压 (矛盾+vault)

# Cron 兼容包装器
python3 /root/scripts/ops/dream_cycle.py  # 等价于 python3 -m dream_cycle
```

45 个单元测试 (pytest)，无 linter 配置、无构建步骤。纯 Python 模块，依赖系统级 Python 包。

## 架构概述

Bio-inspired 记忆整理引擎，模拟人类睡眠周期对 AI agent 记忆进行自主整理。数据源是 mem0 (PostgreSQL)，整理结果写入 Neo4j 知识图谱和 Obsidian Vault。

### 三阶段流水线

```
PG(mem0) → [Stage 1: Shallow Sleep] → [Stage 2: REM] → [Stage 3: Deep Sleep] → Neo4j + Vault
                聚类                    评分+矛盾检测       执行去重/合并/衰减/关系推断
```

- **Stage 1 (stage1.py)** — 四层聚类: 实体级 → 精确hash去重 → pgvector向量聚类 → n-gram文本聚类(fallback)
- **Stage 2 (stage2.py)** — 7维重要性评分(FadeMem双层衰减) + 矛盾检测(关键词预筛 + LLM验证) + Vault候选识别(三重门限) + 跨聚类实体共现关系推断
- **Stage 3 (stage3.py)** — 执行整合: boost标记 → 去重归档(永不删除) → LLM合并 → Neo4j关系回写 → SHY缩减 → 衰减归档 → Vault stub创建 → 语义签名冲突自动处理

### Dream Engine (dream_engine.py)

独立于三阶段之外的 Neo4j 图操作:
- **REM Walk v2**: 从高重要性种子节点随机游走，30%创意跳跃概率，Waking Gate(shared neighbor ≥1)过滤弱边
- **SHY Downscaling**: 突触稳态 — top 20%边保护，其余梯度缩减，低于阈值的剪枝
- **Threat Simulation**: 扫描 CONTRADICTS 边标记矛盾
- **NREM Hebbian**: 高度数节点间连接强化 + 全局α缩减
- **LLM Boost**: 对梦游发现的低置信关系做LLM验证，高共现的从0.3提升到0.6

### 数据访问层

- **db.py**: PG 通过 `docker exec -i postgres psql` stdin管道模式查询(非直连)，state.db/dream_cycle.db 用 sqlite3 直连，Neo4j 用官方 Python driver
- **llm.py**: DashScope API (qwen3.7-max)，通过 curl subprocess 调用，API key 从 `/root/.hermes/config.yaml` 读取
- **similarity.py**: 文本相似度(Jaccard 40% + n-gram 60%)、pgvector余弦距离、批量向量聚类

### 关键设计模式

- **增量处理**: `processed_manifest` 表追踪已处理记忆，`get_incremental_memories()` 只返回新增的
- **FadeMem 双层衰减**: R = e^(-λ·t^β)，volatile(β=1.2, 市场数据快衰减) / normal(β=1.0) / stable(β=0.8, 用户偏好慢衰减)
- **永不删除**: 所有"删除"操作实际是写 `archived: true` 到 PG payload
- **优先级标记**: 含 `⚠️ PERMANENT` / `🔥 HIGH` / `📌 PIN` 的记忆永不归档
- **LLM fallback 链**: 实体提取 LLM优先 → 规则fallback；相似度 pgvector优先 → n-gram fallback
- **并发锁**: `/tmp/dream_cycle.lock` PID文件锁，1小时超时自动接管

### 外部依赖

| 依赖 | 用途 | 连接方式 |
|------|------|----------|
| PostgreSQL (docker: `postgres`) | mem0 记忆存储 | `docker exec` stdin 管道 |
| Neo4j Playground (bolt://100.69.76.69:7687) | 知识图谱 | neo4j Python driver |
| DashScope API (qwen3.7-max) | LLM 合并/验证/实体提取 | curl subprocess |
| state.db (`/root/.hermes/state.db`) | Claude Code session 记录 | sqlite3 |
| dream_cycle.db (`/root/data/dream_cycle.db`) | 梦循环元数据 | sqlite3 |
| Vault (`/root/vault/`) | Obsidian 知识库页面 | 文件系统 |
| Telegram Bot | 报告推送 | curl API |

### 关键文件职责

- **config.py** — 所有可调参数的唯一来源（阈值、权重、路径、API配置）
- **types.py** — v7 dataclass 合约: DreamMemory, MemoryOp, PrepareResult, BudgetSummary, ExecuteResult
- **ops.py** — v7 MemoryBackend 抽象层: DirectBackend (PG直写) + StagingBackend (缓冲) + create_backend() factory
- **orchestrator.py** — 主循环编排 + 锁管理 + 报告生成 + 批量运维操作
- **staging.py** — StagingBuffer + PGProposal + write_staging() + adopt() 安全合约
- **budget.py** — EditBudget: 编辑预算(8/晚) + token预算 + 挂钟预算 + 余弦衰减
- **split.py** — SHA256 确定性 train/val/test 分割 (70/20/10)
- **validation.py** — Held-out 验证: 搜索质量评分 + Gate Safety Probe
- **session.py** — 从 state.db 挖掘 session 信号(纠正/偏好/决策/模式)注入为高优先级记忆
- **health.py** — 自适应触发判断(4条件) + 在线去冗余检查(写入时) + 健康仪表盘
- **entities.py** — 实体提取(LLM + 规则)、停用词表、领域加权关键词
- **vault.py** — Vault stub 创建(含时间感知：过期市场数据不编入概述)

## 测试

```bash
# 运行全部测试 (150 tests)
cd /root/repos/dream-cycle
python3 -m pytest src/dream_cycle/tests/ -v

# 运行特定模块测试
python3 -m pytest src/dream_cycle/tests/test_stage2.py -v
```

测试覆盖: config参数验证、similarity函数、stage2衰减层级和评分逻辑。无集成测试（依赖外部PG/Neo4j）。

## 性能优化

### 批量查询模式 (P11)

避免 N+1 查询问题，所有批量操作使用预取模式：

- **detect_slot_conflicts** (stage3.py): 2次批量查询（IDs+文本 + 向量邻居），而非逐条查询（最坏200次docker exec）
- **batch_resolve_all_conflicts** (orchestrator.py): 1次批量查询所有文本+时间戳，而非每条冲突2-4次查询
- **stage2_rem → orchestrator**: stage2 缓存 `cluster_entities` 到 `rem_results`，orchestrator 复用而非重新提取（避免重复LLM调用）

### API Key 缓存 (llm.py)

`_api_key_cache` 模块级变量缓存 API key，避免每次 `_call_infini()` 都读取 config.yaml。首次调用读取，后续调用直接返回缓存值。

### 预期性能提升

| 操作 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| detect_slot_conflicts (200条记忆) | ~200次 docker exec | 2次 docker exec | 100x |
| batch_resolve_all_conflicts (50条冲突) | ~150次 docker exec | 1次 docker exec | 150x |
| stage2→orchestrator 实体提取 | 2次 LLM 调用/cluster | 1次 LLM 调用/cluster | 2x |
| API key 读取 | 每次 LLM 调用读文件 | 首次读取后缓存 | ~20x I/O |

## 开发注意事项

### 添加新的批量操作时

遵循预取模式：先批量查询构建 lookup map，再在循环中从 map 取值，避免在循环内调用 `pg_query()`。

### 修改 LLM 调用时

如果需要清除 API key 缓存（例如切换账号），重启进程或手动设置 `llm._api_key_cache = None`。
