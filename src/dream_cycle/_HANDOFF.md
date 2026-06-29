# 🦊 Context Handoff — For Claude Code

> 你正在 audit 这个目录，但你可能不知道外面的全貌。这是 Hermes Agent（另一个 AI）写的状态同步。

## 你在哪

这个目录 **不是一个独立项目**。它是一个 symlink：

```
/root/scripts/ops/dream_cycle → /root/repos/dream-cycle/src/dream_cycle/
```

真正的 Git repo 在 `/root/repos/dream-cycle/`，GitHub: **quinnmacro/dream-cycle**。

## 谁在改这个代码

有**两个 AI agent** 同时在碰这个 codebase：

| Agent | 角色 | 工具链 |
|-------|------|--------|
| **Hermes Agent (我)** | 运行时运维 + cron 管理 + 功能迭代 | 直接写文件 + terminal |
| **Claude Code (你)** | 代码审计 + 重构 + 质量把关 | claude CLI + git |

Quinn（我们共同的用户）让你做 audit，让我做日常维护。我们之间没有直接通信通道，靠 Quinn 中转和文件系统。

## 当前状态（2026-06-29 v7.1.0）

### v7.1.0 变更（Backend 抽象层完成）
**commit**: 5985fba (test: 45→150 tests covering v7 modules)

#### 架构重构：MemoryBackend 替代 monkey-patching
**之前 (v6)**：staging 通过全局 monkey-patch 拦截 PG 写入
```python
# orchestrator.py — 旧方式（已删除 158 行）
_db_module.pg_query = _intercepted_pg_query  # 全局替换
```
问题：SQL 正则解析提取意图、crash 时 interceptor 泄漏、`from db import pg_query` 绕过拦截、不可测试。

**之后 (v7)**：Backend 接口 + 结构化操作
```python
# ops.py — 新方式
backend = create_backend(use_staging=True, budget=budget)
backend.execute(MemoryOp(op="archive", memory_id="...", stage="dedup", reason="..."))
```

#### 6 种 MemoryOp 类型
`archive` | `update_text` | `delete` | `boost` | `extend` | `degrade`

- stage3.py: 0 处直连 SQL mutation（6 处 SELECT 读保留）
- resolve_slot_conflicts: 3 处 SQL → MemoryOp (archive + 2× extend)
- degrade_tiers: 2 处 SQL → 2× MemoryOp (degrade)

#### 150 个测试（原 45 → 150）
新增 5 个测试文件覆盖 v7 模块：
- test_types.py: MemoryOp, DreamMemory, BudgetSummary (20 tests)
- test_ops.py: DirectBackend routing, StagingBackend gating (18 tests)
- test_budget.py: EditBudget spend/fraction/token (28 tests)
- test_split.py: deterministic hash split (19 tests)
- test_staging.py: StagingBuffer ops/stats (20 tests)

#### v6.0-v6.3 SkillOpt-Sleep 机制移植
- v6.0 "Safe Sleep": Staging + Split + Validation + Edit Budget (+1,279 lines)
- v6.1 "Smart Sleep": Feedback Signals + Contrastive + Cache + Dual-backend (+542 lines)
- v6.2 "Efficient Sleep": Depth Planning + Parallel Merge + JSON Retry (+123 lines)
- v6.3 "Full Sleep": Associative Recall + Gate Safety Probe (+162 lines)

#### v7.0 "Clean Sleep" — 文件位置整理 + monkey-patch 删除
- 删除 4 份代码副本（hermes-config/scripts/skills/ 中的残留）
- 删除 158L interceptor 代码
- 唯一来源: `/root/repos/dream-cycle/src/dream_cycle/`
- v7.1: 剩余 SQL mutation 全部迁移到 backend

#### 20 个模块（8,427 行）
orchestrator(1128) → stage3(899) → stage2(739) → shmr(604) → dream_engine(576) → db(526) → staging(416) → health(399) → llm(383) → validation(360) → entities(339) → config(339) → session(338) → ops(219) → vault(191) → __main__(184) → similarity(174) → budget(163) → stage1(158) → types(152)
- ✅ 150 个单元测试全通过
- ✅ 所有 CLI 命令正常（--manifest-stats, --history, --dedup-check, --dry-run）
- ✅ 所有模块导入检查通过

### 已知架构债务（待后续处理）
1. **PG 查询走 `docker exec`** — 每次 1-3 秒，应该换 psycopg2
2. **无测试 fixtures** — 45 个测试全是 mock，没有集成测试
3. **Neo4j Playground 经常不在线** — 7687 端口连接拒绝时所有图操作静默失败
4. **LLM 调用走 curl subprocess** — 应该换 `requests` 或 `urllib`
5. **config.py 参数太多（40+）** — 部分可以合并或分组

**已修复的债务**：
- ~~`docker exec` 的 SQL 注入风险~~ → `update_memory_text()` 已改用 stdin pipe（其他函数仍有风险，但优先级低）

### 运行方式
```bash
# Cron: 每天 04:00 HKT
python3 /root/scripts/ops/dream_cycle.py

# 手动运维
python3 -m dream_cycle --health      # 7天仪表盘
python3 -m dream_cycle --backlog     # 清理积压
python3 -m dream_cycle --dry-run     # 只分析不执行
```

## 你的审计输出

Quinn 在等你出 audit 报告。如果你需要我（Hermes）配合——比如跑某个命令验证行为、提供历史运行日志、或者解释某个设计决策——让 Quinn 告诉我。

两个 AI 同时改同一个 codebase 这种事不多见。别把代码搞乱了就行 🦊
