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

## 当前状态（2026-05-31 v3.3.0）

### v3.3.0 变更（Claude Code 审计后修复）
**commit**: 96aa294 (foxcc/feat: v3.3.0)

#### 10 个关键 Bug 修复
**Tier 1 — NameError 崩溃（3个）**
- `__main__.py`: 缺失 `json`, `sqlite3`, `online_dedup_check` 导入 → 3 个 CLI 命令崩溃
- `db.py:236`: 缺失 `text_hash` 导入 → `update_manifest()` 每次必崩
- `stage3.py:262`: 缺失 `_KEYWORD_DOMAIN_BOOST` 导入 → vault 建议处理崩溃

**Tier 2 — 安全漏洞（2个）**
- `db.py update_memory_text()`: `shell=True` + 手工转义 → shell 注入。已改用 stdin pipe + `to_jsonb()`
- `stage3.py resolve_slot_conflicts()`: LLM explanation 单引号截断 SQL → SUPERSEDE 归档静默失败

**Tier 3 — 逻辑错误（3个）**
- `stage2.py:224`: recall_count 回退值用 `len(group)` → 大 cluster 评分虚高。已改为 0
- `stage2.py:384`: vault 门控 `passes_any` 恒真 → 三重门限形同虚设。已删除
- `orchestrator.py:353`: no_memories 跳过路径缺 `_skip_run()` → dream_runs 孤儿记录

**Tier 4 — 一致性（2个）**
- `orchestrator.py` slug 生成缺 `|` 替换 → 与 vault.py 不一致
- `db.py pg_query` 签名撒谎（params 参数是空壳，返回类型标注错误）

#### 性能优化（100-150x 提升）
- `detect_slot_conflicts`: 批量查询 IDs+文本+向量邻居（2次 docker exec vs ~200次）
- `batch_resolve_all_conflicts`: 预取所有文本+时间戳（1次 docker exec vs ~150次）
- `stage2→orchestrator`: 缓存 `cluster_entities` 到 `rem_results`，避免重复 LLM 提取
- `llm.py`: API key 模块级缓存，避免每次调用读 config.yaml

#### 代码质量改进
- `text_hash` 从 `config.py` 移到 `similarity.py`（消除循环依赖）
- 所有导入路径已同步更新（db.py, health.py, stage1.py, tests）

#### 验证状态
- ✅ 45 个单元测试全通过
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
