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

## 当前状态（2026-05-30 晚）

### 刚完成的大重构
- 从 4097 行单文件 → 15 模块包（最大 696 行）
- 加了 45 个单元测试（全绿）
- 修了 P0 致命 bug（float('') 崩溃、无并发锁、僵尸 run）
- 健康度从 46 → 91/100
- 已 push 到 GitHub (commits: 79bffc4, 0f9a621, 431d11f, c3acf66)

### 15 个模块
```
__init__.py    — 版本号
__main__.py    — CLI 入口 (argparse)
config.py      — 所有参数常量 (阈值/权重/路径/API)
db.py          — PG(docker exec) + SQLite + Neo4j 查询
llm.py         — DashScope qwen3.7-max LLM 调用
similarity.py  — 文本相似度 + 向量聚类
entities.py    — 实体提取 (LLM + 规则)
stage1.py      — Shallow Sleep: 四层聚类
stage2.py      — REM: 7维评分 + 矛盾检测 + 关系推断
stage3.py      — Deep Sleep: 去重/合并/衰减/回写
dream_engine.py — Neo4j 图操作 (REM walk/SHY/threat)
session.py     — state.db 信号扫描
vault.py       — Vault stub 创建
health.py      — 自适应触发 + 健康仪表盘
orchestrator.py — 主循环 + 锁 + 报告 + 批量运维
```

### 已知架构债务（你 audit 可能也会发现）
1. **PG 查询走 `docker exec`** — 每次 1-3 秒，应该换 psycopg2
2. **无测试 fixtures** — 45 个测试全是 mock，没有集成测试
3. **`docker exec` 的 SQL 注入风险** — 用字符串拼接而非参数化
4. **Neo4j Playground 经常不在线** — 7687 端口连接拒绝时所有图操作静默失败
5. **LLM 调用走 curl subprocess** — 应该换 `requests` 或 `urllib`
6. **config.py 参数太多（40+）** — 部分可以合并或分组

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
