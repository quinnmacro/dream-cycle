#!/usr/bin/env python3
"""
Hermes Dream Cycle — 记忆自主整理循环
======================================

三阶段管道:
  Stage 1: Shallow Sleep (浅睡) — 记忆聚类分组
  Stage 2: REM (快速眼动) — 重要性评分 + Boost
  Stage 3: Deep Sleep (深睡) — 去重/推断关系/衰减清理

设计参考:
  - OpenClaw Dreaming: Shallow→REM→Deep 三阶段
  - Thoth 4-stage: 去重→丰富→推断→衰减
  - KektorDB: Ebbinghaus 衰减公式
  - Supermemory: 时序感知事实更新

运行时间: off-peak (03:00-05:00 HKT), 在 backup 之后
模型: qwen3.7-max (DashScope, 凌晨独立配额)
数据源: mem0 v2 PG (直连) + Neo4j Playground + state.db
写回目标: PG(去重/merge) + Neo4j(关系推断) + Vault(建议) + Telegram(报告)
"""

import json
import hashlib
import re
import sqlite3
import time
import logging
import logging.handlers
import argparse
import sys
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

# ─── 配置 ──────────────────────────────────────────────────────────────

HKT = timezone(timedelta(hours=8))

# PG 连接 (通过 docker exec)
PG_CONTAINER = "postgres"
PG_USER = "postgres"
PG_DB = "mem0_v2"

# Neo4j 连接
NEO4J_URI = "bolt://100.69.76.69:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "knowledge2026"

# 并发锁文件
DREAM_LOCK = Path("/tmp/dream_cycle.lock")
DREAM_LOCK_TIMEOUT = 3600  # 1小时超时（正常跑完约15-25分钟）

def safe_float(val, default=None) -> float | None:
    """安全 float 转换 — PG 返回空字符串时不崩溃"""
    if val is None or val == '':
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

# state.db
STATE_DB = Path("/root/.hermes/state.db")

# ─── P2-1: 真实 recall 统计 ─────────────────────────────────────────────

_recall_stats_cache: dict | None = None

def get_recall_stats() -> dict[str, int]:
    """
    从 state.db 解析 mem0_search tool_calls，构建查询词 → 调用次数映射
    
    缓存结果（每次 dream cycle 只读一次 state.db）
    Returns: {"query_text": count, ...}
    """
    global _recall_stats_cache
    if _recall_stats_cache is not None:
        return _recall_stats_cache
    
    stats: dict[str, int] = {}
    try:
        conn = sqlite3.connect(str(STATE_DB))
        cursor = conn.execute("""
            SELECT tool_calls FROM messages 
            WHERE tool_calls IS NOT NULL AND tool_calls != '[]'
        """)
        for row in cursor:
            try:
                tc = json.loads(row[0])
                for call in tc:
                    fn = call.get("function", {}).get("name", "")
                    if fn in ("mem0_search", "mem0_profile"):
                        args = call.get("function", {}).get("arguments", "{}")
                        if isinstance(args, str):
                            args = json.loads(args)
                        query = args.get("query", "").strip()
                        if query and len(query) > 3:
                            stats[query] = stats.get(query, 0) + 1
            except:
                continue
        conn.close()
    except Exception as e:
        log.warning(f"⚠️ recall stats 读取失败: {e}")
    
    _recall_stats_cache = stats
    log.info(f"📊 recall stats: {len(stats)} unique queries, {sum(stats.values())} total calls")
    return stats


def match_memory_to_queries(memory_text: str, query_stats: dict[str, int]) -> tuple[int, int]:
    """
    将一条记忆的文本与搜索查询匹配
    
    Returns: (recall_count, session_count)
    - recall_count: 有多少次搜索命中了这条记忆
    - session_count: 有多少个不同查询命中（代理 session diversity）
    """
    text_lower = memory_text.lower()
    recall_count = 0
    matched_queries = set()
    
    for query, count in query_stats.items():
        # 查询词至少 50% 的关键词出现在记忆文本中
        query_words = [w for w in query.lower().split() if len(w) > 2]
        if not query_words:
            continue
        matched_words = sum(1 for w in query_words if w in text_lower)
        if matched_words / len(query_words) >= 0.5:
            recall_count += count
            matched_queries.add(query)
    
    return recall_count, len(matched_queries)

# Vault
VAULT_DIR = Path("/root/vault")

# 梦循环数据库 — 记录每次循环的元数据
DREAM_DB = Path("/root/data/dream_cycle.db")

# 相似度阈值 (向量余弦距离, 越小越相似)
# cosine_dist = 0 -> 完全相同, 0.5 -> 正交, 1.0 -> 完全相反
DEDUP_DIST = 0.10          # <0.10 视为重复 (相似度>0.90)
MERGE_DIST = 0.18          # 0.10-0.18 视为可合并 (相似度0.82-0.90)
CLUSTER_DIST = 0.30        # 0.18-0.30 视为同组 (相似度0.70-0.82)

# 文本相似度阈值 (n-gram fallback, 无向量时使用)
DEDUP_THRESHOLD = 0.92     
MERGE_THRESHOLD = 0.85     
CLUSTER_THRESHOLD = 0.70  

# 重要性评分权重 (7维, v3 — 加 novelty 对齐 SCM ValueTagger)
IMPORTANCE_WEIGHTS = {
    "recency": 0.12,          # 时间衰减 (14天半衰期)
    "frequency": 0.18,        # 短期信号累积 (被引用/搜索次数)
    "query_diversity": 0.12,  # 不同查询上下文命中数
    "domain": 0.18,           # 投资类 > 技术类 > 日常类
    "consolidation": 0.14,    # 多日重现强度 (跨 session 出现)
    "confidence": 0.08,       # 高置信度加分
    "novelty": 0.18,          # 新奇度 — 与已有记忆的语义距离 (SCM 核心)
}

# 自适应触发条件 (来自 SCM + SleepGate)
# 满足任一条件即触发梦循环
TRIGGER_MAX_IDLE_HOURS = 6      # 最长间隔: 6小时无梦循环则触发
TRIGGER_MIN_NEW_MEMORIES = 10   # 新增记忆阈值: >=10条新记忆触发
TRIGGER_CONFLICT_DENSITY = 0.3  # 冲突密度阈值 (SCM: θ_c=0.3)
TRIGGER_MEMORY_ENTROPY = 0.9    # 记忆熵阈值 (SCM: θ_e=0.9)

# NREM Hebbian 强化参数 (来自 SCM)
HEBBIAN_LEARNING_RATE = 0.1     # η: 连接强度增量
HEBBIAN_DOWNSCALE = 0.8         # α: 全局缩减因子 (20%缩减/周期)

# REM 梦游参数 (来自 SCM + claude-brain bio-inspired dream engine)
REM_WALK_LENGTH = 5             # 随机游走步数
REM_SEED_COUNT = 5              # 从前N个高重要性节点出发
REM_MAX_NEW_EDGES = 10          # 每次梦游最多生成的新关联
REM_JUMP_PROBABILITY = 0.30     # 30% 概率创意跳跃到拓扑远但语义近的节点
WAKING_THRESHOLD = 1            # Waking gate: 至少共享1个邻居才提升 provisional edge
SHY_PROTECTION_PCT = 0.20       # SHY: top 20% 边受保护不下调
SHY_DOWNSCALE_FACTOR = 0.15     # SHY: 非保护边最大下调比例
SHY_PRUNE_THRESHOLD = 0.08      # SHY: 下调后低于此阈值的边直接剪枝

# 三重门限提升 (来自 OpenClaw Dreaming)
# 所有条件必须同时满足才能提升到 Vault/长期
PROMOTION_MIN_SCORE = 0.65        # 最低综合分
PROMOTION_MIN_RECALLS = 3         # 最低被召回次数
PROMOTION_MIN_SESSIONS = 2        # 最低跨 session 出现次数

# 优先标记 (永不归档, 来自 Auto-Dream)
PERMANENT_MARKERS = ['⚠️ PERMANENT', '🔥 HIGH', '📌 PIN']

# 衰减参数 (FadeMem 双层差异化 Ebbinghaus: R = e^(-λ·t^β))
# β>1 = super-linear fast decay (volatile), β<1 = sub-linear slow decay (stable)
DECAY_HALF_LIVES = {
    "volatile": 3,    # 天 — 市场数据
    "normal": 7,      # 天 — 项目/工具
    "stable": 30,     # 天 — 用户/基建
}
FADEMEM_BETA = {
    "volatile": 1.2,  # super-linear: 市场数据快速衰减
    "normal": 1.0,    # linear: 标准指数衰减
    "stable": 0.8,    # sub-linear: 用户偏好/基建慢衰减
}
RETENTION_FLOOR = 0.20  # 最低保留率底线 (来自 PowerMem)

# Session Signal Scanning (from Anthropic autoDream / dream-skill)
# 从 state.db 用户消息中提取高价值信号
SIGNAL_CORRECTIONS = [
    "不对", "错了", "actually", "wrong", "别这样", "stop doing", "不是",
    "incorrect", "I said", "I meant", "don't do", "correction", "修改",
]
SIGNAL_PREFERENCES = [
    "我喜欢", "prefer", "always use", "从今以后", "记住", "I like",
    "I want", "going forward", "keep in mind", "make sure", "我的偏好",
]
SIGNAL_DECISIONS = [
    "决定", "我们用", "let's go with", "chosen", "decision", "agreed",
    "the plan is", "switch to", "move to", "pick", "选定", "方案是",
]
SIGNAL_PATTERNS = [
    "又是", "每次", "every time", "again", "keep forgetting", "as usual",
    "same as before", "like last time", "老问题", "反复",
]

# 归档策略 (来自 Auto-Dream: 永不删除, 归档)
ARCHIVE_THRESHOLD_DAYS = 90  # >90天 + 低重要性 → 归档
ARCHIVE_MIN_SCORE = 0.25     # 低于此分数才归档

# DashScope 配置 (用于 LLM 调用)
INFINI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
INFINI_MODEL = "qwen3.7-max"

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler("/var/log/dream-cycle.log", maxBytes=5_000_000, backupCount=3),
    ],
)
log = logging.getLogger("dream")

# ─── 数据库初始化 ──────────────────────────────────────────────────────

def init_dream_db():
    """初始化梦循环元数据库"""
    DREAM_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DREAM_DB))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS dream_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            stage1_clusters INTEGER DEFAULT 0,
            stage2_boosted INTEGER DEFAULT 0,
            stage3_deduped INTEGER DEFAULT 0,
            stage3_inferred INTEGER DEFAULT 0,
            stage3_decayed INTEGER DEFAULT 0,
            stage3_vault_suggestions INTEGER DEFAULT 0,
            error TEXT,
            summary TEXT
        );
        CREATE TABLE IF NOT EXISTS dedup_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dream_run_id INTEGER REFERENCES dream_runs(id),
            kept_id TEXT NOT NULL,
            removed_id TEXT NOT NULL,
            similarity REAL NOT NULL,
            merged_text TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS relation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dream_run_id INTEGER REFERENCES dream_runs(id),
            source_entity TEXT NOT NULL,
            target_entity TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            confidence REAL DEFAULT 0.5,
            method TEXT DEFAULT 'inferred',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS vault_suggestion (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dream_run_id INTEGER REFERENCES dream_runs(id),
            entity TEXT NOT NULL,
            category TEXT,
            frequency INTEGER DEFAULT 0,
            reason TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS processed_manifest (
            memory_id TEXT PRIMARY KEY,
            memory_hash TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            last_processed_at TEXT NOT NULL,
            process_count INTEGER DEFAULT 1,
            last_dream_run_id INTEGER,
            status TEXT DEFAULT 'active'
        );
        CREATE INDEX IF NOT EXISTS idx_manifest_hash ON processed_manifest(memory_hash);
        CREATE INDEX IF NOT EXISTS idx_manifest_status ON processed_manifest(status);
        CREATE TABLE IF NOT EXISTS contradiction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dream_run_id INTEGER REFERENCES dream_runs(id),
            mem1_id TEXT NOT NULL,
            mem2_id TEXT NOT NULL,
            marker TEXT,
            contradiction_type TEXT,
            llm_explanation TEXT,
            verified INTEGER DEFAULT 0,
            resolution TEXT DEFAULT 'pending',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    return conn


# ─── PG 访问层 ─────────────────────────────────────────────────────────

def pg_query(sql: str, params=None) -> list[dict]:
    """通过 docker exec 查询 PG (stdin 管道模式, 避免 Argument list too long)"""
    import subprocess, tempfile
    if params:
        pass
    cmd = ['docker', 'exec', '-i', PG_CONTAINER, 'psql', '-U', PG_USER, '-d', PG_DB, '-t', '-A', '-F', '|']
    result = subprocess.run(cmd, input=sql.encode(), capture_output=True, text=False, timeout=300)
    stdout = result.stdout.decode()
    if result.returncode != 0:
        log.error(f"PG query failed: {result.stderr.decode()[:200]}")
        return []
    rows = []
    for line in stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|")
        rows.append(parts)
    return rows


def get_recent_memories(hours: int = 48) -> list[dict]:
    """获取最近 N 小时的记忆"""
    rows = pg_query(f"""
        SELECT id::text, payload->>'data' as text, payload->>'user_id' as uid,
               payload->>'created_at' as created, payload->>'hash' as hash,
               payload->>'agent_id' as agent_id
        FROM mem0
        WHERE payload->>'created_at' IS NOT NULL
        AND (payload->>'created_at')::timestamptz > NOW() - INTERVAL '{hours} hours'
        AND payload->>'data' IS NOT NULL
        AND LENGTH(payload->>'data') > 20
        ORDER BY payload->>'created_at' DESC
    """)
    memories = []
    for r in rows:
        if len(r) >= 4 and r[1]:  # has text
            memories.append({
                "id": r[0],
                "text": r[1],
                "user_id": r[2],
                "created_at": r[3],
                "hash": r[4] if len(r) > 4 else None,
                "agent_id": r[5] if len(r) > 5 else None,
            })
    return memories


def get_incremental_memories(hours: int = 48) -> list[dict]:
    """
    增量获取 — 只返回上次梦循环未处理的新记忆
    
    使用 processed_manifest 表做 O(1) 查找:
    1. 从 PG 取最近 hours 小时的记忆
    2. 从 manifest 过滤已处理的
    3. 返回增量集合
    """
    all_recent = get_recent_memories(hours)
    if not all_recent:
        return []
    
    conn = sqlite3.connect(str(DREAM_DB))
    try:
        processed_ids = set()
        for row in conn.execute("SELECT memory_id FROM processed_manifest WHERE status = 'active'").fetchall():
            processed_ids.add(row[0])
    finally:
        conn.close()
    
    new_memories = [m for m in all_recent if m["id"] not in processed_ids]
    log.info(f"  📊 增量获取: {len(all_recent)} 条中 {len(new_memories)} 条新增 "
             f"(已处理 {len(all_recent) - len(new_memories)})")
    return new_memories


def update_manifest(memories: list[dict], dream_run_id: int):
    """更新 processed_manifest — 标记已处理"""
    if not memories:
        return
    conn = sqlite3.connect(str(DREAM_DB))
    now = datetime.now(HKT).isoformat()
    for m in memories:
        h = text_hash(m["text"])
        conn.execute("""
            INSERT INTO processed_manifest (memory_id, memory_hash, first_seen_at, last_processed_at, process_count, last_dream_run_id, status)
            VALUES (?, ?, ?, ?, 1, ?, 'active')
            ON CONFLICT(memory_id) DO UPDATE SET
                last_processed_at = ?,
                process_count = process_count + 1,
                last_dream_run_id = ?,
                status = 'active'
        """, (m["id"], h, now, now, dream_run_id, now, dream_run_id))
    conn.commit()
    conn.close()


def mark_manifest_archived(memory_ids: list[str]):
    """标记 manifest 中已归档的记忆"""
    if not memory_ids:
        return
    conn = sqlite3.connect(str(DREAM_DB))
    for mid in memory_ids:
        conn.execute("UPDATE processed_manifest SET status = 'archived' WHERE memory_id = ?", (mid,))
    conn.commit()
    conn.close()


def get_all_memories_with_embeddings() -> list[dict]:
    """获取所有记忆 (含向量用于聚类, 暂时只用文本)"""
    rows = pg_query("""
        SELECT id::text, payload->>'data' as text, payload->>'user_id' as uid,
               payload->>'created_at' as created, payload->>'hash' as hash
        FROM mem0
        WHERE payload->>'data' IS NOT NULL
        AND LENGTH(payload->>'data') > 20
        ORDER BY payload->>'created_at' DESC
    """)
    memories = []
    for r in rows:
        if len(r) >= 4 and r[1]:
            memories.append({
                "id": r[0],
                "text": r[1],
                "user_id": r[2],
                "created_at": r[3],
                "hash": r[4] if len(r) > 4 else None,
            })
    return memories


def delete_memory(memory_id: str) -> bool:
    """删除一条记忆"""
    rows = pg_query(f"DELETE FROM mem0 WHERE id::text = '{memory_id}' RETURNING id::text")
    return len(rows) > 0


def update_memory_text(memory_id: str, new_text: str) -> bool:
    """更新记忆文本"""
    import subprocess
    escaped = new_text.replace("'", "''").replace("\\", "\\\\")
    cmd = f"""docker exec {PG_CONTAINER} psql -U {PG_USER} -d {PG_DB} -c "UPDATE mem0 SET payload = jsonb_set(payload, '{{data}}', '\"{escaped}\"') WHERE id::text = '{memory_id}';" """
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    return result.returncode == 0


# ─── 相似度计算 (不依赖向量, 用文本hash+jaccard) ──────────────────────

def text_hash(text: str) -> str:
    """文本指纹"""
    return hashlib.md5(text.lower().strip().encode()).hexdigest()[:16]


def jaccard_similarity(s1: str, s2: str) -> float:
    """Jaccard 词级相似度 (快速, 不需向量)"""
    w1 = set(s1.lower().split())
    w2 = set(s2.lower().split())
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def ngram_similarity(s1: str, s2: str, n: int = 3) -> float:
    """N-gram 相似度 (比 Jaccard 更敏感)"""
    def ngrams(text, n):
        return set(text[i:i+n] for i in range(len(text)-n+1))
    ng1 = ngrams(s1.lower(), n)
    ng2 = ngrams(s2.lower(), n)
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)


def combined_similarity(s1: str, s2: str) -> float:
    """混合相似度: 0.4*jaccard + 0.6*ngram"""
    return 0.4 * jaccard_similarity(s1, s2) + 0.6 * ngram_similarity(s1, s2)


# ─── 向量相似度 (PG pgvector) ───────────────────────────────────────

def get_vector_neighbors(memory_id: str, limit: int = 10, max_dist: float = 0.30) -> list[dict]:
    """
    用 PG pgvector 找到最近邻 (余弦距离)
    
    Returns: [{"id": ..., "text": ..., "distance": ...}, ...]
    """
    sql = f"""
        SELECT b.id::text, LEFT(b.payload->>'data', 200) as text,
               ROUND((a.vector <=> b.vector)::numeric, 4) as dist
        FROM mem0 a, mem0 b
        WHERE a.id::text = '{memory_id}'
        AND a.id != b.id
        AND (a.vector <=> b.vector) < {max_dist}
        AND b.payload->>'data' IS NOT NULL
        ORDER BY a.vector <=> b.vector
        LIMIT {limit}
    """
    rows = pg_query(sql)
    neighbors = []
    for r in rows:
        if len(r) >= 3 and r[1]:
            dist = safe_float(r[2])
            if dist is None:
                continue
            neighbors.append({
                "id": r[0],
                "text": r[1],
                "distance": dist,
            })
    return neighbors


def batch_vector_clustering(memory_ids: list[str], max_dist: float = 0.30) -> dict[str, list[str]]:
    """
    批量向量聚类 — 对一组 memory_id 找到向量近邻
    
    Returns: {memory_id: [neighbor_id, ...], ...}
    """
    if len(memory_ids) < 2:
        return {}
    
    # 用 UNION ALL 一次查完
    id_list = "','".join(memory_ids)
    sql = f"""
        SELECT a.id::text as source, b.id::text as neighbor,
               ROUND((a.vector <=> b.vector)::numeric, 4) as dist
        FROM mem0 a, mem0 b
        WHERE a.id::text IN ('{id_list}')
        AND b.id::text IN ('{id_list}')
        AND a.id < b.id
        AND (a.vector <=> b.vector) < {max_dist}
        ORDER BY dist
    """
    rows = pg_query(sql)
    
    # 构建邻接表
    graph: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if len(r) >= 3:
            dist = safe_float(r[2])
            if dist is None:
                continue
            src, nbr = r[0], r[1]
            graph[src].add(nbr)
            graph[nbr].add(src)
    
    return {k: list(v) for k, v in graph.items()}


# ─── LLM 合并摘要 ────────────────────────────────────────────────────

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

def dedup_neo4j_relations(new_relations: list[dict]) -> list[dict]:
    """
    P2 语义去重: 查 Neo4j 已有关系，过滤重复
    
    MERGE 语义: 如果已有 (A)-[RELATED_TO]->(B) with conf=0.4，
    新关系 (A)-[RELATED_TO]->(B) with conf=0.6 → 更新 conf (取最大值)
    
    真正去重的是: 相同 source+target 但不同 type 的重复写入
    """
    if not new_relations:
        return new_relations
    
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return new_relations
    
    # 查 Neo4j 已有的 dream_cycle 关系
    existing_pairs: set[str] = set()
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            result = session.run("""
                MATCH (a:Concept)-[r]->(b:Concept)
                WHERE r.source = 'dream_cycle'
                AND a.name IS NOT NULL AND b.name IS NOT NULL
                RETURN a.name as src, b.name as tgt, type(r) as rel_type, r.confidence as conf
            """).data()
            for row in result:
                key = f"{row['src']}|{row['tgt']}|{row['rel_type']}"
                existing_pairs.add(key)
        driver.close()
    except Exception as e:
        log.warning(f"⚠️ Neo4j 去重查询失败: {e}")
        return new_relations
    
    # 过滤重复
    deduped = []
    for rel in new_relations:
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        rel_type = rel.get("type", "RELATED_TO").replace(" ", "_")
        safe_rel_type = ''.join(c for c in rel_type if c.isalnum() or c == '_') or "RELATED_TO"
        key = f"{src}|{tgt}|{safe_rel_type}"
        # 也检查反向 (A→B 和 B→A 视为同一条)
        reverse_key = f"{tgt}|{src}|{safe_rel_type}"
        
        if key not in existing_pairs and reverse_key not in existing_pairs:
            deduped.append(rel)
            existing_pairs.add(key)  # 防止批量内重复
    
    log.info(f"  🔄 Neo4j 关系去重: {len(new_relations)}→{len(deduped)} "
             f"(过滤 {len(new_relations)-len(deduped)} 重复)")
    return deduped


def write_relations_to_neo4j(relations: list[dict]) -> int:
    """
    将推断的关系写入 Neo4j Playground
    
    Args: [{"source": ..., "target": ..., "type": ..., "confidence": ...}]
    Returns: 写入数量
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        log.warning("⚠️ neo4j driver 未安装, 跳过回写")
        return 0
    
    count = 0
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            for rel in relations:
                source = rel.get("source", "").replace("'", "").replace('"', '')
                target = rel.get("target", "").replace("'", "").replace('"', '')
                rel_type = rel.get("type", "RELATED_TO").replace(" ", "_")
                confidence = rel.get("confidence", 0.4)
                
                if not source or not target or len(source) < 2 or len(target) < 2:
                    continue
                
                # 用参数化查询避免 Cypher 注入
                # rel_type 不能参数化, 但已用 replace 清洗
                safe_rel_type = ''.join(c for c in rel_type if c.isalnum() or c == '_')
                if not safe_rel_type:
                    safe_rel_type = "RELATED_TO"
                
                session.run(f"""
                    MERGE (a:Concept {{name: $source}})
                    MERGE (b:Concept {{name: $target}})
                    MERGE (a)-[r:{safe_rel_type}]->(b)
                    SET r.confidence = $conf,
                        r.source = 'dream_cycle',
                        r.created_at = datetime()
                """, source=source, target=target, conf=confidence)
                count += 1
        driver.close()
    except Exception as e:
        log.warning(f"⚠️ Neo4j 写入失败: {e}")
    
    return count


# ─── Vault 自动沉淀 ──────────────────────────────────────────────────

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

def extract_topic_key(text: str) -> str:
    """
    从记忆文本提取主题键 — 用于实体级聚类
    
    策略:
    1. 提取项目名/技能名/仓库名等实体
    2. 提取核心动作词
    3. 组合为主题键
    """
    import re
    
    # 项目/仓库名模式
    repo_patterns = [
        r'(hermes-config|hermes-agent|skills-vendors|vault|mem0-stack|neo4j-playground|quinnpm|Server-Admin|bondTickAnalysis|cv|memory-bridge|Vyakarana)',
        r'([\w-]+)/(?:skills?|repo|project)',
    ]
    
    # 技能名模式
    skill_patterns = [
        r'(?:skill|技能)[s]?\s*[:/]?\s*([a-z][\w-]+)',
        r'([a-z][\w-]+)/(?:SKILL|skill)',
    ]
    
    # 话题关键词
    topic_patterns = [
        r'(?:RBA|Fed|ECB|BOJ|PBOC)\s*(?:framework|decision|rate)',
        r'(?:Bloomberg|terminal|AI)\s*(?:prompt|query|research)',
        r'(?:Vault|wiki|Obsidian)\s*(?:cleanup|restructure|ingest|lint)',
        r'(?:Neo4j|graph)\s*(?:label|query|sync|entity)',
        r'(?:Docker|container)\s*(?:restart|build|deploy|health)',
        r'(?:mem0|memory)\s*(?:plugin|upgrade|model|auth)',
        r'(?:dream|梦)\s*(?:cycle|循环)',
        r'(?:CGB|UST|yield|bond)\s*(?:curve|spread|data)',
        r'(?:COMEX|silver|gold)\s*(?:delivery|inventory)',
    ]
    
    # 排除列表 — 常见误匹配
    EXCLUDE_ENTITIES = {'on', 'from', 'the', 'with', 'for', 'and', 'not', 'but', 'or', 'in', 'at', 'to', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their'}
    
    entities = []
    
    # 提取仓库名
    for p in repo_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            entities.append(m.group(1).lower())
    
    # 提取技能名
    for p in skill_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            entities.append(m.group(1).lower())
    
    # 提取话题
    for p in topic_patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            entities.append(m.group(0).lower().replace(' ', '_'))
    
    if entities:
        filtered = [e for e in set(entities) if e.lower() not in EXCLUDE_ENTITIES and len(e) > 2]
        if filtered:
            return '|'.join(sorted(filtered))
    return ''


# ─── 高质量关键词提取 (替代原始 Counter) ──────────────────────────────

# 综合停用词表 (英文+中文+LLM常见垃圾)
_KEYWORD_STOP_WORDS = frozenset({
    # English
    'about', 'above', 'after', 'again', 'all', 'also', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being',
    'below', 'between', 'both', 'but', 'by', 'can', 'could', 'did', 'do',
    'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'get', 'got', 'had', 'has', 'have', 'having', 'he', 'her',
    'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if',
    'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more',
    'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off',
    'on', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out',
    'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such',
    'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to',
    'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what',
    'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will',
    'with', 'would', 'you', 'your', 'yours', 'yourself', 'yourselves',
    # LLM slop markers
    'user', 'assistant', 'system', 'model', 'output', 'input', 'response',
    'request', 'message', 'content', 'text', 'information', 'note',
    'however', 'therefore', 'thus', 'moreover', 'furthermore', 'additionally',
    'specifically', 'indeed', 'essentially', 'particularly', 'notably',
    'significantly', 'simply', 'actually', 'basically', 'literally',
    'obviously', 'clearly', 'delve', 'regarding', 'ensure', 'leverage',
    # Generic non-entity terms
    'trade', 'framework', 'data', 'system', 'project', 'file', 'update',
    'change', 'feature', 'lines', 'order', 'blocking', 'repository',
    'added', 'updated', 'created', 'removed', 'fixed', 'skill', 'new',
    'old', 'first', 'last', 'only', 'another', 'different', 'important',
    'using', 'based', 'need', 'make', 'like', 'know', 'think', 'want',
    'well', 'much', 'many', 'still', 'even', 'back', 'way', 'thing',
    'things', 'something', 'everything', 'nothing', 'anything', 'already',
    'always', 'never', 'every', 'without', 'within', 'along', 'since',
    # Common verbs (not entities)
    'implemented', 'uses', 'used', 'using', 'created', 'updated', 'removed',
    'deleted', 'added', 'fixed', 'changed', 'modified', 'replaced', 'merged',
    'installed', 'configured', 'deployed', 'started', 'stopped', 'running',
    'working', 'worked', 'requires', 'required', 'supports', 'supported',
    'provides', 'provided', 'includes', 'included', 'contains', 'contained',
    'allows', 'allowed', 'enables', 'enabled', 'prevents', 'prevented',
    'follows', 'followed', 'returns', 'returned', 'accepts', 'accepted',
    'handles', 'handled', 'processes', 'processed', 'generates', 'generated',
    # Common generic nouns (not entities)
    'memory', 'limit', 'service', 'command', 'option', 'parameter', 'value',
    'result', 'output', 'error', 'warning', 'status', 'version', 'number',
    'default', 'method', 'function', 'variable', 'argument', 'example',
    'format', 'type', 'name', 'path', 'directory', 'folder', 'script',
    'code', 'line', 'step', 'process', 'task', 'action', 'check', 'test',
    'access', 'rule', 'entry', 'table', 'column', 'row', 'field', 'key',
    'source', 'target', 'source', 'group', 'item', 'element', 'section',
    'block', 'module', 'component', 'instance', 'resource', 'record',
    # More generic verbs/nouns
    'reduced', 'usage', 'commits', 'commit', 'branch', 'remote', 'local',
    'cache', 'index', 'query', 'response', 'request', 'session', 'client',
    'server', 'host', 'port', 'domain', 'network', 'connection', 'timeout',
    'buffer', 'stream', 'batch', 'chunk', 'payload', 'header', 'token',
    'callback', 'handler', 'listener', 'observer', 'filter', 'mapper',
    'reducer', 'transform', 'convert', 'parse', 'validate', 'encode',
    'decode', 'serialize', 'deserialize', 'normalize', 'sanitize',
    # -ing forms (almost never entities)
    'adding', 'removing', 'updating', 'creating', 'deleting', 'running',
    'loading', 'saving', 'reading', 'writing', 'processing', 'handling',
    'checking', 'testing', 'building', 'deploying', 'starting', 'stopping',
    'monitoring', 'tracking', 'logging', 'reporting', 'notifying',
    'reducing', 'increasing', 'decreasing', 'improving', 'extending',
    'merging', 'splitting', 'moving', 'renaming', 'copying', 'pasting',
    'skipping', 'falling', 'rising', 'dropping', 'changing', 'setting',
    'getting', 'putting', 'calling', 'returning', 'showing', 'hiding',
    # Years
    '2024', '2025', '2026', '2027',
    # Chinese
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
    '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
    '会', '着', '没有', '看', '好', '自己', '这', '他', '她', '它',
    '那', '被', '从', '把', '让', '用', '对', '为', '与', '而',
})

# 领域关键词加权 — 投资术语 3x，技术术语 1.5x
_KEYWORD_DOMAIN_BOOST = {
    # Investment (3x)
    'bonds': 3, 'yield': 3, 'spread': 3, 'CGB': 3, 'UST': 3, 'carry': 3,
    'duration': 3, 'credit': 3, 'curve': 3, 'swap': 3, 'basis': 3,
    'delivery': 3, 'inventory': 3, 'bond': 3, 'rate': 3, 'inflation': 3,
    'fed': 3, 'ecb': 3, 'boj': 3, 'rb': 3, 'pboc': 3, 'macro': 3,
    'fiscal': 3, 'monetary': 3, 'hedge': 3, 'position': 3, 'flow': 3,
    'premium': 3, 'sovereign': 3, 'cme': 3, 'comex': 3, 'silver': 3,
    'gold': 3, 'treasury': 3, 'coupon': 3, 'maturity': 3, 'issuance': 3,
    '利差': 3, '利率': 3, '收益率': 3, '曲线': 3, '债券': 3, '信用': 3,
    '央行': 3, '通胀': 3, '利差': 3, '溢价': 3, '对冲': 3, '仓位': 3,
    # Tech (1.5x)
    'docker': 1.5, 'plugin': 1.5, 'mcp': 1.5, 'mem0': 1.5, 'neo4j': 1.5,
    'config': 1.5, 'deploy': 1.5, 'cron': 1.5, 'api': 1.5, 'hermes': 1.5,
    'vault': 1.5, 'ssh': 1.5, 'tunnel': 1.5, 'token': 1.5, 'oauth': 1.5,
}

# 有意义的实体最小长度
_ENTITY_MIN_LEN = 4


def extract_keywords(texts: list[str], top_n: int = 5, min_score: float = 1.0) -> list[str]:
    """
    从文本列表中提取高质量关键词 — 替代原始 Counter 词频统计
    
    改进:
    1. 综合停用词过滤 (英文+中文+LLM slop)
    2. 标点/URL/代码块清洗
    3. Bigram 短语检测 (两词组合)
    4. 领域感知加权 (投资/技术术语优先)
    5. 缩写识别 (全大写2-5字母)
    6. 最低频率/分数阈值
    """
    import re
    from collections import Counter
    
    combined = " ".join(texts)
    
    # 标点清洗: 去代码块、URL、markdown
    cleaned = re.sub(r'```[\s\S]*?```', '', combined)
    cleaned = re.sub(r'https?://\S+', '', cleaned)
    cleaned = re.sub(r'[^\w\s\u4e00-\u9fff-]', ' ', cleaned)
    
    words = cleaned.split()
    
    # 单词级提取 + 加权
    word_scores: Counter = Counter()
    for w in words:
        w_clean = w.lower().rstrip(',.').strip('-')
        if not w_clean or len(w_clean) < _ENTITY_MIN_LEN:
            continue
        if w_clean in _KEYWORD_STOP_WORDS:
            continue
        if w_clean.isdigit():
            continue
        # 通用 -ing 检测: 任何以 -ing 结尾且不在 domain boost 中的词，99%不是实体
        if w_clean.endswith('ing') and w_clean not in _KEYWORD_DOMAIN_BOOST:
            continue
        
        # 域加权
        boost = _KEYWORD_DOMAIN_BOOST.get(w_clean, 1.0)
        # 缩写加分 (全大写2-5字母 = 可能是重要缩写)
        if w.isupper() and 2 <= len(w) <= 5:
            boost = max(boost, 2.0)
        
        word_scores[w_clean] += boost
    
    # Bigram 检测
    for i in range(len(words) - 1):
        w1 = words[i].lower().rstrip(',.').strip('-')
        w2 = words[i + 1].lower().rstrip(',.').strip('-')
        if (len(w1) >= _ENTITY_MIN_LEN and len(w2) >= _ENTITY_MIN_LEN
                and w1 not in _KEYWORD_STOP_WORDS and w2 not in _KEYWORD_STOP_WORDS
                and not w1.isdigit() and not w2.isdigit()):
            bigram = f"{w1}-{w2}"
            word_scores[bigram] += 1.5
    
    # 过滤 + 排序
    filtered = {w: s for w, s in word_scores.items() if s >= min_score}
    sorted_kw = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    
    return [w for w, _ in sorted_kw[:top_n]]


def llm_extract_entities(texts: list[str], max_entities: int = 5) -> list[str] | None:
    """
    用 LLM 从文本中提取高质量实体 — 替代规则式关键词提取
    
    LLM 理解语义，不会把 'assistant', '2026', 'reducing' 当实体。
    一次调用处理一个 cluster 的所有文本，成本约 300 tokens。
    
    Returns: 实体名列表 or None (API 失败时 fallback 到 extract_keywords)
    """
    combined = "\n".join(f"- {t[:300]}" for i, t in enumerate(texts[:10]))
    prompt = f"""从以下记忆中提取有价值的实体名。只提取真正的专有名词：项目名、工具名、技术名、组织名、金融术语、数据源名。

不要提取：通用动词/名词(user/assistant/system/running/adding)、年份(2026)、停用词、-ing 形式。

记忆内容:
{combined}

只输出 JSON 数组，最多{max_entities}个实体，按重要性排序:
["实体1", "实体2", ...]

无有价值实体时输出: []"""

    result = _call_infini(prompt, max_tokens=200, temperature=0.1)
    if not result:
        return None
    
    # 提取 JSON 数组
    import re
    json_match = re.search(r'\[[\s\S]*?\]', result)
    if not json_match:
        return None
    
    try:
        entities = json.loads(json_match.group())
        if isinstance(entities, list):
            # 基础清洗：空字符串/纯数字/太短的
            clean = [e.strip() for e in entities 
                     if isinstance(e, str) and len(e.strip()) >= 2 
                     and not e.strip().isdigit()]
            return clean[:max_entities] if clean else None
    except json.JSONDecodeError:
        pass
    return None


def extract_entities_with_fallback(texts: list[str], max_entities: int = 5) -> list[str]:
    """
    实体提取：LLM 优先，规则 fallback
    
    LLM 提取的实体质量远高于规则（理解语义、不会提取垃圾词），
    但 API 可能失败（429/超时），此时降级到 extract_keywords()。
    """
    # LLM 优先
    entities = llm_extract_entities(texts, max_entities=max_entities)
    if entities:
        log.info(f"  🤖 LLM 实体提取: {entities}")
        return entities
    
    # Fallback: 规则提取
    log.info(f"  ⚙️ LLM 失败, fallback 到规则提取")
    return extract_keywords(texts, top_n=max_entities, min_score=1.0)


def _is_valid_entity(name: str) -> bool:
    """判断一个字符串是否是有效的实体名 (非停用词、非纯数字、有实际含义)"""
    if not name or len(name) < 2:
        return False
    name_lower = name.lower().rstrip(',.').strip('-')
    if not name_lower:
        return False
    if name_lower in _KEYWORD_STOP_WORDS:
        return False
    if name_lower.isdigit():
        return False
    if name_lower.replace('-', '').isdigit():
        return False
    # 全是同一个字符
    if len(set(name_lower.replace('-', ''))) == 1:
        return False
    # 太短(<=3)且不是已知缩写 → 跳过
    if len(name_lower) <= 3 and name_lower not in _KEYWORD_DOMAIN_BOOST and not name.isupper():
        return False
    # -ing 结尾且不在 domain boost → 几乎不可能是实体
    if name_lower.endswith('ing') and name_lower not in _KEYWORD_DOMAIN_BOOST:
        return False
    return True


def stage1_shallow_sleep(memories: list[dict]) -> dict[str, list[dict]]:
    """
    浅睡: 将记忆按主题聚类分组
    
    方法:
    1. 实体级聚类 (相同项目/技能/话题 → 同组)
    2. 精确去重 (hash)
    3. 文本相似度聚类 (n-gram)
    4. 提取每组的主题关键词
    """
    log.info(f"💤 Stage 1: Shallow Sleep — 聚类 {len(memories)} 条记忆")
    
    clusters: dict[str, list[dict]] = {}  # cluster_id → [memories]
    assigned: dict[str, str] = {}  # memory_id → cluster_id
    
    # ── Phase A: 实体级聚类 (最强信号) ──
    topic_groups: dict[str, list[dict]] = defaultdict(list)
    no_topic: list[dict] = []
    
    for m in memories:
        topic_key = extract_topic_key(m["text"])
        if topic_key:
            topic_groups[topic_key].append(m)
        else:
            no_topic.append(m)
    
    for topic_key, group in topic_groups.items():
        ck = f"topic_{topic_key[:40]}"
        clusters[ck] = group
        for m in group:
            assigned[m["id"]] = ck
    
    log.info(f"  📌 实体聚类: {len(topic_groups)} 主题组, {sum(len(v) for v in topic_groups.values())} 条")
    
    # ── Phase B: 精确去重 (hash) ──
    remaining = [m for m in no_topic if m["id"] not in assigned]
    hash_groups: dict[str, list[dict]] = defaultdict(list)
    for m in remaining:
        h = text_hash(m["text"])
        hash_groups[h].append(m)
    
    exact_dedup_count = 0
    for h, group in hash_groups.items():
        if len(group) > 1:
            ck = f"exact_{h}"
            clusters[ck] = group
            exact_dedup_count += len(group) - 1
            for m in group:
                assigned[m["id"]] = ck
        elif group[0]["id"] not in assigned:
            remaining_unassigned = [m for m in remaining if m["id"] not in assigned]
    
    # ── Phase C: 向量相似度聚类 (pgvector, 精度最高) ──
    remaining = [m for m in memories if m["id"] not in assigned]
    if len(remaining) >= 2:
        remaining_ids = [m["id"] for m in remaining]
        vector_graph = batch_vector_clustering(remaining_ids, max_dist=CLUSTER_DIST)
        
        # 从向量图构建连通分量
        visited: set[str] = set()
        vec_cluster_id = 0
        id_to_mem = {m["id"]: m for m in remaining}
        
        for mid in remaining_ids:
            if mid in visited or mid not in vector_graph:
                continue
            # BFS 找连通分量
            queue = [mid]
            component = []
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for nbr in vector_graph.get(node, []):
                    if nbr not in visited:
                        queue.append(nbr)
            
            if len(component) >= 2:
                ck = f"vec_{vec_cluster_id}"
                clusters[ck] = [id_to_mem[cid] for cid in component if cid in id_to_mem]
                for cid in component:
                    assigned[cid] = ck
                vec_cluster_id += 1
        
        vec_count = sum(1 for k in clusters if k.startswith("vec_"))
        vec_mem_count = sum(len(v) for k, v in clusters.items() if k.startswith("vec_"))
        if vec_count > 0:
            log.info(f"  🔢 向量聚类: {vec_count} 组, {vec_mem_count} 条")
    
    remaining = [m for m in memories if m["id"] not in assigned]
    
    # ── Phase D: 文本相似度聚类 (fallback) ──
    cluster_id = 0
    for i, m1 in enumerate(remaining):
        if m1["id"] in assigned:
            continue
        ck = f"sim_{cluster_id}"
        clusters[ck] = [m1]
        assigned[m1["id"]] = ck
        
        for j in range(i + 1, len(remaining)):
            m2 = remaining[j]
            if m2["id"] in assigned:
                continue
            sim = combined_similarity(m1["text"], m2["text"])
            if sim >= CLUSTER_THRESHOLD:
                clusters[ck].append(m2)
                assigned[m2["id"]] = ck
        
        cluster_id += 1
    
    # Singleton
    for m in memories:
        if m["id"] not in assigned:
            clusters[f"singleton_{m['id'][:8]}"] = [m]
    
    multi_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    log.info(f"  ✅ 聚类完成: {len(clusters)} 组, {len(multi_clusters)} 多条组, "
             f"{exact_dedup_count} 精确重复, {len(topic_groups)} 主题组")
    
    # 统计最有价值的组
    for k, v in sorted(multi_clusters.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        log.info(f"     {k}: {len(v)} 条")
    
    return clusters


# ─── Stage 2: REM (快速眼动) — 重要性评分 ────────────────────────────


def classify_decay_tier(text: str) -> str:
    """
    FadeMem: 分类记忆衰减层级
    
    - volatile (β=1.2): 市场数据、价格、实时新闻 → 快速衰减
    - stable (β=0.8): 用户偏好、基础设施、个人信息 → 慢衰减
    - normal (β=1.0): 默认
    """
    text_lower = text.lower()
    
    # Volatile: market data, prices, time-sensitive
    volatile_kw = [
        "yield", "spread", "bp", "bps", "price", "价格", "利率", "利差",
        "today", "yesterday", "今天", "昨天", "收盘", "开盘", "实时",
        "breaking", "突发", "just announced", "刚发布", "非农", "GDP",
        "CPI", "PMI", "NFP", "Fed meeting", "央行", "data release",
        "stock", "股价", "ticker", "market close", "intraday",
    ]
    
    # Stable: user preferences, infrastructure, personal info
    stable_kw = [
        "prefer", "偏好", "喜欢", "like", "always", "never", "from now on",
        "server", "服务器", "config", "setup", "部署", "infrastructure",
        "password", "key", "credential", "user_id", "api_key",
        "my name", "I am", "我是", "my role", "我的角色", "architecture",
        "framework", "design pattern", "convention", "standard", "规范",
    ]
    
    volatile_hits = sum(1 for kw in volatile_kw if kw in text_lower)
    stable_hits = sum(1 for kw in stable_kw if kw in text_lower)
    
    if volatile_hits >= 2:
        return "volatile"
    elif stable_hits >= 2:
        return "stable"
    elif volatile_hits > stable_hits:
        return "volatile"
    elif stable_hits > volatile_hits:
        return "stable"
    return "normal"


def score_importance(memory: dict, recall_count: int = 0, session_count: int = 0) -> float:
    """
    REM: 6维重要性评分 (对齐 OpenClaw Dreaming)
    
    维度:
    - recency(15%): Ebbinghaus 时间衰减, 14天半衰期
    - frequency(24%): 被搜索/引用次数 (recall_count)
    - query_diversity(15%): 不同 session 命中数
    - domain(20%): 投资(0.9) > 技术(0.6) > 日常(0.3)
    - consolidation(16%): 跨 session 重现强度
    - confidence(10%): 信息密度代理
    """
    scores = {}
    
    # Recency (FadeMem dual-layer: R = e^(-λ·t^β))
    # volatile β=1.2 (fast), stable β=0.8 (slow), normal β=1.0
    try:
        created = memory.get("created_at", "")
        if created:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
            text = memory.get("text", "")
            tier = classify_decay_tier(text)
            half_life = DECAY_HALF_LIVES.get(tier, 7)
            beta = FADEMEM_BETA.get(tier, 1.0)
            # λ = ln(2) / half_life, then R = e^(-λ · t^β)
            lam = math.log(2) / half_life
            retention = math.exp(-lam * (age_days ** beta))
            scores["recency"] = max(RETENTION_FLOOR, retention)
            scores["decay_tier"] = tier  # track for reporting
        else:
            scores["recency"] = 0.5
            scores["decay_tier"] = "normal"
    except:
        scores["recency"] = 0.5
        scores["decay_tier"] = "normal"
    
    # Frequency (被召回次数, 来自 mem0_search 统计)
    scores["frequency"] = min(1.0, math.log2(recall_count + 1) / 4) if recall_count > 0 else 0.1
    
    # Query diversity (跨 session 命中)
    scores["query_diversity"] = min(1.0, session_count / 5) if session_count > 0 else 0.1
    
    # Domain (关键词判定)
    text = memory.get("text", "")
    investment_kw = ["债券", "利率", "利差", "bonds", "yield", "spread", "CGB", "UST",
                     "信用", "carry", "duration", "PM", "trade", "signal", "CME", "COMEX"]
    tech_kw = ["docker", "python", "git", "MCP", "plugin", "skill", "API", "config",
               "Neo4j", "mem0", "cron", "deploy"]
    
    text_lower = text.lower()
    if any(kw.lower() in text_lower for kw in investment_kw):
        scores["domain"] = 0.9
    elif any(kw.lower() in text_lower for kw in tech_kw):
        scores["domain"] = 0.6
    else:
        scores["domain"] = 0.3
    
    # Consolidation (跨天/跨 session 重现)
    # 用 recall_count 和 session_count 的几何平均
    if recall_count >= PROMOTION_MIN_RECALLS and session_count >= PROMOTION_MIN_SESSIONS:
        scores["consolidation"] = min(1.0, math.sqrt(recall_count * session_count) / 4)
    elif session_count >= 2:
        scores["consolidation"] = 0.4
    else:
        scores["consolidation"] = 0.1
    
    # Confidence (长度+唯一词比)
    word_count = len(text.split())
    unique_words = len(set(text.lower().split()))
    if word_count > 50:
        scores["confidence"] = min(1.0, (unique_words / max(word_count, 1)) * 2)
    else:
        scores["confidence"] = 0.3
    
    # Novelty (SCM ValueTagger 核心 — 与已有记忆的语义距离)
    # 新的、意想不到的信息得分更高
    # 用 pgvector 最近邻距离作为 novelty 代理
    # 如果没有向量，用文本相似度 fallback
    try:
        neighbors = get_vector_neighbors(memory["id"], limit=3, max_dist=0.50)
        if neighbors:
            min_dist = min(n["distance"] for n in neighbors)
            # distance 越大 = 越不相似 = 越新奇 → novelty 越高
            scores["novelty"] = min(1.0, min_dist / 0.50)
        else:
            # 无近邻 = 高度独特 = 高 novelty
            scores["novelty"] = 0.9
    except:
        # fallback: 用文本长度和唯一词比估算
        if word_count > 20:
            scores["novelty"] = min(1.0, (unique_words / max(word_count, 1)))
        else:
            scores["novelty"] = 0.5
    
    # 优先标记检测 (永不归档)
    for marker in PERMANENT_MARKERS:
        if marker in text:
            scores["confidence"] = 1.0  # 满分
            break
    
    # 加权平均
    total = sum(IMPORTANCE_WEIGHTS[k] * scores.get(k, 0.1) for k in IMPORTANCE_WEIGHTS)
    return total


# ─── P0: 自适应触发机制 (SCM + SleepGate) ─────────────────────────────

def check_dream_trigger() -> dict:
    """
    检查是否应该触发梦循环
    
    触发条件 (满足任一):
    1. 时间间隔: 距上次梦循环 > TRIGGER_MAX_IDLE_HOURS
    2. 新数据量: 新增未处理记忆 >= TRIGGER_MIN_NEW_MEMORIES
    3. 冲突密度: contradiction_log 中 pending 的比例 > TRIGGER_CONFLICT_DENSITY
    4. 记忆熵: 重要度分布集中度过高 (SCM H > θ_e)
    
    Returns: {"should_trigger": bool, "reasons": [...], "urgency": "low"|"medium"|"high"}
    """
    reasons = []
    urgency = "low"
    
    # 1. 时间间隔检查
    conn = sqlite3.connect(str(DREAM_DB))
    last_run = conn.execute(
        "SELECT finished_at FROM dream_runs WHERE error IS NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    
    if last_run and last_run[0]:
        try:
            last_dt = datetime.fromisoformat(last_run[0])
            idle_hours = (datetime.now(HKT) - last_dt).total_seconds() / 3600
            if idle_hours >= TRIGGER_MAX_IDLE_HOURS:
                reasons.append(f"idle_{idle_hours:.1f}h>={TRIGGER_MAX_IDLE_HOURS}h")
                urgency = "medium"
        except:
            pass
    else:
        reasons.append("no_previous_run")
        urgency = "high"
    
    # 2. 新数据量检查
    new_memories = get_incremental_memories(hours=24)
    if len(new_memories) >= TRIGGER_MIN_NEW_MEMORIES:
        reasons.append(f"new_memories_{len(new_memories)}>={TRIGGER_MIN_NEW_MEMORIES}")
        urgency = "high" if len(new_memories) >= TRIGGER_MIN_NEW_MEMORIES * 3 else urgency
    
    # 3. 冲突密度检查 (从 contradiction_log)
    try:
        conn = sqlite3.connect(str(DREAM_DB))
        total_contra = conn.execute("SELECT COUNT(*) FROM contradiction_log").fetchone()[0]
        pending_contra = conn.execute("SELECT COUNT(*) FROM contradiction_log WHERE resolution='pending'").fetchone()[0]
        conn.close()
        if total_contra > 0:
            conflict_density = pending_contra / total_contra
            if conflict_density >= TRIGGER_CONFLICT_DENSITY:
                reasons.append(f"conflict_density_{conflict_density:.2f}>={TRIGGER_CONFLICT_DENSITY}")
                urgency = "high"
    except:
        pass  # 表可能不存在 (首次运行)
    
    # 4. 记忆熵检查 (重要度分布 — 用 PG 记忆量估算)
    try:
        recent = get_recent_memories(hours=24)
        if len(recent) > 50:
            # 简化熵: 用记忆长度分布的标准差/均值作为 concentration 代理
            lengths = [len(m["text"]) for m in recent]
            mean_len = sum(lengths) / len(lengths)
            if mean_len > 0:
                var_len = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
                cv = (var_len ** 0.5) / mean_len  # coefficient of variation
                # CV 低 = 所有记忆长度相似 = 高集中度 = 高熵
                concentration = 1.0 - min(1.0, cv)
                if concentration >= TRIGGER_MEMORY_ENTROPY:
                    reasons.append(f"entropy_{concentration:.2f}>={TRIGGER_MEMORY_ENTROPY}")
                    urgency = max(urgency, "medium")
    except:
        pass
    
    return {
        "should_trigger": len(reasons) > 0,
        "reasons": reasons,
        "urgency": urgency,
        "new_memory_count": len(new_memories) if new_memories else 0,
    }


# ─── P1: REM 梦游 (SCM 核心 — 随机游走发现隐含关联) ──────────────────

def rem_dream_walk(cluster_entities: list[str] = None) -> list[dict]:
    """
    REM 梦游 v2: Bio-inspired dream engine (from claude-brain)
    
    Upgrades over v1:
    - Creative Teleport (30%): jump to topologically distant but label-shared nodes
    - Revisit Penalty: exp(-visit_count) refractory period
    - Waking Gate: shared neighbor check before promoting provisional edges
    - Edge confidence scoring: path length + shared neighbors → variable confidence
    
    Returns: [{"source": ..., "target": ..., "path": [...], "type": "DREAM_CONNECTION", 
               "confidence": float, "jump_points": [...]}]
    """
    import random, math
    try:
        from neo4j import GraphDatabase
    except ImportError:
        log.warning("⚠️ neo4j driver 未安装, 跳过 REM 梦游")
        return []
    
    # Noise filter shared with v1
    NOISE_NAMES = {'user', 'assistant', 'system', 'from', 'with',
        'that', 'this', 'trade', 'framework', 'data', 'system', 'project',
        'file', 'update', 'change', 'feature', 'lines', 'order', 'skill',
        'memory', 'limit', 'service', 'command', 'parameter', 'value',
        'result', 'output', 'error', 'warning', 'status', 'version',
        'method', 'function', 'code', 'line', 'step', 'process', 'task',
        'action', 'check', 'active', 'repository', 'Related'}
    
    def _revisit_penalty(visit_count: int) -> float:
        """Exponential penalty for revisited nodes (neural refractory period)."""
        if visit_count <= 0:
            return 1.0
        return math.exp(-visit_count)
    
    provisional_edges = []
    promoted_edges = []
    total_jumps = 0
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            # --- Seed selection (v1 logic preserved) ---
            cluster_seed_names = []
            if cluster_entities:
                for ent in cluster_entities[:3]:
                    r = session.run(
                        "MATCH (n:Concept {name: $name}) RETURN n.name as name, COUNT { (n)--() } as degree",
                        name=ent
                    ).data()
                    if r and r[0]["degree"] > 0:
                        cluster_seed_names.append(r[0])
            
            remaining_slots = REM_SEED_COUNT - len(cluster_seed_names)
            neo4j_seeds = []
            if remaining_slots > 0:
                exclude_names = [s["name"] for s in cluster_seed_names] if cluster_seed_names else []
                neo4j_seeds = session.run("""
                    MATCH (n:Concept)
                    WHERE n.name IS NOT NULL AND size(n.name) > 3
                    AND NOT n.name =~ '.*\\d{4}.*'
                    AND NOT n.name ENDS WITH 'ing'
                    AND NOT n.name CONTAINS ','
                    AND NOT n.name IN $noise
                    AND NOT n.name IN $exclude
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n, COUNT(r) as degree
                    ORDER BY degree DESC
                    LIMIT $seed_count
                    RETURN n.name as name, degree
                """, seed_count=remaining_slots, exclude=exclude_names, noise=list(NOISE_NAMES)).data()
            
            seeds = cluster_seed_names + neo4j_seeds
            if not seeds:
                log.info("  💭 REM: 无种子节点，跳过梦游")
                driver.close()
                return []
            
            log.info(f"  💭 REM 梦游 v2: {len(seeds)} 个种子 (jump_p={REM_JUMP_PROBABILITY})")
            
            # --- Walk each seed ---
            for seed in seeds:
                seed_name = seed["name"]
                path = [seed_name]
                visit_counts = {seed_name: 1}
                jump_points = []
                prev_node = None
                current = seed_name
                
                for step in range(REM_WALK_LENGTH):
                    do_jump = random.random() < REM_JUMP_PROBABILITY
                    
                    # Get neighbors
                    neighbors = session.run("""
                        MATCH (c:Concept {name: $name})-[r]-(n:Concept)
                        WHERE n.name IS NOT NULL AND size(n.name) > 3
                        AND NOT n.name =~ '.*\\d{4}.*'
                        AND NOT n.name IN $noise
                        AND n.name <> $prev
                        RETURN n.name as name, COALESCE(r.confidence, 0.5) as weight
                        LIMIT 8
                    """, name=current, prev=prev_node or "", noise=list(NOISE_NAMES)).data()
                    
                    if do_jump or not neighbors:
                        # --- Creative Teleport ---
                        # Jump to a node that shares labels but is NOT a direct neighbor
                        distant = session.run("""
                            MATCH (c:Concept {name: $name})
                            WITH c, labels(c) as my_labels
                            UNWIND my_labels as lbl
                            MATCH (distant:Concept)
                            WHERE lbl IN labels(distant)
                            AND distant.name <> $name
                            AND NOT exists((c)--(distant))
                            AND distant.name IS NOT NULL AND size(distant.name) > 3
                            AND NOT distant.name IN $noise
                            AND NOT distant.name IN $visited
                            OPTIONAL MATCH (distant)-[r2]-()
                            WITH distant, COUNT(r2) as degree
                            WHERE degree > 0
                            RETURN distant.name as name, degree as weight
                            ORDER BY rand()
                            LIMIT 3
                        """, name=current, noise=list(NOISE_NAMES), visited=list(visit_counts.keys())).data()
                        
                        if distant:
                            chosen = random.choice(distant)
                            next_node = chosen["name"]
                            jump_points.append(step)
                            total_jumps += 1
                        elif neighbors:
                            # Fallback to normal neighbor if no distant node found
                            weights = [n["weight"] * _revisit_penalty(visit_counts.get(n["name"], 0))
                                       for n in neighbors]
                            total_w = sum(weights)
                            if total_w <= 0:
                                chosen = random.choice(neighbors)
                            else:
                                r_val = random.random() * total_w
                                cumulative = 0
                                chosen = neighbors[0]
                                for n, w in zip(neighbors, weights):
                                    cumulative += w
                                    if cumulative >= r_val:
                                        chosen = n
                                        break
                            next_node = chosen["name"]
                        else:
                            break  # Dead end
                    else:
                        # --- Normal walk with revisit penalty ---
                        weights = [n["weight"] * _revisit_penalty(visit_counts.get(n["name"], 0))
                                   for n in neighbors]
                        total_w = sum(weights)
                        if total_w <= 0:
                            chosen = random.choice(neighbors)
                        else:
                            r_val = random.random() * total_w
                            cumulative = 0
                            chosen = neighbors[0]
                            for n, w in zip(neighbors, weights):
                                cumulative += w
                                if cumulative >= r_val:
                                    chosen = n
                                    break
                        next_node = chosen["name"]
                    
                    if next_node not in path:
                        path.append(next_node)
                    visit_counts[next_node] = visit_counts.get(next_node, 0) + 1
                    prev_node = current
                    current = next_node
                
                # --- Collect provisional edges (non-adjacent pairs) ---
                if len(path) >= 3:
                    for i in range(len(path)):
                        for j in range(i + 2, len(path)):
                            provisional_edges.append({
                                "source": path[i],
                                "target": path[j],
                                "path": "->".join(path[i:j+1]),
                                "path_len": j - i,
                                "jump_points": jump_points,
                            })
            
            # --- Waking Gate: re-evaluate provisional edges ---
            # Only promote edges where source and target share at least 1 neighbor
            # (topological evidence of real association)
            seen_pairs = set()
            for pe in provisional_edges:
                pair = tuple(sorted((pe["source"], pe["target"])))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                
                # Count shared neighbors
                shared = session.run("""
                    MATCH (a:Concept {name: $src})--(shared:Concept)--(b:Concept {name: $tgt})
                    RETURN count(shared) as cnt
                """, src=pe["source"], tgt=pe["target"]).data()
                
                shared_count = shared[0]["cnt"] if shared else 0
                if shared_count >= WAKING_THRESHOLD:
                    # Confidence based on path length (shorter = stronger) + shared neighbors bonus
                    base_conf = max(0.25, 0.5 - pe["path_len"] * 0.05)
                    neighbor_bonus = min(0.2, shared_count * 0.05)
                    jump_bonus = 0.1 if pe["jump_points"] else 0.0
                    confidence = min(0.7, base_conf + neighbor_bonus + jump_bonus)
                    
                    promoted_edges.append({
                        "source": pe["source"],
                        "target": pe["target"],
                        "path": pe["path"],
                        "type": "DREAM_CONNECTION",
                        "confidence": round(confidence, 3),
                        "shared_neighbors": shared_count,
                        "jump_points": pe["jump_points"],
                    })
        
        driver.close()
    except Exception as e:
        log.warning(f"⚠️ REM 梦游 v2 失败: {e}")
    
    # --- Dedup + limit ---
    seen = set()
    unique_edges = []
    for e in promoted_edges:
        key = f"{e['source']}|{e['target']}"
        rkey = f"{e['target']}|{e['source']}"
        if key not in seen and rkey not in seen:
            seen.add(key)
            unique_edges.append(e)
            if len(unique_edges) >= REM_MAX_NEW_EDGES:
                break
    
    if unique_edges:
        log.info(f"  💭 REM v2: {len(unique_edges)} promoted edges "
                 f"(from {len(provisional_edges)} provisional, {total_jumps} jumps)")
        for e in unique_edges[:5]:
            log.info(f"    {e['source']} → {e['target']} "
                     f"(conf={e['confidence']}, shared={e['shared_neighbors']}, "
                     f"via {e['path'][:60]})")
    else:
        log.info(f"  💭 REM v2: 0 promoted (from {len(provisional_edges)} provisional, "
                 f"{total_jumps} jumps — waking gate filtered all)")
    
    return unique_edges


def rem_shy_downscale() -> dict:
    """
    SHY (Synaptic Homeostasis) Downscaling — from claude-brain.
    
    After dream cycle, rank all edges by weight:
    - Top SHY_PROTECTION_PCT (20%) are protected
    - Remaining edges get gradient downscale (weaker = more downscale)
    - Edges below SHY_PRUNE_THRESHOLD after downscale are removed
    
    This prevents unbounded edge growth while preserving important connections.
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return {"downscaled": 0, "pruned": 0, "protected": 0}
    
    stats = {"downscaled": 0, "pruned": 0, "protected": 0, "total": 0}
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            # Get all edges with confidence, sorted by weight desc
            all_edges = session.run("""
                MATCH (a)-[r]->(b)
                WHERE r.confidence IS NOT NULL
                RETURN elementId(r) as rid, COALESCE(r.confidence, 0.5) as weight,
                       a.name as source, b.name as target, type(r) as rel_type
                ORDER BY weight DESC
            """).data()
            
            stats["total"] = len(all_edges)
            if not all_edges:
                driver.close()
                return stats
            
            protected_count = max(1, int(len(all_edges) * SHY_PROTECTION_PCT))
            unprotected_count = max(1, len(all_edges) - protected_count)
            
            for rank, edge in enumerate(all_edges):
                if rank < protected_count:
                    stats["protected"] += 1
                    continue  # Protected by rank
                
                # Gradient: weakest edges lose the most
                position = (rank - protected_count + 1) / unprotected_count
                scale = 1.0 - SHY_DOWNSCALE_FACTOR * position
                new_weight = edge["weight"] * max(0.0, scale)
                
                if new_weight < SHY_PRUNE_THRESHOLD:
                    # Prune this edge
                    session.run("MATCH ()-[r]->() WHERE elementId(r) = $rid DELETE r", rid=edge["rid"])
                    stats["pruned"] += 1
                else:
                    # Downscale
                    session.run("""
                        MATCH ()-[r]->() WHERE elementId(r) = $rid
                        SET r.confidence = $new_w, r.shy_downscaled = true
                    """, rid=edge["rid"], new_w=round(new_weight, 4))
                    stats["downscaled"] += 1
        
        driver.close()
    except Exception as e:
        log.warning(f"⚠️ SHY downscale 失败: {e}")
    
    return stats


def rem_threat_simulation() -> list[dict]:
    """
    Threat Simulation — from claude-brain.
    
    Scan high-confidence nodes for CONTRADICTS edges and flag them.
    These contradiction alerts surface conflicting information that needs resolution.
    
    Returns: [{"node": ..., "contradicts": ..., "severity": float}]
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return []
    
    threats = []
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            # Find nodes with CONTRADICTS edges
            results = session.run("""
                MATCH (a:Concept)-[r:CONTRADICTS]-(b:Concept)
                WHERE a.name IS NOT NULL AND b.name IS NOT NULL
                WITH a, b, COALESCE(r.confidence, 0.5) as severity
                ORDER BY severity DESC
                LIMIT 20
                RETURN a.name as node, b.name as contradicts, severity
            """).data()
            
            seen = set()
            for r in results:
                pair = tuple(sorted((r["node"], r["contradicts"])))
                if pair in seen:
                    continue
                seen.add(pair)
                threats.append({
                    "node": r["node"],
                    "contradicts": r["contradicts"],
                    "severity": round(r["severity"], 3),
                })
        
        driver.close()
    except Exception as e:
        log.warning(f"⚠️ Threat simulation 失败: {e}")
    
    return threats


def llm_boost_relations(walk_edges: list[dict], clusters: dict[str, list[dict]], 
                         max_boost: int = 10) -> list[dict]:
    """
    LLM Boost: 对梦游发现的低置信度关系(0.3)进行LLM验证并提升
    
    触发条件: 两个实体在clusters中有>=2条共享记忆 (说明关联有事实基础)
    
    流程:
    1. 遍历 walk_edges，检查每对实体的cluster共现频率
    2. 高共现的对 → 调LLM确认关联类型并生成一句话解释
    3. LLM确认 → conf 从 0.3 提升到 0.6
    4. LLM失败或低共现 → 保持 0.3
    
    成本: ~200 tokens/relation, 最多boost 10条
    """
    boosted = []
    not_boosted = []
    
    # 构建实体→记忆映射 (从clusters提取所有记忆文本)
    entity_memories: dict[str, list[str]] = defaultdict(list)
    for ck, group in clusters.items():
        texts = [m["text"][:200] for m in group]
        # 用 extract_entities_with_fallback 提取每个cluster的实体
        entities = extract_entities_with_fallback(texts, max_entities=3)
        for ent in entities:
            entity_memories[ent].extend(texts)
    
    for edge in walk_edges:
        src, tgt = edge.get("source", ""), edge.get("target", "")
        if not src or not tgt:
            not_boosted.append(edge)
            continue
        
        # 检查共现: 两个实体有多少条共享记忆
        src_mems = set(entity_memories.get(src, []))
        tgt_mems = set(entity_memories.get(tgt, []))
        shared = src_mems & tgt_mems
        
        # 也检查: src和tgt是否在同一个cluster出现
        co_cluster_count = 0
        for ck, group in clusters.items():
            texts = [m["text"][:200] for m in group]
            combined_text = " ".join(texts)
            if src.lower() in combined_text.lower() and tgt.lower() in combined_text.lower():
                co_cluster_count += 1
        
        # 触发条件: >=2个cluster共现 OR >=3条共享记忆
        should_boost = co_cluster_count >= 2 or len(shared) >= 3
        
        if not should_boost:
            # 保持低置信度
            edge["boost_attempted"] = False
            edge["boost_reason"] = f"low_cooccurrence(clusters={co_cluster_count},shared_mem={len(shared)})"
            not_boosted.append(edge)
            continue
        
        # LLM 验证
        # 时间感知：检查共享记忆中是否有过期市场数据
        shared_texts = list(shared)[:5] if shared else []
        has_stale_data = any(_is_time_sensitive(t) for t in shared_texts)
        stale_warning = "\n注意：上述记忆片段中可能包含过期的市场数据（价格/利率/利差等），不要基于具体数字建立因果关系，只判断概念层面的结构性关联。" if has_stale_data else ""
        
        prompt = f"""判断以下两个概念之间是否存在有意义的知识关联，并给出关联类型。

概念A: {src}
概念B: {tgt}

相关记忆片段:
{chr(10).join(f'- {t[:150]}...' for t in shared_texts) or '无直接共享记忆'}{stale_warning}

请判断:
1. 关联类型: CAUSE(因果)/PART_OF(包含)/DEPENDS_ON(依赖)/CONTRAST(对比)/EVOLUTION(演变)/RELATED(通用关联)
2. 一句话解释为什么它们关联

只输出一行JSON:
{{"type": "关联类型", "explanation": "一句话解释", "is_valid": true/false}}"""
        
        result = _call_infini(prompt, max_tokens=150, temperature=0.1)
        if result:
            import re
            json_match = re.search(r'\{[^{}]+\}', result)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                    if parsed.get("is_valid", False):
                        edge["confidence"] = 0.6
                        edge["boost_attempted"] = True
                        edge["boost_reason"] = f"llm_verified({parsed.get('type', 'RELATED')}): {parsed.get('explanation', '')[:80]}"
                        edge["relation_type"] = parsed.get("type", "RELATED_TO").replace(" ", "_")
                        boosted.append(edge)
                        log.info(f"  🔥 LLM Boost: {src} → {tgt} conf 0.3→0.6 ({parsed.get('type', '')})")
                        if len(boosted) >= max_boost:
                            break
                        continue
                except json.JSONDecodeError:
                    pass
        
        # LLM失败或判定无效 → 保持 0.3
        edge["boost_attempted"] = True
        edge["boost_reason"] = "llm_failed_or_invalid"
        not_boosted.append(edge)
    
    log.info(f"  🔥 LLM Boost 结果: {len(boosted)} 条提升→0.6, {len(not_boosted)} 条保持0.3")
    return boosted + not_boosted


# ─── P2: NREM Hebbian 强化 (SCM — 强化重要连接) ────────────────────────

def nrem_hebbian_consolidation() -> dict:
    """
    NREM Hebbian 强化: 高重要性概念之间的连接被强化，同时全局缩减
    
    来自 SCM:
    - Hebbian: Δs_ij = η · I(c_i) · I(c_j)
    - Downscaling: s_ij ← α · s_ij (保留相对排名，创建新空间)
    
    Returns: {"strengthened": N, "downscaled": N}
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return {"strengthened": 0, "downscaled": 0}
    
    strengthened = 0
    downscaled = 0
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            # 1. Hebbian 强化: 高度数节点之间的关系加强
            # Neo4j 5.26: 用 COUNT {} 替代 size((a)-[]-())
            result = session.run("""
                MATCH (a:Concept)-[r]->(b:Concept)
                WHERE r.confidence IS NOT NULL
                AND r.source = 'dream_cycle'
                AND COUNT { (a)--() } >= 5
                AND COUNT { (b)--() } >= 5
                RETURN a.name as src, b.name as tgt, 
                       r.confidence as conf, type(r) as rel_type
                LIMIT 50
            """).data()
            
            for row in result:
                old_conf = row.get("conf", 0.5)
                # Hebbian: Δs = η · I(a) · I(b)
                # I(a), I(b) 用 degree 的 log 作为代理
                delta = HEBBIAN_LEARNING_RATE  # 简化: 统一增量
                new_conf = min(1.0, old_conf + delta)
                
                safe_rel_type = ''.join(c for c in row.get("rel_type", "RELATED_TO") 
                                       if c.isalnum() or c == '_') or "RELATED_TO"
                
                session.run(f"""
                    MATCH (a:Concept {{name: $src}})-[r:{safe_rel_type}]->(b:Concept {{name: $tgt}})
                    SET r.confidence = $new_conf,
                        r.last_reinforced = datetime()
                """, src=row["src"], tgt=row["tgt"], new_conf=new_conf)
                strengthened += 1
            
            # 2. 全局缩减: 所有 dream_cycle 关系的 confidence × α
            session.run("""
                MATCH ()-[r]->()
                WHERE r.source = 'dream_cycle' AND r.confidence IS NOT NULL
                SET r.confidence = r.confidence * $alpha
            """, alpha=HEBBIAN_DOWNSCALE)
            
            # 计算被缩减的数量
            downscaled = session.run("""
                MATCH ()-[r]->()
                WHERE r.source = 'dream_cycle'
                RETURN COUNT(r) as cnt
            """).data()
            downscaled = downscaled[0]["cnt"] if downscaled else 0
        
        driver.close()
    except Exception as e:
        log.warning(f"⚠️ NREM Hebbian 失败: {e}")
    
    log.info(f"  🧠 NREM Hebbian: 强化 {strengthened} 条, 缩减 {downscaled} 条")
    return {"strengthened": strengthened, "downscaled": downscaled}


# ─── P3: 语义签名冲突检测 (SleepGate — "同槽不同值") ──────────────────

def detect_slot_conflicts() -> list[dict]:
    """
    语义签名冲突检测: 用 HNSW 索引逐条查找最近邻, 避免全表 cross-join
    
    Returns: [{"mem1": ..., "mem2": ..., "slot_similarity": ..., "value_diff": ...}]
    """
    conflicts = []
    
    # 1. 取最近 7 天有文本的记忆 ID (最多 200 条)
    recent_ids = pg_query("""
        SELECT id::text
        FROM mem0
        WHERE payload->>'archived' IS NULL
        AND LENGTH(payload->>'data') > 30
        AND payload->>'created_at' IS NOT NULL
        AND payload->>'created_at' >= (NOW() - INTERVAL '7 days')::text
        ORDER BY id
        LIMIT 200
    """)
    
    if not recent_ids:
        log.info("  ✅ 无近期记忆, 跳过冲突检测")
        return []
    
    # 2. 对每条记忆, 用 HNSW 索引查最近邻 (高效)
    seen_pairs = set()
    for row in recent_ids:
        mem_id = row[0] if isinstance(row, list) else row
        neighbors = get_vector_neighbors(mem_id, limit=5, max_dist=0.15)
        for n in neighbors:
            nid = n["id"]
            pair_key = tuple(sorted([mem_id, nid]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            # 获取两条记忆的文本
            text_rows = pg_query(f"""
                SELECT a.id::text, LEFT(a.payload->>'data', 300) as t1,
                       b.id::text, LEFT(b.payload->>'data', 300) as t2
                FROM mem0 a, mem0 b
                WHERE a.id::text = '{mem_id}' AND b.id::text = '{nid}'
            """)
            
            if not text_rows or len(text_rows[0]) < 4:
                continue
            r = text_rows[0]
            text1, text2 = r[1], r[3]
            if not text1 or not text2:
                continue
            
            text_sim = combined_similarity(text1, text2)
            vec_dist = n["distance"]
            
            if text_sim < 0.5:  # 语义近但文本差异大 = 槽位冲突
                conflicts.append({
                    "mem1_id": mem_id,
                    "mem1_text": text1,
                    "mem2_id": nid,
                    "mem2_text": text2,
                    "slot_similarity": 1.0 - vec_dist,
                    "value_diff": 1.0 - text_sim,
                    "type": "slot_conflict",
                })
                
                if len(conflicts) >= 10:
                    break
        
        if len(conflicts) >= 10:
            break
    
    if conflicts:
        log.info(f"  🔍 语义签名冲突: {len(conflicts)} 对'同槽不同值'")
        for c in conflicts[:3]:
            log.info(f"    [{c['slot_similarity']:.2f}槽似, {c['value_diff']:.2f}值差] "
                     f"{c['mem1_text'][:60]}... vs {c['mem2_text'][:60]}...")
    else:
        log.info("  ✅ 无语义签名冲突")
    
    return conflicts

def stage2_rem(clusters: dict[str, list[dict]], neo4j_connections: dict = None) -> dict:
    """
    REM: 7维评分 + 三重门限提升 + 矛盾检测
    
    对每条记忆打分, 对每组:
    - 标记最重要的记忆 (keep)
    - 标记可合并/删除的候选
    - 标记 Vault 沉淀候选 (三重门限: 分数+召回+跨session)
    - 检测矛盾事实 (KektorDB Gardener 模式)
    """
    log.info(f"👁️ Stage 2: REM — 评分 {sum(len(v) for v in clusters.values())} 条记忆")
    
    results = {
        "boosted": [],           # 高重要性, 建议 Boost
        "dedup_candidates": [],  # 可去重
        "merge_candidates": [],  # 可合并
        "vault_candidates": [],  # 建议沉淀到 Vault (三重门限)
        "decay_candidates": [],  # 建议归档 (永不删除)
        "contradictions": [],    # 矛盾事实对
    }
    
    # P2-1: 加载真实 recall 统计
    recall_stats = get_recall_stats()
    
    for cluster_key, group in clusters.items():
        # 评分: 优先用真实 recall 统计, fallback 到启发式
        scored = []
        for m in group:
            # P2-1: 真实 recall 匹配
            real_rc, real_sc = match_memory_to_queries(m.get("text", ""), recall_stats)
            
            if real_rc > 0:
                # 有真实搜索命中 → 用真实数据
                rc = real_rc
                sc = real_sc
            else:
                # fallback: 启发式估算
                rc = len(group)  # 同组数 ≈ 被一起召回的次数
                sc = len(set(mem.get("created_at", "")[:10] for mem in group))  # 不同日期数
            
            s = score_importance(m, recall_count=rc, session_count=sc)
            scored.append((m, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # 矛盾检测 (两阶段: 关键词预筛 + LLM 验证)
        # Phase 1: 关键词预筛 — 只匹配中文高置信矛盾（收紧英文模式）
        # P2-3: 加主题重叠过滤 — 两条记忆必须共享关键实体才视为矛盾候选
        CONTRADICTION_MARKERS = [
            ("并非", "而是"), ("不再", "改为"), ("已从", "变为"),
            ("已从", "迁到"), ("不再是", "现在是"),
        ]
        
        def _extract_key_nouns(text: str) -> set[str]:
            """提取文本中的关键名词/实体（简易版）"""
            import re
            # 英文: 大写开头的词 + 全大写的缩写
            en_nouns = set(re.findall(r'\b[A-Z][a-z]{2,}\b|\b[A-Z]{2,}\b', text))
            # 中文: 提取2-4字的中文词组（粗粒度实体）
            cn_nouns = set(re.findall(r'[\u4e00-\u9fff]{2,4}', text))
            # 过滤常见停用词
            stop = {'The', 'This', 'That', 'What', 'How', 'When', 'Where', 'Which',
                    '可以', '但是', '因为', '所以', '如果', '已经', '不是', '而是',
                    '通过', '使用', '进行', '需要', '目前', '现在', '之前', '之后'}
            return (en_nouns | cn_nouns) - stop
        
        def _has_subject_overlap(text1: str, text2: str, min_overlap: int = 1) -> bool:
            """检查两条记忆是否共享关键实体（主题重叠）"""
            nouns1 = _extract_key_nouns(text1)
            nouns2 = _extract_key_nouns(text2)
            overlap = nouns1 & nouns2
            return len(overlap) >= min_overlap
        
        if len(group) >= 2:
            texts = [m["text"] for m, _ in scored]
            texts_lower = [t.lower() for t in texts]
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    t1, t2 = texts_lower[i], texts_lower[j]
                    matched_marker = None
                    for marker_pair in CONTRADICTION_MARKERS:
                        if isinstance(marker_pair, tuple) and len(marker_pair) == 2:
                            if marker_pair[0] in t1 and marker_pair[1] in t2:
                                matched_marker = f"{marker_pair[0]} vs {marker_pair[1]}"
                                break
                            elif marker_pair[1] in t1 and marker_pair[0] in t2:
                                matched_marker = f"{marker_pair[1]} vs {marker_pair[0]}"
                                break
                    # P2-3: 只有主题重叠的对才标记矛盾
                    if matched_marker and _has_subject_overlap(texts[i], texts[j]):
                        results["contradictions"].append({
                            "mem1": scored[i][0], "mem2": scored[j][0],
                            "marker": matched_marker,
                            "score_diff": abs(scored[i][1] - scored[j][1]),
                            "verified": False,  # 待 LLM 验证
                        })
        
        if len(group) == 1:
            m, s = scored[0]
            # P10: 年龄驱动衰减旁路 — >90天且非高重要性 → 强制衰减
            try:
                created = m.get("created_at", "")
                if created:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400
                    if age_days > ARCHIVE_THRESHOLD_DAYS and s < 0.50:
                        results["decay_candidates"].append({
                            "memory": m, "score": s,
                            "reason": f"age_based({age_days:.0f}d>{ARCHIVE_THRESHOLD_DAYS}d,s={s:.2f})"
                        })
                        continue
            except:
                pass
            if s > 0.7:
                results["boosted"].append({"memory": m, "score": s, "reason": "singleton_high_importance"})
            elif s < 0.25:
                results["decay_candidates"].append({"memory": m, "score": s, "reason": "singleton_low_importance"})
            continue
        
        # 多条组: 向量相似度(优先) + 文本相似度(fallback) 判定
        best, best_score = scored[0]
        
        # P4: 多条组也产生boost和decay (原来只给singleton产生)
        for m, s in scored:
            if s >= 0.7:
                results["boosted"].append({"memory": m, "score": s, "reason": "high_importance_in_cluster"})
            elif s < 0.25:
                results["decay_candidates"].append({"memory": m, "score": s, "reason": "low_importance_in_cluster"})
        
        # P2 批量向量去重
        group_ids = [m["id"] for m, _ in scored]
        vec_dedup_cache: dict[str, dict] = {}
        if len(group_ids) >= 2:
            vec_graph = batch_vector_clustering(group_ids, max_dist=DEDUP_DIST)
            # 构建 id→neighbors 映射 (含距离)
            for mid in group_ids:
                vec_dedup_cache[mid] = {}
            # 改用: 一次批量查询 best 对所有 group 成员的向量距离
            best_id = best["id"]
            batch_sql = f"""
                SELECT b.id::text,
                       ROUND((a.vector <=> b.vector)::numeric, 4) as dist
                FROM mem0 a, mem0 b
                WHERE a.id::text = '{best_id}'
                AND b.id::text IN ('{"','".join(group_ids)}')
                AND a.id != b.id
                AND (a.vector <=> b.vector) < 0.20
            """
            batch_rows = pg_query(batch_sql)
            for r in batch_rows:
                if len(r) >= 2:
                    dist_val = safe_float(r[1])
                    if dist_val is not None:
                        vec_dedup_cache[r[0]]["vec_dist_to_best"] = dist_val
        
        for m, s in scored[1:]:
            # P2: 优先用向量距离判断 (pgvector, 精度最高)
            vec_dist = vec_dedup_cache.get(m["id"], {}).get("vec_dist_to_best")
            
            if vec_dist is not None:
                # 余弦距离 → 相似度: sim = 1 - dist
                vec_sim = 1.0 - vec_dist
                if vec_dist < DEDUP_DIST:  # dist<0.10 → sim>0.90 → 精确重复
                    results["dedup_candidates"].append({
                        "keep": best, "remove": m, "similarity": vec_sim,
                        "distance": vec_dist, "method": "vector",
                        "keep_score": best_score, "remove_score": s
                    })
                    continue
                elif vec_dist < MERGE_DIST:  # dist<0.18 → sim>0.82 → 可合并
                    results["merge_candidates"].append({
                        "primary": best, "secondary": m, "similarity": vec_sim,
                        "distance": vec_dist, "method": "vector",
                        "primary_score": best_score, "secondary_score": s
                    })
                    continue
            
            # Fallback: 文本相似度 (ngram+jaccard)
            sim_val = combined_similarity(best["text"], m["text"])
            
            if sim_val >= DEDUP_THRESHOLD:
                results["dedup_candidates"].append({
                    "keep": best, "remove": m, "similarity": sim_val,
                    "method": "ngram",
                    "keep_score": best_score, "remove_score": s
                })
            elif sim_val >= MERGE_THRESHOLD:
                results["merge_candidates"].append({
                    "primary": best, "secondary": m, "similarity": sim_val,
                    "method": "ngram",
                    "primary_score": best_score, "secondary_score": s
                })
        
        # 整组重要性高 → Vault 候选
        # P4: 三重门限放松 — 全通过=high_priority, 任一通过=normal_priority
        passes_all = (best_score >= PROMOTION_MIN_SCORE 
                      and len(group) >= PROMOTION_MIN_RECALLS
                      and len(set(mem.get("created_at", "")[:10] for mem, _ in scored)) >= PROMOTION_MIN_SESSIONS)
        passes_any = (best_score >= 0.5 or len(group) >= 2)
        
        if passes_all or passes_any:
            # 使用 LLM 实体提取 (优先) + 规则 fallback
            all_texts = [mem["text"] for mem, _ in scored]
            top_keywords = extract_entities_with_fallback(all_texts, max_entities=5)
            
            # 时间感知选sample：优先选最新的记忆文本，避免过期市场数据进入概览
            freshest = min(scored, key=lambda ms: _compute_memory_age_days(ms[0].get("created_at")) or 9999)
            sample_text = freshest[0]["text"][:200]
            sample_age = _compute_memory_age_days(freshest[0].get("created_at"))
            
            results["vault_candidates"].append({
                "cluster": cluster_key,
                "memories": [mem["id"] for mem, _ in scored],
                "best_score": best_score,
                "recall_count": len(group),
                "session_count": len(set(mem.get("created_at", "")[:10] for mem, _ in scored)),
                "keywords": top_keywords,
                "sample_text": sample_text,
                "sample_age_days": sample_age,
                "promotion_pass": "all_3_gates" if passes_all else "any_gate",
                "priority": "high" if passes_all else "normal",
            })
    
    # ── Phase 2: LLM 验证预筛矛盾 (限10个，避免 API 过载) ──
    unverified = [c for c in results["contradictions"] if not c.get("verified", False)]
    if unverified:
        to_verify = unverified[:10]  # 最多验证10个
        verified_count = 0
        false_positive_count = 0
        api_fail_count = 0
        for c in to_verify:
            v = llm_verify_contradiction(
                c["mem1"]["text"], c["mem2"]["text"], c["marker"]
            )
            if v is None:
                api_fail_count += 1
                c["verified"] = "api_failed"
            elif v.get("type") == "FALSE_POSITIVE" or not v.get("is_contradiction", False):
                c["verified"] = "false_positive"
                c["llm_explanation"] = v.get("explanation", "")
                false_positive_count += 1
            else:
                c["verified"] = True
                c["contradiction_type"] = v.get("type", "SUPERSEDE")
                c["llm_explanation"] = v.get("explanation", "")
                verified_count += 1
        
        # 过滤掉误报，只保留确认的真矛盾和 API 失败的
        results["contradictions"] = [
            c for c in results["contradictions"]
            if c.get("verified") is True or c.get("verified") == "api_failed"
        ]
        log.info(f"  🔍 LLM 矛盾验证: {verified_count} 确认, "
                 f"{false_positive_count} 误报, {api_fail_count} API失败")
    
    log.info(f"  ✅ REM 完成: {len(results['boosted'])} boost, "
             f"{len(results['dedup_candidates'])} dedup, "
             f"{len(results['merge_candidates'])} merge, "
             f"{len(results['vault_candidates'])} vault, "
             f"{len(results['decay_candidates'])} decay, "
             f"{len(results['contradictions'])} contradictions")
    
    # P2-2: 跨聚类实体共现 → 关系推断
    # 扫描所有 cluster 的实体，找出在 ≥2 个 cluster 共现的实体对
    cross_cluster_relations = []
    entity_clusters: dict[str, set[str]] = defaultdict(set)
    for cluster_key, group in clusters.items():
        cluster_texts = [m["text"][:200] for m in group]
        entities = extract_entities_with_fallback(cluster_texts, max_entities=5)
        for ent in entities:
            if _is_valid_entity(ent):
                entity_clusters[ent].add(cluster_key)
    
    entity_list = list(entity_clusters.keys())
    co_pairs = []
    for i in range(len(entity_list)):
        for j in range(i + 1, len(entity_list)):
            e1, e2 = entity_list[i], entity_list[j]
            shared = entity_clusters[e1] & entity_clusters[e2]
            if len(shared) >= 1:  # P3: 降低阈值，1个共享cluster即可
                co_pairs.append((e1, e2, len(shared)))
    
    co_pairs.sort(key=lambda x: x[2], reverse=True)
    max_rels = 50  # P3: 增加到50条
    for e1, e2, count in co_pairs[:max_rels]:
        conf = min(0.7, 0.3 + count * 0.1)
        cross_cluster_relations.append({
            "source": e1, "target": e2,
            "type": "RELATED_TO", "confidence": conf,
        })
    
    results["cross_cluster_relations"] = cross_cluster_relations
    if cross_cluster_relations:
        log.info(f"  🔗 P2-2 跨聚类共现: {len(co_pairs)} 对, 取 top {len(cross_cluster_relations)}")
    
    return results


# ─── Stage 3: Deep Sleep (深睡) — 整合行动 ────────────────────────────

def stage3_deep_sleep(rem_results: dict, dream_run_id: int, dry_run: bool = False, total_memories: int = 0, total_clusters: int = 0) -> dict:
    """
    Deep Sleep: 执行整合行动
    
    1. 去重: 删除重复记忆
    2. 合并: 合并近似记忆
    3. 关系推断: 识别实体间新关系 → 写入 Neo4j
    4. 衰减清理: 标记低价值记忆
    5. Vault 建议: 生成沉淀建议
    """
    log.info(f"🌊 Stage 3: Deep Sleep — 执行整合{' (dry-run)' if dry_run else ''}")
    
    stats = {
        "deduped": 0,
        "merged": 0,
        "inferred": 0,
        "decayed": 0,
        "vault_suggestions": 0,
    }
    
    conn = sqlite3.connect(str(DREAM_DB))
    
    # P4: 0. Boost — 高重要性记忆标记 (原来只产生不执行)
    # P5: 同时标注 freshness=fresh (和 mem0 plugin Ebbinghaus 联动)
    for item in rem_results.get("boosted", []):
        m = item["memory"]
        score = item["score"]
        if not dry_run:
            # 写入 PG payload: dream_boost + freshness=fresh (P5 联动)
            reason = item.get('reason', '').replace('"', '').replace("'", '')
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{"dream_boost": true, "boost_score": {score:.3f}, "boost_reason": "{reason}", "boosted_at": "{datetime.now(HKT).isoformat()}", "freshness": "fresh", "freshness_source": "dream_cycle_boost"}}' WHERE id::text = '{m['id']}'""")
        log.info(f"  🔥 Boost: {m['id'][:8]} score={score:.3f} ({item.get('reason', '')})")
    stats["boosted"] = len(rem_results.get("boosted", []))
    
    # 1. 去重 → 归档 (永不删除, Auto-Dream 模式)
    for item in rem_results.get("dedup_candidates", []):
        remove_id = item["remove"]["id"]
        keep_id = item["keep"]["id"]
        sim = item["similarity"]
        
        # 优先标记检测 — 带标记的永不归档
        remove_text = item["remove"].get("text", "")
        is_permanent = any(marker in remove_text for marker in PERMANENT_MARKERS)
        if is_permanent:
            log.info(f"  📌 优先标记保护, 跳过归档: {remove_id[:8]}")
            continue
        
        log.info(f"  🗑️ 去重→归档: remove={remove_id[:8]} (sim={sim:.3f}, keep={keep_id[:8]})")
        
        if not dry_run:
            # 归档而非删除: 写入归档标记到 payload
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{\"archived\": true, \"archived_reason\": \"dedup\", \"archived_at\": \"{datetime.now(HKT).isoformat()}\", \"superseded_by\": \"{keep_id}\"}}' WHERE id::text = '{remove_id}'""")
            stats["deduped"] += 1
            conn.execute(
                "INSERT INTO dedup_log (dream_run_id, kept_id, removed_id, similarity) VALUES (?, ?, ?, ?)",
                (dream_run_id, keep_id, remove_id, sim)
            )
        else:
            stats["deduped"] += 1
    
    # 2. 合并 (用 LLM 生成摘要, 删除被合并的)
    merged_texts_log = []
    for item in rem_results.get("merge_candidates", []):
        primary = item["primary"]
        secondary = item["secondary"]
        
        log.info(f"  🔄 合并候选: {primary['id'][:8]} ← {secondary['id'][:8]} "
                 f"(dist={item.get('distance', 'N/A')}, method={item.get('method', 'N/A')})")
        
        if not dry_run:
            merged_text = llm_merge_memories([primary["text"], secondary["text"]])
            if merged_text:
                # 更新 primary 的文本
                if update_memory_text(primary["id"], merged_text):
                    # 删除 secondary
                    if delete_memory(secondary["id"]):
                        stats["merged"] += 1
                        conn.execute(
                            "INSERT INTO dedup_log (dream_run_id, kept_id, removed_id, similarity, merged_text) VALUES (?, ?, ?, ?, ?)",
                            (dream_run_id, primary["id"], secondary["id"],
                             item.get("similarity", 0), merged_text[:500])
                        )
                        merged_texts_log.append(f"{primary['id'][:8]}←{secondary['id'][:8]}: {merged_text[:80]}")
            else:
                log.info(f"    ⏭️ LLM 合并失败, 保留两条")
    
    # 3. 关系推断 + Neo4j 回写
    neo4j_relations = []
    
    # P2-2: 跨聚类实体共现 → 关系推断 (大幅增加关系产出)
    # 从 rem_results 获取预计算的跨聚类关系
    for rel in rem_results.get("cross_cluster_relations", []):
        conn.execute(
            "INSERT INTO relation_log (dream_run_id, source_entity, target_entity, relation_type, confidence, method) VALUES (?, ?, ?, ?, ?, ?)",
            (dream_run_id, rel["source"], rel["target"], rel["type"], rel["confidence"], "cross_cluster_cooccurrence")
        )
        stats["inferred"] += 1
        neo4j_relations.append(rel)
    if rem_results.get("cross_cluster_relations"):
        log.info(f"  🔗 跨聚类共现: {len(rem_results['cross_cluster_relations'])} 条新关系")
    
    # 原有: vault_candidates 关键词关系
    for item in rem_results.get("vault_candidates", []):
        keywords = item.get("keywords", [])
        # 过滤无效实体
        valid_keywords = [k for k in keywords if _is_valid_entity(k)]
        if len(valid_keywords) < 2:
            continue  # 至少需要两个有效实体才能建关系
        # 两两配对推断关系
        for i in range(len(valid_keywords)):
            for j in range(i + 1, min(i + 3, len(valid_keywords))):
                rel_type = "RELATED_TO"  # 通用关系
                conn.execute(
                    "INSERT INTO relation_log (dream_run_id, source_entity, target_entity, relation_type, confidence, method) VALUES (?, ?, ?, ?, ?, ?)",
                    (dream_run_id, valid_keywords[i], valid_keywords[j], rel_type, 0.4, "dream_keyword_cooccurrence")
                )
                stats["inferred"] += 1
                neo4j_relations.append({
                    "source": valid_keywords[i], "target": valid_keywords[j],
                    "type": rel_type, "confidence": 0.4
                })
    
    # P2: 关系去重 — 同源同目标的已有关系不重复写入
    if neo4j_relations and not dry_run:
        # 去重: 查 Neo4j 已有关系，过滤重复
        deduped_relations = dedup_neo4j_relations(neo4j_relations)
        neo4j_written = write_relations_to_neo4j(deduped_relations)
        log.info(f"  🔗 Neo4j 回写: {neo4j_written}/{len(deduped_relations)} 关系 "
                 f"(去重 {len(neo4j_relations)-len(deduped_relations)} 条)")
    
    # 3b. SHY Downscaling (Synaptic Homeostasis — from claude-brain)
    # After writing new edges, downscale weak edges globally to prevent unbounded growth
    if not dry_run:
        shy_stats = rem_shy_downscale()
        if shy_stats["total"] > 0:
            log.info(f"  🧬 SHY: {shy_stats['protected']} protected, "
                     f"{shy_stats['downscaled']} downscaled, {shy_stats['pruned']} pruned "
                     f"(of {shy_stats['total']} total edges)")
    
    # 3c. Threat Simulation (from claude-brain)
    # Scan for contradiction edges between high-confidence nodes
    threats = rem_threat_simulation()
    if threats:
        log.info(f"  ⚠️ Threat: {len(threats)} contradiction edges detected")
        for t in threats[:3]:
            log.info(f"    {t['node']} ↔ {t['contradicts']} (severity={t['severity']})")
    
    # 4. 衰减候选 → 归档 (永不删除)
    # P5: 同时更新 PG payload 的 freshness 字段 (和 mem0 plugin Ebbinghaus 联动)
    for item in rem_results.get("decay_candidates", []):
        m = item["memory"]
        score = item["score"]
        # P5: 根据 score 判断 freshness 标签 (和 mem0 plugin 对齐)
        if score < 0.15:
            freshness = "outdated"
        elif score < 0.25:
            freshness = "stale"
        else:
            freshness = "aging"
        log.info(f"  📉 衰减→归档候选: {m['id'][:8]} (score={score:.3f}, freshness={freshness})")
        if not dry_run:
            # 归档: 写入归档标记到 payload + P5 freshness 联动
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{"archived": true, "archived_reason": "decay", "archived_at": "{datetime.now(HKT).isoformat()}", "decay_score": {score:.3f}, "freshness": "{freshness}", "freshness_source": "dream_cycle_decay"}}' WHERE id::text = '{m['id']}'""")
            mark_manifest_archived([m["id"]])
        stats["decayed"] += 1
    
# 4. 矛盾报告 + P7 自动处理
    contradictions = rem_results.get("contradictions", [])
    if contradictions:
        log.info(f"  ⚡ 矛盾检测: 发现 {len(contradictions)} 对矛盾")
        for c in contradictions[:5]:
            log.info(f"    {c['marker']}: {c['mem1']['id'][:8]} vs {c['mem2']['id'][:8]}")
        
        # P7: 语义签名冲突自动处理 (slot_conflict 类型)
        slot_conflicts = rem_results.get("slot_conflicts_list", [])
        if slot_conflicts and not dry_run:
            processed_conflicts = resolve_slot_conflicts(slot_conflicts, dream_run_id)
            stats["conflicts_resolved"] = len(processed_conflicts)
    
    # 6. 健康评分 (来自 Auto-Dream 5维)
    archived_count = sum(1 for _ in rem_results.get("decay_candidates", []))
    vault_count = len(rem_results.get("vault_candidates", []))
    contradiction_count = len(contradictions)
    
    health = {
        "freshness": min(1.0, (total_memories - archived_count) / max(total_memories, 1)),
        "coverage": min(1.0, vault_count / max(total_clusters, 1)),
        "coherence": min(1.0, 1.0 - contradiction_count / max(total_memories, 1)),
        "efficiency": min(1.0, 1.0 - stats.get("deduped", 0) / max(total_memories, 1)),
        "reachability": min(1.0, stats.get("inferred", 0) / max(total_memories * 0.5, 1)),
    }
    health_score = sum(health[k] * w for k, w in [
        ("freshness", 0.25), ("coverage", 0.25), ("coherence", 0.20),
        ("efficiency", 0.15), ("reachability", 0.15)
    ]) * 100
    
    log.info(f"  💊 健康评分: {health_score:.0f}/100 "
             f"(fresh={health['freshness']:.0%} cov={health['coverage']:.0%} "
             f"coh={health['coherence']:.0%} eff={health['efficiency']:.0%} "
             f"reach={health['reachability']:.0%})")
    
    stats["health_score"] = round(health_score, 1)
    stats["contradictions"] = contradiction_count
    
    # 6. Vault 建议 + 自动沉淀
    for item in rem_results.get("vault_candidates", []):
        # 使用高质量关键词 — 只取有效实体
        keywords = item.get("keywords", [])
        valid_keywords = [k for k in keywords if _is_valid_entity(k)]
        if not valid_keywords:
            continue  # 无有效关键词, 跳过
        
        # 推断 category — 基于关键词的领域加权
        sample = item.get("sample_text", "")
        sample_age = item.get("sample_age_days")
        keywords_lower = [k.lower() for k in valid_keywords]
        investment_kw = {'bonds', 'yield', 'spread', 'cgb', 'ust', 'carry', 'duration',
                        'credit', 'curve', 'swap', 'basis', 'delivery', 'bond', 'rate',
                        'inflation', 'fed', 'ecb', 'boj', 'macro', 'fiscal', 'monetary',
                        'hedge', 'position', 'flow', 'premium', 'sovereign', 'cme', 'comex'}
        tech_kw = {'docker', 'mcp', 'plugin', 'mem0', 'neo4j', 'config', 'deploy', 'cron', 'hermes'}
        
        if any(k in investment_kw for k in keywords_lower):
            category = "markets"
        elif any(k in tech_kw for k in keywords_lower):
            category = "projects"
        else:
            category = "concepts"
        
        # Vault 建议实体名筛选: 优先选有领域加权的词
        domain_keywords = [k for k in valid_keywords if k.lower() in _KEYWORD_DOMAIN_BOOST]
        entity_name = domain_keywords[0] if domain_keywords else valid_keywords[0]
        
        # 如果唯一的实体名太通用 (不在 domain boost 且不是复合词/缩写), 跳过
        if (entity_name.lower() not in _KEYWORD_DOMAIN_BOOST 
            and '-' not in entity_name 
            and not entity_name.isupper()
            and len(entity_name) < 6):
            continue
        
        conn.execute(
            "INSERT INTO vault_suggestion (dream_run_id, entity, category, frequency, reason) VALUES (?, ?, ?, ?, ?)",
            (dream_run_id, entity_name, category,
             len(item.get("memories", [])), f"score={item.get('best_score', 0):.2f}")
        )
        stats["vault_suggestions"] += 1
        
        # 自动沉淀: 创建 Vault 页面骨架 (P3: 门槛从3条降到2条)
        if not dry_run and len(item.get("memories", [])) >= 2:
            vault_path = create_vault_stub(
                entity=entity_name,
                category=category,
                keywords=valid_keywords,
                sample=sample,
                sample_age_days=sample_age,
            )
            if vault_path:
                stats["vault_created"] = stats.get("vault_created", 0) + 1
                # P3: 自动沉淀后标记状态
                conn.execute(
                    "UPDATE vault_suggestion SET status = 'auto_created' WHERE entity = ? AND status = 'pending'",
                    (entity_name,)
                )
    
    conn.commit()
    conn.close()
    
    log.info(f"  ✅ Deep Sleep 完成: deduped={stats['deduped']}, merged={stats['merged']}, "
             f"inferred={stats['inferred']}, decay={stats['decayed']}, vault={stats['vault_suggestions']}")
    
    return stats


# ─── 在线去冗余 (写入时) ──────────────────────────────────────────────

def online_dedup_check(text: str, threshold: float = 0.85) -> dict:
    """
    写入前去冗余检查 — 不等 04:00 梦循环，写入时立刻检查
    
    被 mem0 plugin 的 mem0_conclude 调用:
    1. 计算 text hash → 精确去重
    2. pgvector 查最近邻 → 语义去重
    3. 返回建议: SKIP(重复) / MERGE(近似) / ADD(新增)
    
    Args:
        text: 待写入的文本
        threshold: 语义相似度阈值 (余弦距离, 越小越相似)
    
    Returns:
        {"action": "ADD"|"SKIP"|"MERGE", "reason": "...", "similar_id": "...", "similarity": float}
    """
    # 1. 精确去重: hash 检查
    h = text_hash(text)
    hash_match = pg_query(f"""
        SELECT id::text, LEFT(payload->>'data', 200) as sample
        FROM mem0
        WHERE payload->>'hash' = '{h}'
        AND payload->>'archived' IS NULL
        LIMIT 1
    """)
    if hash_match:
        return {
            "action": "SKIP",
            "reason": f"exact_hash_match",
            "similar_id": hash_match[0][0],
            "similarity": 1.0,
        }
    
    # 2. 语义去重: pgvector 查最近邻
    # 先找最近插入的向量做 anchor
    # (需要先插入向量才能查，所以改用文本相似度快速筛选)
    # 快速文本扫描: 最近100条记忆
    recent = get_recent_memories(hours=72)  # 3天窗口
    best_sim = 0.0
    best_match = None
    
    for m in recent:
        sim = combined_similarity(text, m["text"])
        if sim > best_sim:
            best_sim = sim
            best_match = m
    
    # 余弦距离阈值换算: distance < 0.15 ≈ similarity > 0.85
    if best_sim >= DEDUP_THRESHOLD and best_match:
        return {
            "action": "SKIP",
            "reason": f"semantic_exact (sim={best_sim:.3f})",
            "similar_id": best_match["id"],
            "similarity": best_sim,
        }
    elif best_sim >= MERGE_THRESHOLD and best_match:
        return {
            "action": "MERGE",
            "reason": f"semantic_merge (sim={best_sim:.3f})",
            "similar_id": best_match["id"],
            "similarity": best_sim,
        }
    
    return {"action": "ADD", "reason": "new_unique_memory", "similar_id": None, "similarity": 0.0}


# ─── 主循环 ────────────────────────────────────────────────────────────

def _acquire_lock() -> bool:
    """获取并发锁，防止多个 dream cycle 同时运行"""
    import os, signal
    
    if DREAM_LOCK.exists():
        try:
            pid = int(DREAM_LOCK.read_text().strip())
            # 检查进程是否还活着
            os.kill(pid, 0)
            # 检查是否超时
            lock_age = time.time() - DREAM_LOCK.stat().st_mtime
            if lock_age > DREAM_LOCK_TIMEOUT:
                log.warning(f"🔓 锁超时 ({lock_age:.0f}s > {DREAM_LOCK_TIMEOUT}s), PID {pid} 可能是僵尸, 强制接管")
            else:
                log.error(f"🔒 另一个 dream cycle 正在运行 (PID {pid}, {lock_age:.0f}s 前)")
                return False
        except (ProcessLookupError, ValueError):
            log.warning(f"🔓 发现过期锁文件 (进程已死), 清理并继续")
    
    DREAM_LOCK.write_text(str(os.getpid()))
    return True

def _release_lock():
    """释放并发锁"""
    try:
        DREAM_LOCK.unlink(missing_ok=True)
    except Exception:
        pass

def _cleanup_zombie_runs():
    """清理僵尸 run (started 但 finished_at=NULL 超过 1 小时)"""
    conn = sqlite3.connect(str(DREAM_DB))
    cutoff = (datetime.now(HKT) - timedelta(hours=1)).isoformat()
    cursor = conn.execute("""
        UPDATE dream_runs 
        SET finished_at = ?, error = 'zombie: no finish after 1h'
        WHERE finished_at IS NULL AND started_at < ?
    """, (datetime.now(HKT).isoformat(), cutoff))
    cleaned = cursor.rowcount
    conn.commit()
    conn.close()
    if cleaned > 0:
        log.info(f"🧹 清理了 {cleaned} 个僵尸 run")

def run_dream_cycle(hours: int = 48, dry_run: bool = False, stages: str = "123") -> dict:
    """
    执行完整的梦循环
    
    Args:
        hours: 回溯多少小时的记忆
        dry_run: 只分析不执行
        stages: 执行哪些阶段 (1/2/3/12/23/123)
    """
    start_time = datetime.now(HKT)
    log.info(f"🌙 梦循环启动 @ {start_time.strftime('%Y-%m-%d %H:%M:%S')} HKT "
             f"(hours={hours}, dry_run={dry_run}, stages={stages})")
    
    # 并发锁 — 防止多个实例同时运行
    if not _acquire_lock():
        return {"status": "skipped", "reason": "another_instance_running"}
    
    # 清理僵尸 run
    _cleanup_zombie_runs()
    
    # 初始化
    dream_conn = init_dream_db()
    cursor = dream_conn.execute(
        "INSERT INTO dream_runs (started_at) VALUES (?)",
        (start_time.isoformat(),)
    )
    dream_run_id = cursor.lastrowid
    dream_conn.commit()
    dream_conn.close()
    
    try:
        # 增量获取记忆 (O(1) manifest 过滤)
        memories = get_incremental_memories(hours)
        log.info(f"📊 获取到 {len(memories)} 条新记忆 (最近 {hours} 小时, 增量)")
        
        # 如果增量太少，回退到全量 (首次运行或长时间未运行)
        # 但如果是 0 新增且有已处理记录 → 真的没新数据，不回退
        if len(memories) < 5:
            conn_check = sqlite3.connect(str(DREAM_DB))
            manifest_count = conn_check.execute("SELECT COUNT(*) FROM processed_manifest WHERE status='active'").fetchone()[0]
            conn_check.close()
            
            if manifest_count > 10 and len(memories) == 0:
                log.info(f"  ✅ 无新记忆 (已处理 {manifest_count} 条), 跳过梦循环")
                # 更新 dream_runs 为跳过
                dream_conn = sqlite3.connect(str(DREAM_DB))
                dream_conn.execute("UPDATE dream_runs SET finished_at = ?, summary = ? WHERE id = ?",
                                  (datetime.now(HKT).isoformat(), json.dumps({"status": "skipped_incremental"}), dream_run_id))
                dream_conn.commit()
                dream_conn.close()
                _release_lock()
                return {"status": "skipped", "reason": "no_new_memories_incremental"}
            
            all_recent = get_recent_memories(hours)
            if len(all_recent) > len(memories) * 3:
                log.info(f"  ⚠️ 增量太少({len(memories)}), 回退到全量({len(all_recent)})")
                memories = all_recent
        
        # 挖掘近期 session
        sessions = mine_recent_sessions(hours)
        session_digest = generate_session_digest(sessions)
        log.info(f"📋 近期 session: {len(sessions)} 个")
        log.info(session_digest)
        
        # ── Session Signal Scanning (from Anthropic autoDream) ──
        # 扫描用户消息提取纠正/偏好/决策/模式信号
        session_signals = scan_session_signals(hours)
        total_signals = sum(len(v) for v in session_signals.values())
        if total_signals > 0:
            log.info(f"📡 Session 信号: {total_signals} 条 "
                     f"(纠正={len(session_signals['corrections'])}, "
                     f"偏好={len(session_signals['preferences'])}, "
                     f"决策={len(session_signals['decisions'])}, "
                     f"模式={len(session_signals['patterns'])})")
            # 将信号注入为高优先级记忆候选 (加入 memories 列表)
            for sig_type, sigs in session_signals.items():
                for sig in sigs[:5]:  # 每类最多5条
                    memories.append({
                        "id": f"signal_{sig_type}_{sig['timestamp']:.0f}",
                        "text": f"[SESSION_{sig_type.upper()}] {sig['text']}",
                        "created_at": datetime.fromtimestamp(sig['timestamp'], tz=timezone.utc).isoformat(),
                        "source": "session_signal",
                        "signal_type": sig_type,
                        "session_title": sig.get("session_title", ""),
                    })
        else:
            log.info("📡 Session 信号: 0 条")
        
        if not memories and not sessions:
            log.warning("⚠️ 没有新记忆或session, 跳过梦循环")
            _release_lock()
            return {"status": "skipped", "reason": "no_memories"}
        
        # Stage 1: Shallow Sleep
        clusters = {}
        if "1" in stages:
            clusters = stage1_shallow_sleep(memories)
        
        # Stage 2: REM
        rem_results = {}
        if "2" in stages:
            rem_results = stage2_rem(clusters)
        
        # Stage 3: Deep Sleep
        stats = {}
        if "3" in stages:
            stats = stage3_deep_sleep(
                rem_results, dream_run_id, dry_run,
                total_memories=len(memories),
                total_clusters=len(clusters),
            )
        
        # ── P1: REM 梦游 (从 Neo4j 高重要性节点随机游走) ──
        dream_walk_edges = []
        if "2" in stages and not dry_run:
            # P10: 传当前 cluster 实体给梦游做种子
            # 从 clusters 提取所有实体
            all_cluster_entities = []
            for ck, group in clusters.items():
                texts = [m["text"][:200] for m in group]
                ents = extract_entities_with_fallback(texts, max_entities=3)
                all_cluster_entities.extend(ents)
            
            dream_walk_edges = rem_dream_walk(cluster_entities=all_cluster_entities)
            if dream_walk_edges:
                # P1 扩展: LLM Boost — 对高共现关系从0.3→0.6
                dream_walk_edges = llm_boost_relations(dream_walk_edges, clusters, max_boost=10)
                # 写入 Neo4j
                written = write_relations_to_neo4j(dream_walk_edges)
                stats["dream_walk"] = written
                # 记录到 relation_log
                conn_rl = sqlite3.connect(str(DREAM_DB))
                for e in dream_walk_edges:
                    conn_rl.execute(
                        "INSERT INTO relation_log (dream_run_id, source_entity, target_entity, relation_type, confidence, method) VALUES (?, ?, ?, ?, ?, ?)",
                        (dream_run_id, e["source"], e["target"], "DREAM_WALK", e["confidence"], "rem_dream_walk")
                    )
                conn_rl.commit()
                conn_rl.close()
        
        # ── P2: NREM Hebbian 强化 (关系权重调整) ──
        hebbian_stats = {}
        if "3" in stages and not dry_run:
            hebbian_stats = nrem_hebbian_consolidation()
            stats["hebbian_strengthened"] = hebbian_stats.get("strengthened", 0)
            stats["hebbian_downscaled"] = hebbian_stats.get("downscaled", 0)
        
        # ── P3: 语义签名冲突检测 (同槽不同值) ──
        slot_conflicts = []
        if "2" in stages:
            slot_conflicts = detect_slot_conflicts()
            if slot_conflicts:
                # 记录到 contradiction_log
                conn_sc = sqlite3.connect(str(DREAM_DB))
                for c in slot_conflicts:
                    conn_sc.execute(
                        "INSERT INTO contradiction_log (dream_run_id, mem1_id, mem2_id, marker, contradiction_type, llm_explanation, verified) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (dream_run_id, c["mem1_id"], c["mem2_id"], 
                         f"slot_conflict(sim={c['slot_similarity']:.2f},diff={c['value_diff']:.2f})",
                         "SLOT_CONFLICT", 
                         f"同槽不同值: 槽相似度={c['slot_similarity']:.2f}, 值差异={c['value_diff']:.2f}",
                         0)
                    )
                conn_sc.commit()
                conn_sc.close()
                stats["slot_conflicts"] = len(slot_conflicts)
                # P7: 保存列表供 stage3 自动处理
                rem_results["slot_conflicts_list"] = slot_conflicts
        
        # 更新 manifest (O(1) 增量追踪)
        if memories and not dry_run:
            update_manifest(memories, dream_run_id)
            log.info(f"  📋 Manifest 已更新: {len(memories)} 条标记为已处理")
            
            # 标记已归档的记忆
            archived_ids = []
            for item in rem_results.get("dedup_candidates", []):
                archived_ids.append(item["remove"]["id"])
            if archived_ids:
                mark_manifest_archived(archived_ids)
        
        # 更新 dream_runs
        end_time = datetime.now(HKT)
        dream_conn = sqlite3.connect(str(DREAM_DB))
        dream_conn.execute("""
            UPDATE dream_runs SET
                finished_at = ?,
                stage1_clusters = ?,
                stage2_boosted = ?,
                stage3_deduped = ?,
                stage3_inferred = ?,
                stage3_decayed = ?,
                stage3_vault_suggestions = ?,
                summary = ?
            WHERE id = ?
        """, (
            end_time.isoformat(),
            len(clusters),
            len(rem_results.get("boosted", [])),
            stats.get("deduped", 0),
            stats.get("inferred", 0),
            stats.get("decayed", 0),
            stats.get("vault_suggestions", 0),
            json.dumps({
                "memories_scanned": len(memories),
                "clusters": len(clusters),
                "dedup_candidates": len(rem_results.get("dedup_candidates", [])),
                "merge_candidates": len(rem_results.get("merge_candidates", [])),
                "vault_candidates": len(rem_results.get("vault_candidates", [])),
                "decay_candidates": len(rem_results.get("decay_candidates", [])),
            }, ensure_ascii=False),
            dream_run_id,
        ))
        dream_conn.commit()
        dream_conn.close()
        
        result = {
            "status": "success",
            "dream_run_id": dream_run_id,
            "memories_scanned": len(memories),
            "clusters": len(clusters),
            "deduped": stats.get("deduped", 0),
            "inferred": stats.get("inferred", 0),
            "decayed": stats.get("decayed", 0),
            "vault_suggestions": stats.get("vault_suggestions", 0),
            "boosted": stats.get("boosted", 0),
            "duration_seconds": (end_time - start_time).total_seconds(),
            # P6: 嵌入详情供 format_report 使用
            "rem_results": rem_results,
            "stats": stats,
        }
        
        log.info(f"🌅 梦循环完成 — {result['duration_seconds']:.1f}s")
        _release_lock()
        return result
        
    except Exception as e:
        log.error(f"❌ 梦循环失败: {e}", exc_info=True)
        dream_conn = sqlite3.connect(str(DREAM_DB))
        dream_conn.execute("UPDATE dream_runs SET error = ?, finished_at = ? WHERE id = ?",
                          (str(e), datetime.now(HKT).isoformat(), dream_run_id))
        dream_conn.commit()
        dream_conn.close()
        _release_lock()
        return {"status": "error", "error": str(e)}


def resolve_slot_conflicts(conflicts: list[dict], dream_run_id: int, max_resolve: int = 5) -> list[dict]:
    """
    P7: 语义签名冲突自动处理
    
    对高值差(>0.5)的 slot_conflict:
    1. LLM 判断类型: SUPERSEDE / EXTEND / FALSE_POSITIVE
    2. SUPERSEDE → 旧记忆归档，保留新记忆
    3. EXTEND → 两条都保留，标记 extended
    4. FALSE_POSITIVE → 标记忽略
    
    成本: ~200 tokens/conflict, 最多处理5个
    """
    resolved = []
    
    # 只处理高值差的冲突 (值差>0.5 说明真的不同)
    high_diff = [c for c in conflicts if c.get("value_diff", 0) > 0.5]
    if not high_diff:
        log.info("  🔍 P7: 无高值差冲突需要处理")
        return resolved
    
    log.info(f"  🔍 P7: {len(high_diff)} 个高值差冲突待处理 (限{max_resolve})")
    
    for c in high_diff[:max_resolve]:
        mem1_id = c.get("mem1_id", "")
        mem2_id = c.get("mem2_id", "")
        mem1_text = c.get("mem1_text", "")
        mem2_text = c.get("mem2_text", "")
        
        # LLM 判断
        v = llm_verify_contradiction(mem1_text, mem2_text, f"slot_conflict(sim={c.get('slot_similarity', 0):.2f},diff={c.get('value_diff', 0):.2f})")
        
        if v is None:
            log.info(f"    ⏭️ API失败, 跳过 {mem1_id[:8]} vs {mem2_id[:8]}")
            continue
        
        ctype = v.get("type", "FALSE_POSITIVE")
        explanation = v.get("explanation", "")
        
        if ctype == "SUPERSEDE":
            # 新事实取代旧事实 → 归档旧记忆 (按创建时间判断)
            # 获取两条记忆的创建时间
            rows1 = pg_query(f"SELECT id::text, payload->>'created_at' FROM mem0 WHERE id::text = '{mem1_id}'")
            rows2 = pg_query(f"SELECT id::text, payload->>'created_at' FROM mem0 WHERE id::text = '{mem2_id}'")
            
            older_id = mem1_id  # 默认归档第一条
            newer_id = mem2_id
            if rows1 and rows2:
                # 比较创建时间
                t1 = rows1[0][1] if len(rows1[0]) > 1 else ""
                t2 = rows2[0][1] if len(rows2[0]) > 1 else ""
                if t2 < t1:  # mem2更早
                    older_id, newer_id = mem2_id, mem1_id
            
            log.info(f"    ✅ SUPERSEDE: 归档 {older_id[:8]}, 保留 {newer_id[:8]} ({explanation[:60]})")
            # 归档旧记忆
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{"archived": true, "archived_reason": "slot_supersede", "superseded_by": "{newer_id}", "supersede_explanation": "{explanation[:100].replace('"', '')}", "freshness": "outdated", "freshness_source": "dream_cycle_supersede"}}' WHERE id::text = '{older_id}'""")
            mark_manifest_archived([older_id])
            resolved.append({"type": "SUPERSEDE", "older": older_id, "newer": newer_id, "explanation": explanation})
        
        elif ctype == "EXTEND":
            # 新事实扩展旧事实 → 两条都保留，标记 extended
            log.info(f"    🔗 EXTEND: 两条都保留 ({explanation[:60]})")
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{"extended": true, "extended_by": "{mem2_id}", "extension_type": "{ctype}"}}' WHERE id::text = '{mem1_id}'""")
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{"extended": true, "extends": "{mem1_id}", "extension_type": "{ctype}"}}' WHERE id::text = '{mem2_id}'""")
            resolved.append({"type": "EXTEND", "mem1": mem1_id, "mem2": mem2_id, "explanation": explanation})
        
        else:  # FALSE_POSITIVE
            log.info(f"    ⏭️ FALSE_POSITIVE: 不处理 ({explanation[:60]})")
            resolved.append({"type": "FALSE_POSITIVE", "explanation": explanation})
    
    # 记录到 contradiction_log
    conn_cl = sqlite3.connect(str(DREAM_DB))
    for r in resolved:
        ctype = r["type"]
        ids = [r.get("older", r.get("mem1", "")), r.get("newer", r.get("mem2", ""))]
        conn_cl.execute(
            "UPDATE contradiction_log SET contradiction_type = ?, resolution = ?, llm_explanation = ?, verified = 1 WHERE mem1_id = ? OR mem2_id = ?",
            (ctype, ctype, r.get("explanation", ""), ids[0], ids[1])
        )
    conn_cl.commit()
    conn_cl.close()
    
    log.info(f"  ✅ P7 冲突处理: {len(resolved)} 条 (SUPERSEDE={sum(1 for r in resolved if r['type']=='SUPERSEDE')}, "
             f"EXTEND={sum(1 for r in resolved if r['type']=='EXTEND')}, "
             f"FALSE_POSITIVE={sum(1 for r in resolved if r['type']=='FALSE_POSITIVE')})")
    
    return resolved


# ─── 报告生成 ──────────────────────────────────────────────────────────

def format_report(result: dict, rem_results: dict = None, stats: dict = None) -> str:
    """
    P6: 生成 Telegram 友好的增强报告
    
    新增: Top3 boost + Top3 vault + Top3 语义冲突 + 健康评分
    """
    if result["status"] == "skipped":
        return "💤 Dream Cycle — 无新记忆，跳过"
    
    if result["status"] == "error":
        return f"💤 Dream Cycle — ❌ 错误: {result['error'][:100]}"
    
    lines = [
        "💤 **Dream Cycle v3 完成**",
        "",
        f"📊 扫描: {result['memories_scanned']} 条 | 聚类: {result['clusters']} 组",
        f"🔥 Boost: {result.get('boosted', 0)} | 🔗 推断: {result['inferred']} 关系",
        f"🗑️ 去重: {result['deduped']} | 📉 衰减: {result['decayed']} 候选",
        f"📝 Vault: {result['vault_suggestions']} 建议",
        f"⏱️ {result['duration_seconds']:.1f}s",
    ]
    
    # P6: Top3 Boost 详情
    if rem_results:
        boosted = rem_results.get("boosted", [])
        if boosted:
            lines.append("")
            lines.append("🔥 **Top3 Boost (高重要性记忆)**")
            for b in sorted(boosted, key=lambda x: x["score"], reverse=True)[:3]:
                text_preview = b["memory"]["text"][:60].replace("\n", " ")
                lines.append(f"  • {b['score']:.2f} — {text_preview}... ({b.get('reason', '')})")
        
        # P6: Top3 Vault 候选
        vault = rem_results.get("vault_candidates", [])
        if vault:
            lines.append("")
            lines.append("📝 **Top3 Vault 候选**")
            for v in sorted(vault, key=lambda x: x.get("priority", "normal"), reverse=True)[:3]:
                kw = ", ".join(v.get("keywords", [])[:3])
                priority = v.get("priority", "normal")
                gate = v.get("promotion_pass", "")
                lines.append(f"  • [{priority}] {kw} (score={v['best_score']:.2f}, gate={gate})")
        
        # P6: Top3 语义冲突
        contradictions = rem_results.get("contradictions", [])
        if contradictions:
            lines.append("")
            lines.append("⚡ **Top3 语义冲突**")
            for c in contradictions[:3]:
                m1 = c.get("mem1", {}).get("text", "")[:50].replace("\n", " ")
                m2 = c.get("mem2", {}).get("text", "")[:50].replace("\n", " ")
                vtype = c.get("contradiction_type", c.get("verified", "?"))
                lines.append(f"  • [{vtype}] {m1}... vs {m2}...")
    
    # P6: 健康评分
    if stats and stats.get("health_score"):
        lines.append("")
        lines.append(f"💊 健康评分: **{stats['health_score']:.0f}/100**")
    
    return "\n".join(lines)


def send_telegram_report(report: str):
    """通过 Telegram Bot 发送报告"""
    import subprocess
    # 用 hermes send_message 或直接 curl
    # 从 config 读 bot token + chat_id
    try:
        config_path = Path("/root/.hermes/config.yaml")
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            token = config.get("telegram", {}).get("bot_token", "")
            chat_id = config.get("telegram", {}).get("home_chat_id", "")
            if token and chat_id:
                import urllib.parse
                encoded = urllib.parse.quote(report, safe='')
                url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={encoded}&parse_mode=Markdown"
                subprocess.run(["curl", "-s", url], timeout=10, capture_output=True)
                log.info("📱 Telegram 报告已发送")
                return
        log.warning("⚠️ Telegram 配置不完整，跳过发送")
    except Exception as e:
        log.warning(f"⚠️ Telegram 发送失败: {e}")


# ─── Session 挖掘 — 从 state.db 提取近期会话主题 ──────────────────────

def mine_recent_sessions(hours: int = 24) -> list[dict]:
    """
    从 state.db 提取近期会话主题, 作为梦循环的额外输入
    
    返回:
    - 高频话题
    - 未被 mem0 捕获的知识
    - 跨 session 反复出现的模式
    """
    if not STATE_DB.exists():
        return []
    
    cutoff = time.time() - hours * 3600
    conn = sqlite3.connect(str(STATE_DB))
    
    # 最近 session 的标题和统计
    sessions = conn.execute("""
        SELECT id, title, message_count, tool_call_count, started_at, estimated_cost_usd
        FROM sessions
        WHERE started_at > ? AND title IS NOT NULL
        ORDER BY started_at DESC
    """, (cutoff,)).fetchall()
    
    conn.close()
    
    topics = []
    for s in sessions:
        sid, title, msg_count, tool_count, started, cost = s
        if title and len(title) > 5:
            topics.append({
                "session_id": sid,
                "title": title,
                "message_count": msg_count,
                "tool_call_count": tool_count,
                "cost": cost or 0,
            })
    
    return topics



def scan_session_signals(hours: int = 72) -> dict:
    """
    Session Transcript Scanning (from Anthropic autoDream / dream-skill).
    
    Scan state.db user messages for 4 signal types:
    - corrections: user corrected the agent (highest priority)
    - preferences: explicit preference statements
    - decisions: architectural/tool choices
    - patterns: recurring complaints or repeated requests
    
    Returns: {"corrections": [...], "preferences": [...], "decisions": [...], "patterns": [...]}
    Each item: {"text": str, "session_id": str, "timestamp": float, "signal_type": str}
    """
    if not STATE_DB.exists():
        return {"corrections": [], "preferences": [], "decisions": [], "patterns": []}
    
    cutoff = time.time() - hours * 3600
    signals = {"corrections": [], "preferences": [], "decisions": [], "patterns": []}
    
    try:
        conn = sqlite3.connect(str(STATE_DB))
        cursor = conn.execute("""
            SELECT m.content, m.session_id, m.timestamp, s.title
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE m.role = 'user'
            AND m.timestamp > ?
            AND m.content IS NOT NULL
            AND length(m.content) > 10
            AND length(m.content) < 500
            AND m.content NOT LIKE '[IMPORTANT%'
            AND m.content NOT LIKE 'Tool results%'
            AND m.content NOT LIKE '%SYSTEM:%'
            ORDER BY m.timestamp DESC
            LIMIT 2000
        """, (cutoff,))
        
        for row in cursor:
            text, session_id, ts, title = row
            text_stripped = text.strip()
            text_lower = text_stripped.lower()
            
            # Skip very short or system-like messages
            if len(text_stripped) < 15:
                continue
            
            # Check each signal type (priority order)
            matched_type = None
            matched_kw = None
            
            for kw in SIGNAL_CORRECTIONS:
                if kw.lower() in text_lower:
                    matched_type = "corrections"
                    matched_kw = kw
                    break
            
            if not matched_type:
                for kw in SIGNAL_PREFERENCES:
                    if kw.lower() in text_lower:
                        matched_type = "preferences"
                        matched_kw = kw
                        break
            
            if not matched_type:
                for kw in SIGNAL_DECISIONS:
                    if kw.lower() in text_lower:
                        matched_type = "decisions"
                        matched_kw = kw
                        break
            
            if not matched_type:
                for kw in SIGNAL_PATTERNS:
                    if kw.lower() in text_lower:
                        matched_type = "patterns"
                        matched_kw = kw
                        break
            
            if matched_type:
                signals[matched_type].append({
                    "text": text_stripped[:300],
                    "session_id": session_id,
                    "session_title": title or "untitled",
                    "timestamp": ts,
                    "signal_type": matched_type,
                    "trigger_keyword": matched_kw,
                })
        
        conn.close()
    except Exception as e:
        log.warning(f"⚠️ session signal scan failed: {e}")
    
    return signals


def generate_session_digest(sessions: list[dict]) -> str:
    """生成近期 session 摘要"""
    if not sessions:
        return "无近期 session"
    
    # 按 cost 排序 (高 cost = 深度工作)
    top = sorted(sessions, key=lambda x: x["cost"], reverse=True)[:5]
    
    lines = [f"📋 近期 Top {len(top)} Session:"]
    for s in top:
        lines.append(f"  • {s['title'][:50]} ({s['message_count']}msg, ${s['cost']:.3f})")
    
    return "\n".join(lines)


# ─── P8: 健康仪表盘 ──────────────────────────────────────────────────────

def show_health_dashboard():
    """
    P8: 显示梦循环健康仪表盘 — 7天趋势 + 当前状态
    """
    conn = sqlite3.connect(str(DREAM_DB))
    
    # 7天趋势
    cutoff = (datetime.now(HKT) - timedelta(days=7)).isoformat()
    runs = conn.execute("""
        SELECT id, started_at, stage1_clusters, stage2_boosted, stage3_deduped,
               stage3_inferred, stage3_decayed, stage3_vault_suggestions, summary, error
        FROM dream_runs WHERE started_at > ?
        ORDER BY id ASC
    """, (cutoff,)).fetchall()
    
    # Manifest 统计
    total_manifest = conn.execute("SELECT COUNT(*) FROM processed_manifest").fetchone()[0]
    active_manifest = conn.execute("SELECT COUNT(*) FROM processed_manifest WHERE status='active'").fetchone()[0]
    archived_manifest = conn.execute("SELECT COUNT(*) FROM processed_manifest WHERE status='archived'").fetchone()[0]
    
    # Relation 统计
    total_relations = conn.execute("SELECT COUNT(*) FROM relation_log").fetchone()[0]
    useful_relations = conn.execute("SELECT COUNT(*) FROM relation_log WHERE confidence >= 0.5").fetchone()[0]
    boosted_relations = conn.execute("SELECT COUNT(*) FROM relation_log WHERE confidence >= 0.6").fetchone()[0]
    cross_cluster_rels = conn.execute("SELECT COUNT(*) FROM relation_log WHERE method = 'cross_cluster_cooccurrence'").fetchone()[0]
    
    # Neo4j 实际关系数 (查询 Playground)
    neo4j_total = 0
    neo4j_dream = 0
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        with driver.session() as session:
            neo4j_total = session.run("MATCH ()-[r]->() RETURN count(r)").single()[0]
            neo4j_dream = session.run("MATCH ()-[r]->() WHERE r.source = 'dream_cycle' RETURN count(r)").single()[0]
        driver.close()
    except:
        pass
    
    # Vault suggestion 统计
    total_suggestions = conn.execute("SELECT COUNT(*) FROM vault_suggestion").fetchone()[0]
    pending_suggestions = conn.execute("SELECT COUNT(*) FROM vault_suggestion WHERE status='pending'").fetchone()[0]
    auto_created = conn.execute("SELECT COUNT(*) FROM vault_suggestion WHERE status='auto_created'").fetchone()[0]
    reviewed = conn.execute("SELECT COUNT(*) FROM vault_suggestion WHERE status='reviewed'").fetchone()[0]
    
    # Contradiction 统计
    total_contra = conn.execute("SELECT COUNT(*) FROM contradiction_log").fetchone()[0]
    resolved_contra = conn.execute("SELECT COUNT(*) FROM contradiction_log WHERE resolution != 'pending'").fetchone()[0]
    
    conn.close()
    
    # 记忆总量
    recent_memories = get_recent_memories(hours=168)
    all_memories = get_all_memories_with_embeddings()
    
    print("=" * 50)
    print("🏥 **Dream Cycle Health Dashboard**")
    print("=" * 50)
    
    print(f"\n📊 **记忆状态**")
    print(f"  PG 总量: {len(all_memories)} | 7天新增: {len(recent_memories)}")
    print(f"  Manifest: {active_manifest} active / {archived_manifest} archived / {total_manifest} total")
    
    print(f"\n🔗 **关系网络**")
    print(f"  总关系: {total_relations} | 高置信(>=0.5): {useful_relations} | LLM Boost(>=0.6): {boosted_relations}")
    print(f"  跨聚类: {cross_cluster_rels} | Neo4j: {neo4j_dream}/{neo4j_total}")
    
    print(f"\n📝 **Vault 沉淀**")
    print(f"  建议: {total_suggestions} | pending: {pending_suggestions} | auto_created: {auto_created} | reviewed: {reviewed}")
    
    print(f"\n⚡ **冲突检测**")
    print(f"  总冲突: {total_contra} | 已解决: {resolved_contra} | pending: {total_contra - resolved_contra}")
    
    print(f"\n📈 **7天趋势** ({len(runs)} runs)")
    if runs:
        # Sparkline-style 趋势
        clusters_trend = [str(r[2]) for r in runs]
        dedup_trend = [str(r[4]) for r in runs]     # stage3_deduped (index 4)
        inferred_trend = [str(r[5]) for r in runs]   # stage3_inferred (index 5)
        vault_trend = [str(r[8]) for r in runs]
        
        print(f"  Clusters: {' → '.join(clusters_trend)}")
        print(f"  Deduped:  {' → '.join(dedup_trend)}")
        print(f"  Inferred: {' → '.join(inferred_trend)}")
        print(f"  Vault:    {' → '.join(vault_trend)}")
        
        # 最近一次详情
        last = runs[-1]
        last_status = "✅" if not last[9] else "❌"
        print(f"\n  最近一次: #{last[0]} [{last_status}] {last[1]}")
        summary = last[8]  # summary is column 8 (index 8)
        if summary and summary != "None":
            try:
                s = json.loads(summary)
                print(f"    扫描: {s.get('memories_scanned', '?')} | "
                      f"聚类: {s.get('clusters', '?')} | "
                      f"去重候选: {s.get('dedup_candidates', '?')} | "
                      f"合并候选: {s.get('merge_candidates', '?')}")
            except:
                print(f"    {summary[:100]}")
    else:
        print("  无最近7天记录")
    
    # 健康评分 (简化版)
    # coverage: Neo4j dream关系覆盖了多少记忆 (目标: 10% 记忆有dream关系)
    mem_count = max(len(all_memories), 1)
    coverage = min(1.0, neo4j_dream / max(mem_count * 0.1, 1)) if neo4j_dream > 0 else min(1.0, (useful_relations + auto_created) / max(total_manifest * 0.5, 1))
    
    # coherence: 矛盾解决率
    coherence = min(1.0, 1.0 - (total_contra - resolved_contra) / max(total_contra, 1)) if total_contra > 0 else 1.0
    
    # efficiency: 处理效率
    efficiency = min(1.0, 1.0 - max(0, total_manifest - active_manifest) / max(total_manifest, 1))
    
    # reachability: Neo4j总关系 vs 记忆量 (目标: 30% 连接有向边)
    reachability = min(1.0, neo4j_total / max(mem_count * 0.3, 1)) if neo4j_total > 0 else min(1.0, useful_relations / max(total_manifest * 0.3, 1))
    
    health = {
        "coverage": coverage,
        "coherence": coherence,
        "efficiency": efficiency,
        "reachability": reachability,
    }
    score = sum(health[k] * w for k, w in [("coverage", 0.30), ("coherence", 0.25), ("efficiency", 0.20), ("reachability", 0.25)]) * 100
    
    print(f"\n💊 **综合健康: {score:.0f}/100**")
    print(f"  cov={health['coverage']:.0%} coh={health['coherence']:.0%} eff={health['efficiency']:.0%} reach={health['reachability']:.0%}")
    print("=" * 50)


# ─── P9: Vault Suggestion Review ───────────────────────────────────────

def review_vault_suggestions(max_review: int = 5) -> list[dict]:
    """
    P9: 处理 pending vault suggestion
    
    1. 对 auto_created 的 stub 页 — 用 wiki-ingest --llm 充实内容
    2. 对 pending 的 — LLM 判断是否值得创建stub
    3. 不值得的 → 标记 rejected
    
    Returns: 处理结果列表
    """
    conn = sqlite3.connect(str(DREAM_DB))
    
    # 1. 处理 pending 建议
    pending = conn.execute("""
        SELECT id, entity, category, frequency, reason, dream_run_id
        FROM vault_suggestion WHERE status = 'pending'
        ORDER BY frequency DESC
        LIMIT ?
    """, (max_review,)).fetchall()
    
    # 2. 处理 auto_created 但内容还是stub的
    auto_created = conn.execute("""
        SELECT id, entity, category
        FROM vault_suggestion WHERE status = 'auto_created'
        ORDER BY frequency DESC
    """).fetchall()
    
    conn.close()
    
    reviewed = []
    
    # Process pending: LLM 判断是否值得
    for p in pending:
        sid, entity, category, freq, reason, drid = p
        
        # 检查是否已有 Vault 页面
        slug = entity.lower().replace(" ", "-")[:50]
        cat_map = {"markets": "markets", "investment": "markets", "projects": "projects", 
                   "technology": "concepts", "concepts": "concepts"}
        vault_cat = cat_map.get(category, "concepts")
        filepath = VAULT_DIR / vault_cat / f"{slug}.md"
        
        if filepath.exists():
            # 已有页面 → 标记 reviewed
            conn = sqlite3.connect(str(DREAM_DB))
            conn.execute("UPDATE vault_suggestion SET status = 'reviewed' WHERE id = ?", (sid,))
            conn.commit()
            conn.close()
            reviewed.append({"entity": entity, "action": "already_exists", "status": "reviewed"})
            continue
        
        # 频率太低(<2) → reject
        if freq < 2:
            conn = sqlite3.connect(str(DREAM_DB))
            conn.execute("UPDATE vault_suggestion SET status = 'rejected' WHERE id = ?", (sid,))
            conn.commit()
            conn.close()
            reviewed.append({"entity": entity, "action": "rejected_low_freq", "status": "rejected"})
            continue
        
        # 频率>=2 → 创建 stub (LLM充实概述)
        keywords = [entity]
        sample = f"{entity} (出现 {freq} 次, {reason})"
        vault_path = create_vault_stub(entity, category, keywords, sample, sample_age_days=None)
        if vault_path:
            conn = sqlite3.connect(str(DREAM_DB))
            conn.execute("UPDATE vault_suggestion SET status = 'auto_created' WHERE id = ?", (sid,))
            conn.commit()
            conn.close()
            reviewed.append({"entity": entity, "action": "stub_created", "path": vault_path, "status": "auto_created"})
        else:
            reviewed.append({"entity": entity, "action": "stub_failed", "status": "pending"})
    
    # Process auto_created: 检查内容是否需要充实
    for ac in auto_created:
        sid, entity, category = ac
        slug = entity.lower().replace(" ", "-")[:50]
        cat_map = {"markets": "markets", "investment": "markets", "projects": "projects",
                   "technology": "concepts", "concepts": "concepts"}
        vault_cat = cat_map.get(category, "concepts")
        filepath = VAULT_DIR / vault_cat / f"{slug}.md"
        
        if filepath.exists():
            # 检查内容长度: <500字 = 还是stub → 标记需要充实
            content = filepath.read_text()
            word_count = len(content.split())
            if word_count < 100:
                reviewed.append({"entity": entity, "action": "needs_enrichment", "words": word_count})
            else:
                # 内容已充实 → 标记 reviewed
                conn = sqlite3.connect(str(DREAM_DB))
                conn.execute("UPDATE vault_suggestion SET status = 'reviewed' WHERE id = ?", (sid,))
                conn.commit()
                conn.close()
                reviewed.append({"entity": entity, "action": "enriched", "words": word_count, "status": "reviewed"})
    
    log.info(f"  📝 P9 Vault Review: {len(reviewed)} 条处理")
    for r in reviewed[:5]:
        log.info(f"    {r['entity']}: {r['action']}")
    
    return reviewed


# ─── P10: 批量积压处理 ─────────────────────────────────────────────────

def batch_resolve_all_conflicts(max_per_run: int = 50) -> dict:
    """
    批量处理所有 pending 矛盾 — 每次最多处理 max_per_run 个
    
    从 contradiction_log 读取 pending 记录，获取记忆文本，调用 LLM 分类
    """
    conn = sqlite3.connect(str(DREAM_DB))
    pending = conn.execute("""
        SELECT id, mem1_id, mem2_id, marker
        FROM contradiction_log WHERE resolution = 'pending'
        ORDER BY id
        LIMIT ?
    """, (max_per_run,)).fetchall()
    total_pending = conn.execute("SELECT COUNT(*) FROM contradiction_log WHERE resolution = 'pending'").fetchone()[0]
    conn.close()
    
    if not pending:
        log.info("✅ 无 pending 矛盾需要处理")
        return {"resolved": 0, "remaining": 0}
    
    log.info(f"🔍 批量矛盾处理: {len(pending)}/{total_pending} pending")
    
    resolved_count = 0
    failed = 0
    superseded = 0
    extended = 0
    false_pos = 0
    
    for row in pending:
        cid, mem1_id, mem2_id, marker = row
        
        # 获取记忆文本
        rows1 = pg_query(f"SELECT payload->>'data' FROM mem0 WHERE id::text = '{mem1_id}'")
        rows2 = pg_query(f"SELECT payload->>'data' FROM mem0 WHERE id::text = '{mem2_id}'")
        
        text1 = rows1[0][0] if rows1 and rows1[0][0] else ""
        text2 = rows2[0][0] if rows2 and rows2[0][0] else ""
        
        if not text1 or not text2:
            # 记忆已被删除 → 标记为 FALSE_POSITIVE
            conn = sqlite3.connect(str(DREAM_DB))
            conn.execute("UPDATE contradiction_log SET resolution = 'false_positive', llm_explanation = ? WHERE id = ?",
                        ("memory_deleted", cid))
            conn.commit()
            conn.close()
            false_pos += 1
            resolved_count += 1
            continue
        
        # LLM 判断
        v = llm_verify_contradiction(text1[:300], text2[:300], marker)
        
        if v is None:
            log.warning(f"  ⚠️ API 失败: {mem1_id[:8]} vs {mem2_id[:8]}")
            failed += 1
            continue
        
        ctype = v.get("type", "FALSE_POSITIVE")
        explanation = v.get("explanation", "")[:200].replace('"', '').replace("'", "")
        
        conn = sqlite3.connect(str(DREAM_DB))
        
        if ctype == "SUPERSEDE":
            # 归档较旧的记忆
            rows_t1 = pg_query(f"SELECT payload->>'created_at' FROM mem0 WHERE id::text = '{mem1_id}'")
            rows_t2 = pg_query(f"SELECT payload->>'created_at' FROM mem0 WHERE id::text = '{mem2_id}'")
            t1 = rows_t1[0][0] if rows_t1 and rows_t1[0][0] else ""
            t2 = rows_t2[0][0] if rows_t2 and rows_t2[0][0] else ""
            
            older_id = mem1_id if t1 <= t2 else mem2_id
            newer_id = mem2_id if older_id == mem1_id else mem1_id
            
            pg_query(f"""UPDATE mem0 SET payload = payload || '{{"archived": true, "archived_reason": "slot_supersede", "superseded_by": "{newer_id}"}}' WHERE id::text = '{older_id}'""")
            mark_manifest_archived([older_id])
            superseded += 1
            log.info(f"  ✅ SUPERSEDE: 归档 {older_id[:8]}")
        
        elif ctype == "EXTEND":
            extended += 1
            log.info(f"  🔗 EXTEND: {mem1_id[:8]} ↔ {mem2_id[:8]}")
        
        else:  # FALSE_POSITIVE
            false_pos += 1
        
        conn.execute("""
            UPDATE contradiction_log SET contradiction_type = ?, resolution = ?, 
                   llm_explanation = ?, verified = 1 WHERE id = ?
        """, (f"SLOT_CONFLICT->{ctype}", ctype, explanation, cid))
        conn.commit()
        conn.close()
        resolved_count += 1
    
    remaining = total_pending - resolved_count
    log.info(f"📊 批量矛盾完成: {resolved_count} resolved (S:{superseded} E:{extended} FP:{false_pos} fail:{failed}), {remaining} remaining")
    return {"resolved": resolved_count, "remaining": remaining, "superseded": superseded, "extended": extended, "false_positive": false_pos, "failed": failed}


def batch_review_all_vault(max_per_run: int = 100) -> dict:
    """
    批量处理所有 pending vault 建议
    
    freq>=2 → 创建 stub; freq<2 → reject; 已有页面 → reviewed
    """
    conn = sqlite3.connect(str(DREAM_DB))
    pending = conn.execute("""
        SELECT id, entity, category, frequency, reason
        FROM vault_suggestion WHERE status = 'pending'
        ORDER BY frequency DESC
        LIMIT ?
    """, (max_per_run,)).fetchall()
    total_pending = conn.execute("SELECT COUNT(*) FROM vault_suggestion WHERE status='pending'").fetchone()[0]
    conn.close()
    
    if not pending:
        log.info("✅ 无 pending vault 建议需要处理")
        return {"processed": 0, "remaining": 0}
    
    log.info(f"📝 批量 Vault 审核: {len(pending)}/{total_pending} pending")
    
    created = 0
    rejected = 0
    exists = 0
    failed = 0
    
    for row in pending:
        sid, entity, category, freq, reason = row
        
        # 检查是否已有 Vault 页面
        slug = entity.lower().replace(" ", "-")[:50]
        cat_map = {"markets": "markets", "investment": "markets", "projects": "projects",
                   "technology": "concepts", "concepts": "concepts"}
        vault_cat = cat_map.get(category, "concepts")
        filepath = VAULT_DIR / vault_cat / f"{slug}.md"
        
        conn = sqlite3.connect(str(DREAM_DB))
        
        if filepath.exists():
            conn.execute("UPDATE vault_suggestion SET status = 'reviewed' WHERE id = ?", (sid,))
            conn.commit()
            conn.close()
            exists += 1
            continue
        
        if freq < 2:
            conn.execute("UPDATE vault_suggestion SET status = 'rejected' WHERE id = ?", (sid,))
            conn.commit()
            conn.close()
            rejected += 1
            continue
        
        # 创建 stub
        keywords = [entity]
        sample = f"{entity} (出现 {freq} 次, {reason or ''})"
        vault_path = create_vault_stub(entity, category, keywords, sample, sample_age_days=None)
        if vault_path:
            conn.execute("UPDATE vault_suggestion SET status = 'auto_created' WHERE id = ?", (sid,))
            conn.commit()
            conn.close()
            created += 1
            log.info(f"  📄 创建: {entity} → {vault_path}")
        else:
            conn.close()
            failed += 1
    
    remaining = total_pending - (created + rejected + exists + failed)
    log.info(f"📊 批量 Vault 完成: {created} created, {rejected} rejected, {exists} exists, {failed} failed, {remaining} remaining")
    return {"created": created, "rejected": rejected, "exists": exists, "failed": failed, "remaining": remaining}


# ─── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hermes Dream Cycle — 记忆自主整理")
    parser.add_argument("--hours", type=int, default=48, help="回溯小时数")
    parser.add_argument("--dry-run", action="store_true", help="只分析不执行")
    parser.add_argument("--stages", default="123", help="执行阶段 (1/2/3/123)")
    parser.add_argument("--report", action="store_true", help="只输出报告")
    parser.add_argument("--history", type=int, default=0, help="查看最近N次梦循环记录")
    parser.add_argument("--notify", action="store_true", help="发送 Telegram 报告")
    parser.add_argument("--dedup-check", type=str, help="在线去冗余检查: 传入文本，返回ADD/SKIP/MERGE建议")
    parser.add_argument("--manifest-stats", action="store_true", help="查看manifest统计")
    parser.add_argument("--trigger-check", action="store_true", help="检查是否应该触发梦循环 (自适应触发)")
    parser.add_argument("--health", action="store_true", help="P8: 显示梦循环健康仪表盘 (7天趋势)")
    parser.add_argument("--vault-review", action="store_true", help="P9: 处理 pending vault suggestion")
    parser.add_argument("--resolve-all", action="store_true", help="P10: 批量处理所有 pending 矛盾")
    parser.add_argument("--vault-all", action="store_true", help="P10: 批量处理所有 pending vault 建议")
    parser.add_argument("--backlog", action="store_true", help="P10: 一次性清理所有积压(矛盾+vault)")
    parser.add_argument("--auto", action="store_true", help="自适应模式: 先检查触发条件，满足才执行")
    args = parser.parse_args()
    
    if args.dedup_check:
        result = online_dedup_check(args.dedup_check)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    
    if args.manifest_stats:
        conn = sqlite3.connect(str(DREAM_DB))
        total = conn.execute("SELECT COUNT(*) FROM processed_manifest").fetchone()[0]
        active = conn.execute("SELECT COUNT(*) FROM processed_manifest WHERE status='active'").fetchone()[0]
        archived = conn.execute("SELECT COUNT(*) FROM processed_manifest WHERE status='archived'").fetchone()[0]
        avg_count = conn.execute("SELECT AVG(process_count) FROM processed_manifest").fetchone()[0]
        print(f"📋 Manifest 统计:")
        print(f"  总计: {total}")
        print(f"  活跃: {active}")
        print(f"  已归档: {archived}")
        print(f"  平均处理次数: {avg_count:.1f}")
        conn.close()
        return
    
    if args.trigger_check:
        result = check_dream_trigger()
        status = "🟢 触发" if result["should_trigger"] else "⚪ 不触发"
        print(f"🔍 自适应触发检查: {status}")
        print(f"  紧急度: {result['urgency']}")
        print(f"  新记忆数: {result['new_memory_count']}")
        if result["reasons"]:
            print(f"  触发原因: {', '.join(result['reasons'])}")
        else:
            print(f"  无触发条件满足")
        return result
    
    # P8: 健康仪表盘
    if args.health:
        show_health_dashboard()
        return
    
    # P9: Vault Review
    if args.vault_review:
        results = review_vault_suggestions()
        print(f"📝 Vault Review: {len(results)} 条处理")
        for r in results:
            print(f"  {r['entity']}: {r['action']} → {r.get('status', '?')}")
        return results
    
    # P10: 批量矛盾处理
    if args.resolve_all:
        result = batch_resolve_all_conflicts(max_per_run=50)
        print(f"🔍 批量矛盾处理: {result}")
        return result
    
    # P10: 批量 Vault 审核
    if args.vault_all:
        result = batch_review_all_vault(max_per_run=100)
        print(f"📝 批量 Vault 审核: {result}")
        return result
    
    # P10: 一键清理积压
    if args.backlog:
        print("🧹 开始清理积压...")
        cr = batch_resolve_all_conflicts(max_per_run=50)
        vr = batch_review_all_vault(max_per_run=100)
        print(f"\n📊 积压清理完成:")
        print(f"  矛盾: {cr}")
        print(f"  Vault: {vr}")
        return {"conflicts": cr, "vault": vr}
    
    # 自适应模式: 先检查触发条件
    if args.auto:
        trigger = check_dream_trigger()
        if not trigger["should_trigger"]:
            print(f"💤 自适应模式: 无需触发 ({', '.join(trigger['reasons']) if trigger['reasons'] else '无触发条件'})")
            return {"status": "skipped", "reason": "adaptive_no_trigger"}
        print(f"🌙 自适应模式: 触发! 原因: {', '.join(trigger['reasons'])}, 紧急度: {trigger['urgency']}")
    
    if args.history > 0:
        conn = sqlite3.connect(str(DREAM_DB))
        rows = conn.execute("""
            SELECT id, started_at, stage1_clusters, stage3_deduped, stage3_inferred,
                   stage3_decayed, stage3_vault_suggestions, summary, error
            FROM dream_runs ORDER BY id DESC LIMIT ?
        """, (args.history,)).fetchall()
        for r in rows:
            status = "❌" if r[8] else "✅"
            print(f"#{r[0]} [{status}] {r[1]} clusters={r[2]} dedup={r[3]} infer={r[4]} decay={r[5]} vault={r[6]}")
        conn.close()
        return
    
    result = run_dream_cycle(
        hours=args.hours,
        dry_run=args.dry_run,
        stages=args.stages,
    )
    
    report = format_report(result, rem_results=result.get("rem_results"), stats=result.get("stats"))
    print(report)
    
    # 发送 Telegram 报告
    if args.notify:
        # 附加 session digest
        sessions = mine_recent_sessions(hours=args.hours)
        if sessions:
            session_report = generate_session_digest(sessions)
            report = report + "\n\n" + session_report
        send_telegram_report(report)
    
    if args.report:
        return result
    
    return result


if __name__ == "__main__":
    main()
