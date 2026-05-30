"""
Dream Cycle — Similarity functions — Jaccard, n-gram, combined, pgvector neighbors, clustering
"""

import hashlib
import json
import sqlite3
import subprocess
import logging
from pathlib import Path
from collections import defaultdict
from dream_cycle.config import DREAM_DB, STATE_DB, safe_float, log
from dream_cycle.db import pg_query

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

