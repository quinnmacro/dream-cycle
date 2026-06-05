"""
Dream Cycle — Database operations — PG (mem0), SQLite (dream_cycle.db, state.db), Neo4j Playground
"""

__all__ = [
    "get_recall_stats",
    "init_dream_db",
    "pg_query",
    "get_recent_memories",
    "get_incremental_memories",
    "claim_memories",
    "update_manifest",
    "mark_manifest_archived",
    "get_all_memories_with_embeddings",
    "delete_memory",
    "update_memory_text",
    "dedup_neo4j_relations",
    "write_relations_to_neo4j",
]

import json
import sqlite3
import subprocess
from datetime import datetime
from dream_cycle.config import (
    DREAM_DB,
    STATE_DB,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASS,
    HKT,
    log,
    PG_CONTAINER,
    PG_USER,
    PG_DB,
)
from dream_cycle.similarity import text_hash

# Module-level cache for recall stats (read once per dream cycle)
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
            except Exception:
                continue
        conn.close()
    except Exception as e:
        log.warning(f"⚠️ recall stats 读取失败: {e}")

    _recall_stats_cache = stats
    log.info(
        f"📊 recall stats: {len(stats)} unique queries, {sum(stats.values())} total calls"
    )
    return stats


def init_dream_db():
    """初始化梦循环元数据库"""
    DREAM_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
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
            status TEXT DEFAULT 'active',
            consolidated_at TEXT
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


def pg_query(sql: str) -> list[list[str]]:
    """通过 docker exec 查询 PG (stdin 管道模式, 避免 Argument list too long)"""
    cmd = [
        "docker",
        "exec",
        "-i",
        PG_CONTAINER,
        "psql",
        "-U",
        PG_USER,
        "-d",
        PG_DB,
        "-t",
        "-A",
        "-F",
        "|",
    ]
    result = subprocess.run(
        cmd, input=sql.encode(), capture_output=True, text=False, timeout=300
    )
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
            memories.append(
                {
                    "id": r[0],
                    "text": r[1],
                    "user_id": r[2],
                    "created_at": r[3],
                    "hash": r[4] if len(r) > 4 else None,
                    "agent_id": r[5] if len(r) > 5 else None,
                }
            )
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

    conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
    try:
        processed_ids = set()
        for row in conn.execute(
            "SELECT memory_id FROM processed_manifest WHERE status = 'active'"
        ).fetchall():
            processed_ids.add(row[0])
    finally:
        conn.close()

    new_memories = [m for m in all_recent if m["id"] not in processed_ids]
    log.info(
        f"  📊 增量获取: {len(all_recent)} 条中 {len(new_memories)} 条新增 "
        f"(已处理 {len(all_recent) - len(new_memories)})"
    )
    return new_memories


def claim_memories(
    memory_ids: list[str],
    dream_run_id: int,
    conn: sqlite3.Connection | None = None,
) -> list[str]:
    """
    Atomic claim pattern (from Mnemosyne sleep consolidation).

    Mark memories as claimed by this dream run, preventing concurrent
    dream cycles from processing the same memories.

    Uses UPDATE WHERE consolidated_at IS NULL — only unclaimed rows
    are affected. Returns the list of successfully claimed IDs.

    This is defense-in-depth on top of the PID lock: if two processes
    somehow bypass the lock, the SQL claim prevents double-processing.
    """
    if not memory_ids:
        return []

    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)

    now = datetime.now(HKT).isoformat()
    claimed = []

    try:
        # Ensure column exists (migration for existing DBs)
        try:
            conn.execute(
                "ALTER TABLE processed_manifest ADD COLUMN consolidated_at TEXT"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        for mid in memory_ids:
            cursor = conn.execute(
                """UPDATE processed_manifest
                   SET consolidated_at = ?, last_dream_run_id = ?
                   WHERE memory_id = ? AND consolidated_at IS NULL""",
                (now, dream_run_id, mid),
            )
            if cursor.rowcount > 0:
                claimed.append(mid)

        conn.commit()
    finally:
        if own_conn:
            conn.close()

    if len(claimed) < len(memory_ids):
        log.info(
            f"  🔒 Atomic claim: {len(claimed)}/{len(memory_ids)} claimed "
            f"({len(memory_ids) - len(claimed)} already claimed by concurrent run)"
        )

    return claimed


def update_manifest(memories: list[dict], dream_run_id: int):
    """更新 processed_manifest — 标记已处理"""
    if not memories:
        return
    conn = sqlite3.connect(str(DREAM_DB), timeout=5.0)
    now = datetime.now(HKT).isoformat()
    for m in memories:
        h = text_hash(m["text"])
        conn.execute(
            """
            INSERT INTO processed_manifest (memory_id, memory_hash, first_seen_at, last_processed_at, process_count, last_dream_run_id, status)
            VALUES (?, ?, ?, ?, 1, ?, 'active')
            ON CONFLICT(memory_id) DO UPDATE SET
                last_processed_at = ?,
                process_count = process_count + 1,
                last_dream_run_id = ?,
                status = 'active'
        """,
            (m["id"], h, now, now, dream_run_id, now, dream_run_id),
        )
    conn.commit()
    conn.close()


def mark_manifest_archived(memory_ids: list[str], conn=None):
    """标记 manifest 中已归档的记忆。

    If *conn* is provided, reuses it (avoids write-lock conflict when
    the caller already has an open transaction on dream_cycle.db).
    """
    if not memory_ids:
        return
    own_conn = conn is None
    if own_conn:
        conn = sqlite3.connect(str(DREAM_DB), timeout=30.0)
    for mid in memory_ids:
        conn.execute(
            "UPDATE processed_manifest SET status = 'archived' WHERE memory_id = ?",
            (mid,),
        )
    if own_conn:
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
            memories.append(
                {
                    "id": r[0],
                    "text": r[1],
                    "user_id": r[2],
                    "created_at": r[3],
                    "hash": r[4] if len(r) > 4 else None,
                }
            )
    return memories


def delete_memory(memory_id: str) -> bool:
    """删除一条记忆"""
    rows = pg_query(
        f"DELETE FROM mem0 WHERE id::text = '{memory_id}' RETURNING id::text"
    )
    return len(rows) > 0


def update_memory_text(memory_id: str, new_text: str) -> bool:
    """更新记忆文本 — stdin pipe mode, to_jsonb for safe JSON encoding"""
    safe_id = memory_id.replace("'", "''")
    safe_text = new_text.replace("'", "''")
    sql = (
        f"UPDATE mem0 SET payload = jsonb_set(payload, '{{data}}', "
        f"to_jsonb('{safe_text}'::text)) "
        f"WHERE id::text = '{safe_id}' RETURNING id::text"
    )
    rows = pg_query(sql)
    return len(rows) > 0


# ─── 相似度计算 (不依赖向量, 用文本hash+jaccard) ──────────────────────


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
        safe_rel_type = (
            "".join(c for c in rel_type if c.isalnum() or c == "_") or "RELATED_TO"
        )
        key = f"{src}|{tgt}|{safe_rel_type}"
        # 也检查反向 (A→B 和 B→A 视为同一条)
        reverse_key = f"{tgt}|{src}|{safe_rel_type}"

        if key not in existing_pairs and reverse_key not in existing_pairs:
            deduped.append(rel)
            existing_pairs.add(key)  # 防止批量内重复

    log.info(
        f"  🔄 Neo4j 关系去重: {len(new_relations)}→{len(deduped)} "
        f"(过滤 {len(new_relations) - len(deduped)} 重复)"
    )
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
                source = rel.get("source", "").replace("'", "").replace('"', "")
                target = rel.get("target", "").replace("'", "").replace('"', "")
                rel_type = rel.get("type", "RELATED_TO").replace(" ", "_")
                confidence = rel.get("confidence", 0.4)

                if not source or not target or len(source) < 2 or len(target) < 2:
                    continue

                # 用参数化查询避免 Cypher 注入
                # rel_type 不能参数化, 但已用 replace 清洗
                safe_rel_type = "".join(c for c in rel_type if c.isalnum() or c == "_")
                if not safe_rel_type:
                    safe_rel_type = "RELATED_TO"

                session.run(
                    f"""
                    MERGE (a:Concept {{name: $source}})
                    MERGE (b:Concept {{name: $target}})
                    MERGE (a)-[r:{safe_rel_type}]->(b)
                    SET r.confidence = $conf,
                        r.source = 'dream_cycle',
                        r.created_at = datetime()
                """,
                    source=source,
                    target=target,
                    conf=confidence,
                )
                count += 1
        driver.close()
    except Exception as e:
        log.warning(f"⚠️ Neo4j 写入失败: {e}")

    return count


# ─── Vault 自动沉淀 ──────────────────────────────────────────────────
