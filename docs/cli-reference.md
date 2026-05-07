# CLI Reference

## Core Commands

```bash
# Run dream cycle (default: last 48 hours)
python3 src/dream_cycle.py

# Dry run — no writes, just analysis
python3 src/dream_cycle.py --dry-run

# Extended range — process last 72 hours
python3 src/dream_cycle.py --hours 72

# Run specific stages only
python3 src/dream_cycle.py --stages 1    # Shallow sleep only
python3 src/dream_cycle.py --stages 12   # Shallow + REM
python3 src/dream_cycle.py --stages 123  # All three (default)
```

## Health & Monitoring

```bash
# Health dashboard — 7-day trends + 5-dimension score
python3 src/dream_cycle.py --health

# Check version
python3 src/dream_cycle.py --version
```

## Vault Management

```bash
# Review pending vault suggestions
python3 src/dream_cycle.py --vault-review

# Process auto-created stubs
python3 src/dream_cycle.py --vault-review  # handles both pending and auto_created
```

## NotebookLM Integration

```bash
# Sync vault → NotebookLM (skips stale data automatically)
python3 scripts/vault_to_notebooklm_sync.py

# Dry run — preview what would be synced
python3 scripts/vault_to_notebooklm_sync.py --dry-run

# Sync specific category
python3 scripts/vault_to_notebooklm_sync.py --category markets

# Prune stale research sources (>14 days)
python3 scripts/vault_to_notebooklm_sync.py --prune-stale --dry-run

# Generate audio with time-aware description (auto-injected)
python3 scripts/notebooklm_parallel_generate.py --types audio --poll
```

## Cron Setup

```bash
# Add to crontab (04:00 HKT daily)
0 4 * * * /usr/bin/python3 -u /path/to/dream-cycle/src/dream_cycle.py --hours 48 >> /var/log/dream-cycle.log 2>&1
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEO4J_URI` | Yes | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | Yes | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Yes | — | Neo4j password |
| `INFINI_API_KEY` | Yes | — | LLM API key (Infini AI) |
| `INFINI_BASE_URL` | No | `https://cloud.infini-ai.com/maas/coding/v1` | LLM endpoint |
| `INFINI_MODEL` | No | `deepseek-v3.2` | LLM model name |
| `VAULT_DIR` | No | `/root/vault` | Vault knowledge base path |
| `DREAM_DB` | No | `/root/data/dream_cycle.db` | SQLite database path |

## Output

Dream Cycle produces 5 types of output:

1. **Mem0 PG updates** — boosted/decayed memories (freshness field)
2. **Neo4j relations** — new entity relationships (confidence ≥ 0.3)
3. **Vault stubs** — wiki pages for high-frequency entities
4. **Telegram report** — daily dream summary (via Hermes gateway)
5. **Log file** — `/var/log/dream-cycle.log` (5MB rotating)
