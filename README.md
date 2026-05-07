# Dream Cycle 🌙

> **Autonomous Memory Consolidation for AI Agents** — inspired by sleep neuroscience, built for investment research.

Dream Cycle is a three-stage memory processing system that runs while you're not looking. Like biological sleep, it **consolidates fragmented memories into structured knowledge** — clustering, scoring, deduplicating, and promoting the important stuff to long-term storage.

## Why?

AI agents accumulate thousands of memory fragments per day. Without consolidation:
- 🔴 **Duplicates pile up** — the same fact stored 5 times
- 🔴 **Stale data poisons reasoning** — April's CGB yield presented as today's
- 🔴 **Connections stay buried** — related concepts never linked
- 🔴 **High-value insights evaporate** — no path from ephemeral → durable

Dream Cycle fixes this by running a nightly consolidation pipeline, just like your brain does during REM and deep sleep.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DREAM CYCLE v3                          │
│                                                             │
│  Stage 1: Shallow Sleep  ──→  Cluster memories by topic    │
│  Stage 2: REM Sleep      ──→  Score, Boost, Detect Conflicts│
│  Stage 3: Deep Sleep     ──→  Dedup, Merge, Promote, Decay  │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                │
│  │  Mem0 PG │   │  Neo4j   │   │  Vault   │                │
│  │ (vector) │   │  (graph) │   │  (wiki)  │                │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘                │
│       │              │              │                        │
│       └──────────────┼──────────────┘                        │
│                      │                                       │
│              ┌───────┴───────┐                               │
│              │  Time-Aware   │ ← 7-layer temporal guard      │
│              │  Protection   │   (stale market data blocked) │
│              └───────────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

## Three Stages

### Stage 1: Shallow Sleep (浅睡) — Clustering

Groups similar memories using vector similarity (pgvector) + keyword overlap. Think of it as sorting the day's experiences into thematic piles before filing.

- Vector neighbors: cosine distance < 0.30 → same cluster
- Keyword overlap: shared entities strengthen cluster bonds
- Result: 20-50 clusters per cycle from ~200 new memories

### Stage 2: REM Sleep (快速眼动) — Evaluation

Scores each memory for importance, detects contradictions, and identifies high-value clusters for promotion.

- **Boost**: high-importance memories (score ≥ 0.7) get reinforced in Mem0
- **Dedup**: vector distance < 0.10 → exact duplicate, remove
- **Merge**: distance 0.10-0.18 → near-duplicate, combine
- **Contradiction**: opposing markers (increased/decreased, up/down) → flag for LLM verification
- **Vault candidates**: clusters with high aggregate importance → promote to wiki

### Stage 3: Deep Sleep (深睡) — Action

Executes all the decisions from REM — actually writes to databases, creates files, archives stale data.

- Dedup → delete from Mem0 PG
- Merge → combine texts, keep primary
- Relations → infer entity relationships → write to Neo4j
- Decay → mark low-value memories, apply Ebbinghaus decay
- Vault → create wiki stubs with LLM-enriched overviews
- NotebookLM → sync fresh knowledge, prune stale sources

## Time-Awareness: 7-Layer Protection

Market data has a half-life. A yield quoted in April is dangerous in May. Dream Cycle implements 7 layers of temporal protection:

| Layer | Component | Protection |
|-------|-----------|------------|
| 1 | `_compute_memory_age_days()` | Parse created_at, compute age in days |
| 2 | `_is_time_sensitive()` | 18 regex patterns detect market data (yields, spreads, bp, CGB, UST...) |
| 3 | Sample selection | Freshest memory → sample text, not highest-scored |
| 4 | LLM overview prompt | Stale data → "describe framework, not numbers" |
| 5 | Vault frontmatter | `data_freshness: stale/recent/fresh` metadata |
| 6 | NotebookLM sync | Skip stale sources, prune daily research > 14 days |
| 7 | Audio description | Auto-inject: "Today is {date}. Market data = historical snapshots" |

## Quick Start

```bash
# Run dream cycle (default: last 48 hours)
python3 dream_cycle.py

# Dry run (no writes)
python3 dream_cycle.py --dry-run

# Extended range
python3 dream_cycle.py --hours 72

# Health dashboard
python3 dream_cycle.py --health

# Review vault suggestions
python3 dream_cycle.py --vault-review
```

## Data Sources

| Source | Type | Purpose |
|--------|------|---------|
| Mem0 PostgreSQL | Vector DB | Memory storage + similarity search (pgvector) |
| Neo4j | Graph DB | Entity relationships + knowledge graph |
| Vault | Markdown wiki | Durable knowledge pages (L2 storage) |
| NotebookLM | Google API | Audio overviews + cross-document Q&A |
| Infini AI | LLM API | Entity extraction, relation inference, overview generation |

## Health Score

Dream Cycle computes a 5-dimensional health score (0-100):

| Dimension | Weight | Measures |
|-----------|--------|----------|
| Freshness | 25% | % of memories < 7 days old |
| Coverage | 25% | % of clusters with vault pages |
| Coherence | 20% | % of non-singleton clusters |
| Efficiency | 15% | % of memories deduplicated |
| Reachability | 15% | % of entities with Neo4j connections |

## Configuration

Environment variables (or `.env`):

```bash
# Required
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=***

# LLM
INFINI_BASE_URL=https://cloud.infini-ai.com/maas/coding/v1
INFINI_MODEL=deepseek-v3.2
INFINI_API_KEY=***

# Paths
VAULT_DIR=/root/vault
DREAM_DB=/root/data/dream_cycle.db
```

## Integration with Hermes Agent

Dream Cycle runs as a cron job (04:00 HKT) within the [Hermes Agent](https://github.com/NousResearch/hermes-agent) ecosystem:

- **Memory plugin**: Hermes `mem0_conclude` calls `online_dedup_check()` for real-time dedup
- **Skill**: `meta/dream-cycle` provides quick reference
- **Vault pipeline**: Dream Cycle → wiki-ingest → Obsidian → NotebookLM
- **Telegram**: Daily dream report via Hermes gateway

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v1.0 | 2026-04-19 | Initial: 3 stages + keyword clustering |
| v2.0 | 2026-04-27 | LLM entity extraction + garbage cleanup |
| v3.0 | 2026-05-04 | P1-P5: LLM boost, vector dedup, Vault OR-gate, Ebbinghaus linkage |
| v3.1 | 2026-05-07 | P6-P10: Telegram report, conflict resolution, health dashboard, Neo4j bidirectional |
| v3.2 | 2026-05-07 | Time-awareness: 7-layer protection for market data |

## License

MIT — see [LICENSE](LICENSE)

## Related Projects

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — The AI agent framework Dream Cycle runs in
- [Vyakarana](https://github.com/quinnmacro/Vyakarana) — Architecture decisions (ADR-023)
- [Vault](https://github.com/quinnmacro/vault) — Knowledge base (L2 durable storage)
- [mem0-stack](https://github.com/quinnmacro/mem0-stack) — Self-hosted Mem0 + Neo4j
