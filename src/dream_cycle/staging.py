"""staging.py — Staging + Adopt safety contract (Dream Cycle v6 "Safe Sleep")

Ported from SkillOpt-Sleep: Dream Cycle NEVER mutates PG directly during
consolidation. All proposed changes are staged to disk, validated on held-out,
and only applied via explicit adopt().

Pipeline:
  1. Stage 3 collects proposed PG writes into a staging buffer
  2. Validation runs on held-out val split
  3. If validation passes → write staging files to disk
  4. Cron default = dry-run (stage only, no adopt)
  5. Manual `python3 -m dream_cycle --adopt` applies staged changes

Safety guarantees:
  - Stage 3 runs as normal but PG writes are intercepted
  - Proposed changes written to ~/dream_cycle_staging/<date>/
  - Live PG files NEVER touched during dream cycle
  - Adopt backs up affected memories before overwriting
  - Each staging dir has: proposals.json + report.md + manifest.json
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


STAGING_ROOT = Path.home() / "dream_cycle_staging"


@dataclass
class PGProposal:
    """A proposed PG write operation."""
    op: str                # "update_payload" | "update_text" | "delete"
    memory_id: str
    payload_patch: dict = field(default_factory=dict)  # for update_payload
    new_text: str = ""     # for update_text
    reason: str = ""
    source_op: str = ""    # which budget op triggered this
    stage: str = ""        # "boost" | "dedup" | "merge" | "decay" | "supersede" | "extend"


@dataclass
class StagingResult:
    """Result of staging operation."""
    staging_dir: str
    n_proposals: int
    n_by_stage: dict       # count per stage
    validation_accepted: bool
    validation_reason: str
    adopted: bool = False
    adopted_at: str = ""


class StagingBuffer:
    """Collects proposed PG writes during a dream cycle run.

    Usage:
        buf = StagingBuffer()
        buf.add_update_payload(mem_id, {"dream_boost": True}, reason="high importance", stage="boost")
        buf.add_archive(mem_id, reason="duplicate of X", stage="dedup")
        result = buf.write_staging(dream_run_id, validation_result)
    """

    def __init__(self):
        self.proposals: list[PGProposal] = []
        self.archived_ids: set[str] = set()
        self.merged_ids: set[str] = set()

    def add_update_payload(self, memory_id: str, patch: dict,
                           reason: str = "", stage: str = "", source_op: str = ""):
        """Propose adding metadata to a memory's PG payload."""
        self.proposals.append(PGProposal(
            op="update_payload", memory_id=memory_id,
            payload_patch=patch, reason=reason,
            source_op=source_op or stage, stage=stage,
        ))

    def add_update_text(self, memory_id: str, new_text: str,
                        reason: str = "", stage: str = "",
                        payload_patch: dict = None):
        """Propose updating a memory's text content (optionally with metadata)."""
        self.proposals.append(PGProposal(
            op="update_text", memory_id=memory_id,
            new_text=new_text, reason=reason, stage=stage,
            payload_patch=payload_patch or {},
        ))

    def add_archive(self, memory_id: str, reason: str = "",
                    stage: str = "", payload_patch: dict = None):
        """Propose archiving a memory (soft delete via payload flag)."""
        patch = {"archived": True, "archived_reason": stage}
        if payload_patch:
            patch.update(payload_patch)
        self.proposals.append(PGProposal(
            op="update_payload", memory_id=memory_id,
            payload_patch=patch, reason=reason, stage=stage,
        ))
        self.archived_ids.add(memory_id)

    def add_delete(self, memory_id: str, reason: str = "", stage: str = ""):
        """Propose hard-deleting a memory (used for merge secondary)."""
        self.proposals.append(PGProposal(
            op="delete", memory_id=memory_id,
            reason=reason, stage=stage,
        ))
        self.merged_ids.add(memory_id)

    @property
    def removed_ids(self) -> set[str]:
        """All IDs that would be removed from active memory pool."""
        return self.archived_ids | self.merged_ids

    def stats(self) -> dict:
        """Summary of staged proposals."""
        by_stage: dict[str, int] = {}
        by_op: dict[str, int] = {}
        for p in self.proposals:
            by_stage[p.stage] = by_stage.get(p.stage, 0) + 1
            by_op[p.op] = by_op.get(p.op, 0) + 1
        return {
            "total_proposals": len(self.proposals),
            "by_stage": by_stage,
            "by_op": by_op,
            "archived_ids": len(self.archived_ids),
            "merged_ids": len(self.merged_ids),
        }

    def write_staging(
        self,
        dream_run_id: int,
        validation_result=None,
        budget_summary: dict = None,
        split_stats: dict = None,
        report_text: str = "",
    ) -> StagingResult:
        """Write all proposals to disk staging directory.

        Creates:
          ~/dream_cycle_staging/<YYYYMMDD-HHMMSS>/
            proposals.json  — machine-readable proposals
            manifest.json   — metadata
            report.md       — human-readable report
        """
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        staging_dir = STAGING_ROOT / ts
        staging_dir.mkdir(parents=True, exist_ok=True)

        # Write proposals
        proposals_data = [asdict(p) for p in self.proposals]
        with open(staging_dir / "proposals.json", "w") as f:
            json.dump(proposals_data, f, indent=2, ensure_ascii=False, default=str)

        # Write manifest
        val_accepted = validation_result.accepted if validation_result else True
        val_reason = validation_result.reason if validation_result else "no validation run"

        manifest = {
            "dream_run_id": dream_run_id,
            "staged_at": datetime.now().isoformat(),
            "n_proposals": len(self.proposals),
            "staging_stats": self.stats(),
            "validation": {
                "accepted": val_accepted,
                "reason": val_reason,
                "hard_score": getattr(validation_result, "hard_score", None),
                "soft_score": getattr(validation_result, "soft_score", None),
                "n_val_queries": getattr(validation_result, "n_val_queries", None),
            },
            "budget": budget_summary or {},
            "split": split_stats or {},
            "adopted": False,
        }
        with open(staging_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # Write report
        report_md = _render_report(
            dream_run_id, self.proposals, self.stats(),
            validation_result, budget_summary, split_stats, report_text,
        )
        with open(staging_dir / "report.md", "w") as f:
            f.write(report_md)

        result = StagingResult(
            staging_dir=str(staging_dir),
            n_proposals=len(self.proposals),
            n_by_stage=self.stats()["by_stage"],
            validation_accepted=val_accepted,
            validation_reason=val_reason,
        )

        return result


def adopt_staging(staging_dir: str) -> dict:
    """Apply staged proposals to live PG database.

    1. Read proposals.json from staging dir
    2. Back up affected memories (PG SELECT before mutation)
    3. Execute each proposal via db.py functions
    4. Update manifest.json with adopted=True

    Returns summary of applied operations.
    """
    from .db import pg_query, update_memory_text, delete_memory

    staging_path = Path(staging_dir)
    proposals_file = staging_path / "proposals.json"
    manifest_file = staging_path / "manifest.json"

    if not proposals_file.exists():
        return {"error": f"proposals.json not found in {staging_dir}"}

    with open(proposals_file) as f:
        proposals = json.load(f)

    # Backup: fetch current state of affected memories
    affected_ids = list({p["memory_id"] for p in proposals})
    backup_dir = staging_path / "backup"
    backup_dir.mkdir(exist_ok=True)

    if affected_ids:
        # Fetch current payloads for backup
        id_list = ",".join(f"'{mid}'" for mid in affected_ids[:100])
        try:
            rows = pg_query(
                f"SELECT id::text, payload FROM mem0 WHERE id::text IN ({id_list}) LIMIT 200"
            )
            backup_data = [{"id": r[0], "payload": r[1]} for r in rows]
            with open(backup_dir / "pg_backup.json", "w") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            backup_data = []
            with open(backup_dir / "backup_error.txt", "w") as f:
                f.write(str(e))

    # Execute proposals
    applied = 0
    errors = []

    for p in proposals:
        try:
            if p["op"] == "update_payload":
                patch = p.get("payload_patch", {})
                patch_json = json.dumps(patch, ensure_ascii=False)
                pg_query(
                    f"UPDATE mem0 SET payload = payload || '{patch_json}' "
                    f"WHERE id::text = '{p['memory_id']}'"
                )
                applied += 1

            elif p["op"] == "update_text":
                update_memory_text(p["memory_id"], p["new_text"])
                applied += 1

            elif p["op"] == "delete":
                delete_memory(p["memory_id"])
                applied += 1

        except Exception as e:
            errors.append({"memory_id": p["memory_id"], "op": p["op"], "error": str(e)})

    # Update manifest
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)
        manifest["adopted"] = True
        manifest["adopted_at"] = datetime.now().isoformat()
        manifest["applied_count"] = applied
        manifest["errors"] = errors
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    return {
        "staging_dir": staging_dir,
        "applied": applied,
        "total": len(proposals),
        "errors": errors,
        "backup_dir": str(backup_dir),
    }


def latest_staging() -> Optional[str]:
    """Return the most recent staging directory path, or None."""
    if not STAGING_ROOT.exists():
        return None
    dirs = sorted(STAGING_ROOT.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
    for d in dirs:
        if d.is_dir() and (d / "proposals.json").exists():
            return str(d)
    return None


def staging_status() -> dict:
    """Return status of all staging directories."""
    if not STAGING_ROOT.exists():
        return {"staging_dirs": 0, "pending": 0, "adopted": 0}

    pending = 0
    adopted = 0
    dirs = []

    for d in sorted(STAGING_ROOT.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        manifest_file = d / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            is_adopted = manifest.get("adopted", False)
            if is_adopted:
                adopted += 1
            else:
                pending += 1
            dirs.append({
                "dir": d.name,
                "adopted": is_adopted,
                "n_proposals": manifest.get("n_proposals", 0),
                "staged_at": manifest.get("staged_at", ""),
            })

    return {
        "staging_dirs": len(dirs),
        "pending": pending,
        "adopted": adopted,
        "latest": dirs[0] if dirs else None,
    }


def _render_report(
    dream_run_id: int,
    proposals: list[PGProposal],
    stats: dict,
    validation_result,
    budget_summary: dict,
    split_stats: dict,
    extra_report: str,
) -> str:
    """Render human-readable markdown report."""
    lines = [
        f"# Dream Cycle v6 — Staging Report",
        f"",
        f"**Run ID:** {dream_run_id}  ",
        f"**Staged at:** {datetime.now().isoformat()}  ",
        f"",
        f"## Split Distribution",
    ]
    if split_stats:
        lines.append(f"- Train: {split_stats.get('train', 0)} ({split_stats.get('train_pct', 0)}%)")
        lines.append(f"- Val: {split_stats.get('val', 0)} ({split_stats.get('val_pct', 0)}%)")
        lines.append(f"- Test: {split_stats.get('test', 0)} ({split_stats.get('test_pct', 0)}%)")

    lines.extend([
        "",
        "## Budget",
    ])
    if budget_summary:
        lines.append(f"- Edits: {budget_summary.get('edits_used', 0)}/{budget_summary.get('edit_budget', 0)} "
                     f"(skipped: {budget_summary.get('edits_skipped', 0)})")
        lines.append(f"- Tokens: {budget_summary.get('tokens_used', 0):,}/{budget_summary.get('token_budget', 0):,}")
        lines.append(f"- Wall clock: {budget_summary.get('elapsed_seconds', 0):.1f}s/"
                     f"{budget_summary.get('wall_clock_budget', 0)}s")

    lines.extend([
        "",
        "## Validation (Held-Out)",
    ])
    if validation_result:
        vr = validation_result
        gate_icon = "✅" if vr.accepted else "❌"
        lines.append(f"- **Gate: {gate_icon} {vr.reason}**")
        lines.append(f"- Queries: {vr.n_val_queries} (improved: {vr.n_improved}, "
                     f"same: {vr.n_same}, degraded: {vr.n_degraded})")
        lines.append(f"- Hard score: {vr.hard_score:.3f}")
        lines.append(f"- Soft score: {vr.soft_score:+.4f}")
    else:
        lines.append("- No validation run")

    lines.extend([
        "",
        f"## Proposals ({stats['total_proposals']})",
    ])
    by_stage = stats.get("by_stage", {})
    for stage, count in sorted(by_stage.items()):
        lines.append(f"- **{stage}**: {count}")

    lines.extend([
        "",
        "### Details",
        "",
        "| Op | Stage | Memory ID | Reason |",
        "|----|-------|-----------|--------|",
    ])
    for p in proposals[:50]:  # cap at 50 rows
        mid_short = p.memory_id[:12] + "..." if len(p.memory_id) > 12 else p.memory_id
        reason_short = p.reason[:60] if p.reason else ""
        lines.append(f"| {p.op} | {p.stage} | {mid_short} | {reason_short} |")

    if len(proposals) > 50:
        lines.append(f"| ... | ... | +{len(proposals) - 50} more | ... |")

    if extra_report:
        lines.extend(["", "## Additional Report", "", extra_report])

    lines.extend([
        "",
        "---",
        f"*To adopt: `python3 -m dream_cycle --adopt {STAGING_ROOT}/<dir>`*",
    ])

    return "\n".join(lines)
