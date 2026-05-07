#!/usr/bin/env python3
"""
Vault → NotebookLM sync: detect new Vault files and push to matching notebooks.

Logic:
1. Scan vault categories for new/modified files since last sync
2. Match vault category → notebook by title prefix (using --json for reliable parsing)
3. Import new files to matching notebook (with dedup)
4. Track sync state in ~/.notebooklm/vault_sync_state.json

Usage:
    python3 vault_to_notebooklm_sync.py [--dry-run] [--category markets] [--force]
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

NB_PATH = "/root/.hermes/hermes-agent/venv/bin/notebooklm"
VAULT_ROOT = Path("/root/vault")
SYNC_STATE = Path("/root/.notebooklm/vault_sync_state.json")
SUPPORTED_EXT = {".md", ".pdf", ".txt", ".docx"}
SKIP_PATTERN = re.compile(r'^(archive|backup|old|_|status|inbox)', re.IGNORECASE)

# Category → Notebook title mapping
CATEGORY_MAP = {
    "readings": "Vault: Readings",
    "research": "Vault: Research",
    "markets": "Vault: Markets",
    # "projects": "Vault: Projects",  # Disabled — privacy, user deleted
    "concepts": "Vault: Concepts",
}


def run_nb(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    """Run notebooklm CLI command. Always use --json for structured output."""
    # Add --json flag for commands that support it
    json_commands = {"list", "source"}
    if args and args[0] in json_commands:
        # source subcommands: list, get, etc.
        if len(args) >= 2 and args[0] == "source" and args[1] in ("list", "get"):
            args = args[:2] + ["--json"] + args[2:]
        elif args[0] == "list":
            args = ["list", "--json"] + args[1:]
    return subprocess.run([NB_PATH] + args, capture_output=True, text=True, timeout=timeout)


def run_nb_json(args: list[str], timeout: int = 180) -> dict | None:
    """Run notebooklm CLI command with --json and parse the result."""
    result = run_nb(args, timeout=timeout)
    try:
        data = json.loads(result.stdout)
        # Check for auth errors in JSON response
        if data.get("error") and "Authentication expired" in str(data.get("message", "")):
            print("❌ NotebookLM auth expired. Aborting sync to prevent duplicate creation.")
            print("   Run: python3 /root/scripts/refresh_notebooklm_cookies.py")
            return None
        return data
    except json.JSONDecodeError:
        # Fallback: check stdout for non-JSON auth errors
        if "Authentication expired" in result.stdout or "sign in" in result.stdout.lower():
            print("❌ NotebookLM auth expired. Aborting sync to prevent duplicate creation.")
            print("   Run: python3 /root/scripts/refresh_notebooklm_cookies.py")
            return None
        return None


def load_sync_state() -> dict:
    if SYNC_STATE.exists():
        with open(SYNC_STATE) as f:
            return json.load(f)
    return {"last_sync": None, "synced_files": {}}


def save_sync_state(state: dict):
    with open(SYNC_STATE, "w") as f:
        json.dump(state, f, indent=2)


def get_notebooks() -> dict[str, str]:
    """Get notebook full-ID → title mapping using --json. Handles duplicates."""
    data = run_nb_json(["list"])
    if data is None:
        return {}

    notebooks = {}
    for nb in data.get("notebooks", []):
        nb_id = nb.get("id", "")
        title = nb.get("title", "").strip()
        if not nb_id or not title:
            continue
        if title not in notebooks.values():
            notebooks[nb_id] = title
        else:
            print(f"⚠️ Duplicate notebook skipped: {title} ({nb_id[:8]})")
    return notebooks


def find_matching_notebook(category: str, notebooks: dict, dry_run: bool = False) -> str | None:
    """Find notebook that matches vault category. dry_run mode won't create new notebooks."""
    expected_title = CATEGORY_MAP.get(category)
    if not expected_title:
        return None

    for nb_id, title in notebooks.items():
        if expected_title.lower() in title.lower() or category.lower() in title.lower():
            return nb_id

    # Create notebook if not found (skip in dry_run)
    if dry_run:
        print(f"⚠️ [DRY-RUN] Would create notebook: {expected_title}")
        return None
    print(f"📝 Creating notebook: {expected_title}")
    result = run_nb(["create", expected_title])
    # Parse created notebook ID from stdout (non-JSON output)
    match = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4})', result.stdout)
    if match:
        return match.group(1)
    return None


def get_existing_sources(nb_id: str) -> set[str]:
    """Get set of source titles already in notebook using --json (full titles, no truncation)."""
    run_nb(["use", nb_id])
    data = run_nb_json(["source", "list"])
    if data is None:
        return set()

    titles = set()
    for src in data.get("sources", []):
        title = src.get("title", "").strip()
        if title:
            titles.add(title)
    return titles


def sync_category(category: str, state: dict, dry_run: bool = False, force: bool = False) -> int:
    """Sync vault category to notebook."""
    cat_dir = VAULT_ROOT / category
    if not cat_dir.exists():
        return 0

    notebooks = get_notebooks()
    if not notebooks:
        print(f"⚠️ Auth failed or no notebooks found, skipping {category}")
        return 0

    nb_id = find_matching_notebook(category, notebooks, dry_run=dry_run)
    if not nb_id:
        print(f"⚠️ No matching notebook for category: {category}")
        return 0

    existing_sources = get_existing_sources(nb_id)

    imported = 0
    for f in sorted(cat_dir.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in SUPPORTED_EXT:
            continue
        if SKIP_PATTERN.match(f.name):
            continue

        # Check if already synced (use relative path as key)
        rel_path = str(f.relative_to(VAULT_ROOT))
        if not force and rel_path in state["synced_files"]:
            continue

        # 时间感知：检查 frontmatter 的 data_freshness 字段
        # stale 数据不推送到 NotebookLM（避免过期市场数据污染 audio overview）
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                head = fh.read(2000)  # 只读前2K够解析 frontmatter
            if head.startswith('---'):
                end = head.find('---', 3)
                if end > 0:
                    fm = head[3:end]
                    # 检查 data_freshness
                    for line in fm.split('\n'):
                        if line.strip().startswith('data_freshness:'):
                            freshness = line.split(':', 1)[1].strip().strip('"').strip("'")
                            if freshness == 'stale':
                                print(f"  ⏭️ Skipping stale data: {f.name}")
                                state["synced_files"][rel_path] = {
                                    "status": "skipped_stale",
                                    "time": datetime.now().isoformat()
                                }
                                break
                    else:
                        continue  # frontmatter 没有 data_freshness 字段，正常导入
                    continue  # stale 已跳过
        except Exception:
            pass  # 读取失败不阻断

        # 对 research/daily 目录的文件，检查日期 — 超过14天的日常研究不推送
        # 因为日常研究含市场定价数据，过期后误导性极高
        if '/research/daily/' in rel_path:
            date_match = re.match(r'(\d{4}-\d{2}-\d{2})', f.stem)
            if date_match:
                from datetime import datetime as dt_cls
                try:
                    file_date = dt_cls.strptime(date_match.group(1), "%Y-%m-%d")
                    age_days = (dt_cls.now() - file_date).days
                    if age_days > 14:
                        print(f"  ⏭️ Skipping old daily research ({age_days}d): {f.name}")
                        if rel_path not in state["synced_files"]:
                            state["synced_files"][rel_path] = {
                                "status": "skipped_old_daily",
                                "time": datetime.now().isoformat()
                            }
                        continue
                except ValueError:
                    pass

        # Check dedup in notebook — match by filename (stem or full name)
        base_name = f.stem
        full_name = f.name
        if base_name in existing_sources or full_name in existing_sources:
            state["synced_files"][rel_path] = {"status": "deduped", "time": datetime.now().isoformat()}
            continue

        if dry_run:
            print(f"  📥 Would import: {f.name} → notebook {nb_id[:8]}")
            imported += 1
            continue

        print(f"  📥 Importing: {f.name}")
        result = run_nb(["source", "add", str(f)], timeout=60)
        if result.returncode == 0:
            state["synced_files"][rel_path] = {"status": "imported", "time": datetime.now().isoformat(), "nb_id": nb_id}
            imported += 1
        else:
            print(f"  ⚠️ Failed: {result.stderr[:100]}")
            state["synced_files"][rel_path] = {"status": "failed", "time": datetime.now().isoformat()}

    return imported


def prune_stale_sources(category: str | None = None, max_age_days: int = 14, dry_run: bool = False) -> int:
    """
    清理 NotebookLM 中过期的 research/daily source
    
    过期的日常研究含4月市场定价数据，会误导 NotebookLM audio overview。
    删除超过 max_age_days 的 daily research source。
    
    Returns: 删除的 source 数量
    """
    from datetime import datetime as dt_cls
    
    categories = [category] if category else ["research"]
    total_pruned = 0
    
    for cat in categories:
        notebooks = get_notebooks()
        if not notebooks:
            print(f"⚠️ Auth failed, skipping {cat}")
            continue
        
        nb_id = find_matching_notebook(cat, notebooks, dry_run=True)
        if not nb_id:
            print(f"⚠️ No notebook for {cat}")
            continue
        
        run_nb(["use", nb_id])
        data = run_nb_json(["source", "list"])
        if data is None:
            continue
        
        for src in data.get("sources", []):
            title = src.get("title", "").strip()
            src_id = src.get("id", "")
            if not title or not src_id:
                continue
            
            # 检查标题是否含日期前缀（daily research 格式）
            date_match = re.match(r'(\d{4}-\d{2}-\d{2})', title)
            if not date_match:
                continue
            
            try:
                file_date = dt_cls.strptime(date_match.group(1), "%Y-%m-%d")
                age_days = (dt_cls.now() - file_date).days
                if age_days > max_age_days:
                    if dry_run:
                        print(f"  🗑️ Would remove: {title} ({age_days}d old)")
                    else:
                        result = run_nb(["source", "delete", src_id], timeout=30)
                        print(f"  🗑️ Removed: {title} ({age_days}d old)")
                    total_pruned += 1
            except ValueError:
                continue
    
    return total_pruned


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vault → NotebookLM sync")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--category", help="Specific category to sync")
    parser.add_argument("--force", action="store_true", help="Re-sync already synced files")
    parser.add_argument("--prune-stale", action="store_true", 
                        help="Remove stale research/daily sources from NotebookLM (>14 days old)")
    parser.add_argument("--max-age", type=int, default=14,
                        help="Max age in days for research sources (default: 14)")
    args = parser.parse_args()

    # Auth check via JSON
    data = run_nb_json(["list"])
    if data is None:
        sys.exit(1)

    # Prune stale sources mode
    if args.prune_stale:
        pruned = prune_stale_sources(
            category=args.category,
            max_age_days=args.max_age,
            dry_run=args.dry_run
        )
        print(f"\n📊 Total: {pruned} stale sources {'would be ' if args.dry_run else ''}removed")
        if args.dry_run:
            print("💡 This was a dry run. Run without --dry-run to actually remove.")
        return

    state = load_sync_state()
    total_imported = 0

    categories = [args.category] if args.category else CATEGORY_MAP.keys()

    for cat in categories:
        print(f"\n📂 Syncing: {cat}")
        n = sync_category(cat, state, dry_run=args.dry_run, force=args.force)
        total_imported += n
        print(f"  Imported: {n} files")

    state["last_sync"] = datetime.now().isoformat()
    save_sync_state(state)
    print(f"\n📊 Total: {total_imported} files synced")
    if args.dry_run:
        print("💡 This was a dry run. Run without --dry-run to actually import.")


if __name__ == "__main__":
    main()
