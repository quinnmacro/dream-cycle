#!/usr/bin/env python3
"""
NotebookLM deep operations library.
- Batch import with deduplication
- Smart Q&A with auto-save + vault export
- Artifact generation with polling
- Source health check

Usage:
    python3 notebooklm_ops.py import --notebook <id> --dir /path/to/files/
    python3 notebooklm_ops.py ask --notebook <id> --question "..." --save --export-vault
    python3 notebooklm_ops.py generate --notebook <id> --type mind-map
    python3 notebooklm_ops.py health --notebook <id>
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

NB_PATH = "/root/.hermes/hermes-agent/venv/bin/notebooklm"
PROFILE_DIR = Path("/root/.notebooklm/profiles/default")
VAULT_ROOT = Path("/root/vault")
COOKIE_REFRESH_SCRIPT = Path("/root/scripts/refresh_notebooklm_cookies.sh")


def run_nb(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    """Run notebooklm CLI command using absolute binary path."""
    result = subprocess.run(
        [NB_PATH] + args,
        capture_output=True, text=True, timeout=timeout
    )
    return result


def ensure_auth() -> bool:
    """Check and refresh auth if needed."""
    result = run_nb(["doctor"])
    # Unicode ✓ in table output, check multiple patterns
    if "authenticated" in result.stdout or "All checks passed" in result.stdout:
        return True
    print("⚠️ Auth failed. Attempting cookie refresh...")
    subprocess.run(["bash", str(COOKIE_REFRESH_SCRIPT)], timeout=30)
    result = run_nb(["doctor"])
    return "authenticated" in result.stdout or "All checks passed" in result.stdout


def get_notebook_id(title_or_id: str) -> str | None:
    """Resolve partial notebook ID or title to full ID."""
    result = run_nb(["list"])
    if result.returncode != 0:
        return None
    # Try partial ID match first
    for line in result.stdout.split("\n"):
        if title_or_id[:8] in line:
            match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}', line)
            if match:
                return match.group()
    # Try title match
    for line in result.stdout.split("\n"):
        if title_or_id in line:
            match = re.search(r'[0-9a-f]{8}-[0-9a-f]{4}', line)
            if match:
                return match.group()
    return None


def cmd_import(args):
    """Batch import files to notebook with deduplication."""
    nb_id = get_notebook_id(args.notebook)
    if not nb_id:
        print(f"❌ Notebook not found: {args.notebook}")
        sys.exit(1)

    run_nb(["use", nb_id])

    # Get existing sources for dedup
    existing_result = run_nb(["source", "list"])
    existing_titles = set()
    for line in existing_result.stdout.split("\n"):
        # Extract source titles
        if line.strip() and not line.startswith("┏") and not line.startswith("┃"):
            existing_titles.add(line.strip())

    files_dir = Path(args.dir)
    supported_exts = {".md", ".pdf", ".txt", ".docx", ".epub", ".html"}
    skip_patterns = re.compile(r'^(archive|backup|old|_)', re.IGNORECASE)

    imported = 0
    skipped = 0
    deduped = 0

    for f in sorted(files_dir.iterdir()):
        if f.suffix.lower() not in supported_exts:
            continue
        if skip_patterns.match(f.name):
            print(f"⏭️ Skip (archive): {f.name}")
            skipped += 1
            continue
        # Dedup check — skip if title already in notebook
        base_name = f.stem
        if base_name in existing_titles or f.name in existing_titles:
            print(f"⏭️ Skip (duplicate): {f.name}")
            deduped += 1
            continue

        print(f"📥 Importing: {f.name}")
        result = run_nb(["source", "add", str(f)], timeout=60)
        if result.returncode == 0:
            imported += 1
        else:
            print(f"⚠️ Failed: {result.stderr[:100]}")

    print(f"\n📊 Import complete: {imported} imported, {deduped} deduped, {skipped} skipped")


def cmd_ask(args):
    """Ask question with auto-save and optional vault export."""
    nb_id = get_notebook_id(args.notebook)
    if not nb_id:
        print(f"❌ Notebook not found: {args.notebook}")
        sys.exit(1)

    run_nb(["use", nb_id])

    ask_args = ["ask"]
    if args.save:
        ask_args.extend(["--save-as-note", "--note-title", args.title or args.question[:40]])
    ask_args.append(args.question)

    print(f"❓ Asking: {args.question[:60]}...")
    result = run_nb(ask_args, timeout=180)

    if result.returncode != 0:
        print(f"❌ Ask failed: {result.stderr[:200]}")
        sys.exit(1)

    # Extract answer text
    answer = result.stdout
    # Remove conversation ID line at the end
    answer_clean = re.sub(r'Conversation: [0-9a-f-]+\n.*', '', answer)

    print(answer_clean[:2000])
    if len(answer_clean) > 2000:
        print(f"... ({len(answer_clean)} chars total)")

    if args.export_vault:
        # Export to Vault
        category = args.category or "research"
        date_str = datetime.now().strftime("%Y-%m-%d")
        slug = re.sub(r'[^\w-]', '', args.question[:50].lower().replace(" ", "-"))[:60]
        vault_path = VAULT_ROOT / category / f"{date_str}-{slug}.md"

        # Get notebook title for provenance
        status_result = run_nb(["status"])
        nb_title = "Unknown"
        for line in status_result.stdout.split("\n"):
            if "Notebook" in line:
                nb_title = line.split("—")[-1].strip()

        frontmatter = f"""---
title: {args.question[:60]}
date: {date_str}
category: {category}
source_type: notebooklm
tags: [cross-document-analysis, notebooklm]
maturity: seedling
last-revised: {date_str}
confidence: 0.9
provenance: NotebookLM Q&A (notebook: {nb_title}, ID: {nb_id})
---

# {args.question[:60]}

{answer_clean}

---
*Source: NotebookLM notebook "{nb_title}" ({nb_id})*
*Exported: {date_str}*
"""

        vault_path.write_text(frontmatter)
        print(f"\n✅ Exported to Vault: {vault_path}")
        print(f"   Run: cd ~/vault && git add -A && git commit -m 'add: notebooklm q&a' && git push")


def cmd_generate(args):
    """Generate artifact with polling."""
    nb_id = get_notebook_id(args.notebook)
    if not nb_id:
        print(f"❌ Notebook not found: {args.notebook}")
        sys.exit(1)

    run_nb(["use", nb_id])

    artifact_type = args.type
    
    # 时间感知：audio/podcast 类型自动加 description 引导
    # 防止 NotebookLM 把过期市场数据当现状叙述
    auto_description = ""
    if artifact_type in ("audio", "podcast") and not args.description:
        from datetime import datetime, timezone, timedelta
        HKT = timezone(timedelta(hours=8))
        today = datetime.now(HKT).strftime("%B %d, %Y")
        auto_description = (
            f"Today is {today}. "
            "When sources mention specific market prices, yields, or spreads, "
            "treat them as historical snapshots from their publication date — do NOT present them as current levels. "
            "Focus on structural frameworks, causal mechanisms, and analytical methodology. "
            "If data from different dates conflicts, note the timeline."
        )
    
    print(f"🎨 Generating: {artifact_type}...")
    
    gen_args = ["generate", artifact_type]
    if args.language:
        gen_args.extend(["--language", args.language])
    if args.format:
        gen_args.extend(["--format", args.format])
    if auto_description:
        gen_args.append(auto_description)
        print(f"  📝 Auto-injected time-aware description")
    elif args.description:
        gen_args.append(args.description)

    result = run_nb(gen_args, timeout=300)

    if result.returncode != 0:
        print(f"⚠️ Generation issue: {result.stderr[:200]}")
        if "removed by server" in result.stderr.lower() or "removed by server" in result.stdout.lower():
            print("💡 This may be a quota/rate limit. Wait 24h and retry.")
        sys.exit(1)

    print(result.stdout[:3000])

    # Try to download
    if args.download:
        print(f"\n📥 Downloading...")
        dl_result = run_nb(["download", artifact_type, "--output", args.download_dir or "/root/Desktop/"])
        print(dl_result.stdout[:500])


def cmd_health(args):
    """Check notebook health: auth, sources, indexing status."""
    # Auth check
    auth_ok = ensure_auth()
    print(f"Auth: {'✅' if auth_ok else '❌'}")

    if not auth_ok:
        print("Fix auth first: bash /root/scripts/refresh_notebooklm_cookies.sh")
        return

    # Notebook check
    nb_id = get_notebook_id(args.notebook)
    if not nb_id:
        print(f"❌ Notebook not found: {args.notebook}")
        return

    run_nb(["use", nb_id])

    # Source check
    src_result = run_nb(["source", "list"])
    print(f"\nSources:\n{src_result.stdout[:1000]}")

    # Stale check
    stale_result = run_nb(["source", "stale"])
    if stale_result.stdout.strip():
        print(f"\n⚠️ Stale sources:\n{stale_result.stdout[:500]}")
    else:
        print("\n✅ No stale sources")

    # Note count
    note_result = run_nb(["note", "list"])
    print(f"\nSaved notes:\n{note_result.stdout[:500]}")


def main():
    parser = argparse.ArgumentParser(description="NotebookLM deep operations")
    subparsers = parser.add_subparsers(dest="command")

    # import
    p_import = subparsers.add_parser("import", help="Batch import files with dedup")
    p_import.add_argument("--notebook", required=True, help="Notebook ID or title")
    p_import.add_argument("--dir", required=True, help="Directory of files to import")

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask question with save/export")
    p_ask.add_argument("--notebook", required=True)
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--save", action="store_true", help="Save answer as notebook note")
    p_ask.add_argument("--title", help="Note title (default: question[:40])")
    p_ask.add_argument("--export-vault", action="store_true", help="Export answer to Vault")
    p_ask.add_argument("--category", default="research", help="Vault category")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate artifact")
    p_gen.add_argument("--notebook", required=True)
    p_gen.add_argument("--type", required=True, help="mind-map/audio/video/slide-deck/quiz/flashcards/report/infographic/data-table")
    p_gen.add_argument("--language", help="Language code (e.g. zh_Hans)")
    p_gen.add_argument("--format", help="Audio format: deep-dive/brief/critique/debate")
    p_gen.add_argument("--download", action="store_true")
    p_gen.add_argument("--download-dir", default="/root/")
    p_gen.add_argument("--description", help="Custom description for generation (auto-injected for audio if omitted)")

    # health
    p_health = subparsers.add_parser("health", help="Check notebook health")
    p_health.add_argument("--notebook", required=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Ensure auth before any operation
    ensure_auth()

    if args.command == "import":
        cmd_import(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "health":
        cmd_health(args)


if __name__ == "__main__":
    main()