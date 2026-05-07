#!/usr/bin/env python3
"""
NotebookLM parallel artifact generation for Vault notebooks.

After syncing new sources, generate mindmap + flashcards in parallel
to enrich the knowledge base.

Usage:
    python3 notebooklm_parallel_generate.py [--notebook <id>] [--types mindmap,flashcards] [--poll]
"""

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

NB_PATH = "/root/.hermes/hermes-agent/venv/bin/notebooklm"
DEFAULT_TYPES = ["mind-map", "flashcards"]


def run_nb(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([NB_PATH] + args, capture_output=True, text=True, timeout=timeout)


def get_vault_notebooks() -> list[dict]:
    """Get all Vault: notebooks."""
    result = run_nb(["list", "--json"])
    try:
        data = json.loads(result.stdout)
        # error can be bool True or str — check truthy
        if data.get("error") is not None and data.get("error") is not False:
            return []
        return [nb for nb in data.get("notebooks", [])
                if nb.get("title", "").startswith("Vault:") or "内亚" in nb.get("title", "")]
    except json.JSONDecodeError:
        return []


def get_category_description(nb_title: str) -> str:
    """
    根据笔记本标题自动生成时间感知的 audio description
    
    NotebookLM generate audio 接受 DESCRIPTION 参数控制内容倾向。
    不传=盲编（会把4月价格当现状），传了=引导方向。
    
    关键原则：
    - research 类：强调"结构性框架"，不要具体数字
    - markets 类：强调"机制和驱动因素"，数据仅供参考
    - 通用：标注当前日期让 NotebookLM 知道"现在"是哪天
    """
    from datetime import datetime, timezone, timedelta
    HKT = timezone(timedelta(hours=8))
    today = datetime.now(HKT).strftime("%B %d, %Y")
    
    title_lower = nb_title.lower()
    
    if "research" in title_lower:
        return (
            f"Today is {today}. "
            "Focus on structural frameworks, causal mechanisms, and analytical methodology. "
            "When sources mention specific market prices, yields, or spreads, "
            "treat them as historical snapshots — do NOT present them as current levels. "
            "Instead, explain what drives those numbers and how the framework applies regardless of the exact level. "
            "Emphasize 'why' over 'what number'."
        )
    elif "market" in title_lower:
        return (
            f"Today is {today}. "
            "Focus on market mechanisms, structural drivers, and regime analysis. "
            "When sources mention specific prices, yields, or spreads, "
            "clearly indicate the date of that data point and whether it may be outdated. "
            "Prioritize the causal chain (why moves happen) over the specific level. "
            "If data from different dates conflicts, note the timeline and explain the change."
        )
    elif "内亚" in title_lower or "inner asia" in title_lower or "历史" in title_lower:
        return (
            f"Today is {today}. "
            "Focus on historical causality, cultural dynamics, and structural patterns across centuries. "
            "This is historical research — there are no 'outdated' facts, only evolving interpretations."
        )
    else:
        return (
            f"Today is {today}. "
            "When discussing specific data points or market levels from sources, "
            "note that they reflect the source's publication date and may not be current. "
            "Focus on conceptual understanding and structural relationships."
        )


def generate_artifact(nb_id: str, artifact_type: str, description: str = "") -> dict:
    """Generate a single artifact for a notebook. Returns status dict."""
    args = ["generate", artifact_type, "--notebook", nb_id]
    if description:
        args.append(description)

    start = time.time()
    result = run_nb(args, timeout=300)
    elapsed = time.time() - start

    # Parse artifact ID from output
    artifact_id = None
    match = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4})', result.stdout)
    if match:
        artifact_id = match.group(1)

    status = "success" if result.returncode == 0 else "failed"
    return {
        "nb_id": nb_id,
        "type": artifact_type,
        "artifact_id": artifact_id,
        "status": status,
        "elapsed": round(elapsed, 1),
        "output": result.stdout[:200] if result.returncode != 0 else "",
    }


def poll_artifacts(artifact_ids: list[str], timeout: int = 600) -> dict:
    """Poll until all artifacts are ready."""
    start = time.time()
    completed = set()
    failed = set()

    while time.time() - start < timeout:
        for aid in artifact_ids:
            if aid in completed or aid in failed:
                continue
            result = run_nb(["artifact", "poll", aid], timeout=30)
            if "completed" in result.stdout.lower() or "ready" in result.stdout.lower():
                completed.add(aid)
            elif "failed" in result.stdout.lower():
                failed.add(aid)

        if len(completed) + len(failed) >= len(artifact_ids):
            break
        time.sleep(10)

    return {"completed": len(completed), "failed": len(failed), "timeout": len(artifact_ids) - len(completed) - len(failed)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NotebookLM parallel artifact generation")
    parser.add_argument("--notebook", help="Specific notebook ID")
    parser.add_argument("--types", default="mind-map,flashcards", help="Comma-separated artifact types")
    parser.add_argument("--poll", action="store_true", help="Wait for all artifacts to complete")
    parser.add_argument("--description", default="", help="Custom description for generation")
    args = parser.parse_args()

    artifact_types = [t.strip() for t in args.types.split(",")]
    print(f"🎨 Generating {artifact_types} in parallel")

    # Get notebooks
    if args.notebook:
        result = run_nb(["list", "--json"])
        try:
            data = json.loads(result.stdout)
            if data.get("error") is not None and data.get("error") is not False:
                notebooks = []
            else:
                notebooks = [nb for nb in data.get("notebooks", [])
                         if nb["id"].startswith(args.notebook) or args.notebook in nb.get("title", "")]
        except json.JSONDecodeError:
            notebooks = []
    else:
        notebooks = get_vault_notebooks()

    if not notebooks:
        print("❌ No notebooks found")
        sys.exit(1)

    print(f"📚 {len(notebooks)} notebooks × {len(artifact_types)} artifact types = {len(notebooks) * len(artifact_types)} generations\n")

    # Build all generation tasks
    # 自动为每个笔记本生成时间感知的 description
    # 除非用户手动传了 --description，否则按类别自动选择
    tasks = []
    for nb in notebooks:
        for atype in artifact_types:
            # audio 类型自动注入时间感知 description
            if atype == "audio" and not args.description:
                auto_desc = get_category_description(nb["title"])
            else:
                auto_desc = args.description
            tasks.append((nb["id"], nb["title"], atype, auto_desc))

    # Execute in parallel (max 3 concurrent — avoid rate limiting)
    results = []
    artifact_ids = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_task = {
            executor.submit(generate_artifact, nb_id, atype, desc): (nb_title, atype)
            for nb_id, nb_title, atype, desc in tasks
        }

        for future in as_completed(future_to_task):
            nb_title, atype = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                icon = "✅" if result["status"] == "success" else "❌"
                print(f"  {icon} {nb_title} → {atype} ({result['elapsed']}s)")
                if result["artifact_id"]:
                    artifact_ids.append(result["artifact_id"])
            except Exception as e:
                print(f"  ❌ {nb_title} → {atype}: {e}")

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"\n📊 Generated: {success} success, {failed} failed")

    # Optionally poll for completion
    if args.poll and artifact_ids:
        print(f"\n⏳ Polling {len(artifact_ids)} artifacts...")
        poll_result = poll_artifacts(artifact_ids)
        print(f"  ✅ {poll_result['completed']} completed, ❌ {poll_result['failed']} failed, ⏰ {poll_result['timeout']} timed out")


if __name__ == "__main__":
    main()
