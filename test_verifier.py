"""
Controlled accuracy test for the GPT-5.4-mini verifier.

Sends two known sets to the verifier:
  - POSITIVE: confirmed images from the #GPTImage2 hashtag query (94% precision source)
  - NEGATIVE: rejected images from heuristic curation (comparison grids, real photos, etc.)

Reports per-set accuracy to validate whether the verifier is reliable before
trusting its 80% confirmation rate on uncertain images.

Usage:
    python test_verifier.py --scrape-dir scrape_yolo
    python test_verifier.py --scrape-dir scrape_yolo --model gpt-5.4-mini --sample 50
"""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

HASHTAG_QUERY = "#GPTImage2 OR #GPTimage2"  # substring to match query_used field


def _load_positive_set(scrape_dir: Path, n: int) -> list[dict]:
    """Load confirmed images from the high-precision hashtag query."""
    curated = scrape_dir / "curated_metadata.jsonl"
    pool = []
    with curated.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if (
                r.get("classification") == "confirmed"
                and "#GPTImage2" in r.get("query_used", "")
                and r.get("local_path")
                and Path(r["local_path"]).exists()
            ):
                pool.append(r)
    if not pool:
        print("ERROR: No confirmed hashtag-query images found. Check scrape-dir.")
        sys.exit(1)
    sample = random.sample(pool, min(n, len(pool)))
    print(f"Positive pool (hashtag confirmed): {len(pool)} → sampling {len(sample)}")
    return sample


def _load_negative_set(neg_dir: Path, n: int) -> list[dict]:
    """Load known non-GPT-image-2 images from non_gpt_image_2/metadata.jsonl."""
    meta = neg_dir / "metadata.jsonl"
    if not meta.exists():
        print(f"ERROR: {meta} not found. Run fetch_negatives.py first.")
        sys.exit(1)
    pool = []
    with meta.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("local_path") and Path(r["local_path"]).exists():
                pool.append(r)
    if not pool:
        print("ERROR: No images found in negative directory.")
        sys.exit(1)
    sample = random.sample(pool, min(n, len(pool)))
    by_source = {}
    for r in sample:
        src = r.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
    print(f"Negative pool ({neg_dir}): {len(pool)} total → sampling {len(sample)} {by_source}")
    return sample


async def verify_batch(
    records: list[dict],
    client: AsyncOpenAI,
    model: str,
    sem: asyncio.Semaphore,
) -> list[dict]:
    from verify_openai import _verify_one, DEFAULT_MODEL
    import verify_openai
    verify_openai.DEFAULT_MODEL = model

    tasks = [_verify_one(client, r, sem) for r in records]
    results = []
    for coro in asyncio.as_completed(tasks):
        results.append(await coro)
    return results


def print_set_report(label: str, results: list[dict], want_label: str, threshold: int) -> bool:
    n = len(results)
    counts = {"confirmed": 0, "rejected": 0, "uncertain": 0, "error": 0}
    for r in results:
        lbl = r.get("gpt4o_label", "error")
        counts[lbl] = counts.get(lbl, 0) + 1

    target_count = counts.get(want_label, 0)
    pct = target_count / n * 100 if n else 0
    passes = pct >= threshold

    print(f"\n{'='*54}")
    print(f"  {label}")
    print(f"{'='*54}")
    for lbl in ("confirmed", "rejected", "uncertain", "error"):
        count = counts[lbl]
        bar = "█" * int(count / n * 30) if n else ""
        marker = " ← TARGET" if lbl == want_label else ""
        print(f"  {lbl:<12} {count:>4} / {n}  ({count/n*100:5.1f}%)  {bar}{marker}")

    verdict = "✓ PASS" if passes else "✗ FAIL"
    print(f"\n  Target ({want_label} ≥{threshold}%): {pct:.1f}%  {verdict}")

    # Show misclassified examples
    wrong = [r for r in results if r.get("gpt4o_label") != want_label and r.get("gpt4o_label") != "error"]
    if wrong:
        print(f"\n  Misclassified ({len(wrong)}):")
        for r in wrong[:5]:
            tweet = r.get("text", "").replace("\n", " ")[:80]
            reason = r.get("gpt4o_reason", "")[:80]
            print(f"    [{r.get('gpt4o_label')}] {reason}")
            print(f"           tweet: {tweet}")
    print()
    return passes


async def run(scrape_dir: Path, neg_dir: Path, model: str, sample: int) -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    positives = _load_positive_set(scrape_dir, sample)
    negatives = _load_negative_set(neg_dir, sample)

    print(f"\nRunning verifier ({model}) on {len(positives)} positive + {len(negatives)} negative images...")

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(10)

    pos_results = await verify_batch(positives, client, model, sem)
    neg_results = await verify_batch(negatives, client, model, sem)

    print("\n" + "="*54)
    print("    VERIFIER ACCURACY TEST RESULTS")
    print("="*54)

    pos_pass = print_set_report(
        "POSITIVE SET — known GPT-image-2 (#GPTImage2 hashtag)",
        pos_results, "confirmed", 95
    )
    neg_pass = print_set_report(
        "NEGATIVE SET — known non-GPT-image-2 (heuristic rejected)",
        neg_results, "rejected", 90
    )

    print("="*54)
    if pos_pass and neg_pass:
        print("  OVERALL VERDICT: ✓ PASS — verifier is reliable")
        print("  → Safe to trust 80% confirmation rate on uncertain images")
    elif not pos_pass and not neg_pass:
        print("  OVERALL VERDICT: ✗ FAIL — verifier prompt needs redesign")
    elif not pos_pass:
        print("  OVERALL VERDICT: ✗ FAIL — verifier too strict (rejecting real GPT-image-2)")
    else:
        print("  OVERALL VERDICT: ✗ FAIL — verifier too lenient (accepting non-GPT-image-2)")
    print("="*54 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test GPT verifier accuracy with known positive/negative sets")
    parser.add_argument("--scrape-dir", type=Path, default=Path("scrape_yolo"))
    parser.add_argument("--neg-dir", type=Path, default=Path("non_gpt_image_2"),
                        help="Directory with known non-GPT-image-2 negatives")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--sample", type=int, default=50,
                        help="Images per set (default: 50)")
    args = parser.parse_args()
    asyncio.run(run(args.scrape_dir, args.neg_dir, args.model, args.sample))


if __name__ == "__main__":
    main()
