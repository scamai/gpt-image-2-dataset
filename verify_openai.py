"""
Verify a random sample of confirmed GPT-image-2 images using OpenAI vision.

The default model is `gpt-5.4-mini` (override with --model). Visual assessment
of whether each image is AI-generated and consistent with GPT-image-2 output.
Note: C2PA watermarks are stripped by Twitter's CDN, so this is visual
verification, not cryptographic watermark detection.

The output JSONL fields prefixed `gpt4o_` (e.g. `gpt4o_label`,
`gpt4o_input_tokens`) are legacy names retained for compatibility with
existing analysis tooling — they are populated by whichever model `--model`
selects, not specifically GPT-4o.

Usage:
    python verify_openai.py --scrape-dir scrape_10k              # sample 100 images
    python verify_openai.py --scrape-dir scrape_10k --sample 50  # smaller sample
    python verify_openai.py --scrape-dir scrape_10k --report     # print existing results
"""

import argparse
import asyncio
import base64
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

DEFAULT_SAMPLE = None  # None = all records (no sampling)
DEFAULT_MODEL = "gpt-5.4-mini"

VERIFY_PROMPT = """\
You are verifying images for a research dataset of GPT-image-2 generated content.

Tweet context (the tweet where this image was posted):
{tweet_text}

Examine this image carefully and determine:
1. Is this an AI-generated image (not a real photograph or screenshot)?
2. Is the visual style and quality consistent with GPT-image-2 / ChatGPT's image \
generation feature? GPT-image-2 images tend to have: ultra-high photorealism or \
detailed illustration quality, clean rendering, coherent text if present, and \
consistent lighting. They differ from older models (Stable Diffusion, Midjourney v5) \
which have more obvious artifacts.
3. Is this a standalone generated image — not a comparison grid showing multiple AI \
models, not a screenshot of a chat interface, not a promotional graphic?

Respond ONLY with valid JSON on a single line:
{{"label": "confirmed|rejected|uncertain", "is_ai_generated": true|false, \
"consistent_with_gpt_image_2": true|false, "confidence": "high|medium|low", \
"reason": "one sentence explaining your assessment"}}"""


def _media_type(path: Path) -> str:
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".gif": "image/gif", ".webp": "image/webp"}.get(path.suffix.lower(), "image/jpeg")


def _load_records(scrape_dir: Path, label: str = "confirmed") -> list[dict]:
    curated = scrape_dir / "curated_metadata.jsonl"
    if not curated.exists():
        print(f"ERROR: {curated} not found. Run curate_dataset.py first.")
        sys.exit(1)
    records = []
    with curated.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                match = (label == "all") or (r.get("classification") == label)
                if match and r.get("local_path") and Path(r["local_path"]).exists():
                    records.append(r)
    return records


def _load_existing_results(results_file: Path) -> set[tuple]:
    done = set()
    if results_file.exists():
        with results_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((r["tweet_id"], r.get("media_key", "")))
    return done


async def _verify_one(
    client: AsyncOpenAI,
    record: dict,
    sem: asyncio.Semaphore,
) -> dict:
    path = Path(record["local_path"])
    b64 = base64.standard_b64encode(path.read_bytes()).decode()
    mime = _media_type(path)
    prompt = VERIFY_PROMPT.format(tweet_text=record.get("text", "")[:500])

    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=DEFAULT_MODEL,
                max_completion_tokens=150,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
            result = json.loads(raw)
            return {
                **record,
                "gpt4o_label": result.get("label", "uncertain"),
                "gpt4o_is_ai_generated": result.get("is_ai_generated"),
                "gpt4o_consistent_with_gpt_image_2": result.get("consistent_with_gpt_image_2"),
                "gpt4o_confidence": result.get("confidence", "low"),
                "gpt4o_reason": result.get("reason", ""),
                "gpt4o_verified_at": datetime.now(timezone.utc).isoformat(),
                "gpt4o_input_tokens": resp.usage.prompt_tokens,
                "gpt4o_output_tokens": resp.usage.completion_tokens,
            }
        except Exception as exc:
            return {
                **record,
                "gpt4o_label": "error",
                "gpt4o_reason": str(exc)[:200],
                "gpt4o_verified_at": datetime.now(timezone.utc).isoformat(),
                "gpt4o_input_tokens": 0,
                "gpt4o_output_tokens": 0,
            }


async def run_verification(scrape_dir: Path, sample_size: int | None, label: str, model: str) -> list[dict]:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    results_file = scrape_dir / "verification_results.jsonl"
    pool = _load_records(scrape_dir, label)
    done_keys = _load_existing_results(results_file)

    remaining = [
        r for r in pool
        if (r["tweet_id"], r.get("media_key", "")) not in done_keys
    ]

    if not remaining:
        print("All images already verified. Use --report to see results.")
        return []

    to_verify = random.sample(remaining, min(sample_size, len(remaining))) if sample_size else remaining
    print(f"Pool ({label}): {len(pool)} images")
    print(f"Already verified: {len(done_keys)}")
    print(f"Sending {len(to_verify)} images to {model}...")

    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(10)
    results: list[dict] = []
    completed = 0

    tasks = [_verify_one(client, r, sem) for r in to_verify]
    # patch model into each call via closure captured in _verify_one
    global DEFAULT_MODEL
    DEFAULT_MODEL = model

    with results_file.open("a") as outfile:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            outfile.write(json.dumps(result) + "\n")
            outfile.flush()
            results.append(result)
            completed += 1
            label = result.get("gpt4o_label", "error")
            reason = result.get("gpt4o_reason", "")[:65]
            if completed % 10 == 0 or completed == len(tasks):
                total_in = sum(r.get("gpt4o_input_tokens", 0) for r in results)
                total_out = sum(r.get("gpt4o_output_tokens", 0) for r in results)
                print(f"  [{completed:>3}/{len(tasks)}] {label:<12} tokens: {total_in:,} in / {total_out:,} out")

    return results


def print_report(scrape_dir: Path) -> None:
    results_file = scrape_dir / "verification_results.jsonl"
    if not results_file.exists():
        print("No verification results found.")
        return

    results = []
    with results_file.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    n = len(results)
    if not n:
        print("No results.")
        return

    confirmed_pool = len(_load_records(scrape_dir, "confirmed")) + len(_load_records(scrape_dir, "uncertain"))
    labels = Counter(r.get("gpt4o_label") for r in results)
    ai_gen = sum(1 for r in results if r.get("gpt4o_is_ai_generated") is True)
    gpt_style = sum(1 for r in results if r.get("gpt4o_consistent_with_gpt_image_2") is True)
    total_in = sum(r.get("gpt4o_input_tokens", 0) for r in results)
    total_out = sum(r.get("gpt4o_output_tokens", 0) for r in results)

    # Rough GPT-4o cost: $2.50/1M input, $10/1M output
    cost = (total_in * 2.50 + total_out * 10.0) / 1_000_000

    print("\n" + "=" * 56)
    print("    GPT-4o Verification Report")
    print("=" * 56)
    print(f"Confirmed pool size:  {confirmed_pool}")
    print(f"Sample verified:      {n}")

    print(f"\nGPT-4o classification:")
    for label in ("confirmed", "rejected", "uncertain", "error"):
        count = labels.get(label, 0)
        pct = count / n * 100 if n else 0
        bar = "█" * int(pct / 2.5)
        print(f"  {label:<12} {count:>4}  ({pct:5.1f}%)  {bar}")

    print(f"\nImage-level signals ({n} verified):")
    print(f"  AI-generated (not real photo):      {ai_gen:>4} / {n}  ({ai_gen/n*100:.1f}%)")
    print(f"  Consistent with GPT-image-2 style:  {gpt_style:>4} / {n}  ({gpt_style/n*100:.1f}%)")

    if labels.get("rejected", 0) > 0:
        print(f"\nTop rejection reasons:")
        rej_reasons = Counter(
            r.get("gpt4o_reason", "")
            for r in results if r.get("gpt4o_label") == "rejected"
        )
        for reason, count in rej_reasons.most_common(5):
            print(f"  {count:>3}  {reason[:70]}")

    precision = labels.get("confirmed", 0) / n * 100 if n else 0
    print(f"\nEstimated precision: {precision:.1f}%  {'✓ PASSES 95% threshold' if precision >= 95 else '✗ BELOW 95% threshold — review rejections'}")
    print(f"API cost: ${cost:.2f}  ({total_in:,} input / {total_out:,} output tokens)")
    print(f"Results: {results_file}")
    print("=" * 56 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify confirmed GPT-image-2 images using GPT-4o vision"
    )
    parser.add_argument("--scrape-dir", type=Path, default=Path("scrape_10k"),
                        help="Scrape directory containing curated_metadata.jsonl")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of images to verify (default: all)")
    parser.add_argument("--label", default="confirmed",
                        choices=["confirmed", "uncertain", "all"],
                        help="Which classified records to verify (default: confirmed)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--report", action="store_true",
                        help="Print report from existing results without verifying")
    args = parser.parse_args()

    if args.report:
        print_report(args.scrape_dir)
        return

    results = asyncio.run(run_verification(args.scrape_dir, args.sample, args.label, args.model))
    if results:
        print_report(args.scrape_dir)


if __name__ == "__main__":
    main()
