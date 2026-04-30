"""
Fetch known non-GPT-image-2 images for verifier accuracy testing.

Two sources:
  1. Picsum Photos  — real curated photographs (free, no API key)
  2. Twitter scrape — Midjourney images from BEFORE April 21 2026 (GPT-image-2 launch)

Output: non_gpt_image_2/
    picsum/          ← real photographs
    midjourney/      ← Midjourney-attributed AI images
    metadata.jsonl   ← one record per image with source + local_path

Usage:
    python fetch_negatives.py                     # 50 picsum + $0.50 midjourney scrape
    python fetch_negatives.py --picsum 30         # fewer picsum images
    python fetch_negatives.py --mj-budget 0       # skip midjourney scrape
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path("non_gpt_image_2")
METADATA_FILE = OUTPUT_DIR / "metadata.jsonl"

BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
COST_PER_TWEET = 0.005

# Midjourney images that explicitly exclude GPT-image-2 mentions
# (no end_time needed — exclusion terms ensure these aren't GPT-image-2 images)
MJ_START = "2026-04-19T00:00:00Z"
MJ_QUERY = 'midjourney has:images -is:retweet -"gpt-image-2" -"gpt image 2" -chatgpt -openai'

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _append_meta(record: dict) -> None:
    with METADATA_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ── 1. Picsum Photos (real photographs) ──────────────────────────────────────

async def fetch_picsum(n: int) -> int:
    dest_dir = OUTPUT_DIR / "picsum"
    dest_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %d real photographs from Picsum Photos...", n)
    downloaded = 0

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        for i in range(n):
            seed = i + 100          # seeds 100-149 → stable, varied images
            url = f"https://picsum.photos/seed/{seed}/1024/1024"
            dest = dest_dir / f"picsum_{i+1:03d}.jpg"

            if dest.exists():
                log.info("  [%d/%d] already exists, skipping", i + 1, n)
                downloaded += 1
                continue

            try:
                resp = await client.get(url)
                resp.raise_for_status()
                dest.write_bytes(resp.content)
                _append_meta({
                    "source": "picsum",
                    "type": "real_photograph",
                    "label": "negative",
                    "local_path": str(dest),
                    "image_url": str(resp.url),
                    "text": "",         # no tweet text
                    "query_used": "picsum",
                })
                downloaded += 1
                log.info("  [%d/%d] → %s  (%d KB)", i + 1, n, dest.name, len(resp.content) // 1024)
            except Exception as exc:
                log.warning("  [%d/%d] failed: %s", i + 1, n, exc)

    log.info("Picsum done: %d images", downloaded)
    return downloaded


# ── 2. Twitter Midjourney scrape ──────────────────────────────────────────────

async def scrape_midjourney(budget: float) -> int:
    if not BEARER_TOKEN:
        log.error("TWITTER_BEARER_TOKEN not set — skipping Midjourney scrape")
        return 0
    if budget <= 0:
        log.info("MJ budget=$0 — skipping Midjourney scrape")
        return 0

    dest_dir = OUTPUT_DIR / "midjourney"
    dest_dir.mkdir(parents=True, exist_ok=True)

    max_tweets = int(budget / COST_PER_TWEET)
    log.info("Scraping Midjourney images from Twitter (budget $%.2f = %d tweets)...", budget, max_tweets)
    log.info("Query: %s", MJ_QUERY)

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    base_params = {
        "query": MJ_QUERY,
        "tweet.fields": "id,text,created_at,author_id,public_metrics",
        "user.fields": "id,username,name",
        "media.fields": "media_key,type,url,preview_image_url,width,height",
        "expansions": "attachments.media_keys,author_id",
    }

    tweets_read = 0
    images_collected = 0
    next_token = None
    page = 0
    seen_image_urls: set[str] = set()

    async with httpx.AsyncClient(timeout=30) as client:
        while tweets_read < max_tweets:
            page_size = min(100, max_tweets - tweets_read)
            params = {**base_params, "max_results": max(10, page_size)}
            if next_token:
                params["next_token"] = next_token

            try:
                resp = await client.get(url, params=params, headers=headers)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("x-rate-limit-reset", time.time() + 60)) - int(time.time())
                    log.warning("Rate limited — sleeping %ds", max(wait, 5))
                    await asyncio.sleep(max(wait, 5))
                    continue
                resp.raise_for_status()
            except Exception as exc:
                log.error("Search failed: %s", exc)
                break

            body = resp.json()
            page += 1
            raw_tweets = body.get("data", [])
            includes = body.get("includes", {})
            media_map = {m["media_key"]: m for m in includes.get("media", [])}
            user_map = {u["id"]: u for u in includes.get("users", [])}

            tweets_read += len(raw_tweets)
            log.info("  page %d: +%d tweets | $%.3f spent", page, len(raw_tweets), tweets_read * COST_PER_TWEET)

            # Download images
            for tweet in raw_tweets:
                author = user_map.get(tweet.get("author_id", ""), {})
                for mk in tweet.get("attachments", {}).get("media_keys", []):
                    media = media_map.get(mk)
                    if not media or media.get("type") != "photo":
                        continue
                    image_url = media.get("url") or media.get("preview_image_url")
                    if not image_url or image_url in seen_image_urls:
                        continue
                    seen_image_urls.add(image_url)

                    ext = "png" if ".png" in image_url else "jpg"
                    dest = dest_dir / f"{tweet['id']}_{mk}.{ext}"
                    if dest.exists():
                        continue

                    try:
                        img_resp = await client.get(
                            image_url + "?format=jpg&name=orig",
                            follow_redirects=True, timeout=30,
                        )
                        img_resp.raise_for_status()
                        dest.write_bytes(img_resp.content)
                        _append_meta({
                            "source": "twitter_midjourney",
                            "type": "midjourney_ai",
                            "label": "negative",
                            "local_path": str(dest),
                            "image_url": image_url,
                            "tweet_id": tweet["id"],
                            "media_key": mk,
                            "text": tweet.get("text", ""),
                            "author_username": author.get("username", ""),
                            "query_used": MJ_QUERY,
                        })
                        images_collected += 1
                        log.info("    [img %d] %s", images_collected, dest.name)
                    except Exception as exc:
                        log.warning("    download failed: %s", exc)

            next_token = body.get("meta", {}).get("next_token")
            if not next_token:
                log.info("  No more pages — query exhausted")
                break
            await asyncio.sleep(0.5)

    log.info("Midjourney scrape done: %d images from %d tweets ($%.3f)",
             images_collected, tweets_read, tweets_read * COST_PER_TWEET)
    return images_collected


# ── main ──────────────────────────────────────────────────────────────────────

async def run(picsum_n: int, mj_budget: float) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    existing = []
    if METADATA_FILE.exists():
        with METADATA_FILE.open() as f:
            existing = [json.loads(l) for l in f if l.strip()]
        log.info("Found %d existing records in metadata.jsonl", len(existing))

    pic_count = await fetch_picsum(picsum_n)
    mj_count = await scrape_midjourney(mj_budget)

    all_records = []
    if METADATA_FILE.exists():
        with METADATA_FILE.open() as f:
            all_records = [json.loads(l) for l in f if l.strip()]

    by_source = {}
    for r in all_records:
        src = r.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

    print(f"\n{'='*48}")
    print("    non_gpt_image_2 collection complete")
    print(f"{'='*48}")
    for src, count in by_source.items():
        print(f"  {src:<25} {count:>4} images")
    print(f"  {'TOTAL':<25} {len(all_records):>4} images")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"{'='*48}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch known non-GPT-image-2 negatives")
    parser.add_argument("--picsum", type=int, default=50,
                        help="Number of Picsum real photos to download (default: 50)")
    parser.add_argument("--mj-budget", type=float, default=0.50,
                        help="Twitter budget for Midjourney scrape in USD (default: $0.50, 0=skip)")
    args = parser.parse_args()
    asyncio.run(run(args.picsum, args.mj_budget))


if __name__ == "__main__":
    main()
