"""
One-shot scraper: collect posts with images mentioning GPT-image-2.0 (since Apr 21 2026).

Usage:
    python scrape_gpt_image2.py                           # $2 budget (default)
    python scrape_gpt_image2.py --budget 5.00
    python scrape_gpt_image2.py --budget 0                # unlimited (no budget cap)
    python scrape_gpt_image2.py --output-dir scrape_v2    # custom output directory

Cost: $0.005 per tweet read (X API pay-per-use).
Budget is enforced inside pagination — stops mid-query the moment cap is hit.

Output (default: scrape_output/):
    images/          ← downloaded image files (free — CDN, not X API)
    metadata.jsonl   ← one JSON record per image (append-safe)
    run_log.txt      ← human-readable run summary
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
COST_PER_TWEET = 0.005          # X API: $0.005 per post read
SINCE_DATE = "2026-04-21T00:00:00Z"   # GPT-image-2.0 release day (default)

# Model name variant expansion (coworker recommendation) — catches users who
# write "GPT image2", "ChatGPT image 2", "GPTimage2", etc.
_ALL_VARIANTS = (
    '"GPT image 2" OR "GPT-image-2" OR "GPT image2" OR "GPT-image2" OR '
    '"GPT image-2" OR "GPT-image 2" OR GPTimage2 OR '
    '"ChatGPT image 2" OR "ChatGPT images 2" OR "ChatGPT-image-2" OR ChatGPTimage2'
)

QUERIES = [
    # Hashtag community — highest confirmed rate (94%), now with all variant hashtags
    f'(#GPTImage2 OR #GPTimage2 OR #ChatGPTimage2 OR #GPTimage2) -introducing -"sign up"',

    # Explicit creation language across all model name variants
    f'({_ALL_VARIANTS}) (made OR created OR generated OR "prompt:")',

    # Japanese creation signals with expanded variants
    f'({_ALL_VARIANTS}) (で生成 OR 生成した OR で描い OR 作ってみた OR 試してみた)',

    # Chinese creation signals with expanded variants
    f'({_ALL_VARIANTS}) (提示词 OR 生成的 OR 用它生成 OR chatgpt生成)',

    # ChatGPT image generation with prompt signal
    '"ChatGPT" "image" "prompt" -(vs OR versus OR compare)',

    # Broad coverage — all variants, anti-noise filters
    f'({_ALL_VARIANTS}) -vs -versus -introducing -"sign up" -released -launched',
]

OUTPUT_DIR = Path("scrape_output")
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_FILE = OUTPUT_DIR / "metadata.jsonl"
LOG_FILE = OUTPUT_DIR / "run_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ── budget tracker ────────────────────────────────────────────────────────────

class Budget:
    def __init__(self, dollars: float):
        self.limit = dollars
        self.tweets_read = 0
        self.unlimited = dollars == 0

    def consume(self, n: int) -> bool:
        """Record n tweet reads. Returns False if budget is now exhausted."""
        self.tweets_read += n
        return self.unlimited or self.spent < self.limit

    @property
    def spent(self) -> float:
        return self.tweets_read * COST_PER_TWEET

    @property
    def remaining_tweets(self) -> int:
        if self.unlimited:
            return 10_000
        return max(0, int((self.limit - self.spent) / COST_PER_TWEET))

    def exhausted(self) -> bool:
        return not self.unlimited and self.spent >= self.limit

    def summary(self) -> str:
        cap = "unlimited" if self.unlimited else f"${self.limit:.2f}"
        return f"{self.tweets_read} tweets read · ${self.spent:.3f} spent (budget: {cap})"


# ── Twitter search ────────────────────────────────────────────────────────────

async def search_tweets(
    client: httpx.AsyncClient,
    query: str,
    budget: Budget,
    since_date: str = SINCE_DATE,
) -> list[dict]:
    """
    Page through recent search for `query`, stopping immediately when budget runs out.
    Each page of results is charged against the budget before the next page is fetched.
    """
    url = "https://api.twitter.com/2/tweets/search/recent"
    base_params = {
        "query": f"({query}) has:images -is:retweet",
        "start_time": since_date,
        "tweet.fields": "id,text,created_at,author_id,public_metrics,entities,lang,geo,possibly_sensitive",
        "user.fields": "id,username,name,verified,public_metrics,created_at,description",
        "media.fields": "media_key,type,url,preview_image_url,width,height,alt_text,duration_ms,variants",
        "expansions": "attachments.media_keys,author_id",
    }
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}

    tweets: list[dict] = []
    next_token: str | None = None
    page = 0

    while True:
        # Cap page size to remaining budget (min 10, max 100 — API constraint)
        page_size = max(10, min(100, budget.remaining_tweets))
        params = {**base_params, "max_results": page_size}
        if next_token:
            params["next_token"] = next_token

        resp = await client.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code == 429:
            reset = int(resp.headers.get("x-rate-limit-reset", time.time() + 60))
            wait = max(reset - int(time.time()), 5)
            log.warning("Rate limited — sleeping %ds", wait)
            await asyncio.sleep(wait)
            continue
        resp.raise_for_status()

        body = resp.json()
        page += 1

        raw_tweets = body.get("data", [])
        includes = body.get("includes", {})
        media_map = {m["media_key"]: m for m in includes.get("media", [])}
        user_map = {u["id"]: u for u in includes.get("users", [])}

        for tweet in raw_tweets:
            tweet["_author"] = user_map.get(tweet.get("author_id"), {})
            media_keys = tweet.get("attachments", {}).get("media_keys", [])
            tweet["_media"] = [media_map[k] for k in media_keys if k in media_map]
            tweets.append(tweet)

        ok = budget.consume(len(raw_tweets))
        log.info("  page %d: +%d tweets | %s", page, len(raw_tweets), budget.summary())

        next_token = body.get("meta", {}).get("next_token")
        if not next_token or not ok:
            if not ok:
                log.warning("  budget exhausted — stopping pagination")
            break
        await asyncio.sleep(0.5)

    return tweets


# ── image download ────────────────────────────────────────────────────────────

def _image_url_for_media(media: dict) -> str | None:
    if media.get("type") == "photo":
        url = media.get("url") or media.get("preview_image_url")
        if url:
            return re.sub(r"[?&]name=\w+", "", url) + "?format=jpg&name=orig"
    if media.get("type") in ("video", "animated_gif"):
        return media.get("preview_image_url")
    return None


async def download_image(client: httpx.AsyncClient, url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        resp = await client.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as exc:
        log.warning("  download failed %s — %s", url, exc)
        return False


# ── metadata record ───────────────────────────────────────────────────────────

def build_record(tweet: dict, media: dict, local_path: Path | None, image_url: str) -> dict:
    author = tweet.get("_author", {})
    return {
        # tweet core
        "tweet_id": tweet["id"],
        "tweet_url": f"https://twitter.com/i/web/status/{tweet['id']}",
        "text": tweet.get("text", ""),
        "created_at": tweet.get("created_at"),
        "lang": tweet.get("lang"),
        "possibly_sensitive": tweet.get("possibly_sensitive", False),
        # engagement
        "retweet_count": tweet.get("public_metrics", {}).get("retweet_count"),
        "like_count": tweet.get("public_metrics", {}).get("like_count"),
        "reply_count": tweet.get("public_metrics", {}).get("reply_count"),
        "quote_count": tweet.get("public_metrics", {}).get("quote_count"),
        "impression_count": tweet.get("public_metrics", {}).get("impression_count"),
        # author
        "author_id": tweet.get("author_id"),
        "author_username": author.get("username"),
        "author_name": author.get("name"),
        "author_verified": author.get("verified", False),
        "author_followers": author.get("public_metrics", {}).get("followers_count"),
        "author_following": author.get("public_metrics", {}).get("following_count"),
        "author_tweet_count": author.get("public_metrics", {}).get("tweet_count"),
        "author_created_at": author.get("created_at"),
        "author_description": author.get("description"),
        # media
        "media_key": media.get("media_key"),
        "media_type": media.get("type"),
        "media_width": media.get("width"),
        "media_height": media.get("height"),
        "media_alt_text": media.get("alt_text"),
        "image_url": image_url,
        "local_path": str(local_path) if local_path else None,
        "download_success": local_path is not None and local_path.exists(),
        # scrape metadata
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "query_used": tweet.get("_query_used", ""),
    }


# ── main ──────────────────────────────────────────────────────────────────────

async def run(budget_dollars: float, output_dir: Path, since_date: str = SINCE_DATE):
    global OUTPUT_DIR, IMAGES_DIR, METADATA_FILE, LOG_FILE
    OUTPUT_DIR = output_dir
    IMAGES_DIR = output_dir / "images"
    METADATA_FILE = output_dir / "metadata.jsonl"
    LOG_FILE = output_dir / "run_log.txt"

    if not BEARER_TOKEN:
        log.error("TWITTER_BEARER_TOKEN not set — aborting")
        sys.exit(1)

    budget = Budget(budget_dollars)
    max_tweets = budget.remaining_tweets if not budget.unlimited else "unlimited"
    log.info("Budget: $%.2f → max %s tweets @ $%.3f/tweet", budget_dollars, max_tweets, COST_PER_TWEET)
    log.info("Since: %s | Queries: %d | Output: %s", since_date, len(QUERIES), output_dir)

    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)

    seen_tweet_ids: set[str] = set()
    seen_image_urls: set[str] = set()
    images_collected = 0
    run_start = datetime.now(timezone.utc)

    async with httpx.AsyncClient() as client:
        for query in QUERIES:
            if budget.exhausted():
                log.warning("Budget exhausted — skipping remaining queries")
                break

            log.info("── Query: %s (budget remaining: $%.3f)", query, budget.limit - budget.spent if not budget.unlimited else 999)
            try:
                tweets = await search_tweets(client, query, budget, since_date)
            except httpx.HTTPStatusError as exc:
                log.error("  search failed: %s — %s", exc.response.status_code, exc.response.text[:200])
                continue

            for tweet in tweets:
                if tweet["id"] in seen_tweet_ids:
                    continue
                seen_tweet_ids.add(tweet["id"])
                tweet["_query_used"] = query

                for media in tweet.get("_media", []):
                    image_url = _image_url_for_media(media)
                    if not image_url:
                        continue

                    base_url = re.sub(r"[?&]name=\w+", "", image_url)
                    if base_url in seen_image_urls:
                        continue
                    seen_image_urls.add(base_url)

                    ext = "png" if ".png" in image_url else "gif" if ".gif" in image_url else "jpg"
                    filename = f"{tweet['id']}_{media.get('media_key', 'media')}.{ext}"
                    dest = IMAGES_DIR / filename

                    log.info("  [img %d] %s → %s", images_collected + 1, image_url[:70], filename)
                    ok = await download_image(client, image_url, dest)

                    record = build_record(tweet, media, dest if ok else None, image_url)
                    with METADATA_FILE.open("a") as f:
                        f.write(json.dumps(record) + "\n")

                    images_collected += 1

            await asyncio.sleep(1)

    elapsed = (datetime.now(timezone.utc) - run_start).total_seconds()
    summary = (
        f"Run completed:    {run_start.isoformat()}\n"
        f"Elapsed:          {elapsed:.1f}s\n"
        f"API cost:         {budget.summary()}\n"
        f"Unique tweets:    {len(seen_tweet_ids)}\n"
        f"Images collected: {images_collected}\n"
        f"Metadata:         {METADATA_FILE}\n"
        f"Images dir:       {IMAGES_DIR}\n"
    )
    LOG_FILE.write_text(summary)
    log.info("\n%s", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=2.00,
                        help="Max spend in USD (default $2.00, 0=unlimited)")
    parser.add_argument("--output-dir", type=Path, default=Path("scrape_output"),
                        help="Directory for images, metadata.jsonl, run_log.txt (default: scrape_output)")
    parser.add_argument("--since-hours", type=float, default=None,
                        help="Collect tweets from the last N hours (overrides SINCE_DATE)")
    args = parser.parse_args()

    if args.since_hours is not None:
        from datetime import timedelta
        since_dt = datetime.now(timezone.utc) - timedelta(hours=args.since_hours)
        since_date = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"Using dynamic start_time: {since_date} (last {args.since_hours:.0f} hours)")
    else:
        since_date = SINCE_DATE

    asyncio.run(run(args.budget, args.output_dir, since_date))
