#!/usr/bin/env python3
"""Check uncertain tweets for Twitter's 'Made with AI' label using Playwright."""

import json
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "uncertain_tweets.jsonl"
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "made_with_ai_results.jsonl"
LOAD_TIMEOUT_MS = 12_000
DELAY_MIN, DELAY_MAX = 3, 5
RATE_LIMIT_SLEEP = 60
MAX_RETRIES = 2
LOG_EVERY = 50

# Counters
stats = {"has_label": 0, "no_label": 0, "errors": 0, "login_wall": 0}
start_time = time.time()
interrupted = False


def handle_signal(sig, frame):
    global interrupted
    interrupted = True
    print("\nInterrupted — finishing current tweet and saving...")


signal.signal(signal.SIGINT, handle_signal)


def load_checked_ids(path: str) -> set:
    ids = set()
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ids.add(json.loads(line)["tweet_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return ids


def load_tweets(path: str) -> list:
    tweets = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tweets.append(json.loads(line))
    return tweets


def append_result(path: str, result: dict):
    with open(path, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def elapsed_str():
    secs = int(time.time() - start_time)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m}m"
    return f"{m}m"


def check_tweet(page, tweet: dict) -> dict:
    tweet_id = tweet["tweet_id"]
    tweet_url = tweet["tweet_url"]
    lang = tweet.get("lang", "")

    result = {
        "tweet_id": tweet_id,
        "tweet_url": tweet_url,
        "lang": lang,
        "has_made_with_ai": None,
        "label_text_found": None,
        "status": "error",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = page.goto(tweet_url, wait_until="domcontentloaded", timeout=LOAD_TIMEOUT_MS)

            # Extra wait for dynamic content
            page.wait_for_timeout(3000)

            status_code = resp.status if resp else None

            # Check for 404 / not found
            if status_code == 404:
                result["status"] = "not_found"
                result["has_made_with_ai"] = None
                return result

            page_text = page.content().lower()

            # Check for rate limit / CAPTCHA
            if status_code == 429 or "captcha" in page_text:
                if attempt < MAX_RETRIES:
                    print(f"  Rate limited on {tweet_id}, sleeping {RATE_LIMIT_SLEEP}s...")
                    time.sleep(RATE_LIMIT_SLEEP)
                    continue
                result["status"] = "rate_limited"
                return result

            # Check for login wall
            login_signals = ["log in" in page_text, "sign in to x" in page_text, "sign in to twitter" in page_text]
            if any(login_signals):
                # Some tweets show login prompt but still have content visible
                # Only mark as login_required if we can't see any tweet content
                if "made with ai" not in page_text and "ai-generated" not in page_text:
                    # Check if actual tweet content is visible
                    tweet_preview = tweet.get("text_preview", "")[:30].lower()
                    if tweet_preview and tweet_preview not in page_text:
                        result["status"] = "login_required"
                        result["has_made_with_ai"] = None
                        return result

            # Check for "Made with AI" or "AI-generated"
            if "made with ai" in page_text:
                result["has_made_with_ai"] = True
                result["label_text_found"] = "Made with AI"
                result["status"] = "ok"
                return result
            elif "ai-generated" in page_text:
                result["has_made_with_ai"] = True
                result["label_text_found"] = "AI-generated"
                result["status"] = "ok"
                return result
            else:
                result["has_made_with_ai"] = False
                result["label_text_found"] = None
                result["status"] = "ok"
                return result

        except PlaywrightTimeout:
            if attempt < MAX_RETRIES:
                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
                continue
            result["status"] = "timeout"
            return result
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
                continue
            result["status"] = "error"
            result["label_text_found"] = str(e)[:200]
            return result

    return result


def print_summary():
    total = stats["has_label"] + stats["no_label"] + stats["errors"] + stats["login_wall"]
    if total == 0:
        print("No tweets checked.")
        return
    print("\n=== Results ===")
    print(f"Total checked:       {total}")
    print(f'Has "Made with AI":  {stats["has_label"]} ({100*stats["has_label"]/total:.1f}%)')
    print(f"No label:            {stats['no_label']} ({100*stats['no_label']/total:.1f}%)")
    print(f"Errors/timeouts:     {stats['errors']} ({100*stats['errors']/total:.1f}%)")
    print(f"Login walls:         {stats['login_wall']} ({100*stats['login_wall']/total:.1f}%)")
    print(f"Output: {OUTPUT_FILE}")


def main():
    base_dir = Path(__file__).parent
    input_path = base_dir / INPUT_FILE
    output_path = base_dir / OUTPUT_FILE

    tweets = load_tweets(str(input_path))
    print(f"Loaded {len(tweets)} tweets from {INPUT_FILE}")

    checked_ids = load_checked_ids(str(output_path))
    if checked_ids:
        print(f"Already checked {len(checked_ids)} tweets — resuming")

    remaining = [t for t in tweets if t["tweet_id"] not in checked_ids]
    print(f"Remaining: {len(remaining)} tweets to check\n")

    if not remaining:
        print("All tweets already checked!")
        print_summary()
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()

        checked_count = len(checked_ids)
        total = len(tweets)

        try:
            for i, tweet in enumerate(remaining):
                if interrupted:
                    break

                result = check_tweet(page, tweet)
                append_result(str(output_path), result)

                # Update stats
                if result["has_made_with_ai"] is True:
                    stats["has_label"] += 1
                elif result["has_made_with_ai"] is False:
                    stats["no_label"] += 1
                elif result["status"] == "login_required":
                    stats["login_wall"] += 1
                else:
                    stats["errors"] += 1

                checked_count += 1

                # Progress log
                if checked_count % LOG_EVERY == 0 or i == len(remaining) - 1:
                    print(
                        f"[{checked_count}/{total}] "
                        f"has_label={stats['has_label']} ({100*stats['has_label']/max(1, sum(stats.values())):.1f}%) | "
                        f"no_label={stats['no_label']} | "
                        f"errors={stats['errors']} | "
                        f"login={stats['login_wall']} | "
                        f"elapsed={elapsed_str()}"
                    )

                # Random delay between requests
                if i < len(remaining) - 1 and not interrupted:
                    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        finally:
            browser.close()

    print_summary()


if __name__ == "__main__":
    main()
