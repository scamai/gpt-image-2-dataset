# GPT-Image-2 Twitter Dataset — Collection Code

Code for the paper *GPT-Image-2 in the Wild: A Twitter Dataset of Self-Reported AI-Generated Images from the First Week of Deployment* (Zewde et al., 2026).

This repository contains the scripts used to collect and curate the **10,217-image GPT-Image-2 Twitter Dataset**. It does **not** contain the dataset itself — the dataset is released separately.

---

## What's in here

| Script | Role |
|---|---|
| `scrape_gpt_image2.py` | Twitter API v2 scraper. Six multilingual queries (English, Japanese, Chinese), pay-per-tweet budget cap, paginated, resumable. |
| `curate_dataset.py` | Two-stage curation: media filter (drop video/gif/failed downloads) + multilingual text-heuristic classifier (`confirmed` / `rejected` / `uncertain`). Optional Gemini vision review of uncertain records. |
| `check_made_with_ai.py` | Playwright/Chromium browser automation. Visits each uncertain tweet URL and checks whether Twitter's "Made with AI" badge is rendered. Resumable; writes one JSONL line per tweet. |
| `fetch_negatives.py` | Builds a known-non-GPT-image-2 control set (Picsum real photos + Twitter Midjourney scrape) for verifier benchmarking. |
| `verify_openai.py` | GPT-5.4-mini visual verifier — sends confirmed images to OpenAI's vision API to test whether the model can identify GPT-image-2 outputs. Used to produce the negative result reported in the paper. |
| `test_verifier.py` | Controlled accuracy test: runs `verify_openai.py` against the positive set (hashtag-confirmed) and the negative set (`fetch_negatives.py` output) and reports per-class accuracy. |

---

## Setup

```bash
# Clone and install
git clone https://github.com/scamai/gpt-image-2-dataset.git
cd gpt-image-2-dataset
pip install -r requirements.txt
playwright install chromium   # only if you'll run check_made_with_ai.py

# Configure API keys
cp .env.example .env
# Edit .env and fill in TWITTER_BEARER_TOKEN, OPENAI_API_KEY, GEMINI_API_KEY (only what you need)
```

API access tiers required:

- **Twitter API v2** — Basic (paid) tier. The recent search endpoint costs ~$0.005 per tweet read.
- **OpenAI API** — pay-as-you-go access to `gpt-5.4-mini` (or substitute another vision model).
- **Gemini API** — free tier is sufficient for the optional vision review pass.

---

## End-to-end usage

The full pipeline reproduces the paper's collection process. Each step writes JSONL output that feeds the next.

### 1. Scrape Twitter

```bash
# $2 budget (default) — caps spending mid-pagination
python scrape_gpt_image2.py --budget 2.00 --output-dir scrape_output

# Larger run, custom 24-hour window
python scrape_gpt_image2.py --budget 50.00 --since-hours 24 --output-dir scrape_24h
```

Writes `scrape_output/metadata.jsonl` (one record per image) and `scrape_output/images/{tweet_id}_{media_key}.jpg`.

### 2. Curate

```bash
# Apply text-heuristic classifier
python curate_dataset.py --scrape-dir scrape_output

# Optional: send uncertain records to Gemini vision for a second pass
python curate_dataset.py --scrape-dir scrape_output --gemini-review

# Inspect samples
python curate_dataset.py --scrape-dir scrape_output --show confirmed
python curate_dataset.py --scrape-dir scrape_output --show uncertain
```

Writes `scrape_output/curated_metadata.jsonl` with a `classification` field (`confirmed` / `rejected` / `uncertain`) and a one-line reason.

### 3. Badge check (recover silent creators)

```bash
# Extract uncertain tweets from the curated file into a JSONL with fields:
#   {"tweet_id": "...", "tweet_url": "...", "lang": "...", "text_preview": "..."}

python check_made_with_ai.py uncertain_tweets.jsonl made_with_ai_results.jsonl
```

Headed Chromium browser visits each tweet URL with a 3–5s random delay, looks for "Made with AI" or "AI-generated" text on the page, and writes one JSONL line per tweet. Resumable — re-running picks up where you left off.

### 4. Verifier benchmark (paper §5.3 negative result)

```bash
# Fetch a control set of known non-GPT-image-2 images
python fetch_negatives.py --output-dir non_gpt_image_2

# Run the controlled accuracy test
# --scrape-dir points at the scrape output directory (containing curated_metadata.jsonl);
# the script samples confirmed positives from there and negatives from --neg-dir.
python test_verifier.py \
    --scrape-dir scrape_output \
    --neg-dir non_gpt_image_2 \
    --model gpt-5.4-mini
```

Reports per-class accuracy. The paper found ~94% confirmed on positives but only ~28% rejected on negatives, demonstrating that current vision models cannot distinguish GPT-image-2 outputs from other photorealistic AI images.

---

## Citation

If you use this code or the dataset, please cite:

```bibtex
@article{zewde2026gptimage2,
  author  = {Zewde, Kidus and Ren, Simiao and Shen, Xingyu and Wu, Jenny
             and Zhou, Yuchen and Duong, Tommy and Zhang, Zikang and
             Traister, Ethan and Xie, Kewen},
  title   = {{GPT-Image-2} in the Wild: A {Twitter} Dataset of Self-Reported
             {AI}-Generated Images from the First Week of Deployment},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

Code is released under the Apache License 2.0 — see [`LICENSE`](LICENSE).

The dataset itself (released separately) is governed by its own license; see the dataset README for details.

---

## Notes and caveats

- The scrapers operate on the Twitter API v2 **recent search** endpoint, which only returns tweets from the preceding seven days. This pipeline cannot retroactively reproduce the April 2026 collection window.
- All collected images are subject to the original creators' copyright. This codebase does not redistribute any images.
- Twitter's CDN strips C2PA content credentials and EXIF metadata from uploaded images — provenance verification in this dataset is based on tweet-author self-report and the platform's "Made with AI" label, not cryptographic signatures.
