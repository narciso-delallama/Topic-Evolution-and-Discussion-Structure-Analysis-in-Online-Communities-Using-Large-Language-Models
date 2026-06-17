import json
import re
import csv
import logging
import hashlib
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

INPUT_FOLDER = "data_raw"
OUTPUT_FILE = "complete_tweets_clean_llm_ready.csv"


def clean_text(text):
    if not text:
        return None
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def normalize_for_llm(text):
    if not text:
        return None
    text = clean_text(text)
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    return text.strip()


def build_analysis_text(tweet_text, ref_text, is_repost, is_reply):
    tweet_text = clean_text(tweet_text)
    ref_text = clean_text(ref_text)

    if is_repost and ref_text:
        if tweet_text and tweet_text != ref_text:
            return f"[REPOST] Comment: {tweet_text} | Original: {ref_text}"
        return f"[REPOST] {ref_text}"

    if is_reply and ref_text:
        if tweet_text and tweet_text != ref_text:
            return f"[REPLY] Answer: {tweet_text} | Context: {ref_text}"
        return f"[REPLY] {ref_text}"

    return tweet_text


def make_text_hash(text):
    if not text:
        return None
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def flatten_tweet(record):
    ctx = record.get("referenced_tweet_context") or {}
    raw_dt = record.get("datetime")

    date = hour = year = month = year_month = iso_week = None
    if raw_dt:
        try:
            dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
            date = dt.date().isoformat()
            hour = dt.hour
            year = dt.year
            month = dt.month
            year_month = f"{dt.year}-{dt.month:02d}"
            iso = dt.isocalendar()
            iso_week = f"{iso.year}-W{iso.week:02d}"
        except Exception:
            log.warning(f"Invalid date: {raw_dt}")

    tweet_text = clean_text(record.get("tweet_text", ""))
    ref_text = clean_text(ctx.get("original_tweet_text"))

    is_reply = bool(record.get("is_reply", False))
    is_repost = bool(record.get("is_repost", False))

    analysis_text = build_analysis_text(tweet_text, ref_text, is_repost, is_reply)
    analysis_text_min = normalize_for_llm(analysis_text)

    char_count = len(analysis_text_min) if analysis_text_min else 0
    word_count = len(analysis_text_min.split()) if analysis_text_min else 0
    usable_for_llm = bool(analysis_text_min and word_count >= 3)

    return {
        "public_figure": record.get("public_figure"),
        "tweet_id": record.get("tweet_id"),
        "tweet_url": record.get("tweet_url"),
        "tweet_type": record.get("tweet_type", "original"),
        "is_reply": is_reply,
        "is_repost": is_repost,
        "ref_author": ctx.get("original_author"),
        "ref_url": ctx.get("original_tweet_url"),
        "analysis_text": analysis_text,
        "analysis_text_min": analysis_text_min,
        "text_hash": make_text_hash(analysis_text_min),
        "datetime_utc": raw_dt,
        "date": date,
        "hour": hour,
        "year": year,
        "month": month,
        "year_month": year_month,
        "iso_week": iso_week,
        "char_count": char_count,
        "word_count": word_count,
        "usable_for_llm": usable_for_llm,
        "scrape_date": record.get("scrape_date"),
    }


def load_tweets(input_folder):
    folder = Path(input_folder)
    combined = folder / "all_public_figures_tweets_complete.json"

    seen_ids = set()
    records = []

    def ingest(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for rec in data:
            tid = rec.get("tweet_id")
            if tid and tid not in seen_ids:
                seen_ids.add(tid)
                records.append(rec)

    if combined.exists():
        ingest(combined)
    else:
        for f in folder.glob("*_tweets.json"):
            ingest(f)

    log.info(f"Loaded tweets: {len(records)}")
    return records


def deduplicate(rows):
    seen = set()
    result = []

    for r in rows:
        key = r.get("text_hash")
        if not key:
            result.append(r)
            continue
        if key not in seen:
            seen.add(key)
            result.append(r)

    log.info(f"After deduplication: {len(result)} rows")
    return result


FIELDNAMES = [
    "public_figure", "tweet_id", "tweet_url",
    "tweet_type", "is_reply", "is_repost",
    "ref_author", "ref_url",
    "analysis_text", "analysis_text_min", "text_hash",
    "datetime_utc", "date", "hour", "year", "month", "year_month", "iso_week",
    "char_count", "word_count", "usable_for_llm",
    "scrape_date",
]


def write_csv(rows, output):
    with open(output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Saved CSV: {output} ({len(rows)} rows)")


def main():
    log.info("Starting preprocessing...")
    raw = load_tweets(INPUT_FOLDER)
    rows = [flatten_tweet(r) for r in raw]
    rows = deduplicate(rows)
    rows.sort(key=lambda x: (x["public_figure"] or "", x["datetime_utc"] or ""))
    write_csv(rows, OUTPUT_FILE)
    log.info("Done.")


if __name__ == "__main__":
    main()