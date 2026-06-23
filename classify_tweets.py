import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY = os.getenv("GEMINI_API_KEY2")
if not API_KEY:
    raise EnvironmentError("GEMINI_API_KEY2 was not found in the environment variables.")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.1-flash-lite-preview"

BATCH_SIZE = 25          # 12k tweets / 25 = 480 requests < 500 RPD limit
MAX_WORKERS = 1
RPM_LIMIT = 13           # safe margin under the 15 RPM quota
RPD_LIMIT = 490          # safe margin under the 500 RPD quota
CHECKPOINT_FILE = "classify_checkpoint_v2.csv"


# ---------------------------------------------------------------------------
# Rate limiter  (RPM)
# ---------------------------------------------------------------------------

_rate_lock = threading.Lock()
_request_times: list[float] = []


def _wait_for_rate_limit():
    """Block until sending another request will not exceed RPM_LIMIT.
    Releases the lock before sleeping to avoid blocking other threads."""
    while True:
        with _rate_lock:
            now = time.monotonic()
            window = 60.0

            while _request_times and now - _request_times[0] >= window:
                _request_times.pop(0)

            if len(_request_times) < RPM_LIMIT:
                _request_times.append(time.monotonic())
                return

            sleep_for = window - (now - _request_times[0]) + 0.1

        # sleep outside the lock so other threads are not blocked
        time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Rate limiter  (RPD)
# ---------------------------------------------------------------------------

_daily_lock = threading.Lock()
_daily_requests = 0


def _increment_daily_request():
    global _daily_requests
    with _daily_lock:
        if _daily_requests >= RPD_LIMIT:
            raise RuntimeError(
                f"Daily request limit ({RPD_LIMIT}) reached. "
                "Resume tomorrow or increase BATCH_SIZE."
            )
        _daily_requests += 1


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TweetClassification(BaseModel):
    index: int = Field(description="Tweet index as given in the prompt")

    topic: Literal[
        "Economy and Employment",
        "Welfare, Housing and Social Policy",
        "National Politics and Governance",
        "International Affairs",
        "Immigration and Security",
        "Rights and Equality",
        "Other",
    ] = Field(description="Main topic of the tweet")

    stance: Literal[
        "In favor",
        "Against",
        "Neutral",
        "Unclear",
    ] = Field(description="Detected stance in the tweet")

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence score between 0 and 1",
    )

    short_justification: str = Field(
        description="Very short explanation of the decision, max 25 words"
    )


class BatchClassification(BaseModel):
    results: list[TweetClassification]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_batch_prompt(tweets: list[str]) -> str:
    numbered_tweets = "\n".join(
        f"[{i}] {tweet}" for i, tweet in enumerate(tweets)
    )

    return f"""
You are an expert in political discourse analysis for social media.

Your task is to classify each tweet independently.

----------------------------------------
TASK
----------------------------------------

For EACH tweet, assign:

1. ONE topic from this list:
- Economy and Employment
- Welfare, Housing and Social Policy
- National Politics and Governance
- International Affairs
- Immigration and Security
- Rights and Equality
- Other

2. ONE stance:
- In favor
- Against
- Neutral
- Unclear

3. A confidence score between 0 and 1
- Confidence is an approximate measure of certainty, not an exact probability.

4. A short justification, maximum 25 words

----------------------------------------
IMPORTANT RULES
----------------------------------------

- Classify each tweet independently.
- Do NOT use context outside the tweet.
- Do NOT infer intent that is not explicitly expressed.
- If the tweet is descriptive, informational, ironic, or ambiguous, use "Unclear".
- If no clear opinion is expressed, use "Unclear".
- Only assign "In favor" or "Against" when the stance is clearly explicit.
- Use "Neutral" only when the tweet clearly presents a balanced or non-positioned statement.
- Follow the instructions strictly. Do not deviate from the schema.

----------------------------------------
CONFIDENCE CALIBRATION
----------------------------------------

Use the following scale:

- 0.95: explicit topic and explicit stance, no ambiguity
- 0.85: clear classification but requires minor interpretation
- 0.70: somewhat ambiguous or indirect
- 0.60: very ambiguous, vague, ironic, incomplete, or unclear stance

Avoid always using high values.
Use lower scores when uncertain.

----------------------------------------
EXAMPLES
----------------------------------------

Tweet: "The government must reduce taxes immediately."
Topic: Economy and Employment
Stance: In favor
Confidence: 0.95
Justification: Explicit support for tax reduction.

Tweet: "Another debate in parliament today."
Topic: National Politics and Governance
Stance: Unclear
Confidence: 0.60
Justification: Descriptive, no opinion expressed.

Tweet: "Immigration policies are destroying our country."
Topic: Immigration and Security
Stance: Against
Confidence: 0.95
Justification: Strong negative opinion on immigration.

Tweet: "Housing remains one of the biggest issues in Spain."
Topic: Welfare, Housing and Social Policy
Stance: Unclear
Confidence: 0.70
Justification: Mentions issue without clear stance.

----------------------------------------
OUTPUT FORMAT
----------------------------------------

Return ONLY valid JSON using this schema:

{{
  "results": [
    {{
      "index": int,
      "topic": str,
      "stance": str,
      "confidence": float,
      "short_justification": str
    }}
  ]
}}

The number of results MUST match the number of tweets.
The "index" must match the tweet number.

----------------------------------------
TWEETS
----------------------------------------

{numbered_tweets}
""".strip()


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------

def make_error_result(message: str) -> dict:
    return {
        "topic": "ERROR",
        "stance": "ERROR",
        "confidence": 0.0,
        "short_justification": message[:120],
    }


def classify_batch(tweets: list[str], max_retries: int = 5) -> list[dict]:
    prompt = build_batch_prompt(tweets)
    _increment_daily_request()  # count once per batch, not per retry

    for attempt in range(1, max_retries + 1):
        _wait_for_rate_limit()

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=BatchClassification,
                ),
            )

            raw_text = response.text
            if not raw_text:
                raise ValueError("Empty response received from Gemini.")

            parsed = json.loads(raw_text)
            validated = BatchClassification.model_validate(parsed)

            result_map = {item.index: item for item in validated.results}

            output = []
            for i in range(len(tweets)):
                if i in result_map:
                    item = result_map[i]
                    output.append({
                        "topic": item.topic,
                        "stance": item.stance,
                        "confidence": item.confidence,
                        "short_justification": item.short_justification,
                    })
                else:
                    output.append(make_error_result(f"Missing index {i} in model response."))

            return output

        except RuntimeError:
            # daily limit reached — propagate immediately, do not retry
            raise

        except Exception as e:
            error_msg = str(e)

            is_daily_limit = "RESOURCE_EXHAUSTED" in error_msg and (
                "quota" in error_msg.lower() or "day" in error_msg.lower()
            )
            is_transient = any(
                code in error_msg
                for code in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")
            )

            if is_daily_limit:
                raise RuntimeError(
                    "Daily quota (RPD) exhausted by previous runs. "
                    "Resume tomorrow — checkpoint is saved."
                )

            if is_transient:
                wait_time = min(30 * attempt, 120)
                print(
                    f"  [API error {attempt}/{max_retries}] {error_msg[:120]} "
                    f"— waiting {wait_time}s..."
                )
                time.sleep(wait_time)

            elif attempt == max_retries:
                print(f"  [error] batch failed permanently: {error_msg}")
                raise RuntimeError(error_msg)

            else:
                time.sleep(5)

    return [make_error_result("Unknown batch classification error.") for _ in tweets]


# ---------------------------------------------------------------------------
# Dataframe classification with checkpoint
# ---------------------------------------------------------------------------

def classify_dataframe(
    df: pd.DataFrame,
    text_column: str = "analysis_text",
    date_column: Optional[str] = "date",
    politician_column: Optional[str] = "public_figure",
    only_usable: bool = True,
    batch_size: int = BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
    checkpoint_file: str = CHECKPOINT_FILE,
) -> pd.DataFrame:

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe.")

    if only_usable and "usable_for_llm" in df.columns:
        df = df[df["usable_for_llm"] == True].copy()
        print(f"Filtered to {len(df)} usable tweets.")

    checkpoint_rows = []
    already_done: set[str] = set()

    if os.path.exists(checkpoint_file):
        df_ckpt = pd.read_csv(checkpoint_file)
        df_ckpt["row_id"] = df_ckpt["row_id"].astype(int)
        checkpoint_rows = df_ckpt.to_dict("records")
        already_done = set(df_ckpt["tweet_id"].astype(str))
        print(f"Resuming: {len(already_done)} tweets already classified.")

    pending = df[~df["tweet_id"].astype(str).isin(already_done)].copy()
    pending = pending.reset_index().rename(columns={"index": "original_row_id"})

    print(f"Tweets to classify: {len(pending)}")

    if pending.empty:
        print("Nothing to do. All tweets already classified.")
        return pd.DataFrame(checkpoint_rows)

    batches = [
        pending.iloc[i:i + batch_size].copy()
        for i in range(0, len(pending), batch_size)
    ]

    print(f"Total batches: {len(batches)} ({batch_size} tweets each)")
    print(f"Estimated requests needed: {len(batches)} / {RPD_LIMIT} daily limit")

    checkpoint_lock = threading.Lock()
    completed_tweets = 0
    total_tweets = len(pending)

    def process_batch(batch_df: pd.DataFrame) -> list[dict]:
        nonlocal completed_tweets

        tweets = [str(row[text_column]).strip() for _, row in batch_df.iterrows()]
        results = classify_batch(tweets)

        output_rows = []
        for (_, row), result in zip(batch_df.iterrows(), results):
            output_row = {
                "row_id": int(row["original_row_id"]),
                "tweet_id": row.get("tweet_id"),
                "tweet": str(row[text_column]).strip(),
                "topic": result["topic"],
                "stance": result["stance"],
                "confidence": result["confidence"],
                "short_justification": result["short_justification"],
            }

            if date_column and date_column in batch_df.columns:
                output_row["date"] = row[date_column]

            if politician_column and politician_column in batch_df.columns:
                output_row["politician"] = row[politician_column]

            output_rows.append(output_row)

        with checkpoint_lock:
            completed_tweets += len(output_rows)

            print(
                f"  [{completed_tweets}/{total_tweets}] "
                f"batch of {len(output_rows)} done | "
                f"last tweet_id={output_rows[-1]['tweet_id']} → "
                f"{output_rows[-1]['topic']} / {output_rows[-1]['stance']}"
            )

            checkpoint_exists = os.path.exists(checkpoint_file)
            pd.DataFrame(output_rows).to_csv(
                checkpoint_file,
                mode="a",
                header=not checkpoint_exists,
                index=False,
                encoding="utf-8",
            )

        return output_rows

    new_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_batch, batch): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            try:
                new_results.extend(future.result())
            except RuntimeError as e:
                if "Daily request limit" in str(e):
                    print(f"\n  [STOP] {e}")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                print(f"  [error] unexpected exception in batch: {e}")
            except Exception as e:
                print(f"  [error] unexpected exception in batch: {e}")

    all_results = checkpoint_rows + new_results
    df_results = pd.DataFrame(all_results)
    df_results["row_id"] = df_results["row_id"].astype(int)
    df_results = df_results.sort_values("row_id").reset_index(drop=True)

    return df_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    INPUT_CSV = "complete_tweets_clean_llm_ready.csv"
    OUTPUT_CSV = "classified_all_tweets_final_v2.csv"

    print(f"Loading {INPUT_CSV}...")
    df_input = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df_input)} rows total.")

    usable_rows = (
        df_input[df_input["usable_for_llm"] == True]
        if "usable_for_llm" in df_input.columns
        else df_input
    )

    n_requests = len(usable_rows) / BATCH_SIZE
    estimated_minutes = n_requests / RPM_LIMIT
    print(
        f"Tweets to classify: {len(usable_rows)} → "
        f"~{n_requests:.0f} requests "
        f"(limit: {RPD_LIMIT}/day, {RPM_LIMIT} RPM)"
    )
    print(f"Estimated minimum time: {estimated_minutes:.1f} minutes")

    if n_requests > RPD_LIMIT:
        print(
            f"WARNING: {n_requests:.0f} requests needed but daily limit is {RPD_LIMIT}. "
            "The run will stop mid-way and resume tomorrow via checkpoint."
        )

    t0 = time.time()

    df_output = classify_dataframe(
        df_input,
        text_column="analysis_text", 
        date_column="date",
        politician_column="public_figure",
        only_usable=True,
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS,
        checkpoint_file=CHECKPOINT_FILE,
    )

    elapsed = time.time() - t0

    df_output = df_output.sort_values("row_id").reset_index(drop=True)
    df_output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nDone in {elapsed:.0f}s.")
    print(f"Saved {len(df_output)} tweets to {OUTPUT_CSV}")

    
