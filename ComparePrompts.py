import os
import json
import time
import random
from typing import Literal

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from sklearn.metrics import cohen_kappa_score


load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise EnvironmentError("GEMINI_API_KEY was not found.")

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-3.1-flash-lite-preview"
TEMPERATURE = 0.1

INPUT_CSV = "complete_tweets_clean_llm_ready.csv"
OUTPUT_CSV = "prompt_ab_test_results.csv"

TEXT_COLUMN = "analysis_text"
ID_COLUMN = "tweet_id"
ACTOR_COLUMN = "public_figure"

SAMPLE_SIZE = 500
BATCH_SIZE = 25
SLEEP_SECONDS = 5

PROMPT_V1 = "v1_original"
PROMPT_V2 = "v2_stricter_unclear"


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
    ]

    stance: Literal[
        "In favor",
        "Against",
        "Neutral",
        "Unclear",
    ]

    confidence: float = Field(ge=0.0, le=1.0)

    short_justification: str


class BatchClassification(BaseModel):
    results: list[TweetClassification]


def build_prompt_v1(tweets: list[str]) -> str:
    numbered_tweets = "\n".join(
        f"[{i}] {tweet}" for i, tweet in enumerate(tweets)
    )

    return f"""
You are an expert in political discourse analysis for social media.

Task: Classify each of the following {len(tweets)} tweets independently.

For EACH tweet, assign:

1. Exactly ONE topic from this closed list:
- Economy and Employment (taxes, inflation, jobs, economic policy)
- Welfare, Housing and Social Policy (healthcare, education, housing, social benefits)
- National Politics and Governance (government, laws, political actors, institutions)
- International Affairs (foreign policy, EU, global conflicts, diplomacy)
- Immigration and Security (immigration, borders, crime, public security)
- Rights and Equality (gender equality, civil rights, minority rights)
- Other (if none of the above apply)

2. Exactly ONE stance from this closed list:
- In favor
- Against
- Neutral
- Unclear

3. A confidence score between 0 and 1.
Confidence calibration rules:
- Use 0.95 only when the topic and stance are completely explicit.
- Use 0.80-0.90 when the classification is clear but requires minor interpretation.
- Use 0.60-0.75 when the tweet is ambiguous or context-dependent.
- Use below 0.60 when the tweet is short, vague, ironic, or incomplete.
- Avoid using 1.0 except for extremely obvious cases.

4. A very short justification, maximum 25 words.

Important rules:
- Classify each tweet independently. Do not let one tweet influence another.
- Focus only on the content of each tweet.
- Do not invent context that is not present in the tweet.
- Use the descriptions only as guidance.
- Return the "index" field matching the tweet number shown in brackets.
- The "results" array must contain exactly the same number of items as input tweets.
- In the "topic" field, return only the category name, without parentheses or extra text.
- If the topic is not clearly one of the predefined categories, use Other.
- If the stance cannot be inferred clearly, use Unclear.
- Return only valid JSON, no extra text.

Tweets:
{numbered_tweets}
""".strip()


def build_prompt_v2(tweets: list[str]) -> str:
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


def classify_batch(tweets: list[str], prompt_version: str, max_retries: int = 4) -> list[dict]:
    if prompt_version == PROMPT_V1:
        prompt = build_prompt_v1(tweets)
    elif prompt_version == PROMPT_V2:
        prompt = build_prompt_v2(tweets)
    else:
        raise ValueError(f"Unknown prompt version: {prompt_version}")

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    response_mime_type="application/json",
                    response_schema=BatchClassification,
                ),
            )

            if not response.text:
                raise ValueError("Empty response from Gemini.")

            parsed = json.loads(response.text)
            validated = BatchClassification.model_validate(parsed)

            result_map = {item.index: item for item in validated.results}

            output = []
            for i in range(len(tweets)):
                if i not in result_map:
                    output.append({
                        "topic": "ERROR",
                        "stance": "ERROR",
                        "confidence": 0.0,
                        "short_justification": f"Missing index {i}",
                    })
                else:
                    item = result_map[i]
                    output.append({
                        "topic": item.topic,
                        "stance": item.stance,
                        "confidence": item.confidence,
                        "short_justification": item.short_justification,
                    })

            return output

        except Exception as e:
            wait = min(15 * attempt, 60)
            print(f"[{prompt_version}] Error attempt {attempt}/{max_retries}: {str(e)[:160]}")
            if attempt == max_retries:
                raise
            time.sleep(wait)

    raise RuntimeError("Unexpected classification failure.")


def build_sample(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "usable_for_llm" in df.columns:
        df = df[df["usable_for_llm"] == True].copy()

    df = df.dropna(subset=[TEXT_COLUMN]).copy()

    if ACTOR_COLUMN in df.columns:
        sample = (
            df.groupby(ACTOR_COLUMN, group_keys=False)
            .apply(lambda x: x.sample(
                n=min(len(x), max(1, SAMPLE_SIZE // df[ACTOR_COLUMN].nunique())),
                random_state=42
            ))
        )

        if len(sample) < SAMPLE_SIZE:
            remaining = df.drop(sample.index, errors="ignore")
            extra = remaining.sample(
                n=min(SAMPLE_SIZE - len(sample), len(remaining)),
                random_state=42
            )
            sample = pd.concat([sample, extra])

        sample = sample.sample(frac=1, random_state=42).head(SAMPLE_SIZE)

    else:
        sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

    return sample.reset_index(drop=True)


def run_prompt_on_sample(sample: pd.DataFrame, prompt_version: str) -> pd.DataFrame:
    checkpoint_file = f"checkpoint_{prompt_version}.csv"

    rows = []

    already_done = set()

    if os.path.exists(checkpoint_file):
        df_ckpt = pd.read_csv(checkpoint_file)
        rows = df_ckpt.to_dict("records")
        already_done = set(df_ckpt["tweet_id"].astype(str))
        print(f"[{prompt_version}] Resuming from checkpoint: {len(already_done)} done")

    for start in range(0, len(sample), BATCH_SIZE):
        batch = sample.iloc[start:start + BATCH_SIZE].copy()

        batch = batch[~batch["tweet_id"].astype(str).isin(already_done)]

        if batch.empty:
            continue

        tweets = batch[TEXT_COLUMN].astype(str).tolist()

        print(f"Running {prompt_version}: {start + len(batch)}/{len(sample)}")

        results = classify_batch(tweets, prompt_version=prompt_version)

        batch_rows = []

        for (_, row), result in zip(batch.iterrows(), results):
            output_row = {
                "tweet_id": row.get(ID_COLUMN),
                "public_figure": row.get(ACTOR_COLUMN),
                "date": row.get("date"),
                "tweet": row[TEXT_COLUMN],
                "prompt_version": prompt_version,
                "model_name": MODEL_NAME,
                "temperature": TEMPERATURE,
                "topic": result["topic"],
                "stance": result["stance"],
                "confidence": result["confidence"],
                "short_justification": result["short_justification"],
            }

            batch_rows.append(output_row)

        rows.extend(batch_rows)

        pd.DataFrame(rows).to_csv(
            checkpoint_file,
            index=False,
            encoding="utf-8-sig"
        )

        time.sleep(SLEEP_SECONDS)

    return pd.DataFrame(rows)


def print_distribution(title: str, series: pd.Series):
    print(f"\n{title}")
    counts = series.value_counts(dropna=False)
    perc = series.value_counts(normalize=True, dropna=False) * 100
    summary = pd.DataFrame({"count": counts, "percent": perc.round(2)})
    print(summary)


def analyze_results(df_results: pd.DataFrame):
    print("\n" + "=" * 80)
    print("A/B TEST SUMMARY")
    print("=" * 80)

    for version in [PROMPT_V1, PROMPT_V2]:
        subset = df_results[df_results["prompt_version"] == version]

        print(f"\n--- {version} ---")
        print(f"Rows: {len(subset)}")
        print(f"Mean confidence: {subset['confidence'].mean():.3f}")
        print(f"Std confidence: {subset['confidence'].std():.3f}")
        print(f"Min confidence: {subset['confidence'].min():.3f}")
        print(f"Max confidence: {subset['confidence'].max():.3f}")

        print_distribution("Topic distribution", subset["topic"])
        print_distribution("Stance distribution", subset["stance"])

    wide = df_results.pivot_table(
        index="tweet_id",
        columns="prompt_version",
        values=["topic", "stance", "confidence"],
        aggfunc="first"
    )

    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.dropna().reset_index()

    topic_agreement = (wide[f"topic_{PROMPT_V1}"] == wide[f"topic_{PROMPT_V2}"]).mean()
    stance_agreement = (wide[f"stance_{PROMPT_V1}"] == wide[f"stance_{PROMPT_V2}"]).mean()

    print("\n--- Agreement between prompt versions ---")
    print(f"Topic agreement: {topic_agreement:.3f}")
    print(f"Stance agreement: {stance_agreement:.3f}")

    try:
        topic_kappa = cohen_kappa_score(wide[f"topic_{PROMPT_V1}"], wide[f"topic_{PROMPT_V2}"])
        stance_kappa = cohen_kappa_score(wide[f"stance_{PROMPT_V1}"], wide[f"stance_{PROMPT_V2}"])
        print(f"Topic Cohen's kappa: {topic_kappa:.3f}")
        print(f"Stance Cohen's kappa: {stance_kappa:.3f}")
    except Exception as e:
        print(f"Could not compute kappa: {e}")

    changed_stance = wide[wide[f"stance_{PROMPT_V1}"] != wide[f"stance_{PROMPT_V2}"]]
    print(f"\nTweets with changed stance: {len(changed_stance)}")

    unclear_v1 = (wide[f"stance_{PROMPT_V1}"] == "Unclear").mean() * 100
    unclear_v2 = (wide[f"stance_{PROMPT_V2}"] == "Unclear").mean() * 100
    print(f"Unclear rate v1: {unclear_v1:.2f}%")
    print(f"Unclear rate v2: {unclear_v2:.2f}%")

    avg_conf_v1 = wide[f"confidence_{PROMPT_V1}"].mean()
    avg_conf_v2 = wide[f"confidence_{PROMPT_V2}"].mean()
    print(f"Average confidence v1: {avg_conf_v1:.3f}")
    print(f"Average confidence v2: {avg_conf_v2:.3f}")

    changed_stance.to_csv("prompt_ab_test_changed_stance.csv", index=False, encoding="utf-8-sig")
    print("\nSaved changed stance cases to: prompt_ab_test_changed_stance.csv")


def main():
    random.seed(42)

    print(f"Loading {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    print("Building sample...")
    sample = build_sample(df)
    sample.to_csv("prompt_ab_test_sample.csv", index=False, encoding="utf-8-sig")
    print(f"Sample saved: prompt_ab_test_sample.csv ({len(sample)} tweets)")

    results_v1 = run_prompt_on_sample(sample, PROMPT_V1)
    results_v2 = run_prompt_on_sample(sample, PROMPT_V2)

    df_results = pd.concat([results_v1, results_v2], ignore_index=True)
    df_results.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nSaved full A/B results to: {OUTPUT_CSV}")

    analyze_results(df_results)


if __name__ == "__main__":
    main()