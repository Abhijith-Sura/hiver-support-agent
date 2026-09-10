"""
Preprocess raw Twitter data into structured SpotifyCares conversations.

Pipeline:
  1. Load twcs.csv
  2. Filter to SpotifyCares brand
  3. Reconstruct multi-turn conversation threads
  4. Clean text
  5. Output structured JSONL
"""
import json
import logging
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean a tweet: strip leading @mentions, normalise whitespace, keep emojis."""
    if not isinstance(text, str):
        return ""
    # Remove leading @mentions (replies start with @BrandName)
    text = re.sub(r"^(@\w+\s*)+", "", text).strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def load_raw_data(csv_path: Path) -> pd.DataFrame:
    """Load and lightly validate the raw CSV."""
    logger.info("Loading raw data from %s …", csv_path)
    df = pd.read_csv(csv_path)
    expected_cols = {"tweet_id", "author_id", "inbound", "created_at", "text",
                     "response_tweet_id", "in_response_to_tweet_id"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    logger.info("Loaded %d tweets, %d unique authors.", len(df), df["author_id"].nunique())
    return df


def filter_brand(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Keep only tweets involving the target brand."""
    # Brand outbound tweets
    brand_tweets = df[df["author_id"] == brand].copy()
    brand_tweet_ids = set(brand_tweets["tweet_id"].astype(str))

    # Inbound tweets that the brand replied to
    brand_replies_to = set(
        brand_tweets["in_response_to_tweet_id"].dropna().astype(int).astype(str)
    )
    inbound_tweets = df[df["tweet_id"].astype(str).isin(brand_replies_to)].copy()

    # Also get inbound tweets whose response_tweet_id contains a brand tweet
    # Use vectorized string matching instead of row-by-row loop
    resp_col = df["response_tweet_id"].dropna().astype(str)
    # Most response_tweet_ids are single values; check if any map to brand tweets
    response_matches = resp_col.apply(
        lambda x: any(tid.strip() in brand_tweet_ids for tid in x.split(","))
    )
    inbound_with_brand_response = df.loc[response_matches[response_matches].index].copy()

    all_relevant = pd.concat([brand_tweets, inbound_tweets, inbound_with_brand_response]).drop_duplicates(
        subset=["tweet_id"]
    )
    logger.info("Filtered to %d tweets involving %s.", len(all_relevant), brand)
    return all_relevant


def build_conversation_pairs(df: pd.DataFrame, brand: str, max_conversations: int) -> list[dict]:
    """
    Build conversation pairs: (customer_message, brand_reply, thread_context).

    We find all brand outbound tweets that reply to an inbound tweet,
    then walk backwards to build thread context.
    """
    # Index tweets by tweet_id for fast lookup
    tweet_map = {}
    for _, row in df.iterrows():
        tweet_map[str(row["tweet_id"])] = row.to_dict()

    # Find brand replies to customer messages
    brand_outbound = df[(df["author_id"] == brand) & (df["inbound"] == False)]

    conversations = []
    seen_ids = set()

    for _, brand_row in tqdm(brand_outbound.iterrows(), total=len(brand_outbound),
                              desc="Building conversations"):
        reply_to_id = brand_row.get("in_response_to_tweet_id")
        if pd.isna(reply_to_id):
            continue

        reply_to_id = str(int(float(reply_to_id)))
        if reply_to_id in seen_ids:
            continue
        seen_ids.add(reply_to_id)

        # Get the customer message this brand tweet replies to
        customer_tweet = tweet_map.get(reply_to_id)
        if customer_tweet is None:
            continue

        # Only include if the customer tweet is inbound
        if customer_tweet.get("inbound") != True:
            continue

        customer_text = clean_text(str(customer_tweet.get("text", "")))
        brand_text = clean_text(str(brand_row.get("text", "")))

        if not customer_text or not brand_text:
            continue

        # Build thread context by walking backwards
        thread_context = []
        current_id = customer_tweet.get("in_response_to_tweet_id")
        depth = 0
        while current_id and not pd.isna(current_id) and depth < 5:
            current_id = str(int(float(current_id)))
            prev_tweet = tweet_map.get(current_id)
            if prev_tweet is None:
                break
            prev_text = clean_text(str(prev_tweet.get("text", "")))
            if prev_text:
                prefix = "[brand]" if prev_tweet.get("author_id") == brand else "[customer]"
                thread_context.insert(0, f"{prefix} {prev_text}")
            current_id = prev_tweet.get("in_response_to_tweet_id")
            depth += 1

        conversations.append({
            "conversation_id": f"conv_{len(conversations):05d}",
            "tweet_id": str(customer_tweet["tweet_id"]),
            "response_tweet_id": str(brand_row["tweet_id"]),
            "customer_author_id": str(customer_tweet["author_id"]),
            "customer_message": customer_text,
            "brand_reply": brand_text,
            "thread_context": thread_context,
        })

        if len(conversations) >= max_conversations:
            break

    logger.info("Built %d conversation pairs.", len(conversations))
    return conversations


def preprocess(csv_path: Path | None = None, output_path: Path | None = None) -> Path:
    """Run the full preprocessing pipeline."""
    from src.config import RAW_CSV_PATH, PROCESSED_CONVERSATIONS_PATH, BRAND_AUTHOR_ID, MAX_CONVERSATIONS

    csv_path = csv_path or RAW_CSV_PATH
    output_path = output_path or PROCESSED_CONVERSATIONS_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw_data(csv_path)
    brand_df = filter_brand(df, BRAND_AUTHOR_ID)
    conversations = build_conversation_pairs(brand_df, BRAND_AUTHOR_ID, MAX_CONVERSATIONS)

    # Save as JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    logger.info("Saved %d conversations to %s", len(conversations), output_path)
    return output_path


if __name__ == "__main__":
    preprocess()
