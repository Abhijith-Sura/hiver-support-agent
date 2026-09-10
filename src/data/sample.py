"""
Stratified sampling for the golden evaluation set.
"""
import json
import logging
import random
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Simple keyword heuristic for rough pre-classification (sampling only, NOT the real classifier)
KEYWORD_BUCKETS = {
    "account_access": ["login", "log in", "password", "locked", "can't access", "cant access",
                        "sign in", "signin", "account", "hacked", "reset"],
    "playback_issue": ["play", "playing", "buffer", "skip", "pause", "audio", "sound",
                        "song won't", "not playing", "stops", "stuttering", "quality"],
    "billing": ["charge", "charged", "bill", "payment", "refund", "money", "subscription",
                 "invoice", "price", "pay", "receipt"],
    "premium_features": ["premium", "upgrade", "free trial", "student", "duo", "features",
                          "ad-free", "ads", "offline", "download limit"],
    "playlist_library": ["playlist", "songs", "library", "saved", "liked", "missing songs",
                          "disappeared", "collection", "tracks"],
    "device_sync": ["device", "phone", "desktop", "laptop", "tablet", "connect", "sync",
                     "bluetooth", "speaker", "chromecast", "alexa", "car"],
    "family_plan": ["family", "invite", "member", "address", "household", "family plan",
                     "family premium"],
    "bug_report": ["crash", "bug", "error", "glitch", "freeze", "broken", "update",
                    "not working", "issue", "problem"],
    "feedback_request": ["feature", "wish", "suggest", "would be nice", "please add",
                          "feedback", "request", "idea", "improve"],
}


def rough_classify(text: str) -> str:
    """Keyword-based rough classification for sampling purposes."""
    text_lower = text.lower()
    scores = {}
    for intent, keywords in KEYWORD_BUCKETS.items():
        scores[intent] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def load_conversations(path: Path) -> list[dict]:
    """Load processed conversations from JSONL."""
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    return conversations


def create_golden_set(
    conversations: list[dict],
    total_size: int = 200,
    random_proportion: int = 120,
    rare_coverage: int = 40,
    hard_examples: int = 40,
    seed: int = 42,
) -> list[dict]:
    """Create a stratified golden set sample."""
    random.seed(seed)

    # Rough-classify all conversations
    for conv in conversations:
        conv["rough_intent"] = rough_classify(conv["customer_message"])

    intent_counts = Counter(c["rough_intent"] for c in conversations)
    logger.info("Rough intent distribution: %s", dict(intent_counts))

    # Group by rough intent
    by_intent = {}
    for conv in conversations:
        by_intent.setdefault(conv["rough_intent"], []).append(conv)

    selected_ids = set()
    golden = []

    def add_example(conv: dict) -> bool:
        if conv["conversation_id"] in selected_ids:
            return False
        selected_ids.add(conv["conversation_id"])
        golden.append(conv)
        return True

    # 1. Random proportional sample
    pool = list(conversations)
    random.shuffle(pool)
    count = 0
    for conv in pool:
        if count >= random_proportion:
            break
        if add_example(conv):
            count += 1

    # 2. Rare intent coverage — ensure ≥10 per intent
    all_intents = list(KEYWORD_BUCKETS.keys()) + ["other"]
    for intent in all_intents:
        current_count = sum(1 for g in golden if g["rough_intent"] == intent)
        needed = max(0, 10 - current_count)
        candidates = [c for c in by_intent.get(intent, []) if c["conversation_id"] not in selected_ids]
        random.shuffle(candidates)
        for conv in candidates[:needed]:
            if len(golden) >= total_size - hard_examples:
                break
            add_example(conv)

    # 3. Hard examples
    hard_candidates = []
    for conv in conversations:
        if conv["conversation_id"] in selected_ids:
            continue
        msg = conv["customer_message"]
        word_count = len(msg.split())
        is_hard = (
            word_count < 5
            or word_count > 50
            or len(conv.get("thread_context", [])) >= 2
            or sum(1 for intent, kws in KEYWORD_BUCKETS.items()
                   if any(kw in msg.lower() for kw in kws)) >= 2  # ambiguous
        )
        if is_hard:
            hard_candidates.append(conv)

    random.shuffle(hard_candidates)
    remaining = total_size - len(golden)
    for conv in hard_candidates[:remaining]:
        add_example(conv)

    # Fill any remaining slots randomly
    if len(golden) < total_size:
        remaining_pool = [c for c in conversations if c["conversation_id"] not in selected_ids]
        random.shuffle(remaining_pool)
        for conv in remaining_pool:
            if len(golden) >= total_size:
                break
            add_example(conv)

    # Format output
    output = []
    for i, conv in enumerate(golden):
        output.append({
            "id": f"gs_{i + 1:03d}",
            "tweet_id": conv["tweet_id"],
            "customer_message": conv["customer_message"],
            "thread_context": conv.get("thread_context", []),
            "brand_reply": conv["brand_reply"],
            "rough_intent_bucket": conv["rough_intent"],
            # These will be filled during labelling:
            "gold_intent": "",
            "gold_escalation": "",
            "gold_escalation_reason": "",
            "gold_reply_quality_notes": "",
            "labeller_notes": "",
        })

    logger.info("Created golden set with %d examples.", len(output))
    intent_dist = Counter(e["rough_intent_bucket"] for e in output)
    logger.info("Golden set intent distribution: %s", dict(intent_dist))
    return output


def save_golden_set(examples: list[dict], output_path: Path) -> None:
    """Save golden set as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Saved golden set to %s", output_path)


if __name__ == "__main__":
    from src.config import PROCESSED_CONVERSATIONS_PATH, GOLDEN_SET_PATH, GOLDEN_SET_SIZE

    conversations = load_conversations(PROCESSED_CONVERSATIONS_PATH)
    golden = create_golden_set(conversations, total_size=GOLDEN_SET_SIZE)
    save_golden_set(golden, GOLDEN_SET_PATH)
